"""
Jira Cloud Indexer for RAG Pipeline
Indexes Jira tickets (issues + comments) into Qdrant for hybrid search.
Authenticates via Atlassian REST API v3 with basic auth (email:token).
"""

import asyncio
import base64
import hashlib
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Optional, Callable

import httpx
from qdrant_client.http.models import (
    VectorParams, Distance, PointStruct,
    Filter, FieldCondition, MatchValue,
    SparseVectorParams, SparseIndexParams,
    Prefetch, FusionQuery, Fusion,
)
from sentence_transformers import SentenceTransformer

from qdrant_utils import make_qdrant_client

logger = logging.getLogger(__name__)


# ── Job tracking ─────────────────────────────────────────────────


@dataclass
class JiraIndexJob:
    """Tracks a Jira indexing job."""
    job_id: str
    status: str = "running"
    tickets_found: int = 0
    tickets_indexed: int = 0
    chunks_indexed: int = 0
    errors: List[str] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)
    started_at: Optional[str] = None
    ended_at: Optional[str] = None

    MAX_LOGS = 2000

    def log(self, msg: str):
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        self.logs.append(f"[{ts}] {msg}")
        if len(self.logs) > self.MAX_LOGS:
            self.logs = self.logs[:50] + self.logs[-(self.MAX_LOGS - 50):]

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "tickets_found": self.tickets_found,
            "tickets_indexed": self.tickets_indexed,
            "chunks_indexed": self.chunks_indexed,
            "errors": self.errors[-10:],
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }


# ── ADF (Atlassian Document Format) text extraction ─────────────


def extract_text_from_adf(node: Optional[dict]) -> str:
    """Recursively extract plain text from an ADF JSON tree.

    ADF nodes have {"type": "...", "content": [...]} or {"type": "text", "text": "..."}.
    We walk the tree depth-first, collecting text nodes, inserting newlines
    for block-level elements (paragraph, heading, bulletList, orderedList, etc.).
    """
    if node is None:
        return ""
    if not isinstance(node, dict):
        return ""

    node_type = node.get("type", "")

    # Leaf text node
    if node_type == "text":
        return node.get("text", "")

    # Inline card (link)
    if node_type == "inlineCard":
        attrs = node.get("attrs", {})
        return attrs.get("url", "")

    # Emoji
    if node_type == "emoji":
        attrs = node.get("attrs", {})
        return attrs.get("shortName", "")

    # Mention
    if node_type == "mention":
        attrs = node.get("attrs", {})
        return f"@{attrs.get('text', attrs.get('id', ''))}"

    # Hard break
    if node_type == "hardBreak":
        return "\n"

    # Media / mediaGroup / mediaSingle — skip binary content
    if node_type in ("media", "mediaGroup", "mediaSingle"):
        return ""

    # Recurse into children
    children = node.get("content", [])
    if not isinstance(children, list):
        return ""

    parts = []
    for child in children:
        text = extract_text_from_adf(child)
        if text:
            parts.append(text)

    # Block-level elements: join with newlines
    block_types = {
        "doc", "paragraph", "heading", "blockquote",
        "bulletList", "orderedList", "listItem",
        "codeBlock", "rule", "table", "tableRow", "tableCell",
        "tableHeader", "panel", "expand", "layoutSection",
        "layoutColumn", "decisionList", "decisionItem",
        "taskList", "taskItem",
    }

    if node_type in block_types:
        return "\n".join(parts)

    # Inline elements: join with space
    return " ".join(parts)


# ── Jira Indexer ─────────────────────────────────────────────────


class JiraIndexer:
    COLLECTION_NAME = "jira_tickets"
    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(
        self,
        jira_base_url: Optional[str] = None,
        jira_email: Optional[str] = None,
        jira_api_token: Optional[str] = None,
        qdrant_url: str = "http://localhost:6333",
        qdrant_api_key: Optional[str] = None,
        model: Optional[SentenceTransformer] = None,
        embedding_model: str = "BAAI/bge-base-en-v1.5",
    ):
        # Jira credentials (from params or env)
        self._jira_base_url = (jira_base_url or os.getenv("JIRA_BASE_URL", "")).rstrip("/")
        self._jira_email = jira_email or os.getenv("JIRA_EMAIL", "")
        self._jira_api_token = jira_api_token or os.getenv("JIRA_API_TOKEN", "")

        # Build basic auth header
        if self._jira_email and self._jira_api_token:
            creds = f"{self._jira_email}:{self._jira_api_token}"
            b64 = base64.b64encode(creds.encode()).decode()
            self._auth_header = f"Basic {b64}"
        else:
            self._auth_header = ""

        # Qdrant
        self.client = make_qdrant_client(qdrant_url, qdrant_api_key)
        self._qdrant_url = qdrant_url.rstrip("/")
        self._qdrant_api_key = qdrant_api_key

        # Embedding model
        if model is not None:
            self.model = model
        else:
            self.model = SentenceTransformer(embedding_model)

        self.dim = self.model.get_sentence_embedding_dimension()

        # Rate limiting semaphore
        self._semaphore = asyncio.Semaphore(10)

        self._ensure_collection()

    # ── Collection management ────────────────────────────────────

    def _ensure_collection(self):
        collections = [c.name for c in self.client.get_collections().collections]
        if self.COLLECTION_NAME in collections:
            info = self.client.get_collection(self.COLLECTION_NAME)
            vectors_config = info.config.params.vectors
            if isinstance(vectors_config, dict) and "dense" in vectors_config:
                return
            logger.warning(f"Recreating {self.COLLECTION_NAME} with named vectors")
            self.client.delete_collection(self.COLLECTION_NAME)

        self.client.create_collection(
            collection_name=self.COLLECTION_NAME,
            vectors_config={
                "dense": VectorParams(size=self.dim, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False),
                ),
            },
        )
        logger.info(f"Created hybrid collection: {self.COLLECTION_NAME}")

    # ── ID generation ────────────────────────────────────────────

    @staticmethod
    def _generate_id(ticket_key: str, chunk_idx: int) -> int:
        key = f"jira:{ticket_key}:{chunk_idx}"
        h = hashlib.sha256(key.encode()).hexdigest()
        return int(h[:16], 16)

    # ── Jira API helpers ─────────────────────────────────────────

    def _make_headers(self) -> dict:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self._auth_header:
            headers["Authorization"] = self._auth_header
        return headers

    async def _jira_get(self, path: str, params: Optional[dict] = None) -> dict:
        """Make an authenticated GET request to Jira REST API v3."""
        url = f"{self._jira_base_url}{path}"
        async with self._semaphore:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url, headers=self._make_headers(), params=params)
                resp.raise_for_status()
                return resp.json()

    async def _fetch_issues_jql(
        self,
        jql: str,
        fields: str = "summary,description,status,assignee,reporter,priority,issuetype,labels,components,comment,created,updated,parent,customfield_10014,customfield_10020",
        max_results: int = 50,
    ) -> List[dict]:
        """Fetch all issues matching a JQL query using nextPageToken pagination."""
        all_issues = []
        next_page_token = None

        while True:
            params = {
                "jql": jql,
                "fields": fields,
                "maxResults": str(max_results),
            }
            if next_page_token:
                params["nextPageToken"] = next_page_token

            data = await self._jira_get("/rest/api/3/search/jql", params=params)

            issues = data.get("issues", [])
            all_issues.extend(issues)

            next_page_token = data.get("nextPageToken")
            if not next_page_token or not issues:
                break

        return all_issues

    async def _fetch_single_issue(self, issue_key: str) -> dict:
        """Fetch a single issue by key with all needed fields."""
        fields = "summary,description,status,assignee,reporter,priority,issuetype,labels,components,comment,created,updated,parent,customfield_10014,customfield_10020"
        data = await self._jira_get(f"/rest/api/3/issue/{issue_key}", params={"fields": fields})
        return data

    # ── Issue parsing ────────────────────────────────────────────

    @staticmethod
    def _parse_issue(issue: dict) -> dict:
        """Extract structured data from a Jira issue API response."""
        fields = issue.get("fields", {})
        key = issue.get("key", "")

        # Safe field extraction
        status = ""
        status_obj = fields.get("status")
        if isinstance(status_obj, dict):
            status = status_obj.get("name", "")

        assignee = ""
        assignee_obj = fields.get("assignee")
        if isinstance(assignee_obj, dict):
            assignee = assignee_obj.get("displayName", "")

        reporter = ""
        reporter_obj = fields.get("reporter")
        if isinstance(reporter_obj, dict):
            reporter = reporter_obj.get("displayName", "")

        priority = ""
        priority_obj = fields.get("priority")
        if isinstance(priority_obj, dict):
            priority = priority_obj.get("name", "")

        issue_type = ""
        type_obj = fields.get("issuetype")
        if isinstance(type_obj, dict):
            issue_type = type_obj.get("name", "")

        labels = fields.get("labels", []) or []
        if not isinstance(labels, list):
            labels = []

        components = []
        comp_list = fields.get("components", []) or []
        if isinstance(comp_list, list):
            components = [c.get("name", "") for c in comp_list if isinstance(c, dict)]

        # Sprint (customfield_10020 is common for sprints)
        sprint = ""
        sprint_field = fields.get("customfield_10020")
        if isinstance(sprint_field, list) and sprint_field:
            last_sprint = sprint_field[-1]
            if isinstance(last_sprint, dict):
                sprint = last_sprint.get("name", "")
        elif isinstance(sprint_field, dict):
            sprint = sprint_field.get("name", "")

        # Epic key (customfield_10014 is common for epic link)
        epic_key = ""
        epic_name = ""
        epic_field = fields.get("customfield_10014")
        if isinstance(epic_field, str):
            epic_key = epic_field
        elif isinstance(epic_field, dict):
            epic_key = epic_field.get("key", "")
            epic_name = epic_field.get("fields", {}).get("summary", "") if "fields" in epic_field else ""

        # Parent (for subtasks or child issues)
        parent_obj = fields.get("parent")
        if isinstance(parent_obj, dict) and not epic_key:
            parent_type = ""
            parent_type_obj = parent_obj.get("fields", {}).get("issuetype", {})
            if isinstance(parent_type_obj, dict):
                parent_type = parent_type_obj.get("name", "")
            if parent_type.lower() == "epic":
                epic_key = parent_obj.get("key", "")
                epic_name = parent_obj.get("fields", {}).get("summary", "")

        # Description (ADF JSON)
        description_adf = fields.get("description")
        description = extract_text_from_adf(description_adf).strip()

        # Comments
        comments = []
        comment_container = fields.get("comment", {})
        if isinstance(comment_container, dict):
            comment_list = comment_container.get("comments", [])
        elif isinstance(comment_container, list):
            comment_list = comment_container
        else:
            comment_list = []

        for c in comment_list:
            if not isinstance(c, dict):
                continue
            author_obj = c.get("author", {})
            author_name = author_obj.get("displayName", "") if isinstance(author_obj, dict) else ""
            body_adf = c.get("body")
            body = extract_text_from_adf(body_adf).strip()
            created = c.get("created", "")
            if body:
                comments.append({
                    "author": author_name,
                    "body": body,
                    "created": created,
                })

        project = key.split("-")[0] if "-" in key else ""

        return {
            "key": key,
            "project": project,
            "summary": fields.get("summary", ""),
            "description": description,
            "status": status,
            "assignee": assignee,
            "reporter": reporter,
            "priority": priority,
            "issue_type": issue_type,
            "labels": labels,
            "components": components,
            "sprint": sprint,
            "epic_key": epic_key,
            "epic_name": epic_name,
            "created_at": fields.get("created", ""),
            "updated_at": fields.get("updated", ""),
            "comments": comments,
        }

    # ── Chunking ─────────────────────────────────────────────────

    @staticmethod
    def _build_ticket_chunks(parsed: dict) -> List[dict]:
        """Build chunks for a ticket.

        Chunk 0: summary + description (ticket metadata header).
        Chunk 1+: one per comment.
        """
        key = parsed["key"]
        summary = parsed["summary"]

        # Build metadata header for chunk 0
        meta_lines = [f"[{key}] {summary}"]
        meta_parts = []
        if parsed["issue_type"]:
            meta_parts.append(f"Type: {parsed['issue_type']}")
        if parsed["priority"]:
            meta_parts.append(f"Priority: {parsed['priority']}")
        if parsed["status"]:
            meta_parts.append(f"Status: {parsed['status']}")
        if meta_parts:
            meta_lines.append(" | ".join(meta_parts))

        people_parts = []
        if parsed["assignee"]:
            people_parts.append(f"Assignee: {parsed['assignee']}")
        if parsed["reporter"]:
            people_parts.append(f"Reporter: {parsed['reporter']}")
        if people_parts:
            meta_lines.append(" | ".join(people_parts))

        extra_parts = []
        if parsed["labels"]:
            extra_parts.append(f"Labels: {', '.join(parsed['labels'])}")
        if parsed["epic_name"]:
            extra_parts.append(f"Epic: {parsed['epic_name']}")
        elif parsed["epic_key"]:
            extra_parts.append(f"Epic: {parsed['epic_key']}")
        if extra_parts:
            meta_lines.append(" | ".join(extra_parts))

        header = "\n".join(meta_lines)

        # Chunk 0: ticket summary + description
        chunk0_text = header
        if parsed["description"]:
            chunk0_text += f"\n\n{parsed['description']}"

        chunks = [{
            "text": chunk0_text,
            "chunk_idx": 0,
            "chunk_type": "ticket",
            "comment_author": "",
        }]

        # Chunk 1+: one per comment
        for i, comment in enumerate(parsed["comments"]):
            # Format comment creation date
            created_str = ""
            if comment["created"]:
                try:
                    dt = datetime.fromisoformat(comment["created"].replace("Z", "+00:00"))
                    created_str = dt.strftime("%Y-%m-%d")
                except (ValueError, TypeError):
                    created_str = comment["created"][:10] if len(comment["created"]) >= 10 else comment["created"]

            comment_header = f"[{key}] {summary}\nComment by {comment['author']}"
            if created_str:
                comment_header += f" ({created_str})"
            comment_header += ":"

            comment_text = f"{comment_header}\n\n{comment['body']}"

            chunks.append({
                "text": comment_text,
                "chunk_idx": i + 1,
                "chunk_type": "comment",
                "comment_author": comment["author"],
            })

        return chunks

    # ── Embedding + upsert ───────────────────────────────────────

    async def _embed_and_upsert(self, parsed: dict, chunks: List[dict]) -> int:
        """Generate embeddings and upsert chunks to Qdrant. Returns count."""
        from sparse_encoder import encode_sparse

        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        loop = asyncio.get_event_loop()

        dense_embeddings = await loop.run_in_executor(
            None,
            lambda: self.model.encode(texts, normalize_embeddings=True, batch_size=len(texts))
        )
        sparse_embeddings = await loop.run_in_executor(
            None,
            lambda: encode_sparse(texts)
        )

        now = datetime.now(timezone.utc).isoformat()
        points = []
        for i, chunk in enumerate(chunks):
            payload = {
                "text": chunk["text"],
                "ticket_key": parsed["key"],
                "project": parsed["project"],
                "summary": parsed["summary"],
                "status": parsed["status"],
                "assignee": parsed["assignee"],
                "reporter": parsed["reporter"],
                "priority": parsed["priority"],
                "issue_type": parsed["issue_type"],
                "labels": parsed["labels"],
                "components": parsed["components"],
                "sprint": parsed["sprint"],
                "epic_key": parsed["epic_key"],
                "epic_name": parsed["epic_name"],
                "chunk_type": chunk["chunk_type"],
                "comment_author": chunk["comment_author"],
                "created_at": parsed["created_at"],
                "updated_at": parsed["updated_at"],
                "indexed_at": now,
                "source_type": "jira",
            }

            points.append(PointStruct(
                id=self._generate_id(parsed["key"], chunk["chunk_idx"]),
                vector={
                    "dense": dense_embeddings[i].tolist(),
                    "sparse": sparse_embeddings[i],
                },
                payload=payload,
            ))

        await loop.run_in_executor(
            None,
            lambda: self.client.upsert(
                collection_name=self.COLLECTION_NAME,
                points=points,
            )
        )

        return len(points)

    # ── Public methods ───────────────────────────────────────────

    async def import_all(
        self,
        project: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
    ) -> JiraIndexJob:
        """Import all tickets (optionally filtered by project) into Qdrant."""
        job = JiraIndexJob(
            job_id=uuid.uuid4().hex[:12],
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        try:
            jql = f"project = {project} ORDER BY updated DESC" if project else "ORDER BY updated DESC"
            job.log(f"Fetching issues: {jql}")

            issues = await self._fetch_issues_jql(jql)
            job.tickets_found = len(issues)
            job.log(f"Found {len(issues)} issues")

            if progress_callback:
                progress_callback(job)

            for issue in issues:
                try:
                    parsed = self._parse_issue(issue)
                    chunks = self._build_ticket_chunks(parsed)
                    count = await self._embed_and_upsert(parsed, chunks)
                    job.tickets_indexed += 1
                    job.chunks_indexed += count
                    job.log(f"Indexed {parsed['key']} ({count} chunks)")
                except Exception as e:
                    key = issue.get("key", "?")
                    job.errors.append(f"{key}: {str(e)[:100]}")
                    job.log(f"ERROR indexing {key}: {str(e)[:80]}")

                if progress_callback:
                    progress_callback(job)

            job.status = "completed"
            job.log(f"COMPLETED: {job.tickets_indexed} tickets, {job.chunks_indexed} chunks indexed")

        except Exception as e:
            job.status = "failed"
            job.errors.append(str(e)[:200])
            job.log(f"FAILED: {str(e)[:200]}")

        job.ended_at = datetime.now(timezone.utc).isoformat()
        if progress_callback:
            progress_callback(job)
        return job

    async def sync_recent(
        self,
        since_hours: int = 24,
        progress_callback: Optional[Callable] = None,
    ) -> JiraIndexJob:
        """Sync tickets updated within the last N hours."""
        job = JiraIndexJob(
            job_id=uuid.uuid4().hex[:12],
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        try:
            # Jira JQL supports relative time: updated >= "-24h"
            jql = f'updated >= "-{since_hours}h" ORDER BY updated DESC'
            job.log(f"Syncing recent issues: {jql}")

            issues = await self._fetch_issues_jql(jql)
            job.tickets_found = len(issues)
            job.log(f"Found {len(issues)} recently updated issues")

            if progress_callback:
                progress_callback(job)

            for issue in issues:
                try:
                    parsed = self._parse_issue(issue)
                    chunks = self._build_ticket_chunks(parsed)
                    count = await self._embed_and_upsert(parsed, chunks)
                    job.tickets_indexed += 1
                    job.chunks_indexed += count
                    job.log(f"Synced {parsed['key']} ({count} chunks)")
                except Exception as e:
                    key = issue.get("key", "?")
                    job.errors.append(f"{key}: {str(e)[:100]}")
                    job.log(f"ERROR syncing {key}: {str(e)[:80]}")

                if progress_callback:
                    progress_callback(job)

            job.status = "completed"
            job.log(f"COMPLETED: {job.tickets_indexed} tickets synced, {job.chunks_indexed} chunks")

        except Exception as e:
            job.status = "failed"
            job.errors.append(str(e)[:200])
            job.log(f"FAILED: {str(e)[:200]}")

        job.ended_at = datetime.now(timezone.utc).isoformat()
        if progress_callback:
            progress_callback(job)
        return job

    async def import_ticket(
        self,
        key: str,
        progress_callback: Optional[Callable] = None,
    ) -> JiraIndexJob:
        """Import a single ticket by key."""
        job = JiraIndexJob(
            job_id=uuid.uuid4().hex[:12],
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        try:
            job.log(f"Fetching issue {key}")
            issue = await self._fetch_single_issue(key)
            job.tickets_found = 1

            parsed = self._parse_issue(issue)
            chunks = self._build_ticket_chunks(parsed)
            count = await self._embed_and_upsert(parsed, chunks)
            job.tickets_indexed = 1
            job.chunks_indexed = count
            job.status = "completed"
            job.log(f"COMPLETED: Indexed {key} ({count} chunks)")

        except Exception as e:
            job.status = "failed"
            job.errors.append(str(e)[:200])
            job.log(f"FAILED: {str(e)[:200]}")

        job.ended_at = datetime.now(timezone.utc).isoformat()
        if progress_callback:
            progress_callback(job)
        return job

    async def get_ticket_live(self, key: str) -> dict:
        """Fetch a ticket from Jira live (not from Qdrant) and return parsed data."""
        issue = await self._fetch_single_issue(key)
        parsed = self._parse_issue(issue)
        return parsed

    # ── Search ───────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        project: Optional[str] = None,
        assignee: Optional[str] = None,
        ticket_key: Optional[str] = None,
        issue_type: Optional[str] = None,
    ) -> List[Dict]:
        """Hybrid search on jira_tickets collection with optional filters."""
        from sparse_encoder import encode_sparse_query

        try:
            collections = [c.name for c in self.client.get_collections().collections]
            if self.COLLECTION_NAME not in collections:
                return []

            prefixed_query = f"{self.BGE_QUERY_PREFIX}{query}"
            dense_embedding = self.model.encode(
                prefixed_query, normalize_embeddings=True
            ).tolist()
            sparse_embedding = encode_sparse_query(query)

            conditions = []
            if project:
                conditions.append(
                    FieldCondition(key="project", match=MatchValue(value=project))
                )
            if assignee:
                conditions.append(
                    FieldCondition(key="assignee", match=MatchValue(value=assignee))
                )
            if ticket_key:
                conditions.append(
                    FieldCondition(key="ticket_key", match=MatchValue(value=ticket_key))
                )
            if issue_type:
                conditions.append(
                    FieldCondition(key="issue_type", match=MatchValue(value=issue_type))
                )

            search_filter = Filter(must=conditions) if conditions else None

            results = self.client.query_points(
                collection_name=self.COLLECTION_NAME,
                prefetch=[
                    Prefetch(query=dense_embedding, using="dense", filter=search_filter, limit=top_k * 3),
                    Prefetch(query=sparse_embedding, using="sparse", filter=search_filter, limit=top_k * 3),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=top_k,
                with_payload=True,
            )

            return [
                {
                    "ticket_key": r.payload.get("ticket_key", ""),
                    "project": r.payload.get("project", ""),
                    "summary": r.payload.get("summary", ""),
                    "status": r.payload.get("status", ""),
                    "assignee": r.payload.get("assignee", ""),
                    "priority": r.payload.get("priority", ""),
                    "issue_type": r.payload.get("issue_type", ""),
                    "chunk_type": r.payload.get("chunk_type", ""),
                    "text": r.payload.get("text", ""),
                    "score": round(r.score, 4),
                    "source_type": "jira",
                }
                for r in results.points
            ]

        except Exception as e:
            logger.error(f"Jira search error: {e}")
            return []

    # ── Sources ──────────────────────────────────────────────────

    def get_sources(self) -> List[Dict]:
        """List indexed Jira projects with ticket counts."""
        try:
            collections = [c.name for c in self.client.get_collections().collections]
            if self.COLLECTION_NAME not in collections:
                return []

            projects: Dict[str, Dict] = {}
            offset = None

            while True:
                results = self.client.scroll(
                    collection_name=self.COLLECTION_NAME,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                points, next_offset = results

                for point in points:
                    payload = point.payload or {}
                    project = payload.get("project", "unknown")
                    if project not in projects:
                        projects[project] = {
                            "project": project,
                            "ticket_count": 0,
                            "chunk_count": 0,
                            "tickets": set(),
                            "last_indexed": payload.get("indexed_at", ""),
                        }
                    projects[project]["chunk_count"] += 1
                    ticket_key = payload.get("ticket_key", "")
                    if ticket_key:
                        projects[project]["tickets"].add(ticket_key)
                    indexed_at = payload.get("indexed_at", "")
                    if indexed_at > projects[project]["last_indexed"]:
                        projects[project]["last_indexed"] = indexed_at

                if next_offset is None:
                    break
                offset = next_offset

            # Finalize
            sources = []
            for p in projects.values():
                p["ticket_count"] = len(p["tickets"])
                del p["tickets"]
                sources.append(p)

            return sorted(sources, key=lambda x: x["project"])

        except Exception as e:
            logger.error(f"Error getting Jira sources: {e}")
            return []

    # ── Delete ───────────────────────────────────────────────────

    def delete_project(self, project: str) -> bool:
        """Delete all indexed chunks for a Jira project."""
        try:
            self.client.delete(
                collection_name=self.COLLECTION_NAME,
                points_selector=Filter(
                    must=[FieldCondition(key="project", match=MatchValue(value=project))]
                ),
            )
            logger.info(f"Deleted Jira project: {project}")
            return True
        except Exception as e:
            logger.error(f"Error deleting Jira project {project}: {e}")
            return False
