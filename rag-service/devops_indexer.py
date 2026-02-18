"""
DevOps Indexer & Log Analyzer for RAG Pipeline
Indexes runbooks, postmortems, and infrastructure docs into Qdrant.
Analyzes logs with regex pre-filtering + LLM classification.
"""

import asyncio
import hashlib
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Callable

from qdrant_client.http.models import (
    VectorParams, Distance, PointStruct,
    Filter, FieldCondition, MatchValue,
    SparseVectorParams, SparseIndexParams,
    Prefetch, FusionQuery, Fusion,
)
from sentence_transformers import SentenceTransformer

from qdrant_utils import make_qdrant_client

logger = logging.getLogger(__name__)

# Supported document extensions
DOC_EXTENSIONS = {".md", ".txt", ".rst", ".pdf", ".yaml", ".yml", ".json", ".toml", ".conf", ".cfg"}


@dataclass
class DevOpsIndexJob:
    """Tracks a document indexing job."""
    job_id: str
    path: str
    status: str = "running"
    files_found: int = 0
    files_indexed: int = 0
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
            "path": self.path,
            "status": self.status,
            "files_found": self.files_found,
            "files_indexed": self.files_indexed,
            "chunks_indexed": self.chunks_indexed,
            "errors": self.errors[-10:],
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }


@dataclass
class LogAnalysisResult:
    severity: str  # critical, error, warning, info
    service: str
    error_type: str
    summary: str
    related_runbook: Optional[str] = None
    timestamp: Optional[str] = None
    raw_line: str = ""

    def to_dict(self) -> dict:
        return {
            "severity": self.severity,
            "service": self.service,
            "error_type": self.error_type,
            "summary": self.summary,
            "related_runbook": self.related_runbook,
            "timestamp": self.timestamp,
            "raw_line": self.raw_line[:200],
        }


@dataclass
class LogAnalysisJob:
    job_id: str
    status: str = "running"
    lines_processed: int = 0
    lines_total: int = 0
    categories: Dict[str, int] = field(default_factory=dict)
    results: List[dict] = field(default_factory=list)
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
            "lines_processed": self.lines_processed,
            "lines_total": self.lines_total,
            "categories": self.categories,
            "results_count": len(self.results),
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }


# ── Severity detection keywords ─────────────────────────────────

SEVERITY_KEYWORDS = {
    "critical": ["FATAL", "PANIC", "EMERGENCY", "OOMKilled", "segfault", "kernel panic",
                  "out of memory", "SIGKILL", "SIGSEGV", "core dumped"],
    "error": ["ERROR", "FAIL", "FAILED", "Exception", "Traceback", "refused",
              "denied", "timeout", "SIGTERM", "exit code", "cannot", "unable"],
    "warning": ["WARN", "WARNING", "deprecated", "retry", "retrying", "slow",
                "degraded", "backoff", "throttl"],
}


class DevOpsIndexer:
    COLLECTION_NAME = "devops_docs"
    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        qdrant_api_key: Optional[str] = None,
        model: Optional[SentenceTransformer] = None,
        embedding_model: str = "BAAI/bge-base-en-v1.5",
    ):
        self.client = make_qdrant_client(qdrant_url, qdrant_api_key)
        self._qdrant_url = qdrant_url.rstrip("/")
        self._qdrant_api_key = qdrant_api_key

        if model is not None:
            self.model = model
        else:
            self.model = SentenceTransformer(embedding_model)

        self.dim = self.model.get_sentence_embedding_dimension()
        self._ensure_collection()

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

    @staticmethod
    def _generate_id(source_path: str, chunk_idx: int) -> int:
        key = f"devops:{source_path}:{chunk_idx}"
        h = hashlib.sha256(key.encode()).hexdigest()
        return int(h[:16], 16)

    # ── Document ingestion ────────────────────────────────────────

    async def index_path(
        self,
        path: str,
        recursive: bool = True,
        progress_callback: Optional[Callable] = None,
    ) -> DevOpsIndexJob:
        """Index files from a path (file or directory)."""
        job = DevOpsIndexJob(
            job_id=uuid.uuid4().hex[:12],
            path=path,
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        try:
            p = Path(path)
            files = []

            if p.is_file():
                files = [p]
            elif p.is_dir():
                if recursive:
                    files = [f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in DOC_EXTENSIONS]
                else:
                    files = [f for f in p.glob("*") if f.is_file() and f.suffix.lower() in DOC_EXTENSIONS]
            else:
                job.status = "failed"
                job.errors.append(f"Path not found: {path}")
                job.ended_at = datetime.now(timezone.utc).isoformat()
                return job

            job.files_found = len(files)
            job.log(f"Found {len(files)} files to index")

            for file_path in files:
                try:
                    chunks_count = await self._index_file(file_path, p if p.is_dir() else p.parent)
                    job.files_indexed += 1
                    job.chunks_indexed += chunks_count
                    job.log(f"Indexed {file_path.name} ({chunks_count} chunks)")
                except Exception as e:
                    job.errors.append(f"{file_path.name}: {str(e)[:100]}")
                    job.log(f"ERROR indexing {file_path.name}: {str(e)[:80]}")

                if progress_callback:
                    progress_callback(job)

            job.status = "completed"
            job.log(f"COMPLETED: {job.files_indexed} files, {job.chunks_indexed} chunks indexed")

        except Exception as e:
            job.status = "failed"
            job.errors.append(str(e)[:200])
            job.log(f"FAILED: {str(e)[:200]}")

        job.ended_at = datetime.now(timezone.utc).isoformat()
        if progress_callback:
            progress_callback(job)
        return job

    async def _index_file(self, file_path: Path, base_dir: Path) -> int:
        """Index a single file. Returns number of chunks indexed."""
        from sparse_encoder import encode_sparse

        suffix = file_path.suffix.lower()

        # Extract text based on file type
        if suffix == ".pdf":
            text = self._extract_pdf_text(file_path)
        else:
            text = file_path.read_text(encoding="utf-8", errors="ignore")

        if not text or len(text) < 20:
            return 0

        # Determine doc type from path/content
        rel_path = str(file_path.relative_to(base_dir))
        doc_type = self._classify_doc_type(rel_path, text)
        title = self._extract_title(text, file_path.name)

        # Chunk with context preservation
        chunks = self._chunk_document(text, title)

        if not chunks:
            return 0

        # Embed
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

        # Build points
        now = datetime.now(timezone.utc).isoformat()
        points = []
        for i, chunk in enumerate(chunks):
            points.append(PointStruct(
                id=self._generate_id(rel_path, i),
                vector={
                    "dense": dense_embeddings[i].tolist(),
                    "sparse": sparse_embeddings[i],
                },
                payload={
                    "text": chunk["text"],
                    "title": title,
                    "section": chunk.get("section", ""),
                    "source_path": rel_path,
                    "doc_type": doc_type,
                    "chunk_idx": i,
                    "indexed_at": now,
                    "source_type": "devops",
                },
            ))

        # Upsert
        await loop.run_in_executor(
            None,
            lambda: self.client.upsert(
                collection_name=self.COLLECTION_NAME,
                points=points,
            )
        )

        return len(points)

    @staticmethod
    def _extract_pdf_text(file_path: Path) -> str:
        """Extract text from PDF using pymupdf."""
        try:
            import fitz  # pymupdf
            doc = fitz.open(str(file_path))
            text = ""
            for page in doc:
                text += page.get_text() + "\n\n"
            doc.close()
            return text
        except ImportError:
            logger.warning("PyMuPDF not installed, skipping PDF")
            return ""
        except Exception as e:
            logger.warning(f"PDF extraction failed for {file_path}: {e}")
            return ""

    @staticmethod
    def _extract_title(text: str, filename: str) -> str:
        """Extract title from document content or filename."""
        # Try markdown H1
        h1_match = re.match(r"^#\s+(.+)", text)
        if h1_match:
            return h1_match.group(1).strip()

        # Try first non-empty line
        for line in text.split("\n")[:5]:
            line = line.strip()
            if line and len(line) > 3 and len(line) < 200:
                return line

        # Fallback to filename
        return Path(filename).stem.replace("-", " ").replace("_", " ").title()

    @staticmethod
    def _classify_doc_type(path: str, text: str) -> str:
        """Classify document type from path and content."""
        path_lower = path.lower()
        text_lower = text[:500].lower()

        if "runbook" in path_lower or "playbook" in path_lower:
            return "runbook"
        if "postmortem" in path_lower or "incident" in path_lower or "rca" in path_lower:
            return "postmortem"
        if "config" in path_lower or path_lower.endswith((".yaml", ".yml", ".toml", ".conf", ".cfg")):
            return "config"
        if "procedure" in path_lower or "sop" in path_lower or "how-to" in path_lower:
            return "procedure"
        if "architecture" in path_lower or "design" in path_lower:
            return "architecture"
        if "postmortem" in text_lower or "root cause" in text_lower or "timeline" in text_lower:
            return "postmortem"
        if "steps:" in text_lower or "step 1" in text_lower or "procedure" in text_lower:
            return "procedure"

        return "documentation"

    @staticmethod
    def _chunk_document(
        text: str,
        title: str,
        max_chars: int = 1200,
        overlap: int = 150,
    ) -> List[Dict]:
        """Context-preserving chunking: prepend nearest heading to each chunk."""
        if not text:
            return []

        lines = text.split("\n")
        chunks = []
        current_section = title
        current_lines = []
        current_len = 0

        for line in lines:
            # Track section headings
            heading_match = re.match(r"^(#{1,4})\s+(.+)", line)
            if heading_match:
                # Flush current chunk
                if current_lines and current_len > 50:
                    chunk_text = f"[{current_section}]\n" + "\n".join(current_lines)
                    chunks.append({
                        "text": chunk_text,
                        "section": current_section,
                        "chunk_idx": len(chunks),
                    })
                    # Keep overlap
                    overlap_lines = []
                    ol = 0
                    for l in reversed(current_lines):
                        if ol + len(l) > overlap:
                            break
                        overlap_lines.insert(0, l)
                        ol += len(l)
                    current_lines = overlap_lines
                    current_len = ol

                current_section = heading_match.group(2).strip()
                continue

            current_lines.append(line)
            current_len += len(line) + 1

            if current_len >= max_chars:
                chunk_text = f"[{current_section}]\n" + "\n".join(current_lines)
                chunks.append({
                    "text": chunk_text,
                    "section": current_section,
                    "chunk_idx": len(chunks),
                })
                # Keep overlap
                overlap_lines = []
                ol = 0
                for l in reversed(current_lines):
                    if ol + len(l) > overlap:
                        break
                    overlap_lines.insert(0, l)
                    ol += len(l)
                current_lines = overlap_lines
                current_len = ol

        # Final chunk
        if current_lines:
            text_content = "\n".join(current_lines).strip()
            if text_content and len(text_content) > 20:
                chunk_text = f"[{current_section}]\n" + text_content
                chunks.append({
                    "text": chunk_text,
                    "section": current_section,
                    "chunk_idx": len(chunks),
                })

        return chunks

    # ── Search ────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        doc_type_filter: Optional[str] = None,
    ) -> List[Dict]:
        """Hybrid search on devops_docs collection."""
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
            if doc_type_filter:
                conditions.append(
                    FieldCondition(key="doc_type", match=MatchValue(value=doc_type_filter))
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
                    "title": r.payload.get("title", ""),
                    "section": r.payload.get("section", ""),
                    "source_path": r.payload.get("source_path", ""),
                    "doc_type": r.payload.get("doc_type", ""),
                    "text": r.payload.get("text", ""),
                    "score": round(r.score, 4),
                    "source_type": "devops",
                }
                for r in results.points
            ]

        except Exception as e:
            logger.error(f"DevOps search error: {e}")
            return []

    # ── Sources ───────────────────────────────────────────────────

    def get_sources(self) -> List[Dict]:
        """List indexed document sources."""
        try:
            collections = [c.name for c in self.client.get_collections().collections]
            if self.COLLECTION_NAME not in collections:
                return []

            sources: Dict[str, Dict] = {}
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
                    source = payload.get("source_path", "unknown")
                    if source not in sources:
                        sources[source] = {
                            "source_path": source,
                            "doc_type": payload.get("doc_type", ""),
                            "title": payload.get("title", ""),
                            "chunk_count": 0,
                            "indexed_at": payload.get("indexed_at", ""),
                        }
                    sources[source]["chunk_count"] += 1

                if next_offset is None:
                    break
                offset = next_offset

            return sorted(sources.values(), key=lambda x: x["source_path"])

        except Exception as e:
            logger.error(f"Error getting devops sources: {e}")
            return []

    def delete_source(self, source_path: str) -> bool:
        """Delete all chunks from a source path."""
        try:
            self.client.delete(
                collection_name=self.COLLECTION_NAME,
                points_selector=Filter(
                    must=[FieldCondition(key="source_path", match=MatchValue(value=source_path))]
                ),
            )
            logger.info(f"Deleted devops source: {source_path}")
            return True
        except Exception as e:
            logger.error(f"Error deleting devops source {source_path}: {e}")
            return False


# ── Log Analyzer ──────────────────────────────────────────────────

LOG_CLASSIFICATION_PROMPT = """Classify these log lines. For each line, return a JSON object with:
{
  "severity": "critical|error|warning|info",
  "service": "service name if identifiable",
  "error_type": "timeout|oom|crash|connection_refused|auth_failure|disk_full|config_error|unknown",
  "summary": "one-line human-readable summary"
}

Return a JSON array with one object per input line. Return ONLY the JSON array."""


class LogAnalyzer:
    def __init__(self, llm_client, devops_indexer: DevOpsIndexer):
        self.llm_client = llm_client
        self.devops_indexer = devops_indexer

    async def analyze_logs(
        self,
        log_text: str,
        source_service: str = "unknown",
        progress_callback: Optional[Callable] = None,
    ) -> LogAnalysisJob:
        """Analyze log text: pre-filter with regex, then classify with LLM."""
        job = LogAnalysisJob(
            job_id=uuid.uuid4().hex[:12],
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        try:
            lines = [l.strip() for l in log_text.split("\n") if l.strip()]
            job.lines_total = len(lines)
            job.log(f"Analyzing {len(lines)} log lines")

            # Pre-filter: only send interesting lines to LLM
            interesting = self._prefilter_logs(lines)
            job.log(f"Pre-filtered to {len(interesting)} interesting lines (from {len(lines)})")

            if not interesting:
                job.status = "completed"
                job.lines_processed = len(lines)
                job.categories = {"info": len(lines)}
                job.log("No errors or warnings found")
                job.ended_at = datetime.now(timezone.utc).isoformat()
                return job

            # Classify in batches via LLM
            batch_size = 20
            for i in range(0, len(interesting), batch_size):
                batch = interesting[i:i + batch_size]
                classified = await self._classify_batch(batch)

                for result in classified:
                    sev = result.get("severity", "info")
                    job.categories[sev] = job.categories.get(sev, 0) + 1
                    job.results.append(result)

                job.lines_processed = min(i + batch_size, len(interesting))
                job.log(f"Classified {job.lines_processed}/{len(interesting)} lines")

                if progress_callback:
                    progress_callback(job)

            # Match runbooks for errors/criticals
            error_results = [r for r in job.results if r.get("severity") in ("error", "critical")]
            if error_results:
                await self._match_runbooks(error_results)
                job.log(f"Matched runbooks for {len(error_results)} error/critical entries")

            job.status = "completed"
            job.lines_processed = len(lines)
            job.log(f"COMPLETED: {len(job.results)} classified, categories: {job.categories}")

        except Exception as e:
            job.status = "failed"
            job.log(f"FAILED: {str(e)[:200]}")
            logger.error(f"Log analysis failed: {e}")

        job.ended_at = datetime.now(timezone.utc).isoformat()
        if progress_callback:
            progress_callback(job)
        return job

    @staticmethod
    def _prefilter_logs(lines: List[str]) -> List[str]:
        """Fast regex-based pre-filtering to identify likely error/warning lines."""
        interesting = []
        all_keywords = []
        for sev_keywords in SEVERITY_KEYWORDS.values():
            all_keywords.extend(sev_keywords)

        pattern = re.compile("|".join(re.escape(kw) for kw in all_keywords), re.IGNORECASE)

        for line in lines:
            if pattern.search(line):
                interesting.append(line)

        return interesting

    async def _classify_batch(self, lines: List[str]) -> List[dict]:
        """Send batch of pre-filtered log lines to LLM for classification."""
        numbered = "\n".join(f"[{i+1}] {line}" for i, line in enumerate(lines))

        try:
            loop = asyncio.get_event_loop()
            models = await loop.run_in_executor(None, self.llm_client.models.list)
            model_id = models.data[0].id if models.data else None

            if not model_id:
                return [{"severity": "unknown", "service": "unknown",
                         "error_type": "unknown", "summary": line, "raw_line": line}
                        for line in lines]

            response = await loop.run_in_executor(
                None,
                lambda: self.llm_client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": LOG_CLASSIFICATION_PROMPT},
                        {"role": "user", "content": f"Classify these {len(lines)} log lines:\n\n{numbered}"},
                    ],
                    max_tokens=4096,
                    temperature=0.1,
                    timeout=120,
                ),
            )

            response_text = response.choices[0].message.content
            return self._parse_classifications(response_text, lines)

        except Exception as e:
            logger.warning(f"Log classification failed: {e}")
            return [{"severity": "unknown", "service": "unknown",
                     "error_type": "unknown", "summary": line[:100], "raw_line": line}
                    for line in lines]

    @staticmethod
    def _parse_classifications(response_text: str, lines: List[str]) -> List[dict]:
        """Parse LLM classification response."""
        import json

        text = response_text.strip()
        if "```" in text:
            json_lines = []
            inside = False
            for line in text.split("\n"):
                if line.strip().startswith("```"):
                    inside = not inside
                    continue
                if inside:
                    json_lines.append(line)
            if json_lines:
                text = "\n".join(json_lines)

        try:
            data = json.loads(text)
            if isinstance(data, list):
                # Attach raw lines
                for i, entry in enumerate(data):
                    if i < len(lines):
                        entry["raw_line"] = lines[i]
                return data
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: return basic classification
        return [{"severity": "unknown", "service": "unknown",
                 "error_type": "unknown", "summary": line[:100], "raw_line": line}
                for line in lines]

    async def _match_runbooks(self, results: List[dict]):
        """For each error, search devops_docs for matching runbooks."""
        for result in results:
            error_type = result.get("error_type", "")
            service = result.get("service", "")
            summary = result.get("summary", "")
            query = f"{error_type} {service} {summary}".strip()

            if not query:
                continue

            matches = self.devops_indexer.search(
                query=query, top_k=1, doc_type_filter="runbook"
            )
            if matches and matches[0]["score"] > 0.3:
                result["related_runbook"] = matches[0]["source_path"]
