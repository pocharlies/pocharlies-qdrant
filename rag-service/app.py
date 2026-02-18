"""
RAG Service FastAPI Application
Provides endpoints for code retrieval, web indexing, and tool execution
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI
import validators

from indexer import CodeIndexer
from retriever import CodeRetriever, CodeTools, TOOL_DEFINITIONS
from web_indexer import WebIndexer, CrawlJob
from agent import AgentTask, run_agent_task
from product_indexer import ProductIndexer, ProductSyncJob
from product_classifier import ProductClassifier, ClassificationJob
from translator import TranslationPipeline, TranslationJob
from devops_indexer import DevOpsIndexer, LogAnalyzer, DevOpsIndexJob, LogAnalysisJob
from shopify_client import ShopifyClient
from reranker import get_reranker, init_reranker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://litellm:4000/v1")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "none")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
REPOS_PATH = os.getenv("REPOS_PATH", "/repos")
DEVOPS_DOCS_PATH = os.getenv("DEVOPS_DOCS_PATH", "/devops-docs")
SHOPIFY_SHOP_DOMAIN = os.getenv("SHOPIFY_SHOP_DOMAIN", "")
SHOPIFY_ACCESS_TOKEN = os.getenv("SHOPIFY_ACCESS_TOKEN", "")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "true").lower() == "true"

# Global instances
retriever: CodeRetriever = None
indexer: CodeIndexer = None
web_indexer: WebIndexer = None
competitor_indexer: WebIndexer = None
product_indexer: ProductIndexer = None
product_classifier: ProductClassifier = None
translator: TranslationPipeline = None
devops_indexer: DevOpsIndexer = None
log_analyzer: LogAnalyzer = None
shopify_client: ShopifyClient = None
tools: CodeTools = None
llm_client: OpenAI = None

# Active crawl jobs + queue
crawl_jobs: Dict[str, CrawlJob] = {}
crawl_queue: List[str] = []  # ordered list of job_ids waiting to run
_crawl_running: bool = False

# Product sync jobs
product_sync_jobs: Dict[str, ProductSyncJob] = {}

# Classification jobs
classification_jobs: Dict[str, ClassificationJob] = {}

# Translation jobs
translation_jobs: Dict[str, TranslationJob] = {}

# DevOps jobs
devops_index_jobs: Dict[str, DevOpsIndexJob] = {}
log_analysis_jobs: Dict[str, LogAnalysisJob] = {}

# Agent tasks
agent_tasks: Dict[str, AgentTask] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, indexer, web_indexer, competitor_indexer, tools, llm_client
    global product_indexer, product_classifier, translator
    global devops_indexer, log_analyzer, shopify_client

    logger.info("Initializing RAG service components...")

    retriever = CodeRetriever(
        qdrant_url=QDRANT_URL,
        qdrant_api_key=QDRANT_API_KEY,
        embedding_model=EMBEDDING_MODEL
    )
    indexer = CodeIndexer(
        qdrant_url=QDRANT_URL,
        qdrant_api_key=QDRANT_API_KEY,
        embedding_model=EMBEDDING_MODEL
    )
    # Share the embedding model from CodeIndexer to avoid loading it twice (~440MB)
    web_indexer = WebIndexer(
        qdrant_url=QDRANT_URL,
        qdrant_api_key=QDRANT_API_KEY,
        model=indexer.model,
    )
    competitor_indexer = WebIndexer(
        qdrant_url=QDRANT_URL,
        qdrant_api_key=QDRANT_API_KEY,
        model=indexer.model,
        collection_name="competitor_products",
    )
    product_indexer = ProductIndexer(
        qdrant_url=QDRANT_URL,
        qdrant_api_key=QDRANT_API_KEY,
        model=indexer.model,
    )
    tools = CodeTools(repos_base_path=REPOS_PATH)
    llm_client = OpenAI(
        base_url=VLLM_BASE_URL,
        api_key=LITELLM_API_KEY,
    )

    # Shopify client (optional)
    if SHOPIFY_SHOP_DOMAIN and SHOPIFY_ACCESS_TOKEN:
        shopify_client = ShopifyClient(
            shop_domain=SHOPIFY_SHOP_DOMAIN,
            access_token=SHOPIFY_ACCESS_TOKEN,
        )
        logger.info(f"Shopify client configured for {SHOPIFY_SHOP_DOMAIN}")

    # Product classifier + translator (use LLM)
    product_classifier = ProductClassifier(
        qdrant_client=web_indexer.client,
        model=indexer.model,
        llm_client=llm_client,
    )
    translator = TranslationPipeline(llm_client=llm_client)

    # DevOps indexer + log analyzer
    devops_indexer = DevOpsIndexer(
        qdrant_url=QDRANT_URL,
        qdrant_api_key=QDRANT_API_KEY,
        model=indexer.model,
    )
    log_analyzer = LogAnalyzer(llm_client=llm_client, devops_indexer=devops_indexer)

    # Pre-load BM25 sparse encoder so first search is fast
    try:
        from sparse_encoder import get_bm25_model
        get_bm25_model()
        logger.info("BM25 sparse encoder loaded")
    except Exception as e:
        logger.warning(f"BM25 preload failed (will load on first search): {e}")

    # Reranker (optional, ~568MB model)
    if RERANKER_ENABLED:
        try:
            init_reranker(RERANKER_MODEL)
            logger.info(f"Reranker loaded: {RERANKER_MODEL}")
        except Exception as e:
            logger.warning(f"Reranker load failed (search will work without reranking): {e}")

    logger.info("RAG service initialized successfully")

    yield


app = FastAPI(
    title="Pocharlies Qdrant RAG Service",
    description="Web indexing, code retrieval, and tool execution for RAG pipelines",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve embedded dashboard
app.mount("/static", StaticFiles(directory="static"), name="static")


# Request/Response Models
class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 8
    repo: Optional[str] = None
    language: Optional[str] = None
    rerank: bool = True


class RetrieveResponse(BaseModel):
    results: List[dict]


class IndexRequest(BaseModel):
    repo_path: str
    repo_name: Optional[str] = None


class DeleteRepoRequest(BaseModel):
    repo_name: str


class ChatRequest(BaseModel):
    query: str
    model: str = "qwen-coder"
    use_rag: bool = True
    use_tools: bool = True
    repo: Optional[str] = None
    max_tokens: int = 4096


class ToolRequest(BaseModel):
    tool_name: str
    arguments: dict


class WebIndexRequest(BaseModel):
    url: str
    max_depth: int = 2
    max_pages: int = 50
    smart_mode: bool = True


class WebSearchRequest(BaseModel):
    query: str
    top_k: int = 5
    domain: Optional[str] = None
    rerank: bool = True


class UnifiedSearchRequest(BaseModel):
    query: str
    top_k: int = 10
    collection: str = "all"  # "all" | "web" | "code" | "products" | "competitors" | "devops"
    domain: Optional[str] = None
    repo: Optional[str] = None
    language: Optional[str] = None
    brand: Optional[str] = None
    category: Optional[str] = None
    rerank: bool = True


class AgentTaskRequest(BaseModel):
    prompt: str


class AgentContextRequest(BaseModel):
    message: str


class ProductSyncRequest(BaseModel):
    sync_type: str = "full"  # "full" | "incremental"
    since: Optional[str] = None  # ISO timestamp for incremental


class ProductSearchRequest(BaseModel):
    query: str
    top_k: int = 5
    brand: Optional[str] = None
    category: Optional[str] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    rerank: bool = True


class CompetitorCrawlRequest(BaseModel):
    url: str
    max_depth: int = 2
    max_pages: int = 50
    smart_mode: bool = True


class ClassifyRequest(BaseModel):
    domain: str
    collection: str = "competitor_products"  # which collection to extract from
    batch_size: int = 5


class EntityResolveRequest(BaseModel):
    domain: str


class TranslateRequest(BaseModel):
    texts: List[str]
    source_lang: str = "en"
    target_lang: str = "es"


class DevOpsIndexRequest(BaseModel):
    path: str
    recursive: bool = True


class DevOpsSearchRequest(BaseModel):
    query: str
    top_k: int = 5
    doc_type: Optional[str] = None


class LogAnalyzeRequest(BaseModel):
    log_text: str
    service: str = "unknown"


# ── Dashboard ─────────────────────────────────────────────────

@app.get("/")
async def dashboard():
    """Serve the embedded RAG dashboard."""
    return FileResponse("static/index.html")


@app.get("/api")
async def api_info():
    return {
        "service": "Pocharlies Qdrant RAG Service",
        "version": "2.0.0",
        "endpoints": [
            "/health - Health check",
            "/retrieve - Retrieve code chunks",
            "/index - Index a repository",
            "/delete - Delete a repository from index",
            "/tool - Execute a tool",
            "/tools - Get available tools",
            "/chat - Chat with RAG context",
            "/web/index-url - Start web crawl job",
            "/web/status/{job_id} - SSE crawl progress",
            "/web/sources - List indexed web domains",
            "/web/source/{domain} - Delete web domain",
            "/web/search - Search web content",
            "/search - Unified hybrid search (cross-collection)",
            "/web/collections - Collection stats",
            "/web/jobs - List crawl jobs",
            "/web/jobs/{job_id}/stop - Stop a running/queued crawl job",
            "/web/jobs/{job_id}/resume - Resume a stopped crawl job",
            "/web/logs/{job_id} - Get crawl logs",
        ]
    }


# ── Health ────────────────────────────────────────────────────

@app.get("/health")
async def health():
    qdrant_ok = False
    qdrant_version = None
    qdrant_collections = 0
    try:
        import httpx as _httpx
        headers = {}
        if QDRANT_API_KEY:
            headers["api-key"] = QDRANT_API_KEY
        resp = _httpx.get(f"{QDRANT_URL}/collections", headers=headers, timeout=3.0)
        if resp.status_code == 200:
            qdrant_ok = True
            qdrant_collections = len(resp.json().get("result", {}).get("collections", []))
        # Get Qdrant version
        resp_info = _httpx.get(f"{QDRANT_URL}/", headers=headers, timeout=3.0)
        if resp_info.status_code == 200:
            qdrant_version = resp_info.json().get("version")
    except Exception:
        pass

    # Get per-collection stats
    collection_stats = {}
    if qdrant_ok:
        try:
            collection_stats = {c["name"]: c for c in web_indexer.get_collection_stats()}
        except Exception:
            pass

    return {
        "status": "healthy" if qdrant_ok else "degraded",
        "service": "pocharlies-qdrant-rag",
        "qdrant": {
            "status": "healthy" if qdrant_ok else "unreachable",
            "url": QDRANT_URL,
            "version": qdrant_version,
            "collections_count": qdrant_collections,
            "collections": collection_stats,
        },
        "services": {
            "reranker": RERANKER_ENABLED and get_reranker() is not None,
            "shopify": shopify_client is not None,
            "devops_indexer": devops_indexer is not None,
            "product_indexer": product_indexer is not None,
        },
    }


# ── Code Retrieval ────────────────────────────────────────────

@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_code(request: RetrieveRequest):
    """Retrieve relevant code chunks for a query"""
    try:
        reranker = get_reranker()
        effective_top_k = request.top_k * 4 if (reranker and request.rerank) else request.top_k

        results = retriever.retrieve(
            query=request.query,
            top_k=effective_top_k,
            repo_filter=request.repo,
            language_filter=request.language
        )

        result_dicts = [
            {
                "path": r.path,
                "repo": r.repo,
                "start_line": r.start_line,
                "end_line": r.end_line,
                "text": r.text,
                "score": r.score,
                "symbols": r.symbols
            }
            for r in results
        ]

        if reranker and request.rerank and result_dicts:
            result_dicts = reranker.rerank(request.query, result_dicts, top_k=request.top_k)

        return RetrieveResponse(results=result_dicts)
    except Exception as e:
        logger.error(f"Error retrieving: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index")
async def index_repository(request: IndexRequest):
    """Index a repository into the vector database"""
    try:
        chunks = indexer.index_repository(
            repo_path=request.repo_path,
            repo_name=request.repo_name
        )
        return {"status": "success", "chunks_indexed": chunks}
    except Exception as e:
        logger.error(f"Error indexing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/delete")
async def delete_repository(request: DeleteRepoRequest):
    """Delete a repository from the vector database"""
    try:
        success = indexer.delete_repository(request.repo_name)
        if success:
            return {"status": "success", "message": f"Deleted repository: {request.repo_name}"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete repository")
    except Exception as e:
        logger.error(f"Error deleting: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tool")
async def execute_tool(request: ToolRequest):
    """Execute a tool directly"""
    tool_map = {
        "search_repo": tools.search_repo,
        "read_file": tools.read_file,
        "list_files": tools.list_files,
        "run_command": tools.run_command
    }

    if request.tool_name not in tool_map:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown tool: {request.tool_name}. Available: {list(tool_map.keys())}"
        )

    try:
        result = tool_map[request.tool_name](**request.arguments)
        return {"result": result}
    except Exception as e:
        logger.error(f"Error executing tool: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tools")
async def get_tools():
    """Get available tool definitions"""
    return {"tools": TOOL_DEFINITIONS}


@app.post("/chat")
async def chat_with_rag(request: ChatRequest):
    """Chat endpoint with RAG context injection and tool support."""
    messages = []

    system_prompt = """You are a senior software engineer assistant with access to the codebase through RAG context and tools.

When answering:
1. Always cite file paths when referencing code
2. Use tools to verify information when uncertain
3. Be precise about line numbers
4. If context is insufficient, say what additional files you need

Available tools: search_repo, read_file, list_files, run_command"""

    messages.append({"role": "system", "content": system_prompt})

    if request.use_rag:
        try:
            rag_results = retriever.retrieve(
                query=request.query,
                top_k=8,
                repo_filter=request.repo
            )

            if rag_results:
                context_block = "\n\n".join([
                    f"[{r.repo}/{r.path}#L{r.start_line}-{r.end_line}]\n```\n{r.text}\n```"
                    for r in rag_results
                ])
                messages.append({
                    "role": "user",
                    "content": f"RELEVANT CODE CONTEXT:\n{context_block}\n\nQUESTION: {request.query}"
                })
            else:
                messages.append({"role": "user", "content": request.query})
        except Exception as e:
            logger.warning(f"RAG retrieval failed, continuing without context: {e}")
            messages.append({"role": "user", "content": request.query})
    else:
        messages.append({"role": "user", "content": request.query})

    try:
        call_kwargs = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": 0.2
        }

        if request.use_tools:
            call_kwargs["tools"] = TOOL_DEFINITIONS
            call_kwargs["tool_choice"] = "auto"

        response = llm_client.chat.completions.create(**call_kwargs)

        message = response.choices[0].message

        if message.tool_calls:
            tool_results = []
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                tool_map = {
                    "search_repo": tools.search_repo,
                    "read_file": tools.read_file,
                    "list_files": tools.list_files,
                    "run_command": tools.run_command
                }

                if func_name in tool_map:
                    result = tool_map[func_name](**func_args)
                    tool_results.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "content": result
                    })

            messages.append(message.model_dump())
            messages.extend(tool_results)

            final_response = llm_client.chat.completions.create(
                model=request.model,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=0.2
            )

            return {
                "response": final_response.choices[0].message.content,
                "tools_used": [tc.function.name for tc in message.tool_calls],
                "model": request.model
            }

        return {
            "response": message.content,
            "tools_used": [],
            "model": request.model
        }
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get collection statistics"""
    try:
        collection_info = retriever.client.get_collection(retriever.collection_name)
        return {
            "collection": retriever.collection_name,
            "points_count": collection_info.points_count,
            "vectors_count": collection_info.vectors_count,
            "status": collection_info.status
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Web Indexing Endpoints ──────────────────────────────────────


def _cleanup_old_jobs():
    """Remove completed/failed jobs older than 1 hour, stopped jobs older than 4 hours."""
    now = datetime.now(timezone.utc)
    stale = []
    for jid, job in crawl_jobs.items():
        if job.status in ("completed", "failed", "stopped") and job.ended_at:
            try:
                ended = datetime.fromisoformat(job.ended_at)
                max_age = 14400 if job.status == "stopped" else 3600
                if (now - ended).total_seconds() > max_age:
                    stale.append(jid)
            except (ValueError, TypeError):
                pass
    for jid in stale:
        del crawl_jobs[jid]
    if stale:
        logger.info(f"Cleaned up {len(stale)} old crawl jobs")


async def _process_crawl_queue():
    """Process crawl jobs one at a time from the queue."""
    global _crawl_running
    if _crawl_running:
        return
    _crawl_running = True

    try:
        while crawl_queue:
            _cleanup_old_jobs()

            job_id = crawl_queue[0]
            job = crawl_jobs.get(job_id)
            if not job:
                crawl_queue.pop(0)
                continue

            def progress_cb(j: CrawlJob):
                crawl_jobs[job_id] = j
                j.job_id = job_id

            # Build resume state if this is a resumed job
            initial_visited = None
            initial_bfs_queue = None
            if job.resumed_from:
                old_job = crawl_jobs.get(job.resumed_from)
                if old_job and old_job.visited_urls:
                    from collections import deque
                    initial_visited = set(old_job.visited_urls)
                    initial_bfs_queue = deque(
                        [(url, depth) for url, depth in old_job.pending_bfs_queue]
                    )
                    job.log(f"Resume state loaded: {len(initial_visited)} visited URLs, {len(initial_bfs_queue)} pending in BFS queue")

            # Use competitor_indexer for jobs tagged as competitor crawls
            active_indexer = competitor_indexer if getattr(job, '_competitor', False) else web_indexer

            result = await active_indexer.crawl_and_index(
                start_url=job.url,
                max_depth=job.max_depth,
                max_pages=job.max_pages,
                progress_callback=progress_cb,
                smart_mode=getattr(job, 'smart_mode', True),
                llm_client=llm_client,
                initial_visited=initial_visited,
                initial_bfs_queue=initial_bfs_queue,
            )
            result.job_id = job_id
            result.resumed_from = job.resumed_from
            crawl_jobs[job_id] = result

            crawl_queue.pop(0)
    finally:
        _crawl_running = False


@app.post("/web/index-url")
async def web_index_url(request: WebIndexRequest):
    """Start or queue a web crawl job. Returns job_id for progress tracking."""
    if not validators.url(request.url):
        raise HTTPException(status_code=400, detail=f"Invalid URL: {request.url}")

    if request.max_depth < 0 or request.max_depth > 5:
        raise HTTPException(status_code=400, detail="max_depth must be 0-5")

    if request.max_pages < 1 or request.max_pages > 25000:
        raise HTTPException(status_code=400, detail="max_pages must be 1-25000")

    import uuid
    job_id = uuid.uuid4().hex[:12]

    placeholder_job = CrawlJob(
        job_id=job_id,
        url=request.url,
        max_depth=request.max_depth,
        max_pages=request.max_pages,
        status="queued",
        smart_mode=request.smart_mode,
    )
    crawl_jobs[job_id] = placeholder_job
    crawl_queue.append(job_id)

    queue_pos = len(crawl_queue)
    logger.info(f"Queued crawl job {job_id} for {request.url} (position {queue_pos})")

    asyncio.create_task(_process_crawl_queue())

    return {
        "job_id": job_id,
        "status": "queued",
        "url": request.url,
        "queue_position": queue_pos,
    }


@app.get("/web/status/{job_id}")
async def web_crawl_status(job_id: str):
    """SSE stream of crawl job progress."""
    if job_id not in crawl_jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    async def event_stream():
        prev_data = None
        while True:
            job = crawl_jobs.get(job_id)
            if not job:
                yield f"data: {json.dumps({'error': 'Job disappeared'})}\n\n"
                break

            data = json.dumps(job.to_dict())
            if data != prev_data:
                yield f"data: {data}\n\n"
                prev_data = data

            if job.status in ("completed", "failed", "stopped"):
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/web/sources")
async def web_sources():
    """List indexed web sources grouped by domain."""
    return {"sources": web_indexer.get_sources()}


@app.delete("/web/source/{domain:path}")
async def web_delete_source(domain: str):
    """Delete all indexed content from a domain."""
    success = web_indexer.delete_source(domain)
    if success:
        return {"status": "deleted", "domain": domain}
    raise HTTPException(status_code=500, detail=f"Failed to delete domain: {domain}")


@app.post("/web/search")
async def web_search(request: WebSearchRequest):
    """Search web_pages collection."""
    reranker = get_reranker()
    effective_top_k = request.top_k * 4 if (reranker and request.rerank) else request.top_k

    results = web_indexer.search(
        query=request.query,
        top_k=effective_top_k,
        domain_filter=request.domain,
    )

    if reranker and request.rerank and results:
        results = reranker.rerank(request.query, results, top_k=request.top_k)

    return {"results": results, "query": request.query}


def _cross_collection_rrf(results: List[dict], top_k: int, k: int = 60) -> List[dict]:
    """Client-side RRF to merge results from multiple collections."""
    for i, r in enumerate(results):
        r["_rrf_score"] = 1.0 / (k + i + 1)

    # Group by collection, re-rank within each group by original order
    by_collection: Dict[str, List[dict]] = {}
    for r in results:
        col = r.get("collection", "unknown")
        by_collection.setdefault(col, []).append(r)

    # Assign per-collection RRF ranks
    merged = []
    for col, items in by_collection.items():
        for rank, item in enumerate(items):
            item["_rrf_score"] = 1.0 / (k + rank + 1)
            merged.append(item)

    # Sort by RRF score descending, take top_k
    merged.sort(key=lambda x: x["_rrf_score"], reverse=True)
    merged = merged[:top_k]

    # Normalize scores: top result = 1.0
    if merged:
        max_score = merged[0]["_rrf_score"]
        for r in merged:
            r["score"] = round(r["_rrf_score"] / max_score, 4) if max_score > 0 else 0
            del r["_rrf_score"]

    return merged


@app.post("/search")
async def unified_search(request: UnifiedSearchRequest):
    """Unified hybrid search across all collections."""
    results = []

    if request.collection in ("all", "web"):
        try:
            web_results = web_indexer.search(
                query=request.query,
                top_k=request.top_k,
                domain_filter=request.domain,
            )
            for r in web_results:
                r["collection"] = "web_pages"
                r["source_type"] = "web"
            results.extend(web_results)
        except Exception as e:
            logger.warning(f"Web search failed: {e}")

    if request.collection in ("all", "code"):
        try:
            code_results = retriever.retrieve(
                query=request.query,
                top_k=request.top_k,
                repo_filter=request.repo,
                language_filter=request.language,
            )
            for r in code_results:
                results.append({
                    "path": r.path,
                    "repo": r.repo,
                    "start_line": r.start_line,
                    "end_line": r.end_line,
                    "text": r.text,
                    "score": round(r.score, 4),
                    "symbols": r.symbols,
                    "collection": "code_index",
                    "source_type": "code",
                })
        except Exception as e:
            logger.warning(f"Code search failed: {e}")

    if request.collection in ("all", "products"):
        try:
            prod_results = product_indexer.search(
                query=request.query,
                top_k=request.top_k,
                brand_filter=request.brand,
                category_filter=request.category,
            )
            for r in prod_results:
                r["collection"] = "product_catalog"
            results.extend(prod_results)
        except Exception as e:
            logger.warning(f"Product search failed: {e}")

    if request.collection in ("all", "competitors"):
        try:
            comp_results = competitor_indexer.search(
                query=request.query,
                top_k=request.top_k,
                domain_filter=request.domain,
            )
            for r in comp_results:
                r["collection"] = "competitor_products"
                r["source_type"] = "competitor"
            results.extend(comp_results)
        except Exception as e:
            logger.warning(f"Competitor search failed: {e}")

    if request.collection in ("all", "devops"):
        try:
            devops_results = devops_indexer.search(
                query=request.query,
                top_k=request.top_k,
            )
            for r in devops_results:
                r["collection"] = "devops_docs"
            results.extend(devops_results)
        except Exception as e:
            logger.warning(f"DevOps search failed: {e}")

    if request.collection == "all" and results:
        results = _cross_collection_rrf(results, request.top_k)

    # Rerank merged results
    reranker = get_reranker()
    if reranker and request.rerank and results:
        results = reranker.rerank(request.query, results, top_k=request.top_k)

    return {"results": results, "query": request.query, "collection": request.collection}


@app.get("/web/collections")
async def web_collections():
    """Get stats for all RAG collections (code_index + web_pages)."""
    return {"collections": web_indexer.get_collection_stats()}


@app.post("/web/jobs/{job_id}/stop")
async def web_stop_job(job_id: str):
    """Stop a running or queued crawl job."""
    job = crawl_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    if job.status == "queued":
        if job_id in crawl_queue:
            crawl_queue.remove(job_id)
        job.status = "stopped"
        job.ended_at = datetime.now(timezone.utc).isoformat()
        job.log("Job cancelled while still queued")
        return {"status": "stopped", "job_id": job_id, "message": "Queued job cancelled"}

    if job.status == "running":
        job.cancel_requested = True
        job.log("Stop requested — will halt after current page completes")
        return {"status": "stopping", "job_id": job_id, "message": "Stop signal sent, job will halt gracefully"}

    raise HTTPException(
        status_code=400,
        detail=f"Job is already {job.status}, cannot stop"
    )


@app.post("/web/jobs/{job_id}/resume")
async def web_resume_job(job_id: str):
    """Resume a previously stopped crawl job."""
    old_job = crawl_jobs.get(job_id)
    if not old_job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    if old_job.status != "stopped":
        raise HTTPException(
            status_code=400,
            detail=f"Only stopped jobs can be resumed. Job is: {old_job.status}"
        )

    import uuid
    new_job_id = uuid.uuid4().hex[:12]

    new_job = CrawlJob(
        job_id=new_job_id,
        url=old_job.url,
        max_depth=old_job.max_depth,
        max_pages=old_job.max_pages,
        status="queued",
        smart_mode=old_job.smart_mode,
        resumed_from=job_id,
        pages_scraped=old_job.pages_scraped,
        pages_visited=old_job.pages_visited,
        pages_indexed=old_job.pages_indexed,
        chunks_indexed=old_job.chunks_indexed,
        chunks_queued=old_job.chunks_queued,
    )
    new_job.log(f"Resumed from job {job_id} ({old_job.pages_visited} pages already visited)")

    crawl_jobs[new_job_id] = new_job
    crawl_queue.append(new_job_id)

    queue_pos = len(crawl_queue)
    logger.info(f"Resume job {new_job_id} from {job_id} for {old_job.url} (position {queue_pos})")

    asyncio.create_task(_process_crawl_queue())

    return {
        "job_id": new_job_id,
        "resumed_from": job_id,
        "status": "queued",
        "url": old_job.url,
        "queue_position": queue_pos,
        "pages_already_visited": old_job.pages_visited,
    }


@app.get("/web/jobs")
async def web_jobs():
    """List all crawl jobs with queue position."""
    jobs = []
    for job in crawl_jobs.values():
        d = job.to_dict()
        if job.job_id in crawl_queue:
            d["queue_position"] = crawl_queue.index(job.job_id) + 1
        else:
            d["queue_position"] = None
        jobs.append(d)
    return {"jobs": jobs}


@app.get("/web/logs/{job_id}")
async def web_logs(job_id: str, offset: int = 0):
    """Get logs for a crawl job. Use offset to paginate."""
    job = crawl_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    logs = job.logs[offset:]
    return {
        "job_id": job_id,
        "logs": logs,
        "total": len(job.logs),
        "offset": offset,
        "status": job.status,
    }


# ── Product Catalog Endpoints ─────────────────────────────────


@app.post("/products/sync")
async def product_sync(request: ProductSyncRequest):
    """Start a product catalog sync from Shopify."""
    if not shopify_client:
        raise HTTPException(
            status_code=400,
            detail="Shopify not configured. Set SHOPIFY_SHOP_DOMAIN and SHOPIFY_ACCESS_TOKEN.",
        )

    def progress_cb(j: ProductSyncJob):
        product_sync_jobs[j.job_id] = j

    if request.sync_type == "incremental" and request.since:
        job = await product_indexer.incremental_sync(
            shopify_client, request.since, progress_callback=progress_cb
        )
    else:
        job = await product_indexer.index_all_products(
            shopify_client, progress_callback=progress_cb
        )

    product_sync_jobs[job.job_id] = job
    return job.to_dict()


@app.get("/products/sync/{job_id}")
async def product_sync_status(job_id: str):
    """Get product sync job status."""
    job = product_sync_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job.to_dict()


@app.post("/products/search")
async def product_search(request: ProductSearchRequest):
    """Search product catalog with structured filters."""
    reranker = get_reranker()
    effective_top_k = request.top_k * 4 if (reranker and request.rerank) else request.top_k

    results = product_indexer.search(
        query=request.query,
        top_k=effective_top_k,
        brand_filter=request.brand,
        category_filter=request.category,
        min_price=request.min_price,
        max_price=request.max_price,
    )

    if reranker and request.rerank and results:
        results = reranker.rerank(request.query, results, top_k=request.top_k)

    return {"results": results, "query": request.query}


@app.get("/products/stats")
async def product_stats():
    """Get product catalog stats."""
    return product_indexer.get_stats()


# ── Competitor Indexing Endpoints ─────────────────────────────


@app.post("/competitor/index-url")
async def competitor_index_url(request: CompetitorCrawlRequest):
    """Start crawling a competitor site into competitor_products collection."""
    if not validators.url(request.url):
        raise HTTPException(status_code=400, detail=f"Invalid URL: {request.url}")

    import uuid
    job_id = uuid.uuid4().hex[:12]

    placeholder_job = CrawlJob(
        job_id=job_id,
        url=request.url,
        max_depth=request.max_depth,
        max_pages=request.max_pages,
        status="queued",
        smart_mode=request.smart_mode,
    )
    crawl_jobs[job_id] = placeholder_job
    crawl_queue.append(job_id)

    # Tag this job for competitor indexer
    placeholder_job._competitor = True

    asyncio.create_task(_process_crawl_queue())

    return {"job_id": job_id, "status": "queued", "url": request.url, "collection": "competitor_products"}


@app.get("/competitor/sources")
async def competitor_sources():
    """List indexed competitor domains."""
    return {"sources": competitor_indexer.get_sources()}


@app.delete("/competitor/source/{domain:path}")
async def competitor_delete_source(domain: str):
    """Delete competitor content for a domain."""
    success = competitor_indexer.delete_source(domain)
    if success:
        return {"status": "deleted", "domain": domain}
    raise HTTPException(status_code=500, detail=f"Failed to delete domain: {domain}")


# ── Classification Endpoints ──────────────────────────────────


@app.post("/classify/extract")
async def classify_extract(request: ClassifyRequest):
    """Start product extraction job for a domain."""
    def progress_cb(j: ClassificationJob):
        classification_jobs[j.job_id] = j

    job = await product_classifier.extract_products_from_collection(
        domain=request.domain,
        collection_name=request.collection,
        batch_size=request.batch_size,
        progress_callback=progress_cb,
    )
    classification_jobs[job.job_id] = job
    return job.to_dict()


@app.get("/classify/status/{job_id}")
async def classify_status(job_id: str):
    """Get classification job status."""
    job = classification_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job.to_dict()


@app.get("/classify/products/{job_id}")
async def classify_products(job_id: str):
    """Get extracted products from a classification job."""
    job = classification_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return {"products": job.results, "count": len(job.results)}


@app.post("/classify/resolve")
async def classify_resolve(request: EntityResolveRequest):
    """Run entity resolution for a domain's extracted products against the catalog."""
    # Find the latest classification job for this domain
    domain_job = None
    for job in classification_jobs.values():
        if job.domain == request.domain and job.status == "completed":
            domain_job = job

    if not domain_job or not domain_job.results:
        raise HTTPException(
            status_code=404,
            detail=f"No completed extraction found for domain: {request.domain}. Run /classify/extract first.",
        )

    matches = await product_classifier.resolve_entities(
        extracted_products=domain_job.results,
        product_indexer=product_indexer,
    )
    report = ProductClassifier.generate_price_report(matches)
    return report


# ── Translation Endpoints ─────────────────────────────────────


@app.post("/translate/batch")
async def translate_batch(request: TranslateRequest):
    """Translate a list of texts."""
    job = await translator.translate_batch(
        texts=request.texts,
        source_lang=request.source_lang,
        target_lang=request.target_lang,
    )
    translation_jobs[job.job_id] = job

    return {
        "job_id": job.job_id,
        "status": job.status,
        "translations": job.results,
    }


@app.get("/translate/status/{job_id}")
async def translate_status(job_id: str):
    """Get translation job status."""
    job = translation_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job.to_dict()


@app.post("/translate/normalize")
async def translate_normalize(product: dict):
    """Normalize specs in a product dict."""
    return TranslationPipeline.normalize_specs(product)


# ── DevOps Endpoints ──────────────────────────────────────────


@app.post("/devops/index")
async def devops_index(request: DevOpsIndexRequest):
    """Index documents from a path."""
    def progress_cb(j: DevOpsIndexJob):
        devops_index_jobs[j.job_id] = j

    job = await devops_indexer.index_path(
        path=request.path,
        recursive=request.recursive,
        progress_callback=progress_cb,
    )
    devops_index_jobs[job.job_id] = job
    return job.to_dict()


@app.get("/devops/sources")
async def devops_sources():
    """List indexed DevOps document sources."""
    return {"sources": devops_indexer.get_sources()}


@app.delete("/devops/source/{source_path:path}")
async def devops_delete_source(source_path: str):
    """Delete a DevOps document source."""
    success = devops_indexer.delete_source(source_path)
    if success:
        return {"status": "deleted", "source_path": source_path}
    raise HTTPException(status_code=500, detail=f"Failed to delete: {source_path}")


@app.post("/devops/search")
async def devops_search(request: DevOpsSearchRequest):
    """Search DevOps docs (hybrid)."""
    results = devops_indexer.search(
        query=request.query,
        top_k=request.top_k,
        doc_type_filter=request.doc_type,
    )
    return {"results": results, "query": request.query}


@app.post("/devops/analyze-logs")
async def devops_analyze_logs(request: LogAnalyzeRequest):
    """Start log analysis job."""
    job = await log_analyzer.analyze_logs(
        log_text=request.log_text,
        source_service=request.service,
    )
    log_analysis_jobs[job.job_id] = job
    return {
        "job_id": job.job_id,
        "status": job.status,
        "categories": job.categories,
        "results_count": len(job.results),
    }


@app.get("/devops/analyze-logs/{job_id}")
async def devops_log_analysis_result(job_id: str):
    """Get log analysis results."""
    job = log_analysis_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return {
        **job.to_dict(),
        "results": job.results,
    }


# ── Agent Endpoints ───────────────────────────────────────────


def _cleanup_old_agent_tasks():
    now = datetime.now(timezone.utc)
    stale = []
    for tid, task in agent_tasks.items():
        if task.status in ("completed", "failed", "cancelled") and task.ended_at:
            try:
                ended = datetime.fromisoformat(task.ended_at)
                if (now - ended).total_seconds() > 7200:
                    stale.append(tid)
            except (ValueError, TypeError):
                pass
    for tid in stale:
        del agent_tasks[tid]


@app.get("/agent/status")
async def agent_service_status():
    """Check if LLM is available for agent tasks."""
    try:
        models = llm_client.models.list()
        model_id = models.data[0].id if models.data else None
    except Exception:
        model_id = None

    if model_id:
        return {"available": True, "model_id": model_id}
    return {"available": False, "reason": "No LLM model available. Configure LLM_BASE_URL."}


@app.post("/agent/task")
async def create_agent_task(request: AgentTaskRequest):
    """Create and start a new agent task."""
    _cleanup_old_agent_tasks()

    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    running = sum(1 for t in agent_tasks.values() if t.status == "running")
    if running >= 3:
        raise HTTPException(status_code=429, detail="Maximum 3 concurrent agent tasks")

    import uuid
    task_id = uuid.uuid4().hex[:12]
    task = AgentTask(task_id=task_id, prompt=prompt)
    agent_tasks[task_id] = task

    asyncio.create_task(run_agent_task(task, llm_client, web_indexer, retriever, product_indexer, devops_indexer, log_analyzer))
    logger.info(f"Started agent task {task_id}: {prompt[:80]}")

    return {"task_id": task_id, "status": "running"}


@app.get("/agent/tasks")
async def list_agent_tasks():
    """List all agent tasks."""
    tasks = [t.to_dict() for t in agent_tasks.values()]
    tasks.sort(key=lambda t: t.get("started_at", ""), reverse=True)
    return {"tasks": tasks}


@app.get("/agent/task/{task_id}")
async def get_agent_task(task_id: str):
    task = agent_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    return task.to_dict()


@app.get("/agent/task/{task_id}/stream")
async def agent_task_stream(task_id: str):
    """SSE stream of agent task progress."""
    if task_id not in agent_tasks:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    async def event_stream():
        prev_data = None
        while True:
            task = agent_tasks.get(task_id)
            if not task:
                yield f"data: {json.dumps({'error': 'Task disappeared'})}\n\n"
                break

            data = json.dumps(task.to_dict())
            if data != prev_data:
                yield f"data: {data}\n\n"
                prev_data = data

            if task.status in ("completed", "failed", "cancelled"):
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.get("/agent/task/{task_id}/logs")
async def agent_task_logs(task_id: str, offset: int = 0):
    task = agent_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    return {
        "task_id": task_id,
        "logs": task.logs[offset:],
        "total": len(task.logs),
        "offset": offset,
        "status": task.status,
    }


@app.post("/agent/task/{task_id}/stop")
async def stop_agent_task(task_id: str):
    task = agent_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    if task.status != "running":
        raise HTTPException(status_code=400, detail=f"Task is not running (status: {task.status})")
    task._cancel_event.set()
    task.log("Cancel requested by user")
    return {"task_id": task_id, "status": "cancelling"}


@app.post("/agent/task/{task_id}/context")
async def inject_agent_context(task_id: str, request: AgentContextRequest):
    task = agent_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    if task.status != "running":
        raise HTTPException(status_code=400, detail=f"Task is not running (status: {task.status})")
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    await task._context_queue.put(message)
    task.log(f"Context injected: {message[:200]}")
    return {"task_id": task_id, "injected": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
