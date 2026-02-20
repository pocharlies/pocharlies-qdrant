"""
RAG Service FastAPI Application
Provides endpoints for code retrieval, web indexing, and tool execution
"""

import os
import json
import asyncio
import logging
import uuid
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
import httpx

from indexer import CodeIndexer
from retriever import CodeRetriever, CodeTools, TOOL_DEFINITIONS
from web_indexer import WebIndexer, CrawlJob
from agent import AgentTask, AgentServices, create_agent
from agent.runner import run_task
from agent.redis_session import RedisSession
from agent.session_store import SessionStore
from product_indexer import ProductIndexer, ProductSyncJob
from product_classifier import ProductClassifier, ClassificationJob
from translator import TranslationPipeline, TranslationJob, GlossaryStore, AIRSOFT_GLOSSARY_EN_ES
from devops_indexer import DevOpsIndexer, LogAnalyzer, DevOpsIndexJob, LogAnalysisJob
from jira_indexer import JiraIndexer, JiraImportJob
from confluence_indexer import ConfluenceIndexer, ConfluenceImportJob
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
JIRA_BASE_URL = os.getenv("JIRA_BASE_URL", "")
JIRA_EMAIL = os.getenv("JIRA_EMAIL", "")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN", "")
JIRA_SYNC_ENABLED = os.getenv("JIRA_SYNC_ENABLED", "false").lower() == "true"
JIRA_SYNC_HOUR = int(os.getenv("JIRA_SYNC_HOUR", "2"))
CONFLUENCE_SYNC_ENABLED = os.getenv("CONFLUENCE_SYNC_ENABLED", "false").lower() == "true"
CONFLUENCE_SYNC_HOUR = int(os.getenv("CONFLUENCE_SYNC_HOUR", "3"))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Global instances
retriever: CodeRetriever = None
indexer: CodeIndexer = None
web_indexer: WebIndexer = None
competitor_indexer: WebIndexer = None
product_indexer: ProductIndexer = None
product_classifier: ProductClassifier = None
translator: TranslationPipeline = None
glossary_store: GlossaryStore = None
devops_indexer: DevOpsIndexer = None
log_analyzer: LogAnalyzer = None
jira_indexer: JiraIndexer = None
confluence_indexer: ConfluenceIndexer = None
shopify_client: ShopifyClient = None
tools: CodeTools = None
llm_client: OpenAI = None
llm_model_id: str = None  # auto-discovered at startup
rag_agent = None  # Agent SDK instance, created at startup
redis_client = None  # Async Redis client for session persistence
session_store: SessionStore = None

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

# Jira import jobs
jira_import_jobs: Dict[str, JiraImportJob] = {}

# Confluence import jobs
confluence_import_jobs: Dict[str, ConfluenceImportJob] = {}

# Agent tasks
agent_tasks: Dict[str, AgentTask] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, indexer, web_indexer, competitor_indexer, tools, llm_client, llm_model_id
    global product_indexer, product_classifier, translator
    global devops_indexer, log_analyzer, jira_indexer, confluence_indexer, shopify_client, rag_agent
    global redis_client, session_store, glossary_store

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

    # Auto-discover LLM model ID
    try:
        models = llm_client.models.list()
        llm_model_id = models.data[0].id if models.data else None
        if llm_model_id:
            logger.info(f"LLM model discovered: {llm_model_id}")
        else:
            logger.warning("No LLM models available — chat/agent features will fail")
    except Exception as e:
        logger.warning(f"LLM model discovery failed: {e}")
        llm_model_id = None

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
    # translator is initialized after Redis (needs glossary_store)

    # DevOps indexer + log analyzer
    devops_indexer = DevOpsIndexer(
        qdrant_url=QDRANT_URL,
        qdrant_api_key=QDRANT_API_KEY,
        model=indexer.model,
    )
    log_analyzer = LogAnalyzer(llm_client=llm_client, devops_indexer=devops_indexer)

    # Jira indexer (optional)
    jira_indexer = JiraIndexer(
        qdrant_url=QDRANT_URL,
        qdrant_api_key=QDRANT_API_KEY,
        model=indexer.model,
        jira_base_url=JIRA_BASE_URL,
        jira_email=JIRA_EMAIL,
        jira_api_token=JIRA_API_TOKEN,
    )
    if jira_indexer.configured:
        logger.info(f"Jira indexer configured for {JIRA_BASE_URL}")
    else:
        logger.info("Jira indexer: not configured (missing credentials)")

    # Confluence indexer (optional — reuses Jira credentials if dedicated vars are empty)
    confluence_indexer = ConfluenceIndexer(
        qdrant_url=QDRANT_URL,
        qdrant_api_key=QDRANT_API_KEY,
        model=indexer.model,
        llm_base_url=VLLM_BASE_URL,
        llm_api_key=LITELLM_API_KEY,
    )
    if confluence_indexer.configured:
        logger.info(f"Confluence indexer configured for {confluence_indexer._base_url}")
    else:
        logger.info("Confluence indexer: not configured (missing credentials)")

    # Create SDK agent (after llm_client is ready)
    try:
        rag_agent, _model_id = create_agent(VLLM_BASE_URL, LITELLM_API_KEY)
        logger.info(f"Agent SDK initialized with model: {_model_id}")
    except Exception as e:
        logger.warning(f"Agent SDK init failed (agent endpoints will fail): {e}")

    # Redis for agent session persistence + glossary
    try:
        from redis.asyncio import Redis as AsyncRedis
        redis_client = AsyncRedis.from_url(REDIS_URL, decode_responses=False)
        await redis_client.ping()
        session_store = SessionStore(redis_client)
        logger.info(f"Redis connected: {REDIS_URL}")
    except Exception as e:
        logger.warning(f"Redis connection failed ({e}) — agent sessions will not persist")
        redis_client = None
        session_store = None

    # Translation glossary (Redis-backed for custom terms)
    glossary_store = GlossaryStore(redis=redis_client)
    if redis_client:
        custom_terms = await glossary_store.load()
        logger.info(f"Glossary loaded: {len(AIRSOFT_GLOSSARY_EN_ES)} built-in + {len(custom_terms)} custom terms")
    translator = TranslationPipeline(llm_client=llm_client, glossary_store=glossary_store)

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

    # Nightly Jira sync scheduler
    scheduler = None
    if JIRA_SYNC_ENABLED and jira_indexer.configured:
        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler
            from apscheduler.triggers.cron import CronTrigger

            scheduler = AsyncIOScheduler()

            async def _nightly_jira_sync():
                try:
                    logger.info("Nightly Jira sync starting...")
                    job = await jira_indexer.sync_recent(since_hours=25)
                    jira_import_jobs[job.job_id] = job
                    logger.info(f"Nightly Jira sync done: {job.tickets_indexed} tickets, {job.chunks_indexed} chunks")
                except Exception as e:
                    logger.error(f"Nightly Jira sync failed: {e}")

            scheduler.add_job(
                _nightly_jira_sync,
                CronTrigger(hour=JIRA_SYNC_HOUR, minute=0),
                id="nightly_jira_sync",
                replace_existing=True,
            )
            scheduler.start()
            logger.info(f"Jira nightly sync scheduled at {JIRA_SYNC_HOUR}:00 UTC")
        except Exception as e:
            logger.warning(f"Failed to start Jira scheduler: {e}")

    # Nightly Confluence sync scheduler
    if CONFLUENCE_SYNC_ENABLED and confluence_indexer.configured:
        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler
            from apscheduler.triggers.cron import CronTrigger

            if scheduler is None:
                scheduler = AsyncIOScheduler()
                scheduler.start()

            async def _nightly_confluence_sync():
                try:
                    logger.info("Nightly Confluence sync starting...")
                    job = await confluence_indexer.sync_recent(since_hours=25)
                    confluence_import_jobs[job.job_id] = job
                    logger.info(f"Nightly Confluence sync done: {job.pages_indexed} pages, {job.chunks_indexed} chunks")
                except Exception as e:
                    logger.error(f"Nightly Confluence sync failed: {e}")

            scheduler.add_job(
                _nightly_confluence_sync,
                CronTrigger(hour=CONFLUENCE_SYNC_HOUR, minute=0),
                id="nightly_confluence_sync",
                replace_existing=True,
            )
            logger.info(f"Confluence nightly sync scheduled at {CONFLUENCE_SYNC_HOUR}:00 UTC")
        except Exception as e:
            logger.warning(f"Failed to start Confluence scheduler: {e}")

    logger.info("RAG service initialized successfully")

    yield

    # Shutdown scheduler
    if scheduler:
        scheduler.shutdown(wait=False)


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
    model: Optional[str] = None  # auto-discovered at startup
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
    collection: str = "all"  # "all" | "web" | "code" | "products" | "competitors" | "devops" | "jira" | "confluence"
    domain: Optional[str] = None
    repo: Optional[str] = None
    language: Optional[str] = None
    brand: Optional[str] = None
    category: Optional[str] = None
    # Jira-specific filters
    project: Optional[str] = None
    assignee: Optional[str] = None
    issue_type: Optional[str] = None
    # Confluence-specific filters
    space: Optional[str] = None
    author: Optional[str] = None
    department: Optional[str] = None
    topic: Optional[str] = None
    rerank: bool = True


class AgentTaskRequest(BaseModel):
    prompt: str


class AgentContinueRequest(BaseModel):
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
    use_rag: bool = True
    resource_type: str = "product"


class GlossaryEntryRequest(BaseModel):
    source: str
    target: str


class GlossaryBulkRequest(BaseModel):
    entries: Dict[str, str]  # source_term -> target_term


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


class JiraImportRequest(BaseModel):
    project: Optional[str] = None


class JiraSyncRequest(BaseModel):
    since_hours: int = 24


class ConfluenceImportRequest(BaseModel):
    space: Optional[str] = None

class ConfluenceSyncRequest(BaseModel):
    since_hours: int = 24


class JiraSearchRequest(BaseModel):
    query: str = ""
    top_k: int = 8
    project: Optional[str] = None
    assignee: Optional[str] = None
    ticket_key: Optional[str] = None
    issue_type: Optional[str] = None


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

    # Get per-collection stats from all known collections
    collection_stats = {}
    if qdrant_ok:
        try:
            collection_stats = {c["name"]: c for c in web_indexer.get_collection_stats()}
        except Exception:
            pass
        # Add collections from other indexers not covered by web_indexer
        import httpx as _httpx2
        headers2 = {}
        if QDRANT_API_KEY:
            headers2["api-key"] = QDRANT_API_KEY
        all_colls_resp = resp.json().get("result", {}).get("collections", [])
        for coll_info in all_colls_resp:
            cname = coll_info.get("name", "")
            if cname and cname not in collection_stats:
                try:
                    cr = _httpx2.get(f"{QDRANT_URL}/collections/{cname}", headers=headers2, timeout=3.0)
                    if cr.status_code == 200:
                        cdata = cr.json().get("result", {})
                        vec_count = cdata.get("vectors_count")
                        if vec_count is None:
                            vec_count = cdata.get("indexed_vectors_count", 0)
                        collection_stats[cname] = {
                            "name": cname,
                            "points_count": cdata.get("points_count", 0),
                            "vectors_count": vec_count or 0,
                            "status": cdata.get("status", "unknown"),
                        }
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
        "redis": {
            "status": "healthy" if redis_client else "not configured",
        },
        "services": {
            "reranker": RERANKER_ENABLED and get_reranker() is not None,
            "shopify": shopify_client is not None,
            "devops_indexer": devops_indexer is not None,
            "jira_indexer": jira_indexer is not None and jira_indexer.configured,
            "confluence_indexer": confluence_indexer is not None and confluence_indexer.configured,
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
    # Resolve model: use request model, or auto-discovered, or fail
    model = request.model or llm_model_id
    if not model:
        raise HTTPException(status_code=503, detail="No LLM model available. Check LLM_BASE_URL configuration.")

    messages = []

    system_prompt = """You are a knowledgeable assistant with access to indexed content from multiple sources:
- **Product Catalog**: Products from the Shopify store (titles, descriptions, prices, brands)
- **Web Pages**: Crawled website content from various indexed sites
- **Competitor Products**: Product data from competitor stores
- **Code Index**: Source code from indexed repositories
- **DevOps Docs**: DevOps documentation and runbooks

When answering:
1. Use the provided RAG context to give factual, data-driven answers
2. Cite sources (domain, product name, collection) when referencing specific data
3. If comparing prices or products, present the data in a clear table or list
4. If context is insufficient, say what data might be missing and suggest indexing more sources
5. Be specific — use exact names, prices, and URLs from the context"""

    messages.append({"role": "system", "content": system_prompt})

    if request.use_rag:
        context_sections = []

        # Search across ALL collections for relevant context
        try:
            # Product catalog
            prod_results = product_indexer.search(query=request.query, top_k=5)
            if prod_results:
                prod_block = "\n".join([
                    f"- **{r.get('title', 'Unknown')}** | Price: {r.get('price', 'N/A')} | Brand: {r.get('brand', 'N/A')} | {r.get('text', '')[:200]}"
                    for r in prod_results
                ])
                context_sections.append(f"## Product Catalog Results\n{prod_block}")
        except Exception as e:
            logger.warning(f"Product search failed in chat: {e}")

        try:
            # Web pages (crawled content)
            web_results = web_indexer.search(query=request.query, top_k=5)
            if web_results:
                web_block = "\n".join([
                    f"- [{r.get('title', 'Untitled')}]({r.get('url', '')}) ({r.get('domain', 'unknown')})\n  {r.get('text', '')[:300]}"
                    for r in web_results
                ])
                context_sections.append(f"## Web Page Results\n{web_block}")
        except Exception as e:
            logger.warning(f"Web search failed in chat: {e}")

        try:
            # Competitor products
            comp_results = competitor_indexer.search(query=request.query, top_k=5)
            if comp_results:
                comp_block = "\n".join([
                    f"- [{r.get('title', 'Untitled')}]({r.get('url', '')}) ({r.get('domain', 'unknown')})\n  {r.get('text', '')[:300]}"
                    for r in comp_results
                ])
                context_sections.append(f"## Competitor Store Results\n{comp_block}")
        except Exception as e:
            logger.warning(f"Competitor search failed in chat: {e}")

        try:
            # Code (only if query seems code-related)
            code_keywords = {"code", "function", "class", "file", "bug", "error", "import", "def ", "api", "endpoint"}
            if any(kw in request.query.lower() for kw in code_keywords) or request.repo:
                code_results = retriever.retrieve(query=request.query, top_k=5, repo_filter=request.repo)
                if code_results:
                    code_block = "\n\n".join([
                        f"[{r.repo}/{r.path}#L{r.start_line}-{r.end_line}]\n```\n{r.text}\n```"
                        for r in code_results
                    ])
                    context_sections.append(f"## Code Results\n{code_block}")
        except Exception as e:
            logger.warning(f"Code search failed in chat: {e}")

        try:
            # DevOps docs
            devops_keywords = {"deploy", "docker", "kubernetes", "k8s", "ci", "cd", "pipeline", "server", "nginx", "devops", "log", "monitor"}
            if any(kw in request.query.lower() for kw in devops_keywords):
                devops_results = devops_indexer.search(query=request.query, top_k=5)
                if devops_results:
                    devops_block = "\n".join([
                        f"- {r.get('title', 'Untitled')}: {r.get('text', '')[:300]}"
                        for r in devops_results
                    ])
                    context_sections.append(f"## DevOps Docs Results\n{devops_block}")
        except Exception as e:
            logger.warning(f"DevOps search failed in chat: {e}")

        try:
            # Jira tickets
            jira_keywords = {"ticket", "jira", "bug", "task", "story", "sprint", "epic", "issue", "backlog"}
            if any(kw in request.query.lower() for kw in jira_keywords):
                jira_results = jira_indexer.search(query=request.query, top_k=5)
                if jira_results:
                    jira_block = "\n".join([
                        f"- [{r.get('ticket_key', '')}] {r.get('summary', '')} ({r.get('status', '')}) — {r.get('text', '')[:300]}"
                        for r in jira_results
                    ])
                    context_sections.append(f"## Jira Ticket Results\n{jira_block}")
        except Exception as e:
            logger.warning(f"Jira search failed in chat: {e}")

        try:
            # Confluence pages
            confluence_keywords = {"confluence", "wiki", "page", "documentation", "docs", "knowledge", "onboarding", "process"}
            if any(kw in request.query.lower() for kw in confluence_keywords):
                conf_results = confluence_indexer.search(query=request.query, top_k=5)
                if conf_results:
                    conf_block = "\n".join([
                        f"- [{r.get('page_title', '')}] ({r.get('space_key', '')}) — {r.get('text', '')[:300]}"
                        for r in conf_results
                    ])
                    context_sections.append(f"## Confluence Page Results\n{conf_block}")
        except Exception as e:
            logger.warning(f"Confluence search failed in chat: {e}")

        if context_sections:
            full_context = "\n\n".join(context_sections)
            messages.append({
                "role": "user",
                "content": f"Here is relevant context from indexed sources:\n\n{full_context}\n\n---\nQUESTION: {request.query}"
            })
        else:
            messages.append({
                "role": "user",
                "content": f"No relevant indexed content found for this query.\n\nQUESTION: {request.query}"
            })
    else:
        messages.append({"role": "user", "content": request.query})

    # Log context stats for debugging
    user_msg = messages[-1]["content"] if messages else ""
    context_len = len(user_msg)
    has_context = "RAG context" in user_msg or "indexed sources" in user_msg
    logger.info(f"Chat: query='{request.query[:60]}', model={model}, context={context_len} chars, has_rag={has_context}")

    try:
        call_kwargs = {
            "model": model,
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
                model=model,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=0.2
            )

            return {
                "response": final_response.choices[0].message.content,
                "tools_used": [tc.function.name for tc in message.tool_calls],
                "model": model
            }

        return {
            "response": message.content,
            "tools_used": [],
            "model": model
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

    if request.collection in ("all", "jira"):
        try:
            jira_results = jira_indexer.search(
                query=request.query,
                top_k=request.top_k,
                project=request.project,
                assignee=request.assignee,
                issue_type=request.issue_type,
            )
            for r in jira_results:
                r["collection"] = "jira_tickets"
            results.extend(jira_results)
        except Exception as e:
            logger.warning(f"Jira search failed: {e}")

    if request.collection in ("all", "confluence"):
        try:
            conf_results = confluence_indexer.search(
                query=request.query,
                top_k=request.top_k,
                space=request.space,
                author=request.author,
                department=request.department,
                topic=request.topic,
            )
            for r in conf_results:
                r["collection"] = "confluence_pages"
            results.extend(conf_results)
        except Exception as e:
            logger.warning(f"Confluence search failed: {e}")

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
    """Start a product catalog sync from Shopify (runs in background)."""
    if not shopify_client:
        raise HTTPException(
            status_code=400,
            detail="Shopify not configured. Set SHOPIFY_SHOP_DOMAIN and SHOPIFY_ACCESS_TOKEN.",
        )

    import uuid as _uuid
    job_id = _uuid.uuid4().hex[:12]

    # Create a placeholder job so the frontend can poll immediately
    placeholder = ProductSyncJob(
        job_id=job_id,
        sync_type=request.sync_type or "full",
        status="running",
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    product_sync_jobs[job_id] = placeholder

    def progress_cb(j: ProductSyncJob):
        product_sync_jobs[j.job_id] = j

    async def _run_sync():
        try:
            if request.sync_type == "incremental" and request.since:
                job = await product_indexer.incremental_sync(
                    shopify_client, request.since, progress_callback=progress_cb
                )
            else:
                job = await product_indexer.index_all_products(
                    shopify_client, progress_callback=progress_cb
                )
            product_sync_jobs[job.job_id] = job
        except Exception as e:
            placeholder.status = "failed"
            placeholder.errors.append(str(e)[:200])
            placeholder.log(f"FAILED: {str(e)[:200]}")

    asyncio.create_task(_run_sync())

    return {"job_id": job_id, "status": "running", "sync_type": placeholder.sync_type}


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
    """Translate a list of texts with glossary + optional RAG context."""
    rag_context = None

    if request.use_rag and product_indexer is not None and request.texts:
        try:
            # Search for similar products to provide domain context
            combined_query = " ".join(request.texts)[:500]
            similar = product_indexer.search(query=combined_query, top_k=5)
            if similar:
                lines = []
                for r in similar:
                    title = r.get("title", "Unknown")
                    brand = r.get("brand", "")
                    category = r.get("category", "")
                    # Include structured metadata for terminology reference
                    parts = [f"Product: {title}"]
                    if brand:
                        parts.append(f"Brand: {brand}")
                    if category:
                        parts.append(f"Category: {category}")
                    # Include a snippet of the description for context
                    text = r.get("text", "")[:150]
                    if text:
                        parts.append(f"Description: {text}")
                    lines.append(" | ".join(parts))
                rag_context = "\n".join(lines)
                logger.info(f"RAG context built for translation: {len(similar)} similar products")
        except Exception as e:
            logger.warning(f"RAG context lookup failed, continuing without: {e}")

    job = await translator.translate_batch(
        texts=request.texts,
        source_lang=request.source_lang,
        target_lang=request.target_lang,
        rag_context=rag_context,
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


# ── Glossary Endpoints ────────────────────────────────────────


@app.get("/glossary")
async def glossary_list():
    """List all glossary entries (built-in + custom)."""
    custom = await glossary_store.get_all()
    return {
        "builtin_count": len(AIRSOFT_GLOSSARY_EN_ES),
        "custom_count": len(custom),
        "builtin": AIRSOFT_GLOSSARY_EN_ES,
        "custom": custom,
    }


@app.post("/glossary")
async def glossary_add(request: GlossaryEntryRequest):
    """Add or update a custom glossary entry."""
    if not request.source.strip() or not request.target.strip():
        raise HTTPException(status_code=400, detail="Both source and target terms are required")
    await glossary_store.add(request.source, request.target)
    return {"status": "added", "source": request.source.strip().lower(), "target": request.target.strip()}


@app.post("/glossary/bulk")
async def glossary_bulk_add(request: GlossaryBulkRequest):
    """Add multiple glossary entries at once."""
    count = await glossary_store.add_bulk(request.entries)
    return {"status": "added", "count": count}


@app.delete("/glossary/{term:path}")
async def glossary_delete(term: str):
    """Remove a custom glossary entry."""
    existed = await glossary_store.remove(term)
    if not existed:
        raise HTTPException(status_code=404, detail=f"Term not found: {term}")
    return {"status": "deleted", "term": term}


@app.get("/glossary/test")
async def glossary_test(text: str, source_lang: str = "en", target_lang: str = "es"):
    """Test which glossary terms would be matched for a given text."""
    relevant = glossary_store.get_relevant(text, source_lang, target_lang)
    return {
        "text": text,
        "matched_terms": len(relevant),
        "glossary": relevant,
    }


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


# ── Jira Endpoints ────────────────────────────────────────────


@app.post("/jira/import")
async def jira_import(request: JiraImportRequest):
    """Start importing Jira tickets (runs in background, returns job_id immediately)."""
    if not jira_indexer or not jira_indexer.configured:
        raise HTTPException(status_code=400, detail="Jira not configured. Set JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN.")

    job_id = uuid.uuid4().hex[:12]

    async def _run():
        def progress_cb(j: JiraImportJob):
            jira_import_jobs[j.job_id] = j
        job = await jira_indexer.import_all(project=request.project, progress_callback=progress_cb, job_id=job_id)
        jira_import_jobs[job.job_id] = job

    asyncio.get_running_loop().create_task(_run())
    return {"job_id": job_id, "status": "running"}


@app.get("/jira/import/status/{job_id}")
async def jira_import_status(job_id: str):
    """Get Jira import job status."""
    job = jira_import_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job.to_dict()


@app.post("/jira/sync")
async def jira_sync(request: JiraSyncRequest):
    """Sync recently updated Jira tickets (runs in background, returns job_id immediately)."""
    if not jira_indexer or not jira_indexer.configured:
        raise HTTPException(status_code=400, detail="Jira not configured.")

    job_id = uuid.uuid4().hex[:12]

    async def _run():
        def progress_cb(j: JiraImportJob):
            jira_import_jobs[j.job_id] = j
        job = await jira_indexer.sync_recent(since_hours=request.since_hours, progress_callback=progress_cb, job_id=job_id)
        jira_import_jobs[job.job_id] = job

    asyncio.get_running_loop().create_task(_run())
    return {"job_id": job_id, "status": "running"}


@app.get("/jira/search")
async def jira_search(
    query: str = "",
    top_k: int = 8,
    project: Optional[str] = None,
    assignee: Optional[str] = None,
    ticket_key: Optional[str] = None,
    issue_type: Optional[str] = None,
):
    """Search indexed Jira tickets."""
    if not jira_indexer:
        raise HTTPException(status_code=400, detail="Jira indexer not initialized.")

    results = jira_indexer.search(
        query=query,
        top_k=top_k,
        project=project,
        assignee=assignee,
        ticket_key=ticket_key,
        issue_type=issue_type,
    )
    return {"results": results, "query": query}


@app.get("/jira/sources")
async def jira_sources():
    """List indexed Jira projects with ticket/chunk counts."""
    if not jira_indexer:
        raise HTTPException(status_code=400, detail="Jira indexer not initialized.")
    return {"sources": jira_indexer.get_sources()}


@app.delete("/jira/source/{project}")
async def jira_delete_project(project: str):
    """Delete all indexed data for a Jira project."""
    if not jira_indexer:
        raise HTTPException(status_code=400, detail="Jira indexer not initialized.")
    success = jira_indexer.delete_project(project)
    if success:
        return {"status": "deleted", "project": project}
    raise HTTPException(status_code=500, detail=f"Failed to delete project: {project}")


@app.get("/jira/filters")
async def jira_filters():
    """Return unique assignees and projects for search filter dropdowns."""
    if not jira_indexer or not jira_indexer.configured:
        return {"assignees": [], "projects": []}
    return jira_indexer.get_filter_options()


@app.get("/jira/status")
async def jira_status():
    """Get comprehensive Jira indexer status for the dashboard."""
    configured = jira_indexer is not None and jira_indexer.configured
    result = {
        "configured": configured,
        "base_url": jira_indexer._jira_base_url if configured else None,
        "email": jira_indexer._jira_email if configured else None,
        "nightly_sync_enabled": os.environ.get("JIRA_SYNC_ENABLED", "false").lower() == "true",
        "nightly_sync_hour": int(os.environ.get("JIRA_SYNC_HOUR", "2")),
    }

    # Active/recent jobs
    active_jobs = []
    for j in jira_import_jobs.values():
        active_jobs.append(j.to_dict())
    # Sort by started_at descending, limit to 5
    active_jobs.sort(key=lambda x: x.get("started_at", ""), reverse=True)
    result["jobs"] = active_jobs[:5]

    # Collection stats from Qdrant
    if configured:
        try:
            sources = jira_indexer.get_sources()
            result["sources"] = sources
            result["total_tickets"] = sum(s.get("ticket_count", 0) for s in sources)
            result["total_chunks"] = sum(s.get("chunk_count", 0) for s in sources)
            # Find the latest last_indexed across all projects
            last_times = [s.get("last_indexed") for s in sources if s.get("last_indexed")]
            result["last_indexed"] = max(last_times) if last_times else None
        except Exception:
            result["sources"] = []
            result["total_tickets"] = 0
            result["total_chunks"] = 0
            result["last_indexed"] = None

    return result


@app.get("/jira/ticket/{key}")
async def jira_ticket_live(key: str):
    """Fetch a ticket from Jira API (live, not from Qdrant index)."""
    if not jira_indexer or not jira_indexer.configured:
        raise HTTPException(status_code=400, detail="Jira not configured.")
    try:
        return await jira_indexer.get_ticket_live(key)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Jira API error: {e.response.text[:200]}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])


# ── Confluence Endpoints ─────────────────────────────────────


@app.post("/confluence/import")
async def confluence_import(request: ConfluenceImportRequest):
    """Start importing Confluence pages (runs in background)."""
    if not confluence_indexer or not confluence_indexer.configured:
        raise HTTPException(status_code=400, detail="Confluence not configured.")

    job_id = uuid.uuid4().hex[:12]

    async def _run():
        def progress_cb(j: ConfluenceImportJob):
            confluence_import_jobs[j.job_id] = j
        job = await confluence_indexer.import_all(space=request.space, progress_callback=progress_cb, job_id=job_id)
        confluence_import_jobs[job.job_id] = job

    asyncio.get_running_loop().create_task(_run())
    return {"job_id": job_id, "status": "running"}


@app.get("/confluence/import/status/{job_id}")
async def confluence_import_status(job_id: str):
    """Get Confluence import job status."""
    job = confluence_import_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()


@app.post("/confluence/sync")
async def confluence_sync(request: ConfluenceSyncRequest):
    """Sync recently updated Confluence pages."""
    if not confluence_indexer or not confluence_indexer.configured:
        raise HTTPException(status_code=400, detail="Confluence not configured.")

    job_id = uuid.uuid4().hex[:12]

    async def _run():
        def progress_cb(j: ConfluenceImportJob):
            confluence_import_jobs[j.job_id] = j
        job = await confluence_indexer.sync_recent(since_hours=request.since_hours, progress_callback=progress_cb, job_id=job_id)
        confluence_import_jobs[job.job_id] = job

    asyncio.get_running_loop().create_task(_run())
    return {"job_id": job_id, "status": "running"}


@app.get("/confluence/search")
async def confluence_search(
    query: str = "",
    top_k: int = 10,
    space: Optional[str] = None,
    author: Optional[str] = None,
    label: Optional[str] = None,
    department: Optional[str] = None,
    topic: Optional[str] = None,
):
    """Search indexed Confluence pages."""
    if not confluence_indexer:
        raise HTTPException(status_code=400, detail="Confluence indexer not initialized.")
    results = confluence_indexer.search(
        query=query, top_k=top_k, space=space, author=author,
        label=label, department=department, topic=topic,
    )
    return {"results": results, "query": query, "collection": "confluence"}


@app.get("/confluence/sources")
async def confluence_sources():
    """List indexed Confluence spaces with page/chunk counts."""
    if not confluence_indexer:
        return {"sources": []}
    return {"sources": confluence_indexer.get_sources()}


@app.delete("/confluence/source/{space_key}")
async def confluence_delete_source(space_key: str):
    """Delete all indexed data for a Confluence space."""
    if not confluence_indexer:
        raise HTTPException(status_code=400, detail="Confluence indexer not initialized.")
    ok = confluence_indexer.delete_space(space_key)
    if not ok:
        raise HTTPException(status_code=500, detail="Delete failed")
    return {"deleted": space_key}


@app.get("/confluence/filters")
async def confluence_filters():
    """Return unique spaces, authors, departments, and topics for search filters."""
    if not confluence_indexer or not confluence_indexer.configured:
        return {"spaces": [], "authors": [], "departments": [], "topics": []}
    return confluence_indexer.get_filters()


@app.get("/confluence/spaces")
async def confluence_spaces():
    """List available Confluence spaces (live from API)."""
    if not confluence_indexer or not confluence_indexer.configured:
        raise HTTPException(status_code=400, detail="Confluence not configured.")
    spaces = await confluence_indexer.fetch_spaces()
    return {"spaces": [{"key": s.get("key", ""), "name": s.get("name", ""), "type": s.get("type", "")} for s in spaces]}


@app.get("/confluence/status")
async def confluence_status():
    """Get comprehensive Confluence indexer status for the dashboard."""
    configured = confluence_indexer is not None and confluence_indexer.configured
    sources = confluence_indexer.get_sources() if configured else []
    total_pages = sum(s["page_count"] for s in sources)
    total_chunks = sum(s["chunk_count"] for s in sources)
    last_times = [s["last_indexed"] for s in sources if s.get("last_indexed")]
    last_indexed = max(last_times) if last_times else None

    active_jobs = []
    for j in confluence_import_jobs.values():
        active_jobs.append(j.to_dict())
    active_jobs.sort(key=lambda x: x.get("started_at", ""), reverse=True)

    return {
        "configured": configured,
        "base_url": confluence_indexer._base_url if configured else None,
        "total_pages": total_pages,
        "total_chunks": total_chunks,
        "sources": sources,
        "last_indexed": last_indexed,
        "nightly_sync_enabled": CONFLUENCE_SYNC_ENABLED,
        "nightly_sync_hour": CONFLUENCE_SYNC_HOUR,
        "jobs": active_jobs[:5],
    }


# ── Agent Endpoints ───────────────────────────────────────────


def _cleanup_old_agent_tasks():
    """Remove in-memory tasks older than 5 minutes (Redis has the persistent copy)."""
    now = datetime.now(timezone.utc)
    stale = []
    for tid, task in agent_tasks.items():
        if task.status in ("completed", "failed", "cancelled") and task.ended_at:
            try:
                ended = datetime.fromisoformat(task.ended_at)
                if (now - ended).total_seconds() > 300:
                    stale.append(tid)
            except (ValueError, TypeError):
                pass
    for tid in stale:
        del agent_tasks[tid]


def _make_services(task: AgentTask = None) -> AgentServices:
    return AgentServices(
        web_indexer=web_indexer,
        retriever=retriever,
        product_indexer=product_indexer,
        devops_indexer=devops_indexer,
        log_analyzer=log_analyzer,
        llm_client=llm_client,
        task=task,
    )


@app.get("/agent/status")
async def agent_service_status():
    """Check if LLM is available for agent tasks."""
    try:
        models = llm_client.models.list()
        model_id = models.data[0].id if models.data else None
    except Exception:
        model_id = None

    if model_id:
        return {"available": True, "model_id": model_id, "redis": redis_client is not None}
    return {"available": False, "reason": "No LLM model available. Configure LLM_BASE_URL."}


@app.post("/agent/task")
async def create_agent_task(request: AgentTaskRequest):
    """Create and start a new agent task."""
    _cleanup_old_agent_tasks()

    if not rag_agent:
        raise HTTPException(status_code=503, detail="Agent not initialized — LLM may be unavailable")

    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    running = sum(1 for t in agent_tasks.values() if t.status == "running")
    if running >= 3:
        raise HTTPException(status_code=429, detail="Maximum 3 concurrent agent tasks")

    import uuid
    task_id = uuid.uuid4().hex[:12]
    task = AgentTask(
        task_id=task_id,
        prompt=prompt,
        started_at=datetime.now(timezone.utc).isoformat(),
        source="web",
    )

    # Wire Redis persistence callbacks
    if session_store:
        task._on_log = session_store.add_log
        task._on_step = session_store.add_step
        await session_store.create_task(task, source="web")

    agent_tasks[task_id] = task

    session = RedisSession(task_id, redis_client) if redis_client else None
    services = _make_services(task)

    async def _run():
        await run_task(rag_agent, services, prompt, task=task,
                       session=session, session_store=session_store)

    asyncio.create_task(_run())
    logger.info(f"Started agent task {task_id}: {prompt[:80]}")

    return {"task_id": task_id, "status": "running"}


@app.post("/agent/task/{task_id}/message")
async def send_agent_message(task_id: str, request: AgentContinueRequest):
    """Send a message to an agent task.

    - If the task is **running**: queues the message — it will be processed
      automatically after the current agent run finishes.
    - If the task is **completed/failed**: restarts the session with the
      new message (like the old /continue endpoint).
    """
    if not rag_agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    if not session_store or not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available for session resume")

    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Check in-memory first (running tasks), then Redis
    mem_task = agent_tasks.get(task_id)
    if mem_task and mem_task.status == "running":
        # Task is running — queue the message for after current run
        queue_len = await session_store.push_message(task_id, message)
        mem_task.log(f"User message queued (position {queue_len}): {message[:200]}")
        mem_task.add_step("user_message", f"[queued] {message}")
        logger.info(f"Queued message for running task {task_id}: {message[:80]}")
        return {"task_id": task_id, "status": "queued", "queue_position": queue_len}

    # Task is not running in memory — check Redis
    existing = await session_store.get_task(task_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Session not found: {task_id}")

    if existing["status"] == "running":
        # Running in Redis but not in memory (e.g. after restart) — queue it
        queue_len = await session_store.push_message(task_id, message)
        logger.info(f"Queued message for task {task_id} (Redis-only): {message[:80]}")
        return {"task_id": task_id, "status": "queued", "queue_position": queue_len}

    # Task is completed/failed — restart the session
    task = AgentTask(
        task_id=task_id,
        prompt=existing.get("prompt", ""),
        started_at=existing.get("started_at", ""),
        source=existing.get("source", "web"),
    )
    task._on_log = session_store.add_log
    task._on_step = session_store.add_step
    task.status = "running"
    await session_store.update_task(task)

    agent_tasks[task_id] = task

    session = RedisSession(task_id, redis_client)
    services = _make_services(task)

    task.add_step("user_message", message)
    task.log(f"Resume with: {message[:200]}")

    async def _run():
        await run_task(rag_agent, services, message, task=task,
                       session=session, session_store=session_store)

    asyncio.create_task(_run())
    logger.info(f"Continued agent task {task_id}: {message[:80]}")

    return {"task_id": task_id, "status": "running", "continued": True}


@app.post("/agent/task/{task_id}/continue")
async def continue_agent_task(task_id: str, request: AgentContinueRequest):
    """Legacy endpoint — redirects to /message. Kept for backwards compatibility."""
    return await send_agent_message(task_id, request)


@app.get("/agent/tasks")
async def list_agent_tasks():
    """List all agent tasks (from Redis if available, else in-memory)."""
    if session_store:
        try:
            tasks = await session_store.list_tasks(limit=50)
            # Merge in-memory running tasks (freshest data)
            redis_ids = {t["task_id"] for t in tasks}
            for tid, mem_task in agent_tasks.items():
                if tid not in redis_ids:
                    tasks.append(mem_task.to_dict())
            tasks.sort(key=lambda t: t.get("started_at", ""), reverse=True)
            return {"tasks": tasks}
        except Exception as e:
            logger.warning(f"Redis list_tasks failed, falling back to memory: {e}")

    tasks = [t.to_dict() for t in agent_tasks.values()]
    tasks.sort(key=lambda t: t.get("started_at", ""), reverse=True)
    return {"tasks": tasks}


@app.get("/agent/task/{task_id}")
async def get_agent_task(task_id: str):
    # In-memory first (running tasks have freshest data)
    if task_id in agent_tasks:
        return agent_tasks[task_id].to_dict()
    # Then Redis
    if session_store:
        task_data = await session_store.get_task(task_id)
        if task_data:
            task_data["steps"] = await session_store.get_steps(task_id, last_n=50)
            return task_data
    raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")


@app.get("/agent/task/{task_id}/stream")
async def agent_task_stream(task_id: str):
    """SSE stream of agent task progress."""
    if task_id not in agent_tasks:
        raise HTTPException(status_code=404, detail=f"Task not found in active tasks: {task_id}")

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
    # In-memory first
    if task_id in agent_tasks:
        task = agent_tasks[task_id]
        return {
            "task_id": task_id,
            "logs": task.logs[offset:],
            "total": len(task.logs),
            "offset": offset,
            "status": task.status,
        }
    # Then Redis
    if session_store:
        logs, total = await session_store.get_logs(task_id, offset=offset)
        task_data = await session_store.get_task(task_id)
        if task_data:
            return {
                "task_id": task_id,
                "logs": logs,
                "total": total,
                "offset": offset,
                "status": task_data.get("status", "unknown"),
            }
    raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")


@app.get("/agent/task/{task_id}/history")
async def get_agent_conversation_history(task_id: str):
    """Get the full SDK conversation history for a session (for debugging/inspection)."""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available")
    session = RedisSession(task_id, redis_client)
    items = await session.get_items()
    return {"task_id": task_id, "items": items, "count": len(items)}


@app.get("/agent/task/{task_id}/steps")
async def get_agent_all_steps(task_id: str):
    """Get ALL steps for a task (not just last 50)."""
    if session_store:
        steps = await session_store.get_all_steps(task_id)
        if steps:
            return {"task_id": task_id, "steps": steps, "count": len(steps)}
    # Fallback to in-memory
    if task_id in agent_tasks:
        return {"task_id": task_id, "steps": agent_tasks[task_id].steps, "count": len(agent_tasks[task_id].steps)}
    raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
