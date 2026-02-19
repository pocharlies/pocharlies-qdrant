# Agent SDK Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the hand-rolled agent loop in `agent.py` with OpenAI Agents SDK, organized as an `agent/` package with CLI support.

**Architecture:** New `agent/` package with 4 files (`__init__.py`, `tools.py`, `runner.py`, `cli.py`) replaces monolithic `agent.py`. All 10 existing tools get `@function_tool` wrappers. `app.py` endpoints swap to new imports. Old `agent.py` renamed to `agent_legacy.py` for safety.

**Tech Stack:** `openai-agents>=0.7.0` (Agent/Runner/function_tool), `typer>=0.15.0` (CLI), `OpenAIChatCompletionsModel` (vLLM compatibility)

**Design doc:** `docs/plans/2026-02-19-agent-sdk-migration-design.md`

---

### Task 1: Add dependencies

**Files:**
- Modify: `rag-service/requirements.txt`

**Step 1: Add openai-agents and typer to requirements.txt**

Add these two lines at the end of `rag-service/requirements.txt`:

```
openai-agents>=0.7.0
typer>=0.15.0
```

The full file after edit:

```
fastapi==0.109.0
uvicorn==0.27.0
qdrant-client>=1.12.0,<2.0.0
sentence-transformers==2.3.1
fastembed>=0.4.0
openai==1.12.0
gitpython==3.1.41
pydantic==2.5.3
python-dotenv==1.0.0
aiofiles==23.2.1
httpx==0.26.0
beautifulsoup4==4.12.3
trafilatura==1.8.1
validators==0.22.0
curl_cffi>=0.7.0
redis[hiredis]>=5.0.0
PyMuPDF>=1.24.0
langdetect>=1.0.9
openai-agents>=0.7.0
typer>=0.15.0
```

**Step 2: Verify pip can resolve them (local check)**

Run: `pip install --dry-run openai-agents>=0.7.0 typer>=0.15.0`
Expected: No conflicts

**Step 3: Commit**

```bash
git add rag-service/requirements.txt
git commit -m "deps: add openai-agents and typer for agent SDK migration"
```

---

### Task 2: Create agent/ package with AgentServices and create_agent()

**Files:**
- Create: `rag-service/agent/__init__.py`

**Step 1: Create agent/__init__.py**

```python
"""
Agent package — OpenAI Agents SDK wrapper for RAG service.
Replaces the hand-rolled agent loop from agent_legacy.py.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone

from openai import OpenAI, AsyncOpenAI
from agents import Agent, ModelSettings
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel

logger = logging.getLogger(__name__)


# ── AgentTask (carried over from agent_legacy.py, minus cancel/inject) ──


@dataclass
class AgentTask:
    """Tracks an autonomous agent task."""

    task_id: str
    prompt: str
    status: str = "running"  # running | completed | failed
    steps: List[Dict[str, Any]] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    model_id: Optional[str] = None
    tools_called: List[str] = field(default_factory=list)
    summary: Optional[str] = None
    error: Optional[str] = None

    MAX_LOGS = 3000

    def log(self, msg: str):
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        entry = f"[{ts}] {msg}"
        self.logs.append(entry)
        if len(self.logs) > self.MAX_LOGS:
            self.logs = self.logs[:50] + self.logs[-(self.MAX_LOGS - 50):]

    def add_step(self, step_type: str, content: str):
        self.steps.append({
            "type": step_type,
            "content": content[:2000],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "status": self.status,
            "steps": self.steps[-50:],
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "model_id": self.model_id,
            "tools_called": self.tools_called,
            "summary": self.summary,
            "error": self.error,
            "log_count": len(self.logs),
        }


# ── AgentServices context (passed to tools via RunContextWrapper) ──


@dataclass
class AgentServices:
    """Shared services passed to all tools via RunContextWrapper[AgentServices]."""
    web_indexer: Any  # WebIndexer
    retriever: Any  # Optional[CodeRetriever]
    product_indexer: Any  # Optional[ProductIndexer]
    devops_indexer: Any  # Optional[DevOpsIndexer]
    log_analyzer: Any  # Optional[LogAnalyzer]
    llm_client: Any  # OpenAI
    task: Optional[AgentTask] = None


# ── System prompt (unchanged from agent_legacy.py) ──


SYSTEM_PROMPT = """You are an autonomous AI agent managing a multi-purpose RAG (Retrieval-Augmented Generation) system for an airsoft e-commerce business and DevOps operations. You have tools to:

**Web & Content:**
1. **web_search** — Search the internet to find relevant URLs
2. **analyze_site** — Check if a URL is accessible and crawlable (detects Cloudflare, SPAs, bot protection)
3. **crawl_website** — Crawl and index a website into the RAG vector database
4. **search_indexed** — Search already-indexed content (web pages, code repos, or both)
5. **list_collections** — View database statistics for all collections
6. **list_sources** — See what domains are already indexed
7. **delete_source** — Remove a domain's content from the index

**Product Catalog:**
8. **search_products** — Search the Shopify product catalog. REQUIRES `query` parameter (always pass a search string).

**DevOps & SRE:**
9. **search_devops** — Search DevOps documentation (runbooks, postmortems, config, procedures)
10. **analyze_logs** — Analyze log text to classify errors, identify services, and match runbooks

## IMPORTANT RULES — Tool Usage

**Always pass ALL required parameters.** Every tool call MUST include its required arguments:
- `search_products` REQUIRES `query` (a text string). If you want to browse by category, use the category name as the query: `search_products(query="pistol", category="pistol")`.
- `search_indexed` REQUIRES `query`.
- `crawl_website` REQUIRES `url`.
- `web_search` REQUIRES `query`.

**Before crawling, ALWAYS analyze the site first:**
1. Call `analyze_site(url)` — check the `crawlable` field and `verdict`
2. If verdict says BLOCKED or SPA DETECTED, do NOT attempt to crawl — it will fail
3. If blocked by Cloudflare, try alternative URLs or inform the user
4. If SPA detected, the crawler cannot render JavaScript — suggest a different approach

**Crawl failure recovery:**
- If a crawl returns 0 pages indexed, READ the diagnostics and suggestions carefully
- If smart_mode analysis failed, retry with `smart_mode=false`
- If 0 pages visited, the site blocks bots — do NOT retry the same URL
- If pages were visited but 0 indexed, content was below minimum length or filtered as boilerplate

**Workflow:**
1. Check `list_sources` first — don't re-crawl indexed domains
2. Use `analyze_site` before crawling to check accessibility
3. Only crawl if the site is marked `crawlable: true`
4. After crawling, use `search_indexed` to verify content was indexed
5. Report progress and findings clearly

When finished, provide a clear summary of what you accomplished."""


# ── Agent factory ──


def create_agent(vllm_base_url: str, api_key: str = "none") -> tuple:
    """Create the agent and return (agent, model_id).

    Returns:
        Tuple of (Agent, str) — the agent instance and discovered model ID.
    """
    from .tools import ALL_TOOLS

    async_client = AsyncOpenAI(base_url=vllm_base_url, api_key=api_key)

    # Auto-discover model ID from vLLM
    sync_client = OpenAI(base_url=vllm_base_url, api_key=api_key)
    try:
        models = sync_client.models.list()
        model_id = models.data[0].id if models.data else "default"
    except Exception as e:
        logger.warning(f"Failed to discover model: {e}")
        model_id = "default"

    agent = Agent(
        name="pocharlies-rag",
        instructions=SYSTEM_PROMPT,
        tools=ALL_TOOLS,
        model=OpenAIChatCompletionsModel(model=model_id, openai_client=async_client),
        model_settings=ModelSettings(temperature=0.3, max_tokens=4096),
    )

    logger.info(f"Created agent with model: {model_id}")
    return agent, model_id
```

**Step 2: Commit**

```bash
git add rag-service/agent/__init__.py
git commit -m "feat(agent): add agent package with AgentTask, AgentServices, create_agent"
```

---

### Task 3: Create agent/tools.py — migrate all 10 tools

**Files:**
- Create: `rag-service/agent/tools.py`

This is the largest task. Each tool follows the same pattern: `@function_tool` decorator, `ctx: RunContextWrapper[AgentServices]` as first param, same logic body as current `agent.py`.

**Step 1: Create agent/tools.py with all 10 tools**

```python
"""
Agent tools — @function_tool wrappers around existing service methods.
Each tool pulls services from RunContextWrapper[AgentServices].
"""

import json
import logging
from typing import Optional

from agents import function_tool, RunContextWrapper

logger = logging.getLogger(__name__)


# Type alias for context
from . import AgentServices
Ctx = RunContextWrapper[AgentServices]


# ── Web & Content tools ──────────────────────────────────────


@function_tool
async def crawl_website(
    ctx: Ctx,
    url: str,
    max_depth: int = 2,
    max_pages: int = 50,
    smart_mode: bool = True,
) -> str:
    """Crawl a website and index its content into the RAG vector database.

    Args:
        url: Starting URL to crawl.
        max_depth: Link depth 0-5 (default: 2).
        max_pages: Max pages 1-500 (default: 50).
        smart_mode: Use AI site analysis (default: true).
    """
    svc = ctx.context
    if svc.task:
        svc.task.log(f"TOOL crawl_website: {url} (depth={max_depth}, pages={max_pages}, smart={smart_mode})")
        svc.task.add_step("tool_call", f"crawl_website({url}, depth={max_depth}, pages={max_pages})")
    try:
        job = await svc.web_indexer.crawl_and_index(
            start_url=url,
            max_depth=int(max_depth),
            max_pages=int(max_pages),
            smart_mode=bool(smart_mode),
            llm_client=svc.llm_client,
        )
        result = f"Crawled {url}: {job.pages_indexed} pages indexed, {job.chunks_indexed} chunks, {job.pages_visited} visited. Status: {job.status}"
        if job.pages_indexed == 0:
            diagnostics = []
            if job.errors:
                diagnostics.append(f"Errors: {'; '.join(job.errors[:3])}")
            if job.analysis_status == "failed":
                diagnostics.append("Smart mode analysis failed (LLM could not parse site structure)")
            if job.pages_visited == 0:
                diagnostics.append("No pages could be fetched — site may be blocking bots (Cloudflare/WAF), returning empty HTML, or the URL may be incorrect")
            elif job.pages_visited > 0 and job.pages_scraped == 0:
                diagnostics.append(f"Visited {job.pages_visited} pages but none had extractable content — site likely uses JavaScript rendering (SPA) which the crawler cannot execute")
            elif job.pages_scraped > 0:
                diagnostics.append(f"Scraped {job.pages_scraped} pages but content was below minimum threshold or all content was filtered as boilerplate")
            if diagnostics:
                result += "\nDIAGNOSTICS: " + " | ".join(diagnostics)
            result += "\nSUGGESTIONS: Try with smart_mode=false if smart analysis failed. If 0 pages visited, the site blocks bots — try a different URL or use analyze_site first to check accessibility."
            recent_logs = [l for l in job.logs[-5:] if l] if hasattr(job, 'logs') else []
            if recent_logs:
                result += "\nRecent crawl logs:\n" + "\n".join(recent_logs)
        if svc.task:
            svc.task.add_step("tool_result", result)
            svc.task.log(f"RESULT: {result[:300]}")
        return result
    except Exception as e:
        err = f"Crawl failed for {url}: {str(e)[:300]}"
        if svc.task:
            svc.task.add_step("tool_result", err)
            svc.task.log(f"ERROR: {err}")
        return err


@function_tool
async def search_indexed(
    ctx: Ctx,
    query: str,
    collection: str = "all",
    domain: Optional[str] = None,
    repo: Optional[str] = None,
    top_k: int = 5,
) -> str:
    """Search the RAG database for indexed content. Supports web pages, code repositories, or both.

    Args:
        query: Search query.
        collection: Collection to search: 'all' (both), 'web' (web pages), 'code' (code repos). Default: all.
        domain: Filter by domain (web only, optional).
        repo: Filter by repository name (code only, optional).
        top_k: Number of results (default: 5).
    """
    svc = ctx.context
    if svc.task:
        svc.task.log(f"TOOL search_indexed: '{query}' (collection={collection}, domain={domain}, repo={repo})")
        svc.task.add_step("tool_call", f"search_indexed('{query}', collection={collection})")
    try:
        formatted = []
        top_k = int(top_k)

        if collection in ("all", "web"):
            web_results = svc.web_indexer.search(query=query, top_k=top_k, domain_filter=domain)
            for r in web_results:
                formatted.append(f"[WEB {r['score']:.2f}] {r.get('title', 'Untitled')} — {r['url']}\n{r['text'][:300]}")

        if collection in ("all", "code") and svc.retriever:
            code_results = svc.retriever.retrieve(query=query, top_k=top_k, repo_filter=repo)
            for r in code_results:
                symbols_str = f" | symbols: {', '.join(r.symbols)}" if r.symbols else ""
                formatted.append(f"[CODE {r.score:.2f}] {r.repo}/{r.path}#L{r.start_line}-{r.end_line}{symbols_str}\n{r.text[:300]}")

        if not formatted:
            msg = f"No results found for '{query}'"
            if svc.task:
                svc.task.add_step("tool_result", msg)
            return msg

        result = "\n\n".join(formatted)
        if svc.task:
            svc.task.add_step("tool_result", f"{len(formatted)} results found")
            svc.task.log(f"RESULT: {len(formatted)} results")
        return result
    except Exception as e:
        err = f"Search failed: {str(e)[:300]}"
        if svc.task:
            svc.task.add_step("tool_result", err)
        return err


@function_tool
async def list_collections(ctx: Ctx) -> str:
    """Get Qdrant vector database collection statistics."""
    svc = ctx.context
    if svc.task:
        svc.task.log("TOOL list_collections")
        svc.task.add_step("tool_call", "list_collections()")
    try:
        stats = svc.web_indexer.get_collection_stats()
        result = json.dumps(stats, indent=2)
        if svc.task:
            svc.task.add_step("tool_result", result)
        return result
    except Exception as e:
        err = f"Failed: {str(e)[:300]}"
        if svc.task:
            svc.task.add_step("tool_result", err)
        return err


@function_tool
async def list_sources(ctx: Ctx) -> str:
    """List all indexed web domains with page and chunk counts."""
    svc = ctx.context
    if svc.task:
        svc.task.log("TOOL list_sources")
        svc.task.add_step("tool_call", "list_sources()")
    try:
        sources = svc.web_indexer.get_sources()
        if not sources:
            msg = "No sources indexed."
            if svc.task:
                svc.task.add_step("tool_result", msg)
            return msg
        lines = [f"{s['domain']}: {s['url_count']} pages, {s['chunk_count']} chunks" for s in sources]
        result = "\n".join(lines)
        if svc.task:
            svc.task.add_step("tool_result", f"{len(sources)} domains")
        return result
    except Exception as e:
        err = f"Failed: {str(e)[:300]}"
        if svc.task:
            svc.task.add_step("tool_result", err)
        return err


@function_tool
async def delete_source(ctx: Ctx, domain: str) -> str:
    """Delete all indexed content from a specific domain.

    Args:
        domain: Domain to delete (e.g. 'example.com').
    """
    svc = ctx.context
    if svc.task:
        svc.task.log(f"TOOL delete_source: {domain}")
        svc.task.add_step("tool_call", f"delete_source('{domain}')")
    try:
        success = svc.web_indexer.delete_source(domain)
        result = f"Deleted {domain}" if success else f"Failed to delete {domain}"
        if svc.task:
            svc.task.add_step("tool_result", result)
        return result
    except Exception as e:
        err = f"Failed: {str(e)[:300]}"
        if svc.task:
            svc.task.add_step("tool_result", err)
        return err


@function_tool
async def web_search(ctx: Ctx, query: str) -> str:
    """Search the internet to find URLs and information. Use to discover websites to crawl.

    Args:
        query: Search query.
    """
    import httpx as _httpx
    from bs4 import BeautifulSoup

    svc = ctx.context
    if svc.task:
        svc.task.log(f"TOOL web_search: '{query}'")
        svc.task.add_step("tool_call", f"web_search('{query}')")
    try:
        async with _httpx.AsyncClient(
            follow_redirects=True,
            timeout=15.0,
            headers={"User-Agent": "Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36"},
        ) as client:
            resp = await client.get("https://html.duckduckgo.com/html/", params={"q": query})
            soup = BeautifulSoup(resp.text, "html.parser")
            results = []
            for a in soup.select("a.result__a")[:10]:
                href = a.get("href", "")
                title = a.get_text(strip=True)
                if href and title:
                    results.append(f"{title}\n  {href}")

        if not results:
            msg = f"No web results for '{query}'"
            if svc.task:
                svc.task.add_step("tool_result", msg)
            return msg

        result = "\n\n".join(results)
        if svc.task:
            svc.task.add_step("tool_result", f"{len(results)} web results")
            svc.task.log(f"RESULT: {len(results)} web results")
        return result
    except Exception as e:
        err = f"Web search failed: {str(e)[:300]}"
        if svc.task:
            svc.task.add_step("tool_result", err)
        return err


@function_tool
async def analyze_site(ctx: Ctx, url: str) -> str:
    """Probe a URL to check accessibility, detect Cloudflare, and estimate content.

    Args:
        url: URL to analyze.
    """
    import httpx as _httpx
    from bs4 import BeautifulSoup

    svc = ctx.context
    if svc.task:
        svc.task.log(f"TOOL analyze_site: {url}")
        svc.task.add_step("tool_call", f"analyze_site('{url}')")
    try:
        async with _httpx.AsyncClient(
            follow_redirects=True,
            timeout=10.0,
            headers={"User-Agent": "Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36"},
        ) as client:
            resp = await client.get(url)

        body_lower = resp.text.lower() if resp.text else ""
        cf_indicators = ["cloudflare", "cf-browser-verification", "just a moment", "cf-challenge",
                         "checking your browser", "please wait", "ddos-guard", "sucuri"]
        cf_matches = [i for i in cf_indicators if i in body_lower]

        soup = BeautifulSoup(resp.text, "html.parser")
        title_tag = soup.find("title")
        links = soup.find_all("a", href=True)
        text_content = soup.get_text(separator=" ", strip=True)
        scripts = soup.find_all("script")
        has_react_root = bool(soup.find(id="root") or soup.find(id="app") or soup.find(id="__next"))

        is_spa = has_react_root and len(text_content) < 500 and len(scripts) > 5
        is_blocked = bool(cf_matches) or resp.status_code in (403, 503, 429)

        info = {
            "url": str(resp.url),
            "status_code": resp.status_code,
            "content_type": resp.headers.get("content-type", "unknown"),
            "content_length": len(resp.text) if resp.text else 0,
            "text_content_length": len(text_content),
            "cloudflare_detected": bool(cf_matches),
            "cloudflare_indicators": cf_matches[:3] if cf_matches else [],
            "title": title_tag.get_text(strip=True)[:200] if title_tag else "",
            "link_count": len(links),
            "script_count": len(scripts),
            "is_spa_likely": is_spa,
            "crawlable": not is_blocked and not is_spa and len(text_content) > 100,
        }

        warnings = []
        if is_blocked:
            warnings.append("BLOCKED: Site uses bot protection (Cloudflare/WAF).")
        if is_spa:
            warnings.append("SPA DETECTED: Site uses JavaScript rendering. Crawler cannot execute JS.")
        if resp.status_code >= 400:
            warnings.append(f"HTTP ERROR: Status {resp.status_code}.")
        if len(text_content) < 100 and not is_spa and not is_blocked:
            warnings.append("LOW CONTENT: Very little text content found.")
        if not warnings:
            warnings.append("CRAWLABLE: Site appears accessible and has extractable content.")

        info["verdict"] = " | ".join(warnings)

        result = json.dumps(info, indent=2)
        if svc.task:
            svc.task.add_step("tool_result", result)
        return result
    except Exception as e:
        err = f"Site analysis failed: {str(e)[:300]}. The URL may be unreachable or have invalid SSL."
        if svc.task:
            svc.task.add_step("tool_result", err)
        return err


# ── Product Catalog tools ────────────────────────────────────


@function_tool
async def search_products(
    ctx: Ctx,
    query: str,
    brand: Optional[str] = None,
    category: Optional[str] = None,
    top_k: int = 5,
) -> str:
    """Search the product catalog (Shopify products indexed in Qdrant). Supports brand, category, and price filters.

    Args:
        query: Search query (e.g. 'Tokyo Marui Hi-Capa').
        brand: Filter by brand name (optional).
        category: Filter by category: gbb, aeg, sniper, pistol, shotgun, smg, accessory, gear, etc.
        top_k: Number of results (default: 5).
    """
    svc = ctx.context
    if not svc.product_indexer:
        return "Product catalog not configured."

    # Defensive: LLM sometimes omits query
    if not query:
        parts = []
        if brand:
            parts.append(str(brand))
        if category:
            parts.append(str(category))
        query = " ".join(parts) if parts else "all products"

    if svc.task:
        svc.task.log(f"TOOL search_products: '{query}' (brand={brand}, category={category})")
        svc.task.add_step("tool_call", f"search_products('{query}', brand={brand}, category={category})")
    try:
        results = svc.product_indexer.search(
            query=query,
            top_k=int(top_k),
            brand_filter=brand,
            category_filter=category,
        )
        if not results:
            msg = f"No products found for '{query}'"
            if svc.task:
                svc.task.add_step("tool_result", msg)
            return msg

        formatted = []
        for r in results:
            price_str = f" EUR{r['price']}" if r.get('price') else ""
            formatted.append(f"[PRODUCT {r['score']:.2f}] {r.get('brand', '')} {r.get('title', '')}{price_str}\n  SKU: {r.get('sku', 'N/A')} | Category: {r.get('category', 'N/A')}\n  {r['text'][:200]}")

        result = "\n\n".join(formatted)
        if svc.task:
            svc.task.add_step("tool_result", f"{len(formatted)} products found")
            svc.task.log(f"RESULT: {len(formatted)} products")
        return result
    except Exception as e:
        err = f"Product search failed: {str(e)[:300]}"
        if svc.task:
            svc.task.add_step("tool_result", err)
        return err


# ── DevOps & SRE tools ───────────────────────────────────────


@function_tool
async def search_devops(
    ctx: Ctx,
    query: str,
    doc_type: Optional[str] = None,
    top_k: int = 5,
) -> str:
    """Search DevOps documentation: runbooks, postmortems, config files, procedures.

    Args:
        query: Search query (e.g. 'vLLM OOM recovery').
        doc_type: Filter by doc type: runbook, postmortem, config, procedure, architecture, documentation.
        top_k: Number of results (default: 5).
    """
    svc = ctx.context
    if not svc.devops_indexer:
        return "DevOps indexer not configured."

    if not query:
        query = doc_type if doc_type else "devops documentation"

    if svc.task:
        svc.task.log(f"TOOL search_devops: '{query}' (doc_type={doc_type})")
        svc.task.add_step("tool_call", f"search_devops('{query}', doc_type={doc_type})")
    try:
        results = svc.devops_indexer.search(
            query=query,
            top_k=int(top_k),
            doc_type_filter=doc_type,
        )
        if not results:
            msg = f"No DevOps docs found for '{query}'"
            if svc.task:
                svc.task.add_step("tool_result", msg)
            return msg

        formatted = []
        for r in results:
            formatted.append(f"[DEVOPS {r['score']:.2f}] {r.get('title', 'Untitled')} ({r.get('doc_type', 'unknown')})\n  Source: {r.get('source_path', 'N/A')}\n  {r['text'][:300]}")

        result = "\n\n".join(formatted)
        if svc.task:
            svc.task.add_step("tool_result", f"{len(formatted)} docs found")
            svc.task.log(f"RESULT: {len(formatted)} devops docs")
        return result
    except Exception as e:
        err = f"DevOps search failed: {str(e)[:300]}"
        if svc.task:
            svc.task.add_step("tool_result", err)
        return err


@function_tool
async def analyze_logs(
    ctx: Ctx,
    log_text: str,
    service: str = "unknown",
) -> str:
    """Analyze log text to classify errors by severity, identify services, and match related runbooks.

    Args:
        log_text: Log text to analyze (error logs, container output, etc.).
        service: Source service name (e.g. 'vllm', 'litellm', 'nginx').
    """
    svc = ctx.context
    if not svc.log_analyzer:
        return "Log analyzer not configured."

    if svc.task:
        svc.task.log(f"TOOL analyze_logs: {len(log_text)} chars from '{service}'")
        svc.task.add_step("tool_call", f"analyze_logs(service={service}, {len(log_text)} chars)")
    try:
        job = await svc.log_analyzer.analyze_logs(log_text=log_text, source_service=service)
        if not job.results:
            msg = "No notable errors or warnings found in the log text."
            if svc.task:
                svc.task.add_step("tool_result", msg)
            return msg

        formatted = []
        for r in job.results:
            runbooks_str = ""
            if r.get("related_runbooks"):
                runbooks_str = "\n  Related runbooks: " + ", ".join(
                    rb.get("title", "?") for rb in r["related_runbooks"][:3]
                )
            formatted.append(
                f"[{r.get('severity', 'unknown').upper()}] {r.get('error_type', 'unknown')} — {r.get('summary', '')}"
                f"\n  Service: {r.get('service', service)}"
                f"{runbooks_str}"
            )

        result = "\n\n".join(formatted)
        if svc.task:
            svc.task.add_step("tool_result", f"{len(formatted)} issues found")
            svc.task.log(f"RESULT: {len(formatted)} log issues")
        return result
    except Exception as e:
        err = f"Log analysis failed: {str(e)[:300]}"
        if svc.task:
            svc.task.add_step("tool_result", err)
        return err


# ── Export list ───────────────────────────────────────────────

ALL_TOOLS = [
    crawl_website,
    search_indexed,
    list_collections,
    list_sources,
    delete_source,
    web_search,
    analyze_site,
    search_products,
    search_devops,
    analyze_logs,
]
```

**Step 2: Commit**

```bash
git add rag-service/agent/tools.py
git commit -m "feat(agent): add 10 @function_tool wrappers for all agent tools"
```

---

### Task 4: Create agent/runner.py — streaming runner with AgentTask tracking

**Files:**
- Create: `rag-service/agent/runner.py`

**Step 1: Create agent/runner.py**

```python
"""
Agent runner — executes agent tasks with streaming event tracking.
Used by both FastAPI (background task) and CLI.
"""

import logging
import uuid
from datetime import datetime, timezone

from agents import Runner, ItemHelpers

from . import AgentTask, AgentServices

logger = logging.getLogger(__name__)


async def run_task(agent, services: AgentServices, prompt: str, max_turns: int = 30) -> AgentTask:
    """Run an agent task with streaming event tracking.

    Creates an AgentTask, runs the agent with streaming, and populates
    the task with step/log data as events arrive. Returns the completed task.
    """
    task = AgentTask(
        task_id=uuid.uuid4().hex[:12],
        prompt=prompt,
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    services.task = task

    # Discover model_id from agent
    try:
        model = agent.model
        task.model_id = getattr(model, 'model', None) or str(model)
    except Exception:
        task.model_id = "unknown"

    task.log(f"Starting with model: {task.model_id}")
    task.add_step("thinking", f"Planning approach for: {prompt}")

    try:
        result = Runner.run_streamed(
            agent,
            input=prompt,
            context=services,
            max_turns=max_turns,
        )

        async for event in result.stream_events():
            if event.type == "run_item_stream_event":
                item = event.item
                if item.type == "tool_call_item":
                    tool_name = getattr(item, 'name', None) or getattr(item, 'raw_item', {}).get('name', 'unknown')
                    task.add_step("tool_call", str(tool_name))
                    task.log(f"Tool call: {tool_name}")
                    if tool_name not in task.tools_called:
                        task.tools_called.append(tool_name)
                elif item.type == "tool_call_output_item":
                    output = str(getattr(item, 'output', ''))[:2000]
                    task.add_step("tool_result", output)
                elif item.type == "message_output_item":
                    text = ItemHelpers.text_message_output(item)
                    task.add_step("response", text)
                    task.log(f"Agent response: {text[:300]}")

        task.summary = result.final_output
        task.status = "completed"
        task.log(f"COMPLETED: {task.summary[:300] if task.summary else 'No summary'}")

    except Exception as e:
        task.status = "failed"
        task.error = str(e)[:500]
        task.log(f"FAILED: {str(e)[:400]}")
        logger.error(f"Agent task {task.task_id} failed: {e}", exc_info=True)

    task.ended_at = datetime.now(timezone.utc).isoformat()
    task.log(f"Task ended: {task.status}")
    return task
```

**Step 2: Commit**

```bash
git add rag-service/agent/runner.py
git commit -m "feat(agent): add streaming runner with AgentTask event tracking"
```

---

### Task 5: Create agent/cli.py — typer CLI entry point

**Files:**
- Create: `rag-service/agent/cli.py`

**Step 1: Create agent/cli.py**

```python
"""
CLI entry point for the RAG agent.
Usage: python -m agent.cli "Index all Kubernetes migration guides"
"""

import os
import sys
import logging

import typer
from agents import Runner

app = typer.Typer(help="Pocharlies RAG Agent CLI")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def _build_services():
    """Initialize services from environment variables (same as app.py lifespan)."""
    from qdrant_utils import make_qdrant_client
    from sentence_transformers import SentenceTransformer
    from openai import OpenAI

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    vllm_base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    llm_api_key = os.getenv("LITELLM_API_KEY", "none")

    logger.info("Loading embedding model...")
    model = SentenceTransformer(embedding_model)

    from web_indexer import WebIndexer
    from retriever import CodeRetriever
    from product_indexer import ProductIndexer
    from devops_indexer import DevOpsIndexer, LogAnalyzer

    web_indexer = WebIndexer(qdrant_url=qdrant_url, qdrant_api_key=qdrant_api_key, model=model)
    retriever = CodeRetriever(qdrant_url=qdrant_url, qdrant_api_key=qdrant_api_key, embedding_model=embedding_model)
    product_indexer = ProductIndexer(qdrant_url=qdrant_url, qdrant_api_key=qdrant_api_key, model=model)
    devops_indexer = DevOpsIndexer(qdrant_url=qdrant_url, qdrant_api_key=qdrant_api_key, model=model)
    llm_client = OpenAI(base_url=vllm_base_url, api_key=llm_api_key)
    log_analyzer = LogAnalyzer(llm_client=llm_client, devops_indexer=devops_indexer)

    from . import AgentServices
    return AgentServices(
        web_indexer=web_indexer,
        retriever=retriever,
        product_indexer=product_indexer,
        devops_indexer=devops_indexer,
        log_analyzer=log_analyzer,
        llm_client=llm_client,
    )


@app.command()
def run(
    prompt: str = typer.Argument(..., help="Task for the agent to perform"),
    max_turns: int = typer.Option(30, help="Maximum agent iterations"),
    vllm_url: str = typer.Option(
        None, "--vllm-url", envvar="VLLM_BASE_URL",
        help="vLLM API base URL (default: $VLLM_BASE_URL or http://localhost:8000/v1)",
    ),
):
    """Run the RAG agent with a natural language prompt."""
    vllm_url = vllm_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    api_key = os.getenv("LITELLM_API_KEY", "none")

    logger.info(f"Connecting to LLM at {vllm_url}...")
    from . import create_agent
    agent, model_id = create_agent(vllm_url, api_key)
    typer.echo(f"Model: {model_id}")

    logger.info("Initializing services...")
    services = _build_services()

    typer.echo(f"Running: {prompt}")
    typer.echo("---")

    result = Runner.run_sync(agent, input=prompt, context=services, max_turns=max_turns)
    typer.echo(result.final_output)


@app.command()
def status(
    vllm_url: str = typer.Option(
        None, "--vllm-url", envvar="VLLM_BASE_URL",
        help="vLLM API base URL",
    ),
):
    """Check if the LLM and Qdrant are available."""
    from openai import OpenAI
    from qdrant_utils import make_qdrant_client

    vllm_url = vllm_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    api_key = os.getenv("LITELLM_API_KEY", "none")

    # LLM check
    try:
        client = OpenAI(base_url=vllm_url, api_key=api_key)
        models = client.models.list()
        model_id = models.data[0].id if models.data else None
        typer.echo(f"LLM: OK ({model_id})")
    except Exception as e:
        typer.echo(f"LLM: FAIL ({e})")

    # Qdrant check
    try:
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        qdrant = make_qdrant_client(qdrant_url, qdrant_api_key)
        collections = [c.name for c in qdrant.get_collections().collections]
        typer.echo(f"Qdrant: OK ({len(collections)} collections: {', '.join(collections)})")
    except Exception as e:
        typer.echo(f"Qdrant: FAIL ({e})")


if __name__ == "__main__":
    app()
```

**Step 2: Create `rag-service/agent/__main__.py` for `python -m agent` support**

```python
from .cli import app

app()
```

**Step 3: Commit**

```bash
git add rag-service/agent/cli.py rag-service/agent/__main__.py
git commit -m "feat(agent): add typer CLI with run and status commands"
```

---

### Task 6: Update app.py — swap agent imports and endpoints

**Files:**
- Modify: `rag-service/app.py:25` (import line)
- Modify: `rag-service/app.py:82` (agent_tasks dict type)
- Modify: `rag-service/app.py:86-167` (lifespan — add agent creation)
- Modify: `rag-service/app.py:1349-1458` (agent endpoints — swap to new runner)

**Step 1: Change the import at line 25**

Replace:
```python
from agent import AgentTask, run_agent_task
```
With:
```python
from agent import AgentTask, AgentServices, create_agent
from agent.runner import run_task
```

**Step 2: Add agent instance to lifespan globals (line 87)**

Add `global ... rag_agent` to the lifespan function globals line. After the `log_analyzer` init (around line 148), add:

```python
    # Create SDK agent (after llm_client is ready)
    rag_agent, _model_id = create_agent(VLLM_BASE_URL, LITELLM_API_KEY)
    logger.info(f"Agent SDK initialized with model: {_model_id}")
```

Add `rag_agent = None` near the other global declarations (around line 61).

**Step 3: Update POST /agent/task endpoint (line 1349-1370)**

Replace the `asyncio.create_task(run_agent_task(...))` call at line 1367 with:

```python
    services = AgentServices(
        web_indexer=web_indexer,
        retriever=retriever,
        product_indexer=product_indexer,
        devops_indexer=devops_indexer,
        log_analyzer=log_analyzer,
        llm_client=llm_client,
    )

    async def _run_and_store(task_id: str):
        task = await run_task(rag_agent, services, prompt)
        agent_tasks[task_id] = task

    # Create placeholder task for immediate response
    task = AgentTask(task_id=task_id, prompt=prompt)
    task.started_at = datetime.now(timezone.utc).isoformat()
    agent_tasks[task_id] = task

    asyncio.create_task(_run_and_store(task_id))
```

Wait — this needs adjustment. The `run_task` function creates its own AgentTask internally. We need the task to be shared so the polling endpoint works. Let me restructure: have `run_task` accept an optional pre-created task, or have it update the dict reference.

Better approach: modify `run_task` to accept an existing `AgentTask` and update it in-place, or have the endpoint pass the tasks dict. Simplest: let `run_task` create the task, but store it immediately in `agent_tasks` before the background coroutine starts.

Revised approach for the endpoint:

```python
@app.post("/agent/task")
async def create_agent_task(request: AgentTaskRequest):
    _cleanup_old_agent_tasks()
    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    running = sum(1 for t in agent_tasks.values() if t.status == "running")
    if running >= 3:
        raise HTTPException(status_code=429, detail="Maximum 3 concurrent agent tasks")

    task_id = uuid.uuid4().hex[:12]
    task = AgentTask(task_id=task_id, prompt=prompt, started_at=datetime.now(timezone.utc).isoformat())
    agent_tasks[task_id] = task

    services = AgentServices(
        web_indexer=web_indexer,
        retriever=retriever,
        product_indexer=product_indexer,
        devops_indexer=devops_indexer,
        log_analyzer=log_analyzer,
        llm_client=llm_client,
        task=task,
    )

    async def _run():
        from agent.runner import run_task as _run_task
        await _run_task(rag_agent, services, prompt, task=task)

    asyncio.create_task(_run())
    logger.info(f"Started agent task {task_id}: {prompt[:80]}")
    return {"task_id": task_id, "status": "running"}
```

This requires a small change to `runner.py`: accept an optional `task` parameter instead of always creating one. Update `run_task` signature:

```python
async def run_task(agent, services: AgentServices, prompt: str, max_turns: int = 30, task: AgentTask = None) -> AgentTask:
    if task is None:
        task = AgentTask(
            task_id=uuid.uuid4().hex[:12],
            prompt=prompt,
            started_at=datetime.now(timezone.utc).isoformat(),
        )
    services.task = task
    # ... rest unchanged
```

**Step 4: Remove /agent/task/{task_id}/stop endpoint (lines 1434-1443)**

The stop endpoint used `_cancel_event` which no longer exists. Remove it entirely — the SDK doesn't support mid-run cancellation without custom hooks.

**Step 5: Remove /agent/task/{task_id}/context endpoint (lines 1446-1458)**

The context injection endpoint used `_context_queue` which no longer exists. Remove it.

**Step 6: Commit**

```bash
git add rag-service/app.py
git commit -m "refactor(app): swap agent endpoints to use Agents SDK runner"
```

---

### Task 7: Rename old agent.py and clean up

**Files:**
- Rename: `rag-service/agent.py` → `rag-service/agent_legacy.py`

**Step 1: Rename agent.py**

```bash
cd rag-service
mv agent.py agent_legacy.py
```

**Step 2: Verify no other files import from `agent` module directly**

Run: `grep -rn "from agent import\|import agent" rag-service/ --include="*.py" | grep -v agent_legacy | grep -v agent/`

Should only show `app.py` (which we already updated).

**Step 3: Commit**

```bash
git add rag-service/agent.py rag-service/agent_legacy.py
git commit -m "refactor: rename old agent.py to agent_legacy.py for safety"
```

---

### Task 8: Rebuild Docker image and test

**Files:**
- No file changes — testing the build and runtime

**Step 1: Rebuild the Docker image**

```bash
cd /home/ubuntu/skirmshopshopifyapp/pocharlies-qdrant
docker compose build rag-service
```

Expected: Build succeeds, `pip install` picks up `openai-agents` and `typer`.

**Step 2: Restart the service**

```bash
docker compose up -d rag-service
```

**Step 3: Check health**

```bash
curl -s http://127.0.0.1:5000/health | python3 -m json.tool
```

Expected: `{"status": "healthy", ...}`

**Step 4: Check agent status**

```bash
curl -s http://127.0.0.1:5000/agent/status | python3 -m json.tool
```

Expected: `{"available": true, "model_id": "..."}`

**Step 5: Run a simple agent task via API**

```bash
curl -s -X POST http://127.0.0.1:5000/agent/task \
  -H "Content-Type: application/json" \
  -d '{"prompt": "List all indexed sources and report how many collections exist"}' | python3 -m json.tool
```

Expected: Returns `{"task_id": "...", "status": "running"}`

**Step 6: Poll task status**

```bash
# Replace TASK_ID with the returned task_id
curl -s http://127.0.0.1:5000/agent/task/TASK_ID | python3 -m json.tool
```

Expected: Task completes with `status: "completed"`, `tools_called` includes `list_sources` and `list_collections`.

**Step 7: Commit (if any fixes were needed)**

```bash
git add -A && git commit -m "fix: address issues found during agent SDK integration testing"
```

---

### Task 9: Test CLI entry point

**Files:**
- No file changes — testing

**Step 1: Test CLI status command inside the container**

```bash
docker exec pocharlies-rag python -m agent status
```

Expected: Shows LLM status and Qdrant collections.

**Step 2: Test CLI run command**

```bash
docker exec pocharlies-rag python -m agent run "List all indexed sources"
```

Expected: Agent runs, calls tools, prints final summary.

**Step 3: Commit (if any fixes)**

```bash
git add -A && git commit -m "fix: address CLI testing issues"
```

---

### Task 10: Delete agent_legacy.py

Only do this after Tasks 8 and 9 pass successfully.

**Files:**
- Delete: `rag-service/agent_legacy.py`

**Step 1: Delete the legacy file**

```bash
rm rag-service/agent_legacy.py
```

**Step 2: Commit**

```bash
git add rag-service/agent_legacy.py
git commit -m "cleanup: remove agent_legacy.py after successful SDK migration"
```

---

## Summary

| Task | Description | Files | Est. |
|------|-------------|-------|------|
| 1 | Add dependencies | requirements.txt | 2 min |
| 2 | Create agent/__init__.py | agent/__init__.py | 5 min |
| 3 | Create agent/tools.py (10 tools) | agent/tools.py | 10 min |
| 4 | Create agent/runner.py | agent/runner.py | 5 min |
| 5 | Create agent/cli.py | agent/cli.py, agent/__main__.py | 5 min |
| 6 | Update app.py | app.py | 10 min |
| 7 | Rename old agent.py | agent.py → agent_legacy.py | 2 min |
| 8 | Docker build + API test | (no files) | 10 min |
| 9 | CLI test | (no files) | 5 min |
| 10 | Delete legacy file | agent_legacy.py | 1 min |
