"""
Autonomous AI Agent for RAG Service
Background task execution with tool calling via local LLM (OpenAI-compatible API).
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)


# ── Data Model ────────────────────────────────────────────────


@dataclass
class AgentTask:
    """Tracks an autonomous agent task. Follows CrawlJob pattern."""

    task_id: str
    prompt: str
    status: str = "running"  # running | completed | failed | cancelled
    steps: List[Dict[str, Any]] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    model_id: Optional[str] = None
    tools_called: List[str] = field(default_factory=list)
    summary: Optional[str] = None
    error: Optional[str] = None

    # Runtime control (not serialized)
    _cancel_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    _context_queue: asyncio.Queue = field(default_factory=asyncio.Queue, repr=False)

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


# ── Tool Definitions (OpenAI function-calling format) ─────────


AGENT_TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "crawl_website",
            "description": "Crawl a website and index its content into the RAG vector database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Starting URL to crawl"},
                    "max_depth": {"type": "integer", "description": "Link depth 0-5 (default: 2)"},
                    "max_pages": {"type": "integer", "description": "Max pages 1-500 (default: 50)"},
                    "smart_mode": {"type": "boolean", "description": "Use AI site analysis (default: true)"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_indexed",
            "description": "Search the RAG database for indexed content. Supports web pages, code repositories, or both.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "collection": {"type": "string", "enum": ["all", "web", "code"], "description": "Collection to search: 'all' (both), 'web' (web pages), 'code' (code repos). Default: all"},
                    "domain": {"type": "string", "description": "Filter by domain (web only, optional)"},
                    "repo": {"type": "string", "description": "Filter by repository name (code only, optional)"},
                    "top_k": {"type": "integer", "description": "Number of results (default: 5)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_collections",
            "description": "Get Qdrant vector database collection statistics.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_sources",
            "description": "List all indexed web domains with page and chunk counts.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_source",
            "description": "Delete all indexed content from a specific domain.",
            "parameters": {
                "type": "object",
                "properties": {
                    "domain": {"type": "string", "description": "Domain to delete (e.g. 'example.com')"},
                },
                "required": ["domain"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the internet to find URLs and information. Use to discover websites to crawl.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_site",
            "description": "Probe a URL to check accessibility, detect Cloudflare, and estimate content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to analyze"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Search the product catalog (Shopify products indexed in Qdrant). Supports brand, category, and price filters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query (e.g. 'Tokyo Marui Hi-Capa')"},
                    "brand": {"type": "string", "description": "Filter by brand name (optional)"},
                    "category": {"type": "string", "description": "Filter by category: gbb, aeg, sniper, pistol, shotgun, smg, accessory, gear, etc."},
                    "top_k": {"type": "integer", "description": "Number of results (default: 5)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_devops",
            "description": "Search DevOps documentation: runbooks, postmortems, config files, procedures.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query (e.g. 'vLLM OOM recovery')"},
                    "doc_type": {"type": "string", "description": "Filter by doc type: runbook, postmortem, config, procedure, architecture, documentation"},
                    "top_k": {"type": "integer", "description": "Number of results (default: 5)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_logs",
            "description": "Analyze log text to classify errors by severity, identify services, and match related runbooks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "log_text": {"type": "string", "description": "Log text to analyze (error logs, container output, etc.)"},
                    "service": {"type": "string", "description": "Source service name (e.g. 'vllm', 'litellm', 'nginx')"},
                },
                "required": ["log_text"],
            },
        },
    },
]


# ── Tool Implementations ──────────────────────────────────────


async def tool_crawl_website(web_indexer, task, llm_client, url, max_depth=2, max_pages=50, smart_mode=True):
    task.log(f"TOOL crawl_website: {url} (depth={max_depth}, pages={max_pages}, smart={smart_mode})")
    task.add_step("tool_call", f"crawl_website({url}, depth={max_depth}, pages={max_pages})")
    try:
        job = await web_indexer.crawl_and_index(
            start_url=url,
            max_depth=int(max_depth),
            max_pages=int(max_pages),
            smart_mode=bool(smart_mode),
            llm_client=llm_client,
        )
        result = f"Crawled {url}: {job.pages_indexed} pages indexed, {job.chunks_indexed} chunks. Status: {job.status}"
        task.add_step("tool_result", result)
        task.log(f"RESULT: {result}")
        return result
    except Exception as e:
        err = f"Crawl failed for {url}: {str(e)[:300]}"
        task.add_step("tool_result", err)
        task.log(f"ERROR: {err}")
        return err


async def tool_search_indexed(web_indexer, retriever, task, query, collection="all", domain=None, repo=None, top_k=5):
    task.log(f"TOOL search_indexed: '{query}' (collection={collection}, domain={domain}, repo={repo}, top_k={top_k})")
    task.add_step("tool_call", f"search_indexed('{query}', collection={collection})")
    try:
        formatted = []
        top_k = int(top_k)

        if collection in ("all", "web"):
            web_results = web_indexer.search(query=query, top_k=top_k, domain_filter=domain)
            for r in web_results:
                formatted.append(f"[WEB {r['score']:.2f}] {r.get('title', 'Untitled')} — {r['url']}\n{r['text'][:300]}")

        if collection in ("all", "code") and retriever:
            code_results = retriever.retrieve(query=query, top_k=top_k, repo_filter=repo)
            for r in code_results:
                symbols_str = f" | symbols: {', '.join(r.symbols)}" if r.symbols else ""
                formatted.append(f"[CODE {r.score:.2f}] {r.repo}/{r.path}#L{r.start_line}-{r.end_line}{symbols_str}\n{r.text[:300]}")

        if not formatted:
            msg = f"No results found for '{query}'"
            task.add_step("tool_result", msg)
            return msg

        result = "\n\n".join(formatted)
        task.add_step("tool_result", f"{len(formatted)} results found")
        task.log(f"RESULT: {len(formatted)} results")
        return result
    except Exception as e:
        err = f"Search failed: {str(e)[:300]}"
        task.add_step("tool_result", err)
        return err


async def tool_list_collections(web_indexer, task):
    task.log("TOOL list_collections")
    task.add_step("tool_call", "list_collections()")
    try:
        stats = web_indexer.get_collection_stats()
        result = json.dumps(stats, indent=2)
        task.add_step("tool_result", result)
        return result
    except Exception as e:
        err = f"Failed: {str(e)[:300]}"
        task.add_step("tool_result", err)
        return err


async def tool_list_sources(web_indexer, task):
    task.log("TOOL list_sources")
    task.add_step("tool_call", "list_sources()")
    try:
        sources = web_indexer.get_sources()
        if not sources:
            msg = "No sources indexed."
            task.add_step("tool_result", msg)
            return msg
        lines = [f"{s['domain']}: {s['url_count']} pages, {s['chunk_count']} chunks" for s in sources]
        result = "\n".join(lines)
        task.add_step("tool_result", f"{len(sources)} domains")
        return result
    except Exception as e:
        err = f"Failed: {str(e)[:300]}"
        task.add_step("tool_result", err)
        return err


async def tool_delete_source(web_indexer, task, domain):
    task.log(f"TOOL delete_source: {domain}")
    task.add_step("tool_call", f"delete_source('{domain}')")
    try:
        success = web_indexer.delete_source(domain)
        result = f"Deleted {domain}" if success else f"Failed to delete {domain}"
        task.add_step("tool_result", result)
        return result
    except Exception as e:
        err = f"Failed: {str(e)[:300]}"
        task.add_step("tool_result", err)
        return err


async def tool_web_search(task, query):
    import httpx as _httpx
    from bs4 import BeautifulSoup

    task.log(f"TOOL web_search: '{query}'")
    task.add_step("tool_call", f"web_search('{query}')")
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
            task.add_step("tool_result", msg)
            return msg

        result = "\n\n".join(results)
        task.add_step("tool_result", f"{len(results)} web results")
        task.log(f"RESULT: {len(results)} web results")
        return result
    except Exception as e:
        err = f"Web search failed: {str(e)[:300]}"
        task.add_step("tool_result", err)
        return err


async def tool_analyze_site(task, url):
    import httpx as _httpx
    from bs4 import BeautifulSoup

    task.log(f"TOOL analyze_site: {url}")
    task.add_step("tool_call", f"analyze_site('{url}')")
    try:
        async with _httpx.AsyncClient(
            follow_redirects=True,
            timeout=10.0,
            headers={"User-Agent": "Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36"},
        ) as client:
            resp = await client.get(url)

        body_lower = resp.text.lower() if resp.text else ""
        cf_indicators = ["cloudflare", "cf-browser-verification", "just a moment", "cf-challenge"]
        has_cf = any(i in body_lower for i in cf_indicators)

        soup = BeautifulSoup(resp.text, "html.parser")
        title_tag = soup.find("title")
        links = soup.find_all("a", href=True)

        info = {
            "url": str(resp.url),
            "status_code": resp.status_code,
            "content_type": resp.headers.get("content-type", "unknown"),
            "content_length": len(resp.text) if resp.text else 0,
            "cloudflare_detected": has_cf,
            "title": title_tag.get_text(strip=True)[:200] if title_tag else "",
            "link_count": len(links),
        }

        result = json.dumps(info, indent=2)
        task.add_step("tool_result", result)
        return result
    except Exception as e:
        err = f"Site analysis failed: {str(e)[:300]}"
        task.add_step("tool_result", err)
        return err


async def tool_search_products(product_indexer, task, query, brand=None, category=None, top_k=5):
    task.log(f"TOOL search_products: '{query}' (brand={brand}, category={category})")
    task.add_step("tool_call", f"search_products('{query}', brand={brand}, category={category})")
    try:
        results = product_indexer.search(
            query=query,
            top_k=int(top_k),
            brand_filter=brand,
            category_filter=category,
        )
        if not results:
            msg = f"No products found for '{query}'"
            task.add_step("tool_result", msg)
            return msg

        formatted = []
        for r in results:
            price_str = f" €{r['price']}" if r.get('price') else ""
            formatted.append(f"[PRODUCT {r['score']:.2f}] {r.get('brand', '')} {r.get('title', '')}{price_str}\n  SKU: {r.get('sku', 'N/A')} | Category: {r.get('category', 'N/A')}\n  {r['text'][:200]}")

        result = "\n\n".join(formatted)
        task.add_step("tool_result", f"{len(formatted)} products found")
        task.log(f"RESULT: {len(formatted)} products")
        return result
    except Exception as e:
        err = f"Product search failed: {str(e)[:300]}"
        task.add_step("tool_result", err)
        return err


async def tool_search_devops(devops_indexer, task, query, doc_type=None, top_k=5):
    task.log(f"TOOL search_devops: '{query}' (doc_type={doc_type})")
    task.add_step("tool_call", f"search_devops('{query}', doc_type={doc_type})")
    try:
        results = devops_indexer.search(
            query=query,
            top_k=int(top_k),
            doc_type_filter=doc_type,
        )
        if not results:
            msg = f"No DevOps docs found for '{query}'"
            task.add_step("tool_result", msg)
            return msg

        formatted = []
        for r in results:
            formatted.append(f"[DEVOPS {r['score']:.2f}] {r.get('title', 'Untitled')} ({r.get('doc_type', 'unknown')})\n  Source: {r.get('source_path', 'N/A')}\n  {r['text'][:300]}")

        result = "\n\n".join(formatted)
        task.add_step("tool_result", f"{len(formatted)} docs found")
        task.log(f"RESULT: {len(formatted)} devops docs")
        return result
    except Exception as e:
        err = f"DevOps search failed: {str(e)[:300]}"
        task.add_step("tool_result", err)
        return err


async def tool_analyze_logs(log_analyzer, task, log_text, service="unknown"):
    task.log(f"TOOL analyze_logs: {len(log_text)} chars from '{service}'")
    task.add_step("tool_call", f"analyze_logs(service={service}, {len(log_text)} chars)")
    try:
        job = await log_analyzer.analyze_logs(log_text=log_text, source_service=service)
        if not job.results:
            msg = "No notable errors or warnings found in the log text."
            task.add_step("tool_result", msg)
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
        task.add_step("tool_result", f"{len(formatted)} issues found ({', '.join(f'{k}: {v}' for k, v in job.categories.items())})")
        task.log(f"RESULT: {len(formatted)} log issues")
        return result
    except Exception as e:
        err = f"Log analysis failed: {str(e)[:300]}"
        task.add_step("tool_result", err)
        return err


# ── Agent Runner ──────────────────────────────────────────────


class AgentCancelled(Exception):
    pass


AGENT_SYSTEM_PROMPT = """You are an autonomous AI agent managing a multi-purpose RAG (Retrieval-Augmented Generation) system for an airsoft e-commerce business and DevOps operations. You have tools to:

**Web & Content:**
1. **web_search** — Search the internet to find relevant URLs
2. **analyze_site** — Check if a URL is accessible before crawling
3. **crawl_website** — Crawl and index a website into the vector database
4. **search_indexed** — Search already-indexed content (web pages, code repos, or both)
5. **list_collections** — View database statistics for all collections
6. **list_sources** — See what domains are already indexed
7. **delete_source** — Remove a domain's content from the index

**Product Catalog:**
8. **search_products** — Search the Shopify product catalog with brand/category filters

**DevOps & SRE:**
9. **search_devops** — Search DevOps documentation (runbooks, postmortems, config, procedures)
10. **analyze_logs** — Analyze log text to classify errors, identify services, and match runbooks

When given a task:
- Plan your approach step by step
- Use web_search to find relevant URLs when needed
- Use analyze_site to check URLs before committing to crawl them
- Use crawl_website to index useful content
- Use search_indexed to verify content was indexed correctly
- Use search_products for product catalog queries
- Use search_devops and analyze_logs for DevOps/SRE tasks
- Report your progress and findings clearly

Be efficient: don't crawl sites that are already indexed (check with list_sources first).
Be thorough: verify your work by searching after indexing.
When finished, provide a clear summary of what you accomplished."""

MAX_AGENT_ITERATIONS = 30


async def run_agent_task(task: AgentTask, llm_client, web_indexer, retriever=None, product_indexer=None, devops_indexer=None, log_analyzer=None):
    """Execute the agent loop. Runs as asyncio background task."""
    task.started_at = datetime.now(timezone.utc).isoformat()
    task.status = "running"

    # Discover model
    model_id = None
    try:
        models = await asyncio.get_event_loop().run_in_executor(
            None, llm_client.models.list
        )
        model_id = models.data[0].id if models.data else None
    except Exception as e:
        logger.warning(f"Failed to discover model: {e}")

    if not model_id:
        task.status = "failed"
        task.error = "No LLM model available. Check LLM_BASE_URL."
        task.log("FAILED: No LLM model available")
        task.ended_at = datetime.now(timezone.utc).isoformat()
        return

    task.model_id = model_id
    task.log(f"Starting with model: {model_id}")
    task.add_step("thinking", f"Planning approach for: {task.prompt}")

    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": task.prompt},
    ]

    tool_dispatch = {
        "crawl_website": lambda **kw: tool_crawl_website(web_indexer, task, llm_client, **kw),
        "search_indexed": lambda **kw: tool_search_indexed(web_indexer, retriever, task, **kw),
        "list_collections": lambda **kw: tool_list_collections(web_indexer, task),
        "list_sources": lambda **kw: tool_list_sources(web_indexer, task),
        "delete_source": lambda **kw: tool_delete_source(web_indexer, task, **kw),
        "web_search": lambda **kw: tool_web_search(task, **kw),
        "analyze_site": lambda **kw: tool_analyze_site(task, **kw),
    }
    if product_indexer:
        tool_dispatch["search_products"] = lambda **kw: tool_search_products(product_indexer, task, **kw)
    if devops_indexer:
        tool_dispatch["search_devops"] = lambda **kw: tool_search_devops(devops_indexer, task, **kw)
    if log_analyzer:
        tool_dispatch["analyze_logs"] = lambda **kw: tool_analyze_logs(log_analyzer, task, **kw)

    try:
        for iteration in range(MAX_AGENT_ITERATIONS):
            # Check cancellation
            if task._cancel_event.is_set():
                raise AgentCancelled()

            # Drain context injection queue
            while not task._context_queue.empty():
                try:
                    injected = task._context_queue.get_nowait()
                    messages.append({"role": "user", "content": injected})
                    task.log(f"CONTEXT INJECTED: {injected[:200]}")
                    task.add_step("context_injected", injected)
                except asyncio.QueueEmpty:
                    break

            # Call LLM (non-blocking)
            task.log(f"LLM call (iteration {iteration + 1})...")
            try:
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: llm_client.chat.completions.create(
                        model=model_id,
                        messages=messages,
                        tools=AGENT_TOOL_DEFINITIONS,
                        tool_choice="auto",
                        max_tokens=4096,
                        temperature=0.3,
                    ),
                )
            except Exception as e:
                task.log(f"LLM call failed: {str(e)[:300]}")
                task.status = "failed"
                task.error = f"LLM error: {str(e)[:300]}"
                break

            message = response.choices[0].message

            # No tool calls = agent is done
            if not message.tool_calls:
                content = message.content or ""
                task.log(f"Agent response: {content[:300]}")
                task.add_step("response", content)
                task.summary = content
                task.status = "completed"
                messages.append({"role": "assistant", "content": content})
                break

            # Process tool calls
            # Build assistant message dict for conversation history
            assistant_msg = {"role": "assistant", "content": message.content or ""}
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]
            messages.append(assistant_msg)

            for tc in message.tool_calls:
                # Check cancellation before each tool
                if task._cancel_event.is_set():
                    raise AgentCancelled()

                func_name = tc.function.name
                try:
                    func_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    func_args = {}

                task.log(f"Tool: {func_name}({json.dumps(func_args)[:200]})")

                if func_name not in task.tools_called:
                    task.tools_called.append(func_name)

                if func_name in tool_dispatch:
                    result = await tool_dispatch[func_name](**func_args)
                else:
                    result = f"Unknown tool: {func_name}"
                    task.log(f"WARNING: Unknown tool '{func_name}'")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result)[:8000],
                })

        else:
            task.log(f"Hit max iterations ({MAX_AGENT_ITERATIONS})")
            task.status = "completed"
            task.summary = "Task reached maximum iteration limit."

    except AgentCancelled:
        task.status = "cancelled"
        task.log("Task cancelled by user")
        task.add_step("cancelled", "Task was cancelled by the user")

    except Exception as e:
        task.status = "failed"
        task.error = str(e)[:500]
        task.log(f"FATAL: {str(e)[:400]}")
        logger.error(f"Agent task {task.task_id} failed: {e}", exc_info=True)

    finally:
        task.ended_at = datetime.now(timezone.utc).isoformat()
        task.log(f"Task ended: {task.status}")
