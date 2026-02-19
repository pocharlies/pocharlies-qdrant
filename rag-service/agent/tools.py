"""
Agent tools — @function_tool wrappers around existing service methods.
Each tool pulls services from RunContextWrapper[AgentServices].
"""

import json
import logging
from typing import Optional

from agents import function_tool, RunContextWrapper

logger = logging.getLogger(__name__)

from . import AgentServices
Ctx = RunContextWrapper[AgentServices]


# ── Web & Content tools ────────────────────────────────────


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


# ── Product Catalog tools ────────────────────────────


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


# ── DevOps & SRE tools ───────────────────────────────


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


# ── Export list ───────────────────────────────────────

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
