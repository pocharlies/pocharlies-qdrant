# Agent SDK Migration Design

Replace the hand-rolled agent loop in `agent.py` with OpenAI Agents SDK, reorganized as an `agent/` package with CLI support.

## Motivation

The current `agent.py` (776 lines) implements a manual tool-calling loop: call LLM, parse tool_calls, dispatch, append results, repeat. The OpenAI Agents SDK provides this loop natively with streaming, automatic schema generation, typed context injection, and `max_turns` control. Migrating eliminates custom plumbing while keeping all existing tool logic.

## Scope

**Replace:** `agent.py` (hand-rolled loop, manual tool dispatch, JSON tool definitions)
**Keep:** All indexers (`web_indexer.py`, `product_indexer.py`, `devops_indexer.py`, etc.), `app.py` (with minor endpoint changes), `shopify_client.py`, all other services.
**Add:** CLI entry point for terminal-based agent interaction.

## Architecture

### File Structure

```
rag-service/
  agent/                  # NEW package (replaces agent.py)
    __init__.py           # Exports: create_agent(), AgentServices, ALL_TOOLS
    tools.py              # 10 @function_tool wrappers
    runner.py             # run_task() — streaming runner with AgentTask tracking
    cli.py                # typer CLI entry point
  agent.py                # DELETED
  app.py                  # MODIFIED — agent endpoints import from agent/
  requirements.txt        # MODIFIED — +openai-agents, +typer
```

### Dependencies

```
openai-agents>=0.7.0     # Agent framework (replaces hand-rolled loop)
typer>=0.15.0             # CLI interface
```

## Design

### 1. Context Object (agent/__init__.py)

All tools receive shared services via `RunContextWrapper[AgentServices]`:

```python
@dataclass
class AgentServices:
    web_indexer: WebIndexer
    retriever: Optional[CodeRetriever]
    product_indexer: Optional[ProductIndexer]
    devops_indexer: Optional[DevOpsIndexer]
    log_analyzer: Optional[LogAnalyzer]
    llm_client: OpenAI
    task: Optional[AgentTask] = None  # None in CLI mode
```

### 2. Tools (agent/tools.py)

Each of the 10 current tools becomes a `@function_tool` that pulls services from context. The SDK auto-generates JSON schemas from type annotations and docstrings.

Tools to migrate:
1. `crawl_website(url, max_depth, max_pages, smart_mode)` — web crawling
2. `search_indexed(query, collection, domain, repo, top_k)` — RAG search
3. `list_collections()` — Qdrant stats
4. `list_sources()` — indexed domains
5. `delete_source(domain)` — remove domain content
6. `web_search(query)` — DuckDuckGo search
7. `analyze_site(url)` — site accessibility check
8. `search_products(query, brand, category, top_k)` — product catalog search
9. `search_devops(query, doc_type, top_k)` — DevOps doc search
10. `analyze_logs(log_text, service)` — log analysis

Pattern for each tool:

```python
@function_tool
async def crawl_website(
    ctx: RunContextWrapper[AgentServices],
    url: str,
    max_depth: int = 2,
    max_pages: int = 50,
    smart_mode: bool = True,
) -> str:
    """Crawl a website and index its content into the RAG vector database."""
    svc = ctx.context
    if svc.task:
        svc.task.log(f"TOOL crawl_website: {url}")
        svc.task.add_step("tool_call", f"crawl_website({url}, depth={max_depth})")

    job = await svc.web_indexer.crawl_and_index(
        start_url=url, max_depth=max_depth, max_pages=max_pages,
        smart_mode=smart_mode, llm_client=svc.llm_client,
    )
    result = f"Crawled {url}: {job.pages_indexed} pages, {job.chunks_indexed} chunks"
    # ... diagnostics formatting (same logic as current agent.py)

    if svc.task:
        svc.task.add_step("tool_result", result)
    return result
```

### 3. Agent Creation (agent/__init__.py)

```python
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel

def create_agent(vllm_base_url: str, api_key: str = "none") -> Agent:
    client = AsyncOpenAI(base_url=vllm_base_url, api_key=api_key)

    # Auto-discover model ID from vLLM
    sync_client = OpenAI(base_url=vllm_base_url, api_key=api_key)
    models = sync_client.models.list()
    model_id = models.data[0].id if models.data else "default"

    return Agent(
        name="pocharlies-rag",
        instructions=SYSTEM_PROMPT,
        tools=ALL_TOOLS,
        model=OpenAIChatCompletionsModel(model=model_id, openai_client=client),
        model_settings=ModelSettings(temperature=0.3, max_tokens=4096),
    )
```

Critical: Must use `OpenAIChatCompletionsModel`, not the default model string. vLLM only exposes Chat Completions API, not OpenAI's Responses API.

### 4. Streaming Runner (agent/runner.py)

Runs the agent as an async background task with live event tracking:

```python
async def run_task(agent: Agent, services: AgentServices, prompt: str) -> AgentTask:
    task = AgentTask(task_id=uuid4().hex[:12], prompt=prompt, ...)
    services.task = task

    result = Runner.run_streamed(
        agent, input=prompt,
        context=services,
        max_turns=30,
    )

    async for event in result.stream_events():
        if event.type == "run_item_stream_event":
            if event.item.type == "tool_call_item":
                task.add_step("tool_call", event.item.name)
                if event.item.name not in task.tools_called:
                    task.tools_called.append(event.item.name)
            elif event.item.type == "tool_call_output_item":
                task.add_step("tool_result", str(event.item.output)[:2000])
            elif event.item.type == "message_output_item":
                text = ItemHelpers.text_message_output(event.item)
                task.add_step("response", text)

    task.summary = result.final_output
    task.status = "completed"
    task.ended_at = datetime.now(timezone.utc).isoformat()
    return task
```

The `AgentTask` dataclass stays as-is (minus `_cancel_event` and `_context_queue` fields). The web UI polls `GET /agent/tasks` — same API, same data shape.

### 5. FastAPI Integration (app.py changes)

Minimal changes:

```python
# Replace:
from agent import AgentTask, run_agent_task
# With:
from agent import create_agent, AgentServices, AgentTask
from agent.runner import run_task

# In lifespan — create agent once:
agent = create_agent(VLLM_BASE_URL, LITELLM_API_KEY)

# In POST /agent/run — use new runner:
services = AgentServices(
    web_indexer=web_indexer, retriever=retriever,
    product_indexer=product_indexer, devops_indexer=devops_indexer,
    log_analyzer=log_analyzer, llm_client=llm_client,
)
asyncio.create_task(run_task(agent, services, request.prompt))
```

### 6. CLI (agent/cli.py)

```python
import typer
from agents import Runner

app = typer.Typer()

@app.command()
def run(prompt: str, max_turns: int = 30, vllm_url: str = "http://localhost:8000/v1"):
    """Run the RAG agent from the command line."""
    agent = create_agent(vllm_url)
    services = AgentServices(...)  # init indexers from env vars
    result = Runner.run_sync(agent, input=prompt, context=services, max_turns=max_turns)
    print(result.final_output)
```

Usage: `python -m agent.cli run "Index all Kubernetes ingress migration guides"`

### System Prompt

Keep the existing `AGENT_SYSTEM_PROMPT` unchanged. It works well and the SDK doesn't impose any prompt format requirements.

## What Gets Deleted

- `agent.py` (776 lines) — entire file
- `AGENT_TOOL_DEFINITIONS` (142 lines of JSON) — replaced by SDK auto-schema
- `run_agent_task()` loop (117 lines) — replaced by `Runner.run_streamed()`
- Manual `tool_dispatch` dict (16 lines) — replaced by SDK routing
- `AgentCancelled` exception class — no longer needed
- `_cancel_event` and `_context_queue` fields on AgentTask — dropped per design decision

## Error Handling

- Tool errors: SDK catches exceptions from `@function_tool` and returns error message to LLM (same as current `TypeError` catch)
- LLM errors: SDK raises `ModelHTTPError` — caught in `run_task()`, sets `task.status = "failed"`
- Timeouts: `@function_tool(timeout=N)` per-tool, plus SDK `max_turns` for overall loop

## Migration Safety

1. Keep `agent.py` until new code is tested (rename to `agent_legacy.py`)
2. New `agent/` package is additive — no existing imports break until we swap the endpoints
3. `AgentTask.to_dict()` output shape is unchanged — web UI works without changes
4. Test with: `python -m agent.cli run "list all indexed sources"` before wiring to FastAPI
