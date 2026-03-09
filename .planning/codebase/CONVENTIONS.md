# Coding Conventions

**Analysis Date:** 2026-03-09

## Language & Runtime

**Language:** Python 3.11 (specified in all Dockerfiles)
**No linting/formatting tools configured.** No `.flake8`, `.pylintrc`, `.prettierrc`, `pyproject.toml`, `ruff.toml`, or `eslint` config exists. Code style is enforced by convention only.

## Naming Patterns

**Files:**
- Use `snake_case.py` for all modules: `shopify_client.py`, `web_indexer.py`, `product_classifier.py`
- Suffix with role: `*_indexer.py` for indexing, `*_client.py` for API clients, `*_handler.py` for event handlers
- No `__init__.py` barrel files with re-exports (they are empty except `rag-service/agent/__init__.py`)

**Classes:**
- PascalCase: `ShopifyClient`, `ProductIndexer`, `WebIndexer`, `MCPManager`
- Use descriptive suffixes matching domain: `*Client`, `*Indexer`, `*Store`, `*Handler`, `*Pipeline`
- Dataclasses for job/task tracking: `CrawlJob`, `ProductSyncJob`, `TranslationJob`, `AgentTask`

**Functions/Methods:**
- snake_case: `extract_product_text()`, `index_all_products()`, `incremental_sync()`
- Private methods with single underscore: `_ensure_collection()`, `_process_product_batch()`, `_translate_chunk()`
- Async functions use same naming (no `async_` prefix)

**Variables:**
- Module-level constants in UPPER_SNAKE_CASE: `QDRANT_URL`, `EMBEDDING_MODEL`, `COLLECTION_NAME`
- Module-level singleton instances in lowercase: `web_indexer`, `llm_client`, `redis_client`
- Private module globals with underscore: `_bm25_model`, `_reranker_instance`, `_crawl_running`

**FastAPI Routers (agent-service):**
- Named `router` within each API module
- Imported and aliased in `main.py`: `from api.chat import router as chat_router`

## Code Style

**Formatting:**
- No automated formatter configured
- Indentation: 4 spaces (Python standard)
- Line length: no enforced limit, lines regularly exceed 120 characters
- String quotes: double quotes `"` predominantly, with occasional single quotes for dictionary keys

**Type Hints:**
- Use `typing` module types in rag-service: `List`, `Dict`, `Optional`, `Callable`
- Use Python 3.10+ syntax in agent-service: `list[dict]`, `str | None`, `dict[str, Any]`
- Function signatures generally include type hints for parameters and return values
- Class attributes in dataclasses are fully typed

**Docstrings:**
- Module-level docstrings on every file (triple-quoted, one-line or multi-line)
- Class docstrings present on most classes (brief, one line)
- Method docstrings on public methods; typically describe purpose and key behavior
- Use `Args:` block in some methods (inconsistent)
- No docstrings on private methods

## Import Organization

**Order:**
1. Standard library (`os`, `json`, `asyncio`, `logging`, `uuid`, `hashlib`, etc.)
2. Third-party packages (`fastapi`, `pydantic`, `httpx`, `qdrant_client`, `sentence_transformers`, etc.)
3. Local/project imports (`from web_indexer import WebIndexer`, `from config import settings`)

**Notes:**
- No path aliases (no `pyproject.toml` with `tool.setuptools.package-dir`)
- Imports are absolute within each service (flat module layout in rag-service, package-based in agent-service)
- Lazy imports used in several places to avoid circular dependencies:
  - `rag-service/webhook_handler.py`: `from shopify_graphql import ShopifyGraphQL` inside method
  - `rag-service/product_indexer.py`: `from sparse_encoder import encode_sparse` inside method
  - `rag-service/agent/cli.py`: `from . import create_agent` inside function
- No `__all__` exports in any module

## Configuration Patterns

**rag-service:**
- Module-level `os.getenv()` calls with defaults at the top of `app.py`
- No settings class; all config is bare `os.getenv()` in `app.py` lines 44-54
- Global mutable state for service instances (lines 57-73 of `app.py`)
- Initialized in `lifespan()` async context manager

**agent-service:**
- Pydantic `BaseSettings` class in `config.py` with env var binding
- Single `settings = Settings()` singleton imported throughout
- Clean separation: `config.py` owns all env vars, other modules import `settings`

**mcp-server:**
- Module-level `os.getenv()` like rag-service (lines 50-55 of `server.py`)

**Prescriptive rule:** For new code in agent-service, use the `Settings` pattern from `agent-service/config.py`. For rag-service and mcp-server, use module-level `os.getenv()` at the top of the file.

## Error Handling

**General Pattern:**
- Broad `except Exception as e:` catches with logging, returning safe defaults
- No custom exception classes anywhere in the codebase
- Services return empty lists `[]` or `None` on failure rather than raising
- Job/task objects track errors in status fields: `job.status = "failed"`, `job.errors.append(str(e)[:200])`

**FastAPI endpoints (agent-service):**
- Use `HTTPException` for 404/503 errors
- Let unexpected exceptions propagate (FastAPI returns 500 automatically)
- Pattern in `api/chat.py`: catch, log with `logger.exception()`, re-raise

**Tools (rag-service/agent/tools.py):**
- Every `@function_tool` wraps body in try/except
- Returns error string on failure: `return f"Crawl failed for {url}: {str(e)[:300]}"`
- Never raises from tools (agent SDK expects string returns)

**Async operations:**
- `loop.run_in_executor(None, lambda: ...)` to run blocking calls (embedding, Qdrant upserts)
- Semaphore pattern for concurrency control: `asyncio.Semaphore(self.MAX_CONCURRENT_CHUNKS)`

**Prescriptive rule:** Always catch exceptions at service boundaries and return safe defaults. Log with `logger.error()` or `logger.warning()`. Truncate error messages to 200-300 chars in job status fields.

## Logging

**Framework:** Python stdlib `logging`

**Setup patterns:**
- rag-service: `logging.basicConfig(level=logging.INFO)` at module level, `logger = logging.getLogger(__name__)`
- agent-service: `logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")` in `main.py`, named loggers like `logging.getLogger("agent.api.chat")`
- mcp-server: `logging.basicConfig(level=logging.INFO, stream=sys.stderr)` (stdout reserved for MCP JSON-RPC)

**Log message style:**
- f-strings for interpolation: `logger.info(f"Fetched {len(products)} products")`
- agent-service uses %-style: `logger.info("Supervisor graph compiled with %d tools", len(tools))`
- Both styles coexist; no enforced standard

**Prescriptive rule:** Use `logger = logging.getLogger(__name__)` at module level. Use f-strings for log messages in rag-service/mcp-server, %-style in agent-service. Log at INFO for operations, WARNING for recoverable issues, ERROR for failures.

## Comments

**When to Comment:**
- Section dividers using Unicode box-drawing characters: `# -- Text & metadata extraction -----`
- Inline comments for non-obvious logic: `# Shopify rate limit: ~2 req/sec`
- Constants explained: `# Indicators of Cloudflare/bot protection`

**Style:**
- Section headers: `# -- Section Name --` with Unicode line characters (em dash variants)
- Agent-service uses ASCII comment blocks: `# ---------------------------------------------------------------------------`

## Pydantic Models

**Agent-service pattern:**
```python
class ChatRequest(BaseModel):
    """Incoming chat message."""
    message: str = Field(..., description="User message text")
    thread_id: str | None = Field(None, description="Conversation thread ID.")

class ChatResponse(BaseModel):
    """Agent response payload."""
    thread_id: str = Field(..., description="Conversation thread ID")
    message: str = Field(..., description="Agent reply text")
    tool_calls: list[dict] = Field(default_factory=list, description="Tool calls made")
```

**Prescriptive rules:**
- Use `Field(...)` for required fields with descriptions
- Use `Field(None)` or `Field(default_factory=...)` for optional fields
- Include `model_config = {"from_attributes": True}` when mapping from ORM models
- Place Pydantic models at the top of each API module, before endpoint functions

## Dataclass Job Tracking Pattern

All long-running operations use a consistent `@dataclass` job tracker:

```python
@dataclass
class SomeJob:
    job_id: str
    status: str = "running"  # running | completed | failed
    items_processed: int = 0
    items_total: int = 0
    logs: List[str] = field(default_factory=list)
    started_at: Optional[str] = None
    ended_at: Optional[str] = None

    MAX_LOGS = 3000

    def log(self, msg: str):
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        self.logs.append(f"[{ts}] {msg}")
        if len(self.logs) > self.MAX_LOGS:
            self.logs = self.logs[:50] + self.logs[-(self.MAX_LOGS - 50):]

    def to_dict(self) -> dict:
        return { ... }
```

**Used in:** `CrawlJob`, `ProductSyncJob`, `ClassificationJob`, `TranslationJob`, `AgentTask`

**Prescriptive rule:** New long-running operations must follow this pattern. Include `job_id`, `status`, `log()` with MAX_LOGS rotation, and `to_dict()` for API serialization.

## Module Design

**Exports:** No barrel files. Import directly from the module: `from product_indexer import ProductIndexer`

**Singleton Pattern:**
- Module-level private global + getter function:
  ```python
  _reranker_instance: Optional["Reranker"] = None

  def get_reranker() -> Optional[Reranker]:
      return _reranker_instance

  def init_reranker(model_name: str) -> Reranker:
      global _reranker_instance
      _reranker_instance = Reranker(model_name)
      return _reranker_instance
  ```
- Used for expensive-to-load ML models: `sparse_encoder.py`, `reranker.py`

**Service Initialization:**
- rag-service: All service instances created in `app.py:lifespan()`, stored as module-level globals
- agent-service: Service instances stored on `app.state` (FastAPI pattern): `app.state.supervisor`, `app.state.mcp_manager`

**Prescriptive rule:** In agent-service, attach runtime state to `app.state`. In rag-service, use module-level globals initialized in `lifespan()`.

## Redis Key Naming

Consistent prefix-based naming for Redis keys:
- `agent:task:{task_id}` - Task metadata hash
- `agent:tasks` - Sorted set of task IDs
- `agent:session:{session_id}` - Session history list
- `catalog:sync:{sync_id}` - Sync state hash
- `catalog:hash:{item_key}` - Content dedup hashes
- `translation:glossary:{source}:{target}` - Glossary entries

**Prescriptive rule:** Use colon-separated hierarchical keys. Define `PREFIX` as a class constant.

---

*Convention analysis: 2026-03-09*
