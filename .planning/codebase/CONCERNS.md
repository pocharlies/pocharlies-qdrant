# Codebase Concerns

**Analysis Date:** 2026-03-09

## Tech Debt

**Monolithic rag-service/app.py (1943 lines):**
- Issue: `rag-service/app.py` is a single 1943-line file containing all FastAPI route handlers, startup logic, and ~15 global mutable variables. It handles web indexing, product sync, translation, classification, agent tasks, chat, orders, webhooks, and health endpoints in one file.
- Files: `rag-service/app.py`
- Impact: Difficult to navigate, test, or modify any single feature without risk of breaking others. Merge conflicts are likely when multiple changes are in flight.
- Fix approach: Split into a FastAPI router-per-domain structure (e.g., `api/crawl.py`, `api/products.py`, `api/translate.py`, `api/orders.py`, `api/webhooks.py`, `api/agent.py`). Move global state into a dependency-injected services container.

**Monolithic mcp-server/server.py (1604 lines):**
- Issue: `mcp-server/server.py` contains all MCP tool definitions, web crawling logic, search functions, embedding logic, and helper utilities in a single file. It duplicates functionality already present in `rag-service/` (crawling, search, embedding).
- Files: `mcp-server/server.py`
- Impact: Bug fixes or feature changes in crawling/search logic must be applied in two places. The MCP server has its own SentenceTransformer model instance rather than delegating to the RAG service for all operations.
- Fix approach: Remove duplicated logic from `mcp-server/server.py`. The MCP server should delegate all search/crawl/index operations to the RAG service via HTTP (it already does for some tools via `_pocharlies_get`/`_pocharlies_post`). Keep only the MCP protocol wiring and tool definitions in the server.

**Global mutable state in rag-service:**
- Issue: At least 15 global variables are set during startup via `global` keyword in the lifespan function: `web_indexer`, `product_indexer`, `catalog_indexer`, `llm_client`, `redis_client`, `shopify_client`, `rag_agent`, etc. Job tracking uses module-level dicts (`crawl_jobs`, `product_sync_jobs`, `agent_tasks`, etc.).
- Files: `rag-service/app.py` (lines 57-91)
- Impact: Untestable without monkeypatching globals. Not safe for multi-worker uvicorn (the default config sets `UVICORN_WORKERS=40`), since in-memory dicts are per-process. Job status queries may return 404 if the request lands on a different worker than the one running the job.
- Fix approach: Replace in-memory job dicts with Redis-backed storage (some Redis infrastructure already exists for agent tasks via `SessionStore`). Use FastAPI dependency injection instead of globals.

**Deprecated asyncio.get_event_loop() usage:**
- Issue: Eight call sites use `asyncio.get_event_loop()` to run sync functions in executors. This is deprecated in Python 3.10+ and will emit DeprecationWarnings.
- Files: `rag-service/catalog_indexer.py` (lines 114, 278), `rag-service/web_indexer.py` (lines 1394, 1440), `rag-service/translator.py` (line 472), `rag-service/product_indexer.py` (lines 157, 236, 304)
- Impact: Logs get polluted with DeprecationWarnings. Future Python versions may remove this API.
- Fix approach: Replace with `asyncio.get_running_loop()` everywhere, or use `asyncio.to_thread()` (Python 3.9+).

**Duplicate return statement in chat endpoint:**
- Issue: The `/chat` endpoint in `rag-service/app.py` has two identical return statements at lines 536-540 and 542-546. The second return is dead code.
- Files: `rag-service/app.py` (lines 536-546)
- Impact: No runtime impact (dead code), but indicates sloppy editing and reduces readability.
- Fix approach: Remove the second return statement (lines 542-546).

**Webhook error response not using HTTPException:**
- Issue: When the webhook handler is not configured, the endpoint returns `{"error": "..."}, 503` as a tuple. FastAPI does not interpret this as an HTTP 503 response -- it will return 200 with the tuple serialized.
- Files: `rag-service/app.py` (line 1327)
- Impact: Shopify will receive a 200 OK when the webhook handler is not configured, masking the error.
- Fix approach: Replace `return {"error": "Webhook handler not configured"}, 503` with `raise HTTPException(status_code=503, detail="Webhook handler not configured")`.

## Known Bugs

**Webhook handler returns 200 instead of 503:**
- Symptoms: When `SHOPIFY_WEBHOOK_SECRET` is not set, the webhook endpoint at `/webhooks/shopify` returns HTTP 200 with a tuple body instead of HTTP 503.
- Files: `rag-service/app.py` (line 1327)
- Trigger: Call `/webhooks/shopify` when webhook handler is not initialized.
- Workaround: Ensure `SHOPIFY_WEBHOOK_SECRET` is always set.

**Dead code in /chat endpoint:**
- Symptoms: Second return statement is unreachable. Also, the first return has inconsistent indentation (`"model": model` is indented differently).
- Files: `rag-service/app.py` (lines 536-546)
- Trigger: N/A -- dead code, no runtime effect.
- Workaround: None needed; does not affect behavior.

## Security Considerations

**No authentication on any HTTP API:**
- Risk: All API endpoints across all three services (rag-service port 5000, mcp-server port 8002, agent-service port 8100) have no authentication middleware. Anyone who can reach these ports can invoke crawls, search products, access order data (including customer PII: names, emails, addresses, phone numbers), send agent commands, and delete indexed content.
- Files: `rag-service/app.py` (all endpoints), `agent-service/main.py` (all endpoints), `mcp-server/server.py` (all MCP tools)
- Current mitigation: All services bind to `127.0.0.1` in `docker-compose.yml` (lines 36, 90, 122), so they are only accessible from the host machine. External access presumably goes through an nginx reverse proxy.
- Recommendations: Add API key or Bearer token authentication middleware to at least the rag-service and agent-service. The order endpoints expose customer PII (names, emails, addresses, phone numbers) and should have authentication even for internal use. Consider adding rate limiting beyond the existing 3-concurrent-agent-tasks limit.

**CORS allows all origins:**
- Risk: `rag-service/app.py` sets `allow_origins=["*"]` with `allow_credentials=True`. This allows any website to make authenticated cross-origin requests.
- Files: `rag-service/app.py` (lines 224-230)
- Current mitigation: Services are bound to localhost only.
- Recommendations: Restrict `allow_origins` to the specific domains that need access (e.g., the agent-service dashboard domain).

**Default PostgreSQL password:**
- Risk: `docker-compose.yml` sets `POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-agent-secret}`. If the env var is not set, the password defaults to the hardcoded string `agent-secret`.
- Files: `docker-compose.yml` (line 105)
- Current mitigation: PostgreSQL is not exposed to the host (no `ports:` mapping), only accessible within the Docker network.
- Recommendations: Remove the default value so deployment fails fast if the password is not configured.

**Exception messages exposed to clients:**
- Risk: Multiple endpoints pass raw `str(e)` into HTTP 500 error responses (e.g., `raise HTTPException(status_code=500, detail=str(e))`). This can leak internal paths, database errors, or stack traces.
- Files: `rag-service/app.py` (lines 548-549, 898-899, 917-918), `agent-service/api/health.py` (multiple), `agent-service/api/chat.py` (line 55)
- Current mitigation: None.
- Recommendations: Log the full exception server-side and return a generic error message to clients.

**Order/customer PII accessible without auth:**
- Risk: The `/orders`, `/orders/{id}`, and `/orders/{id}/fulfillments` endpoints in `rag-service/app.py` return customer names, emails, phone numbers, and shipping addresses. The MCP server also exposes `search_orders` and `get_order` tools.
- Files: `rag-service/app.py` (lines 1880-1943), `mcp-server/server.py` (order tools), `rag-service/shopify_graphql.py` (lines 536-610)
- Current mitigation: Services bound to localhost only.
- Recommendations: Add authentication to order endpoints. Consider masking email/phone in non-admin contexts.

## Performance Bottlenecks

**Multi-worker uvicorn with in-memory state:**
- Problem: The rag-service Dockerfile configures `UVICORN_WORKERS=${UVICORN_WORKERS:-4}` (docker-compose default is 40). Each worker loads its own ~440MB embedding model, ~568MB reranker model, and BM25 model. With 40 workers, this requires approximately 40GB+ of RAM just for model duplicates.
- Files: `rag-service/Dockerfile` (line 21), `docker-compose.yml` (line 50, `UVICORN_WORKERS:-40`)
- Cause: Each uvicorn worker forks the process, duplicating all in-memory objects including ML models and job tracking dicts.
- Improvement path: Reduce `UVICORN_WORKERS` to 1-4 (embedding/reranking are CPU-bound, not I/O-bound). Use `--workers 1` with async concurrency for the event loop. If multi-worker is truly needed, use a shared model server or move models to a separate service.

**Synchronous embedding in async endpoints:**
- Problem: All embedding operations (SentenceTransformer.encode) are synchronous and run in the default thread pool via `run_in_executor(None, ...)`. The default executor has limited threads, which can become a bottleneck under load.
- Files: `rag-service/product_indexer.py` (lines 305-316), `rag-service/catalog_indexer.py` (lines 115-119, 279-283), `rag-service/web_indexer.py` (lines 1394, 1440)
- Cause: `SentenceTransformer.encode()` is CPU-bound and holds the GIL during numpy operations.
- Improvement path: Use a dedicated `ThreadPoolExecutor` with a bounded size (e.g., 2-4 threads) so embedding requests don't starve other I/O tasks. Consider moving to an external embedding service for horizontal scaling.

**MCP server loads its own SentenceTransformer:**
- Problem: The MCP server loads an independent copy of the SentenceTransformer embedding model (~440MB), even though many of its tools just proxy to the RAG service.
- Files: `mcp-server/server.py` (lines 52, embedded model loading)
- Cause: The MCP server directly uses Qdrant for search instead of delegating all queries to the RAG service.
- Improvement path: Route all search/embed operations through the RAG service HTTP API. Remove the embedded model from the MCP server entirely.

## Fragile Areas

**rag-service/app.py global state initialization:**
- Files: `rag-service/app.py` (lines 57-91, 94-214)
- Why fragile: The lifespan function initializes ~15 globals sequentially. If any component (e.g., Redis, Shopify) fails to initialize, downstream components that depend on it (e.g., glossary_store, webhook_handler) may be initialized with None dependencies. The failure mode is a warning log, not a crash, so the service runs in a degraded state that is hard to detect.
- Safe modification: When adding new dependencies, add them after their prerequisites and include a null check. Test with Redis/Shopify/LLM unavailable to verify graceful degradation.
- Test coverage: None -- no test files exist in the project.

**Webhook processing fire-and-forget:**
- Files: `rag-service/app.py` (lines 1345-1351), `rag-service/webhook_handler.py`
- Why fragile: Webhook processing is dispatched via `asyncio.create_task(_process())` with no error propagation, retry logic, or persistence. If the task fails (e.g., Qdrant is temporarily down), the product update is silently lost.
- Safe modification: Add a webhook event queue (Redis list) with retry logic. Log failed webhook events to a dead-letter list for manual replay.
- Test coverage: None.

**MCP Manager lifecycle (context manager abuse):**
- Files: `agent-service/mcp_client/manager.py` (lines 54-71, 156-166)
- Why fragile: `MCPServerConnection.connect()` manually calls `__aenter__()` on context managers and stores them, with `disconnect()` calling `__aexit__()`. If connection fails partway through (e.g., after `_cm.__aenter__()` succeeds but `session.initialize()` fails), the SSE connection may leak.
- Safe modification: Use `contextlib.AsyncExitStack` to manage nested context managers properly.
- Test coverage: None.

**Agent supervisor recursion limit:**
- Files: `agent-service/graphs/supervisor.py` (line 89)
- Why fragile: `graph.recursion_limit = 16` caps the total graph steps (supervisor + tools round-trips). With complex multi-step tasks, the agent may hit this limit and raise a `RecursionError` that propagates as an unhandled exception to the client.
- Safe modification: Add explicit error handling for `RecursionError` in the chat endpoint. Consider making the limit configurable.
- Test coverage: None.

## Scaling Limits

**In-memory job tracking dicts:**
- Current capacity: Job dicts (`crawl_jobs`, `product_sync_jobs`, `translation_jobs`, `agent_tasks`) grow without bound. A cleanup function (`_cleanup_old_agent_tasks`) trims to 100 entries, but only for agent tasks.
- Limit: With 40 uvicorn workers, each worker has isolated dicts. A status poll for a job may hit a different worker than the one running the job, returning 404.
- Scaling path: Move all job tracking to Redis (the `SessionStore` pattern already exists for agent tasks). Add TTL-based expiry.

**Qdrant as the only search backend:**
- Current capacity: Qdrant is a single instance with persistent volume. Point counts are in the low thousands.
- Limit: Single-instance Qdrant becomes a bottleneck with hundreds of thousands of products or heavy concurrent search load.
- Scaling path: Qdrant supports clustering. Monitor memory usage and add sharding when collection sizes grow.

**Redis without persistence configuration:**
- Current capacity: Redis is configured with `appendonly yes` and 512MB max memory with LRU eviction.
- Limit: At 512MB, glossary entries, content hashes, sync state, and session data will start getting evicted under memory pressure. Loss of content hashes forces re-embedding of all unchanged products on next sync.
- Scaling path: Increase `maxmemory`. Consider separating cache data (can be evicted) from stateful data (sync cursors, glossary entries) into different Redis databases or instances.

## Dependencies at Risk

**OpenAI Agents SDK (`agents` package):**
- Risk: The `agents` package used in `rag-service/agent/__init__.py` and `rag-service/agent/tools.py` is the OpenAI Agents SDK, which is relatively new and may have breaking API changes.
- Impact: Agent tool definitions and the agent factory would need rewriting if the SDK API changes.
- Migration plan: The agent-service uses LangGraph instead, which is more mature. Consider converging on a single agent framework.

**Two different agent frameworks in use:**
- Risk: The project uses both the OpenAI Agents SDK (in `rag-service/agent/`) and LangGraph (in `agent-service/graphs/`). These serve overlapping purposes.
- Impact: Maintenance burden of keeping two agent implementations in sync. Developers must understand two different abstractions.
- Migration plan: The `agent-service` with LangGraph appears to be the newer, more capable implementation. Consider deprecating the `rag-service/agent/` SDK agent in favor of routing all agent tasks through the `agent-service`.

## Missing Critical Features

**No test suite:**
- Problem: The entire project has zero test files. No unit tests, no integration tests, no end-to-end tests.
- Blocks: Safe refactoring, confident deployments, automated CI/CD pipelines.

**No database migrations:**
- Problem: The agent-service creates database tables via `Base.metadata.create_all` at startup (`agent-service/main.py` line 30). There is no migration tool (like Alembic) for schema changes.
- Blocks: Schema changes require dropping and recreating tables, losing all task history data.

**No request validation for sensitive operations:**
- Problem: Destructive operations like `/delete/{collection}`, `/products/delete-all`, and agent task creation lack confirmation mechanisms or audit logging.
- Blocks: Accidental data loss from API misuse.

## Test Coverage Gaps

**Entire codebase is untested:**
- What's not tested: All functionality -- web crawling, product indexing, search, translation, webhooks, agent orchestration, MCP tools, chat, orders.
- Files: Every `.py` file in the project.
- Risk: Any refactoring or feature addition could break existing functionality without detection. The identified bugs (duplicate return, webhook 200-instead-of-503) would have been caught by even basic tests.
- Priority: High -- start with the most fragile areas: `rag-service/webhook_handler.py`, `rag-service/shopify_graphql.py` (data flattening), `rag-service/translator.py` (parse_translations), `agent-service/mcp_client/manager.py`.

---

*Concerns audit: 2026-03-09*
