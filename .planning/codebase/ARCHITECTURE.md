# Architecture

**Analysis Date:** 2026-03-09

## Pattern Overview

**Overall:** Multi-service microservices architecture with three Python services (rag-service, mcp-server, agent-service) orchestrated via Docker Compose. Each service runs independently as a Docker container. Inter-service communication uses HTTP REST and MCP SSE protocol.

**Key Characteristics:**
- Three-tier agent architecture: agent-service (LangGraph orchestrator) -> mcp-server (MCP tool bridge) -> rag-service (domain logic and data)
- Hybrid vector search (dense + sparse) via Qdrant for all retrieval operations
- Two independent agent systems: LangGraph-based supervisor (agent-service) and OpenAI Agents SDK agent (rag-service)
- Event-driven Shopify webhook processing for real-time catalog sync
- Redis for session persistence, content hash dedup, sync state, and glossary storage
- PostgreSQL for LangGraph checkpointing and task persistence (agent-service only)

## Layers

**Agent Orchestration Layer (agent-service):**
- Purpose: LangGraph-based supervisor agent that receives user instructions and delegates to tools
- Location: `agent-service/`
- Contains: FastAPI app, LangGraph supervisor graph, MCP client manager, task/health APIs
- Depends on: MCP server (via SSE), PostgreSQL, Redis, LLM (via OpenAI-compatible API)
- Used by: External clients (web UI, API consumers)

**MCP Tool Bridge (mcp-server):**
- Purpose: Exposes RAG service capabilities as MCP tools for the agent-service to consume
- Location: `mcp-server/`
- Contains: Single FastMCP server with 25+ tools covering search, crawl, products, orders, translation
- Depends on: Qdrant (direct), RAG service (HTTP proxy for some operations), embedding model
- Used by: agent-service (via MCP SSE protocol)

**RAG & Domain Logic Layer (rag-service):**
- Purpose: Core business logic -- web crawling, product indexing, search, translation, Shopify integration, competitor analysis
- Location: `rag-service/`
- Contains: FastAPI app with 60+ REST endpoints, embedded OpenAI Agents SDK agent, all indexers and pipelines
- Depends on: Qdrant, Redis, Shopify Admin API, LLM (via LiteLLM), embedding model
- Used by: MCP server (HTTP calls), direct web UI consumers, CLI users

**Data Storage Layer:**
- Purpose: Persistent storage for vectors, state, and caching
- Location: Docker volumes (managed via `docker-compose.yml`)
- Contains: Qdrant (vector DB), PostgreSQL (relational), Redis (cache/sessions)
- Depends on: Nothing (infrastructure services)
- Used by: All application services

## Data Flow

**User Chat via Agent Service:**

1. User sends message to `POST /api/chat` or WebSocket `/api/chat/ws` on agent-service (port 8100)
2. `agent-service/api/chat.py` invokes the LangGraph supervisor graph with the message
3. Supervisor graph (`agent-service/graphs/supervisor.py`) calls ChatOpenAI with bound MCP tools
4. LLM decides which tools to call; ToolNode executes them via MCP client
5. MCP client (`agent-service/mcp_client/manager.py`) sends tool call over SSE to mcp-server (port 8002)
6. MCP server (`mcp-server/server.py`) executes the tool -- either directly against Qdrant or by proxying to rag-service (port 5000)
7. Results flow back through the chain: MCP -> ToolNode -> supervisor -> user

**Product Catalog Sync:**

1. Request hits `POST /products/sync` or `POST /catalog/full-sync` on rag-service
2. `rag-service/app.py` starts background task with `ProductIndexer` or `CatalogIndexer`
3. `shopify_client.py` fetches products via Shopify REST API (paginated, rate-limited)
4. Alternatively, `shopify_graphql.py` uses Shopify bulk operations for large exports
5. Text extraction: `ShopifyClient.extract_product_text()` + `parse_airsoft_specs()` for domain-specific metadata
6. Hybrid embedding: dense vectors (SentenceTransformer BGE) + sparse vectors (BM25 via fastembed)
7. Content hash dedup via `sync_state.ContentHashStore` (Redis-backed SHA-256)
8. Points upserted to Qdrant `product_catalog`, `product_collections`, or `product_pages` collections

**Shopify Webhook Real-Time Sync:**

1. Shopify sends webhook to `POST /webhooks/shopify` on rag-service
2. `webhook_handler.py` verifies HMAC-SHA256 signature
3. Routes by topic: products/create, products/update, products/delete, collections/*
4. Fetches full entity via GraphQL (`shopify_graphql.py`) for creates/updates
5. Re-embeds and upserts single item to Qdrant (bypasses batch pipeline)

**Hybrid Search Flow:**

1. Query arrives at any search endpoint (rag-service, mcp-server, or via agent)
2. Query is prefixed with BGE instruction prefix for dense encoding
3. Dense embedding via SentenceTransformer, sparse embedding via BM25
4. Qdrant hybrid query: two Prefetch branches (dense + sparse) fused with Reciprocal Rank Fusion (RRF)
5. Optional reranking via CrossEncoder (`reranker.py`) if enabled
6. Results returned with scores, metadata, and text snippets

**Translation Pipeline:**

1. Request to `POST /translate/batch` with texts, source_lang, target_lang
2. `translator.py` does token-aware bin packing into LLM-sized batches
3. Glossary matching: built-in 20-language glossary (`glossary_data.py`) + custom Redis-backed terms
4. LLM translation with glossary-aware system prompt
5. Adaptive retry: bisects failed batches recursively up to 3 levels deep
6. JSON array parsing of LLM response with fallback to numbered-marker parsing

**State Management:**

- **LangGraph conversation state**: PostgreSQL via `AsyncPostgresSaver` (agent-service only)
- **Agent task state**: Redis sorted sets + hashes (`agent:task:{id}`, `agent:tasks` index)
- **Agent session history**: Redis lists (`agent:session:{id}`) via `RedisSession`
- **Sync state**: Redis hashes (`catalog:sync:{id}`) via `SyncStateStore`
- **Content dedup hashes**: Redis strings (`catalog:hash:{item_key}`) via `ContentHashStore`
- **Glossary terms**: Redis hashes (`translation:glossary:{src}:{tgt}`) via `GlossaryStore`
- **Crawl jobs / sync jobs**: In-memory dicts in rag-service (not persistent across restarts)

## Key Abstractions

**Indexers:**
- Purpose: Embed content and upsert to Qdrant collections with hybrid vectors
- Examples: `rag-service/web_indexer.py`, `rag-service/product_indexer.py`, `rag-service/catalog_indexer.py`
- Pattern: Each indexer owns a Qdrant collection, creates it at init, handles embedding + upserting + searching. All use named vectors (`dense` + `sparse`) with RRF fusion.

**Job Tracking Dataclasses:**
- Purpose: Track progress of long-running background operations
- Examples: `CrawlJob` in `rag-service/web_indexer.py`, `ProductSyncJob` in `rag-service/product_indexer.py`, `ClassificationJob` in `rag-service/product_classifier.py`, `TranslationJob` in `rag-service/translator.py`, `AgentTask` in `rag-service/agent/__init__.py`
- Pattern: Dataclass with `job_id`, `status` (running/completed/failed), `logs` list with max cap, `to_dict()` method. Stored in in-memory dicts keyed by job_id, polled via GET endpoints.

**MCP Tool Wrapping:**
- Purpose: Convert MCP tool schemas into LangChain `StructuredTool` objects for LangGraph
- Examples: `agent-service/mcp_client/manager.py` (`MCPServerConnection._wrap_tool`)
- Pattern: JSON Schema from MCP `list_tools` -> dynamic Pydantic model via `create_model` -> `StructuredTool.from_function` with async coroutine that calls `session.call_tool`

**Shopify Client Hierarchy:**
- Purpose: Interface with Shopify Admin API (REST + GraphQL)
- Examples: `rag-service/shopify_client.py` (REST), `rag-service/shopify_graphql.py` (GraphQL + bulk ops)
- Pattern: `ShopifyClient` uses REST for simple CRUD, lazy-initializes `ShopifyGraphQL` for bulk operations. Both provide `flatten_graphql_*` methods to normalize GraphQL response shapes to REST-like dicts.

## Entry Points

**rag-service FastAPI (`rag-service/app.py`):**
- Location: `rag-service/app.py`
- Triggers: HTTP requests on port 5000 (internal to Docker network, mapped to `127.0.0.1:5000` on host)
- Responsibilities: 60+ REST endpoints for web crawling, product sync, search, translation, agent tasks, Shopify webhooks, order management, glossary management. Serves embedded dashboard UI.

**agent-service FastAPI (`agent-service/main.py`):**
- Location: `agent-service/main.py`
- Triggers: HTTP requests on port 8100 (mapped to `127.0.0.1:8100` on host)
- Responsibilities: Chat API (REST + WebSocket), task management, MCP server management, health/stats. Serves chat UI.

**mcp-server FastMCP (`mcp-server/server.py`):**
- Location: `mcp-server/server.py`
- Triggers: MCP SSE connections on port 8000 (mapped to `127.0.0.1:8002` on host)
- Responsibilities: Exposes 25+ MCP tools for search, crawl, products, orders, translation. Used by agent-service's MCPManager.

**RAG Agent CLI (`rag-service/agent/__main__.py`):**
- Location: `rag-service/agent/__main__.py` (invoked via `python -m agent "prompt"`)
- Triggers: Command-line execution
- Responsibilities: CLI interface to the OpenAI Agents SDK agent with optional Redis persistence

## Error Handling

**Strategy:** Exception-based with catch-and-log at service boundaries. No custom exception hierarchy.

**Patterns:**
- FastAPI endpoints wrap operations in try/except, return HTTP error responses
- MCP tools return error strings rather than raising (e.g., `return f"Error: {e}"`)
- Job dataclasses capture errors in `errors` list and set `status = "failed"`
- Translation pipeline uses bisect-retry: on LLM failure, splits batch in half and retries each half (up to 3 levels deep), but raises immediately on connectivity errors (502, 503)
- Webhook handler logs errors but does not retry -- relies on Shopify's retry mechanism
- Agent runner catches all exceptions, sets task status to "failed" with error message

## Cross-Cutting Concerns

**Logging:** Python `logging` module throughout. Each module creates its own logger via `logging.getLogger(__name__)` or named loggers like `logging.getLogger("agent.supervisor")`. No structured logging framework.

**Validation:** Pydantic models for API request/response validation in both FastAPI services. MCP tool parameters validated via JSON Schema converted to dynamic Pydantic models.

**Authentication:** No authentication on any service. All services bind to `127.0.0.1` (host-only) via Docker port mappings. Inter-service communication uses Docker network names. Shopify webhook requests are verified via HMAC-SHA256 signature. Qdrant protected by API key. Shopify Admin API uses access token.

**Configuration:** Environment variables throughout. agent-service uses `pydantic_settings.BaseSettings` (`agent-service/config.py`). rag-service and mcp-server use raw `os.getenv()` calls. All config passed via `docker-compose.yml` environment blocks. `.env` file present for secrets (git-ignored).

---

*Architecture analysis: 2026-03-09*
