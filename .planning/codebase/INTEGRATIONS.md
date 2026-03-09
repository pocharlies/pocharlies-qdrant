# External Integrations

**Analysis Date:** 2026-03-09

## APIs & External Services

**Shopify Admin API:**
- Purpose: Product catalog management, order lookup, collection/page sync
- REST client: `rag-service/shopify_client.py` (API version 2025-01)
- GraphQL client: `rag-service/shopify_graphql.py` (API version 2025-01)
- Auth: `SHOPIFY_ACCESS_TOKEN` (X-Shopify-Access-Token header)
- Shop domain: `SHOPIFY_SHOP_DOMAIN` (e.g., `skirmshop-es.myshopify.com`)
- Capabilities:
  - Product CRUD + pagination via REST and GraphQL
  - Bulk operations (async export of products, collections, pages)
  - Order queries (list, search, detail, fulfillments, refunds)
  - Collection and page management
  - Incremental sync via `updated_at` queries
- Rate limiting: Shopify leaky bucket, handled in both clients (0.5s sleep for REST, cost-based throttle for GraphQL)

**LLM Service (via LiteLLM/vLLM):**
- Purpose: AI agent reasoning, product classification, translation, smart crawl analysis
- Client: OpenAI Python SDK (`openai` package) pointed at self-hosted endpoint
- Auth: `LLM_API_KEY` (default: "none" for local deployments)
- Base URL: `LLM_BASE_URL` (default: `http://host.docker.internal:8000/v1`)
- Used in:
  - `rag-service/app.py` - OpenAI client for RAG agent, classifier, translator
  - `agent-service/graphs/supervisor.py` - ChatOpenAI (LangChain) for supervisor agent
  - `rag-service/agent/__init__.py` - OpenAI Agents SDK model
- Model selection: `LLM_MODEL` env var (default: `qwen3next-80b` in agent-service)

**HuggingFace Models (local inference):**
- Dense embeddings: `BAAI/bge-base-en-v1.5` via sentence-transformers
  - Loaded once at startup, shared across indexers (`rag-service/app.py` line 105)
  - ~440MB model cached at `HF_CACHE` path
- Sparse embeddings: `Qdrant/bm25` via fastembed (`rag-service/sparse_encoder.py`)
- Reranker: `BAAI/bge-reranker-v2-m3` via CrossEncoder (`rag-service/reranker.py`)
  - Toggled by `RERANKER_ENABLED` env var

**DuckDuckGo (web search):**
- Purpose: Internet search to discover URLs for crawling
- Implementation: HTML scraping of `https://html.duckduckgo.com/html/`
- Client: httpx (`rag-service/agent/tools.py`, `web_search` function)
- No API key required (scrapes HTML results page)

**MCP Protocol (Model Context Protocol):**
- Purpose: Dynamic tool discovery between agent-service and mcp-server
- Server: `mcp-server/server.py` - FastMCP server exposing RAG tools via SSE
- Client: `agent-service/mcp_client/manager.py` - Connects to MCP servers, wraps tools for LangGraph
- Config: `agent-service/mcp_client/mcp_servers.json`
- Default connection: `pocharlies-rag` SSE server at `http://pocharlies-mcp:8000/sse`
- Supports hot-reload: add/remove MCP servers at runtime via `/api/mcp/servers` API

## Data Storage

**Databases:**

**Qdrant Vector Database:**
- Connection: `QDRANT_URL` (default: `http://qdrant:6333`) + `QDRANT_API_KEY`
- Client: `qdrant-client` Python SDK via `rag-service/qdrant_utils.py`
- Collections:
  - Default web content collection (configurable name)
  - `competitor_products` - Competitor website content
  - Product catalog collection (via `rag-service/product_indexer.py`)
  - `product_collections` - Shopify collections
  - `product_pages` - Shopify pages
- Vector types: Hybrid search (dense BGE + sparse BM25 + Reciprocal Rank Fusion)
- Shared volume: `skirmshopshopifyapp_qdrant_data` (external, shared with main skirmshop app)

**PostgreSQL 16:**
- Connection: `DATABASE_URL` = `postgresql+asyncpg://agent:{password}@pocharlies-postgres:5432/langgraph`
- ORM: SQLAlchemy async (`agent-service/state/database.py`)
- Schema defined in `agent-service/state/models.py`:
  - `tasks` table: UUID id, thread_id, trigger, status, prompt, summary, tools_used (JSONB), timestamps
  - `task_logs` table: auto-increment id, task_id FK, level, message, data (JSONB), created_at
- LangGraph checkpointer tables (auto-created by `langgraph-checkpoint-postgres`)

**Redis 7:**
- Connection: `REDIS_URL`
  - DB 0: rag-service (`redis://pocharlies-redis:6379/0`)
  - DB 1: agent-service (`redis://pocharlies-redis:6379/1`)
- Client: `redis[hiredis]` async (`redis.asyncio.Redis`)
- Data stored:
  - `agent:session:{id}` - OpenAI Agents SDK conversation history (7-day TTL) (`rag-service/agent/redis_session.py`)
  - `agent:task:{id}` - Agent task metadata, steps, logs, message queues (`rag-service/agent/session_store.py`)
  - `agent:tasks` - Sorted set index of tasks by timestamp
  - `catalog:sync:{id}` - Sync operation records (7-day TTL) (`rag-service/sync_state.py`)
  - `catalog:sync:history` - Sorted set of last 50 sync IDs
  - `catalog:hash:{key}` - Content SHA-256 hashes for dedup (`rag-service/sync_state.py`)
  - Custom glossary terms (via `GlossaryStore`)

**File Storage:**
- Local filesystem only (Docker volumes)
- HuggingFace model cache: `${HF_CACHE:-~/.cache/huggingface}` mounted into containers

**Caching:**
- Redis for session/state persistence (see above)
- Content hash dedup prevents re-embedding unchanged Shopify items (`rag-service/sync_state.py`)
- Embedding model singleton loaded once at startup, shared across indexers
- BM25 sparse encoder singleton (`rag-service/sparse_encoder.py`)
- Reranker singleton (`rag-service/reranker.py`)

## Authentication & Identity

**Auth Provider:**
- None (no user authentication on the services themselves)
- Shopify webhook HMAC-SHA256 verification (`rag-service/webhook_handler.py`)
- Qdrant API key authentication (`QDRANT_API_KEY`)
- Services are internal-only (bound to `127.0.0.1` in docker-compose)
- Nginx reverse proxy handles TLS termination; no auth layer at proxy

## Monitoring & Observability

**Error Tracking:**
- None (no Sentry, Datadog, etc.)

**Logs:**
- Python `logging` module to stdout/stderr
- Format: `%(asctime)s %(name)s %(levelname)s %(message)s` (agent-service)
- Default level: INFO
- Nginx access/error logs at `/var/log/nginx/{service}-access.log`
- Docker container logs via `docker logs`

**Health Checks:**
- rag-service: `GET /health` (HTTP endpoint + Docker healthcheck via curl)
  - Docker: `curl -f --max-time 25 http://localhost:5000/health` every 60s
- agent-service: `GET /api/health` (`agent-service/api/health.py`)
- PostgreSQL: `pg_isready -U agent -d langgraph` every 10s

## CI/CD & Deployment

**Hosting:**
- Self-hosted Linux server (Ubuntu, likely ARM/aarch64 based on User-Agent headers)
- Docker Compose on bare metal

**CI Pipeline:**
- None detected (no GitHub Actions, no CI config files)

**Deployment Process:**
- Manual `docker-compose build && docker-compose up -d` (inferred)
- devops-docs directory exists for operational documentation

**TLS/SSL:**
- Let's Encrypt certificates managed by Certbot
- Nginx handles TLS termination for all services
- Domains: `rag.e-dani.com`, `agent.e-dani.com`, `qdrant.e-dani.com`

## Environment Configuration

**Required env vars:**
- `QDRANT_API_KEY` - Vector database authentication (required, enforced in docker-compose)
- `SHOPIFY_SHOP_DOMAIN` - Shopify store domain (optional, disables Shopify features if unset)
- `SHOPIFY_ACCESS_TOKEN` - Shopify Admin API token (optional, paired with domain)

**Optional env vars with defaults:**
- `LLM_BASE_URL` (default: `http://host.docker.internal:8000/v1`)
- `LLM_API_KEY` (default: `none`)
- `LLM_MODEL` (default: `qwen3next-80b` in agent-service)
- `EMBEDDING_MODEL` (default: `BAAI/bge-base-en-v1.5`)
- `RERANKER_MODEL` (default: `BAAI/bge-reranker-v2-m3`)
- `RERANKER_ENABLED` (default: `true`)
- `POSTGRES_PASSWORD` (default: `agent-secret`)
- `UVICORN_WORKERS` (default: `40` for rag-service, `1` for agent-service)
- `UVICORN_MAX_CONCURRENT` (default: `500`)
- `TRANSLATE_MAX_CHUNKS` / `TRANSLATE_MAX_INPUT_TOKENS` - Translation pipeline limits
- `TRANSLATE_MODEL` - Override model for translation
- `SHOPIFY_WEBHOOK_SECRET` - HMAC secret for webhook verification

**Secrets location:**
- `.env` file at project root (not committed to git)
- `.env.example` documents structure (committed)

## Webhooks & Callbacks

**Incoming:**
- `POST /webhooks/shopify` (`rag-service/app.py` line 1323)
  - HMAC-SHA256 verified via `X-Shopify-Hmac-SHA256` header
  - Topics handled: `products/create`, `products/update`, `products/delete`, `collections/create`, `collections/update`, `collections/delete`, `inventory_levels/update`
  - Responds immediately with 200, processes async in background task
  - Handler: `rag-service/webhook_handler.py` - fetches full item via GraphQL, re-embeds, upserts to Qdrant

**Outgoing:**
- None (no outbound webhook calls to external services)

## Inter-Service Communication

**Service topology:**
```
agent-service (port 8100)
    -> mcp-server (SSE at port 8000, mapped to 8002)
        -> rag-service (HTTP at port 5000) via POCHARLIES_RAG_URL
    -> postgres (port 5432) via DATABASE_URL
    -> redis DB 1 via REDIS_URL
    -> LLM endpoint via LLM_BASE_URL (host.docker.internal:8000)

rag-service (port 5000)
    -> qdrant (ports 6333/6334) via QDRANT_URL
    -> redis DB 0 via REDIS_URL
    -> Shopify Admin API (external HTTPS)
    -> LLM endpoint via VLLM_BASE_URL (host.docker.internal:8000)
    -> DuckDuckGo (external HTTPS, web search)

mcp-server (port 8000)
    -> qdrant (ports 6333/6334) via QDRANT_URL
    -> rag-service (port 5000) via POCHARLIES_RAG_URL
```

**Network:** All services on `skirmshop-network` (external Docker network shared with other apps)

**DNS:** Containers use Cloudflare (1.1.1.1) and Google (8.8.8.8) DNS (rag-service only)

---

*Integration audit: 2026-03-09*
