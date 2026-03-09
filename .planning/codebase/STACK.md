# Technology Stack

**Analysis Date:** 2026-03-09

## Languages

**Primary:**
- Python 3.11 - All three services (rag-service, agent-service, mcp-server)

**Secondary:**
- HTML/CSS/JS - Static UIs served by FastAPI (`rag-service/static/`, `agent-service/static/`)
- SQL - PostgreSQL schema via SQLAlchemy ORM (`agent-service/state/models.py`)
- GraphQL - Shopify Admin API queries (`rag-service/shopify_graphql.py`)

## Runtime

**Environment:**
- Python 3.11-slim (Docker base image for all services)
- Docker + Docker Compose for orchestration

**Package Manager:**
- pip (requirements.txt per service, no lock files)
- Lockfile: missing (no `requirements.lock` or `pip-compile` output)

## Frameworks

**Core:**
- FastAPI >=0.115.0 (agent-service) / 0.109.0 (rag-service) - HTTP/WebSocket API layer
- LangGraph >=0.4.0 - Agent orchestration graph (agent-service)
- LangChain (langchain-openai >=0.3.0, langchain-core >=0.3.0) - LLM integration for agent-service
- OpenAI Agents SDK (openai-agents >=0.7.0) - Agent framework in rag-service
- MCP (Model Context Protocol) >=1.0.0 - Tool discovery/execution protocol

**ML/AI:**
- sentence-transformers 2.3.1 - Dense embeddings (BAAI/bge-base-en-v1.5)
- fastembed >=0.4.0 - BM25 sparse embeddings (Qdrant/bm25)
- CrossEncoder (sentence-transformers) - Reranking (BAAI/bge-reranker-v2-m3)

**Testing:**
- Not detected (no test framework, no test files)

**Build/Dev:**
- Docker Compose - Multi-container orchestration (`docker-compose.yml`)
- Uvicorn >=0.27.0 - ASGI server for all Python services
- Nginx - Reverse proxy with TLS (Let's Encrypt/Certbot)

## Key Dependencies

**Critical:**
- `qdrant-client` >=1.12.0,<2.0.0 - Vector database client (core data store)
- `openai` >=2.9.0,<3 - OpenAI-compatible LLM client (used with LiteLLM proxy)
- `langchain-openai` >=0.3.0 - LangChain LLM wrapper for agent-service supervisor
- `langgraph` >=0.4.0 - State graph for agent orchestration
- `langgraph-checkpoint-postgres` >=2.0.0 - Durable agent state checkpointing
- `mcp` >=1.0.0 - Model Context Protocol client/server SDK
- `sentence-transformers` 2.3.1 - Embedding model loading and inference
- `redis[hiredis]` >=5.0.0 - Caching, session persistence, sync state, content dedup

**Infrastructure:**
- `sqlalchemy[asyncio]` >=2.0.0 - Async ORM for PostgreSQL (agent-service)
- `asyncpg` >=0.30.0 - PostgreSQL async driver
- `psycopg[binary]` >=3.2.0 - PostgreSQL driver for LangGraph checkpointer
- `httpx` >=0.26.0 - Async HTTP client (Shopify API, web crawling, inter-service calls)
- `pydantic` >=2.5.0 - Data validation (all services)
- `pydantic-settings` >=2.0.0 - Environment-based config (`agent-service/config.py`)

**Content Processing:**
- `beautifulsoup4` 4.12.3 - HTML parsing (crawled pages, Shopify product descriptions)
- `trafilatura` 1.8.1 - Web content extraction (main text from HTML)
- `curl_cffi` >=0.7.0 - Cloudflare bypass HTTP client (optional, for protected sites)
- `PyMuPDF` >=1.24.0 - PDF text extraction
- `langdetect` >=1.0.9 - Language detection for content
- `gitpython` 3.1.41 - Git repo crawling/indexing

**Scheduling:**
- `apscheduler` >=3.10.0,<4 - Task scheduling in rag-service
- `apscheduler` >=4.0.0a5 - Scheduler in agent-service (alpha version, listed but not yet wired)

**CLI:**
- `typer` >=0.15.0 - CLI interface for rag-service agent (`rag-service/agent/cli.py`)

## Configuration

**Environment:**
- `.env` file at project root (loaded by Docker Compose and pydantic-settings)
- `.env.example` documents all required variables
- Environment variables passed through `docker-compose.yml` with defaults
- agent-service uses `pydantic-settings` (`agent-service/config.py`) for typed config
- rag-service uses raw `os.getenv()` calls in `app.py`

**Key env vars (from `.env.example`):**
- `QDRANT_URL` / `QDRANT_API_KEY` - Vector database connection
- `LLM_BASE_URL` / `LLM_API_KEY` - LLM endpoint (LiteLLM proxy)
- `SHOPIFY_SHOP_DOMAIN` / `SHOPIFY_ACCESS_TOKEN` - Shopify Admin API
- `EMBEDDING_MODEL` - HuggingFace model name (default: BAAI/bge-base-en-v1.5)
- `RERANKER_MODEL` / `RERANKER_ENABLED` - CrossEncoder reranking config
- `REDIS_URL` - Redis connection string
- `SHOPIFY_WEBHOOK_SECRET` - HMAC verification for Shopify webhooks
- `POSTGRES_PASSWORD` - PostgreSQL password for agent state DB

**Build:**
- `docker-compose.yml` - Defines all 5 services + volumes + network
- `rag-service/Dockerfile` - Python 3.11-slim + git, ripgrep, curl
- `agent-service/Dockerfile` - Python 3.11-slim + curl
- `mcp-server/Dockerfile` - Python 3.11-slim + curl

## Data Stores

**Qdrant (Vector DB):**
- Image: `qdrant/qdrant:latest`
- Ports: 6333 (REST), 6334 (gRPC)
- Volume: `qdrant_data` (external, shared with skirmshop app)
- Collections: web content, products, competitor products, collections, pages

**PostgreSQL 16:**
- Image: `postgres:16-alpine`
- Database: `langgraph` (user: `agent`)
- Purpose: LangGraph checkpoint persistence + task/log storage
- Tables: `tasks`, `task_logs` (via SQLAlchemy), plus LangGraph internal tables

**Redis 7:**
- Image: `redis:7-alpine`
- DB 0: rag-service (sessions, glossary, sync state, content hashes)
- DB 1: agent-service (general cache)
- Config: AOF persistence, 512MB max, allkeys-lru eviction
- Volume: `redis_data`

## Platform Requirements

**Development:**
- Docker + Docker Compose
- Access to HuggingFace model cache (`~/.cache/huggingface`)
- LLM endpoint (LiteLLM or vLLM) at `LLM_BASE_URL`

**Production:**
- Linux host (Ubuntu, aarch64 architecture based on User-Agent strings)
- Nginx reverse proxy with TLS (Let's Encrypt/Certbot)
- Domains: `rag.e-dani.com` (port 5000), `agent.e-dani.com` (port 8100), `qdrant.e-dani.com` (port 6333)
- Docker network: `skirmshop-network` (external, shared with other skirmshop services)
- GPU recommended for embedding model inference (sentence-transformers)

---

*Stack analysis: 2026-03-09*
