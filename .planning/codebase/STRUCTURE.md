# Codebase Structure

**Analysis Date:** 2026-03-09

## Directory Layout

```
pocharlies-qdrant/
├── agent-service/           # LangGraph agent orchestrator (FastAPI, port 8100)
│   ├── api/                 # FastAPI route modules
│   │   ├── chat.py          # Chat REST + WebSocket endpoints
│   │   ├── health.py        # Health check + dependency status + stats
│   │   ├── mcp_api.py       # MCP server management CRUD endpoints
│   │   └── tasks.py         # Task listing + log retrieval
│   ├── graphs/              # LangGraph graph definitions
│   │   └── supervisor.py    # ReAct supervisor agent with MCP tools
│   ├── mcp_client/          # MCP protocol client
│   │   ├── manager.py       # MCPManager + MCPServerConnection classes
│   │   └── mcp_servers.json # MCP server connection config
│   ├── state/               # Database models + checkpointing
│   │   ├── checkpointer.py  # LangGraph AsyncPostgresSaver setup
│   │   ├── database.py      # SQLAlchemy async engine + session factory
│   │   └── models.py        # Task + TaskLog ORM models
│   ├── static/              # Embedded chat UI
│   │   ├── index.html       # Chat interface HTML
│   │   ├── app.js           # Chat UI JavaScript
│   │   └── style.css        # Chat UI styles
│   ├── config.py            # Pydantic Settings (env vars)
│   ├── main.py              # FastAPI app + lifespan (entry point)
│   └── Dockerfile           # Container build
│
├── mcp-server/              # MCP tool server (FastMCP, SSE on port 8000)
│   ├── server.py            # Single-file MCP server with 25+ tools
│   ├── Dockerfile           # Container build
│   └── .venv/               # Local Python venv (not committed)
│
├── rag-service/             # RAG pipeline service (FastAPI, port 5000)
│   ├── agent/               # OpenAI Agents SDK integration
│   │   ├── __init__.py      # AgentTask, AgentServices, create_agent(), SYSTEM_PROMPT
│   │   ├── __main__.py      # CLI entry point (python -m agent)
│   │   ├── cli.py           # Typer CLI with run + status commands
│   │   ├── redis_session.py # Redis-backed SessionABC for conversation history
│   │   ├── runner.py        # Agent task executor with streaming events
│   │   ├── session_store.py # Redis-backed task metadata store
│   │   └── tools.py         # @function_tool wrappers (crawl, search, etc.)
│   ├── static/              # Embedded dashboard UI
│   │   ├── index.html       # Dashboard HTML
│   │   ├── app.js           # Dashboard JavaScript
│   │   └── style.css        # Dashboard styles
│   ├── app.py               # FastAPI app (1943 lines) — all REST endpoints + lifespan
│   ├── web_indexer.py        # Web crawler + Qdrant indexer (1666 lines)
│   ├── product_indexer.py    # Shopify product indexer with hybrid search
│   ├── catalog_indexer.py    # Collection + page indexer (extends product pipeline)
│   ├── product_classifier.py # LLM-based competitor product extraction + entity resolution
│   ├── shopify_client.py     # Shopify REST Admin API client
│   ├── shopify_graphql.py    # Shopify GraphQL + bulk operations client
│   ├── translator.py         # LLM translation pipeline with glossary
│   ├── glossary_data.py      # Built-in 20-language airsoft terminology glossary
│   ├── reranker.py           # CrossEncoder reranker (singleton)
│   ├── sparse_encoder.py     # BM25 sparse encoder (singleton, fastembed)
│   ├── qdrant_utils.py       # Shared QdrantClient factory
│   ├── sync_state.py         # Redis-backed sync state + content hash dedup
│   ├── webhook_handler.py    # Shopify webhook HMAC verification + routing
│   └── Dockerfile            # Container build
│
├── docs/                    # Documentation directory
│   └── plans/               # Planning documents
├── devops-docs/             # DevOps documentation (empty)
├── repos/                   # External repo references (empty)
├── .planning/               # GSD planning directory
│   └── codebase/            # Codebase analysis documents (this file)
├── .claude/                 # Claude Code configuration
├── docker-compose.yml       # Full stack orchestration (5 services + 3 volumes)
├── .env                     # Environment secrets (git-ignored)
├── .env.example             # Template for required env vars
└── .gitignore               # Git ignore rules
```

## Directory Purposes

**`agent-service/`:**
- Purpose: LangGraph-based agent orchestrator that provides chat interface and task management
- Contains: FastAPI app, LangGraph supervisor graph, MCP client, PostgreSQL models, static chat UI
- Key files: `main.py` (entry point), `graphs/supervisor.py` (agent graph), `mcp_client/manager.py` (MCP connectivity)

**`agent-service/api/`:**
- Purpose: FastAPI router modules, one per API domain
- Contains: Route handlers organized by concern (chat, tasks, MCP management, health)
- Key files: `chat.py` (REST + WebSocket chat), `mcp_api.py` (server CRUD + tool listing)

**`agent-service/graphs/`:**
- Purpose: LangGraph graph definitions
- Contains: Supervisor ReAct agent graph builder
- Key files: `supervisor.py` (the only graph -- supervisor with tool-calling loop)

**`agent-service/mcp_client/`:**
- Purpose: MCP protocol client for connecting to MCP servers
- Contains: Connection manager, JSON config for server URLs
- Key files: `manager.py` (MCPManager class), `mcp_servers.json` (server registry)

**`agent-service/state/`:**
- Purpose: Database layer for persistent task storage and LangGraph checkpointing
- Contains: SQLAlchemy async models, engine setup, checkpointer factory
- Key files: `models.py` (Task + TaskLog tables), `database.py` (engine + session), `checkpointer.py` (Postgres checkpointer)

**`mcp-server/`:**
- Purpose: MCP tool bridge between agent-service and rag-service
- Contains: Single Python file implementing FastMCP server with all tools
- Key files: `server.py` (1604 lines -- all tools, crawling logic, search, Qdrant access)

**`rag-service/`:**
- Purpose: Core domain logic -- all RAG pipelines, Shopify integration, translation, search
- Contains: FastAPI app, indexers, Shopify clients, translation pipeline, agent SDK integration
- Key files: `app.py` (1943 lines -- all endpoints), `web_indexer.py` (1666 lines -- web crawler), `product_indexer.py`, `translator.py`

**`rag-service/agent/`:**
- Purpose: OpenAI Agents SDK integration for autonomous multi-step tasks
- Contains: Agent factory, tool definitions, runner, Redis session persistence, CLI
- Key files: `__init__.py` (AgentTask + create_agent), `tools.py` (8 function tools), `runner.py` (task executor)

## Key File Locations

**Entry Points:**
- `agent-service/main.py`: Agent orchestrator FastAPI app (start with uvicorn)
- `rag-service/app.py`: RAG service FastAPI app (start with uvicorn)
- `mcp-server/server.py`: MCP server (starts itself via `mcp.run()`)
- `rag-service/agent/__main__.py`: CLI agent entry point (`python -m agent "prompt"`)
- `docker-compose.yml`: Full stack orchestration

**Configuration:**
- `agent-service/config.py`: Agent service settings via pydantic-settings
- `agent-service/mcp_client/mcp_servers.json`: MCP server connection registry
- `docker-compose.yml`: Service definitions, ports, environment variables, dependencies
- `.env.example`: Template for required environment variables
- `.env`: Actual secrets (never read or commit contents)

**Core Domain Logic:**
- `rag-service/web_indexer.py`: Web crawling, content extraction (trafilatura + BeautifulSoup), hybrid indexing
- `rag-service/product_indexer.py`: Shopify product catalog indexing with hybrid search
- `rag-service/catalog_indexer.py`: Shopify collection + page indexing
- `rag-service/shopify_client.py`: Shopify REST Admin API client with metadata extraction
- `rag-service/shopify_graphql.py`: Shopify GraphQL client with bulk operations
- `rag-service/translator.py`: LLM-based translation with glossary awareness
- `rag-service/product_classifier.py`: Competitor product extraction via LLM + entity resolution
- `rag-service/webhook_handler.py`: Real-time Shopify webhook processing

**Search & Retrieval:**
- `rag-service/sparse_encoder.py`: BM25 sparse encoder (singleton, fastembed)
- `rag-service/reranker.py`: CrossEncoder reranker (singleton)
- `rag-service/qdrant_utils.py`: QdrantClient factory (handles HTTP/HTTPS)

**State Management:**
- `agent-service/state/models.py`: SQLAlchemy Task + TaskLog models
- `agent-service/state/database.py`: Async engine + session factory
- `agent-service/state/checkpointer.py`: LangGraph Postgres checkpointer
- `rag-service/sync_state.py`: Redis-backed SyncStateStore + ContentHashStore
- `rag-service/agent/session_store.py`: Redis-backed SessionStore for agent tasks
- `rag-service/agent/redis_session.py`: Redis-backed SessionABC for conversation history

**Static UIs:**
- `rag-service/static/index.html`: RAG dashboard (crawling, products, translation, agent)
- `agent-service/static/index.html`: Agent chat interface

## Naming Conventions

**Files:**
- `snake_case.py`: All Python modules (e.g., `web_indexer.py`, `shopify_client.py`)
- Single-word preferred where possible: `app.py`, `config.py`, `main.py`
- Underscore-separated compound names: `product_indexer.py`, `catalog_indexer.py`
- Agent tools in dedicated file: `tools.py` (not per-tool files)

**Directories:**
- `snake_case/`: All directory names (e.g., `agent-service/`, `mcp_client/`, `rag-service/`)
- Service directories use hyphen: `agent-service/`, `rag-service/`, `mcp-server/`
- Internal packages use underscore: `mcp_client/`

**Classes:**
- `PascalCase`: All classes (e.g., `WebIndexer`, `ProductIndexer`, `MCPManager`, `ShopifyClient`)
- Suffix pattern: `*Indexer` for indexing classes, `*Store` for persistence classes, `*Job` for tracking dataclasses

**Functions:**
- `snake_case`: All functions and methods (e.g., `create_supervisor`, `search_products`, `index_all_products`)
- Private methods prefixed with underscore: `_ensure_collection`, `_process_product_batch`

**Constants:**
- `UPPER_SNAKE_CASE`: Module-level constants (e.g., `QDRANT_URL`, `SYSTEM_PROMPT`, `BGE_QUERY_PREFIX`)

**Qdrant Collections:**
- `snake_case` strings: `web_pages`, `product_catalog`, `product_collections`, `product_pages`, `competitor_products`

## Where to Add New Code

**New API Endpoint (agent-service):**
- Create router module: `agent-service/api/{domain}.py`
- Follow pattern: `APIRouter(prefix="/api/{domain}", tags=["{Domain}"])`
- Register in: `agent-service/main.py` via `app.include_router()`
- Add Pydantic request/response models in the same router file

**New API Endpoint (rag-service):**
- Add directly to: `rag-service/app.py` (monolithic file -- all routes are here)
- Follow pattern: `@app.post("/path")` with Pydantic request model class defined above the endpoint
- Group with related endpoints by comment section

**New MCP Tool:**
- Add to: `mcp-server/server.py` using `@mcp.tool()` decorator
- If the tool proxies to rag-service, use the `async with httpx.AsyncClient()` pattern
- If the tool accesses Qdrant directly, use `ctx.request_context.lifespan_context` for shared resources

**New Indexer:**
- Create: `rag-service/{domain}_indexer.py`
- Follow pattern: Class with `__init__(qdrant_url, qdrant_api_key, model)`, `_ensure_collection()`, `search()`, `get_stats()`
- Use named vectors: `{"dense": VectorParams(...)}` + `{"sparse": SparseVectorParams(...)}`
- Initialize in: `rag-service/app.py` lifespan function, assign to module-level global

**New Agent Tool (rag-service agent):**
- Add to: `rag-service/agent/tools.py`
- Use `@function_tool` decorator from `agents` package
- First parameter: `ctx: Ctx` (which is `RunContextWrapper[AgentServices]`)
- Access services via `ctx.context` (has `.web_indexer`, `.product_indexer`, `.llm_client`)
- Add to `ALL_TOOLS` list at bottom of file

**New SQLAlchemy Model (agent-service):**
- Add to: `agent-service/state/models.py`
- Extend `Base` (DeclarativeBase)
- Tables auto-created at startup via `Base.metadata.create_all` in `main.py` lifespan

**New Background Job Type:**
- Create dataclass following `CrawlJob`/`ProductSyncJob` pattern in the relevant service file
- Add in-memory dict for tracking: `{job_type}_jobs: Dict[str, JobClass] = {}`
- Add POST endpoint to start, GET endpoint to poll status

**New Shopify Integration:**
- REST operations: Add to `rag-service/shopify_client.py`
- GraphQL operations: Add to `rag-service/shopify_graphql.py`
- Add `flatten_graphql_{entity}` static method for response normalization
- Webhook handling: Add topic handler in `rag-service/webhook_handler.py`

## Special Directories

**`.venv/` (in mcp-server):**
- Purpose: Python virtual environment for MCP server
- Generated: Yes
- Committed: No (should be gitignored)

**`static/` (in rag-service and agent-service):**
- Purpose: Embedded web dashboard and chat UIs (vanilla HTML/CSS/JS)
- Generated: No (hand-written)
- Committed: Yes

**`repos/`:**
- Purpose: Placeholder for external repository references
- Generated: No
- Committed: Yes (empty)
- Note: Currently empty, may have been used for code repository indexing

**`docs/plans/`:**
- Purpose: Planning and design documents
- Generated: No
- Committed: Yes

**`.planning/codebase/`:**
- Purpose: GSD codebase analysis documents (auto-generated)
- Generated: Yes
- Committed: Yes

---

*Structure analysis: 2026-03-09*
