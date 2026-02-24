# LangGraph Agent Orchestrator — Design Document

**Date**: 2026-02-24
**Status**: Approved
**Approach**: A — LangGraph + Custom Orchestration

---

## 1. Overview

A new `agent-service` that runs 24/7 as a production-grade LangGraph orchestrator for
Pocharlies airsoft e-commerce. It receives instructions via chat, cron schedules, or
system events, reasons with a supervisor agent, loads tools dynamically from MCP servers,
and spawns specialized sub-workflows for complex multi-step tasks.

### Goals
- Replace the simple fire-and-forget agent with a persistent, stateful orchestrator
- Support chat, scheduled, and event-driven execution patterns
- Dynamically load tools from any MCP server (own + third-party)
- Durable execution — survives restarts, resumes in-flight workflows
- Human-in-the-loop approval for destructive or expensive actions
- Full REST API with Swagger documentation
- Chat UI for day-to-day interaction

### Non-goals (Phase 1)
- Visual workflow builder / drag-and-drop
- Multi-tenant / multi-user auth
- Dashboard analytics (Phase 4)

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────┐
│  agent-service (FastAPI + LangGraph)                     │
│                                                          │
│  ┌───────────┐  ┌────────────┐  ┌───────────┐           │
│  │ Chat API  │  │ Scheduler  │  │  Event    │           │
│  │(WebSocket)│  │(APScheduler)│  │ Listener  │           │
│  └─────┬─────┘  └──────┬─────┘  └─────┬─────┘           │
│        └────────┬───────┴──────────────┘                 │
│           ┌─────▼──────┐                                 │
│           │ Supervisor  │  (LangGraph ReAct StateGraph)  │
│           │   Agent     │                                │
│           └─────┬──────┘                                 │
│      ┌──────┬───┴───┬──────┐                             │
│      ▼      ▼       ▼      ▼                             │
│   Crawl  Research  Guide  Competitor                     │
│   Graph  Graph    Graph   Graph                          │
│   (sub-workflows, each a LangGraph graph)                │
│                                                          │
│   ┌──────────────────────────────────────┐               │
│   │ MCP Manager                          │               │
│   │  pocharlies-rag/ (24 tools via SSE)  │               │
│   │  github/ (stdio)                     │               │
│   │  brave-search/ (stdio)               │               │
│   │  ... (any MCP server)                │               │
│   └──────────────────────────────────────┘               │
│                                                          │
│   PostgresSaver ──► Postgres (durable state)             │
│   Redis ──► pub/sub + caching                            │
└────────────────┬─────────────────────────────────────────┘
                 │ HTTP calls
                 ▼
           rag-service:5000
```

### Tech Stack

| Component          | Technology                                      |
|--------------------|-------------------------------------------------|
| Agent framework    | LangGraph (StateGraph + Functional API)         |
| LLM                | LiteLLM proxy (self-hosted, existing)           |
| State persistence  | AsyncPostgresSaver (LangGraph native)           |
| Task/schedule DB   | PostgreSQL 16                                   |
| Scheduler          | APScheduler 4 + Postgres job store              |
| Events             | Redis pub/sub                                   |
| Tool protocol      | MCP (SSE + stdio transports)                    |
| API                | FastAPI + WebSocket                             |
| UI                 | Vanilla HTML/JS/CSS (no framework)              |
| Deployment         | Docker Compose (alongside existing services)    |

---

## 3. Project Structure

```
agent-service/
├── Dockerfile
├── requirements.txt
├── main.py                  # FastAPI app + lifespan (scheduler, Redis listener)
├── config.py                # Pydantic BaseSettings from env vars
│
├── graphs/                  # LangGraph graph definitions
│   ├── supervisor.py        # Supervisor ReAct agent (the brain)
│   ├── crawl_workflow.py    # Web crawling sub-workflow
│   ├── research_workflow.py # Deep research (forums, docs, competitors)
│   ├── guide_workflow.py    # Content generation (product guides, how-tos)
│   └── competitor_workflow.py # Price monitoring + competitor intel
│
├── mcp/                     # MCP client layer
│   ├── manager.py           # Connect, discover, route
│   ├── adapter.py           # MCP tool -> LangGraph @tool converter
│   └── mcp_servers.json     # Server configurations
│
├── scheduler/               # APScheduler cron + event triggers
│   ├── scheduler.py         # APScheduler setup with Postgres job store
│   └── jobs.py              # Predefined job definitions
│
├── state/                   # State management
│   ├── models.py            # Pydantic models (WorkflowState, TaskResult, etc.)
│   └── checkpointer.py      # AsyncPostgresSaver setup
│
├── api/                     # API layer (FastAPI routers)
│   ├── chat.py              # WebSocket chat endpoint
│   ├── workflows.py         # CRUD for workflows
│   ├── tasks.py             # Task status, history, logs
│   ├── schedules.py         # Cron job management
│   ├── mcp_api.py           # MCP server management
│   ├── events.py            # Event subscription management
│   └── health.py            # Health checks + stats
│
└── static/                  # Chat UI
    ├── index.html
    ├── style.css
    └── app.js
```

---

## 4. Supervisor Agent

The supervisor is a LangGraph ReAct StateGraph with Postgres checkpointing. All
instructions — chat, scheduled, event-driven — route through it.

### State Schema

```python
class SupervisorState(TypedDict):
    messages: Annotated[list, add_messages]  # Conversation history
    active_workflow: str | None              # Currently running sub-workflow
    workflow_results: dict                   # Accumulated results
    task_queue: list[dict]                   # Pending tasks from scheduler/events
    metadata: dict                           # Thread context (user_id, trigger_source)
```

### Behavior

- **Chat mode**: User message -> supervisor reasons -> calls tools or spawns sub-workflows -> responds
- **Scheduled trigger**: Cron fires -> injects `[SCHEDULED] ...` message -> supervisor executes
- **Event trigger**: Redis event -> injects `[EVENT] ...` message -> supervisor decides action
- **Human-in-the-loop**: Uses `interrupt()` for destructive/expensive actions, UI renders approval buttons

### LLM Configuration

```python
from langchain.chat_models import init_chat_model

model = init_chat_model(
    model="openai/your-model",
    base_url=settings.LLM_BASE_URL,
    api_key=settings.LLM_API_KEY,
    temperature=0.2,
)
```

---

## 5. MCP Client Integration

Tools are loaded dynamically from MCP servers — no hardcoded HTTP wrappers.

### Configuration: `mcp_servers.json`

```json
{
  "servers": {
    "pocharlies-rag": {
      "type": "sse",
      "url": "http://rag-service:5000/mcp/sse",
      "description": "Pocharlies RAG"
    },
    "github": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {"GITHUB_TOKEN": "${GITHUB_TOKEN}"}
    }
  }
}
```

### MCP Manager Responsibilities

1. **Boot**: Read config, connect to each server (SSE for HTTP, stdio for local processes)
2. **Discovery**: Call `tools/list` on each, collect all tool schemas
3. **Convert**: Wrap each MCP tool as LangGraph `@tool`, namespaced `server__tool_name`
4. **Execute**: Route LLM tool calls to the correct MCP server
5. **Hot reload**: Watch config for changes, reconnect without restart
6. **Health**: Track connection status, skip unavailable servers gracefully

### Existing MCP Server Change

Add SSE transport to `mcp-server/server.py` (one-liner with FastMCP) so the
agent-service can connect as a client. Alternatively, run MCP server as stdio
subprocess from agent-service.

---

## 6. Sub-Workflow Graphs

Four specialized LangGraph graphs for complex multi-step operations.

### 6a. Crawl Workflow

```
START -> analyze_site -> decide_strategy -> crawl_pages -> verify_indexing -> report
```

State: `target_url, strategy, job_id, pages_indexed, status, report`

### 6b. Research Workflow

```
START -> plan_research -> [parallel: web_search, crawl_pages, search_rag, search_competitors] -> synthesize -> store_findings
```

State: `topic, research_plan, findings, synthesis, status`

### 6c. Guide Workflow

```
START -> plan_outline -> research_gaps -> write_sections -> review -> [interrupt: approval] -> finalize -> index_to_rag
```

State: `topic, outline, product_data, sections, final_guide, approved, status`

### 6d. Competitor Workflow

```
START -> crawl_competitor -> classify_products -> compare_prices -> detect_changes -> generate_report
```

State: `competitor_domain, previous_snapshot, current_products, price_comparison, changes_detected, report, status`

### Design Decisions

- Each sub-workflow has its own state schema
- All use the same MCP tools (passed from supervisor)
- All use Postgres checkpointing — resume on restart
- Can be triggered directly via API or through the supervisor

---

## 7. Scheduler & Events

### Scheduler (APScheduler 4 + Postgres)

```python
scheduler = AsyncScheduler(
    data_store=PostgresDataStore(engine),
    role=SchedulerRole.both,
)
```

Predefined jobs (configurable via API):

| Job               | Cron       | Action                      |
|-------------------|------------|-----------------------------|
| product_sync      | 0 6 * * *  | Sync Shopify catalog daily  |
| competitor_evike  | 0 3 * * 1  | Crawl+compare evike weekly  |
| rag_health_check  | */30 * * * | Check RAG infra every 30min |
| stale_content     | 0 2 1 * *  | Detect outdated RAG content |

### Events (Redis Pub/Sub)

Channels: `agent:products`, `agent:crawl`, `agent:competitor`, `agent:translation`, `agent:command`

Event listener runs as a background task, creates new supervisor threads for each event.

### Task Tracking (Postgres)

```sql
CREATE TABLE tasks (
    id          UUID PRIMARY KEY,
    thread_id   TEXT NOT NULL,
    trigger     TEXT NOT NULL,        -- chat | schedule | event
    trigger_ref TEXT,
    status      TEXT DEFAULT 'pending',
    prompt      TEXT NOT NULL,
    summary     TEXT,
    tools_used  JSONB DEFAULT '[]',
    started_at  TIMESTAMPTZ,
    ended_at    TIMESTAMPTZ,
    error       TEXT,
    metadata    JSONB DEFAULT '{}'
);

CREATE TABLE task_logs (
    id          SERIAL PRIMARY KEY,
    task_id     UUID REFERENCES tasks(id),
    level       TEXT,
    message     TEXT,
    data        JSONB,
    created_at  TIMESTAMPTZ DEFAULT now()
);
```

---

## 8. API Layer (Swagger at /docs)

FastAPI with tagged router groups:

| Group       | Endpoints                                          |
|-------------|----------------------------------------------------|
| Chat        | `WS /ws/chat`, `POST /api/chat`, `GET /api/chat/threads`, `GET /api/chat/threads/{id}`, `DELETE /api/chat/threads/{id}` |
| Tasks       | `GET /api/tasks`, `GET /api/tasks/{id}`, `GET /api/tasks/{id}/logs`, `POST /api/tasks/{id}/cancel`, `POST /api/tasks/{id}/approve`, `POST /api/tasks/{id}/reject` |
| Schedules   | `GET /api/schedules`, `POST /api/schedules`, `PATCH /api/schedules/{id}`, `DELETE /api/schedules/{id}`, `POST /api/schedules/{id}/run`, `GET /api/schedules/{id}/history` |
| Workflows   | `POST /api/workflows/crawl`, `POST /api/workflows/research`, `POST /api/workflows/guide`, `POST /api/workflows/competitor`, `GET /api/workflows/{id}` |
| MCP Servers | `GET /api/mcp/servers`, `POST /api/mcp/servers`, `DELETE /api/mcp/servers/{name}`, `GET /api/mcp/servers/{name}/tools`, `POST /api/mcp/servers/reload`, `GET /api/mcp/tools` |
| Events      | `GET /api/events/channels`, `POST /api/events/emit`, `GET /api/events/history` |
| System      | `GET /api/health`, `GET /api/health/dependencies`, `GET /api/stats` |

All endpoints have Pydantic request/response models, proper status codes, and examples.

---

## 9. Chat Interface

Single-page HTML + vanilla JS + WebSocket. Served by FastAPI StaticFiles.

Features:
- Left sidebar: thread history + active schedules
- Main area: streaming chat with tool call visualization
- Inline interrupt handling (approve/reject buttons)
- Status bar: connection state, MCP server count, active jobs
- Progress bar for running workflows

WebSocket protocol:
- Client sends: `message`, `approve`, `reject`
- Server streams: `message`, `tool_call`, `tool_result`, `progress`, `interrupt`, `done`

---

## 10. Docker Compose Changes

Add two services:

```yaml
postgres:
  image: postgres:16-alpine
  volumes:
    - postgres_data:/var/lib/postgresql/data
  environment:
    POSTGRES_DB: langgraph
    POSTGRES_USER: agent
    POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}

agent-service:
  build: ./agent-service
  depends_on: [postgres, redis, rag-service]
  ports:
    - "127.0.0.1:8100:8100"
  environment:
    - DATABASE_URL=postgresql+asyncpg://agent:${POSTGRES_PASSWORD}@postgres:5432/langgraph
    - REDIS_URL=redis://pocharlies-redis:6379/1
    - RAG_SERVICE_URL=http://rag-service:5000
    - LLM_BASE_URL=${VLLM_BASE_URL}
    - LLM_API_KEY=${LITELLM_API_KEY}
```

### Existing Code Changes
- `mcp-server/server.py`: Add SSE transport endpoint (one-liner with FastMCP)
- Everything else: untouched

---

## 11. Implementation Phases

| Phase   | Scope                                                    |
|---------|----------------------------------------------------------|
| Phase 1 | Supervisor + MCP client + Postgres + chat API + basic UI |
| Phase 2 | 4 sub-workflows (crawl, research, guide, competitor)     |
| Phase 3 | Scheduler + events + cron management API                 |
| Phase 4 | Dashboard UI (workflow builder, analytics, monitoring)    |
