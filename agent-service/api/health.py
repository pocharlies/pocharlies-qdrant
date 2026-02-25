"""Health & Stats API — system health checks and operational statistics."""

import logging

import httpx
import redis.asyncio as redis
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field
from sqlalchemy import func, select, text

from config import settings
from state.database import async_session
from state.models import Task

logger = logging.getLogger("agent.api.health")

router = APIRouter(tags=["System"])


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Simple health check response."""

    status: str = "ok"
    service: str = "agent-orchestrator"
    version: str = "1.0.0"


class DependencyStatus(BaseModel):
    """Status of a single dependency."""

    status: str
    detail: str = ""


class DependenciesResponse(BaseModel):
    """Aggregated dependency health check."""

    status: str = Field(description="'ok' if all dependencies healthy, 'degraded' otherwise")
    dependencies: dict[str, DependencyStatus]


class StatsResponse(BaseModel):
    """Operational statistics."""

    tasks: dict[str, int] = Field(default_factory=dict, description="Task counts by status")
    mcp_servers: int = 0
    mcp_tools: int = 0


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/api/health", response_model=HealthResponse, summary="Health check")
async def health() -> HealthResponse:
    """Simple OK check to verify the service is running."""
    return HealthResponse()


@router.get(
    "/api/health/dependencies",
    response_model=DependenciesResponse,
    summary="Dependency health",
)
async def health_dependencies(request: Request) -> DependenciesResponse:
    """Check connectivity to all external dependencies."""
    deps: dict[str, DependencyStatus] = {}

    # --- Postgres ---
    try:
        async with async_session() as session:
            await session.execute(text("SELECT 1"))
        deps["postgres"] = DependencyStatus(status="ok")
    except Exception as exc:
        deps["postgres"] = DependencyStatus(status="error", detail=str(exc))

    # --- Redis ---
    try:
        r = redis.from_url(settings.redis_url)
        try:
            await r.ping()
            deps["redis"] = DependencyStatus(status="ok")
        finally:
            await r.aclose()
    except Exception as exc:
        deps["redis"] = DependencyStatus(status="error", detail=str(exc))

    # --- RAG service ---
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.rag_service_url}/health")
            if resp.status_code == 200:
                deps["rag_service"] = DependencyStatus(status="ok")
            else:
                deps["rag_service"] = DependencyStatus(
                    status="error",
                    detail=f"HTTP {resp.status_code}",
                )
    except Exception as exc:
        deps["rag_service"] = DependencyStatus(status="error", detail=str(exc))

    # --- MCP servers ---
    manager = getattr(request.app.state, "mcp_manager", None)
    if manager is not None:
        statuses = manager.get_server_status()
        total = len(statuses)
        connected = sum(1 for s in statuses if s["connected"])
        deps["mcp"] = DependencyStatus(
            status="ok" if connected == total else "degraded",
            detail=f"{connected}/{total} servers connected",
        )
    else:
        deps["mcp"] = DependencyStatus(status="error", detail="MCP manager not initialized")

    # Overall status
    all_ok = all(d.status == "ok" for d in deps.values())
    overall = "ok" if all_ok else "degraded"

    return DependenciesResponse(status=overall, dependencies=deps)


@router.get("/api/stats", response_model=StatsResponse, summary="System statistics")
async def stats(request: Request) -> StatsResponse:
    """Return task counts by status and MCP server/tool counts."""

    # --- Task counts by status ---
    task_counts: dict[str, int] = {}
    try:
        async with async_session() as session:
            stmt = select(Task.status, func.count(Task.id)).group_by(Task.status)
            result = await session.execute(stmt)
            for status_val, count_val in result.all():
                task_counts[status_val] = count_val
    except Exception as exc:
        logger.warning("Failed to query task counts: %s", exc)

    # --- MCP stats ---
    mcp_servers = 0
    mcp_tools = 0
    manager = getattr(request.app.state, "mcp_manager", None)
    if manager is not None:
        mcp_servers = len(manager.servers)
        mcp_tools = len(manager.get_all_tools())

    return StatsResponse(
        tasks=task_counts,
        mcp_servers=mcp_servers,
        mcp_tools=mcp_tools,
    )
