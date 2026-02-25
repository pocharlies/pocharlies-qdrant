"""MCP API — manage MCP servers and tools at runtime."""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from graphs.supervisor import create_supervisor

logger = logging.getLogger("agent.api.mcp")

router = APIRouter(prefix="/api/mcp", tags=["MCP Servers"])


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ServerStatusResponse(BaseModel):
    """Status of a single MCP server."""

    name: str
    type: str | None = None
    connected: bool
    tools_count: int
    description: str = ""


class AddServerRequest(BaseModel):
    """Request body for adding a new MCP server."""

    name: str = Field(..., description="Unique server name")
    type: str = Field("sse", description="Server type: 'sse' or 'stdio'")
    url: str | None = Field(None, description="SSE server URL (for type=sse)")
    command: str | None = Field(None, description="Command to run (for type=stdio)")
    args: list[str] = Field(default_factory=list, description="Command arguments (for type=stdio)")
    env: dict[str, str] = Field(default_factory=dict, description="Environment variables")
    description: str = Field("", description="Human-readable description")


class ToolResponse(BaseModel):
    """A single tool exposed by an MCP server."""

    name: str
    description: str = ""
    server: str = ""


class ReloadResponse(BaseModel):
    """Result of a reload operation."""

    status: str
    servers: int
    tools: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_mcp_manager(request: Request):
    """Retrieve the MCPManager from app state."""
    manager = getattr(request.app.state, "mcp_manager", None)
    if manager is None:
        raise HTTPException(status_code=503, detail="MCP manager not initialized")
    return manager


async def _rebuild_supervisor(request: Request) -> None:
    """Rebuild the supervisor graph with current MCP tools and update app state."""
    manager = _get_mcp_manager(request)
    tools = manager.get_all_tools()
    checkpointer = getattr(request.app.state, "checkpointer", None)
    graph = create_supervisor(checkpointer, tools)
    request.app.state.supervisor = graph
    logger.info("Supervisor rebuilt with %d tools", len(tools))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/servers",
    response_model=list[ServerStatusResponse],
    summary="List MCP servers",
)
async def list_servers(request: Request) -> list[ServerStatusResponse]:
    """Return status of all connected MCP servers."""
    manager = _get_mcp_manager(request)
    return [ServerStatusResponse(**s) for s in manager.get_server_status()]


@router.post(
    "/servers",
    response_model=ServerStatusResponse,
    status_code=201,
    summary="Add MCP server",
)
async def add_server(body: AddServerRequest, request: Request) -> ServerStatusResponse:
    """Add and connect a new MCP server, then rebuild the supervisor graph."""
    manager = _get_mcp_manager(request)

    # Build the config dict expected by MCPManager
    config: dict[str, Any] = {"type": body.type, "description": body.description}
    if body.url:
        config["url"] = body.url
    if body.command:
        config["command"] = body.command
    if body.args:
        config["args"] = body.args
    if body.env:
        config["env"] = body.env

    await manager.add_server(body.name, config)
    await _rebuild_supervisor(request)

    conn = manager.servers.get(body.name)
    return ServerStatusResponse(
        name=body.name,
        type=body.type,
        connected=conn.connected if conn else False,
        tools_count=len(conn.tools) if conn else 0,
        description=body.description,
    )


@router.delete(
    "/servers/{name}",
    summary="Remove MCP server",
)
async def remove_server(name: str, request: Request) -> dict:
    """Disconnect and remove an MCP server, then rebuild the supervisor graph."""
    manager = _get_mcp_manager(request)
    if name not in manager.servers:
        raise HTTPException(status_code=404, detail=f"Server '{name}' not found")

    await manager.remove_server(name)
    await _rebuild_supervisor(request)

    return {"status": "removed", "name": name}


@router.get(
    "/servers/{name}/tools",
    response_model=list[ToolResponse],
    summary="List tools from one server",
)
async def list_server_tools(name: str, request: Request) -> list[ToolResponse]:
    """Return tools exposed by a specific MCP server."""
    manager = _get_mcp_manager(request)
    conn = manager.servers.get(name)
    if conn is None:
        raise HTTPException(status_code=404, detail=f"Server '{name}' not found")

    return [
        ToolResponse(name=t.name, description=t.description or "", server=name)
        for t in conn.tools
    ]


@router.get(
    "/tools",
    response_model=list[ToolResponse],
    summary="List all tools from all servers",
)
async def list_all_tools(request: Request) -> list[ToolResponse]:
    """Return every tool from every connected MCP server."""
    manager = _get_mcp_manager(request)
    tools: list[ToolResponse] = []
    for server_name, conn in manager.servers.items():
        for t in conn.tools:
            tools.append(
                ToolResponse(name=t.name, description=t.description or "", server=server_name)
            )
    return tools


@router.post(
    "/servers/reload",
    response_model=ReloadResponse,
    summary="Reload all MCP servers",
)
async def reload_servers(request: Request) -> ReloadResponse:
    """Hot-reload all MCP servers and rebuild the supervisor graph."""
    manager = _get_mcp_manager(request)
    await manager.reload()
    await _rebuild_supervisor(request)

    all_tools = manager.get_all_tools()
    return ReloadResponse(
        status="reloaded",
        servers=len(manager.servers),
        tools=len(all_tools),
    )
