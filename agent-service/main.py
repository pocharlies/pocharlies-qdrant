"""Pocharlies Agent Orchestrator - FastAPI entrypoint."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

logger = logging.getLogger("agent")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Agent service starting...")
    yield
    logger.info("Agent service shutting down...")


app = FastAPI(
    title="Pocharlies Agent Orchestrator",
    description=(
        "Production-grade LangGraph agent for Pocharlies airsoft e-commerce.\n\n"
        "## Capabilities\n"
        "- **Chat**: Converse with the agent to give instructions\n"
        "- **Workflows**: Automated multi-step operations\n"
        "- **Schedules**: Cron jobs for recurring tasks\n"
        "- **MCP**: Dynamic tool loading from any MCP server\n"
        "- **Events**: React to system events in real-time\n"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.get("/api/health", tags=["System"])
async def health():
    return {"status": "ok", "service": "agent-orchestrator"}
