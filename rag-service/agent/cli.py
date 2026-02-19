"""
CLI entry point for the RAG agent.
Usage: python -m agent "Index all Kubernetes migration guides"
"""

import asyncio
import os
import uuid
import logging
from datetime import datetime, timezone

import typer
from agents import Runner

app = typer.Typer(help="Pocharlies RAG Agent CLI")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def _build_services():
    """Initialize services from environment variables (same as app.py lifespan)."""
    from qdrant_utils import make_qdrant_client
    from sentence_transformers import SentenceTransformer
    from openai import OpenAI

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    vllm_base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    llm_api_key = os.getenv("LITELLM_API_KEY", "none")

    logger.info("Loading embedding model...")
    model = SentenceTransformer(embedding_model)

    from web_indexer import WebIndexer
    from retriever import CodeRetriever
    from product_indexer import ProductIndexer
    from devops_indexer import DevOpsIndexer, LogAnalyzer

    web_indexer = WebIndexer(qdrant_url=qdrant_url, qdrant_api_key=qdrant_api_key, model=model)
    retriever = CodeRetriever(qdrant_url=qdrant_url, qdrant_api_key=qdrant_api_key, embedding_model=embedding_model)
    product_indexer = ProductIndexer(qdrant_url=qdrant_url, qdrant_api_key=qdrant_api_key, model=model)
    devops_indexer = DevOpsIndexer(qdrant_url=qdrant_url, qdrant_api_key=qdrant_api_key, model=model)
    llm_client = OpenAI(base_url=vllm_base_url, api_key=llm_api_key)
    log_analyzer = LogAnalyzer(llm_client=llm_client, devops_indexer=devops_indexer)

    from . import AgentServices
    return AgentServices(
        web_indexer=web_indexer,
        retriever=retriever,
        product_indexer=product_indexer,
        devops_indexer=devops_indexer,
        log_analyzer=log_analyzer,
        llm_client=llm_client,
    )


@app.command()
def run(
    prompt: str = typer.Argument(..., help="Task for the agent to perform"),
    max_turns: int = typer.Option(30, help="Maximum agent iterations"),
    vllm_url: str = typer.Option(
        None, "--vllm-url", envvar="VLLM_BASE_URL",
        help="vLLM API base URL (default: $VLLM_BASE_URL or http://localhost:8000/v1)",
    ),
):
    """Run the RAG agent with a natural language prompt."""
    vllm_url = vllm_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    api_key = os.getenv("LITELLM_API_KEY", "none")

    logger.info(f"Connecting to LLM at {vllm_url}...")
    from . import create_agent, AgentTask
    agent, model_id = create_agent(vllm_url, api_key)
    typer.echo(f"Model: {model_id}")

    logger.info("Initializing services...")
    services = _build_services()

    typer.echo(f"Running: {prompt}")
    typer.echo("---")

    asyncio.run(_run_with_persistence(agent, services, prompt, max_turns))


async def _run_with_persistence(agent, services, prompt: str, max_turns: int):
    """Run agent with Redis persistence if available, fallback to plain run."""
    from . import AgentTask
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    redis_client = None
    session_store = None
    session = None

    try:
        from redis.asyncio import Redis as AsyncRedis
        redis_client = AsyncRedis.from_url(redis_url, decode_responses=False)
        await redis_client.ping()

        from .session_store import SessionStore
        from .redis_session import RedisSession

        session_store = SessionStore(redis_client)

        task_id = uuid.uuid4().hex[:12]
        task = AgentTask(
            task_id=task_id,
            prompt=prompt,
            started_at=datetime.now(timezone.utc).isoformat(),
            source="cli",
        )
        task._on_log = session_store.add_log
        task._on_step = session_store.add_step
        await session_store.create_task(task, source="cli")

        session = RedisSession(task_id, redis_client)

        from .runner import run_task
        result = await run_task(
            agent, services, prompt,
            max_turns=max_turns, task=task,
            session=session, session_store=session_store,
        )
        typer.echo(result.summary or "No output")
        typer.echo(f"\nTask ID: {task_id} (visible on dashboard)")

    except ImportError:
        logger.warning("redis package not available, running without persistence")
        result = Runner.run_sync(agent, input=prompt, context=services, max_turns=max_turns)
        typer.echo(result.final_output)
    except Exception as e:
        if session_store:
            # Redis was available but something else failed
            raise
        logger.warning(f"Redis unavailable ({e}), running without persistence")
        result = Runner.run_sync(agent, input=prompt, context=services, max_turns=max_turns)
        typer.echo(result.final_output)
    finally:
        if redis_client:
            await redis_client.aclose()


@app.command()
def status(
    vllm_url: str = typer.Option(
        None, "--vllm-url", envvar="VLLM_BASE_URL",
        help="vLLM API base URL",
    ),
):
    """Check if the LLM and Qdrant are available."""
    from openai import OpenAI
    from qdrant_utils import make_qdrant_client

    vllm_url = vllm_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    api_key = os.getenv("LITELLM_API_KEY", "none")

    try:
        client = OpenAI(base_url=vllm_url, api_key=api_key)
        models = client.models.list()
        model_id = models.data[0].id if models.data else None
        typer.echo(f"LLM: OK ({model_id})")
    except Exception as e:
        typer.echo(f"LLM: FAIL ({e})")

    try:
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        qdrant = make_qdrant_client(qdrant_url, qdrant_api_key)
        collections = [c.name for c in qdrant.get_collections().collections]
        typer.echo(f"Qdrant: OK ({len(collections)} collections: {', '.join(collections)})")
    except Exception as e:
        typer.echo(f"Qdrant: FAIL ({e})")

    # Check Redis
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    try:
        import redis
        r = redis.from_url(redis_url)
        r.ping()
        task_count = r.zcard("agent:tasks")
        typer.echo(f"Redis: OK ({task_count} sessions)")
    except Exception as e:
        typer.echo(f"Redis: FAIL ({e})")


if __name__ == "__main__":
    app()
