"""
CLI entry point for the RAG agent.
Usage: python -m agent "Index all Kubernetes migration guides"
"""

import os
import logging

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
    from . import create_agent
    agent, model_id = create_agent(vllm_url, api_key)
    typer.echo(f"Model: {model_id}")

    logger.info("Initializing services...")
    services = _build_services()

    typer.echo(f"Running: {prompt}")
    typer.echo("---")

    result = Runner.run_sync(agent, input=prompt, context=services, max_turns=max_turns)
    typer.echo(result.final_output)


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


if __name__ == "__main__":
    app()
