"""Shared Qdrant client utilities."""

from typing import Optional
from urllib.parse import urlparse

from qdrant_client import QdrantClient


def make_qdrant_client(url: str, api_key: Optional[str] = None) -> QdrantClient:
    """Create QdrantClient, handling HTTPS URLs (qdrant_client 1.7 needs explicit port/https)."""
    parsed = urlparse(url)
    if parsed.scheme == "https":
        return QdrantClient(
            host=parsed.hostname,
            port=parsed.port or 443,
            https=True,
            api_key=api_key,
        )
    return QdrantClient(url=url, api_key=api_key)
