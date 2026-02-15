"""
Singleton BM25 sparse encoder using fastembed.
Shared across indexers and search for hybrid vector generation.
"""

import logging
from typing import List

from fastembed import SparseTextEmbedding
from qdrant_client.http.models import SparseVector

logger = logging.getLogger(__name__)

_bm25_model = None


def get_bm25_model() -> SparseTextEmbedding:
    global _bm25_model
    if _bm25_model is None:
        logger.info("Loading BM25 sparse encoder (Qdrant/bm25)...")
        _bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        logger.info("BM25 sparse encoder loaded")
    return _bm25_model


def encode_sparse(texts: List[str]) -> List[SparseVector]:
    """Batch encode documents into Qdrant SparseVector objects."""
    model = get_bm25_model()
    return [
        SparseVector(
            indices=emb.indices.tolist(),
            values=emb.values.tolist(),
        )
        for emb in model.embed(texts)
    ]


def encode_sparse_query(text: str) -> SparseVector:
    """Encode a single query (BM25 query weighting)."""
    model = get_bm25_model()
    emb = list(model.query_embed(text))[0]
    return SparseVector(
        indices=emb.indices.tolist(),
        values=emb.values.tolist(),
    )
