"""
Reranker module for RAG Pipeline
Uses CrossEncoder (bge-reranker-v2-m3) to rerank search results for higher relevance.
"""

import logging
from typing import List, Dict, Optional

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

_reranker_instance: Optional["Reranker"] = None


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        logger.info(f"Loading reranker model: {model_name}...")
        self.model = CrossEncoder(model_name, max_length=512)
        self.model_name = model_name
        logger.info("Reranker model loaded")

    def rerank(
        self,
        query: str,
        results: List[Dict],
        text_key: str = "text",
        top_k: int = 5,
    ) -> List[Dict]:
        """Rerank results by cross-encoder relevance score.

        Args:
            query: The user query
            results: List of result dicts, each must have `text_key` field
            text_key: Key in result dict containing the text to score
            top_k: Number of results to return after reranking
        """
        if not results:
            return results

        pairs = [(query, r.get(text_key, "")) for r in results]
        scores = self.model.predict(pairs)

        for i, r in enumerate(results):
            r["rerank_score"] = float(scores[i])

        reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)

        # Normalize: top result gets score 1.0
        if reranked:
            max_score = reranked[0]["rerank_score"]
            for r in reranked:
                r["score"] = round(r["rerank_score"] / max_score, 4) if max_score > 0 else 0
                del r["rerank_score"]

        return reranked[:top_k]


def get_reranker() -> Optional[Reranker]:
    """Singleton accessor. Returns None if reranking is disabled."""
    return _reranker_instance


def init_reranker(model_name: str = "BAAI/bge-reranker-v2-m3") -> Reranker:
    """Initialize the singleton reranker. Call once at startup."""
    global _reranker_instance
    _reranker_instance = Reranker(model_name)
    return _reranker_instance
