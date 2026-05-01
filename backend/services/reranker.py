"""
reranker.py — Cross-encoder reranker

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (CPU-friendly)
"""

from sentence_transformers.cross_encoder import CrossEncoder
from core.config import settings

_reranker = None

def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        print(f"[Reranker] Loading model: {settings.RERANKER_MODEL}")
        _reranker = CrossEncoder(settings.RERANKER_MODEL, max_length=512)
    return _reranker


def rerank(query: str, documents: list[dict], top_n: int | None = None) -> list[dict]:
    """
    Score each document against the query and return top-N by relevance.
    Each document dict must have a "content" key.
    """
    if not documents:
        return []

    if top_n is None:
        top_n = settings.RERANK_TOP_N

    reranker = _get_reranker()
    pairs    = [(query, doc["content"]) for doc in documents]
    scores   = reranker.predict(pairs)

    scored = [(float(score), doc) for score, doc in zip(scores, documents)]
    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, doc in scored[:top_n]:
        enriched = doc.copy()
        enriched["rerank_score"] = round(score, 4)
        results.append(enriched)

    return results
