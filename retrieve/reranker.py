"""
retrieve/reranker.py — Cross-encoder reranking for precision improvement.

Why cross-encoder:
  - Bi-encoders (used in vector search) encode query and document independently.
    They cannot model fine-grained query-document interaction.
  - Cross-encoders see (query, document) as a single sequence — they capture
    nuanced relevance signals that bi-encoders miss.
  - Cost: ~500ms for 10 pairs on CPU. Worth it for the last-mile precision.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - 80MB (vs 400MB+ for BGE rerankers)
  - Trained on MS MARCO passage ranking
  - Improves MRR@10 by ~8-12% over bi-encoder alone
"""

import time
from typing import List

from sentence_transformers import CrossEncoder

from config import Config
from retrieve.vector_search import SearchResult
from utils.logger import get_logger

log = get_logger(__name__)

_reranker: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        log.info(f"Loading cross-encoder: {Config.RERANKER_MODEL} (~80MB, first load downloads)")
        _reranker = CrossEncoder(Config.RERANKER_MODEL)
        log.info("Cross-encoder loaded and cached")
    return _reranker


def rerank(
    query: str,
    candidates: List[SearchResult],
    top_k: int = None,
) -> List[SearchResult]:
    """
    Score each candidate with the cross-encoder and return top_k by score.

    Args:
        query: The user's question.
        candidates: Candidates from RRF (up to VECTOR_TOP_K results).
        top_k: How many to return (default: Config.RERANK_TOP_K = 3).

    Returns:
        Top_k SearchResult objects sorted by cross-encoder score descending.
        SearchResult.score is replaced with the cross-encoder logit score.
    """
    if top_k is None:
        top_k = Config.RERANK_TOP_K

    if not candidates:
        return []

    reranker = _get_reranker()
    pairs = [(query, c.text) for c in candidates]

    t0 = time.time()
    scores = reranker.predict(pairs, show_progress_bar=False)
    elapsed = round((time.time() - t0) * 1000)

    scored = sorted(zip(scores.tolist(), candidates), key=lambda x: x[0], reverse=True)

    result: List[SearchResult] = []
    for score, candidate in scored[:top_k]:
        result.append(SearchResult(
            chunk_id=candidate.chunk_id,
            text=candidate.text,
            score=float(score),
            source_file=candidate.source_file,
            page_number=candidate.page_number,
            chunk_index=candidate.chunk_index,
        ))

    log.debug(
        f"Reranked ({elapsed}ms): {len(candidates)} → {len(result)}, "
        f"top score={result[0].score:.3f}" if result else "reranked: empty"
    )
    return result