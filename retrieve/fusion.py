"""
retrieve/fusion.py — Reciprocal Rank Fusion (RRF) for hybrid search.

RRF merges results from two ranked lists without requiring score normalization.
Each chunk gets a score: score = Σ 1/(k + rank_i) summed across all lists it appears in.

Why k=60:
  - The constant k dampens the contribution of very high ranks.
  - k=60 is the empirically validated default from the original RRF paper
    (Cormack, Clarke, Buettcher 2009).
  - Lower k over-weights top-ranked items; higher k under-uses rank information.

Example with k=60:
  - Rank 1 in both lists: 1/61 + 1/61 = 0.0328
  - Rank 1 in one list only: 1/61 = 0.0164
  - Rank 10 in both lists: 1/70 + 1/70 = 0.0286
"""

from typing import Dict, List

from config import Config
from retrieve.vector_search import SearchResult
from utils.logger import get_logger

log = get_logger(__name__)


def reciprocal_rank_fusion(
    vector_results: List[SearchResult],
    bm25_results: List[SearchResult],
    k: int = None,
) -> List[SearchResult]:
    """
    Merge and re-rank two result lists using Reciprocal Rank Fusion.

    Args:
        vector_results: Ranked results from vector search.
        bm25_results: Ranked results from BM25 search.
        k: RRF constant (default: Config.RRF_K = 60).

    Returns:
        Deduplicated list sorted by RRF score descending.
        SearchResult.score is replaced with the RRF score.
    """
    if k is None:
        k = Config.RRF_K

    rrf_scores: Dict[str, float] = {}
    chunk_lookup: Dict[str, SearchResult] = {}

    for rank, result in enumerate(vector_results, start=1):
        rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0.0) + 1.0 / (k + rank)
        chunk_lookup[result.chunk_id] = result

    for rank, result in enumerate(bm25_results, start=1):
        rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0.0) + 1.0 / (k + rank)
        if result.chunk_id not in chunk_lookup:
            chunk_lookup[result.chunk_id] = result

    sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

    fused: List[SearchResult] = []
    for chunk_id in sorted_ids:
        r = chunk_lookup[chunk_id]
        fused.append(SearchResult(
            chunk_id=r.chunk_id,
            text=r.text,
            score=rrf_scores[chunk_id],
            source_file=r.source_file,
            page_number=r.page_number,
            chunk_index=r.chunk_index,
        ))

    log.debug(
        f"RRF: {len(vector_results)} vector + {len(bm25_results)} BM25 "
        f"→ {len(fused)} unique (top RRF={fused[0].score:.4f})" if fused else "RRF: empty inputs"
    )
    return fused