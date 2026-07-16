"""
retrieve/__init__.py — Full retrieval pipeline orchestrator.

Pipeline:
  1. Semantic cache check (cosine >= 0.95 → instant return)
  2. Vector search (ChromaDB, top 10)
  3. BM25 search (sparse, top 10)
  4. Reciprocal Rank Fusion (k=60, merged top 10)
  5. Cross-encoder reranking (top 10 → top 3)

Public API:
  retrieve(query) → RetrievalResult
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from retrieve.bm25_search import bm25_search
from retrieve.cache import check_cache
from retrieve.fusion import reciprocal_rank_fusion
from retrieve.reranker import rerank
from retrieve.vector_search import SearchResult, vector_search
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class RetrievalResult:
    """Result from the retrieval pipeline."""
    chunks: List[SearchResult]          # Top-3 reranked chunks (empty on cache hit)
    cache_hit: bool                     # True if answer was served from cache
    cached_answer: Optional[str]        # The cached answer (only set when cache_hit=True)
    latency_ms: Dict[str, int] = field(default_factory=dict)


def retrieve(query: str) -> RetrievalResult:
    """
    Run the full retrieval pipeline for a query.

    On cache hit:  Returns RetrievalResult with cache_hit=True, cached_answer set.
    On cache miss: Returns RetrievalResult with cache_hit=False, chunks populated.

    Args:
        query: The user's question.

    Returns:
        RetrievalResult with chunks (or cached answer) and latency breakdown.
    """
    timings: Dict[str, int] = {}

    # ── Step 1: Semantic cache ─────────────────────────────────────────────────
    t0 = time.time()
    cached_answer = check_cache(query)
    timings["cache_ms"] = round((time.time() - t0) * 1000)

    if cached_answer is not None:
        log.info(f"Cache HIT — served in {timings['cache_ms']}ms")
        return RetrievalResult(
            chunks=[],
            cache_hit=True,
            cached_answer=cached_answer,
            latency_ms=timings,
        )

    # ── Step 2: Vector search ──────────────────────────────────────────────────
    t1 = time.time()
    vector_results = vector_search(query)
    timings["vector_ms"] = round((time.time() - t1) * 1000)

    # ── Step 3: BM25 search ────────────────────────────────────────────────────
    t2 = time.time()
    bm25_results = bm25_search(query)
    timings["bm25_ms"] = round((time.time() - t2) * 1000)

    # ── Step 4: Reciprocal Rank Fusion ─────────────────────────────────────────
    t3 = time.time()
    fused = reciprocal_rank_fusion(vector_results, bm25_results)
    timings["rrf_ms"] = round((time.time() - t3) * 1000)

    # ── Step 5: Cross-encoder reranking ───────────────────────────────────────
    t4 = time.time()
    top_chunks = rerank(query, fused)
    timings["rerank_ms"] = round((time.time() - t4) * 1000)

    timings["total_ms"] = sum(timings.values())

    log.info(
        f"Retrieval MISS | "
        f"cache={timings['cache_ms']}ms | "
        f"vector={timings['vector_ms']}ms | "
        f"bm25={timings['bm25_ms']}ms | "
        f"rrf={timings['rrf_ms']}ms | "
        f"rerank={timings['rerank_ms']}ms | "
        f"→ {len(top_chunks)} chunks"
    )

    return RetrievalResult(
        chunks=top_chunks,
        cache_hit=False,
        cached_answer=None,
        latency_ms=timings,
    )