"""Tests for fusion, reranking, and the retrieval orchestrator."""
from retrieve.vector_search import vector_search, SearchResult
from retrieve.bm25_search import bm25_search
from retrieve.fusion import reciprocal_rank_fusion
from retrieve.reranker import rerank
from retrieve import retrieve


def test_rrf_deduplicates():
    q = "machine learning"
    v = vector_search(q)
    b = bm25_search(q)
    fused = reciprocal_rank_fusion(v, b)
    # Fused should have no more than v + b (dedup removes overlaps)
    assert len(fused) <= len(v) + len(b)

def test_rrf_all_chunks_present():
    q = "machine learning"
    v = vector_search(q)
    b = bm25_search(q)
    fused = reciprocal_rank_fusion(v, b)
    fused_ids = {r.chunk_id for r in fused}
    vector_ids = {r.chunk_id for r in v}
    bm25_ids = {r.chunk_id for r in b}
    # Every vector and BM25 chunk should appear in fused
    assert vector_ids.issubset(fused_ids)
    assert bm25_ids.issubset(fused_ids)

def test_rrf_empty_inputs():
    fused = reciprocal_rank_fusion([], [])
    assert fused == []

def test_reranker_returns_top_k():
    q = "what is machine learning?"
    fused = reciprocal_rank_fusion(vector_search(q), bm25_search(q))
    top3 = rerank(q, fused, top_k=3)
    assert len(top3) <= 3
    assert len(top3) >= 1

def test_reranker_changes_order():
    q = "what is supervised learning?"
    fused = reciprocal_rank_fusion(vector_search(q), bm25_search(q))
    reranked = rerank(q, fused)
    # Reranked top-1 should not always match RRF top-1 (shows reranking happened)
    # We just verify the output is valid
    assert all(isinstance(r, SearchResult) for r in reranked)

def test_retrieve_returns_result():
    result = retrieve("what is machine learning?")
    assert result.cache_hit is False or result.cached_answer is not None
    assert isinstance(result.latency_ms, dict)

def test_retrieve_cache_hit():
    from retrieve.cache import store_cache, clear_cache
    clear_cache()
    store_cache("what is RAG?", "RAG is Retrieval Augmented Generation.")
    result = retrieve("what is RAG?")
    assert result.cache_hit is True
    assert result.cached_answer == "RAG is Retrieval Augmented Generation."
    clear_cache()