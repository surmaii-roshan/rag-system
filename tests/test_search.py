"""Tests for vector search and BM25 search."""
from retrieve.vector_search import vector_search, SearchResult
from retrieve.bm25_search import bm25_search
from config import Config


def test_vector_returns_results():
    results = vector_search("what is machine learning?")
    assert len(results) > 0
    assert all(isinstance(r, SearchResult) for r in results)

def test_vector_scores_normalized():
    results = vector_search("machine learning")
    for r in results:
        assert 0.0 <= r.score <= 1.0, f"Score out of range: {r.score}"

def test_vector_top_k():
    results = vector_search("machine learning", top_k=3)
    assert len(results) <= 3

def test_bm25_keyword_match():
    # BM25 should excel at exact keyword matches
    results = bm25_search("supervised learning algorithms")
    assert len(results) > 0
    # Top result should have the words in it
    top_text = results[0].text.lower()
    assert "supervised" in top_text or "learning" in top_text

def test_bm25_returns_search_results():
    results = bm25_search("reinforcement learning")
    assert all(isinstance(r, SearchResult) for r in results)

def test_empty_query_vector():
    results = vector_search("")
    assert isinstance(results, list)

def test_unknown_query_returns_results():
    # Even for unknown topics, vector search returns candidates
    results = vector_search("quantum entanglement in superconductors")
    assert isinstance(results, list)