"""
retrieve/vector_search.py — Semantic vector search via ChromaDB.

SearchResult is the universal data structure flowing through the
entire retrieval pipeline (vector → BM25 → RRF → reranker → LLM).
"""

import time
from dataclasses import dataclass
from typing import List

from config import Config
from ingest.embedder import embed_query
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class SearchResult:
    """Universal result object used across vector search, BM25, RRF, and reranking."""
    chunk_id: str
    text: str
    score: float        # Meaning changes per stage: similarity → BM25 → RRF → cross-encoder
    source_file: str
    page_number: int
    chunk_index: int


def vector_search(query: str, top_k: int = None) -> List[SearchResult]:
    """
    Search ChromaDB for the top_k most semantically similar chunks.

    Args:
        query: The user's question.
        top_k: Number of results to return (defaults to Config.VECTOR_TOP_K).

    Returns:
        List of SearchResult sorted by similarity score descending.
    """
    from ingest.store import get_collection

    if top_k is None:
        top_k = Config.VECTOR_TOP_K

    collection = get_collection()
    count = collection.count()

    if count == 0:
        log.warning("ChromaDB collection is empty. Run ingest first.")
        return []

    # Clamp top_k to available documents
    effective_k = min(top_k, count)

    embedding = embed_query(query)

    t0 = time.time()
    results = collection.query(
        query_embeddings=[embedding],
        n_results=effective_k,
        include=["documents", "distances", "metadatas"],
    )
    elapsed = round((time.time() - t0) * 1000)

    search_results: List[SearchResult] = []
    for i, chunk_id in enumerate(results["ids"][0]):
        # ChromaDB returns cosine distance [0,2] — convert to similarity [0,1]
        distance = results["distances"][0][i]
        similarity = max(0.0, 1.0 - distance)

        meta = results["metadatas"][0][i]
        search_results.append(SearchResult(
            chunk_id=chunk_id,
            text=results["documents"][0][i],
            score=similarity,
            source_file=meta.get("source_file", "unknown"),
            page_number=meta.get("page_number", 0),
            chunk_index=meta.get("chunk_index", 0),
        ))

    log.debug(
        f"Vector search ({elapsed}ms): {len(search_results)} results, "
        f"top score={search_results[0].score:.3f}" if search_results else "no results"
    )
    return search_results