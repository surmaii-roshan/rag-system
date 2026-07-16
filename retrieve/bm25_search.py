"""
retrieve/bm25_search.py — BM25 sparse retrieval.

Builds and persists an index at ingest time.
Loads and queries the index at retrieval time (Phase 3).

Index format (bm25_index.json):
{
  "tokenized_corpus": [["token", ...], ...],
  "chunk_ids":        ["uuid", ...],
  "chunk_texts":      ["full text", ...],
  "chunk_metadatas":  [{"source_file": ..., ...}, ...]
}

The index is append-safe: re-running ingest adds new chunks
without losing existing ones, deduplicating by chunk_id.
"""

import json
import re
from typing import List

from rank_bm25 import BM25Okapi

from config import Config
from ingest.models import Chunk
from utils.logger import get_logger

log = get_logger(__name__)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())

# ─── Index Building (called during ingest) ────────────────────────────────────

def build_bm25_index(new_chunks: List[Chunk]) -> None:
    """
    Build or update the BM25 index with new chunks.

    Append-safe: loads existing index first, appends only chunks
    whose chunk_id is not already present, then saves.

    Args:
        new_chunks: Newly ingested Chunk objects to add to the index.
    """
    if not new_chunks:
        return

    # Load existing index (or start fresh)
    existing: dict = {
        "tokenized_corpus": [],
        "chunk_ids": [],
        "chunk_texts": [],
        "chunk_metadatas": [],
    }

    if Config.BM25_INDEX_PATH.exists():
        with open(Config.BM25_INDEX_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
        log.debug(f"Loaded existing BM25 index: {len(existing['chunk_ids'])} chunks")

    existing_ids = set(existing["chunk_ids"])
    added = 0

    for chunk in new_chunks:
        if chunk.chunk_id in existing_ids:
            continue

        existing["tokenized_corpus"].append(_tokenize(chunk.text))
        existing["chunk_ids"].append(chunk.chunk_id)
        existing["chunk_texts"].append(chunk.text)
        existing["chunk_metadatas"].append(
            {
                "source_file": chunk.source_file,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
            }
        )
        added += 1

    Config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(Config.BM25_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False)

    total = len(existing["chunk_ids"])
    log.info(
        f"BM25 index updated: +{added} new chunks, {total} total → {Config.BM25_INDEX_PATH}"
    )


# ─── Index Loading (used in Phase 3 retrieval) ────────────────────────────────

def load_bm25_index():
    """
    Load the persisted BM25 index from disk.

    Returns:
        (BM25Okapi, chunk_ids, chunk_texts, chunk_metadatas)

    Raises:
        FileNotFoundError: If the index doesn't exist (run ingest first).
    """
    if not Config.BM25_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"BM25 index not found at {Config.BM25_INDEX_PATH}. Run ingest first."
        )

    with open(Config.BM25_INDEX_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    bm25 = BM25Okapi(data["tokenized_corpus"])
    log.debug(f"BM25 index loaded: {len(data['chunk_ids'])} chunks")

    return (
        bm25,
        data["chunk_ids"],
        data["chunk_texts"],
        data["chunk_metadatas"],
    )


# ─── Index Querying (Phase 3) ─────────────────────────────────────────────────

def bm25_search(query: str, top_k: int = None) -> List["SearchResult"]:
    """
    Search the BM25 index for keyword-relevant chunks.

    Catches exact matches that embedding models miss:
    error codes, proper nouns, technical acronyms.

    Args:
        query: The user's question.
        top_k: Number of results (defaults to Config.BM25_TOP_K).

    Returns:
        List of SearchResult sorted by BM25 score descending.
        Returns [] if the index doesn't exist yet.
    """
    from retrieve.vector_search import SearchResult

    if top_k is None:
        top_k = Config.BM25_TOP_K

    try:
        bm25, chunk_ids, chunk_texts, chunk_metadatas = load_bm25_index()
    except FileNotFoundError:
        log.warning("BM25 index not found — run ingest first. Returning empty results.")
        return []

    tokenized_query = _tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    # Sort indices by descending score
    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:top_k]

    results: List[SearchResult] = []

    for idx in top_indices:
        if scores[idx] <= 0:
            continue

        meta = chunk_metadatas[idx]

        results.append(
            SearchResult(
                chunk_id=chunk_ids[idx],
                text=chunk_texts[idx],
                score=float(scores[idx]),
                source_file=meta.get("source_file", "unknown"),
                page_number=meta.get("page_number", 0),
                chunk_index=meta.get("chunk_index", 0),
            )
        )

    if results:
        log.debug(
            f"BM25 search: {len(results)} results, top score={results[0].score:.3f}"
        )
    else:
        log.debug("BM25: no results")

    return results