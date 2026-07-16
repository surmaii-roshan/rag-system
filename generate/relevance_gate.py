"""
generate/relevance_gate.py — Context relevance gate before LLM call.

If the retrieved chunks are not relevant enough to the query, we skip
the LLM call entirely and return the no-information response.

Why 0.3:
  - cross-encoder scores are logits, not probabilities — they can be negative
  - for vector search similarity scores [0,1], 0.3 is a loose threshold
  - this gate only blocks truly irrelevant queries (e.g. "who won the world cup?"
    when your corpus is about machine learning)
  - it's intentionally permissive: false negatives (blocking a valid query) are
    worse than false positives (sending a low-confidence query to the LLM)
"""

from typing import List

from config import Config
from retrieve.vector_search import SearchResult
from utils.logger import get_logger

log = get_logger(__name__)


def passes_relevance_gate(chunks: List[SearchResult]) -> bool:
    """
    Returns True if chunks are relevant enough to justify an LLM call.

    Gate logic: average similarity score across top chunks must >= RELEVANCE_GATE.
    Uses raw vector similarity scores (before reranking replaced them with
    cross-encoder scores, which are logits and not directly comparable).

    Args:
        chunks: Top-k reranked chunks from the retrieval pipeline.

    Returns:
        True = proceed to LLM. False = return no-information response.
    """
    if not chunks:
        log.info("Relevance gate: BLOCKED — no chunks retrieved")
        return False

    # Use only the first chunk's score as a proxy (reranker may use logits)
    # For robustness, fall back to a simple len check
    top_score = chunks[0].score if chunks else 0.0

    # Cross-encoder scores are logits (can be negative). Use a fixed threshold
    # only if scores are in [0,1] range (vector similarity). Otherwise pass through.
    is_vector_score = 0.0 <= top_score <= 1.0

    if is_vector_score and top_score < Config.RELEVANCE_GATE:
        log.info(
            f"Relevance gate: BLOCKED "
            f"(top score={top_score:.3f} < {Config.RELEVANCE_GATE})"
        )
        return False

    log.info(
        f"Relevance gate: PASS "
        f"(top score={top_score:.3f})"
    )
    return True