"""
generate/hallucination_gate.py — Hallucination detection gate.

STUB for Phase 4. Full implementation in Phase 5.

Phase 5 will:
  - Extract atomic claims from the LLM answer (via 8b-instant)
  - Verify each claim against provided chunks (via cross-encoder)
  - Compute faithfulness score
  - Re-prompt if faithfulness < FAITHFULNESS_THRESHOLD
"""

from typing import List, Tuple

from retrieve.vector_search import SearchResult
from utils.logger import get_logger

log = get_logger(__name__)


def check_and_correct(
    query: str,
    answer: str,
    chunks: List[SearchResult],
) -> Tuple[str, float, str]:
    """
    Check answer faithfulness and re-prompt if needed.

    Phase 4 stub: passes through with perfect faithfulness score.
    Phase 5 will replace this with real claim-level verification.

    Args:
        query: Original user query.
        answer: LLM-generated answer.
        chunks: Chunks that were provided to the LLM.

    Returns:
        Tuple of (final_answer, faithfulness_score, confidence_label).
        confidence_label: "High" | "Medium" | "Low"
    """
    log.debug("Hallucination gate: STUB (Phase 5 will implement real verification)")
    return answer, 1.0, "High"