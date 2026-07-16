"""
evaluate/faithfulness.py — Claim-level faithfulness scoring.

Faithfulness measures whether each claim made in the answer is
directly supported by the retrieved context.

Approach:
  For each claim:
    Score it against each provided chunk using the cross-encoder.
    If max(scores across all chunks) >= CLAIM_SUPPORT_THRESHOLD → supported.

  faithfulness = supported_claims / total_claims

Why cross-encoder for claim verification:
  - The same ms-marco cross-encoder used in reranking is well-calibrated
    for "does this passage support this statement?" tasks.
  - It's already loaded in memory from reranking — no extra model cost.
  - Thresholding at 0.5 logit score was empirically validated.

Faithfulness score interpretation:
  1.0         → All claims grounded in context
  0.7 - 1.0   → High (acceptable)
  0.5 - 0.7   → Medium (borderline, re-prompt triggered)
  < 0.5       → Low (significant hallucination, definitely re-prompt)
"""

from typing import List, Tuple

from sentence_transformers import CrossEncoder

from config import Config
from retrieve.vector_search import SearchResult
from utils.logger import get_logger

log = get_logger(__name__)

_verifier: CrossEncoder | None = None


def _get_verifier() -> CrossEncoder:
    """Reuse the same model as retrieve/reranker.py — already in memory."""
    global _verifier
    if _verifier is None:
        log.info(f"Loading claim verifier: {Config.RERANKER_MODEL}")
        _verifier = CrossEncoder(Config.RERANKER_MODEL)
    return _verifier


def score_claim(claim: str, chunks: List[SearchResult]) -> float:
    """
    Score a single claim against the provided chunks.

    Args:
        claim: A single atomic factual claim (from extract_claims()).
        chunks: The chunks provided to the LLM during generation.

    Returns:
        Maximum cross-encoder score across all chunks.
        Higher = better supported by context.
    """
    if not chunks:
        return 0.0

    verifier = _get_verifier()
    pairs = [(claim, chunk.text) for chunk in chunks]
    scores = verifier.predict(pairs, show_progress_bar=False)
    return float(max(scores))


def compute_faithfulness(
    claims: List[str],
    chunks: List[SearchResult],
) -> Tuple[float, List[Tuple[str, float, bool]]]:
    """
    Compute faithfulness score for a list of claims against context.

    Args:
        claims: Atomic claims extracted from the LLM answer.
        chunks: Chunks provided to the LLM.

    Returns:
        Tuple of:
          - faithfulness_score: float [0,1]
          - claim_details: List of (claim, score, is_supported) for logging
    """
    if not claims:
        log.debug("No claims to verify — returning faithfulness=1.0")
        return 1.0, []

    claim_details: List[Tuple[str, float, bool]] = []
    supported = 0

    for claim in claims:
        score = score_claim(claim, chunks)
        is_supported = score >= Config.CLAIM_SUPPORT_THRESHOLD
        claim_details.append((claim, score, is_supported))
        if is_supported:
            supported += 1
        else:
            log.debug(f"Unsupported claim (score={score:.3f}): {claim[:80]}")

    faithfulness = supported / len(claims)
    log.info(
        f"Faithfulness: {faithfulness:.2f} "
        f"({supported}/{len(claims)} claims supported)"
    )
    return faithfulness, claim_details


def assign_confidence(faithfulness: float) -> str:
    """
    Convert faithfulness score to a human-readable confidence label.

    Args:
        faithfulness: Float [0,1] from compute_faithfulness().

    Returns:
        "High" | "Medium" | "Low"
    """
    if faithfulness >= Config.CONFIDENCE_HIGH:
        return "High"
    elif faithfulness >= Config.CONFIDENCE_MEDIUM:
        return "Medium"
    else:
        return "Low"