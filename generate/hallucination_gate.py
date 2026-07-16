"""
generate/hallucination_gate.py — Full hallucination gate implementation.

Replaces the Phase 4 stub with real claim-level verification and re-prompting.

Two-stage re-prompt strategy:
  Stage 1: Original answer has faithfulness < FAITHFULNESS_THRESHOLD (0.7)
           → Issue grounded re-prompt: "only state what is directly stated"
           → Re-run verification on re-prompted answer
  Stage 2: Accept re-prompted answer regardless of score.
           (We don't loop indefinitely — one re-prompt attempt only)
"""

from typing import List, Tuple

from config import Config
from evaluate.claims import extract_claims
from evaluate.faithfulness import assign_confidence, compute_faithfulness
from evaluate.hallucination_logger import log_hallucination
from retrieve.vector_search import SearchResult
from utils.logger import get_logger

log = get_logger(__name__)

_REPROMPT_SYSTEM = """You are a strictly factual assistant. A previous answer contained statements not fully supported by the provided context.

Rules:
1. ONLY state facts that are DIRECTLY AND EXPLICITLY written in the context below.
2. Do NOT infer, extrapolate, or generalize beyond what is stated.
3. Cite every claim with [Source: filename, Page: N].
4. If you cannot answer strictly from the context, say: "I don't have enough information in the provided documents to answer this question."
5. Be more conservative than your previous answer."""


def check_and_correct(
    query: str,
    answer: str,
    chunks: List[SearchResult],
) -> Tuple[str, float, str]:
    """
    Verify answer faithfulness and re-prompt once if below threshold.

    Args:
        query: Original user query.
        answer: LLM-generated answer to verify.
        chunks: Chunks provided to the LLM during generation.

    Returns:
        Tuple of (final_answer, faithfulness_score, confidence_label).
    """
    # ── Extract claims ──────────────────────────────────────────────────────
    claims = extract_claims(answer)

    if not claims:
        # No claims to verify → no hallucination risk
        log.debug("No claims extracted — skipping faithfulness check")
        return answer, 1.0, "High"

    # ── Compute faithfulness ─────────────────────────────────────────────────
    faithfulness, claim_details = compute_faithfulness(claims, chunks)
    confidence = assign_confidence(faithfulness)

    # ── Check gate ───────────────────────────────────────────────────────────
    if faithfulness >= Config.FAITHFULNESS_THRESHOLD:
        log.info(f"Hallucination gate: PASS (faithfulness={faithfulness:.2f})")
        return answer, faithfulness, confidence

    # ── Below threshold — log and re-prompt ──────────────────────────────────
    log.warning(
        f"Hallucination gate: TRIGGERED (faithfulness={faithfulness:.2f} < {Config.FAITHFULNESS_THRESHOLD}). "
        f"Issuing re-prompt."
    )

    reprompted_answer = _reprompt(query, answer, chunks, claim_details)

    # ── Verify re-prompted answer ────────────────────────────────────────────
    reprompt_claims = extract_claims(reprompted_answer)
    if reprompt_claims:
        faithfulness_after, _ = compute_faithfulness(reprompt_claims, chunks)
    else:
        faithfulness_after = 1.0

    confidence_after = assign_confidence(faithfulness_after)

    # ── Log the hallucination event ──────────────────────────────────────────
    log_hallucination(
        query=query,
        original_answer=answer,
        claim_details=claim_details,
        faithfulness_before=faithfulness,
        faithfulness_after=faithfulness_after,
        reprompt_issued=True,
    )

    log.info(
        f"Re-prompt result: faithfulness {faithfulness:.2f} → {faithfulness_after:.2f}"
    )

    return reprompted_answer, faithfulness_after, confidence_after


def _reprompt(
    query: str,
    original_answer: str,
    chunks: List[SearchResult],
    claim_details: List[Tuple[str, float, bool]],
) -> str:
    """
    Issue a corrective re-prompt to the LLM.

    Includes:
      - The original context
      - The original answer
      - Which specific claims were unsupported
      - Strict grounding instructions
    """
    from generate.groq_client import generate as _generate
    from generate.prompts import build_grounded_prompt

    unsupported = [
        f"  - \"{claim}\" (confidence score: {score:.2f})"
        for claim, score, supported in claim_details
        if not supported
    ]

    _, context_prompt = build_grounded_prompt(query, chunks)

    reprompt_user = f"""{context_prompt}

---

Your previous answer contained the following claims that could not be verified in the context:

{chr(10).join(unsupported)}

Please provide a corrected answer that only states facts directly supported by the context above.
Do not repeat the unsupported claims. If you cannot answer, say so explicitly."""

    try:
        reprompted, _ = _generate(
            prompt=reprompt_user,
            system_prompt=_REPROMPT_SYSTEM,
        )
        return reprompted
    except RuntimeError as e:
        log.error(f"Re-prompt generation failed: {e}")
        return original_answer  # Fall back to original if re-prompt fails