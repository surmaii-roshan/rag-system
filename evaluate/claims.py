"""
evaluate/claims.py — Atomic claim extraction from LLM answers.

Uses 8b-instant (cheapest, fastest Groq model) to extract claims.
We use a different model than generation to preserve the primary model's
RPM budget for actual RAG queries.

What counts as a claim:
  - A single, verifiable factual statement
  - "Machine learning is a subset of AI"  ← claim
  - "It is used in many fields"           ← too vague, but still extracted
  - "In conclusion, ..."                  ← not a claim, ignored
"""

import json
import re
from typing import List
from utils.logger import get_logger

log = get_logger(__name__)

_CLAIM_SYSTEM = """You are a precise claim extractor. Your task is to decompose a statement into its atomic factual claims.

Rules:
1. Extract ONLY objective, verifiable statements. Skip opinions and meta-commentary.
2. Each claim must be a complete, standalone sentence.
3. Keep each claim concise (under 30 words).
4. Return ONLY a JSON array of strings. No preamble, no explanation.

Example input: "Machine learning is a subset of AI. It includes supervised, unsupervised, and reinforcement learning."
Example output: ["Machine learning is a subset of AI.", "Machine learning includes supervised learning.", "Machine learning includes unsupervised learning.", "Machine learning includes reinforcement learning."]"""


def extract_claims(answer: str) -> List[str]:
    from generate.groq_client import generate_meta
    """
    Extract atomic factual claims from an LLM answer.

    Uses llama-3.1-8b-instant via generate_meta() to preserve
    the primary model's rate limit budget.

    Args:
        answer: The full LLM-generated answer text.

    Returns:
        List of atomic claim strings.
        Returns [] if extraction fails (non-blocking).
    """
    if not answer.strip():
        return []

    # Skip no-information responses
    if "don't have enough information" in answer.lower():
        return []

    prompt = f"Extract atomic claims from this text:\n\n{answer}"

    try:
        raw = generate_meta(prompt=prompt, system_prompt=_CLAIM_SYSTEM)
        claims = _parse_json_claims(raw)
        log.debug(f"Extracted {len(claims)} claims from answer")
        return claims

    except Exception as e:
        log.warning(f"Claim extraction failed: {e}. Returning empty claim list.")
        return []


def _parse_json_claims(raw: str) -> List[str]:
    """
    Parse JSON array from LLM output.
    Handles cases where the LLM wraps the JSON in markdown code blocks.
    """
    # Strip markdown code blocks if present
    raw = re.sub(r"```(?:json)?\s*", "", raw)
    raw = raw.strip()

    # Find the JSON array
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if not match:
        log.warning(f"No JSON array found in claim extraction output: {raw[:100]}")
        return []

    try:
        claims = json.loads(match.group(0))
        if isinstance(claims, list):
            return [str(c).strip() for c in claims if isinstance(c, str) and c.strip()]
        return []
    except json.JSONDecodeError as e:
        log.warning(f"JSON parse error in claim extraction: {e}")
        return []