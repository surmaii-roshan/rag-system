"""
evaluate/hallucination_logger.py — JSONL log for hallucination events.

Every time faithfulness < FAITHFULNESS_THRESHOLD, we log:
  - The original query
  - The original answer
  - Each unsupported claim with its score
  - Whether a re-prompt was issued
  - The faithfulness score after re-prompting

This log is invaluable for post-hoc analysis and prompt tuning.
"""

import json
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from config import Config
from utils.logger import get_logger

log = get_logger(__name__)


def log_hallucination(
    query: str,
    original_answer: str,
    claim_details: List[Tuple[str, float, bool]],
    faithfulness_before: float,
    faithfulness_after: Optional[float] = None,
    reprompt_issued: bool = False,
) -> None:
    """
    Append a hallucination event to the JSONL log file.

    Args:
        query: The original user query.
        original_answer: The first LLM answer (before re-prompting).
        claim_details: List of (claim, score, is_supported) tuples.
        faithfulness_before: Faithfulness of original_answer.
        faithfulness_after: Faithfulness after re-prompting (if applicable).
        reprompt_issued: Whether a re-prompt was issued.
    """
    Config.LOGS_DIR.mkdir(exist_ok=True)

    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "faithfulness_before": round(faithfulness_before, 4),
        "faithfulness_after": round(faithfulness_after, 4) if faithfulness_after is not None else None,
        "reprompt_issued": reprompt_issued,
        "unsupported_claims": [
            {"claim": claim, "score": round(score, 4)}
            for claim, score, supported in claim_details
            if not supported
        ],
        "answer_preview": original_answer[:300],
    }

    try:
        with open(Config.HALLUCINATION_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
        log.debug(f"Hallucination event logged to {Config.HALLUCINATION_LOG}")
    except Exception as e:
        log.error(f"Failed to write hallucination log: {e}")


def summarize_hallucination_log() -> dict:
    """
    Read the JSONL log and return summary statistics.

    Returns:
        Dict with total_events, reprompt_rate, avg_faithfulness_before,
        avg_faithfulness_after, most_common_unsupported.
    """
    if not Config.HALLUCINATION_LOG.exists():
        return {"total_events": 0}

    events = []
    with open(Config.HALLUCINATION_LOG, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not events:
        return {"total_events": 0}

    reprompt_count = sum(1 for e in events if e.get("reprompt_issued"))
    faith_before = [e["faithfulness_before"] for e in events]
    faith_after = [e["faithfulness_after"] for e in events if e.get("faithfulness_after") is not None]

    return {
        "total_events": len(events),
        "reprompt_rate": reprompt_count / len(events),
        "avg_faithfulness_before": sum(faith_before) / len(faith_before),
        "avg_faithfulness_after": sum(faith_after) / len(faith_after) if faith_after else None,
    }