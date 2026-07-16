"""
evaluate/runner.py — Evaluation runner with rate-limit throttling.

Runs all test questions sequentially with time.sleep(EVAL_SLEEP_SECONDS)
between each one to avoid hitting Groq's free-tier RPM limit.

Progress is written to a results JSON file that can be compared across runs.
"""

import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from config import Config
from evaluate.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness_metric,
)
from generate import answer_question
from retrieve import retrieve
from utils.logger import get_logger

log = get_logger(__name__)


def _load_test_set() -> List[Dict[str, Any]]:
    if not Config.TEST_SET_PATH.exists():
        raise FileNotFoundError(
            f"Test set not found at {Config.TEST_SET_PATH}. "
            f"Create evaluate/test_set.json first."
        )
    with open(Config.TEST_SET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def run_evaluation(
    save_results: bool = True,
    question_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run the full evaluation suite.

    Computes 4 metrics per question, then averages across all questions.
    Sleeps EVAL_SLEEP_SECONDS (3s) between questions to stay under Groq RPM.

    Args:
        save_results: If True, writes results to Config.RESULTS_PATH.
        question_ids: If set, only runs the specified question IDs (for testing).

    Returns:
        Dict with per-question results and aggregate metrics.
    """
    test_set = _load_test_set()

    if question_ids:
        test_set = [q for q in test_set if q["id"] in question_ids]

    log.info(f"Starting evaluation: {len(test_set)} questions")
    log.info(f"Throttle: {Config.EVAL_SLEEP_SECONDS}s between questions")

    question_results = []
    all_precision = []
    all_recall = []
    all_faithfulness = []
    all_relevancy = []

    for i, item in enumerate(test_set, start=1):
        qid = item["id"]
        question = item["question"]
        expected_source = item.get("expected_source")
        expected_keywords = item.get("answer_contains", [])
        has_answer = item.get("has_answer", True)

        print(f"\n[{i:02d}/{len(test_set):02d}] {qid}: {question[:60]}...")

        # ── Retrieve ──────────────────────────────────────────────────────────
        retrieval = retrieve(question)

        # ── Compute context precision ─────────────────────────────────────────
        if retrieval.cache_hit or not retrieval.chunks:
            precision = 1.0 if retrieval.cache_hit else 0.0
        else:
            precision = context_precision(question, retrieval.chunks)
        all_precision.append(precision)

        # ── Generate answer ───────────────────────────────────────────────────
        result = answer_question(question)
        answer = result.answer

        # ── Compute context recall ────────────────────────────────────────────
        recall = context_recall(answer, expected_keywords)
        all_recall.append(recall)

        # ── Compute faithfulness ──────────────────────────────────────────────
        if result.from_cache or not retrieval.chunks:
            faith = result.faithfulness
        else:
            faith = faithfulness_metric(answer, retrieval.chunks)
        all_faithfulness.append(faith)

        # ── Compute answer relevancy ──────────────────────────────────────────
        relevancy = answer_relevancy(question, answer)
        all_relevancy.append(relevancy)

        # ── Check source accuracy ─────────────────────────────────────────────
        source_correct: Optional[bool] = None
        if expected_source and retrieval.chunks:
            retrieved_sources = {c.source_file for c in retrieval.chunks}
            source_correct = expected_source in retrieved_sources
        elif not has_answer:
            source_correct = True  # No-answer questions don't need a source

        q_result = {
            "id": qid,
            "question": question,
            "answer_preview": answer[:200],
            "has_answer": has_answer,
            "expected_source": expected_source,
            "source_correct": source_correct,
            "model_used": result.model_used,
            "from_cache": result.from_cache,
            "metrics": {
                "context_precision": round(precision, 4),
                "context_recall": round(recall, 4),
                "faithfulness": round(faith, 4),
                "answer_relevancy": round(relevancy, 4),
            },
        }
        question_results.append(q_result)

        print(
            f"    P={precision:.2f}  R={recall:.2f}  F={faith:.2f}  Rel={relevancy:.2f}"
            f"  {'✅' if source_correct else '❌' if source_correct is False else '—'}"
            f"  [{result.model_used.split('/')[-1] if '/' in result.model_used else result.model_used}]"
        )

        # ── Throttle (skip sleep after last question) ─────────────────────────
        if i < len(test_set):
            log.debug(f"Sleeping {Config.EVAL_SLEEP_SECONDS}s to avoid RPM limit...")
            time.sleep(Config.EVAL_SLEEP_SECONDS)

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    source_accuracy = None
    checked = [r for r in question_results if r["source_correct"] is not None]
    if checked:
        source_accuracy = sum(1 for r in checked if r["source_correct"]) / len(checked)

    aggregate = {
        "context_precision": round(sum(all_precision) / len(all_precision), 4),
        "context_recall": round(sum(all_recall) / len(all_recall), 4),
        "faithfulness": round(sum(all_faithfulness) / len(all_faithfulness), 4),
        "answer_relevancy": round(sum(all_relevancy) / len(all_relevancy), 4),
        "source_accuracy": round(source_accuracy, 4) if source_accuracy else None,
    }

    full_results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_questions": len(test_set),
        "aggregate": aggregate,
        "questions": question_results,
    }

    if save_results:
        Config.EVAL_DIR.mkdir(exist_ok=True)
        with open(Config.RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(full_results, f, indent=2)
        log.info(f"Results saved to {Config.RESULTS_PATH}")

    return full_results


def save_baseline(results: Dict[str, Any]) -> None:
    """
    Save current results as the V1 baseline for future comparison.
    Only call this once after the first clean evaluation run.
    """
    Config.EVAL_DIR.mkdir(exist_ok=True)
    with open(Config.BASELINE_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log.info(f"Baseline saved to {Config.BASELINE_PATH}")


def compare_to_baseline(current_results: Dict[str, Any]) -> None:
    """
    Print a side-by-side comparison of current results vs baseline.
    """
    if not Config.BASELINE_PATH.exists():
        print("No baseline found. Run --save-baseline first.")
        return

    with open(Config.BASELINE_PATH, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    curr_agg = current_results["aggregate"]
    base_agg = baseline["aggregate"]

    print("\n📊 Metric Comparison vs Baseline")
    print("─" * 55)
    print(f"{'Metric':<22} {'Baseline':>10} {'Current':>10} {'Delta':>10}")
    print("─" * 55)

    metrics = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
    for m in metrics:
        b = base_agg.get(m, 0)
        c = curr_agg.get(m, 0)
        delta = c - b
        arrow = "↑" if delta > 0.005 else ("↓" if delta < -0.005 else "→")
        print(f"{m:<22} {b:>10.4f} {c:>10.4f} {f'{arrow}{abs(delta):.4f}':>10}")

    print("─" * 55)