"""Tests for the evaluation runner (fast subset)."""
import json
from evaluate.runner import run_evaluation


def test_test_set_loads():
    from evaluate.runner import _load_test_set
    test_set = _load_test_set()
    assert len(test_set) == 20
    for item in test_set:
        assert "id" in item
        assert "question" in item
        assert "has_answer" in item


def test_quick_eval_single_question():
    """Run eval on just q01 to verify the runner works."""
    results = run_evaluation(save_results=False, question_ids=["q01"])
    assert results["n_questions"] == 1
    agg = results["aggregate"]
    assert 0.0 <= agg["context_precision"] <= 1.0
    assert 0.0 <= agg["context_recall"] <= 1.0
    assert 0.0 <= agg["faithfulness"] <= 1.0
    assert 0.0 <= agg["answer_relevancy"] <= 1.0


def test_results_structure():
    results = run_evaluation(save_results=False, question_ids=["q01", "q13"])
    assert "aggregate" in results
    assert "questions" in results
    assert len(results["questions"]) == 2
    for q in results["questions"]:
        assert "metrics" in q
        assert "answer_preview" in q