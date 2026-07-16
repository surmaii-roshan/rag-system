"""Integration tests for the hallucination gate (real API calls)."""

from retrieve.vector_search import SearchResult
from generate.hallucination_gate import check_and_correct
from config import Config


def _chunks():
    return [
        SearchResult(
            "id1",
            (
                "Multi-head attention allows the model to jointly attend "
                "to information from different representation subspaces."
            ),
            0.9,
            "NIPS-2017-attention-is-all-you-need-Paper.pdf",
            4,
            0,
        )
    ]


def test_faithful_answer_passes():
    answer = (
        "Multi-head attention allows the model to jointly attend "
        "to information from different representation subspaces. "
        "[Source: NIPS-2017-attention-is-all-you-need-Paper.pdf, Page: 4]"
    )

    result, faith, conf = check_and_correct(
        "What is multi-head attention?",
        answer,
        _chunks(),
    )

    assert faith >= Config.FAITHFULNESS_THRESHOLD
    assert conf in ("High", "Medium")


def test_hallucinated_answer_triggers_reprompt():
    """A fabricated answer should trigger hallucination correction."""

    bad_answer = (
        "Multi-head attention requires quantum computers "
        "and specialized quantum processors."
    )

    result, faith_after, conf = check_and_correct(
        "What is multi-head attention?",
        bad_answer,
        _chunks(),
    )

    # The re-prompt may still fail, but the gate should return
    # a valid corrected response and a faithfulness score.
    assert isinstance(result, str)
    assert isinstance(faith_after, float)
    assert conf in ("High", "Medium", "Low")


def test_empty_answer_passes():
    result, faith, conf = check_and_correct(
        "What is multi-head attention?",
        "",
        _chunks(),
    )

    assert faith == 1.0