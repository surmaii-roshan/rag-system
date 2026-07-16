"""Tests for faithfulness scoring."""

from retrieve.vector_search import SearchResult
from evaluate.faithfulness import (
    score_claim,
    compute_faithfulness,
    assign_confidence,
)
from config import Config


def _chunk(text, score=0.9):
    return SearchResult(
        "id1",
        text,
        score,
        "NIPS-2017-attention-is-all-you-need-Paper.pdf",
        4,
        0,
    )


def test_grounded_claim_scores_high():
    chunk = _chunk(
        "Multi-head attention allows the model to jointly attend "
        "to information from different representation subspaces."
    )

    score = score_claim(
        "Multi-head attention allows the model to jointly attend "
        "to information from different representation subspaces.",
        [chunk],
    )

    assert score > Config.CLAIM_SUPPORT_THRESHOLD


def test_unrelated_claim_scores_low():
    chunk = _chunk(
        "Multi-head attention allows the model to jointly attend "
        "to information from different representation subspaces."
    )

    score = score_claim(
        "The Eiffel Tower is located in Paris.",
        [chunk],
    )

    assert score < Config.CLAIM_SUPPORT_THRESHOLD


def test_faithfulness_all_supported():
    chunk = _chunk(
        "Scaled dot-product attention computes attention weights "
        "using queries, keys, and values."
    )

    claims = [
        "Scaled dot-product attention computes attention weights "
        "using queries, keys, and values."
    ]

    faith, details = compute_faithfulness(claims, [chunk])

    assert faith == 1.0
    assert details[0][2] is True


def test_faithfulness_empty_chunks():
    faith, _ = compute_faithfulness(
        ["Multi-head attention uses multiple heads."],
        [],
    )

    assert faith == 0.0


def test_faithfulness_empty_claims():
    faith, details = compute_faithfulness(
        [],
        [_chunk("Transformer models use self-attention.")],
    )

    assert faith == 1.0
    assert details == []


def test_assign_confidence_labels():
    assert assign_confidence(1.0) == "High"
    assert assign_confidence(0.95) == "High"
    assert assign_confidence(0.75) == "Medium"
    assert assign_confidence(0.50) == "Low"
    assert assign_confidence(0.0) == "Low"