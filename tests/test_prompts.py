"""Tests for prompt building and citation parsing."""

from retrieve.vector_search import SearchResult
from generate.prompts import (
    build_grounded_prompt,
    count_prompt_tokens,
    NO_INFORMATION_RESPONSE,
)
from generate.citation import extract_citations, verify_citations


def _make_chunk(text, source="NIPS-2017-attention-is-all-you-need-Paper.pdf", page=4):
    return SearchResult("id1", text, 0.9, source, page, 0)


# ---------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------

def test_prompt_contains_source_label():
    chunks = [
        _make_chunk(
            "Multi-head attention allows the model to jointly attend to information from different representation subspaces."
        )
    ]

    _, user_prompt = build_grounded_prompt(
        "What is multi-head attention?",
        chunks,
    )

    assert "Source: NIPS-2017-attention-is-all-you-need-Paper.pdf" in user_prompt
    assert "Page: 4" in user_prompt


def test_prompt_contains_query():
    chunks = [
        _make_chunk(
            "Scaled dot-product attention computes attention weights using queries, keys and values."
        )
    ]

    _, user_prompt = build_grounded_prompt(
        "Explain scaled dot-product attention.",
        chunks,
    )

    assert "Explain scaled dot-product attention." in user_prompt


def test_prompt_contains_context():
    context = (
        "Multi-head attention allows the model to jointly attend "
        "to information from different representation subspaces."
    )

    chunks = [_make_chunk(context)]

    _, user_prompt = build_grounded_prompt(
        "What is multi-head attention?",
        chunks,
    )

    assert context in user_prompt


def test_token_count_positive():
    chunks = [
        _make_chunk(
            "Scaled dot-product attention computes attention scores."
        )
    ]

    sys_p, usr_p = build_grounded_prompt(
        "What is scaled dot-product attention?",
        chunks,
    )

    assert count_prompt_tokens(sys_p, usr_p) > 0


# ---------------------------------------------------------------------
# Citation parsing
# ---------------------------------------------------------------------

def test_citation_extraction():
    answer = (
        "Multi-head attention improves representation learning "
        "[Source: NIPS-2017-attention-is-all-you-need-Paper.pdf, Page: 4]. "
        "RAG combines retrieval and generation "
        "[Source: NeurIPS-2020-retrieval-augmented-generation-for-knowledge-intensive-nlp-tasks-Paper.pdf, Page: 2]."
    )

    citations = extract_citations(answer)

    assert (
        "NIPS-2017-attention-is-all-you-need-Paper.pdf",
        4,
    ) in citations

    assert (
        "NeurIPS-2020-retrieval-augmented-generation-for-knowledge-intensive-nlp-tasks-Paper.pdf",
        2,
    ) in citations


def test_phantom_citation_detected():
    answer = (
        "Attention mechanism "
        "[Source: NIPS-2017-attention-is-all-you-need-Paper.pdf, Page: 4] "
        "and "
        "[Source: imaginary-paper.pdf, Page: 99]."
    )

    chunks = [
        _make_chunk(
            "Multi-head attention allows the model to jointly attend...",
            "NIPS-2017-attention-is-all-you-need-Paper.pdf",
            4,
        )
    ]

    valid, phantom = verify_citations(answer, chunks)

    assert (
        "NIPS-2017-attention-is-all-you-need-Paper.pdf",
        4,
    ) in valid

    assert (
        "imaginary-paper.pdf",
        99,
    ) in phantom


def test_no_citations_in_answer():
    answer = "Transformers rely heavily on self-attention."

    assert extract_citations(answer) == []


def test_no_information_response_constant():
    assert "don't have enough information" in NO_INFORMATION_RESPONSE.lower()