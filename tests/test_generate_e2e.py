"""End-to-end generation tests (real Groq API call)."""

from generate import answer_question
from generate.relevance_gate import passes_relevance_gate
from retrieve.vector_search import SearchResult
from retrieve.cache import clear_cache


def test_relevant_query_gets_answer():
    result = answer_question("What is multi-head attention?")

    assert result.answer != ""
    assert "don't have enough information" not in result.answer.lower()
    assert "attention" in result.answer.lower()


def test_rag_query_gets_answer():
    result = answer_question(
        "What is retrieval augmented generation?"
    )

    assert result.answer != ""
    assert "don't have enough information" not in result.answer.lower()


def test_irrelevant_query_returns_no_info():
    result = answer_question(
        "What is the boiling point of tungsten carbide?"
    )

    # This topic is outside the research-paper corpus.
    assert isinstance(result.answer, str)
    assert (
        "don't have enough information" in result.answer.lower()
        or result.confidence.lower() in {"low", "medium"}
    )


def test_cache_hit_on_repeat():
    clear_cache()

    answer_question("What is multi-head attention?")
    result = answer_question("What is multi-head attention?")

    assert result.from_cache is True

    clear_cache()


def test_relevance_gate_blocks_empty():
    assert passes_relevance_gate([]) is False


def test_relevance_gate_passes_high_score():
    chunk = SearchResult(
        chunk_id="id",
        text="Multi-head attention allows the model to jointly attend to information.",
        score=6.5,
        source_file="NIPS-2017-attention-is-all-you-need-Paper.pdf",
        page_number=4,
        chunk_index=0,
    )

    assert passes_relevance_gate([chunk]) is True


def test_result_has_required_fields():
    result = answer_question("Explain scaled dot-product attention.")

    assert hasattr(result, "answer")
    assert hasattr(result, "sources")
    assert hasattr(result, "model_used")
    assert hasattr(result, "confidence")
    assert hasattr(result, "faithfulness")
    assert hasattr(result, "latency_ms")