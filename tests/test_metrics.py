"""Tests for evaluation metrics (no API calls)."""
import numpy as np
from retrieve.vector_search import SearchResult
from evaluate.metrics import context_recall, answer_relevancy


def _chunk(text, score=0.5):
    return SearchResult("id", text, score, "test.txt", 1, 0)


class TestContextRecall:
    def test_all_keywords_found(self):
        answer = "Machine learning is a subset of artificial intelligence."
        assert context_recall(answer, ["machine learning", "subset"]) == 1.0

    def test_no_keywords_found(self):
        answer = "The weather is sunny today."
        assert context_recall(answer, ["machine learning", "subset"]) == 0.0

    def test_partial_keywords(self):
        answer = "Machine learning is interesting."
        recall = context_recall(answer, ["machine learning", "subset", "AI"])
        assert 0 < recall < 1.0

    def test_empty_keywords_returns_one(self):
        # No-answer questions have empty keywords — vacuously recalled
        assert context_recall("any answer", []) == 1.0

    def test_case_insensitive(self):
        answer = "MACHINE LEARNING is a subset of AI"
        assert context_recall(answer, ["machine learning", "subset"]) == 1.0


class TestAnswerRelevancy:
    def test_high_relevancy(self):
        q = "What is machine learning?"
        a = "Machine learning is a subset of artificial intelligence."
        score = answer_relevancy(q, a)
        assert score > 0.7, f"Expected > 0.7, got {score}"

    def test_no_info_response_is_zero(self):
        q = "What is ML?"
        a = "I don't have enough information in the provided documents to answer this question."
        assert answer_relevancy(q, a) == 0.0

    def test_empty_answer_is_zero(self):
        assert answer_relevancy("Question?", "") == 0.0

    def test_output_in_range(self):
        score = answer_relevancy("What is AI?", "Artificial intelligence is broad.")
        assert 0.0 <= score <= 1.0