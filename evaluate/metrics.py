"""
evaluate/metrics.py — Four RAGAS-inspired evaluation metrics.

All metrics return float [0,1] where 1.0 is perfect.

Metric implementations:
  context_precision:  % of retrieved chunks that are relevant (cross-encoder > 0.0)
  context_recall:     % of expected keywords found in the combined retrieved text
  faithfulness:       reuses Phase 5 compute_faithfulness (claim-level)
  answer_relevancy:   cosine similarity between question and answer embeddings
"""

from typing import List

import numpy as np
from sentence_transformers import CrossEncoder

from config import Config
from ingest.embedder import embed_query
from retrieve.vector_search import SearchResult
from utils.logger import get_logger

log = get_logger(__name__)

_precision_model: CrossEncoder | None = None


def _get_precision_model() -> CrossEncoder:
    """Reuse the ms-marco cross-encoder for precision scoring."""
    global _precision_model
    if _precision_model is None:
        _precision_model = CrossEncoder(Config.RERANKER_MODEL)
    return _precision_model


# ─── Metric 1: Context Precision ─────────────────────────────────────────────

def context_precision(
    question: str,
    retrieved_chunks: List[SearchResult],
) -> float:
    """
    Measures: Are the retrieved chunks actually relevant to the question?

    Method:
      Score each chunk against the question using the cross-encoder.
      A chunk is "relevant" if its score > 0.0 (positive logit).
      Precision = relevant_chunks / total_retrieved_chunks.

    A high score means retrieval is signal-rich (not noise-heavy).

    Args:
        question: The evaluation question.
        retrieved_chunks: Top-k chunks returned by the retrieval pipeline.

    Returns:
        Precision score [0,1].
    """
    if not retrieved_chunks:
        return 0.0

    model = _get_precision_model()
    pairs = [(question, chunk.text) for chunk in retrieved_chunks]
    scores = model.predict(pairs, show_progress_bar=False)

    relevant = sum(1 for s in scores if s > 0.0)
    precision = relevant / len(retrieved_chunks)

    log.debug(f"Context precision: {precision:.3f} ({relevant}/{len(retrieved_chunks)} relevant)")
    return precision


# ─── Metric 2: Context Recall ─────────────────────────────────────────────────

def context_recall(
    answer: str,
    expected_keywords: List[str],
) -> float:
    """
    Measures: Does the answer capture the key information?

    Method:
      Check what % of expected_keywords appear (case-insensitive) in the answer.
      This is a lightweight proxy for semantic recall.

    Note: Real RAGAS uses an LLM-judge for recall. We use keyword matching
    to stay within free-tier limits. It's less nuanced but fully deterministic.

    Args:
        answer: The generated answer text.
        expected_keywords: List of keywords expected in the answer (from test set).

    Returns:
        Recall score [0,1]. Returns 1.0 for no-answer questions (vacuously true).
    """
    if not expected_keywords:
        return 1.0  # No-answer questions are vacuously "recalled"

    answer_lower = answer.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    recall = found / len(expected_keywords)

    log.debug(f"Context recall: {recall:.3f} ({found}/{len(expected_keywords)} keywords)")
    return recall


# ─── Metric 3: Faithfulness ───────────────────────────────────────────────────

def faithfulness_metric(
    answer: str,
    chunks: List[SearchResult],
) -> float:
    """
    Measures: Is the answer grounded in the retrieved context?

    Delegates to Phase 5 compute_faithfulness for consistency.
    The same cross-encoder-based claim verification is used during
    both runtime (hallucination gate) and evaluation.

    Args:
        answer: The generated answer.
        chunks: Chunks that were provided to the LLM.

    Returns:
        Faithfulness score [0,1].
    """
    from evaluate.claims import extract_claims
    from evaluate.faithfulness import compute_faithfulness

    claims = extract_claims(answer)
    if not claims:
        return 1.0  # No claims = no hallucinations

    faith, _ = compute_faithfulness(claims, chunks)
    return faith


# ─── Metric 4: Answer Relevancy ──────────────────────────────────────────────

def answer_relevancy(question: str, answer: str) -> float:
    """
    Measures: Is the answer semantically on-topic with the question?

    Method:
      Cosine similarity between question embedding and answer embedding.
      Uses the same all-MiniLM-L6-v2 model as ingestion.

    A high score means the answer is topically related to the question.
    A low score means the answer may be "I don't know" or off-topic.

    Note: Cosine similarity on sentence embeddings captures semantic
    relatedness well — "machine learning" and "subset of AI" map close.

    Args:
        question: The original question.
        answer: The generated answer.

    Returns:
        Relevancy score [0,1].
    """
    if not answer.strip() or "don't have enough information" in answer.lower():
        return 0.0  # No-info response is not "relevant" to the question

    q_emb = np.array(embed_query(question))
    a_emb = np.array(embed_query(answer))

    # Cosine similarity
    norm_q = np.linalg.norm(q_emb)
    norm_a = np.linalg.norm(a_emb)

    if norm_q == 0 or norm_a == 0:
        return 0.0

    similarity = float(np.dot(q_emb, a_emb) / (norm_q * norm_a))
    # Clamp to [0,1] (cosine can be slightly negative for unrelated text)
    return max(0.0, similarity)