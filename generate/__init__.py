"""
generate/__init__.py — Generation pipeline orchestrator.

Two public functions:
  answer(query, chunks)  → GenerationResult
    Pure generation from retrieved chunks. No cache involved.
    Called by answer_question() and by evaluation pipeline.

  answer_question(query) → GenerationResult
    Full pipeline: retrieve + generate + cache store.
    Called by CLI, Gradio UI, and demo script.

Dynamic chunk truncation:
  If the full prompt (system + 3 chunks) nears the TPM safety threshold,
  the 3rd chunk is dropped before sending to Groq. This keeps us under
  the free-tier token-per-minute limit without failing the request.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config import Config
from generate.citation import verify_citations
from generate.groq_client import generate as _generate
from generate.hallucination_gate import check_and_correct
from generate.prompts import (
    NO_INFORMATION_RESPONSE,
    build_grounded_prompt,
    count_prompt_tokens,
)
from generate.relevance_gate import passes_relevance_gate
from retrieve.vector_search import SearchResult
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class GenerationResult:
    """Complete result from the generation pipeline."""
    answer: str
    sources: List[tuple]            # List of (filename, page_number) valid citations
    model_used: str                 # Which Groq model served this request
    latency_ms: Dict[str, int] = field(default_factory=dict)
    confidence: str = "High"        # "High" | "Medium" | "Low"
    faithfulness: float = 1.0       # [0,1] — Phase 5 populates this
    chunks_used: int = 0            # How many chunks were in the prompt
    from_cache: bool = False        # True if served from semantic cache
    skipped: bool = False           # True if relevance gate blocked LLM call


def answer(query: str, chunks: List[SearchResult]) -> GenerationResult:
    """
    Generate a grounded answer from retrieved chunks.

    Pipeline:
      1. Relevance gate (skip LLM if context score < 0.3)
      2. Dynamic chunk truncation (drop 3rd if prompt > TPM_SAFETY_THRESHOLD)
      3. Build grounded prompt with [Source: file, Page: N] labels
      4. Generate with Groq (Scout 17B → 70B fallback)
      5. Parse and verify citations (phantom detection)
      6. Hallucination gate (stub in Phase 4, real in Phase 5)
      7. Return GenerationResult

    Args:
        query: User's question.
        chunks: Top-k reranked chunks from retrieve().

    Returns:
        GenerationResult with answer, sources, confidence, and timings.
    """
    t_start = time.time()
    timings: Dict[str, int] = {}

    # ── Step 1: Relevance gate ──────────────────────────────────────────────
    if not passes_relevance_gate(chunks):
        return GenerationResult(
            answer=NO_INFORMATION_RESPONSE,
            sources=[],
            model_used="none",
            latency_ms={"total_ms": round((time.time() - t_start) * 1000)},
            confidence="Low",
            faithfulness=0.0,
            chunks_used=0,
            skipped=True,
        )

    # ── Step 2: Dynamic chunk truncation ────────────────────────────────────
    active_chunks = list(chunks)
    system_prompt, user_prompt = build_grounded_prompt(query, active_chunks)
    token_count = count_prompt_tokens(system_prompt, user_prompt)

    if token_count > Config.TPM_SAFETY_THRESHOLD and len(active_chunks) > 2:
        log.warning(
            f"Prompt is {token_count} tokens (>{Config.TPM_SAFETY_THRESHOLD}). "
            f"Dropping 3rd chunk to stay within TPM limit."
        )
        active_chunks = active_chunks[:2]
        system_prompt, user_prompt = build_grounded_prompt(query, active_chunks)
        token_count = count_prompt_tokens(system_prompt, user_prompt)

    log.info(f"Final prompt: {token_count} tokens, {len(active_chunks)} chunks")

    # ── Step 3: Generate ────────────────────────────────────────────────────
    t_gen = time.time()
    try:
        raw_answer, model_used = _generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
        )
    except RuntimeError as e:
        log.error(f"Generation failed: {e}")
        return GenerationResult(
            answer="Generation failed. Please check your API key and rate limits.",
            sources=[],
            model_used="none",
            latency_ms={"total_ms": round((time.time() - t_start) * 1000)},
            confidence="Low",
            faithfulness=0.0,
            chunks_used=len(active_chunks),
        )
    timings["generation_ms"] = round((time.time() - t_gen) * 1000)

    # ── Step 4: Citation verification ───────────────────────────────────────
    valid_citations, phantom_citations = verify_citations(raw_answer, active_chunks)

    # ── Step 5: Hallucination gate (stub Phase 4, real Phase 5) ────────────
    t_faith = time.time()
    final_answer, faithfulness, confidence = check_and_correct(
        query=query,
        answer=raw_answer,
        chunks=active_chunks,
    )
    timings["faithfulness_ms"] = round((time.time() - t_faith) * 1000)

    timings["total_ms"] = round((time.time() - t_start) * 1000)

    log.info(
        f"Generation complete: model={model_used}, "
        f"faithfulness={faithfulness:.2f}, confidence={confidence}, "
        f"total={timings['total_ms']}ms"
    )

    return GenerationResult(
        answer=final_answer,
        sources=valid_citations,
        model_used=model_used,
        latency_ms=timings,
        confidence=confidence,
        faithfulness=faithfulness,
        chunks_used=len(active_chunks),
    )


def answer_question(query: str) -> GenerationResult:
    """
    Full pipeline entry point: retrieve → generate → cache.

    This is what CLI commands, Gradio UI, and demo.py should call.

    Cache storage:
      - Only caches answers that passed the relevance gate (not skipped)
      - Only caches answers with faithfulness >= FAITHFULNESS_THRESHOLD
        (prevents caching hallucinated answers)

    Args:
        query: User's question.

    Returns:
        GenerationResult. Check .from_cache to know if it was a cache hit.
    """
    from retrieve import retrieve
    from retrieve.cache import store_cache

    # ── Check semantic cache ──────────────────────────────────────────────
    retrieval = retrieve(query)

    if retrieval.cache_hit:
        log.info("Served from semantic cache")
        return GenerationResult(
            answer=retrieval.cached_answer,
            sources=[],
            model_used="cache",
            latency_ms=retrieval.latency_ms,
            confidence="High",
            faithfulness=1.0,
            chunks_used=0,
            from_cache=True,
        )

    # ── Generate from retrieved chunks ────────────────────────────────────
    result = answer(query, retrieval.chunks)
    result.latency_ms.update(retrieval.latency_ms)

    # ── Store to cache if trustworthy ─────────────────────────────────────
    if (
        not result.skipped
        and result.faithfulness >= Config.FAITHFULNESS_THRESHOLD
        and result.answer != NO_INFORMATION_RESPONSE
    ):
        store_cache(query, result.answer)

    return result