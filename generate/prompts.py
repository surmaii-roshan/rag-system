"""
generate/prompts.py — Grounded prompt templates.

The system prompt is the most important hallucination-prevention mechanism.
Every rule is there for a reason:
  - Rule 1: Prevents outside knowledge contamination
  - Rule 2: Creates verifiable citations we can parse
  - Rule 3: Handles partial knowledge honestly
  - Rule 4: Standard "I don't know" signal for the relevance gate
  - Rule 5: Prevents phantom citations
"""

from typing import List, Tuple

import tiktoken

from retrieve.vector_search import SearchResult

SYSTEM_PROMPT = """You are a precise technical assistant. Answer the user's question using ONLY the context provided below.

Rules you must follow without exception:
1. Answer ONLY from the provided context. Never use outside knowledge or training data.
2. For every factual claim you make, cite the source using this exact format: [Source: filename, Page: N]
3. If the context partially answers the question, answer what you can and explicitly state what information is missing.
4. If the context does NOT contain information relevant to the question, respond with EXACTLY this phrase: "I don't have enough information in the provided documents to answer this question."
5. Never fabricate citations. Only cite sources that actually appear in the context below."""

NO_INFORMATION_RESPONSE = "I don't have enough information in the provided documents to answer this question."


def build_grounded_prompt(
    query: str,
    chunks: List[SearchResult],
) -> Tuple[str, str]:
    """
    Build the (system_prompt, user_prompt) pair for grounded generation.

    Each chunk is labeled with its source and page so the LLM can cite correctly.

    Args:
        query: The user's question.
        chunks: Top-k retrieved chunks from the retrieval pipeline.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(
            f"[Context {i} — Source: {chunk.source_file}, Page: {chunk.page_number}]\n"
            f"{chunk.text}"
        )

    context_block = "\n\n---\n\n".join(context_parts)

    user_prompt = f"""Context:

{context_block}

---

Question: {query}

Answer (remember to cite sources using [Source: filename, Page: N]):"""

    return SYSTEM_PROMPT, user_prompt


def count_prompt_tokens(system_prompt: str, user_prompt: str) -> int:
    """
    Estimate total prompt token count using cl100k_base tokenizer.
    Used by the dynamic chunk truncation logic in generate/__init__.py.

    Args:
        system_prompt: The system message.
        user_prompt: The user message.

    Returns:
        Approximate total token count.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    # +4 tokens per message for ChatML formatting overhead
    return (
        len(enc.encode(system_prompt)) + 4
        + len(enc.encode(user_prompt)) + 4
    )