"""
ingest/chunker.py — Recursive character chunker with token-based overlap.

Strategy:
  1. Try to split at paragraph breaks (\n\n) first.
  2. If a piece is still too large, fall back to line breaks (\n).
  3. Then sentence endings (". "), then spaces, then characters.
  4. After splitting, prepend the tail of the previous chunk to each chunk
     to create CHUNK_OVERLAP tokens of context continuity.
"""

import uuid
from typing import List

import tiktoken

from config import Config
from ingest.models import Chunk, Document
from utils.logger import get_logger

log = get_logger(__name__)

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _count_tokens(text: str, enc: tiktoken.Encoding) -> int:
    return len(enc.encode(text))


def _split_by_separator(text: str, separator: str) -> List[str]:
    if separator == "":
        # Last resort: split into individual characters (rare edge case)
        return list(text)
    return [p.strip() for p in text.split(separator) if p.strip()]


def _merge_splits(
    splits: List[str],
    separator: str,
    enc: tiktoken.Encoding,
    max_tokens: int,
) -> List[str]:
    """
    Greedily join splits together until adding the next one would exceed max_tokens.
    Any single split that is itself larger than max_tokens is passed through as-is
    and handled by the recursive caller.
    """
    chunks: List[str] = []
    current_parts: List[str] = []
    current_len = 0

    for split in splits:
        split_len = _count_tokens(split, enc)
        sep_len = _count_tokens(separator, enc) if current_parts else 0

        if current_len + sep_len + split_len <= max_tokens:
            current_parts.append(split)
            current_len += sep_len + split_len
        else:
            if current_parts:
                chunks.append(separator.join(current_parts))
            current_parts = [split]
            current_len = split_len

    if current_parts:
        chunks.append(separator.join(current_parts))

    return chunks


def _chunk_text(
    text: str,
    enc: tiktoken.Encoding,
    max_tokens: int,
    separators: List[str],
) -> List[str]:
    """Recursively split text so no chunk exceeds max_tokens."""
    if _count_tokens(text, enc) <= max_tokens:
        return [text]

    separator = separators[0]
    remaining = separators[1:]

    splits = _split_by_separator(text, separator)

    # If this separator doesn't appear in the text, try the next one
    if len(splits) <= 1 and remaining:
        return _chunk_text(text, enc, max_tokens, remaining)

    merged = _merge_splits(splits, separator, enc, max_tokens)

    # Any merged chunk that is still too large gets recursed
    result: List[str] = []
    for chunk in merged:
        if _count_tokens(chunk, enc) > max_tokens and remaining:
            result.extend(_chunk_text(chunk, enc, max_tokens, remaining))
        else:
            result.append(chunk)

    return result


def _add_overlap(
    chunks: List[str],
    enc: tiktoken.Encoding,
    overlap_tokens: int,
) -> List[str]:
    """
    Prepend the last `overlap_tokens` tokens of chunk[i-1] to chunk[i].
    This ensures context is not lost at chunk boundaries.
    """
    if overlap_tokens == 0 or len(chunks) <= 1:
        return chunks

    overlapped = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tokens = enc.encode(chunks[i - 1])
        if len(prev_tokens) >= overlap_tokens:
            overlap_text = enc.decode(prev_tokens[-overlap_tokens:])
        else:
            overlap_text = chunks[i - 1]
        overlapped.append(overlap_text + " " + chunks[i])

    return overlapped


def chunk_documents(documents: List[Document]) -> List[Chunk]:
    """
    Convert a list of Documents into a flat list of Chunks.

    Args:
        documents: Output from load_document().

    Returns:
        List of Chunk objects ready for embedding and storage.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    chunks: List[Chunk] = []
    global_index = 0

    for doc in documents:
        text = doc.content.strip()
        if not text:
            continue

        raw = _chunk_text(text, enc, Config.CHUNK_SIZE, _SEPARATORS)
        raw = [c.strip() for c in raw if c.strip()]
        with_overlap = _add_overlap(raw, enc, Config.CHUNK_OVERLAP)

        for chunk_text in with_overlap:
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
            chunks.append(Chunk(
                text=chunk_text,
                chunk_id=str(uuid.uuid4()),
                chunk_index=global_index,
                source_file=doc.source_file,
                page_number=doc.page_number,
                token_count=_count_tokens(chunk_text, enc),
            ))
            global_index += 1

    log.info(f"Chunked {len(documents)} document(s) → {len(chunks)} chunks")
    return chunks