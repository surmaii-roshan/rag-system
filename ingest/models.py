"""
ingest/models.py — Core data structures for the ingestion pipeline.

Document: raw extracted content from a file (one per page for PDFs).
Chunk: a text segment ready for embedding and storage.
"""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Document:
    """Raw text extracted from a source file."""
    content: str
    source_file: str
    page_number: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A tokenized, sized segment of a Document ready for embedding."""
    text: str
    chunk_id: str        # UUID — unique key in ChromaDB
    chunk_index: int     # Global position across all chunks ingested
    source_file: str
    page_number: int
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)