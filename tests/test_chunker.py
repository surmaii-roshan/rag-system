"""Tests for ingest/chunker.py"""
from ingest.loader import load_document
from ingest.chunker import chunk_documents
from ingest.models import Document
from config import Config
from pathlib import Path


def test_chunk_count_reasonable():
    docs = load_document(Path("tests/samples/sample.txt"))
    chunks = chunk_documents(docs)
    # Small doc should produce at least 1 chunk
    assert len(chunks) >= 1

def test_no_empty_chunks():
    docs = load_document(Path("tests/samples/sample.txt"))
    chunks = chunk_documents(docs)
    for c in chunks:
        assert c.text.strip() != ""

def test_chunk_token_size():
    # Tokens should be <= CHUNK_SIZE + CHUNK_OVERLAP (overlap adds extra tokens)
    docs = load_document(Path("tests/samples/sample.txt"))
    chunks = chunk_documents(docs)
    max_allowed = Config.CHUNK_SIZE + Config.CHUNK_OVERLAP + 10  # small buffer
    for c in chunks:
        assert c.token_count <= max_allowed, f"Chunk too large: {c.token_count} tokens"

def test_chunk_metadata():
    docs = load_document(Path("tests/samples/sample.txt"))
    chunks = chunk_documents(docs)
    for c in chunks:
        assert c.source_file == "sample.txt"
        assert c.chunk_id != ""
        assert c.chunk_index >= 0

def test_chunk_indices_sequential():
    docs = load_document(Path("tests/samples/sample.txt"))
    chunks = chunk_documents(docs)
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))

def test_empty_document():
    chunks = chunk_documents([Document(content="   ", source_file="test.txt")])
    assert chunks == []