"""Tests for ingest/loader.py"""
from pathlib import Path
from ingest.loader import load_document, SUPPORTED_EXTENSIONS


SAMPLES = Path("tests/samples")


def test_load_txt():
    docs = load_document(SAMPLES / "sample.txt")
    assert len(docs) == 1
    assert docs[0].source_file == "sample.txt"
    assert docs[0].page_number == 1
    assert len(docs[0].content) > 100

def test_load_md():
    docs = load_document(SAMPLES / "sample.md")
    assert len(docs) == 1
    assert "RAG" in docs[0].content or "Retrieval" in docs[0].content

def test_unsupported_format(tmp_path):
    f = tmp_path / "test.xyz"
    f.write_text("hello")
    docs = load_document(f)
    assert docs == []

def test_missing_file():
    docs = load_document("nonexistent_file.txt")
    assert docs == []

def test_empty_file(tmp_path):
    f = tmp_path / "empty.txt"
    f.write_text("   \n  ")
    docs = load_document(f)
    assert docs == []

def test_supported_extensions():
    assert ".pdf" in SUPPORTED_EXTENSIONS
    assert ".docx" in SUPPORTED_EXTENSIONS
    assert ".html" in SUPPORTED_EXTENSIONS