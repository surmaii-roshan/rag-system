"""
End-to-end ingestion test.
Warning: This test actually calls ChromaDB and sentence-transformers.
First run will download the embedding model (~80MB).
"""
import shutil
from pathlib import Path
import pytest
from ingest import ingest_directory
from ingest.store import get_collection_count


TEST_DOCS_DIR = Path("tests/samples")


def test_ingest_produces_chunks(tmp_path):
    """Ingesting real documents should create chunks in ChromaDB."""
    # Use a temp chroma dir so we don't pollute the real one
    import chromadb
    from config import Config

    # Patch chroma dir for this test
    original = Config.CHROMA_DIR
    Config.CHROMA_DIR = tmp_path / "test_chroma"
    Config.MANIFEST_PATH = tmp_path / "manifest.json"
    Config.BM25_INDEX_PATH = tmp_path / "bm25_index.json"

    try:
        stats = ingest_directory(TEST_DOCS_DIR)
        assert stats["processed"] >= 1
        assert stats["chunks_created"] >= 1
        assert stats["skipped"] == 0
    finally:
        Config.CHROMA_DIR = original
        Config.MANIFEST_PATH = original.parent / "manifest.json"
        Config.BM25_INDEX_PATH = original.parent / "bm25_index.json"


def test_dedup_on_second_run(tmp_path):
    """Running ingest twice on the same directory should skip all files."""
    from config import Config

    Config.CHROMA_DIR = tmp_path / "test_chroma"
    Config.MANIFEST_PATH = tmp_path / "manifest.json"
    Config.BM25_INDEX_PATH = tmp_path / "bm25_index.json"

    try:
        stats1 = ingest_directory(TEST_DOCS_DIR)
        stats2 = ingest_directory(TEST_DOCS_DIR)

        assert stats2["processed"] == 0
        assert stats2["skipped"] == stats1["processed"]
        assert stats2["chunks_created"] == 0
    finally:
        Config.CHROMA_DIR = Path("data/chroma_db")
        Config.MANIFEST_PATH = Path("data/manifest.json")
        Config.BM25_INDEX_PATH = Path("data/bm25_index.json")