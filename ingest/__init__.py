"""
ingest/__init__.py — Ingestion pipeline orchestrator.

Chains: load → dedup → chunk → embed → store (ChromaDB) → index (BM25)

Usage:
    from ingest import ingest_directory
    stats = ingest_directory("./data/documents")
"""

from pathlib import Path
from typing import List

from tqdm import tqdm

from config import Config
from ingest.dedup import (
    hash_file, is_duplicate, load_manifest,
    save_manifest, update_manifest,
)
from ingest.embedder import embed_chunks
from ingest.chunker import chunk_documents
from ingest.loader import load_document, SUPPORTED_EXTENSIONS
from ingest.models import Chunk
from ingest.store import store_chunks
from retrieve.bm25_search import build_bm25_index
from utils.logger import get_logger

log = get_logger(__name__)


def ingest_directory(directory: str | Path) -> dict:
    """
    Ingest all supported documents from a directory.

    Pipeline per file:
        hash → dedup check → load → chunk → (collect all) →
        embed all → store ChromaDB → build BM25 → save manifest

    Embeddings are batched across ALL new files together for efficiency.

    Args:
        directory: Path to folder containing documents.

    Returns:
        dict with keys: processed, skipped, chunks_created
    """
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Document directory not found: {directory}")

    # Collect all supported files
    files = sorted([
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ])

    if not files:
        log.warning(f"No supported documents found in {directory}")
        return {"processed": 0, "skipped": 0, "chunks_created": 0}

    log.info(f"Found {len(files)} file(s) in {directory}")

    manifest = load_manifest()
    all_new_chunks: List[Chunk] = []
    stats = {"processed": 0, "skipped": 0, "chunks_created": 0}

    # ── Phase 1: Load & chunk (fast, local) ──────────────────────────────────
    for file_path in tqdm(files, desc="Loading & chunking"):
        file_hash = hash_file(file_path)

        if is_duplicate(file_hash, manifest):
            log.info(f"Skipping duplicate: {file_path.name}")
            stats["skipped"] += 1
            continue

        documents = load_document(file_path)
        if not documents:
            log.warning(f"No content extracted from {file_path.name} — skipping")
            continue

        chunks = chunk_documents(documents)
        if not chunks:
            log.warning(f"No chunks produced from {file_path.name} — skipping")
            continue

        all_new_chunks.extend(chunks)
        update_manifest(file_hash, file_path.name, manifest)
        stats["processed"] += 1
        log.info(f"  {file_path.name}: {len(documents)} doc(s), {len(chunks)} chunks")

    if not all_new_chunks:
        log.info("No new chunks to embed — all files were duplicates or empty")
        return stats

    stats["chunks_created"] = len(all_new_chunks)

    # ── Phase 2: Embed all new chunks together (GPU-friendly batching) ────────
    log.info(f"Embedding {len(all_new_chunks)} chunks...")
    embeddings = embed_chunks(all_new_chunks)

    # ── Phase 3: Store in ChromaDB ────────────────────────────────────────────
    store_chunks(all_new_chunks, embeddings)

    # ── Phase 4: Update BM25 index ────────────────────────────────────────────
    build_bm25_index(all_new_chunks)

    # ── Phase 5: Persist manifest ─────────────────────────────────────────────
    save_manifest(manifest)

    log.info(
        f"Ingestion done: {stats['processed']} file(s) processed, "
        f"{stats['skipped']} skipped, "
        f"{stats['chunks_created']} chunks created"
    )
    return stats