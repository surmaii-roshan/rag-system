"""
ingest/store.py — ChromaDB persistent storage for chunks and their embeddings.

ChromaDB 1.x (Rust core) API used here:
  - chromadb.PersistentClient(path=str)  ← path must be str, not Path
  - client.get_or_create_collection(name, metadata)
  - collection.upsert(ids, embeddings, documents, metadatas)
  - collection.count()
"""

from datetime import datetime, timezone
from typing import List

import chromadb

from config import Config
from ingest.models import Chunk
from utils.logger import get_logger

log = get_logger(__name__)

# Module-level singletons: one client + one collection per process
_client = None
_collection = None
# ChromaDB hard limit per upsert call
_CHROMA_BATCH_LIMIT = 500


def _get_collection():
    """Lazily initialise the ChromaDB client and collection."""
    global _client, _collection
    if _collection is None:
        Config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(Config.CHROMA_DIR))
        _collection = _client.get_or_create_collection(
            name=Config.CHUNKS_COLLECTION,
            metadata={"hnsw:space": "cosine"},   # cosine similarity for semantic search
        )
        log.info(
            f"ChromaDB collection '{Config.CHUNKS_COLLECTION}' "
            f"ready ({_collection.count()} existing chunks)"
        )
    return _collection


def store_chunks(chunks: List[Chunk], embeddings: List[List[float]]) -> None:
    """
    Upsert chunks + embeddings into ChromaDB.
    Splits into batches of 500 to stay within ChromaDB's per-call limit.

    Args:
        chunks: List of Chunk objects (provides IDs and metadata).
        embeddings: Parallel list of embedding vectors from embed_chunks().
    """
    if not chunks:
        log.warning("store_chunks called with empty list — nothing to do")
        return

    collection = _get_collection()
    now = datetime.now(timezone.utc).isoformat()

    ids = [c.chunk_id for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [
        {
            "source_file": c.source_file,
            "chunk_index": c.chunk_index,
            "page_number": c.page_number,
            "token_count": c.token_count,
            "ingested_at": now,
        }
        for c in chunks
    ]

    total = len(chunks)
    stored = 0

    for start in range(0, total, _CHROMA_BATCH_LIMIT):
        end = min(start + _CHROMA_BATCH_LIMIT, total)
        collection.upsert(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
        stored += end - start
        log.debug(f"Stored batch {start}:{end} ({stored}/{total})")

    log.info(f"Stored {total} chunks in ChromaDB (total now: {collection.count()})")


def get_collection_count() -> int:
    """Return the total number of chunks currently in ChromaDB."""
    return _get_collection().count()


def get_collection():
    """Expose the ChromaDB collection for use by retrieve/vector_search.py."""
    return _get_collection()