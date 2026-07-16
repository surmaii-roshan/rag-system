"""
ingest/embedder.py — Batch embedding using sentence-transformers.

Model: all-MiniLM-L6-v2 (80MB, local, no API cost)
  - 384-dimensional embeddings
  - Loaded once and cached in module scope (_model)
  - First load downloads the model from HuggingFace (~80MB, takes 1-2 min)
"""

from typing import List

from sentence_transformers import SentenceTransformer

from config import Config
from ingest.models import Chunk
from utils.logger import get_logger

log = get_logger(__name__)

# Module-level cache: model loaded once per process
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        log.info(f"Loading embedding model: {Config.EMBED_MODEL} (first load downloads ~80MB)")
        _model = SentenceTransformer(Config.EMBED_MODEL)
        log.info("Embedding model loaded and cached")
    return _model


def embed_chunks(chunks: List[Chunk]) -> List[List[float]]:
    """
    Embed a list of Chunks in batches.

    Args:
        chunks: List of Chunk objects with .text fields.

    Returns:
        List of embedding vectors (List[float]), one per chunk.
        Same order as input chunks.
    """
    if not chunks:
        return []

    model = _get_model()
    texts = [chunk.text for chunk in chunks]

    log.info(f"Embedding {len(texts)} chunks in batches of {Config.EMBED_BATCH_SIZE}...")

    # convert_to_numpy=True returns ndarray — call .tolist() for ChromaDB compatibility
    embeddings = model.encode(
        texts,
        batch_size=Config.EMBED_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    log.info(f"Embedding complete. Shape: {embeddings.shape}")
    return embeddings.tolist()


def embed_query(query: str) -> List[float]:
    """
    Embed a single query string. Used at retrieval time.

    Args:
        query: The user's question.

    Returns:
        Single embedding vector as List[float].
    """
    model = _get_model()
    embedding = model.encode(query, convert_to_numpy=True)
    return embedding.tolist()