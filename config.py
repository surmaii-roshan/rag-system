"""
config.py — Single source of truth for all constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Groq Model Tier Strategy
    # Primary: MoE model, highest TPM on Groq free tier
    PRIMARY_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
    # Fallback: dense model, used when primary 429s persistently
    FALLBACK_MODEL = "llama-3.3-70b-versatile"
    # Meta-task: cheap model for claim extraction (preserves primary budget)
    META_TASK_MODEL = "llama-3.1-8b-instant"

    # Generation Parameters
    TEMPERATURE = 0.1          # Low = grounded, not creative
    MAX_TOKENS = 1024          # Max output tokens per generation
    TPM_SAFETY_THRESHOLD = 30_000  # Drop 3rd chunk if prompt nears this

    # Chunking
    CHUNK_SIZE = 512           # Tokens per chunk
    CHUNK_OVERLAP = 64         # Overlap between consecutive chunks
    EMBED_BATCH_SIZE = 64      # Chunks per embedding batch

    # Retrieval Thresholds
    CACHE_THRESHOLD = 0.95     # Cosine similarity to trigger cache hit
    CACHE_MAX_ENTRIES = 200    # LRU eviction limit
    VECTOR_TOP_K = 10          # Candidates from vector search
    BM25_TOP_K = 10            # Candidates from BM25 search
    RRF_K = 60                 # RRF constant (standard value)
    RERANK_TOP_K = 3           # Final chunks sent to LLM

    # Generation Thresholds
    RELEVANCE_GATE = 0.3       # Skip LLM if avg chunk similarity below this
    FAITHFULNESS_THRESHOLD = 0.7   # Trigger hallucination gate below this
    CLAIM_SUPPORT_THRESHOLD = 0.5  # Cross-encoder score to mark claim supported

    # Confidence Labels
    CONFIDENCE_HIGH = 0.9
    CONFIDENCE_MEDIUM = 0.7
    # Below CONFIDENCE_MEDIUM = "Low"

    # Local Embedding / Reranker Models
    EMBED_MODEL = "all-MiniLM-L6-v2"
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # File Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    DOCUMENTS_DIR = DATA_DIR / "documents"
    CHROMA_DIR = DATA_DIR / "chroma_db"
    BM25_INDEX_PATH = DATA_DIR / "bm25_index.json"
    MANIFEST_PATH = DATA_DIR / "manifest.json"
    LOGS_DIR = BASE_DIR / "logs"
    LOG_FILE = LOGS_DIR / "rag.log"
    HALLUCINATION_LOG = LOGS_DIR / "hallucinations.jsonl"

    # ChromaDB Collection Names
    CHUNKS_COLLECTION = "document_chunks"
    CACHE_COLLECTION = "query_cache"

    # API Key
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # Eval
    EVAL_DIR = BASE_DIR / "evaluate"
    TEST_SET_PATH = EVAL_DIR / "test_set.json"
    RESULTS_PATH = EVAL_DIR / "results.json"
    BASELINE_PATH = EVAL_DIR / "baseline_v1.json"
    EVAL_SLEEP_SECONDS = 3     # Sleep between eval questions to avoid RPM limit