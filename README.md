# Production RAG System (Free Tier)

A production-grade Retrieval-Augmented Generation system running entirely
on free-tier infrastructure. Built with Groq, ChromaDB, and local embeddings.

## Status
🚧 **In development** — V1 implementation in progress.

## Stack
- **LLM**: Groq (llama-4-scout-17b / llama-3.3-70b / llama-3.1-8b)
- **Embeddings**: all-MiniLM-L6-v2 (local)
- **Vector store**: ChromaDB (local)
- **Sparse search**: BM25 (rank-bm25)
- **Reranker**: ms-marco-MiniLM-L-6-v2 (local)
- **UI**: Gradio

## Setup
```bash
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cp .env.example .env   # then paste your GROQ_API_KEY