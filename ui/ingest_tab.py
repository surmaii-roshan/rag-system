"""
ui/ingest_tab.py — Gradio ingest tab: file upload → ingest pipeline.
"""

import shutil
import tempfile
from pathlib import Path
from typing import Iterator

import gradio as gr

from ingest import ingest_directory
from ingest.loader import SUPPORTED_EXTENSIONS
from ingest.store import get_collection_count
from utils.logger import get_logger

log = get_logger(__name__)

_ACCEPT_TYPES = [ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS]


def _status_html(message: str, color: str = "#374151") -> str:
    return (
        f'<div style="padding:10px; border-radius:6px; '
        f'background:#f9fafb; border:1px solid #e5e7eb; '
        f'font-size:13px; color:{color};">'
        f'{message}</div>'
    )


def run_ingest(files) -> str:
    """Handler for the ingest button click."""
    if not files:
        return _status_html("⚠️ No files selected. Upload at least one document.", "#d97706")

    # Copy uploaded files to a temp directory and ingest
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        for file in files:
            src = Path(file.name)
            if src.suffix.lower() not in SUPPORTED_EXTENSIONS:
                log.warning(f"Skipping unsupported file: {src.name}")
                continue
            shutil.copy(src, tmp_path / src.name)

        try:
            stats = ingest_directory(tmp_path)
        except Exception as e:
            log.error(f"Ingest failed: {e}")
            return _status_html(f"❌ Ingest error: {e}", "#dc2626")

    total_chunks = get_collection_count()
    color = "#166534" if stats["processed"] > 0 else "#d97706"

    return _status_html(
        f"✅ Ingest complete!<br>"
        f"&nbsp;&nbsp;• Files processed: <strong>{stats['processed']}</strong><br>"
        f"&nbsp;&nbsp;• Files skipped (duplicates): <strong>{stats['skipped']}</strong><br>"
        f"&nbsp;&nbsp;• New chunks created: <strong>{stats['chunks_created']}</strong><br>"
        f"&nbsp;&nbsp;• Total chunks in store: <strong>{total_chunks}</strong>",
        color,
    )


def get_corpus_status() -> str:
    """Show current corpus stats."""
    try:
        count = get_collection_count()
        color = "#166534" if count > 0 else "#d97706"
        return _status_html(
            f"📚 Corpus contains <strong>{count}</strong> chunks in ChromaDB.",
            color,
        )
    except Exception as e:
        return _status_html(f"❌ Could not reach ChromaDB: {e}", "#dc2626")


def build_ingest_tab() -> gr.Tab:
    """Build and return the Ingest tab component."""
    with gr.Tab("📄 Ingest") as tab:
        gr.Markdown("""
### Document Ingestion
Upload your documents and they will be chunked, embedded, and indexed automatically.
**Supported formats**: PDF, DOCX, TXT, MD, HTML

Duplicate files are detected via SHA-256 hashing and skipped automatically.
""")
        corpus_status = gr.HTML(value=get_corpus_status())

        with gr.Row():
            file_input = gr.File(
                label="Upload Documents",
                file_count="multiple",
                file_types=[f".{ext}" for ext in _ACCEPT_TYPES],
                elem_id="file-uploader",
            )

        with gr.Row():
            ingest_btn = gr.Button("🚀 Ingest Documents", variant="primary", scale=2)
            refresh_btn = gr.Button("🔄 Refresh Status", variant="secondary", scale=1)

        result_html = gr.HTML(label="Ingest Result")

        ingest_btn.click(
            fn=run_ingest,
            inputs=[file_input],
            outputs=[result_html],
        )

        refresh_btn.click(
            fn=get_corpus_status,
            inputs=[],
            outputs=[corpus_status],
        )

    return tab