"""
ui/app_ui.py — Main Gradio application with Chat + Ingest + Eval tabs.

Theme: Soft, clean, professional.
Chat supports multi-turn conversation history display.
Each answer shows confidence badge, sources, and model metadata.
"""

import gradio as gr

from generate import answer_question
from ui.components import confidence_badge, meta_html, sources_html
from ui.ingest_tab import build_ingest_tab
from ui.eval_tab import build_eval_tab
from utils.logger import get_logger

log = get_logger(__name__)

CSS = """
#rag-title { text-align: center; padding: 8px 0; }
#chat-container { height: 520px; }
#answer-meta { margin-top: 6px; }
.confidence-high { color: #166534; }
.confidence-low  { color: #dc2626; }
footer { display: none !important; }
"""


def _build_response_html(result) -> str:
    """Combine answer, confidence badge, sources, and meta into one HTML block."""
    badge = confidence_badge(result.confidence, result.faithfulness)
    src = sources_html(result.sources)
    meta = meta_html(result.model_used, result.latency_ms, result.from_cache)

    return (
        f'<div style="padding:10px 0;">'
        f'{badge}<br/>'
        f'{src}'
        f'{meta}'
        f'</div>'
    )


def chat_fn(message: str, history: list) -> tuple[list, str, str]:
    """
    Main chat handler.

    Args:
        message: The user's question.
        history: List of {"role": ..., "content": ...} dicts (Gradio 5 format).

    Returns:
        Updated history, cleared input, metadata HTML.
    """
    if not message.strip():
        return history, "", ""

    log.info(f"UI query: {message[:80]}")

    try:
        result = answer_question(message)
    except Exception as e:
        log.error(f"answer_question failed: {e}")
        result = None

    if result is None:
        bot_content = "❌ An error occurred. Check your API key and try again."
        meta = ""
    else:
        bot_content = result.answer
        meta = _build_response_html(result)

    history = history or []
    history.append({"role": "user",      "content": message})
    history.append({"role": "assistant", "content": bot_content})

    return history, "", meta


def clear_fn() -> tuple[list, str, str]:
    """Clear chat history."""
    return [], "", ""


def build_app() -> gr.Blocks:
    """Build and return the full Gradio app."""
    with gr.Blocks(
        title="Production RAG System",
        theme=gr.themes.Soft(),
        css=CSS,
    ) as app:

        gr.HTML("""
<div id="rag-title">
  <h1 style="font-size:24px; font-weight:700; color:#1e293b; margin:0;">
    🔍 Production RAG System
  </h1>
  <p style="color:#64748b; font-size:14px; margin:4px 0 0 0;">
    Free-tier · Groq LLM · ChromaDB · Hybrid Retrieval · Hallucination Detection
  </p>
</div>
""")

        with gr.Tabs():
            # ── Tab 1: Chat ────────────────────────────────────────────────────
            with gr.Tab("💬 Chat"):
                chatbot = gr.Chatbot(
                    label="RAG Assistant",
                    type="messages",
                    height=480,
                    elem_id="chat-container",
                    show_label=False,
                    avatar_images=(
                        None,  # user avatar (None = default)
                        None,  # bot avatar
                    ),
                )

                answer_meta = gr.HTML(
                    value="",
                    elem_id="answer-meta",
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Ask a question about your documents...",
                        show_label=False,
                        scale=9,
                        container=False,
                        elem_id="msg-input",
                    )
                    send_btn = gr.Button(
                        "Send",
                        variant="primary",
                        scale=1,
                        min_width=80,
                    )

                with gr.Row():
                    clear_btn = gr.Button(
                        "🗑️ Clear",
                        variant="secondary",
                        size="sm",
                    )
                    gr.Markdown(
                        "*Answers cite sources with `[Source: file, Page: N]` format.*",
                        elem_id="hint-text",
                    )

                # Wire up events
                send_btn.click(
                    fn=chat_fn,
                    inputs=[msg_input, chatbot],
                    outputs=[chatbot, msg_input, answer_meta],
                )

                msg_input.submit(
                    fn=chat_fn,
                    inputs=[msg_input, chatbot],
                    outputs=[chatbot, msg_input, answer_meta],
                )

                clear_btn.click(
                    fn=clear_fn,
                    inputs=[],
                    outputs=[chatbot, msg_input, answer_meta],
                )

            # ── Tab 2: Ingest ──────────────────────────────────────────────────
            build_ingest_tab()

            # ── Tab 3: Evaluation ──────────────────────────────────────────────
            build_eval_tab()

    return app


def launch(share: bool = False, port: int = 7860) -> None:
    """Launch the Gradio app."""
    app = build_app()
    log.info(f"Launching Gradio UI on port {port}")
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=share,
        show_error=True,
    )