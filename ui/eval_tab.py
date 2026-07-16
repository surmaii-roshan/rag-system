"""
ui/eval_tab.py — Gradio evaluation tab with metric visualization.
"""

import json

import gradio as gr

from config import Config
from ui.components import metric_bar
from utils.logger import get_logger

log = get_logger(__name__)

_TARGETS = {
    "context_precision": 0.60,
    "context_recall": 0.50,
    "faithfulness": 0.70,
    "answer_relevancy": 0.65,
}

_LABELS = {
    "context_precision": "Context Precision",
    "context_recall":    "Context Recall",
    "faithfulness":      "Faithfulness",
    "answer_relevancy":  "Answer Relevancy",
}


def _load_results_html(path) -> str:
    if not path.exists():
        return (
            '<div style="color:#d97706; padding:10px;">'
            '⚠️ No results found. Run <code>python app.py evaluate</code> first.'
            '</div>'
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return f'<div style="color:#dc2626;">❌ Error loading results: {e}</div>'

    agg = data.get("aggregate", {})
    ts = data.get("timestamp", "unknown")
    n = data.get("n_questions", "?")

    bars = "".join(
        metric_bar(_LABELS[m], agg.get(m, 0), _TARGETS[m])
        for m in _TARGETS
    )

    src_acc = agg.get("source_accuracy")
    src_line = (
        f'<p style="font-size:12px; color:#6b7280;">Source Accuracy: '
        f'<strong>{src_acc:.4f}</strong></p>'
        if src_acc is not None else ""
    )

    return (
        f'<div style="padding:12px; background:#f9fafb; border-radius:8px; '
        f'border:1px solid #e5e7eb;">'
        f'<p style="font-size:12px; color:#6b7280;">Run on {ts} · {n} questions</p>'
        f'{bars}'
        f'{src_line}'
        f'</div>'
    )


def run_quick_eval(n_questions: int) -> str:
    """Run evaluation on first N questions and return results HTML."""
    from evaluate.runner import run_evaluation, _load_test_set

    try:
        test_set = _load_test_set()
        ids = [q["id"] for q in test_set[:int(n_questions)]]
        results = run_evaluation(save_results=True, question_ids=ids)
        agg = results["aggregate"]

        bars = "".join(
            metric_bar(_LABELS[m], agg.get(m, 0), _TARGETS[m])
            for m in _TARGETS
        )
        return (
            f'<div style="padding:12px; background:#f9fafb; border-radius:8px;">'
            f'<p style="font-size:12px; color:#6b7280;">Quick eval: {n_questions} questions</p>'
            f'{bars}'
            f'</div>'
        )
    except Exception as e:
        return f'<div style="color:#dc2626;">❌ Eval failed: {e}</div>'


def build_eval_tab() -> gr.Tab:
    """Build and return the Evaluation tab component."""
    with gr.Tab("📊 Evaluation") as tab:
        gr.Markdown("""
### Evaluation Dashboard
Run the evaluation suite to measure system quality across 4 metrics.
Full evaluation (20 questions) takes ~3 minutes due to API rate limiting.
""")

        with gr.Row():
            load_btn = gr.Button("📂 Load Latest Results", variant="secondary")
            with gr.Column():
                n_slider = gr.Slider(
                    minimum=1, maximum=20, value=5, step=1,
                    label="Questions for Quick Eval",
                )
                quick_btn = gr.Button("⚡ Quick Eval", variant="primary")

        results_display = gr.HTML(
            value=_load_results_html(Config.RESULTS_PATH),
            label="Metric Results",
        )

        baseline_display = gr.HTML(
            value=(
                '<div style="margin-top:12px; font-weight:600;">Baseline (V1)</div>'
                + _load_results_html(Config.BASELINE_PATH)
            ),
            label="V1 Baseline",
        )

        load_btn.click(
            fn=lambda: _load_results_html(Config.RESULTS_PATH),
            inputs=[],
            outputs=[results_display],
        )

        quick_btn.click(
            fn=run_quick_eval,
            inputs=[n_slider],
            outputs=[results_display],
        )

    return tab