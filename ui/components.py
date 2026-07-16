"""
ui/components.py — Reusable Gradio UI helpers and formatters.

These functions convert backend results into HTML/Markdown strings
that render nicely inside Gradio components.
"""

from typing import List, Tuple


def confidence_badge(confidence: str, faithfulness: float) -> str:
    """
    Render a colored confidence badge as HTML.

    Args:
        confidence: "High" | "Medium" | "Low"
        faithfulness: Float [0,1]

    Returns:
        HTML string with inline-styled badge.
    """
    colors = {
        "High":   ("#22c55e", "#f0fdf4"),  # green
        "Medium": ("#f59e0b", "#fffbeb"),  # amber
        "Low":    ("#ef4444", "#fef2f2"),  # red
    }
    fg, bg = colors.get(confidence, ("#6b7280", "#f9fafb"))

    return (
        f'<span style="'
        f'background:{bg}; color:{fg}; '
        f'border:1px solid {fg}; '
        f'border-radius:4px; '
        f'padding:2px 8px; '
        f'font-size:12px; '
        f'font-weight:600; '
        f'font-family:monospace;">'
        f'{confidence} ({faithfulness:.0%})'
        f'</span>'
    )


def sources_html(sources: List[Tuple[str, int]]) -> str:
    """
    Render source citations as an HTML list.

    Args:
        sources: List of (filename, page_number) tuples.

    Returns:
        HTML string or empty string if no sources.
    """
    if not sources:
        return ""

    items = "".join(
        f'<li style="margin:2px 0;">'
        f'📄 <code style="font-size:12px;">{fname}</code>'
        f' — Page {page}'
        f'</li>'
        for fname, page in sources
    )
    return (
        f'<div style="margin-top:8px; font-size:13px; color:#6b7280;">'
        f'<strong>Sources:</strong>'
        f'<ul style="margin:4px 0; padding-left:20px;">{items}</ul>'
        f'</div>'
    )


def meta_html(model: str, latency_ms: dict, from_cache: bool) -> str:
    """
    Render generation metadata as a small footer line.
    """
    total = latency_ms.get("total_ms", latency_ms.get("total", "?"))

    if from_cache:
        return (
            f'<div style="font-size:11px; color:#9ca3af; margin-top:4px;">'
            f'📦 Served from semantic cache · {total}ms'
            f'</div>'
        )

    model_short = model.split("/")[-1] if "/" in model else model

    return (
        f'<div style="font-size:11px; color:#9ca3af; margin-top:4px;">'
        f'🤖 {model_short} · ⏱ {total}ms'
        f'</div>'
    )


def metric_bar(label: str, value: float, target: float) -> str:
    """
    Render a metric as an HTML progress bar with pass/fail color.

    Args:
        label: Metric name.
        value: Current value [0,1].
        target: Pass threshold [0,1].

    Returns:
        HTML string with label and progress bar.
    """
    pct = int(value * 100)
    color = "#22c55e" if value >= target else "#ef4444"
    status = "✅" if value >= target else "❌"

    return (
        f'<div style="margin:6px 0;">'
        f'<div style="display:flex; justify-content:space-between; font-size:13px;">'
        f'<span>{status} {label}</span>'
        f'<span style="font-weight:600;">{value:.4f}</span>'
        f'</div>'
        f'<div style="background:#e5e7eb; border-radius:4px; height:8px; margin-top:3px;">'
        f'<div style="width:{pct}%; background:{color}; border-radius:4px; height:8px;"></div>'
        f'</div>'
        f'</div>'
    )