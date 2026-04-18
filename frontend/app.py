# frontend/app.py
"""
TacoLLM — Gradio frontend.

Public helpers (used by tests):
    render_taco_card(taco, title) -> str  (HTML)
    format_debug_info(metadata)   -> str  (Markdown)

Entry point:
    build_app() -> gr.Blocks
"""

import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

import requests


def render_taco_card(taco: Optional[Dict[str, Any]], title: str = "") -> str:
    """Return styled HTML for a taco dict. Returns an error card if taco is None."""
    if taco is None:
        return (
            '<div style="border:1px solid #f87171;border-radius:8px;padding:16px;'
            'background:#fef2f2;">'
            '<p style="color:#dc2626;font-weight:bold;">Failed to generate taco</p>'
            '<p style="color:#dc2626;">Check that the backend is running at localhost:8000</p>'
            "</div>"
        )

    name = taco.get("name", "Unnamed Taco")
    calories = taco.get("calories", "—")
    protein = taco.get("protein", "—")
    carbs = taco.get("carbs", "—")
    fat = taco.get("fat", "—")
    ingredients = taco.get("ingredients", [])
    tags = taco.get("dietary_tags", [])
    spice = taco.get("spice_level", "—")
    reasoning = taco.get("reasoning", "")

    spice_display = {"mild": "mild 🌶", "medium": "medium 🌶🌶", "hot": "hot 🌶🌶🌶"}.get(
        spice, spice
    )

    tag_pills = " ".join(
        f'<span style="background:#d1fae5;color:#065f46;padding:2px 8px;'
        f'border-radius:12px;font-size:0.75rem;">{t}</span>'
        for t in tags
    )
    ingredients_html = "".join(f"<li>{i}</li>" for i in ingredients)

    macro = (
        '<div style="display:grid;grid-template-columns:repeat(4,1fr);'
        'gap:8px;margin-bottom:12px;">'
    )
    for value, label in [(calories, "cal"), (protein, "protein"), (carbs, "carbs"), (fat, "fat")]:
        suffix = "" if label == "cal" else "g"
        macro += (
            f'<div style="text-align:center;background:#f9fafb;border-radius:6px;padding:6px;">'
            f'<div style="font-size:1rem;font-weight:bold;">{value}{suffix}</div>'
            f'<div style="font-size:0.7rem;color:#6b7280;">{label}</div></div>'
        )
    macro += "</div>"

    return (
        '<div style="border:1px solid #e5e7eb;border-radius:8px;padding:16px;background:#fff;">'
        f'<h3 style="margin:0 0 8px 0;font-size:1.1rem;">{name}</h3>'
        f"{macro}"
        f'<div style="margin-bottom:8px;">{tag_pills}</div>'
        f'<div style="margin-bottom:8px;font-size:0.85rem;">Spice: {spice_display}</div>'
        f'<ul style="margin:0 0 8px 0;padding-left:20px;font-size:0.85rem;">{ingredients_html}</ul>'
        f'<p style="margin:0;font-size:0.8rem;color:#6b7280;font-style:italic;">{reasoning}</p>'
        "</div>"
    )


def format_debug_info(metadata: Dict[str, Any]) -> str:
    """Return a Markdown string summarising inference metadata."""
    lines = [
        f"**Session ID:** `{metadata.get('session_id', 'unknown')}`",
        f"**Model:** `{metadata.get('model', '—')}`",
        f"**Valid JSON:** `{metadata.get('valid_json', '—')}`",
        f"**Inference Attempts:** `{metadata.get('inference_attempts', '—')}`",
        "",
        "**Parsed Constraints:**",
        f"```json\n{json.dumps(metadata.get('parsed_constraints', {}), indent=2)}\n```",
    ]
    issues = metadata.get("validation_issues", [])
    if issues:
        lines.append("**Validation Issues:**")
        for issue in issues:
            lines.append(f"- {issue}")
    else:
        lines.append("**Validation Issues:** none")
    return "\n".join(lines)


import gradio as gr

from client import generate_taco, health_check


def get_health_badge() -> str:
    """Returns an HTML badge indicating backend status."""
    try:
        data = health_check()
        status = data.get("status", "unknown")
        model = data.get("active_model", "—")
        color = "#10b981" if status == "ok" else "#f59e0b"
        return (
            f'<span style="color:{color};font-weight:bold;">&#9679; API online</span>'
            f" (model: {model})"
        )
    except Exception:
        return (
            '<span style="color:#ef4444;font-weight:bold;">&#9679; API offline</span>'
            " — run <code>uvicorn app.main:app --app-dir ../backend</code> first"
        )


def _call_model(
    message: str, session_id: str, model: str
) -> tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Returns (taco_dict_or_None, metadata_dict). Never raises."""
    try:
        response = generate_taco(message, session_id, model=model)
        return response.get("data"), response.get("metadata", {})
    except requests.HTTPError as exc:
        return None, {"error": str(exc), "model": model}
    except requests.ConnectionError:
        return None, {"error": "connection_error", "model": model}


def submit(
    message: str,
    history: list,
    session_id: str,
) -> tuple:
    """
    Gradio event handler for the Send button and Enter key.

    Returns:
        (history, base_card_html, lora_card_html, debug_md, session_id, cleared_input)
    """
    if not message.strip():
        return history, "", "", "", session_id, ""

    history = history + [[message, None]]

    with ThreadPoolExecutor(max_workers=2) as executor:
        base_future = executor.submit(_call_model, message, session_id, "base")
        lora_future = executor.submit(_call_model, message, session_id, "lora")
        base_taco, base_meta = base_future.result()
        lora_taco, lora_meta = lora_future.result()

    base_html = render_taco_card(base_taco)
    lora_html = render_taco_card(lora_taco)
    debug_md = format_debug_info(lora_meta)

    history[-1][1] = "Done! Taco cards updated below."

    return history, base_html, lora_html, debug_md, session_id, ""


def build_app() -> "gr.Blocks":
    """Construct and return the Gradio Blocks application."""
    with gr.Blocks(title="TacoLLM", theme=gr.themes.Soft()) as demo:
        session_state = gr.State(value=lambda: str(uuid.uuid4()))

        gr.HTML(
            '<div style="display:flex;align-items:center;justify-content:space-between;'
            'padding:8px 0;">'
            '<h1 style="margin:0;">TacoLLM</h1>'
            f'<div>{get_health_badge()}</div>'
            "</div>"
            '<p style="color:#6b7280;margin:0 0 16px 0;">'
            "Constraint-aware taco recommendations &mdash; base model vs. LoRA side by side."
            "</p>"
        )

        chatbot = gr.Chatbot(label="Conversation", height=220)

        with gr.Row():
            msg_input = gr.Textbox(
                placeholder='Try: "High protein taco under 400 calories, no dairy"',
                label="Your request",
                scale=9,
                show_label=False,
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Base Model")
                base_card = gr.HTML(value="<p style='color:#9ca3af'>Waiting for input...</p>")
            with gr.Column():
                gr.Markdown("### LoRA Model")
                lora_card = gr.HTML(value="<p style='color:#9ca3af'>Waiting for input...</p>")

        with gr.Accordion("Debug / Metadata", open=False):
            debug_out = gr.Markdown(value="_Submit a request to see metadata._")

        outputs = [chatbot, base_card, lora_card, debug_out, session_state, msg_input]

        send_btn.click(
            fn=submit,
            inputs=[msg_input, chatbot, session_state],
            outputs=outputs,
        )
        msg_input.submit(
            fn=submit,
            inputs=[msg_input, chatbot, session_state],
            outputs=outputs,
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
