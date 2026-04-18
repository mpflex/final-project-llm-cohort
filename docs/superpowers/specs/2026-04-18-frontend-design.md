# TacoLLM Frontend — Design Spec
**Date:** 2026-04-18
**Status:** Approved

---

## Summary

A Gradio-based frontend that sends natural language taco requests to the FastAPI backend and renders side-by-side taco recommendation cards from the base model and the LoRA-adapted model. Keeps the entire stack in Python. Calls the existing `/generate-taco` API over HTTP — the API layer remains intact and demonstrable.

---

## Architecture

Two processes:

```
User browser
    ↓  (Gradio UI at localhost:7860)
frontend/app.py   (gr.Blocks)
    ↓  (HTTP POST /generate-taco × 2 — base and lora in parallel)
backend FastAPI   (localhost:8000)
    ↓
InferencePipeline → TacoValidator → SessionMemory
```

Two source files:

| File | Responsibility |
|---|---|
| `frontend/client.py` | Thin `requests` wrapper for `/generate-taco` and `/health` |
| `frontend/app.py` | Gradio `gr.Blocks` layout, event wiring, card rendering |

A `session_id` (UUID4) is generated once per page load, stored in `gr.State`, and passed with every request so the backend's `SessionMemory` tracks constraints across turns.

---

## Layout

Single `gr.Blocks` page, vertical flow:

```
┌─────────────────────────────────────────────┐
│  TacoLLM  — constraint-aware taco recs      │  Markdown header + API status badge
├─────────────────────────────────────────────┤
│  [ text input                       ] [Send]│  Textbox + Button
│  [ Chat history (user/assistant)         ]  │  Chatbot component
├──────────────────┬──────────────────────────┤
│  Base Model      │  LoRA Model              │  2 x gr.Column
│  ┌────────────┐  │  ┌────────────┐          │
│  │ Taco Card  │  │  │ Taco Card  │          │  gr.HTML each
│  └────────────┘  │  └────────────┘          │
├─────────────────────────────────────────────┤
│  ▶ Debug / Metadata  (Accordion)            │  gr.Accordion (closed by default)
│    parsed_constraints | validation_issues   │
│    inference_attempts | session_id          │
└─────────────────────────────────────────────┘
```

### Taco Card (HTML)

Each card renders:
- Taco name as a heading
- Macro grid: calories / protein / carbs / fat
- Dietary tags as pill badges
- Spice level indicator
- Ingredients list
- Reasoning text

### Chat History

Shows user messages and a brief assistant summary ("Done! Cards updated below."). Full structured data lives in the cards, not the chat bubble.

---

## Data Flow

On **Submit:**
1. Disable Send button
2. Append user message to chat history immediately
3. Call `/generate-taco` twice in parallel via `ThreadPoolExecutor` — `model="base"` and `model="lora"` — both with the same `session_id`
4. Render both taco cards
5. Update debug accordion with metadata from the LoRA response
6. Append brief assistant message to chat history
7. Re-enable Send button

On **Page load:**
- Generate fresh `session_id` (UUID4)
- Call `/health`; show inline status badge in header ("API online" / "API offline")

---

## Error Handling

| Scenario | Behaviour |
|---|---|
| Backend unreachable | Red error card in both columns: "Backend offline — run `uvicorn app.main:app` first" |
| One model returns 422 | Error card in that column only; other card renders normally |
| Both models fail | Error card in both columns; no assistant message appended to chat |

---

## File Structure

```
frontend/
├── app.py               # Gradio Blocks layout and event wiring
├── client.py            # HTTP client for FastAPI backend
├── pyproject.toml       # gradio, requests deps
├── tests/
│   ├── test_client.py   # unit tests for client.py (requests mocked)
│   └── test_app.py      # unit tests for render_taco_card, format_debug_info
```

---

## Testing

**`test_client.py`** (mocks `requests`):
- `generate_taco` returns parsed dict on 200
- `generate_taco` raises on 422
- `generate_taco` raises on connection error
- `health_check` returns status dict

**`test_app.py`** (pure function tests, no running server):
- `render_taco_card(taco_dict)` returns HTML containing the taco name
- `render_taco_card` handles missing fields gracefully
- `format_debug_info(metadata)` returns expected string

Run with `uv run pytest` from `frontend/`.

---

## Dependencies

```
gradio>=4.0.0
requests>=2.31.0
```

Dev: `pytest>=8.2.0`, `pytest-cov`

---

## Out of Scope

- Authentication / multi-user sessions
- Deployment beyond localhost
- Eval results tab (stretch goal only)
- Mobile responsive design
