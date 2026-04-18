# Architecture

## System diagram

```
User browser
    ↓  (Gradio UI — http://127.0.0.1:7860)
frontend/app.py   (gr.Blocks)
    ↓  (HTTP POST /generate-taco × 2, parallel via ThreadPoolExecutor)
    ↓  (HTTP GET /health on page load)
backend/app/main.py   (FastAPI — http://localhost:8000)
    ↓
ConstraintParser  →  SessionMemory
    ↓
InferencePipeline
    ├── base model  (meta-llama/Llama-3.2-3B-Instruct)
    └── lora model  (base + checkpoints/tacollm-lora-v1)
    ↓
TacoValidator
    ↓
API response → Gradio HTML card
```

## Component responsibilities

### `frontend/app.py`

- Renders a single `gr.Blocks` page with a chat input, two parallel HTML taco cards (base vs. LoRA), and a collapsible debug accordion
- Generates a UUID `session_id` per page load, stored in `gr.State`, passed with every request
- Calls `/generate-taco` twice in parallel (base + lora) using `ThreadPoolExecutor`
- Calls `/health` on startup and renders an inline status badge

### `frontend/client.py`

- Thin `requests` wrapper for `/health` and `/generate-taco`
- Raises on non-2xx responses; caller handles errors and renders error cards

### `backend/app/main.py`

- FastAPI app with CORS middleware
- `/health`, `/generate-taco`, `/session/{id}`, `/evaluate` routes
- Singletons: `InferencePipeline`, `ConstraintParser`, `TacoValidator`, `SessionMemory`

### `backend/app/inference.py`

- Loads `meta-llama/Llama-3.2-3B-Instruct` on startup
- Lazy-loads LoRA adapter (`backend/checkpoints/tacollm-lora-v1/`) on first `model="lora"` request
- Sampling: `temperature=0.3`, `top_p=0.9`, `max_new_tokens=512`, `repetition_penalty=1.1`
- Retry logic: up to 2 attempts with a corrective prompt on JSON parse failure

### `backend/app/parser.py`

- Regex + keyword-based constraint extractor
- Returns a structured dict: `{max_calories, high_protein, no_dairy, vegan, spice_level, ...}`

### `backend/app/validator.py`

- Checks output against constraints: calorie thresholds, ingredient exclusions (dairy, meat keywords), dietary tag consistency

### `backend/app/memory.py`

- In-memory session store keyed by `session_id`
- Merges new constraints on each turn; current message always wins over stored preferences

### `backend/evaluation/`

- `metrics.py`: JSON validity, field completeness, constraint satisfaction, contradiction rate
- `compare_models.py`: runs base and lora side-by-side on all eval prompts
- `run_eval.py`: orchestrates the full pipeline, returns comparison dict

## Data flow (single request)

```
1. User types "high protein taco under 400 calories"
2. frontend/app.py → POST /generate-taco (model=base) + POST /generate-taco (model=lora), parallel
3. main.py: load session prefs from SessionMemory
4. ConstraintParser.extract("high protein taco under 400 calories")
   → {high_protein: true, max_calories: 400}
5. Merge with session prefs
6. InferencePipeline.generate(message, constraints, model_variant)
   → build system + user prompt
   → tokenize + generate (LLaMA-3 instruct format)
   → extract JSON from output
   → retry if parse fails (up to 2 attempts)
7. TacoValidator.validate(result, constraints)
   → check calorie threshold, ingredient exclusions
8. SessionMemory.update(session_id, constraints)
9. Return {data: taco, metadata: {model, valid_json, attempts, constraints, issues}}
10. frontend renders HTML card + debug markdown
```
