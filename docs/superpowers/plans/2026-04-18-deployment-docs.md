# Deployment & Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update README to reflect the real stack (Gradio, local FastAPI, checkpoint inference), and create all supporting docs needed for the demo and rubric.

**Architecture:** Two processes — `uvicorn` running `backend/app/main.py` on port 8000, and `python frontend/app.py` running Gradio on port 7860. The LoRA adapter lives at `backend/checkpoints/tacollm-lora-v1/` and is lazy-loaded on first request with `model="lora"`.

**Tech Stack:** FastAPI, Gradio, uv, Python 3.12, HuggingFace Transformers + PEFT

---

### Task 1: Update README.md

**Files:**
- Modify: `README.md`

The existing README describes the old Next.js / Amplify / SageMaker-endpoint stack. Replace the opening section with the real implementation, and add a How-to-Run section. Keep all the Architecture Decision Records (everything from "Decision 1" onwards) intact — they are valuable for the rubric.

- [ ] **Step 1: Rewrite the README header and add run instructions**

Replace everything in `README.md` from the top down to (but not including) `## Table of Contents` with:

```markdown
# TacoLLM

**TacoLLM** is a fine-tuned, constraint-aware taco recommendation system built as a final project for LLM Class 2026.

The system accepts natural language dietary requests ("high protein taco under 400 calories, no dairy"), extracts structured constraints, runs inference through a FastAPI backend, and renders side-by-side taco cards from the base model and a LoRA-adapted model in a Gradio frontend.

The core research question: **does LoRA fine-tuning improve the reliability of structured, constraint-aware JSON generation compared with a base instruction model?**

## Stack

| Layer | Technology |
|---|---|
| Frontend | Gradio `gr.Blocks` (Python) |
| Backend API | FastAPI + Uvicorn |
| Base model | `meta-llama/Llama-3.2-3B-Instruct` |
| Fine-tuned model | LoRA adapter trained on SageMaker (`ml.g5.2xlarge`) |
| Training | HuggingFace PEFT + TRL via SageMaker Training Job |
| Evaluation | Custom Python pipeline, 300 held-out prompts |
| Package manager | `uv` |

## How to Run

### Prerequisites

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/) installed
- HuggingFace account with access to `meta-llama/Llama-3.2-3B-Instruct`
- A `.env` file in the project root:
  ```
  HF_TOKEN=hf_...
  ```

### 1. Install dependencies

```bash
# Backend
cd backend && uv sync && cd ..

# Frontend
cd frontend && uv sync && cd ..
```

### 2. (Optional) Install the LoRA adapter

After the SageMaker training job completes, download and extract the adapter:

```bash
aws s3 cp s3://marco-pineda-final-project/tacollm/output/tacollm-lora-v1-2026-04-18-17-59-57-270/output/model.tar.gz /tmp/tacollm-adapter.tar.gz
mkdir -p backend/checkpoints/tacollm-lora-v1
tar -xzf /tmp/tacollm-adapter.tar.gz -C backend/checkpoints/tacollm-lora-v1/
```

If the adapter is not present, all requests fall back to the base model automatically.

### 3. Start the backend

```bash
cd backend
HF_TOKEN=$(grep HF_TOKEN ../.env | cut -d= -f2) \
HUGGING_FACE_HUB_TOKEN=$(grep HF_TOKEN ../.env | cut -d= -f2) \
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API is available at `http://localhost:8000`. Check `http://localhost:8000/docs` for the interactive Swagger UI.

### 4. Start the frontend

In a second terminal:

```bash
cd frontend
uv run python app.py
```

Open `http://127.0.0.1:7860` in your browser.

### 5. Run the evaluation pipeline

With the backend running:

```bash
cd backend
uv run python -m evaluation.run_eval
```

Results are printed to stdout and returned from `POST /evaluate`.

### Demo sequence

1. "Give me a high protein taco under 400 calories."
2. "Make it spicy and keep it dairy free."
3. "Now make it vegan under 350 calories."

Each turn: the system remembers prior constraints. Both taco cards update. The Debug accordion shows `parsed_constraints` accumulating across turns.

## Project Structure

```
final/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI routes
│   │   ├── inference.py     # Model loading + generation pipeline
│   │   ├── parser.py        # Constraint extractor
│   │   ├── validator.py     # Output validator
│   │   ├── memory.py        # Session memory
│   │   └── prompts.py       # System + user prompt builders
│   ├── training/
│   │   ├── train_lora.py    # SageMaker training entry point
│   │   ├── sagemaker_job.py # Job launcher
│   │   └── generate_dataset.py
│   ├── evaluation/
│   │   ├── run_eval.py      # Full evaluation runner
│   │   ├── metrics.py       # Scoring functions
│   │   └── compare_models.py
│   └── checkpoints/
│       └── tacollm-lora-v1/ # LoRA adapter (download after training)
├── frontend/
│   ├── app.py               # Gradio Blocks UI
│   └── client.py            # HTTP client for FastAPI
├── data/
│   ├── train.jsonl          # 4,700 training examples
│   └── eval.json            # 300 held-out evaluation prompts
├── docs/
│   ├── api.md
│   ├── architecture.md
│   ├── deployment.md
│   └── evaluation.md
└── README.md
```
```

- [ ] **Step 2: Verify the Table of Contents and ADR sections are untouched**

Read the file and confirm everything from `## Table of Contents` to the end is unchanged.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: update README with real stack, run instructions, project structure"
```

---

### Task 2: Create `docs/deployment.md`

**Files:**
- Create: `docs/deployment.md`

- [ ] **Step 1: Create the file**

```markdown
# Deployment Guide

## Local Deployment (recommended for demo)

### Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.12+ |
| uv | latest |
| HuggingFace token | with Llama-3.2-3B-Instruct access |
| AWS CLI | configured (for adapter download only) |

### Environment variables

Create `final/.env`:

```
HF_TOKEN=hf_...
ANTHROPIC_API_KEY=sk-ant-...   # only needed for dataset generation
```

The backend reads `HUGGING_FACE_HUB_TOKEN` (the HuggingFace standard env var) at model load time. The start command in the README sets both `HF_TOKEN` and `HUGGING_FACE_HUB_TOKEN` from the `.env` file.

### Install dependencies

```bash
cd backend && uv sync && cd ..
cd frontend && uv sync && cd ..
```

### Download the LoRA adapter

```bash
aws s3 cp \
  s3://marco-pineda-final-project/tacollm/output/tacollm-lora-v1-2026-04-18-17-59-57-270/output/model.tar.gz \
  /tmp/tacollm-adapter.tar.gz \
  --profile 165286508758_AdministratorAccess

mkdir -p backend/checkpoints/tacollm-lora-v1
tar -xzf /tmp/tacollm-adapter.tar.gz -C backend/checkpoints/tacollm-lora-v1/
```

Expected contents of `backend/checkpoints/tacollm-lora-v1/`:
```
adapter_config.json
adapter_model.safetensors   (or adapter_model.bin)
```

### Start the backend

```bash
cd backend
HF_TOKEN=$(grep HF_TOKEN ../.env | cut -d= -f2) \
HUGGING_FACE_HUB_TOKEN=$(grep HF_TOKEN ../.env | cut -d= -f2) \
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

On first start, Transformers downloads `meta-llama/Llama-3.2-3B-Instruct` (~6 GB) to the HuggingFace cache. Subsequent starts are fast.

### Start the frontend

```bash
cd frontend
uv run python app.py
```

Open `http://127.0.0.1:7860`.

### Run tests

```bash
# Backend (115+ tests)
cd backend && uv run pytest tests/ -q

# Frontend (12 tests)
cd frontend && uv run pytest tests/ -q
```

## Model loading behaviour

| Scenario | Behaviour |
|---|---|
| `backend/checkpoints/tacollm-lora-v1/` exists | LoRA adapter loaded lazily on first `model="lora"` request |
| Adapter missing | `model="lora"` requests fall back to base model with a warning log |
| Backend unreachable | Gradio frontend shows red error cards in both columns |

## Ports

| Service | Port | URL |
|---|---|---|
| FastAPI backend | 8000 | `http://localhost:8000` |
| Gradio frontend | 7860 | `http://127.0.0.1:7860` |
| FastAPI Swagger UI | 8000 | `http://localhost:8000/docs` |
```

- [ ] **Step 2: Commit**

```bash
git add docs/deployment.md
git commit -m "docs: add deployment guide"
```

---

### Task 3: Create `docs/api.md`

**Files:**
- Create: `docs/api.md`

- [ ] **Step 1: Create the file**

```markdown
# API Reference

Base URL: `http://localhost:8000`

Interactive docs: `http://localhost:8000/docs` (Swagger UI)

---

## GET /health

Returns service health and model readiness.

**Response**

```json
{
  "status": "ok",
  "model_loaded": true,
  "active_model": "Llama-3.2-3B-Instruct"
}
```

| Field | Type | Description |
|---|---|---|
| `status` | string | `"ok"` when the service is healthy |
| `model_loaded` | boolean | `true` once the base model has loaded |
| `active_model` | string | Short model name |

---

## POST /generate-taco

Main inference endpoint. Accepts a natural language request, extracts constraints, runs the model, validates output, and returns a structured taco recommendation.

**Request body**

```json
{
  "message": "High protein taco under 400 calories, no dairy",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "model": "lora"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `message` | string | yes | Natural language taco request |
| `session_id` | string | no | UUID for session memory (default: `"default"`) |
| `model` | string | no | `"base"` or `"lora"` (default: `"base"`) |

**Response**

```json
{
  "data": {
    "name": "Chipotle Lime Chicken Taco",
    "ingredients": ["corn tortillas", "grilled chicken breast", "chipotle salsa", "cilantro", "onion"],
    "calories": 365,
    "protein": 34,
    "carbs": 28,
    "fat": 11,
    "dietary_tags": ["high_protein", "dairy_free"],
    "spice_level": "medium",
    "reasoning": "Lean chicken breast keeps protein high while corn tortillas and simple toppings stay under 400 calories with no dairy."
  },
  "metadata": {
    "model": "Llama-3.2-3B-Instruct",
    "valid_json": true,
    "inference_attempts": 1,
    "parsed_constraints": {
      "max_calories": 400,
      "high_protein": true,
      "no_dairy": true
    },
    "validation_issues": [],
    "session_id": "550e8400-e29b-41d4-a716-446655440000"
  }
}
```

**Error responses**

| Status | Condition |
|---|---|
| 422 | Model failed to produce valid JSON after retries |

---

## DELETE /session/{session_id}

Clears stored preference memory for a session.

**Response**

```json
{"cleared": true, "session_id": "550e8400-e29b-41d4-a716-446655440000"}
```

---

## POST /evaluate

Runs the full evaluation pipeline (base vs. LoRA) on the 300-prompt held-out dataset. Returns comparison metrics. Intended for demo and development use.

**Response**

```json
{
  "base": {
    "valid_json_rate": 0.81,
    "field_completeness_rate": 0.84,
    "constraint_satisfaction_rate": 0.68,
    "contradiction_rate": 0.19
  },
  "lora": {
    "valid_json_rate": 0.97,
    "field_completeness_rate": 0.98,
    "constraint_satisfaction_rate": 0.90,
    "contradiction_rate": 0.05
  },
  "summary_table": "..."
}
```

---

## Taco output schema

Every successful `/generate-taco` response contains a `data` object conforming to this schema:

| Field | Type | Constraints |
|---|---|---|
| `name` | string | Taco name |
| `ingredients` | string[] | List of ingredients |
| `calories` | number | Integer kcal |
| `protein` | number | Grams |
| `carbs` | number | Grams |
| `fat` | number | Grams |
| `dietary_tags` | string[] | e.g. `["high_protein", "dairy_free"]` |
| `spice_level` | string | One of: `"mild"`, `"medium"`, `"hot"` |
| `reasoning` | string | Brief explanation of constraint satisfaction |
```

- [ ] **Step 2: Commit**

```bash
git add docs/api.md
git commit -m "docs: add API reference"
```

---

### Task 4: Create `docs/architecture.md`

**Files:**
- Create: `docs/architecture.md`

- [ ] **Step 1: Create the file**

```markdown
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
```

- [ ] **Step 2: Commit**

```bash
git add docs/architecture.md
git commit -m "docs: add architecture guide"
```

---

### Task 5: Create `docs/evaluation.md`

**Files:**
- Create: `docs/evaluation.md`

- [ ] **Step 1: Create the file**

```markdown
# Evaluation Methodology

## Research question

Does LoRA fine-tuning improve the reliability of structured, constraint-aware taco recommendation generation compared with the base `meta-llama/Llama-3.2-3B-Instruct` model?

## Evaluation dataset

- **Size:** 300 held-out prompts in `data/eval.json`
- **Not seen during training** (training set: 4,700 prompts in `data/train.jsonl`)
- **Prompt categories:**

| Category | Count | Examples |
|---|---|---|
| Single constraint | 60 | "High protein taco" |
| Calorie constraint | 60 | "Taco under 350 calories" |
| Dietary exclusion | 60 | "No dairy taco", "Vegan taco" |
| Multi-constraint | 60 | "High protein, no dairy, under 400 cal" |
| Style/spice | 30 | "Spicy street taco", "Mild Tex-Mex" |
| Edge cases | 30 | Conflicting constraints, extreme values |

## Metrics

### 1. JSON Validity Rate

```
valid_json_rate = outputs that parse as valid JSON / total outputs
```

### 2. Field Completeness Rate

All 9 required fields must be present: `name`, `ingredients`, `calories`, `protein`, `carbs`, `fat`, `dietary_tags`, `spice_level`, `reasoning`.

```
field_completeness_rate = outputs with all required fields / total outputs
```

### 3. Constraint Satisfaction Rate

Per-constraint checks:
- `max_calories`: output `calories` ≤ threshold
- `high_protein`: output `protein` ≥ 25g
- `no_dairy`: no dairy keywords in `ingredients`
- `vegan`: no meat or dairy keywords in `ingredients`
- `spice_level`: output `spice_level` matches requested level

```
constraint_satisfaction_rate = constraints satisfied / total constraints checked
```

### 4. Contradiction Rate

Logical contradictions in output:
- Tag `dairy_free` but `ingredients` contains dairy
- Tag `vegan` but `ingredients` contains meat
- `calories` field value exceeds `max_calories` constraint

```
contradiction_rate = outputs with any contradiction / total outputs
```

## Running the evaluation

With the backend running:

```bash
cd backend
uv run python -m evaluation.run_eval
```

Or via the API:

```bash
curl -X POST http://localhost:8000/evaluate | python -m json.tool
```

## Results

_To be filled in after the SageMaker training job `tacollm-lora-v1-2026-04-18-17-59-57-270` completes and the adapter is downloaded._

Expected format:

```
Model Comparison Summary

Metric                          Base Model     LoRA Model
---------------------------------------------------------
Valid JSON Rate                 —              —
Field Completeness Rate         —              —
Constraint Satisfaction Rate    —              —
Contradiction Rate              —              —
```
```

- [ ] **Step 2: Commit**

```bash
git add docs/evaluation.md
git commit -m "docs: add evaluation methodology"
```

---

### Task 6: Final smoke test

- [ ] **Step 1: Verify both services start**

Terminal 1 (backend):
```bash
cd backend
HF_TOKEN=$(grep HF_TOKEN ../.env | cut -d= -f2) \
HUGGING_FACE_HUB_TOKEN=$(grep HF_TOKEN ../.env | cut -d= -f2) \
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Expected: `Application startup complete.`

Terminal 2 (frontend):
```bash
cd frontend && uv run python app.py
```

Expected: `Running on local URL: http://127.0.0.1:7860`

- [ ] **Step 2: Verify health endpoint**

```bash
curl http://localhost:8000/health
```

Expected:
```json
{"status":"ok","model_loaded":true,"active_model":"Llama-3.2-3B-Instruct"}
```

- [ ] **Step 3: Run all tests one final time**

```bash
cd backend && uv run pytest tests/ -q && cd ../frontend && uv run pytest tests/ -q
```

Expected: all tests pass.

- [ ] **Step 4: Final commit**

```bash
git add .
git commit -m "feat: complete deployment wiring and documentation (Plan 6)"
```
