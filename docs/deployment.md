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
  s3://marco-pineda-final-project/tacollm/output/tacollm-lora-v1-2026-04-19-00-15-49-687/output/model.tar.gz \
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
