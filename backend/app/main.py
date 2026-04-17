"""
TacoLLM — FastAPI Backend
Main application entry point.
"""

from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .inference import InferencePipeline
from .memory import SessionMemory
from .parser import ConstraintParser
from .validator import TacoValidator

app = FastAPI(
    title="TacoLLM API",
    description="Constraint-aware taco recommendation system powered by a fine-tuned LLaMA model.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Module singletons (loaded once on startup)
pipeline = InferencePipeline()
parser = ConstraintParser()
validator = TacoValidator()
memory = SessionMemory()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"
    model: Optional[str] = "base"  # "base" | "lora"


class GenerateResponse(BaseModel):
    data: Dict[str, Any]
    metadata: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    active_model: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Returns service health and model readiness."""
    return {
        "status": "ok",
        "model_loaded": pipeline.is_loaded(),
        "active_model": pipeline.active_model_name(),
    }


@app.post("/generate-taco", response_model=GenerateResponse, tags=["Inference"])
def generate_taco(request: GenerateRequest):
    """
    Main inference endpoint.

    Accepts a natural language taco request, extracts constraints,
    builds a prompt, runs the model, validates output, and returns
    a structured taco recommendation.
    """
    # 1. Load session memory
    session_prefs = memory.get(request.session_id)

    # 2. Extract constraints from current message
    constraints = parser.extract(request.message)

    # 3. Merge with session memory (memory fills gaps, current message wins)
    merged_constraints = {**session_prefs, **constraints}

    # 4. Run inference (with optional retry)
    result, valid_json, attempts = pipeline.generate(
        user_message=request.message,
        constraints=merged_constraints,
        model_variant=request.model,
    )

    if result is None:
        raise HTTPException(
            status_code=422,
            detail="Model failed to produce valid JSON after retries.",
        )

    # 5. Validate output against constraints
    validation_issues = validator.validate(result, merged_constraints)

    # 6. Persist useful preferences to session memory
    memory.update(request.session_id, constraints)

    return {
        "data": result,
        "metadata": {
            "model": pipeline.active_model_name(),
            "valid_json": valid_json,
            "inference_attempts": attempts,
            "parsed_constraints": merged_constraints,
            "validation_issues": validation_issues,
            "session_id": request.session_id,
        },
    }


@app.delete("/session/{session_id}", tags=["System"])
def clear_session(session_id: str):
    """Clears stored memory for a session."""
    memory.clear(session_id)
    return {"cleared": True, "session_id": session_id}


@app.post("/evaluate", tags=["Evaluation"])
def run_evaluation():
    """
    Triggers the evaluation pipeline. Returns comparison metrics
    between the base and LoRA models on the held-out eval dataset.
    Only intended for development / demo use.
    """
    from evaluation.run_eval import run_full_evaluation

    results = run_full_evaluation(pipeline)
    return results


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
