# ADR-001: TacoLLM Architecture Decisions

**Date:** 2026-04-15
**Status:** Accepted
**Author:** Marco Pineda

---

## Table of Contents

1. [Project Motivation](#1-project-motivation)
2. [User Experience & Design Considerations](#2-user-experience--design-considerations)
3. [LLM-Specific Feature Decisions](#3-llm-specific-feature-decisions)
4. [Memory & Conversation History](#4-memory--conversation-history)
5. [Architecture & Infrastructure](#5-architecture--infrastructure)
6. [Dataset Generation Strategy](#6-dataset-generation-strategy)
7. [Fine-Tuning Approach](#7-fine-tuning-approach)
8. [Evaluation Strategy](#8-evaluation-strategy)

---

## 1. Project Motivation

### Why Tacos?

This project centers on tacos for three distinct reasons:

**Personal and cultural connection.** Tacos are part of my cultural heritage. Building something that reflects a meaningful personal domain makes the project feel authentic rather than arbitrary.

**Creative ambition.** A taco recommendation system is memorable, visually expressive, and easy to demo without extensive domain knowledge from the audience. It makes structured generation feel fun instead of sterile.

**Technical fit.** The taco domain is narrow enough to fine-tune on meaningfully but rich enough to test constraint-following in interesting ways (dietary, caloric, macro, and stylistic constraints all apply naturally). This makes the domain ideal for evaluating whether LoRA adaptation actually improves structured output reliability — which is the real research question of this project.

---

## 2. User Experience & Design Considerations

### User Stories

**As a user, I want to:**
- Type a natural language request ("high protein taco, no dairy, under 400 calories") and receive a structured, readable recommendation.
- Have my dietary preferences remembered within a session so I don't have to repeat myself on follow-up messages.
- See a polished taco card that renders the recommendation with name, ingredients, macros, tags, spice level, and reasoning.
- Reset my session preferences when I want to start fresh.

**As a demo presenter, I want to:**
- Show a clean, fast chat experience that does not look like a developer prototype.
- Demonstrate constraint chaining across turns (turn 1: "high protein", turn 2: "make it spicy" — system remembers the protein constraint).
- Show a debug panel that reveals parsed constraints and validation results without cluttering the main UI.
- Show the evaluation comparison chart between base and fine-tuned model without leaving the app.

### UI Components Decided

| Component | Decision |
|-----------|----------|
| Chat window | Scrollable message history, user and assistant bubbles |
| Input box | Single-line natural language entry, submit on Enter |
| Taco card | Rendered from structured JSON: name, ingredients list, macro table, dietary tags, spice badge, reasoning |
| Debug panel | Hidden by default, toggled in dev mode — shows parsed constraints, model used, JSON validity, inference attempts |
| Session reset button | Clears memory and conversation history |

### Design Principles

- **Polished but minimal.** The UI should look intentional, not feature-heavy. Tailwind CSS gives us that with low effort.
- **Card-first rendering.** The taco recommendation renders as a card, not raw JSON. The JSON is still visible in debug mode for demo purposes.
- **No streaming.** Given the output is structured JSON that feeds a rendered card, streaming would result in a visually broken mid-render card. The better experience is a loading indicator while inference runs, followed by an instant full card render. See Section 3 for full rationale.

---

## 3. LLM-Specific Feature Decisions

### Streaming: Not Implemented

**Decision:** No token-by-token streaming.

**Rationale:** TacoLLM outputs strict JSON that is immediately parsed and rendered as a UI card. Streaming JSON mid-generation produces invalid JSON at every intermediate state, meaning the frontend cannot render anything useful until the full response arrives anyway. A clean loading animation followed by a full card reveal is a better experience than watching characters appear in a broken JSON blob.

This also simplifies the SageMaker endpoint configuration — standard synchronous inference rather than a streaming endpoint.

**Future consideration:** If the model is ever modified to produce a narrative or conversational prefix before the JSON, streaming could be re-evaluated.

### Retry Logic: Implemented

**Decision:** The inference pipeline retries up to 2 times if the model output fails JSON parsing.

**Rationale:** Small instruction models occasionally produce malformed JSON, especially under novel constraint combinations. A corrective retry prompt (explicitly instructing the model to return only raw JSON) recovers most of these failures without requiring the user to re-submit.

**Implementation:** On retry, the user prompt is prefixed with a corrective instruction. The number of attempts is surfaced in the response metadata for observability.

### Stop Button: Not Implemented

**Decision:** No stop/cancel button for in-flight inference.

**Rationale:** Inference requests are short (under ~5 seconds with a 3B model). Cancellation adds frontend/backend complexity for negligible user benefit. If latency grows (e.g., due to cold SageMaker starts), a timeout + error message is the simpler solution.

### Model Selection: User-Selectable in API

**Decision:** The `/generate-taco` endpoint accepts a `model` parameter (`"base"` or `"lora"`). The frontend defaults to `"lora"` post-training.

**Rationale:** Being able to switch between base and fine-tuned model at request time enables live A/B demonstration during the presentation without restarting any service.

### Sampling Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| temperature | 0.3 | Low temperature increases JSON reliability and reduces hallucinated field values |
| top_p | 0.9 | Nucleus sampling to retain some diversity |
| max_new_tokens | 512 | Sufficient to complete the full JSON schema |
| do_sample | true | Avoids greedy repetition artifacts |
| repetition_penalty | 1.1 | Mild penalty to prevent repeated ingredient lists |

---

## 4. Memory & Conversation History

### Decision: Simple Session-Based Preference Memory

**Chosen approach:** Store extracted user preferences (e.g., `high_protein: true`, `no_dairy: true`) in backend session state, keyed by `session_id`. Merge stored preferences with new constraints on each request. Current message always wins over stored memory. Session clears on explicit user action.

**Alternatives considered and rejected:**

| Approach | Why Rejected |
|----------|--------------|
| **Summarization** (summarize conversation after N turns) | Adds latency and complexity for a chat app that rarely exceeds 5–10 turns in practice. Overkill for the demo context. |
| **Mem0 / vector-based memory** | Requires an external memory service and persistent user identity. Adds infrastructure dependencies that do not improve the project's core evaluation goal. |
| **Full conversation history in prompt** | Works for short conversations but grows the prompt unbounded. Small models degrade with long contexts. |

**Why this approach is correct for this project:**

The memory system is not a graded feature in isolation — it is a supporting mechanism for the constraint-following task. Session-based preference merging satisfies the rubric requirement for a "chat interface" with multi-turn context without adding infrastructure complexity that could destabilize the demo. The tradeoffs between this approach and alternatives (Mem0, summarization) are explicitly documented here, which is itself valuable for the presentation.

**Session memory schema:**
```json
{
  "preferences": {
    "high_protein": true,
    "no_dairy": true,
    "max_calories": 400,
    "spice_level": "hot"
  }
}
```

---

## 5. Architecture & Infrastructure

### System Architecture

```
User (Browser)
    |
    v
Next.js Frontend (AWS Amplify)
    |
    | HTTPS POST /generate-taco
    v
FastAPI Backend (AWS EC2 or App Runner)
    |
    |-- Constraint Parser
    |-- Session Memory
    |-- Prompt Builder
    |
    | InvokeEndpoint
    v
AWS SageMaker Inference Endpoint
    |
    | (Llama-3.2-3B-Instruct + LoRA adapter)
    v
Structured JSON Response
    |
    v
FastAPI Validator
    |
    v
Frontend Taco Card Render
```

### Infrastructure Decisions

| Component | Technology | Decision Rationale |
|-----------|------------|-------------------|
| Frontend | Next.js + TypeScript + Tailwind | Modern, type-safe, easy to make polished |
| Frontend hosting | AWS Amplify | Git-connected auto-deploy, stays in AWS ecosystem, free tier generous |
| Backend API | FastAPI + Uvicorn | Clean async Python, excellent Pydantic schema support, easy to document |
| Model inference | AWS SageMaker Endpoint | Managed GPU inference, scales to zero, integrates with S3 for checkpoint loading |
| Model training | AWS SageMaker Training Job | Manages GPU provisioning, stores artifacts to S3 automatically |
| Model checkpoint storage | AWS S3 | Native SageMaker integration, versioned |
| Base model | `meta-llama/Llama-3.2-3B-Instruct` | Small enough to train and serve economically, strong instruction following |

### Why SageMaker Over EC2 for Inference

**Decision:** SageMaker managed endpoint rather than a raw EC2 GPU instance.

**Rationale:**
- Managed GPU provisioning and auto-scaling without manual instance management.
- Native integration with S3 for loading LoRA checkpoints.
- Health checks, rollback, and endpoint versioning are built in.
- Demonstrates production-grade MLOps practices, which strengthens the rubric score for "Production Environment & Programmability."

**Tradeoff acknowledged:** SageMaker cold starts can add latency (~30–60s if endpoint scales to zero). For the demo, the endpoint will be kept warm. Cold start behavior is documented as a known limitation.

---

## 6. Dataset Generation Strategy

### Decision: Synthetic Generation via Claude API

**Dataset size:** 5,000 training examples (required by course).

**Method:** A Python script calls the Claude API with diverse constraint templates to generate instruction/output pairs. Each example contains a natural language instruction and a valid structured JSON taco recommendation.

**Why Claude API:**
- Generates high-quality, varied examples far faster than manual curation.
- Produces realistic nutritional values, diverse ingredient combinations, and varied constraint scenarios.
- Output can be validated against the schema programmatically before saving.
- The generation script itself becomes a project artifact that demonstrates prompt engineering skill.

**Dataset composition targets:**

| Category | Approx. % of dataset |
|----------|----------------------|
| Single constraint (e.g., high protein only) | 20% |
| Two constraints (e.g., vegan + low carb) | 30% |
| Three+ constraints (e.g., keto + no dairy + spicy) | 25% |
| Style-based (street taco, Tex-Mex, breakfast) | 10% |
| Edge / contradictory constraints | 10% |
| Follow-up / memory-aware prompts | 5% |

**Validation:** Each generated example is parsed against the required schema before inclusion. Examples failing schema validation or containing obvious contradictions (e.g., "vegan" with chicken) are discarded.

**Train/eval split:** 4,700 training / 300 held-out evaluation (not used during fine-tuning).

---

## 7. Fine-Tuning Approach

### Decision: LoRA via PEFT on Llama-3.2-3B-Instruct

**Method:** Parameter-Efficient Fine-Tuning (LoRA) using the Hugging Face PEFT library. Full weights are frozen; only low-rank adapter matrices are trained.

**Why LoRA:**
- Explicitly required by the course rubric.
- Economical on GPU memory — enables fine-tuning a 3B model on a single SageMaker GPU instance.
- Adapter is small (~10–50MB), easy to store on S3 and load at inference time.
- Clean separation from base weights enables live base vs. fine-tuned comparison.

**Target hyperparameters (to be tuned):**

| Parameter | Starting Value |
|-----------|---------------|
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| Target modules | q_proj, v_proj |
| Learning rate | 2e-4 |
| Batch size | 4 (with gradient accumulation x4) |
| Epochs | 3 |
| Max sequence length | 512 |
| Optimizer | AdamW (paged) |

---

## 8. Evaluation Strategy

### Research Question

Does LoRA fine-tuning improve the reliability of structured, constraint-aware taco recommendation generation compared with the base instruction model?

### Evaluation Dataset

300 held-out prompts not seen during training, spanning all constraint categories.

### Metrics

| Metric | What It Measures |
|--------|-----------------|
| JSON Validity Rate | % of outputs that parse as valid JSON |
| Field Completeness Rate | % of outputs containing all required schema fields |
| Constraint Satisfaction Rate | % of constraints correctly honored (calorie limits, dietary exclusions, etc.) |
| Contradiction Rate | % of outputs containing logical contradictions (e.g., vegan taco with beef) |
| Human Quality Score (optional) | 1–5 subjective plausibility and flavor coherence score |

### Comparison

| System | Description |
|--------|-------------|
| System A (baseline) | Base `Llama-3.2-3B-Instruct` with prompt engineering only |
| System B (fine-tuned) | LoRA-adapted model with same inference wrapper |

Results will be presented as a side-by-side table in the final presentation and surfaced via the `/evaluate` endpoint in the live demo.

---

## Summary of Key Decisions

| Decision | Choice | Primary Reason |
|----------|--------|----------------|
| Domain | Tacos | Cultural connection, creative, technically well-scoped |
| Streaming | No | JSON output incompatible with partial rendering |
| Memory | Session-based preference store | Sufficient for demo, explainable, zero extra infra |
| Stop button | Not implemented | Inference is short; complexity not justified |
| Inference hosting | SageMaker endpoint | Managed GPU, MLOps story, S3 integration |
| Frontend hosting | AWS Amplify | Git-connected deploy, stays in AWS ecosystem |
| Dataset generation | Claude API (5,000 rows) | Quality, speed, and scale |
| Fine-tuning method | LoRA / PEFT | Rubric requirement, GPU-efficient, easily swappable |
