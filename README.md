# Action Plan

TacoLLM is a fine-tuned, constraint-aware taco recommendation system built as a final project for LLM Class 2026. The system accepts natural language dietary requests, extracts structured constraints, and returns a JSON taco recommendation rendered as a polished UI card. The core research question is whether LoRA fine-tuning improves the reliability of structured, constraint-aware generation compared to a base instruction model.

The project is deployed as a production-grade AWS stack: a Next.js frontend on Amplify, a FastAPI backend, and a SageMaker managed inference endpoint serving a LoRA-adapted `Llama-3.2-3B-Instruct` model. A 5,000-row synthetic dataset is used, split into training and held-out evaluation sets. The evaluation pipeline compares base vs. fine-tuned model on JSON validity, constraint satisfaction, field completeness, and contradiction rate.

The domain is tacos — chosen for personal and cultural reasons, and because the narrow-but-rich constraint space (calorie limits, dietary restrictions, macros, style) makes it an ideal test bed for structured generation reliability.

## Table of Contents

### Decisions
1. [Why Tacos as the Domain](#decision-1-why-tacos-as-the-domain)
2. [Session-Based Preference Memory](#decision-3-session-based-preference-memory-vs-mem0-or-summarization)
3. [AWS SageMaker Endpoint for Model Inference](#decision-4-aws-sagemaker-endpoint-for-model-inference-vs-raw-ec2)
4. [AWS Amplify for Frontend Hosting](#decision-5-aws-amplify-for-frontend-hosting-vs-vercel-or-ec2)
5. [Dataset Generation](#decision-6--dataset-generation)
6. [LoRA Fine-Tuning via PEFT](#decision-7-lora-fine-tuning-via-peft-vs-full-fine-tune-or-prompt-engineering-only)

### Further Detail
- [LLM-Specific Feature Decisions](#llm-specific-feature-decisions)
  - [Streaming](#streaming-not-implemented)
  - [Retry Logic](#retry-logic-implemented)
  - [Stop Button](#stop-button-not-implemented)
  - [Model Selection](#model-selection-user-selectable-in-api)
  - [Sampling Configuration](#sampling-configuration)
- [Memory & Conversation History](#memory--conversation-history)
- [Evaluation Strategy](#evaluation-strategy)
- [User Experience & Design Considerations](#user-experience--design-considerations)

---

### Decision 1: Why Tacos as the Domain

Tacos are part of my cultural heritage and something I genuinely care about. Building a project around a personally meaningful domain makes it feel authentic rather than arbitrary. Tacos are also memorable and easy to demo — an audience does not need domain expertise to understand a taco recommendation. Most importantly, the taco domain is technically well-scoped: narrow enough to fine-tune a model on meaningfully, yet rich enough to express diverse and intersecting constraints (calorie limits, vegan/keto/dairy-free, spice level, style) that stress-test constraint-following reliability. The project frames itself not as a novelty chatbot but as a constrained generation reliability study — the tacos are the interface, the evaluation is the contribution.

#### Pros:

- Personally motivated — reflects cultural identity and genuine interest
- Memorable and presentation-friendly — audience immediately understands the domain
- Constraint-rich domain naturally tests the model's structured output reliability
- Small enough scope to finish cleanly within the course timeline

#### Cons:

- Could be perceived as a novelty project if not framed carefully — requires deliberate academic positioning in the presentation
- Domain-specific fine-tuning means the model is not generalizable beyond tacos

---

### Decision 3: Session-Based Preference Memory (vs. Mem0 or Summarization)

The backend stores extracted user preferences (e.g., `high_protein: true`, `no_dairy: true`) in server-side session state keyed by `session_id`. These preferences are merged into each subsequent prompt so users do not have to repeat themselves across turns. The current message always overrides stored memory. Sessions are cleared on explicit user action.

Two alternatives were considered and rejected:

- **Summarization:** After N turns, summarize the conversation and use the summary as rolling context. This handles very long conversations but adds latency, complexity, and a second LLM call per request. In practice, taco recommendation sessions rarely exceed 5–10 turns, making this overhead unjustified.
- **Mem0 / vector-based memory:** Persistent cross-session memory using an external store. This would require a separate infrastructure dependency and persistent user identity — significant complexity for a feature that is not the project's core evaluation goal.

Session-based memory satisfies the rubric requirement for a contextual chat interface, is explainable in 30 seconds during the demo, and introduces zero additional infrastructure.

#### Pros:

- Zero additional infrastructure — no external memory service required
- Explainable in a demo — "the system remembers your preferences within a session"
- Directly satisfies the rubric's multi-turn chat requirement
- Tradeoffs vs. alternatives are documented and defensible in the presentation

#### Cons:

- Memory does not persist across sessions — preferences are lost on page reload or session clear
- Does not handle conversational nuance — only stores explicit constraint flags, not free-form context

---

### Decision 4: AWS SageMaker Endpoint for Model Inference (vs. Raw EC2)

Model inference is served via an AWS SageMaker managed endpoint rather than a manually managed EC2 GPU instance. The LoRA adapter checkpoint is stored in S3 and loaded into the SageMaker container at endpoint startup.

A raw EC2 GPU instance was the primary alternative. EC2 gives more direct control but requires manual GPU provisioning, health check configuration, instance lifecycle management, and scaling logic. SageMaker handles all of this as a managed service and provides native S3 integration for checkpoint loading — meaning training artifacts flow directly from the SageMaker Training Job to the inference endpoint with no manual file transfer.

#### Pros:

- Managed GPU provisioning — no manual instance lifecycle management
- Native S3 integration — LoRA checkpoint flows directly from training job to endpoint
- Built-in health checks, endpoint versioning, and rollback support
- Demonstrates production-grade MLOps practices, strengthening the rubric's "Production Environment" score

#### Cons:

- Cold start latency (~30–60 seconds if endpoint scales to zero) — endpoint must be kept warm for demos
- Higher cost than a persistent EC2 instance for always-on workloads
- More configuration overhead than SSH-ing into an EC2 box and running uvicorn

---

### Decision 5: AWS Amplify for Frontend Hosting (vs. Vercel or EC2)

The Next.js frontend is hosted on AWS Amplify. Amplify connects directly to the GitHub repository and auto-deploys on push, with built-in HTTPS, CDN distribution, and environment variable management.

Vercel was the main alternative and is arguably simpler. However, Amplify keeps the entire stack within AWS, which simplifies IAM permission management between the frontend and backend, avoids cross-cloud networking considerations, and presents a cleaner architecture story in the presentation. EC2 for the frontend was ruled out immediately — running a Next.js server on EC2 is unnecessary operational overhead when a managed hosting service exists.

#### Pros:

- Stays fully within the AWS ecosystem — consistent IAM, networking, and billing
- Git-connected auto-deploy — push to main triggers a production build automatically
- Free tier is generous for a class project with low traffic
- Built-in HTTPS and CDN with no additional configuration

#### Cons:

- Slightly more configuration than Vercel for a pure Next.js project
- Amplify build times can be slower than Vercel's optimized Next.js pipeline
- AWS console UX is more complex than Vercel's dashboard

---

### Decision 6:  Dataset Generation 

The dataset is split into 4,700 training examples and 300 held-out evaluation examples.

Manual curation was considered making this a tough endeavor at this scale — hand-writing 5,000 quality examples is not feasible within the project timeline. A purely template-based deterministic script was also considered but produces lower variety in ingredient combinations, nutritional values, and reasoning text, which would limit what the model learns. Ultimately, Claude API generation + manual curation would produces rich, varied, realistic examples at speed, and the generation script itself becomes a project artifact demonstrating prompt engineering skill.

#### Pros:

- High quality and variety — realistic ingredient combinations, plausible nutritional values, diverse constraint scenarios
- Scalable — 5,000 examples generated in minutes rather than weeks
- Generation script is a demonstrable project artifact
- Schema validation filters bad examples before they enter training

#### Cons:

- Dataset quality depends on the generation prompts — poorly written templates produce biased or repetitive data
- API cost for generating 5,000 examples (mitigated by Claude's low per-token cost at this scale)
- Generated data may not capture the full diversity of real user phrasing

---

### Decision 7: LoRA Fine-Tuning via PEFT (vs. Full Fine-Tune or Prompt Engineering Only)

The model is adapted using Low-Rank Adaptation (LoRA) via the Hugging Face PEFT library on top of `meta-llama/Llama-3.2-3B-Instruct`. Only low-rank adapter matrices are trained; base weights are frozen.

Full fine-tuning was ruled out — updating all parameters of a 3B model requires significantly more GPU memory and training time, with diminishing returns for a domain-adaptation task of this scope. Prompt engineering only (no fine-tuning) was also considered as a baseline but does not satisfy the course rubric's requirement for a trained model, and the project's central research question is specifically about whether fine-tuning improves over prompting.

LoRA is GPU-efficient, produces a small adapter (~10–50MB) that loads cleanly on top of the base model in SageMaker, and enables the live base vs. fine-tuned comparison that is central to the demo and evaluation.

#### Pros:

- GPU-efficient — trains on a single SageMaker GPU instance without model parallelism
- Adapter is small and portable — stores easily in S3, loads quickly at inference time
- Satisfies the course rubric's fine-tuning requirement explicitly
- Enables clean base vs. fine-tuned comparison at inference time by switching adapters
- Established technique with well-documented hyperparameter guidance

#### Cons:

- LoRA may not capture all domain-specific patterns that a full fine-tune would learn
- Hyperparameter sensitivity (rank, alpha, target modules) requires experimentation
- If the base model is already strong at JSON generation, gains may be modest and harder to show clearly

# Further Detail

## LLM-Specific Feature Decisions

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

## Memory & Conversation History

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

## Evaluation Strategy

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

## User Experience & Design Considerations

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
- **No streaming.** Given the output is structured JSON that feeds a rendered card, streaming would result in a visually broken mid-render card. The better experience is a loading indicator while inference runs, followed by an instant full card render. 
