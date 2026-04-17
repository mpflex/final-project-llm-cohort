# CLAUDE.md v2 — TacoLLM A-Level Final Project

## Project Title

**TacoLLM: A Fine-Tuned, Constraint-Aware Taco Recommendation System with Evaluation Pipeline**

---

## Executive Summary

TacoLLM is an LLM final project designed to score highly against the course rubric by combining:

1. a working taco-themed chat application,
2. a fine-tuned or LoRA-adapted model,
3. a documented inference pipeline,
4. a production-style API deployment,
5. a thoughtful evaluation dataset and scoring process,
6. clear technical documentation and demo flow.

The project is intentionally designed to be more than a simple chat wrapper. It treats the taco domain as a constrained generation problem and evaluates whether fine-tuning improves structured output reliability and constraint adherence.

This project directly targets all rubric sections:

- **Model & Inference**
- **Innovation & Creativity**
- **Production Environment & Programmability**
- **Technical Documentation**
- **Demo & Presentation**

---

## Why This Project Fits the Rubric

### Rubric Alignment

#### I. Model & Inference (40 points)

This project includes:
- a functional base model,
- LoRA fine-tuning or equivalent adaptation,
- meaningful outputs,
- an evaluation dataset,
- a comparison between base and adapted model behavior.

#### Innovation & Creativity (20 points)

This project is not just "generate a taco recipe." It frames the problem as:
- **constraint-aware generation,**
- **structured JSON generation,**
- **evaluation of reliability under user constraints.**

That makes it a more serious LLM systems project.

#### II. Production Environment & Programmability (30 points)

This project includes:
- a deployable inference service,
- a documented API endpoint,
- a complete inference pipeline,
- explicit sampling parameters,
- validation and retry logic.

#### III. Documentation & Presentation (30 points)

This document, plus the project README and demo plan, covers:
- model choice,
- dataset design,
- fine-tuning approach,
- evaluation methodology,
- deployment architecture,
- API behavior,
- demo narrative.

---

## Core Project Idea

A user chats with TacoLLM and requests tacos under specific constraints such as:

- "Give me a high protein taco under 400 calories."
- "Make me a vegan taco with low carbs."
- "I want a spicy taco with no dairy."
- "Give me a keto taco that still feels authentic."

The system responds with a structured taco recommendation in JSON, which the frontend renders as a polished UI card.

The real technical goal is not just generation. The goal is to determine whether a fine-tuned model is better than a base model at:

- following user constraints,
- producing valid structured output,
- avoiding obvious contradictions,
- maintaining response consistency.

---

## High-Level Research Question

**Does LoRA fine-tuning improve the reliability of structured, constraint-aware taco recommendation generation compared with a base instruction model?**

This question gives the project an academic framing and helps justify the evaluation section.

---

## Project Objectives

### Primary Objectives

1. Build a taco-themed chat interface that feels polished and usable.
2. Use an LLM to generate taco recommendations in strict JSON format.
3. Fine-tune or LoRA-adapt a model on a taco instruction dataset.
4. Create and use an evaluation dataset to compare model behavior.
5. Deploy the inference pipeline in a production-style environment.
6. Document the system well enough to support an excellent presentation.

### Secondary Objectives

1. Demonstrate prompt engineering and output control.
2. Show measurable gains from model adaptation.
3. Keep the domain small enough to finish cleanly.
4. Make the demo memorable and easy to explain.

---

## Scope

### In Scope

- Taco recommendation generation
- Structured output enforcement
- Constraint extraction
- Fine-tuning or LoRA adaptation
- Base vs fine-tuned evaluation
- API deployment
- Simple chat frontend
- Documentation and presentation assets

### Out of Scope

- Multi-agent orchestration
- Large-scale RAG system
- Massive recipe corpus scraping
- Mobile app development
- Multi-user authentication
- Real-time production traffic handling

This scope is intentional. The project should feel ambitious, but still be finishable.

---

## Final Deliverables

The finished project should include all of the following:

### Code Deliverables

- frontend chat application
- backend inference API
- fine-tuning scripts
- evaluation scripts
- utility modules for parsing and validation
- deployment configuration

### Documentation Deliverables

- `CLAUDE.md` or equivalent build plan
- project `README.md`
- evaluation methodology writeup
- API documentation
- deployment instructions
- demo script
- presentation outline

### Model Deliverables

- base model benchmark results
- fine-tuned or LoRA model checkpoint
- evaluation dataset
- evaluation results summary

---

## System Overview

### System Flow

```text
User Message
→ Constraint Extraction
→ Prompt Builder
→ Model Inference
→ JSON Validation
→ Optional Retry / Repair
→ API Response
→ Frontend Rendering
```

### Extended Flow with Evaluation

```text
Evaluation Input
→ Base Model Inference
→ Parse + Score
→ Fine-Tuned Model Inference
→ Parse + Score
→ Metric Aggregation
→ Comparison Report
```

---

## Architecture

### Frontend

A lightweight chat interface built in Next.js.

Responsibilities:
- collect user input,
- display conversation history,
- send requests to backend,
- render structured taco output as cards,
- optionally show debug metadata in development mode.

### Backend

A Python FastAPI service is recommended because it cleanly supports:
- model loading,
- inference endpoints,
- evaluation scripts,
- deployment to local or cloud environments.

Responsibilities:
- parse requests,
- construct prompts,
- call model,
- validate output,
- return structured JSON,
- expose evaluation or health endpoints if desired.

### Model Layer

Use a base instruction-following model suitable for local fine-tuning with LoRA. Good candidate families:
- Mistral
- Llama-family instruct variants
- Qwen instruct variants

Responsibilities:
- accept prompt,
- generate taco JSON,
- support base and fine-tuned checkpoints.

### Evaluation Layer

A Python scoring pipeline that compares:
- JSON validity,
- constraint satisfaction,
- completeness,
- contradiction rate,
- optional subjective quality.

---

## Recommended Tech Stack

### Frontend
- Next.js
- React
- TypeScript
- Tailwind CSS

### Backend
- FastAPI
- Uvicorn

### Model / ML
- Hugging Face Transformers
- PEFT
- TRL or standard trainer utilities
- PyTorch

### Evaluation
- Python
- Pandas
- JSON / CSV reports

### Deployment
- Local machine for baseline deployment
- Optional AWS EC2 / Lambda-compatible architecture if feasible
- Docker optional but helpful

---

## Why FastAPI Is Recommended

FastAPI gives the cleanest path to full rubric coverage because it makes it easy to show:
- API endpoint accessibility,
- production-style inference,
- clean request/response schemas,
- deployment readiness,
- local or cloud portability.

A local FastAPI endpoint is enough to satisfy the environment criterion if it is functional and documented well.

---

## Model Strategy

### Base Model

Choose a small instruction-tuned open model that can reasonably run locally or in your environment. The model should:
- support instruction following,
- be able to generate JSON,
- be practical for LoRA fine-tuning.

### Fine-Tuned Model

Apply LoRA or PEFT fine-tuning on a taco-domain instruction dataset.

The fine-tuned model should learn:
- taco recommendation style,
- structured JSON formatting,
- constraint adherence,
- domain consistency.

### Why LoRA

LoRA is explicitly named in the rubric and is the most efficient way to satisfy the "trained model" requirement without overcommitting to a full fine-tune.

---

## Task Definition

### Primary Task

Generate structured taco recommendations from natural language instructions.

### Input

Natural language user request such as:
- "High protein taco under 400 calories"
- "Vegan taco, spicy, no dairy"
- "Low carb taco with chicken and lots of flavor"

### Output

Strict JSON in this form:

```json
{
  "name": "Chipotle Lime Chicken Taco",
  "ingredients": [
    "corn tortillas",
    "grilled chicken breast",
    "cabbage slaw",
    "pickled onions",
    "chipotle salsa"
  ],
  "calories": 365,
  "protein": 34,
  "carbs": 28,
  "fat": 11,
  "dietary_tags": ["high_protein", "dairy_free"],
  "spice_level": "medium",
  "reasoning": "This taco uses lean chicken breast for high protein and avoids dairy while staying under the calorie target."
}
```

---

## Structured Output Contract

### Required Fields

Every model response must contain:

- `name`
- `ingredients`
- `calories`
- `protein`
- `carbs`
- `fat`
- `dietary_tags`
- `spice_level`
- `reasoning`

### Rules

- Output must be valid JSON.
- No text may appear before or after the JSON.
- Numeric values must be numbers, not strings.
- `ingredients` must be an array of strings.
- `dietary_tags` must be an array of strings.
- `spice_level` must be one of: `mild`, `medium`, `hot`.
- `reasoning` must be concise and constraint-aware.

---

## Constraint Handling

Constraint handling is a major part of the project and should be visible in both the code and presentation.

### Supported Constraint Categories

#### Calorie Constraints
Examples:
- under 400 calories
- low calorie
- under 500 kcal

#### Macro Constraints
Examples:
- high protein
- low carb
- high fat
- keto friendly

#### Ingredient Constraints
Examples:
- no dairy
- no beef
- chicken only
- vegetarian
- vegan

#### Style / Preference Constraints
Examples:
- spicy
- authentic
- Tex-Mex
- street taco
- healthy

---

## Constraint Extraction Plan

Use deterministic parsing first. This is easier to document and evaluate than relying fully on the model.

### Example Parsed Object

```json
{
  "max_calories": 400,
  "high_protein": true,
  "low_carb": false,
  "no_dairy": true,
  "vegan": false,
  "preferred_style": "street",
  "spice_level": "hot"
}
```

### Extraction Rules

1. Scan for calorie thresholds using regex.
2. Scan for known dietary phrases.
3. Scan for macro-related phrases.
4. Scan for style and spice preferences.
5. Include parsed constraints in the inference prompt.

This makes the pipeline more robust and easier to explain during demo.

---

## Simple Memory Design

Memory is not the central grading criterion, but it strengthens the app and can improve the presentation.

### Goal

Store recent user preferences across turns so that follow-up prompts feel context-aware.

### Example

Turn 1:
"I like high protein tacos and no dairy."

Turn 2:
"Make it spicy."

The system should remember:
- high protein
- no dairy

### MVP Memory Storage

Use short-session memory in the frontend or backend:
- store last known user preferences in session state,
- append them to subsequent prompt construction.

### Memory Example Object

```json
{
  "preferences": {
    "high_protein": true,
    "no_dairy": true,
    "preferred_spice": "hot"
  }
}
```

### Why Memory Helps

It gives the project a more complete "chat interface" feel without adding much implementation burden.

---

## Training Dataset Plan

### Goal

Create an instruction dataset for fine-tuning the model to:
- produce valid JSON,
- follow taco constraints,
- remain consistent.

### Dataset Format

Each example should contain:
- instruction,
- optional input,
- expected structured output.

### Example Entry

```json
{
  "instruction": "Generate a high protein taco under 400 calories with no dairy.",
  "output": {
    "name": "Fire-Grilled Chicken Taco",
    "ingredients": [
      "corn tortillas",
      "grilled chicken breast",
      "salsa roja",
      "cilantro",
      "onion"
    ],
    "calories": 345,
    "protein": 33,
    "carbs": 24,
    "fat": 9,
    "dietary_tags": ["high_protein", "dairy_free"],
    "spice_level": "medium",
    "reasoning": "Lean chicken keeps protein high while simple toppings and corn tortillas keep calories low and dairy out."
  }
}
```

### Dataset Size

Target a manageable dataset size such as:
- 150 to 500 examples if generating manually or semi-manually,
- more if you can synthesize examples programmatically and clean them.

A smaller, cleaner dataset is better than a huge noisy one.

### Dataset Composition

Ensure the dataset spans:
- protein-focused tacos,
- vegan tacos,
- low-carb tacos,
- dairy-free tacos,
- style-based tacos,
- spicy / mild variations,
- edge cases with multiple simultaneous constraints.

---

## Data Generation Strategy

### Recommended Approach

Use a semi-synthetic dataset generation pipeline:

1. Create templates for user requests.
2. Generate structured taco examples manually or with controlled scripting.
3. Review outputs for consistency.
4. Save as JSONL for fine-tuning.

### Why This Works

It is faster than collecting real taco dialogues and easier to control for evaluation.

---

## Fine-Tuning Plan

### Objective

Train a LoRA adapter that improves:
- JSON validity,
- adherence to constraints,
- domain consistency.

### Training Method

Use PEFT / LoRA on an instruction model.

### Suggested Process

1. Load base instruct model.
2. Format training examples into prompt-completion pairs.
3. Apply LoRA config.
4. Train for a small number of epochs.
5. Save adapter checkpoint.
6. Evaluate against base model.

### Important Notes

- Keep training parameters modest.
- Prioritize a clean training run over aggressive optimization.
- Document hyperparameters clearly.

### Hyperparameters to Record

Document at least:
- learning rate,
- batch size,
- LoRA rank,
- LoRA alpha,
- number of epochs,
- max sequence length,
- optimizer,
- training runtime.

This supports strong technical documentation.

---

## Evaluation Plan

This is one of the most important sections for an A.

### Core Evaluation Question

Does the fine-tuned model outperform the base model on taco generation reliability?

### Evaluation Dataset

Create a held-out dataset separate from the training data.

Target around:
- 30 to 100 evaluation prompts,
- enough variety to demonstrate meaningful comparison.

### Evaluation Prompt Categories

Include:
- simple constraint prompts,
- multi-constraint prompts,
- contradictory prompts,
- ingredient exclusion prompts,
- style-based prompts,
- follow-up or memory-aware prompts if supported.

### Example Evaluation Cases

1. "High protein taco under 400 calories"
2. "Vegan taco with low carbs"
3. "No dairy spicy chicken taco"
4. "Keto taco that feels authentic"
5. "Street taco under 350 calories with beef"
6. "Vegetarian taco, mild, under 300 calories"

---

## Evaluation Metrics

### 1. JSON Validity Rate

Measure:
- how often the response parses successfully as JSON.

Formula:

```text
valid_json_rate = valid_json_outputs / total_outputs
```

### 2. Constraint Satisfaction Rate

Measure whether the output follows requested constraints.

Examples:
- calorie under threshold,
- no forbidden ingredients,
- protein target approximately met,
- dietary constraints respected.

Formula:

```text
constraint_satisfaction_rate = satisfied_constraints / total_constraints_checked
```

### 3. Required Field Completeness

Measure whether all required keys are present.

Formula:

```text
field_completeness_rate = outputs_with_all_fields / total_outputs
```

### 4. Contradiction Rate

Examples of contradictions:
- "vegan" taco includes chicken,
- "no dairy" taco includes cheese,
- "under 400 calories" output says 520 calories.

Formula:

```text
contradiction_rate = contradictory_outputs / total_outputs
```

### 5. Optional Human Quality Score

You may also score outputs on a 1 to 5 scale for:
- plausibility,
- flavor coherence,
- usefulness.

This is optional, but can strengthen the presentation if done honestly.

---

## Base vs Fine-Tuned Comparison

You should compare at least two systems:

### System A
Base model with prompt engineering only

### System B
Fine-tuned / LoRA model with same inference wrapper

Optional:
### System C
Base model plus stronger formatting prompt, if you want a middle baseline

---

## Expected Result Narrative

Your presentation should be able to say something like:

> The fine-tuned model improved JSON validity, reduced contradictions, and followed user constraints more consistently than the base model. The biggest gains were seen in multi-constraint prompts such as low-calorie, dairy-free, high-protein requests.

That kind of conclusion makes the project feel rigorous.

---

## Inference Pipeline

This section matters for the rubric.

### Pipeline Stages

#### 1. Receive Request
The API accepts a user message and optional memory context.

#### 2. Extract Constraints
Regex or rule-based parser extracts structured constraints.

#### 3. Build Prompt
Prompt includes:
- system instructions,
- structured output schema,
- extracted constraints,
- memory preferences if present.

#### 4. Call Model
Run generation using explicit sampling settings.

#### 5. Parse Output
Attempt to parse JSON.

#### 6. Validate Output
Check required fields and obvious contradictions.

#### 7. Retry or Repair
If invalid:
- retry once with a corrective prompt,
- or reject with a safe error.

#### 8. Return API Response
Return clean JSON to frontend.

---

## Sampling Method

The rubric explicitly mentions sampling method, so document it.

### Recommended Sampling Settings

For deterministic, structured output:
- `temperature`: low, such as `0.2` to `0.4`
- `top_p`: `0.9`
- `max_new_tokens`: enough to complete structured JSON
- `do_sample`: true or false depending on model behavior

### Why Low Temperature

Low temperature reduces randomness and helps structured output consistency.

### Suggested Documentation Language

> A low-temperature sampling strategy was used to balance response diversity with reliable JSON generation. This was important because the task required consistent constraint-aware structured outputs rather than highly creative free-form text.

---

## Production Environment Plan

### Recommended Environment

Deploy the backend as a local FastAPI service first. This satisfies the requirement if:
- it runs reliably,
- it is accessible via API endpoint,
- it is documented clearly.

Optional stretch:
- deploy to AWS EC2 or containerized cloud instance.

### Required Endpoint

At minimum:

```text
POST /generate-taco
```

Optional:
```text
GET /health
POST /evaluate
```

---

## API Design

### POST /generate-taco

#### Request

```json
{
  "message": "Generate a spicy high protein taco under 400 calories with no dairy.",
  "memory": {
    "high_protein": true,
    "no_dairy": true
  }
}
```

#### Response

```json
{
  "data": {
    "name": "Chipotle Chicken Street Taco",
    "ingredients": [
      "corn tortillas",
      "grilled chicken breast",
      "shredded cabbage",
      "pickled onion",
      "chipotle salsa"
    ],
    "calories": 372,
    "protein": 35,
    "carbs": 25,
    "fat": 10,
    "dietary_tags": ["high_protein", "dairy_free"],
    "spice_level": "hot",
    "reasoning": "This taco uses lean chicken for high protein, avoids dairy, and stays under the requested calorie limit."
  },
  "metadata": {
    "model": "tacollm-lora-v1",
    "valid_json": true
  }
}
```

### GET /health

Returns service health and model readiness.

Example:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

## Frontend Plan

### Main UI Components

#### Chat Window
Shows user and assistant messages.

#### Input Box
Allows natural language prompt entry.

#### Taco Card
Renders structured response fields:
- taco name,
- ingredients,
- macros,
- tags,
- reasoning.

#### Optional Debug Panel
In development mode, show:
- parsed constraints,
- selected model,
- JSON validation status.

### UX Goal

The UI should look polished enough for demo, but simple enough to finish.

---

## Folder Structure

```text
tacollm/
├── frontend/
│   ├── app/
│   │   ├── page.tsx
│   │   └── api-proxy/
│   ├── components/
│   │   ├── Chat.tsx
│   │   ├── TacoCard.tsx
│   │   └── MessageBubble.tsx
│   └── lib/
│       └── types.ts
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── inference.py
│   │   ├── prompts.py
│   │   ├── parser.py
│   │   ├── validator.py
│   │   └── memory.py
│   ├── training/
│   │   ├── train_lora.py
│   │   ├── dataset_builder.py
│   │   └── format_data.py
│   ├── evaluation/
│   │   ├── eval_dataset.json
│   │   ├── run_eval.py
│   │   ├── metrics.py
│   │   └── compare_models.py
│   └── requirements.txt
├── data/
│   ├── train.jsonl
│   └── eval.json
├── docs/
│   ├── architecture.md
│   ├── api.md
│   ├── evaluation.md
│   ├── deployment.md
│   └── presentation.md
├── README.md
└── CLAUDE.md
```

---

## Prompt Design

### System Prompt Goals

The system prompt should enforce:
- taco expert role,
- strict JSON output,
- realistic ingredient selection,
- constraint obedience,
- no extra explanation outside JSON.

### Example System Prompt

```text
You are TacoLLM, an expert taco recommendation assistant.

Your task is to generate realistic taco recommendations that satisfy user dietary and nutrition constraints.

You must return only valid JSON.
Do not include markdown.
Do not include commentary outside the JSON.
Use the exact required schema.

Rules:
- Respect all calorie and dietary constraints.
- Do not include forbidden ingredients.
- Keep ingredient lists realistic.
- Ensure calories, protein, carbs, and fat are plausible.
- reasoning must briefly explain how the taco satisfies the constraints.
```

### Dynamic Prompt Content

Add:
- user request,
- parsed constraints,
- memory preferences,
- exact schema example.

---

## Validation Rules

### JSON Validation

The output must parse as JSON.

### Schema Validation

Check:
- all required keys present,
- numeric types valid,
- arrays valid,
- spice level allowed.

### Constraint Validation

Check simple contradictions such as:
- dairy requested to be excluded but cheese appears,
- vegan but meat included,
- calorie threshold exceeded.

### Retry Strategy

If parsing fails:
1. retry once with a stronger corrective instruction,
2. if still invalid, return an error payload.

This makes the inference pipeline more robust and easy to explain.

---

## Innovation Positioning

To score well on innovation, present the project as a constrained generation reliability study, not just a novelty taco bot.

### Strong Positioning Statement

> TacoLLM explores how lightweight fine-tuning can improve the reliability of small instruction models on a structured, constraint-sensitive generation task. The taco domain provides a fun interface, but the underlying contribution is an evaluation-driven comparison of base and adapted model performance.

That sounds much more advanced than "I made a taco chatbot."

---

## Documentation Plan

You should produce documentation in stages.

### README.md

Must include:
- project overview,
- how to run frontend,
- how to run backend,
- how to run evaluation,
- sample request and response.

### docs/architecture.md

Include:
- system components,
- request flow,
- model pipeline,
- deployment overview.

### docs/api.md

Include:
- endpoint list,
- request schemas,
- response schemas,
- error handling.

### docs/evaluation.md

Include:
- evaluation dataset creation,
- metrics,
- scoring logic,
- result interpretation.

### docs/deployment.md

Include:
- environment setup,
- local run instructions,
- dependency installation,
- how to access the API.

### docs/presentation.md

Include:
- demo order,
- talking points,
- key charts,
- likely questions.

---

## Presentation Plan

### Presentation Narrative

#### Slide 1 — Problem
LLMs often struggle with structured outputs and user constraints.

#### Slide 2 — Project
TacoLLM is a fine-tuned taco recommendation system that produces structured, constraint-aware outputs.

#### Slide 3 — Model Approach
Base model plus LoRA fine-tuning on taco instruction data.

#### Slide 4 — Inference Pipeline
Constraint extraction, prompt building, generation, validation, API response.

#### Slide 5 — Evaluation
Held-out dataset comparing base vs fine-tuned model.

#### Slide 6 — Results
Show validity and constraint adherence gains.

#### Slide 7 — Demo
Live request through chat interface.

#### Slide 8 — Reflection
What improved, what still needs work, and why evaluation mattered.

---

## Demo Plan

A good demo should be short, clean, and predictable.

### Demo Sequence

#### Demo 1
Prompt:
"Give me a high protein taco under 400 calories."

Show:
- parsed constraints,
- structured taco card,
- API response success.

#### Demo 2
Prompt:
"Make it spicy and keep it dairy free."

Show:
- memory carrying prior constraints,
- updated taco output.

#### Demo 3
Prompt:
"Now make it vegan under 350 calories."

Show:
- constraint shift,
- correct ingredient changes,
- system flexibility.

#### Demo 4
Show evaluation chart comparing base vs fine-tuned model.

This combination demonstrates both product polish and technical rigor.

---

## Evaluation Result Reporting Format

At the end of evaluation, generate a small report table.

### Example

```text
Model Comparison Summary

Metric                          Base Model     LoRA Model
--------------------------------------------------------
Valid JSON Rate                 0.81           0.97
Field Completeness Rate         0.84           0.98
Constraint Satisfaction Rate    0.68           0.90
Contradiction Rate              0.19           0.05
Avg Human Quality Score         3.4            4.3
```

Even if your gains are smaller, this format makes the project feel rigorous.

---

## Risks and Mitigations

### Risk 1
Fine-tuning takes too long or is unstable.

### Mitigation
Use a smaller model and smaller dataset.
Prioritize a completed LoRA run over perfection.

### Risk 2
The model does not produce valid JSON consistently.

### Mitigation
Use stronger schema prompting, low temperature, and retry logic.

### Risk 3
Evaluation becomes too subjective.

### Mitigation
Use objective metrics like:
- JSON validity,
- field completeness,
- explicit constraint checks.

### Risk 4
Frontend takes too long.

### Mitigation
Keep frontend minimal and polished, not feature-heavy.

---

## What Makes This an A-Level Version

This version is A-level because it includes all of the following:

1. a working model,
2. LoRA or fine-tuning,
3. an evaluation dataset,
4. measurable comparison,
5. a documented inference pipeline,
6. API deployment,
7. a clean demo,
8. a credible research-style framing.

The difference between a good project and an A-level project is the evaluation and framing.

---

## Development Phases

### Phase 1 — Project Setup
- initialize frontend and backend
- select model
- define schema
- implement base API route

### Phase 2 — Base Inference Pipeline
- constraint parser
- prompt builder
- JSON validator
- health endpoint

### Phase 3 — Frontend Chat UI
- input box
- message rendering
- taco card output

### Phase 4 — Dataset Creation
- build train dataset
- build held-out eval dataset

### Phase 5 — LoRA Fine-Tuning
- run training
- save adapter
- load checkpoint

### Phase 6 — Evaluation
- compare base vs fine-tuned
- generate metrics report
- create chart for presentation

### Phase 7 — Documentation
- complete README and docs
- write demo script
- finalize architecture notes

### Phase 8 — Presentation Prep
- rehearse demo
- prepare fallback screenshots
- polish narrative

---

## Minimum Viable A-Version

If time gets tight, the minimum version that can still score well should include:

- one working base model,
- one successful LoRA run,
- one held-out eval dataset,
- one model comparison report,
- one FastAPI endpoint,
- one simple frontend,
- complete documentation.

Do not sacrifice evaluation for fancy features.

---

## Stretch Goals

Only do these if the core project is already complete.

### Stretch Goal 1
Add multiple taco styles such as:
- street,
- Tex-Mex,
- healthy,
- breakfast.

### Stretch Goal 2
Add a nutrition lookup helper for ingredients.

### Stretch Goal 3
Add a small chart page in the frontend showing evaluation results.

### Stretch Goal 4
Add a debug mode that shows parsed constraints and validation steps.

These are nice extras, but not necessary for the grade.

---

## Definition of Done

The project is done when all of the following are true:

1. The model can be queried through an API endpoint.
2. The frontend can send user prompts and display structured results.
3. A LoRA or fine-tuned version of the model exists and runs.
4. A held-out evaluation dataset has been created.
5. Base vs fine-tuned comparison has been executed.
6. Metrics are documented clearly.
7. The inference pipeline is documented.
8. Sampling method is documented.
9. Deployment instructions are complete.
10. Demo is rehearsed and easy to follow.

---

## Final Positioning for Instructor

If you need a concise one-paragraph explanation of the project, use this:

> TacoLLM is a fine-tuned, constraint-aware LLM system for taco recommendation generation. The project evaluates whether LoRA adaptation improves a base instruction model’s ability to produce valid structured JSON while respecting nutritional and dietary constraints. It includes a production-style inference API, a chat frontend, a held-out evaluation dataset, explicit sampling strategy, and a comparative evaluation pipeline.

---

## Final Advice

Keep the domain fun, but keep the framing serious.

The tacos make the project memorable.
The evaluation makes the project credible.
The deployment makes the project complete.
The documentation makes the project score highly.

That combination is what turns this from a themed demo into an A-level final project.
