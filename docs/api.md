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
