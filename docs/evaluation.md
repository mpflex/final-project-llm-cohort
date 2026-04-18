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
