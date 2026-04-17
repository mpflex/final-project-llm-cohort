# Evaluation Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `backend/evaluation/` — metrics, model comparison, and `run_full_evaluation(pipeline)` — so the `/evaluate` API endpoint works end-to-end.

**Architecture:** Three focused modules: `metrics.py` (pure scoring functions), `compare_models.py` (comparison report formatter), and `run_eval.py` (orchestrator that loads the held-out eval dataset, runs inference via the injected `InferencePipeline`, scores both model variants, and returns a structured JSON report). All modules use absolute imports so they work both from pytest (`backend/` on sys.path) and from the FastAPI app.

**Tech Stack:** Python stdlib only (json, pathlib, logging) — no new dependencies. pytest + unittest.mock for tests.

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `backend/evaluation/__init__.py` | Makes `evaluation` a package |
| Create | `backend/evaluation/metrics.py` | Pure scoring functions operating on result-list dicts |
| Create | `backend/evaluation/compare_models.py` | Delta comparison + text table formatter |
| Create | `backend/evaluation/run_eval.py` | Loads eval dataset, runs both models, aggregates |
| Create | `backend/tests/test_metrics.py` | Unit tests for all metric functions |
| Create | `backend/tests/test_compare_models.py` | Unit tests for comparison logic and table format |
| Create | `backend/tests/test_run_eval.py` | Integration tests with mocked pipeline |
| Modify | `backend/app/main.py:140` | Fix `..evaluation` relative import → absolute `evaluation` import |
| Modify | `backend/pyproject.toml` | Add `evaluation` to coverage paths |

---

## Result Dict Schema

Every `evaluate_model()` call produces a list of these dicts. All downstream functions operate on this shape.

```python
{
    "instruction": str,        # original user prompt
    "valid_json": bool,        # did pipeline.generate return non-None?
    "parsed": dict | None,     # the taco dict returned by pipeline, or None
    "constraints": dict,       # output of ConstraintParser.extract(instruction)
    "validation_issues": list[str],  # TacoValidator.validate() output, or ["Invalid JSON output"]
    "attempts": int,           # number of inference attempts made
}
```

---

## Task 1: Package skeleton + fix main.py import

**Files:**
- Create: `backend/evaluation/__init__.py`
- Modify: `backend/app/main.py:140`

The existing `from ..evaluation.run_eval import run_full_evaluation` uses a relative import that walks up from `app/` to `backend/` — but `backend/` has no `__init__.py`, so Python can't resolve `..`. Replace it with an absolute import.

- [ ] **Step 1: Create the evaluation package**

```python
# backend/evaluation/__init__.py
```
(Empty file — just makes `evaluation` a package.)

- [ ] **Step 2: Fix the import in main.py**

Change line 140 in `backend/app/main.py` from:
```python
    from ..evaluation.run_eval import run_full_evaluation
```
to:
```python
    from evaluation.run_eval import run_full_evaluation
```

- [ ] **Step 3: Verify existing tests still pass**

Run from `backend/`:
```
uv run pytest tests/ -q
```
Expected: `115 passed`

- [ ] **Step 4: Commit**

```bash
git add backend/evaluation/__init__.py backend/app/main.py
git commit -m "feat: add evaluation package skeleton, fix relative import in main.py"
```

---

## Task 2: metrics.py — pure scoring functions

**Files:**
- Create: `backend/evaluation/metrics.py`
- Create: `backend/tests/test_metrics.py`

### Metric definitions

- `json_validity_rate` — fraction of results where `valid_json=True`
- `field_completeness_rate` — fraction of results where `valid_json=True` AND no `"Missing required field"` in issues
- `constraint_satisfaction_rate` — fraction of results where `valid_json=True` AND no constraint-violation issues (calorie / dairy / beef / vegan)
- `contradiction_rate` — fraction of results where `valid_json=True` AND has at least one constraint-violation issue

All four use `len(results)` as denominator (including failed/invalid-JSON entries). Returns `0.0` for empty input.

Constraint-violation issues are identified by these substrings: `"exceeds max"`, `"but no_dairy constraint is set"`, `"but no_beef constraint is set"`, `"but vegan constraint is set"`.

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/test_metrics.py
"""Tests for evaluation/metrics.py"""

import pytest
from evaluation.metrics import (
    aggregate_metrics,
    constraint_satisfaction_rate,
    contradiction_rate,
    field_completeness_rate,
    json_validity_rate,
)


def _r(valid_json=True, issues=None):
    """Helper — build a minimal result dict."""
    return {
        "instruction": "test prompt",
        "valid_json": valid_json,
        "parsed": {} if valid_json else None,
        "constraints": {},
        "validation_issues": issues or [],
        "attempts": 1,
    }


# ---------------------------------------------------------------------------
# json_validity_rate
# ---------------------------------------------------------------------------


def test_json_validity_rate_empty():
    assert json_validity_rate([]) == 0.0


def test_json_validity_rate_all_valid():
    results = [_r(True), _r(True)]
    assert json_validity_rate(results) == 1.0


def test_json_validity_rate_all_invalid():
    results = [_r(False), _r(False)]
    assert json_validity_rate(results) == 0.0


def test_json_validity_rate_mixed():
    results = [_r(True), _r(False)]
    assert json_validity_rate(results) == 0.5


# ---------------------------------------------------------------------------
# field_completeness_rate
# ---------------------------------------------------------------------------


def test_field_completeness_rate_empty():
    assert field_completeness_rate([]) == 0.0


def test_field_completeness_rate_complete():
    results = [_r(True, []), _r(True, [])]
    assert field_completeness_rate(results) == 1.0


def test_field_completeness_rate_missing_field():
    results = [_r(True, ["Missing required field: name"]), _r(True, [])]
    assert field_completeness_rate(results) == 0.5


def test_field_completeness_rate_invalid_json_counts_incomplete():
    results = [_r(False, ["Invalid JSON output"]), _r(True, [])]
    assert field_completeness_rate(results) == 0.5


# ---------------------------------------------------------------------------
# constraint_satisfaction_rate
# ---------------------------------------------------------------------------


def test_constraint_satisfaction_rate_empty():
    assert constraint_satisfaction_rate([]) == 0.0


def test_constraint_satisfaction_rate_all_satisfied():
    results = [_r(True, []), _r(True, [])]
    assert constraint_satisfaction_rate(results) == 1.0


def test_constraint_satisfaction_rate_calorie_violation():
    results = [_r(True, ["Calories 500 exceeds max 400"]), _r(True, [])]
    assert constraint_satisfaction_rate(results) == 0.5


def test_constraint_satisfaction_rate_dairy_violation():
    results = [_r(True, ["Contains dairy ingredient 'cheese' but no_dairy constraint is set"])]
    assert constraint_satisfaction_rate(results) == 0.0


def test_constraint_satisfaction_rate_vegan_violation():
    results = [_r(True, ["Contains meat ingredient 'chicken' but vegan constraint is set"])]
    assert constraint_satisfaction_rate(results) == 0.0


def test_constraint_satisfaction_rate_beef_violation():
    results = [_r(True, ["Contains beef ingredient 'beef' but no_beef constraint is set"])]
    assert constraint_satisfaction_rate(results) == 0.0


def test_constraint_satisfaction_rate_invalid_json_penalises():
    # Invalid JSON is not satisfied — denominator is total
    results = [_r(False, ["Invalid JSON output"]), _r(True, [])]
    assert constraint_satisfaction_rate(results) == 0.5


# ---------------------------------------------------------------------------
# contradiction_rate
# ---------------------------------------------------------------------------


def test_contradiction_rate_empty():
    assert contradiction_rate([]) == 0.0


def test_contradiction_rate_no_contradictions():
    results = [_r(True, []), _r(True, [])]
    assert contradiction_rate(results) == 0.0


def test_contradiction_rate_with_contradiction():
    results = [
        _r(True, ["Calories 500 exceeds max 400"]),
        _r(True, []),
    ]
    assert contradiction_rate(results) == 0.5


def test_contradiction_rate_invalid_json_not_counted():
    # Invalid JSON is not a contradiction — only valid-JSON outputs can contradict
    results = [_r(False, ["Invalid JSON output"]), _r(True, [])]
    assert contradiction_rate(results) == 0.0


# ---------------------------------------------------------------------------
# aggregate_metrics
# ---------------------------------------------------------------------------


def test_aggregate_metrics_returns_all_keys():
    results = [_r(True, []), _r(False)]
    out = aggregate_metrics(results)
    assert set(out.keys()) == {
        "total",
        "json_validity_rate",
        "field_completeness_rate",
        "constraint_satisfaction_rate",
        "contradiction_rate",
    }


def test_aggregate_metrics_total():
    results = [_r(), _r(), _r()]
    assert aggregate_metrics(results)["total"] == 3


def test_aggregate_metrics_values_rounded_to_4dp():
    results = [_r(True), _r(True), _r(False)]
    out = aggregate_metrics(results)
    # 2/3 valid = 0.6667
    assert out["json_validity_rate"] == round(2 / 3, 4)


def test_aggregate_metrics_empty():
    out = aggregate_metrics([])
    assert out["total"] == 0
    assert out["json_validity_rate"] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_metrics.py -v
```
Expected: `ImportError` or `ModuleNotFoundError` for `evaluation.metrics`

- [ ] **Step 3: Implement metrics.py**

```python
# backend/evaluation/metrics.py
"""
TacoLLM — Evaluation Metrics

Pure scoring functions operating on result-list dicts produced by evaluate_model().
Each function accepts a list of result dicts and returns a float in [0.0, 1.0].
"""

from typing import Any, Dict, List

_CONSTRAINT_VIOLATION_MARKERS = [
    "exceeds max",
    "but no_dairy constraint is set",
    "but no_beef constraint is set",
    "but vegan constraint is set",
]


def json_validity_rate(results: List[Dict[str, Any]]) -> float:
    """Fraction of results where the model produced valid JSON."""
    if not results:
        return 0.0
    return sum(1 for r in results if r["valid_json"]) / len(results)


def field_completeness_rate(results: List[Dict[str, Any]]) -> float:
    """Fraction of results with valid JSON and all required fields present."""
    if not results:
        return 0.0
    complete = sum(
        1
        for r in results
        if r["valid_json"]
        and not any("Missing required field" in issue for issue in r["validation_issues"])
    )
    return complete / len(results)


def constraint_satisfaction_rate(results: List[Dict[str, Any]]) -> float:
    """Fraction of results with valid JSON and no constraint violations."""
    if not results:
        return 0.0
    satisfied = sum(
        1
        for r in results
        if r["valid_json"]
        and not any(
            any(marker in issue for marker in _CONSTRAINT_VIOLATION_MARKERS)
            for issue in r["validation_issues"]
        )
    )
    return satisfied / len(results)


def contradiction_rate(results: List[Dict[str, Any]]) -> float:
    """Fraction of results with valid JSON that contain a constraint contradiction."""
    if not results:
        return 0.0
    contradictions = sum(
        1
        for r in results
        if r["valid_json"]
        and any(
            any(marker in issue for marker in _CONSTRAINT_VIOLATION_MARKERS)
            for issue in r["validation_issues"]
        )
    )
    return contradictions / len(results)


def aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute and return all evaluation metrics for a result list."""
    return {
        "total": len(results),
        "json_validity_rate": round(json_validity_rate(results), 4),
        "field_completeness_rate": round(field_completeness_rate(results), 4),
        "constraint_satisfaction_rate": round(constraint_satisfaction_rate(results), 4),
        "contradiction_rate": round(contradiction_rate(results), 4),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest tests/test_metrics.py -v
```
Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
git add backend/evaluation/metrics.py backend/tests/test_metrics.py
git commit -m "feat: add evaluation metrics with full test coverage"
```

---

## Task 3: compare_models.py — delta comparison and table formatter

**Files:**
- Create: `backend/evaluation/compare_models.py`
- Create: `backend/tests/test_compare_models.py`

`compare_models()` takes two metric dicts (from `aggregate_metrics`) and returns a per-metric dict of `{base, lora, delta}`. `format_comparison_table()` returns a human-readable string suitable for logging and the demo.

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/test_compare_models.py
"""Tests for evaluation/compare_models.py"""

from evaluation.compare_models import compare_models, format_comparison_table

_BASE = {
    "total": 10,
    "json_validity_rate": 0.8,
    "field_completeness_rate": 0.8,
    "constraint_satisfaction_rate": 0.6,
    "contradiction_rate": 0.2,
}

_LORA = {
    "total": 10,
    "json_validity_rate": 0.95,
    "field_completeness_rate": 0.95,
    "constraint_satisfaction_rate": 0.85,
    "contradiction_rate": 0.05,
}


def test_compare_models_keys():
    result = compare_models(_BASE, _LORA)
    assert set(result.keys()) == {
        "json_validity_rate",
        "field_completeness_rate",
        "constraint_satisfaction_rate",
        "contradiction_rate",
    }


def test_compare_models_base_value():
    result = compare_models(_BASE, _LORA)
    assert result["json_validity_rate"]["base"] == 0.8


def test_compare_models_lora_value():
    result = compare_models(_BASE, _LORA)
    assert result["json_validity_rate"]["lora"] == 0.95


def test_compare_models_delta():
    result = compare_models(_BASE, _LORA)
    assert result["json_validity_rate"]["delta"] == round(0.95 - 0.8, 4)


def test_compare_models_equal_metrics_delta_zero():
    result = compare_models(_BASE, _BASE)
    for key in result:
        assert result[key]["delta"] == 0.0


def test_compare_models_negative_delta():
    # Base outperforms lora on a metric
    result = compare_models(_LORA, _BASE)
    assert result["json_validity_rate"]["delta"] < 0


def test_format_comparison_table_is_string():
    comparison = compare_models(_BASE, _LORA)
    table = format_comparison_table(comparison)
    assert isinstance(table, str)


def test_format_comparison_table_contains_metric_names():
    comparison = compare_models(_BASE, _LORA)
    table = format_comparison_table(comparison)
    assert "json_validity_rate" in table
    assert "contradiction_rate" in table


def test_format_comparison_table_contains_values():
    comparison = compare_models(_BASE, _LORA)
    table = format_comparison_table(comparison)
    assert "0.8000" in table
    assert "0.9500" in table
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_compare_models.py -v
```
Expected: `ImportError` for `evaluation.compare_models`

- [ ] **Step 3: Implement compare_models.py**

```python
# backend/evaluation/compare_models.py
"""
TacoLLM — Model Comparison Report

Takes two aggregate-metric dicts (one per model variant) and produces
a per-metric comparison with delta values and a printable text table.
"""

from typing import Any, Dict

_METRIC_KEYS = [
    "json_validity_rate",
    "field_completeness_rate",
    "constraint_satisfaction_rate",
    "contradiction_rate",
]


def compare_models(
    base_metrics: Dict[str, Any],
    lora_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Return a per-metric dict of {base, lora, delta} for each tracked metric.
    Delta = lora - base (positive means LoRA improved).
    """
    comparison = {}
    for key in _METRIC_KEYS:
        base_val = base_metrics.get(key, 0.0)
        lora_val = lora_metrics.get(key, 0.0)
        comparison[key] = {
            "base": base_val,
            "lora": lora_val,
            "delta": round(lora_val - base_val, 4),
        }
    return comparison


def format_comparison_table(comparison: Dict[str, Any]) -> str:
    """Return a human-readable text table of model comparison metrics."""
    header = f"{'Metric':<35} {'Base':>10} {'LoRA':>10} {'Delta':>10}"
    divider = "-" * 67
    rows = [header, divider]
    for key, vals in comparison.items():
        rows.append(
            f"{key:<35} {vals['base']:>10.4f} {vals['lora']:>10.4f} {vals['delta']:>+10.4f}"
        )
    return "\n".join(rows)
```

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest tests/test_compare_models.py -v
```
Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
git add backend/evaluation/compare_models.py backend/tests/test_compare_models.py
git commit -m "feat: add model comparison report with text table formatter"
```

---

## Task 4: run_eval.py — dataset loader and model evaluator

**Files:**
- Create: `backend/evaluation/run_eval.py`
- Create: `backend/tests/test_run_eval.py`

`load_eval_dataset(path)` loads `data/eval.json` (default path resolved relative to this file's location). `evaluate_model(pipeline, dataset, model_variant)` calls `pipeline.generate()` for each item and returns a list of result dicts. `run_full_evaluation(pipeline)` orchestrates both model variants and returns the final report.

The eval dataset path: `Path(__file__).resolve().parent.parent.parent / "data" / "eval.json"` resolves to `final/data/eval.json` from `final/backend/evaluation/run_eval.py`.

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/test_run_eval.py
"""Tests for evaluation/run_eval.py"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evaluation.run_eval import evaluate_model, load_eval_dataset, run_full_evaluation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_TACO = {
    "name": "Test Taco",
    "ingredients": ["corn tortilla", "chicken"],
    "calories": 350,
    "protein": 30,
    "carbs": 25,
    "fat": 10,
    "dietary_tags": ["high_protein"],
    "spice_level": "medium",
    "reasoning": "Lean chicken keeps protein high.",
}

_DATASET = [
    {"instruction": "Give me a high protein taco under 400 calories.", "output": _VALID_TACO},
    {"instruction": "Make me a vegan taco.", "output": _VALID_TACO},
]


def _mock_pipeline(taco=_VALID_TACO, valid_json=True, attempts=1):
    pipeline = MagicMock()
    pipeline.generate.return_value = (taco, valid_json, attempts)
    return pipeline


# ---------------------------------------------------------------------------
# load_eval_dataset
# ---------------------------------------------------------------------------


def test_load_eval_dataset_from_temp_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(_DATASET, f)
        tmp_path = Path(f.name)
    loaded = load_eval_dataset(tmp_path)
    assert len(loaded) == 2
    assert loaded[0]["instruction"] == "Give me a high protein taco under 400 calories."


def test_load_eval_dataset_real_file():
    """Smoke test: the actual eval.json loads without error."""
    real_path = Path(__file__).resolve().parent.parent.parent / "data" / "eval.json"
    dataset = load_eval_dataset(real_path)
    assert len(dataset) > 0
    assert "instruction" in dataset[0]


# ---------------------------------------------------------------------------
# evaluate_model
# ---------------------------------------------------------------------------


def test_evaluate_model_result_count():
    pipeline = _mock_pipeline()
    results = evaluate_model(pipeline, _DATASET, "base")
    assert len(results) == 2


def test_evaluate_model_calls_pipeline_for_each_item():
    pipeline = _mock_pipeline()
    evaluate_model(pipeline, _DATASET, "base")
    assert pipeline.generate.call_count == 2


def test_evaluate_model_passes_model_variant():
    pipeline = _mock_pipeline()
    evaluate_model(pipeline, _DATASET, "lora")
    for call in pipeline.generate.call_args_list:
        assert call.kwargs["model_variant"] == "lora"


def test_evaluate_model_result_shape():
    pipeline = _mock_pipeline()
    results = evaluate_model(pipeline, _DATASET, "base")
    r = results[0]
    assert "instruction" in r
    assert "valid_json" in r
    assert "parsed" in r
    assert "constraints" in r
    assert "validation_issues" in r
    assert "attempts" in r


def test_evaluate_model_valid_json_flag_propagated():
    pipeline = _mock_pipeline(valid_json=True)
    results = evaluate_model(pipeline, _DATASET, "base")
    assert all(r["valid_json"] for r in results)


def test_evaluate_model_invalid_json_sets_flag_false():
    pipeline = _mock_pipeline(taco=None, valid_json=False)
    results = evaluate_model(pipeline, _DATASET, "base")
    assert all(not r["valid_json"] for r in results)


def test_evaluate_model_invalid_json_sets_fallback_issue():
    pipeline = _mock_pipeline(taco=None, valid_json=False)
    results = evaluate_model(pipeline, _DATASET, "base")
    assert results[0]["validation_issues"] == ["Invalid JSON output"]


def test_evaluate_model_extracts_constraints_from_instruction():
    pipeline = _mock_pipeline()
    results = evaluate_model(pipeline, _DATASET, "base")
    # "under 400 calories" in first instruction → max_calories=400
    assert results[0]["constraints"].get("max_calories") == 400


# ---------------------------------------------------------------------------
# run_full_evaluation
# ---------------------------------------------------------------------------


def test_run_full_evaluation_keys():
    pipeline = _mock_pipeline()
    result = run_full_evaluation(pipeline)
    assert set(result.keys()) == {"base", "lora", "comparison", "summary_table"}


def test_run_full_evaluation_base_has_metric_keys():
    pipeline = _mock_pipeline()
    result = run_full_evaluation(pipeline)
    assert "json_validity_rate" in result["base"]
    assert "total" in result["base"]


def test_run_full_evaluation_comparison_has_delta():
    pipeline = _mock_pipeline()
    result = run_full_evaluation(pipeline)
    assert "delta" in result["comparison"]["json_validity_rate"]


def test_run_full_evaluation_summary_table_is_string():
    pipeline = _mock_pipeline()
    result = run_full_evaluation(pipeline)
    assert isinstance(result["summary_table"], str)


def test_run_full_evaluation_calls_both_variants():
    pipeline = _mock_pipeline()
    run_full_evaluation(pipeline)
    variants_called = {call.kwargs["model_variant"] for call in pipeline.generate.call_args_list}
    assert variants_called == {"base", "lora"}
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_run_eval.py -v
```
Expected: `ImportError` for `evaluation.run_eval`

- [ ] **Step 3: Implement run_eval.py**

```python
# backend/evaluation/run_eval.py
"""
TacoLLM — Evaluation Runner

Loads the held-out eval dataset, runs inference through both model variants,
scores the outputs, and returns a structured comparison report.

Entry point: run_full_evaluation(pipeline) — called by the /evaluate API endpoint.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from app.parser import ConstraintParser
from app.validator import TacoValidator

from .compare_models import compare_models, format_comparison_table
from .metrics import aggregate_metrics

logger = logging.getLogger(__name__)

EVAL_DATASET_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "eval.json"

_parser = ConstraintParser()
_validator = TacoValidator()


def load_eval_dataset(path: Path = EVAL_DATASET_PATH) -> List[Dict[str, Any]]:
    """Load and return the evaluation dataset from a JSON file."""
    with open(path) as f:
        return json.load(f)


def evaluate_model(
    pipeline,
    dataset: List[Dict[str, Any]],
    model_variant: str,
) -> List[Dict[str, Any]]:
    """
    Run inference for every item in the dataset using the given model variant.

    Returns a list of result dicts ready for metrics scoring.
    """
    results = []
    for item in dataset:
        instruction = item["instruction"]
        constraints = _parser.extract(instruction)
        taco, valid_json, attempts = pipeline.generate(
            user_message=instruction,
            constraints=constraints,
            model_variant=model_variant,
        )
        issues = (
            _validator.validate(taco, constraints)
            if taco is not None
            else ["Invalid JSON output"]
        )
        results.append(
            {
                "instruction": instruction,
                "valid_json": valid_json,
                "parsed": taco,
                "constraints": constraints,
                "validation_issues": issues,
                "attempts": attempts,
            }
        )
    return results


def run_full_evaluation(pipeline) -> Dict[str, Any]:
    """
    Entry point called by the /evaluate API endpoint.

    Runs both model variants against the held-out eval dataset,
    computes metrics, and returns a structured comparison report.
    """
    logger.info("Loading eval dataset ...")
    dataset = load_eval_dataset()
    logger.info(f"Evaluating {len(dataset)} prompts × 2 models ...")

    base_results = evaluate_model(pipeline, dataset, "base")
    lora_results = evaluate_model(pipeline, dataset, "lora")

    base_metrics = aggregate_metrics(base_results)
    lora_metrics = aggregate_metrics(lora_results)
    comparison = compare_models(base_metrics, lora_metrics)

    table = format_comparison_table(comparison)
    logger.info("\n%s", table)

    return {
        "base": base_metrics,
        "lora": lora_metrics,
        "comparison": comparison,
        "summary_table": table,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest tests/test_run_eval.py -v
```
Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
git add backend/evaluation/run_eval.py backend/tests/test_run_eval.py
git commit -m "feat: add evaluation runner with dataset loader and full pipeline orchestration"
```

---

## Task 5: Update coverage config and run full suite

**Files:**
- Modify: `backend/pyproject.toml`

The current `addopts` only covers `app/`. Add `evaluation` to the coverage targets.

- [ ] **Step 1: Update pyproject.toml**

Change `addopts` in `[tool.pytest.ini_options]` from:
```toml
addopts = "--cov=app --cov-report=term-missing"
```
to:
```toml
addopts = "--cov=app --cov-report=term-missing --cov=evaluation"
```

- [ ] **Step 2: Run full test suite**

```
uv run pytest tests/ -q
```
Expected: `145+ passed` (115 existing + ~30 new). Coverage report shows both `app/` and `evaluation/`.

- [ ] **Step 3: Commit**

```bash
git add backend/pyproject.toml
git commit -m "chore: add evaluation package to pytest coverage paths"
```

---

## Self-Review

### Spec Coverage

| Requirement | Covered by |
|---|---|
| `run_full_evaluation(pipeline)` callable from `/evaluate` endpoint | Task 1 (import fix) + Task 4 |
| JSON validity metric | Task 2 |
| Field completeness metric | Task 2 |
| Constraint satisfaction metric | Task 2 |
| Contradiction rate metric | Task 2 |
| Base vs LoRA comparison | Task 3 |
| Printable comparison table | Task 3 |
| Eval dataset loaded from `data/eval.json` | Task 4 |
| Uses `InferencePipeline.generate()` interface | Task 4 |
| Uses `TacoValidator.validate()` for scoring | Task 4 |
| Uses `ConstraintParser.extract()` for constraint extraction | Task 4 |

### Type Consistency Check

- `evaluate_model()` returns `List[Dict]` matching the result-dict schema defined at top
- `aggregate_metrics()` accepts that same list shape → metrics.py ✓
- `compare_models()` accepts two `aggregate_metrics()` outputs → Task 3 tests verify ✓
- `run_full_evaluation()` returns `{base, lora, comparison, summary_table}` → tests verify ✓
- `pipeline.generate(user_message, constraints, model_variant)` matches `InferencePipeline.generate()` signature in `inference.py:161` ✓

### Placeholder Scan

None found.
