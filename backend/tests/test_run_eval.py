# backend/tests/test_run_eval.py
"""Tests for evaluation/run_eval.py"""

import json
from pathlib import Path
from unittest.mock import MagicMock

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


def test_load_eval_dataset_from_temp_file(tmp_path):
    p = tmp_path / "eval.json"
    p.write_text(json.dumps(_DATASET))
    loaded = load_eval_dataset(p)
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
