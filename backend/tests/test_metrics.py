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
