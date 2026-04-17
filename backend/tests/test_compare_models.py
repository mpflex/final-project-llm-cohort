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
