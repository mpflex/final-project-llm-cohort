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
