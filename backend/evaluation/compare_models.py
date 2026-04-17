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
