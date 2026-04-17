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
    logger.info("Evaluating %d prompts × 2 models ...", len(dataset))

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
