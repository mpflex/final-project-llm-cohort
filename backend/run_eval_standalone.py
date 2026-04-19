"""
Standalone evaluation runner — no FastAPI server required.

Usage (from backend/):
    uv run python run_eval_standalone.py
    uv run python run_eval_standalone.py --limit 30   # quick subset
    uv run python run_eval_standalone.py --output results.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit eval to first N prompts")
    parser.add_argument("--output", default="eval_results.json", help="Output JSON file")
    args = parser.parse_args()

    logger.info("Loading inference pipeline (this downloads the base model on first run)...")
    from app.inference import InferencePipeline

    pipeline = InferencePipeline()
    if not pipeline.is_loaded():
        logger.error("Model failed to load. Check HF_TOKEN and network access.")
        sys.exit(1)

    from evaluation.run_eval import load_eval_dataset, evaluate_model
    from evaluation.metrics import aggregate_metrics
    from evaluation.compare_models import compare_models, format_comparison_table

    dataset = load_eval_dataset()
    if args.limit:
        dataset = dataset[: args.limit]
        logger.info(f"Limiting to {args.limit} prompts")
    logger.info(f"Running eval on {len(dataset)} prompts × 2 models ...")

    base_results = evaluate_model(pipeline, dataset, "base")
    lora_results = evaluate_model(pipeline, dataset, "lora")

    base_metrics = aggregate_metrics(base_results)
    lora_metrics = aggregate_metrics(lora_results)
    comparison = compare_models(base_metrics, lora_metrics)
    table = format_comparison_table(comparison)

    print("\n" + table)

    output = {
        "num_prompts": len(dataset),
        "base": base_metrics,
        "lora": lora_metrics,
        "comparison": comparison,
        "summary_table": table,
    }

    Path(args.output).write_text(json.dumps(output, indent=2))
    logger.info(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
