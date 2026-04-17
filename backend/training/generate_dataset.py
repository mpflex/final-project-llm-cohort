"""
TacoLLM — Dataset Generation Script

Generates taco instruction/output pairs using the Claude API.
Validates each example before inclusion; skips invalid outputs.

Usage:
    cd backend
    ANTHROPIC_API_KEY=sk-... uv run python -m training.generate_dataset
    ANTHROPIC_API_KEY=sk-... uv run python -m training.generate_dataset --count 5000 --output ../data

Options:
    --count         Total examples to generate (default: 5000)
    --output        Output directory path   (default: ../data)
    --train-split   Fraction for training   (default: 0.94 → 4700/5000)
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

import anthropic

from .dataset_validator import validate_example
from .prompt_templates import get_category_counts, get_prompt_for_category

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are TacoLLM, an expert taco recommendation assistant.
Generate a realistic taco recommendation as valid JSON. No markdown, no commentary.

Required schema:
{
  "name": "string",
  "ingredients": ["array of strings"],
  "calories": number,
  "protein": number,
  "carbs": number,
  "fat": number,
  "dietary_tags": ["array of strings"],
  "spice_level": "mild" | "medium" | "hot",
  "reasoning": "string"
}

Rules:
- Return ONLY valid JSON starting with { and ending with }
- Respect all constraints in the user request
- Use realistic, plausible nutritional values
- Keep ingredients lists realistic (4-7 items)
- spice_level must be exactly: mild, medium, or hot"""


def parse_taco_output(text: str) -> dict[str, Any] | None:
    """
    Extract a JSON object from raw model output.
    Strips markdown code fences if present, then attempts json.loads.
    Falls back to regex extraction of a {...} block.
    Returns None if all parsing attempts fail.
    """
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def call_claude(client: anthropic.Anthropic, instruction: str) -> dict[str, Any] | None:
    """
    Call the Claude API with a taco generation instruction.
    Returns parsed taco dict or None if the call or parsing fails.

    Uses claude-haiku-4-5-20251001 for speed and cost efficiency at dataset scale.
    Temperature set to 1.0 (API max) for maximum variety.
    """
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            temperature=1.0,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": instruction}],
        )
        return parse_taco_output(response.content[0].text)
    except Exception as exc:
        logger.debug(f"Claude API call failed: {exc}")
        return None


def generate_examples(client: anthropic.Anthropic, total: int) -> list[dict[str, Any]]:
    """
    Generate `total` validated examples distributed across all categories.
    Each category slot allows up to 4x attempts to fill its quota.
    """
    category_counts = get_category_counts(total)
    examples: list[dict[str, Any]] = []

    for category, count in category_counts.items():
        logger.info(f"Generating {count} '{category}' examples...")
        generated = 0
        attempts = 0
        max_attempts = count * 4

        while generated < count and attempts < max_attempts:
            instruction = get_prompt_for_category(category)
            output = call_claude(client, instruction)
            attempts += 1

            if output is None:
                continue

            example = {"instruction": instruction, "output": output}
            if validate_example(example):
                examples.append(example)
                generated += 1
                if generated % 100 == 0:
                    logger.info(f"  {category}: {generated}/{count}")

        logger.info(f"  {category}: completed {generated}/{count} (attempts: {attempts})")

    return examples


def save_dataset(
    examples: list[dict[str, Any]],
    output_dir: Path,
    train_split: float,
) -> tuple[int, int]:
    """
    Split examples into train/eval sets and save to disk.

    Saves:
    - train.jsonl   — one JSON object per line
    - eval.json     — JSON array (indented for readability)

    Returns (train_count, eval_count).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    split_idx = int(len(examples) * train_split)
    train = examples[:split_idx]
    eval_set = examples[split_idx:]

    train_path = output_dir / "train.jsonl"
    with open(train_path, "w") as f:
        for ex in train:
            f.write(json.dumps(ex) + "\n")

    eval_path = output_dir / "eval.json"
    with open(eval_path, "w") as f:
        json.dump(eval_set, f, indent=2)

    return len(train), len(eval_set)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TacoLLM training dataset")
    parser.add_argument("--count", type=int, default=5000, help="Total examples to generate")
    parser.add_argument("--output", type=str, default="../data", help="Output directory")
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.94,
        help="Fraction for training set (default: 0.94)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    logger.info(f"Generating {args.count} examples...")

    examples = generate_examples(client, args.count)
    logger.info(f"Generated {len(examples)} valid examples")

    train_count, eval_count = save_dataset(examples, Path(args.output), args.train_split)
    logger.info(f"Saved {train_count} train + {eval_count} eval examples to {args.output}")


if __name__ == "__main__":
    main()
