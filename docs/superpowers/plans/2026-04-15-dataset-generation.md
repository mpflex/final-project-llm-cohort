# Dataset Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate 5,000 validated taco instruction/output pairs using the Claude API and save them as a JSONL training set and JSON eval set.

**Architecture:** Three focused modules — `prompt_templates.py` generates varied natural language instructions across 6 constraint categories, `dataset_validator.py` wraps the existing `TacoValidator` to check generated examples before inclusion, and `generate_dataset.py` is the runnable script that drives the Claude API calls, collects validated examples, and splits/saves the dataset. The generation script is not unit-tested directly (it requires a live API key), but its two pure utility functions (`parse_taco_output` and `save_dataset`) have full test coverage.

**Tech Stack:** Python 3.12, `anthropic` SDK (already in `pyproject.toml`), `pytest`, `uv`

---

## File Map

| File | Status | Responsibility |
|------|--------|----------------|
| `backend/training/__init__.py` | Create | Package marker |
| `backend/training/prompt_templates.py` | Create | Instruction generators + category distribution |
| `backend/training/dataset_validator.py` | Create | Per-example schema validation wrapping TacoValidator |
| `backend/training/generate_dataset.py` | Create | Claude API driver, JSON parsing, file save/split |
| `backend/tests/test_prompt_templates.py` | Create | Full coverage of prompt_templates |
| `backend/tests/test_dataset_validator.py` | Create | Full coverage of dataset_validator |
| `backend/tests/test_generate_dataset.py` | Create | Coverage of pure functions in generate_dataset |
| `data/.gitignore` | Create | Prevents large dataset files from being committed |

---

## Task 1: Training Module Setup

**Files:**
- Create: `backend/training/__init__.py`
- Create: `data/.gitignore`

- [ ] **Step 1: Create the training package marker**

Create `backend/training/__init__.py` as an empty file (just a newline).

- [ ] **Step 2: Create `data/.gitignore`**

Create `data/.gitignore`:

```
train.jsonl
eval.json
*.jsonl
*.json
!.gitignore
```

- [ ] **Step 3: Verify the training package is importable**

```bash
cd /Users/mpineda/Desktop/llm-class-2026/llm-class-2026-winter-cohort/backend
uv run python -c "import training; print('training package OK')"
```

Expected output: `training package OK`

---

## Task 2: Prompt Templates — TDD

**Files:**
- Create: `backend/tests/test_prompt_templates.py`
- Create: `backend/training/prompt_templates.py`

- [ ] **Step 1: Write failing tests**

Create `backend/tests/test_prompt_templates.py`:

```python
import pytest

from training.prompt_templates import (
    CATEGORY_DISTRIBUTION,
    edge_case_prompt,
    followup_prompt,
    get_category_counts,
    get_prompt_for_category,
    single_constraint_prompt,
    style_prompt,
    three_constraint_prompt,
    two_constraint_prompt,
)


class TestSingleConstraintPrompt:
    def test_returns_non_empty_string(self):
        assert isinstance(single_constraint_prompt(), str)
        assert len(single_constraint_prompt()) > 0

    def test_varies_on_repeated_calls(self):
        results = {single_constraint_prompt() for _ in range(50)}
        assert len(results) > 1


class TestTwoConstraintPrompt:
    def test_returns_non_empty_string(self):
        result = two_constraint_prompt()
        assert isinstance(result, str) and len(result) > 0


class TestThreeConstraintPrompt:
    def test_returns_non_empty_string(self):
        result = three_constraint_prompt()
        assert isinstance(result, str) and len(result) > 0


class TestStylePrompt:
    def test_returns_non_empty_string(self):
        result = style_prompt()
        assert isinstance(result, str) and len(result) > 0


class TestEdgeCasePrompt:
    def test_returns_non_empty_string(self):
        result = edge_case_prompt()
        assert isinstance(result, str) and len(result) > 0


class TestFollowupPrompt:
    def test_returns_non_empty_string(self):
        result = followup_prompt()
        assert isinstance(result, str) and len(result) > 0


class TestGetPromptForCategory:
    def test_all_valid_categories_return_strings(self):
        for cat in CATEGORY_DISTRIBUTION:
            result = get_prompt_for_category(cat)
            assert isinstance(result, str) and len(result) > 0

    def test_unknown_category_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown category"):
            get_prompt_for_category("unknown")

    def test_single_category_varies(self):
        results = {get_prompt_for_category("single") for _ in range(50)}
        assert len(results) > 1

    def test_two_category_varies(self):
        results = {get_prompt_for_category("two") for _ in range(50)}
        assert len(results) > 1


class TestGetCategoryCounts:
    def test_counts_sum_to_total_5000(self):
        counts = get_category_counts(5000)
        assert sum(counts.values()) == 5000

    def test_counts_sum_to_total_100(self):
        counts = get_category_counts(100)
        assert sum(counts.values()) == 100

    def test_all_categories_present(self):
        counts = get_category_counts(5000)
        assert set(counts.keys()) == set(CATEGORY_DISTRIBUTION.keys())

    def test_no_zero_counts_for_large_total(self):
        counts = get_category_counts(5000)
        assert all(v > 0 for v in counts.values())

    def test_single_is_20_percent(self):
        counts = get_category_counts(5000)
        assert counts["single"] == 1000

    def test_two_is_30_percent(self):
        counts = get_category_counts(5000)
        assert counts["two"] == 1500
```

- [ ] **Step 2: Run tests — verify they all fail**

```bash
cd /Users/mpineda/Desktop/llm-class-2026/llm-class-2026-winter-cohort/backend
uv run pytest tests/test_prompt_templates.py -v
```

Expected: `ImportError` — `prompt_templates` does not exist yet.

- [ ] **Step 3: Implement `prompt_templates.py`**

Create `backend/training/prompt_templates.py`:

```python
"""
TacoLLM — Dataset Prompt Templates

Generates natural language instruction prompts for dataset generation.
Templates span all constraint categories defined in the ADR:
  single (20%), two (30%), three+ (25%), style (10%), edge (10%), followup (5%)
"""

import random
from typing import Dict

_CALORIE_THRESHOLDS = [300, 350, 400, 450, 500]
_PROTEINS = ["chicken", "beef", "pork", "fish", "shrimp", "carnitas", "carne asada", "turkey"]
_STYLES = ["street", "tex-mex", "breakfast", "healthy", "authentic Mexican"]
_SPICE_WORDS = ["mild", "medium", "spicy", "extra spicy"]

CATEGORY_DISTRIBUTION: Dict[str, float] = {
    "single": 0.20,
    "two": 0.30,
    "three": 0.25,
    "style": 0.10,
    "edge": 0.10,
    "followup": 0.05,
}


def single_constraint_prompt() -> str:
    """Return a prompt with exactly one dietary/nutritional constraint."""
    cal = random.choice(_CALORIE_THRESHOLDS)
    spice = random.choice(_SPICE_WORDS)
    protein = random.choice(_PROTEINS)
    style = random.choice(_STYLES)
    options = [
        "Give me a high protein taco.",
        "Make me a vegan taco.",
        f"I want a taco under {cal} calories.",
        "Give me a low-carb taco.",
        "Make me a dairy-free taco.",
        f"I want a {spice} taco.",
        "Give me a keto-friendly taco.",
        "Make me a vegetarian taco.",
        f"I want a {style} style taco.",
        "Give me a gluten-free taco.",
        f"Make me a {protein} taco.",
        "I want a protein-packed taco.",
    ]
    return random.choice(options)


def two_constraint_prompt() -> str:
    """Return a prompt with two intersecting constraints."""
    cal = random.choice(_CALORIE_THRESHOLDS)
    spice = random.choice(_SPICE_WORDS)
    protein = random.choice(_PROTEINS)
    style = random.choice(_STYLES)
    options = [
        f"Give me a high protein taco under {cal} calories.",
        f"Make me a vegan taco that is {spice}.",
        f"I want a dairy-free taco under {cal} calories.",
        f"Give me a low-carb taco with {protein}.",
        "Make me a keto taco with no dairy.",
        f"I want a vegetarian taco under {cal} calories.",
        f"Give me a high protein {style} taco.",
        f"Make me a {spice} taco with no beef.",
        f"I want a gluten-free taco with {protein}.",
        "Give me a dairy-free high protein taco.",
        f"Make me a low-carb {spice} taco.",
        f"I want a vegan taco under {cal} calories.",
    ]
    return random.choice(options)


def three_constraint_prompt() -> str:
    """Return a prompt with three or more intersecting constraints."""
    cal = random.choice(_CALORIE_THRESHOLDS)
    spice = random.choice(_SPICE_WORDS)
    protein = random.choice(_PROTEINS)
    style = random.choice(_STYLES)
    options = [
        f"Give me a high protein taco under {cal} calories with no dairy.",
        f"Make me a vegan low-carb taco that is {spice}.",
        f"I want a keto taco under {cal} calories that is {spice}.",
        f"Give me a dairy-free high protein taco under {cal} calories.",
        f"Make me a gluten-free vegetarian taco that is {spice} and under {cal} calories.",
        f"I want a high protein {style} taco with no dairy.",
        f"Give me a keto {spice} taco with {protein}.",
        f"Make me a vegan {style} taco under {cal} calories.",
        f"I want a low-carb dairy-free taco with {protein} under {cal} calories.",
        f"Give me a high protein {spice} {style} taco.",
    ]
    return random.choice(options)


def style_prompt() -> str:
    """Return a style or cuisine-focused prompt."""
    cal = random.choice(_CALORIE_THRESHOLDS)
    spice = random.choice(_SPICE_WORDS)
    protein = random.choice(_PROTEINS)
    options = [
        "Give me an authentic street taco.",
        "Make me a classic tex-mex taco.",
        "I want a breakfast taco.",
        "Give me a healthy taco.",
        f"Make me a street taco with {protein}.",
        f"I want a tex-mex style taco under {cal} calories.",
        "Give me an authentic Mexican taco.",
        f"Make me a healthy taco that is {spice}.",
        f"I want a breakfast taco with {protein}.",
        f"Give me a street-style taco under {cal} calories.",
    ]
    return random.choice(options)


def edge_case_prompt() -> str:
    """Return a challenging or contradictory constraint prompt."""
    options = [
        "Give me a high protein keto vegan taco under 300 calories.",
        "Make me a super low calorie taco under 200 calories that is still filling.",
        "I want an ultra-high protein taco with over 50g of protein.",
        "Give me a taco with no beef, no chicken, no pork, and no fish.",
        "Make me a keto vegan taco.",
        "I want the spiciest taco possible under 350 calories.",
        "Give me a taco with minimal carbs, high protein, and no dairy.",
        "Make me a filling taco under 250 calories.",
        "I want a taco with at least 5 different vegetables.",
        "Give me a no-compromise taco: high protein, low carb, and dairy-free.",
    ]
    return random.choice(options)


def followup_prompt() -> str:
    """Return a follow-up prompt that references prior preferences."""
    spice = random.choice(_SPICE_WORDS)
    protein = random.choice(_PROTEINS)
    options = [
        f"Same as before but make it {spice}.",
        f"Keep the same dietary restrictions but use {protein} instead.",
        "Make it a bit spicier.",
        "Can you make a lighter version of that?",
        "Same thing but vegan.",
        "Make it under 350 calories this time.",
        "Keep the protein high but make it dairy-free.",
        "Same constraints but make it a street taco style.",
        "Make it milder for me.",
        "Keep the same but add more protein.",
    ]
    return random.choice(options)


def get_prompt_for_category(category: str) -> str:
    """Return a random instruction prompt for the given category name."""
    generators = {
        "single": single_constraint_prompt,
        "two": two_constraint_prompt,
        "three": three_constraint_prompt,
        "style": style_prompt,
        "edge": edge_case_prompt,
        "followup": followup_prompt,
    }
    if category not in generators:
        raise ValueError(
            f"Unknown category: {category!r}. Must be one of: {list(generators)}"
        )
    return generators[category]()


def get_category_counts(total: int) -> Dict[str, int]:
    """
    Return per-category example counts that sum exactly to `total`.

    Distributes according to CATEGORY_DISTRIBUTION, with the last
    category absorbing any rounding remainder.
    """
    categories = list(CATEGORY_DISTRIBUTION.keys())
    counts: Dict[str, int] = {}
    remaining = total
    for cat in categories[:-1]:
        counts[cat] = round(total * CATEGORY_DISTRIBUTION[cat])
        remaining -= counts[cat]
    counts[categories[-1]] = remaining
    return counts
```

- [ ] **Step 4: Run tests — verify they all pass**

```bash
cd /Users/mpineda/Desktop/llm-class-2026/llm-class-2026-winter-cohort/backend
uv run pytest tests/test_prompt_templates.py -v
```

Expected: all PASS. (Note: `test_varies_on_repeated_calls` is probabilistic but extremely unlikely to fail with 50 calls across 12 options.)

- [ ] **Step 5: Lint and format**

```bash
cd /Users/mpineda/Desktop/llm-class-2026/llm-class-2026-winter-cohort/backend
uv run ruff check training/prompt_templates.py tests/test_prompt_templates.py
uv run ruff format training/prompt_templates.py tests/test_prompt_templates.py
```

Expected: exits 0.

---

## Task 3: Dataset Validator — TDD

**Files:**
- Create: `backend/tests/test_dataset_validator.py`
- Create: `backend/training/dataset_validator.py`

- [ ] **Step 1: Write failing tests**

Create `backend/tests/test_dataset_validator.py`:

```python
import pytest

from training.dataset_validator import validate_example


@pytest.fixture
def valid_example():
    return {
        "instruction": "Give me a high protein taco under 400 calories.",
        "output": {
            "name": "Chipotle Chicken Taco",
            "ingredients": [
                "corn tortillas",
                "grilled chicken breast",
                "salsa roja",
                "cilantro",
                "onion",
            ],
            "calories": 345,
            "protein": 33,
            "carbs": 24,
            "fat": 9,
            "dietary_tags": ["high_protein"],
            "spice_level": "medium",
            "reasoning": "Lean chicken keeps protein high.",
        },
    }


class TestValidateExample:
    def test_valid_example_passes(self, valid_example):
        assert validate_example(valid_example) is True

    def test_missing_instruction_fails(self, valid_example):
        del valid_example["instruction"]
        assert validate_example(valid_example) is False

    def test_empty_instruction_fails(self, valid_example):
        valid_example["instruction"] = ""
        assert validate_example(valid_example) is False

    def test_whitespace_only_instruction_fails(self, valid_example):
        valid_example["instruction"] = "   "
        assert validate_example(valid_example) is False

    def test_non_string_instruction_fails(self, valid_example):
        valid_example["instruction"] = 123
        assert validate_example(valid_example) is False

    def test_missing_output_fails(self, valid_example):
        del valid_example["output"]
        assert validate_example(valid_example) is False

    def test_non_dict_output_fails(self, valid_example):
        valid_example["output"] = "not a dict"
        assert validate_example(valid_example) is False

    def test_output_missing_required_field_fails(self, valid_example):
        del valid_example["output"]["name"]
        assert validate_example(valid_example) is False

    def test_output_invalid_spice_level_fails(self, valid_example):
        valid_example["output"]["spice_level"] = "nuclear"
        assert validate_example(valid_example) is False

    def test_output_string_calories_fails(self, valid_example):
        valid_example["output"]["calories"] = "345"
        assert validate_example(valid_example) is False

    def test_schema_only_no_constraint_checking(self, valid_example):
        # Dataset validator checks schema only, not constraint adherence.
        # A dairy ingredient with no_dairy context still passes schema.
        valid_example["output"]["ingredients"] = ["corn tortillas", "cheese", "salsa"]
        assert validate_example(valid_example) is True
```

- [ ] **Step 2: Run tests — verify they all fail**

```bash
cd /Users/mpineda/Desktop/llm-class-2026/llm-class-2026-winter-cohort/backend
uv run pytest tests/test_dataset_validator.py -v
```

Expected: `ImportError` — `dataset_validator` does not exist yet.

- [ ] **Step 3: Implement `dataset_validator.py`**

Create `backend/training/dataset_validator.py`:

```python
"""
TacoLLM — Dataset Example Validator

Validates a generated training example before inclusion in the dataset.
Wraps app.validator.TacoValidator with dataset-specific structure checks.

Note: only schema validity is checked here, not constraint adherence.
The training set intentionally does not penalize the model for responding
to a "high protein" prompt with a lower-protein taco — the fine-tuning
signal comes from the example pairing, not from constraint post-hoc filtering.
"""

from typing import Any, Dict

from app.validator import TacoValidator

_validator = TacoValidator()


def validate_example(example: Dict[str, Any]) -> bool:
    """
    Returns True if the example is structurally valid for training.

    Checks:
    - "instruction" is a non-empty string
    - "output" is a dict that passes all TacoValidator schema checks
    """
    instruction = example.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        return False

    output = example.get("output")
    if not isinstance(output, dict):
        return False

    # Empty constraints dict — schema check only, no constraint adherence
    return len(_validator.validate(output, {})) == 0
```

- [ ] **Step 4: Run tests — verify they all pass**

```bash
cd /Users/mpineda/Desktop/llm-class-2026/llm-class-2026-winter-cohort/backend
uv run pytest tests/test_dataset_validator.py -v
```

Expected: all 11 PASS.

- [ ] **Step 5: Lint and format**

```bash
cd /Users/mpineda/Desktop/llm-class-2026/llm-class-2026-winter-cohort/backend
uv run ruff check training/dataset_validator.py tests/test_dataset_validator.py
uv run ruff format training/dataset_validator.py tests/test_dataset_validator.py
```

Expected: exits 0.

---

## Task 4: Generation Script + Pure Function Tests — TDD

**Files:**
- Create: `backend/tests/test_generate_dataset.py`
- Create: `backend/training/generate_dataset.py`

- [ ] **Step 1: Write failing tests for pure functions**

Create `backend/tests/test_generate_dataset.py`:

```python
import json

import pytest

from training.generate_dataset import parse_taco_output, save_dataset


class TestParseTacoOutput:
    def test_parses_clean_json(self):
        text = '{"name": "Chicken Taco", "calories": 350}'
        result = parse_taco_output(text)
        assert result == {"name": "Chicken Taco", "calories": 350}

    def test_strips_markdown_json_fence(self):
        text = '```json\n{"name": "test"}\n```'
        result = parse_taco_output(text)
        assert result == {"name": "test"}

    def test_strips_plain_markdown_fence(self):
        text = '```\n{"name": "test"}\n```'
        result = parse_taco_output(text)
        assert result == {"name": "test"}

    def test_extracts_json_from_surrounding_text(self):
        text = 'Here is your taco: {"name": "test", "calories": 300} Enjoy!'
        result = parse_taco_output(text)
        assert result is not None
        assert result["name"] == "test"

    def test_returns_none_for_invalid_json(self):
        assert parse_taco_output("not json at all") is None

    def test_returns_none_for_empty_string(self):
        assert parse_taco_output("") is None


class TestSaveDataset:
    def test_creates_train_jsonl_and_eval_json(self, tmp_path):
        examples = [{"instruction": f"p{i}", "output": {"n": i}} for i in range(10)]
        save_dataset(examples, tmp_path, train_split=0.8)
        assert (tmp_path / "train.jsonl").exists()
        assert (tmp_path / "eval.json").exists()

    def test_returns_correct_counts(self, tmp_path):
        examples = [{"instruction": f"p{i}", "output": {}} for i in range(100)]
        train_count, eval_count = save_dataset(examples, tmp_path, train_split=0.9)
        assert train_count == 90
        assert eval_count == 10

    def test_train_jsonl_one_line_per_example(self, tmp_path):
        examples = [{"instruction": f"p{i}", "output": {}} for i in range(5)]
        save_dataset(examples, tmp_path, train_split=1.0)
        lines = (tmp_path / "train.jsonl").read_text().strip().splitlines()
        assert len(lines) == 5
        for line in lines:
            obj = json.loads(line)
            assert "instruction" in obj

    def test_eval_json_is_list(self, tmp_path):
        examples = [{"instruction": f"p{i}", "output": {}} for i in range(10)]
        save_dataset(examples, tmp_path, train_split=0.8)
        data = json.loads((tmp_path / "eval.json").read_text())
        assert isinstance(data, list)
        assert len(data) == 2

    def test_creates_output_dir_if_missing(self, tmp_path):
        nested = tmp_path / "deep" / "nested"
        examples = [{"instruction": "test", "output": {}}]
        save_dataset(examples, nested, train_split=1.0)
        assert (nested / "train.jsonl").exists()
```

- [ ] **Step 2: Run tests — verify they all fail**

```bash
cd /Users/mpineda/Desktop/llm-class-2026/llm-class-2026-winter-cohort/backend
uv run pytest tests/test_generate_dataset.py -v
```

Expected: `ImportError` — `generate_dataset` does not exist yet.

- [ ] **Step 3: Implement `generate_dataset.py`**

Create `backend/training/generate_dataset.py`:

```python
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

    Uses claude-haiku-4-5 for speed and cost efficiency at dataset scale.
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


def generate_examples(
    client: anthropic.Anthropic, total: int
) -> list[dict[str, Any]]:
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

    train_count, eval_count = save_dataset(
        examples, Path(args.output), args.train_split
    )
    logger.info(
        f"Saved {train_count} train + {eval_count} eval examples to {args.output}"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests — verify they all pass**

```bash
cd /Users/mpineda/Desktop/llm-class-2026/llm-class-2026-winter-cohort/backend
uv run pytest tests/test_generate_dataset.py -v
```

Expected: all 11 PASS.

- [ ] **Step 5: Lint and format**

```bash
cd /Users/mpineda/Desktop/llm-class-2026/llm-class-2026-winter-cohort/backend
uv run ruff check training/generate_dataset.py tests/test_generate_dataset.py
uv run ruff format training/generate_dataset.py tests/test_generate_dataset.py
```

Expected: exits 0.

---

## Task 5: Full Suite Gate + Smoke Test Instructions

- [ ] **Step 1: Run full test suite**

```bash
cd /Users/mpineda/Desktop/llm-class-2026/llm-class-2026-winter-cohort/backend
uv run pytest --cov=app --cov=training --cov-report=term-missing -v
```

Expected results:
- All previously passing tests still pass (64 from Plan 1)
- `tests/test_prompt_templates.py` — all pass
- `tests/test_dataset_validator.py` — 11 pass
- `tests/test_generate_dataset.py` — 11 pass
- `training/prompt_templates.py` — ~100% coverage
- `training/dataset_validator.py` — 100% coverage
- `training/generate_dataset.py` — `parse_taco_output` and `save_dataset` covered; `call_claude`, `generate_examples`, `main` at 0% (expected — require live API)

- [ ] **Step 2: Run ruff on all source**

```bash
cd /Users/mpineda/Desktop/llm-class-2026/llm-class-2026-winter-cohort/backend
uv run ruff check app/ training/ tests/
uv run ruff format --check app/ training/ tests/
```

Expected: exits 0 on both.

- [ ] **Step 3: Smoke test instructions (manual — requires ANTHROPIC_API_KEY)**

To verify the generation script works end-to-end before running the full 5,000 row job:

```bash
cd /Users/mpineda/Desktop/llm-class-2026/llm-class-2026-winter-cohort/backend
ANTHROPIC_API_KEY=<your-key> uv run python -m training.generate_dataset --count 10 --output ../data
```

Expected output:
```
HH:MM:SS INFO Generating 10 examples...
HH:MM:SS INFO Generating 2 'single' examples...
HH:MM:SS INFO   single: completed 2/2 (attempts: ...)
HH:MM:SS INFO Generating 3 'two' examples...
...
HH:MM:SS INFO Generated 10 valid examples
HH:MM:SS INFO Saved 9 train + 1 eval examples to ../data
```

Verify output files:
```bash
wc -l /Users/mpineda/Desktop/llm-class-2026/llm-class-2026-winter-cohort/data/train.jsonl
# Expected: 9 lines

python3 -c "import json; data=json.load(open('../data/eval.json')); print(len(data), 'eval examples')"
# Expected: 1 eval examples
```

Once smoke test passes, run the full dataset generation:
```bash
cd /Users/mpineda/Desktop/llm-class-2026/llm-class-2026-winter-cohort/backend
ANTHROPIC_API_KEY=<your-key> uv run python -m training.generate_dataset --count 5000 --output ../data
```

This will take approximately 30–60 minutes depending on API rate limits. The script logs progress every 100 examples per category.
