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
