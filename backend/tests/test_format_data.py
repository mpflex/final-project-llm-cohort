import json

import pytest

from training.format_data import TRAINING_SYSTEM_PROMPT, format_training_example


@pytest.fixture
def example():
    return {
        "instruction": "Give me a high protein taco under 400 calories.",
        "output": {
            "name": "Chipotle Chicken Taco",
            "ingredients": ["corn tortilla", "grilled chicken", "salsa"],
            "calories": 350,
            "protein": 32,
            "carbs": 20,
            "fat": 8,
            "dietary_tags": ["high_protein"],
            "spice_level": "medium",
            "reasoning": "Lean chicken keeps protein high.",
        },
    }


class TestFormatTrainingExample:
    def test_returns_string(self, example):
        assert isinstance(format_training_example(example), str)

    def test_starts_with_begin_token(self, example):
        assert format_training_example(example).startswith("<|begin_of_text|>")

    def test_contains_system_header(self, example):
        assert "<|start_header_id|>system<|end_header_id|>" in format_training_example(example)

    def test_contains_system_prompt(self, example):
        assert TRAINING_SYSTEM_PROMPT in format_training_example(example)

    def test_contains_user_header(self, example):
        assert "<|start_header_id|>user<|end_header_id|>" in format_training_example(example)

    def test_contains_instruction(self, example):
        assert example["instruction"] in format_training_example(example)

    def test_contains_assistant_header(self, example):
        assert "<|start_header_id|>assistant<|end_header_id|>" in format_training_example(example)

    def test_contains_compact_output_json(self, example):
        # Output JSON must be compact (no indent) to keep sequence lengths short
        assert json.dumps(example["output"]) in format_training_example(example)

    def test_ends_with_eot(self, example):
        assert format_training_example(example).endswith("<|eot_id|>")

    def test_exactly_three_eot_tokens(self, example):
        # One after system, one after user, one after assistant
        assert format_training_example(example).count("<|eot_id|>") == 3

    def test_different_instructions_produce_different_strings(self):
        ex1 = {"instruction": "Prompt A", "output": {"x": 1}}
        ex2 = {"instruction": "Prompt B", "output": {"x": 1}}
        assert format_training_example(ex1) != format_training_example(ex2)

    def test_different_outputs_produce_different_strings(self):
        ex1 = {"instruction": "Same", "output": {"x": 1}}
        ex2 = {"instruction": "Same", "output": {"x": 2}}
        assert format_training_example(ex1) != format_training_example(ex2)
