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
