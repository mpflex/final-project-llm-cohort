import pytest

from app.validator import TacoValidator


@pytest.fixture
def validator():
    return TacoValidator()


@pytest.fixture
def valid_taco():
    return {
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
        "dietary_tags": ["high_protein", "dairy_free"],
        "spice_level": "medium",
        "reasoning": "Lean chicken keeps protein high.",
    }


class TestRequiredFields:
    def test_valid_taco_returns_no_issues(self, validator, valid_taco):
        assert validator.validate(valid_taco, {}) == []

    def test_missing_name_flagged(self, validator, valid_taco):
        del valid_taco["name"]
        issues = validator.validate(valid_taco, {})
        assert any("name" in i for i in issues)

    def test_missing_calories_flagged(self, validator, valid_taco):
        del valid_taco["calories"]
        issues = validator.validate(valid_taco, {})
        assert any("calories" in i for i in issues)

    def test_empty_taco_flags_all_nine_fields(self, validator):
        issues = validator.validate({}, {})
        assert len(issues) == 9


class TestSpiceLevel:
    def test_invalid_spice_level_flagged(self, validator, valid_taco):
        valid_taco["spice_level"] = "nuclear"
        issues = validator.validate(valid_taco, {})
        assert any("spice_level" in i for i in issues)

    def test_mild_is_valid(self, validator, valid_taco):
        valid_taco["spice_level"] = "mild"
        assert validator.validate(valid_taco, {}) == []

    def test_hot_is_valid(self, validator, valid_taco):
        valid_taco["spice_level"] = "hot"
        assert validator.validate(valid_taco, {}) == []


class TestNumericFields:
    def test_string_calories_flagged(self, validator, valid_taco):
        valid_taco["calories"] = "345"
        issues = validator.validate(valid_taco, {})
        assert any("calories" in i for i in issues)

    def test_string_protein_flagged(self, validator, valid_taco):
        valid_taco["protein"] = "high"
        issues = validator.validate(valid_taco, {})
        assert any("protein" in i for i in issues)

    def test_float_calories_valid(self, validator, valid_taco):
        valid_taco["calories"] = 345.5
        assert validator.validate(valid_taco, {}) == []


class TestArrayFields:
    def test_string_ingredients_flagged(self, validator, valid_taco):
        valid_taco["ingredients"] = "chicken"
        issues = validator.validate(valid_taco, {})
        assert any("ingredients" in i for i in issues)

    def test_string_dietary_tags_flagged(self, validator, valid_taco):
        valid_taco["dietary_tags"] = "high_protein"
        issues = validator.validate(valid_taco, {})
        assert any("dietary_tags" in i for i in issues)


class TestCalorieConstraint:
    def test_calories_exceeding_max_flagged(self, validator, valid_taco):
        valid_taco["calories"] = 500
        issues = validator.validate(valid_taco, {"max_calories": 400})
        assert any("Calories" in i for i in issues)

    def test_calories_at_max_pass(self, validator, valid_taco):
        valid_taco["calories"] = 400
        assert validator.validate(valid_taco, {"max_calories": 400}) == []

    def test_no_calorie_constraint_passes(self, validator, valid_taco):
        valid_taco["calories"] = 800
        assert validator.validate(valid_taco, {}) == []


class TestDairyConstraint:
    def test_sour_cream_flagged_when_no_dairy(self, validator, valid_taco):
        valid_taco["ingredients"] = ["corn tortillas", "chicken", "sour cream", "salsa"]
        issues = validator.validate(valid_taco, {"no_dairy": True})
        assert any("dairy" in i.lower() for i in issues)

    def test_cheese_flagged_when_no_dairy(self, validator, valid_taco):
        valid_taco["ingredients"] = ["flour tortilla", "chicken", "cheddar cheese"]
        issues = validator.validate(valid_taco, {"no_dairy": True})
        assert any("dairy" in i.lower() for i in issues)

    def test_dairy_free_ingredients_pass(self, validator, valid_taco):
        assert validator.validate(valid_taco, {"no_dairy": True}) == []

    def test_jackfruit_not_flagged_as_dairy(self, validator, valid_taco):
        valid_taco["ingredients"] = ["corn tortillas", "jackfruit", "salsa verde", "cilantro"]
        assert validator.validate(valid_taco, {"no_dairy": True}) == []


class TestVeganConstraint:
    def test_chicken_flagged_when_vegan(self, validator, valid_taco):
        issues = validator.validate(valid_taco, {"vegan": True})
        assert any("vegan" in i.lower() for i in issues)

    def test_vegan_taco_passes(self, validator, valid_taco):
        valid_taco["ingredients"] = [
            "corn tortillas",
            "black beans",
            "salsa verde",
            "avocado",
            "cilantro",
        ]
        assert validator.validate(valid_taco, {"vegan": True}) == []

    def test_jackfruit_not_flagged_as_meat_vegan(self, validator, valid_taco):
        valid_taco["ingredients"] = ["corn tortillas", "jackfruit", "salsa verde", "avocado"]
        assert validator.validate(valid_taco, {"vegan": True}) == []


class TestNoBeefConstraint:
    def test_beef_flagged_when_no_beef(self, validator, valid_taco):
        valid_taco["ingredients"] = ["flour tortilla", "ground beef", "salsa", "onion"]
        issues = validator.validate(valid_taco, {"no_beef": True})
        assert any("beef" in i.lower() for i in issues)

    def test_chicken_passes_no_beef(self, validator, valid_taco):
        assert validator.validate(valid_taco, {"no_beef": True}) == []
