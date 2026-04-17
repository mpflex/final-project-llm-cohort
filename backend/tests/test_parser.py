import pytest

from app.parser import ConstraintParser


@pytest.fixture
def parser():
    return ConstraintParser()


class TestCalorieExtraction:
    def test_extracts_under_calories(self, parser):
        result = parser.extract("Give me a taco under 400 calories")
        assert result["max_calories"] == 400

    def test_extracts_under_kcal(self, parser):
        result = parser.extract("I want something under 350 kcal")
        assert result["max_calories"] == 350

    def test_no_calorie_returns_no_key(self, parser):
        result = parser.extract("Give me a spicy taco")
        assert "max_calories" not in result


class TestMacroExtraction:
    def test_extracts_high_protein(self, parser):
        result = parser.extract("high protein taco")
        assert result["high_protein"] is True

    def test_extracts_protein_rich(self, parser):
        result = parser.extract("protein-rich option please")
        assert result["high_protein"] is True

    def test_extracts_low_carb(self, parser):
        result = parser.extract("low carb taco please")
        assert result["low_carb"] is True

    def test_keto_sets_low_carb_and_keto(self, parser):
        result = parser.extract("keto taco")
        assert result["keto"] is True
        assert result["low_carb"] is True


class TestDietaryExtraction:
    def test_vegan_sets_vegan_and_vegetarian(self, parser):
        result = parser.extract("I want a vegan taco")
        assert result["vegan"] is True
        assert result["vegetarian"] is True

    def test_vegetarian_does_not_set_vegan(self, parser):
        result = parser.extract("vegetarian taco please")
        assert result["vegetarian"] is True
        assert "vegan" not in result

    def test_no_dairy(self, parser):
        result = parser.extract("no dairy taco")
        assert result["no_dairy"] is True

    def test_dairy_free(self, parser):
        result = parser.extract("dairy-free option")
        assert result["no_dairy"] is True

    def test_without_dairy(self, parser):
        result = parser.extract("without dairy")
        assert result["no_dairy"] is True

    def test_no_beef(self, parser):
        result = parser.extract("no beef please")
        assert result["no_beef"] is True

    def test_no_gluten(self, parser):
        result = parser.extract("gluten-free taco")
        assert result["no_gluten"] is True


class TestSpiceLevelExtraction:
    def test_spicy_maps_to_hot(self, parser):
        result = parser.extract("make it spicy")
        assert result["spice_level"] == "hot"

    def test_hot_maps_to_hot(self, parser):
        result = parser.extract("I want it hot")
        assert result["spice_level"] == "hot"

    def test_mild(self, parser):
        result = parser.extract("mild taco please")
        assert result["spice_level"] == "mild"

    def test_medium(self, parser):
        result = parser.extract("medium spice level")
        assert result["spice_level"] == "medium"

    def test_no_spice_returns_no_key(self, parser):
        result = parser.extract("high protein taco")
        assert "spice_level" not in result


class TestStyleExtraction:
    def test_street_taco(self, parser):
        result = parser.extract("I want a street taco")
        assert result["preferred_style"] == "street"

    def test_tex_mex(self, parser):
        result = parser.extract("tex-mex style taco")
        assert result["preferred_style"] == "tex-mex"

    def test_breakfast_taco(self, parser):
        result = parser.extract("breakfast taco please")
        assert result["preferred_style"] == "breakfast"

    def test_healthy_style(self, parser):
        result = parser.extract("healthy taco option")
        assert result["preferred_style"] == "healthy"


class TestMultiConstraint:
    def test_combines_multiple_constraints(self, parser):
        result = parser.extract("high protein taco under 400 calories, no dairy, spicy")
        assert result["max_calories"] == 400
        assert result["high_protein"] is True
        assert result["no_dairy"] is True
        assert result["spice_level"] == "hot"

    def test_empty_message_returns_empty_dict(self, parser):
        result = parser.extract("")
        assert result == {}

    def test_unrelated_message_returns_empty_dict(self, parser):
        result = parser.extract("hello there")
        assert result == {}
