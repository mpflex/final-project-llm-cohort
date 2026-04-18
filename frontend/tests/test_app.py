# frontend/tests/test_app.py
from app import format_debug_info, render_taco_card

_VALID_TACO = {
    "name": "Chipotle Chicken Taco",
    "ingredients": ["corn tortilla", "grilled chicken", "salsa roja"],
    "calories": 350,
    "protein": 30,
    "carbs": 25,
    "fat": 10,
    "dietary_tags": ["high_protein", "dairy_free"],
    "spice_level": "medium",
    "reasoning": "Lean chicken keeps protein high.",
}


def test_render_taco_card_contains_name():
    html = render_taco_card(_VALID_TACO)
    assert "Chipotle Chicken Taco" in html


def test_render_taco_card_contains_macros():
    html = render_taco_card(_VALID_TACO)
    assert "350" in html
    assert "30" in html


def test_render_taco_card_contains_tags():
    html = render_taco_card(_VALID_TACO)
    assert "high_protein" in html


def test_render_taco_card_none_returns_error_html():
    html = render_taco_card(None)
    lower = html.lower()
    assert "error" in lower or "failed" in lower or "offline" in lower


def test_render_taco_card_handles_missing_fields():
    html = render_taco_card({"name": "Minimal Taco"})
    assert "Minimal Taco" in html


def test_format_debug_info_contains_session_and_constraints():
    metadata = {
        "session_id": "abc-123",
        "parsed_constraints": {"max_calories": 400},
        "validation_issues": [],
        "inference_attempts": 1,
        "model": "lora",
        "valid_json": True,
    }
    result = format_debug_info(metadata)
    assert "abc-123" in result
    assert "max_calories" in result


def test_format_debug_info_shows_validation_issues():
    metadata = {
        "session_id": "x",
        "parsed_constraints": {},
        "validation_issues": ["Calories 520 exceeds max 400"],
        "inference_attempts": 2,
        "model": "base",
        "valid_json": True,
    }
    result = format_debug_info(metadata)
    assert "Calories 520 exceeds max 400" in result
