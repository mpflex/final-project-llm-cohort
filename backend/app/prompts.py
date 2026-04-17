"""
TacoLLM — Prompt Builder

Constructs system and user prompts dynamically from user messages
and parsed constraints.
"""

from typing import Any, Dict

# JSON schema example injected into every prompt
SCHEMA_EXAMPLE = """{
  "name": "Chipotle Lime Chicken Taco",
  "ingredients": ["corn tortillas", "grilled chicken breast", "cabbage slaw", "pickled onions", "chipotle salsa"],
  "calories": 365,
  "protein": 34,
  "carbs": 28,
  "fat": 11,
  "dietary_tags": ["high_protein", "dairy_free"],
  "spice_level": "medium",
  "reasoning": "This taco uses lean chicken breast for high protein and avoids dairy while staying under the calorie target."
}"""


def build_system_prompt() -> str:
    return (
        """You are TacoLLM, an expert taco recommendation assistant.

Your task is to generate realistic taco recommendations that satisfy user dietary and nutrition constraints.

RULES:
- You must return ONLY valid JSON. No markdown. No commentary. No text before or after the JSON.
- Respect ALL calorie and dietary constraints provided.
- Do not include forbidden ingredients.
- Keep ingredient lists realistic (4–7 items).
- Ensure calories, protein, carbs, and fat are numerically plausible.
- spice_level must be exactly one of: mild, medium, hot
- dietary_tags must be an array of strings.
- reasoning must briefly explain how the taco satisfies the constraints.

REQUIRED SCHEMA:
- name (string)
- ingredients (array of strings)
- calories (number)
- protein (number)
- carbs (number)
- fat (number)
- dietary_tags (array of strings)
- spice_level (string: mild | medium | hot)
- reasoning (string)

EXAMPLE OUTPUT:
"""
        + SCHEMA_EXAMPLE
    )


def build_user_prompt(
    message: str,
    constraints: Dict[str, Any],
    retry: bool = False,
    attempt: int = 1,
) -> str:
    """
    Build the user turn of the prompt.
    On retry, add a stronger corrective instruction.
    """
    constraint_lines = _format_constraints(constraints)

    base = f"""User request: {message}

Extracted constraints:
{constraint_lines}

Respond with ONLY the JSON object. Begin your response with {{ and end with }}."""

    if retry:
        base = (
            "[IMPORTANT: Your previous response was not valid JSON. "
            "You MUST respond with ONLY a raw JSON object. "
            "No explanation. No markdown. Start immediately with {]\n\n" + base
        )

    return base


def _format_constraints(constraints: Dict[str, Any]) -> str:
    if not constraints:
        return "- None specified"

    lines = []
    label_map = {
        "max_calories": "Maximum calories",
        "min_protein": "Minimum protein (g)",
        "low_carb": "Low carb",
        "high_protein": "High protein",
        "vegan": "Vegan",
        "vegetarian": "Vegetarian",
        "no_dairy": "No dairy",
        "no_beef": "No beef",
        "no_gluten": "No gluten",
        "spice_level": "Spice level",
        "preferred_style": "Preferred style",
        "keto": "Keto",
    }

    for key, value in constraints.items():
        label = label_map.get(key, key.replace("_", " ").title())
        if isinstance(value, bool) and value:
            lines.append(f"- {label}: Yes")
        elif isinstance(value, bool) and not value:
            continue  # Don't show false flags
        elif value is not None:
            lines.append(f"- {label}: {value}")

    return "\n".join(lines) if lines else "- None specified"
