"""
TacoLLM — Taco Output Validator

Validates a generated taco JSON against the required schema
and checks it against the user's parsed constraints.
"""

import re
from typing import Any, Dict, List

REQUIRED_FIELDS = [
    "name",
    "ingredients",
    "calories",
    "protein",
    "carbs",
    "fat",
    "dietary_tags",
    "spice_level",
    "reasoning",
]
VALID_SPICE_LEVELS = {"mild", "medium", "hot"}

_DAIRY_KEYWORDS = [
    "cheese",
    "sour cream",
    "crema",
    "queso",
    "milk",
    "butter",
    "cream",
    "yogurt",
    "cotija",
    "oaxaca",
    "cheddar",
    "jack",
    "fresco",
]
_MEAT_KEYWORDS = [
    "chicken",
    "beef",
    "pork",
    "carnitas",
    "steak",
    "carne",
    "chorizo",
    "al pastor",
    "barbacoa",
    "birria",
    "fish",
    "shrimp",
    "seafood",
    "turkey",
    "lamb",
]
_BEEF_KEYWORDS = ["beef", "carne asada", "ground beef", "barbacoa", "carne molida"]


class TacoValidator:
    """
    Validates a taco recommendation dict against the schema contract
    and the user's parsed constraints. Returns a list of issue strings;
    an empty list means the taco is valid.
    """

    def validate(self, taco: Dict[str, Any], constraints: Dict[str, Any]) -> List[str]:
        issues: List[str] = []

        for field in REQUIRED_FIELDS:
            if field not in taco:
                issues.append(f"Missing required field: {field}")

        if issues:
            return issues

        self._check_spice_level(taco, issues)
        self._check_numeric_fields(taco, issues)
        self._check_array_fields(taco, issues)

        if issues:
            return issues

        ingredients = [i.lower() for i in taco.get("ingredients", [])]
        self._check_calorie_constraint(taco, constraints, issues)
        self._check_no_dairy(ingredients, constraints, issues)
        self._check_no_beef(ingredients, constraints, issues)
        self._check_vegan(ingredients, constraints, issues)

        return issues

    def _check_spice_level(self, taco: Dict[str, Any], issues: List[str]) -> None:
        if taco.get("spice_level") not in VALID_SPICE_LEVELS:
            issues.append(
                f"Invalid spice_level '{taco.get('spice_level')}'. Must be: mild, medium, hot"
            )

    def _check_numeric_fields(self, taco: Dict[str, Any], issues: List[str]) -> None:
        for field in ("calories", "protein", "carbs", "fat"):
            if not isinstance(taco.get(field), (int, float)):
                issues.append(
                    f"Field '{field}' must be a number, got {type(taco.get(field)).__name__}"
                )

    def _check_array_fields(self, taco: Dict[str, Any], issues: List[str]) -> None:
        for field in ("ingredients", "dietary_tags"):
            if not isinstance(taco.get(field), list):
                issues.append(f"Field '{field}' must be an array")

    def _check_calorie_constraint(
        self, taco: Dict[str, Any], constraints: Dict[str, Any], issues: List[str]
    ) -> None:
        max_cal = constraints.get("max_calories")
        if max_cal is not None and isinstance(taco.get("calories"), (int, float)):
            if taco["calories"] > max_cal:
                issues.append(f"Calories {taco['calories']} exceeds max {max_cal}")

    def _check_no_dairy(
        self, ingredients: List[str], constraints: Dict[str, Any], issues: List[str]
    ) -> None:
        if not constraints.get("no_dairy"):
            return
        for ingredient in ingredients:
            for kw in _DAIRY_KEYWORDS:
                if re.search(rf"\b{re.escape(kw)}\b", ingredient):
                    issues.append(
                        f"Contains dairy ingredient '{kw}' but no_dairy constraint is set"
                    )
                    return

    def _check_no_beef(
        self, ingredients: List[str], constraints: Dict[str, Any], issues: List[str]
    ) -> None:
        if not constraints.get("no_beef"):
            return
        for ingredient in ingredients:
            for kw in _BEEF_KEYWORDS:
                if re.search(rf"\b{re.escape(kw)}\b", ingredient):
                    issues.append(f"Contains beef ingredient '{kw}' but no_beef constraint is set")
                    return

    def _check_vegan(
        self, ingredients: List[str], constraints: Dict[str, Any], issues: List[str]
    ) -> None:
        if not constraints.get("vegan"):
            return
        for ingredient in ingredients:
            for kw in _MEAT_KEYWORDS:
                if re.search(rf"\b{re.escape(kw)}\b", ingredient):
                    issues.append(f"Contains meat ingredient '{kw}' but vegan constraint is set")
                    return
        for ingredient in ingredients:
            for kw in _DAIRY_KEYWORDS:
                if re.search(rf"\b{re.escape(kw)}\b", ingredient):
                    issues.append(f"Contains dairy ingredient '{kw}' but vegan constraint is set")
                    return
