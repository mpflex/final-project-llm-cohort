"""
TacoLLM — Constraint Parser

Extracts structured dietary and nutritional constraints from
free-text user messages using deterministic regex rules.
"""

import re
from typing import Any, Dict


class ConstraintParser:
    """
    Deterministic constraint extraction from natural language.

    Returns a dict of constraint keys matching the keys used in
    SessionMemory, prompt builders, and the validator.
    """

    def extract(self, message: str) -> Dict[str, Any]:
        msg = message.lower()
        constraints: Dict[str, Any] = {}

        self._extract_calories(msg, constraints)
        self._extract_macros(msg, constraints)
        self._extract_dietary(msg, constraints)
        self._extract_spice(msg, constraints)
        self._extract_style(msg, constraints)

        return constraints

    # ------------------------------------------------------------------
    # Private extraction helpers
    # ------------------------------------------------------------------

    def _extract_calories(self, msg: str, out: Dict[str, Any]) -> None:
        match = re.search(r"under\s+(\d+)\s*(?:calories?|kcal|cal)\b", msg)
        if match:
            out["max_calories"] = int(match.group(1))
            return
        match = re.search(r"(\d+)\s*(?:calories?|kcal|cal)\s+(?:or\s+)?(?:under|less|max)\b", msg)
        if match:
            out["max_calories"] = int(match.group(1))

    def _extract_macros(self, msg: str, out: Dict[str, Any]) -> None:
        if re.search(r"high[- ]protein|protein[- ]rich|protein[- ]packed", msg):
            out["high_protein"] = True
        if re.search(r"low[- ]carb", msg):
            out["low_carb"] = True
        if re.search(r"\bketo\b", msg):
            out["keto"] = True
            out["low_carb"] = True

    def _extract_dietary(self, msg: str, out: Dict[str, Any]) -> None:
        if re.search(r"\bvegan\b", msg):
            out["vegan"] = True
            out["vegetarian"] = True
        elif re.search(r"\bvegetarian\b", msg):
            out["vegetarian"] = True

        if re.search(r"no\s+dairy|dairy[- ]free|without\s+dairy", msg):
            out["no_dairy"] = True
        if re.search(r"no\s+beef|beef[- ]free|without\s+beef", msg):
            out["no_beef"] = True
        if re.search(r"no\s+gluten|gluten[- ]free|without\s+gluten", msg):
            out["no_gluten"] = True

    def _extract_spice(self, msg: str, out: Dict[str, Any]) -> None:
        # \bspice\b intentionally excluded: too broad, matches "spice" in phrases like
        # "medium spice level" which should route to medium, not hot.
        if re.search(r"\bhot\b|extra\s+spicy|very\s+spicy|\bspicy\b", msg):
            out["spice_level"] = "hot"
        elif re.search(r"\bmild\b", msg):
            out["spice_level"] = "mild"
        elif re.search(r"\bmedium\b", msg):
            out["spice_level"] = "medium"

    def _extract_style(self, msg: str, out: Dict[str, Any]) -> None:
        if re.search(r"street\s+taco", msg):
            out["preferred_style"] = "street"
        elif re.search(r"tex[- ]mex", msg):
            out["preferred_style"] = "tex-mex"
        elif re.search(r"breakfast\s+taco", msg):
            out["preferred_style"] = "breakfast"
        elif re.search(r"\bhealthy\b", msg):
            out["preferred_style"] = "healthy"
