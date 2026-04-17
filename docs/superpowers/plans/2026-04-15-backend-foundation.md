# Backend Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the three empty backend modules (`parser.py`, `validator.py`, `memory.py`), wire up the project with `uv` and `ruff`, and achieve full test coverage.

**Architecture:** `ConstraintParser` uses regex to extract structured constraints from free-text. `TacoValidator` checks a taco JSON against those constraints and the required schema. `SessionMemory` holds per-session preference state in-process. All three are already imported and called by `main.py` — this plan makes them real.

**Tech Stack:** Python 3.12, uv, FastAPI, pytest, ruff

---

## File Map

| File | Status | Responsibility |
|------|--------|----------------|
| `backend/pyproject.toml` | Create | uv project config, ruff config, pytest config |
| `backend/app/__init__.py` | Create | Package marker |
| `backend/app/parser.py` | Complete (currently 1 empty line) | Regex-based constraint extraction |
| `backend/app/validator.py` | Complete (currently 1 empty line) | Schema + constraint validation |
| `backend/app/memory.py` | Complete (currently 1 empty line) | In-process session preference store |
| `backend/tests/__init__.py` | Create | Test package marker |
| `backend/tests/test_parser.py` | Create | Full coverage of ConstraintParser |
| `backend/tests/test_validator.py` | Create | Full coverage of TacoValidator |
| `backend/tests/test_memory.py` | Create | Full coverage of SessionMemory |

---

## Task 1: Project Setup — `pyproject.toml`

**Files:**
- Create: `backend/pyproject.toml`

- [ ] **Step 1: Create `backend/pyproject.toml`**

```toml
[project]
name = "tacollm-backend"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.111.0",
    "uvicorn[standard]>=0.30.0",
    "pydantic>=2.7.0",
    "transformers>=4.41.0",
    "peft>=0.11.0",
    "torch>=2.3.0",
    "accelerate>=0.30.0",
    "boto3>=1.34.0",
    "sagemaker>=2.220.0",
    "anthropic>=0.28.0",
]

[dependency-groups]
dev = [
    "pytest>=8.2.0",
    "pytest-cov>=5.0.0",
    "ruff>=0.4.0",
    "httpx>=0.27.0",
]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I"]
ignore = ["E501"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--cov=app --cov-report=term-missing"
```

- [ ] **Step 2: Create package markers**

Create `backend/app/__init__.py` — empty file:
```python
```

Create `backend/tests/__init__.py` — empty file:
```python
```

- [ ] **Step 3: Install dependencies**

```bash
cd backend
uv sync --dev
```

Expected: uv resolves and installs all packages into `.venv/`.

- [ ] **Step 4: Verify ruff works**

```bash
cd backend
uv run ruff check app/
```

Expected: exits 0 (no files to lint yet / empty files pass).


---

## Task 2: `ConstraintParser` — TDD

**Files:**
- Create: `backend/tests/test_parser.py`
- Complete: `backend/app/parser.py`

- [ ] **Step 1: Write failing tests**

Create `backend/tests/test_parser.py`:

```python
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
```

- [ ] **Step 2: Run tests — verify they all fail**

```bash
cd backend
uv run pytest tests/test_parser.py -v
```

Expected: `ModuleNotFoundError` or `ImportError` — `ConstraintParser` does not exist yet.

- [ ] **Step 3: Implement `ConstraintParser`**

Write `backend/app/parser.py`:

```python
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
        if re.search(r"\bhot\b|extra\s+spicy|very\s+spicy|\bspicy\b|\bspice\b", msg):
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
```

- [ ] **Step 4: Run tests — verify they all pass**

```bash
cd backend
uv run pytest tests/test_parser.py -v
```

Expected: all tests PASS, 0 failures.

- [ ] **Step 5: Lint and format**

```bash
cd backend
uv run ruff check app/parser.py tests/test_parser.py
uv run ruff format app/parser.py tests/test_parser.py
```

Expected: exits 0.


---

## Task 3: `TacoValidator` — TDD

**Files:**
- Create: `backend/tests/test_validator.py`
- Complete: `backend/app/validator.py`

- [ ] **Step 1: Write failing tests**

Create `backend/tests/test_validator.py`:

```python
import pytest
from app.validator import TacoValidator


@pytest.fixture
def validator():
    return TacoValidator()


@pytest.fixture
def valid_taco():
    return {
        "name": "Chipotle Chicken Taco",
        "ingredients": ["corn tortillas", "grilled chicken breast", "salsa roja", "cilantro", "onion"],
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


class TestVeganConstraint:
    def test_chicken_flagged_when_vegan(self, validator, valid_taco):
        issues = validator.validate(valid_taco, {"vegan": True})
        assert any("vegan" in i.lower() for i in issues)

    def test_vegan_taco_passes(self, validator, valid_taco):
        valid_taco["ingredients"] = ["corn tortillas", "black beans", "salsa verde", "avocado", "cilantro"]
        assert validator.validate(valid_taco, {"vegan": True}) == []


class TestNoBeefConstraint:
    def test_beef_flagged_when_no_beef(self, validator, valid_taco):
        valid_taco["ingredients"] = ["flour tortilla", "ground beef", "salsa", "onion"]
        issues = validator.validate(valid_taco, {"no_beef": True})
        assert any("beef" in i.lower() for i in issues)

    def test_chicken_passes_no_beef(self, validator, valid_taco):
        assert validator.validate(valid_taco, {"no_beef": True}) == []
```

- [ ] **Step 2: Run tests — verify they all fail**

```bash
cd backend
uv run pytest tests/test_validator.py -v
```

Expected: `ImportError` — `TacoValidator` does not exist yet.

- [ ] **Step 3: Implement `TacoValidator`**

Write `backend/app/validator.py`:

```python
"""
TacoLLM — Taco Output Validator

Validates a generated taco JSON against the required schema
and checks it against the user's parsed constraints.
"""

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
    "cheese", "sour cream", "crema", "queso", "milk", "butter",
    "cream", "yogurt", "cotija", "oaxaca", "cheddar", "jack", "fresco",
]
_MEAT_KEYWORDS = [
    "chicken", "beef", "pork", "carnitas", "steak", "carne", "chorizo",
    "al pastor", "barbacoa", "birria", "fish", "shrimp", "seafood",
    "turkey", "lamb",
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

        ingredients_str = " ".join(i.lower() for i in taco.get("ingredients", []))
        self._check_calorie_constraint(taco, constraints, issues)
        self._check_no_dairy(ingredients_str, constraints, issues)
        self._check_no_beef(ingredients_str, constraints, issues)
        self._check_vegan(ingredients_str, constraints, issues)

        return issues

    # ------------------------------------------------------------------

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
        self, ingredients_str: str, constraints: Dict[str, Any], issues: List[str]
    ) -> None:
        if not constraints.get("no_dairy"):
            return
        for kw in _DAIRY_KEYWORDS:
            if kw in ingredients_str:
                issues.append(f"Contains dairy ingredient '{kw}' but no_dairy constraint is set")
                return

    def _check_no_beef(
        self, ingredients_str: str, constraints: Dict[str, Any], issues: List[str]
    ) -> None:
        if not constraints.get("no_beef"):
            return
        for kw in _BEEF_KEYWORDS:
            if kw in ingredients_str:
                issues.append(f"Contains beef ingredient '{kw}' but no_beef constraint is set")
                return

    def _check_vegan(
        self, ingredients_str: str, constraints: Dict[str, Any], issues: List[str]
    ) -> None:
        if not constraints.get("vegan"):
            return
        for kw in _MEAT_KEYWORDS:
            if kw in ingredients_str:
                issues.append(f"Contains meat ingredient '{kw}' but vegan constraint is set")
                return
        for kw in _DAIRY_KEYWORDS:
            if kw in ingredients_str:
                issues.append(f"Contains dairy ingredient '{kw}' but vegan constraint is set")
                return
```

- [ ] **Step 4: Run tests — verify they all pass**

```bash
cd backend
uv run pytest tests/test_validator.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Lint and format**

```bash
cd backend
uv run ruff check app/validator.py tests/test_validator.py
uv run ruff format app/validator.py tests/test_validator.py
```

Expected: exits 0.


---

## Task 4: `SessionMemory` — TDD

**Files:**
- Create: `backend/tests/test_memory.py`
- Complete: `backend/app/memory.py`

- [ ] **Step 1: Write failing tests**

Create `backend/tests/test_memory.py`:

```python
import pytest
from app.memory import SessionMemory


@pytest.fixture
def memory():
    return SessionMemory()


class TestGet:
    def test_unknown_session_returns_empty_dict(self, memory):
        assert memory.get("new_session") == {}

    def test_get_returns_copy_not_reference(self, memory):
        memory.update("s1", {"high_protein": True})
        result = memory.get("s1")
        result["injected"] = True
        assert "injected" not in memory.get("s1")


class TestUpdate:
    def test_stores_truthy_constraint(self, memory):
        memory.update("s1", {"high_protein": True})
        assert memory.get("s1")["high_protein"] is True

    def test_stores_integer_constraint(self, memory):
        memory.update("s1", {"max_calories": 400})
        assert memory.get("s1")["max_calories"] == 400

    def test_stores_string_constraint(self, memory):
        memory.update("s1", {"spice_level": "hot"})
        assert memory.get("s1")["spice_level"] == "hot"

    def test_merges_with_existing_preferences(self, memory):
        memory.update("s1", {"high_protein": True})
        memory.update("s1", {"no_dairy": True})
        result = memory.get("s1")
        assert result["high_protein"] is True
        assert result["no_dairy"] is True

    def test_false_value_not_stored(self, memory):
        memory.update("s1", {"vegan": False, "high_protein": True})
        result = memory.get("s1")
        assert "vegan" not in result
        assert result["high_protein"] is True

    def test_none_value_not_stored(self, memory):
        memory.update("s1", {"max_calories": None})
        assert "max_calories" not in memory.get("s1")

    def test_later_value_overwrites_earlier(self, memory):
        memory.update("s1", {"spice_level": "mild"})
        memory.update("s1", {"spice_level": "hot"})
        assert memory.get("s1")["spice_level"] == "hot"


class TestClear:
    def test_clear_removes_all_preferences(self, memory):
        memory.update("s1", {"high_protein": True, "no_dairy": True})
        memory.clear("s1")
        assert memory.get("s1") == {}

    def test_clear_nonexistent_session_does_not_raise(self, memory):
        memory.clear("does_not_exist")

    def test_clear_one_session_leaves_others_intact(self, memory):
        memory.update("s1", {"high_protein": True})
        memory.update("s2", {"vegan": True})
        memory.clear("s1")
        assert memory.get("s2")["vegan"] is True


class TestIsolation:
    def test_sessions_do_not_share_state(self, memory):
        memory.update("s1", {"high_protein": True})
        memory.update("s2", {"vegan": True})
        assert "vegan" not in memory.get("s1")
        assert "high_protein" not in memory.get("s2")
```

- [ ] **Step 2: Run tests — verify they all fail**

```bash
cd backend
uv run pytest tests/test_memory.py -v
```

Expected: `ImportError` — `SessionMemory` does not exist yet.

- [ ] **Step 3: Implement `SessionMemory`**

Write `backend/app/memory.py`:

```python
"""
TacoLLM — Session Memory

Stores per-session user constraint preferences in memory.
Preferences persist for the lifetime of the server process;
they are cleared on explicit user action or server restart.
"""

from typing import Any, Dict


class SessionMemory:
    """
    Key-value preference store keyed by session_id.

    Only truthy values are stored — False and None are silently dropped
    so that the absence of a key means "no preference" throughout the
    rest of the pipeline.
    """

    def __init__(self) -> None:
        self._store: Dict[str, Dict[str, Any]] = {}

    def get(self, session_id: str) -> Dict[str, Any]:
        """Return a copy of stored preferences for the session."""
        return dict(self._store.get(session_id, {}))

    def update(self, session_id: str, constraints: Dict[str, Any]) -> None:
        """
        Merge new constraints into the session store.
        False and None values are ignored.
        """
        if session_id not in self._store:
            self._store[session_id] = {}
        for key, value in constraints.items():
            if value is not None and value is not False:
                self._store[session_id][key] = value

    def clear(self, session_id: str) -> None:
        """Remove all stored preferences for the session."""
        self._store.pop(session_id, None)
```

- [ ] **Step 4: Run tests — verify they all pass**

```bash
cd backend
uv run pytest tests/test_memory.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Lint and format**

```bash
cd backend
uv run ruff check app/memory.py tests/test_memory.py
uv run ruff format app/memory.py tests/test_memory.py
```

Expected: exits 0.

---

## Task 5: Full Suite — Coverage Gate

- [ ] **Step 1: Run full test suite with coverage**

```bash
cd backend
uv run pytest --cov=app --cov-report=term-missing -v
```

Expected output (approximate):
```
tests/test_memory.py ............ PASSED
tests/test_parser.py ............ PASSED
tests/test_validator.py ......... PASSED

---------- coverage: app ----------
app/memory.py          100%
app/parser.py          100%
app/validator.py       100%
app/prompts.py          --   (not yet covered — OK for this plan)
app/inference.py        --   (requires GPU — excluded from this plan)
app/main.py             --   (integration — excluded from this plan)
```

- [ ] **Step 2: Run ruff on all backend source**

```bash
cd backend
uv run ruff check app/ tests/
uv run ruff format --check app/ tests/
```

Expected: exits 0 on both.
