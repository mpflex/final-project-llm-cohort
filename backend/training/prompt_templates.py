"""
TacoLLM — Dataset Prompt Templates

Generates natural language instruction prompts for dataset generation.
Templates span all constraint categories defined in the ADR:
  single (20%), two (30%), three+ (25%), style (10%), edge (10%), followup (5%)
"""

import random
from typing import Dict

_CALORIE_THRESHOLDS = [300, 350, 400, 450, 500]
_PROTEINS = ["chicken", "beef", "pork", "fish", "shrimp", "carnitas", "carne asada", "turkey"]
_STYLES = ["street", "tex-mex", "breakfast", "healthy", "authentic Mexican"]
_SPICE_WORDS = ["mild", "medium", "spicy", "extra spicy"]

CATEGORY_DISTRIBUTION: Dict[str, float] = {
    "single": 0.20,
    "two": 0.30,
    "three": 0.25,
    "style": 0.10,
    "edge": 0.10,
    "followup": 0.05,
}


def single_constraint_prompt() -> str:
    """Return a prompt with exactly one dietary/nutritional constraint."""
    cal = random.choice(_CALORIE_THRESHOLDS)
    spice = random.choice(_SPICE_WORDS)
    protein = random.choice(_PROTEINS)
    style = random.choice(_STYLES)
    options = [
        "Give me a high protein taco.",
        "Make me a vegan taco.",
        f"I want a taco under {cal} calories.",
        "Give me a low-carb taco.",
        "Make me a dairy-free taco.",
        f"I want a {spice} taco.",
        "Give me a keto-friendly taco.",
        "Make me a vegetarian taco.",
        f"I want a {style} style taco.",
        "Give me a gluten-free taco.",
        f"Make me a {protein} taco.",
        "I want a protein-packed taco.",
    ]
    return random.choice(options)


def two_constraint_prompt() -> str:
    """Return a prompt with two intersecting constraints."""
    cal = random.choice(_CALORIE_THRESHOLDS)
    spice = random.choice(_SPICE_WORDS)
    protein = random.choice(_PROTEINS)
    style = random.choice(_STYLES)
    options = [
        f"Give me a high protein taco under {cal} calories.",
        f"Make me a vegan taco that is {spice}.",
        f"I want a dairy-free taco under {cal} calories.",
        f"Give me a low-carb taco with {protein}.",
        "Make me a keto taco with no dairy.",
        f"I want a vegetarian taco under {cal} calories.",
        f"Give me a high protein {style} taco.",
        f"Make me a {spice} taco with no beef.",
        f"I want a gluten-free taco with {protein}.",
        "Give me a dairy-free high protein taco.",
        f"Make me a low-carb {spice} taco.",
        f"I want a vegan taco under {cal} calories.",
    ]
    return random.choice(options)


def three_constraint_prompt() -> str:
    """Return a prompt with three or more intersecting constraints."""
    cal = random.choice(_CALORIE_THRESHOLDS)
    spice = random.choice(_SPICE_WORDS)
    protein = random.choice(_PROTEINS)
    style = random.choice(_STYLES)
    options = [
        f"Give me a high protein taco under {cal} calories with no dairy.",
        f"Make me a vegan low-carb taco that is {spice}.",
        f"I want a keto taco under {cal} calories that is {spice}.",
        f"Give me a dairy-free high protein taco under {cal} calories.",
        f"Make me a gluten-free vegetarian taco that is {spice} and under {cal} calories.",
        f"I want a high protein {style} taco with no dairy.",
        f"Give me a keto {spice} taco with {protein}.",
        f"Make me a vegan {style} taco under {cal} calories.",
        f"I want a low-carb dairy-free taco with {protein} under {cal} calories.",
        f"Give me a high protein {spice} {style} taco.",
    ]
    return random.choice(options)


def style_prompt() -> str:
    """Return a style or cuisine-focused prompt."""
    cal = random.choice(_CALORIE_THRESHOLDS)
    spice = random.choice(_SPICE_WORDS)
    protein = random.choice(_PROTEINS)
    options = [
        "Give me an authentic street taco.",
        "Make me a classic tex-mex taco.",
        "I want a breakfast taco.",
        "Give me a healthy taco.",
        f"Make me a street taco with {protein}.",
        f"I want a tex-mex style taco under {cal} calories.",
        "Give me an authentic Mexican taco.",
        f"Make me a healthy taco that is {spice}.",
        f"I want a breakfast taco with {protein}.",
        f"Give me a street-style taco under {cal} calories.",
    ]
    return random.choice(options)


def edge_case_prompt() -> str:
    """Return a challenging or contradictory constraint prompt."""
    options = [
        "Give me a high protein keto vegan taco under 300 calories.",
        "Make me a super low calorie taco under 200 calories that is still filling.",
        "I want an ultra-high protein taco with over 50g of protein.",
        "Give me a taco with no beef, no chicken, no pork, and no fish.",
        "Make me a keto vegan taco.",
        "I want the spiciest taco possible under 350 calories.",
        "Give me a taco with minimal carbs, high protein, and no dairy.",
        "Make me a filling taco under 250 calories.",
        "I want a taco with at least 5 different vegetables.",
        "Give me a no-compromise taco: high protein, low carb, and dairy-free.",
    ]
    return random.choice(options)


def followup_prompt() -> str:
    """Return a follow-up prompt that references prior preferences."""
    spice = random.choice(_SPICE_WORDS)
    protein = random.choice(_PROTEINS)
    options = [
        f"Same as before but make it {spice}.",
        f"Keep the same dietary restrictions but use {protein} instead.",
        "Make it a bit spicier.",
        "Can you make a lighter version of that?",
        "Same thing but vegan.",
        "Make it under 350 calories this time.",
        "Keep the protein high but make it dairy-free.",
        "Same constraints but make it a street taco style.",
        "Make it milder for me.",
        "Keep the same but add more protein.",
    ]
    return random.choice(options)


def get_prompt_for_category(category: str) -> str:
    """Return a random instruction prompt for the given category name."""
    generators = {
        "single": single_constraint_prompt,
        "two": two_constraint_prompt,
        "three": three_constraint_prompt,
        "style": style_prompt,
        "edge": edge_case_prompt,
        "followup": followup_prompt,
    }
    if category not in generators:
        raise ValueError(f"Unknown category: {category!r}. Must be one of: {list(generators)}")
    return generators[category]()


def get_category_counts(total: int) -> Dict[str, int]:
    """
    Return per-category example counts that sum exactly to `total`.

    Distributes according to CATEGORY_DISTRIBUTION, with the last
    category absorbing any rounding remainder.
    """
    categories = list(CATEGORY_DISTRIBUTION.keys())
    counts: Dict[str, int] = {}
    remaining = total
    for cat in categories[:-1]:
        counts[cat] = round(total * CATEGORY_DISTRIBUTION[cat])
        remaining -= counts[cat]
    counts[categories[-1]] = remaining
    return counts
