"""
TacoLLM — Training Data Formatter

Converts raw training examples (instruction + output dict) into
formatted LLaMA-3 chat strings for SFT with TRL's SFTTrainer.

The prompt format exactly matches app/inference.py:_format_chat() so that
the fine-tuned model learns to respond in the expected inference format.
"""

import json
from typing import Any, Dict

# System prompt — must stay in sync with app/prompts.py:build_system_prompt().
# Duplicated here to keep training/ independent of the app/ package.
TRAINING_SYSTEM_PROMPT = """You are TacoLLM, an expert taco recommendation assistant.

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
- reasoning (string)"""


def format_training_example(example: Dict[str, Any]) -> str:
    """
    Format a training example as a LLaMA-3 chat completion string.

    The assistant turn ends with <|eot_id|> to teach the model the correct
    stop boundary. Output JSON is serialized without indentation to keep
    sequence lengths short.
    """
    instruction = example["instruction"]
    output_str = json.dumps(example["output"])
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{TRAINING_SYSTEM_PROMPT}"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{instruction}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{output_str}"
        "<|eot_id|>"
    )
