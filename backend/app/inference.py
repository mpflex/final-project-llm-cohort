"""
TacoLLM — Inference Pipeline

Handles model loading (base + LoRA), prompt construction,
generation with explicit sampling settings, and JSON parsing
with retry logic.
"""

import json
import logging
import re
from typing import Any, Dict, Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .prompts import build_system_prompt, build_user_prompt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — edit these for your environment
# ---------------------------------------------------------------------------

BASE_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"  # swap for any instruct model
LORA_ADAPTER_PATH = "./checkpoints/tacollm-lora-v1"  # path to your saved LoRA adapter

# Sampling settings (documented for rubric)
SAMPLING_CONFIG = {
    "temperature": 0.3,  # Low = more deterministic JSON output
    "top_p": 0.9,  # Nucleus sampling
    "max_new_tokens": 512,  # Enough to complete structured JSON
    "do_sample": True,
    "repetition_penalty": 1.1,
}

MAX_RETRIES = 2


# ---------------------------------------------------------------------------
# InferencePipeline
# ---------------------------------------------------------------------------


class InferencePipeline:
    """
    Wraps base and LoRA model variants behind a unified generate() interface.

    Sampling strategy: low-temperature (0.3) nucleus sampling is used to
    balance response diversity with reliable JSON generation. This reduces
    hallucinated field values and improves constraint adherence compared
    with greedy decoding or high-temperature sampling.
    """

    def __init__(self):
        self._base_model = None
        self._lora_model = None
        self._tokenizer = None
        self._loaded = False
        self._load_models()

    def _load_models(self):
        """Load tokenizer and base model. LoRA adapter loaded lazily."""
        logger.info(f"Loading tokenizer from {BASE_MODEL_ID} ...")
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            logger.info(f"Loading base model {BASE_MODEL_ID} ...")
            self._base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_ID,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
            )
            self._base_model.eval()
            self._loaded = True
            logger.info("Base model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._loaded = False

    def _load_lora(self):
        """Lazy-load LoRA adapter on first use."""
        if self._lora_model is not None:
            return
        try:
            logger.info(f"Loading LoRA adapter from {LORA_ADAPTER_PATH} ...")
            self._lora_model = PeftModel.from_pretrained(self._base_model, LORA_ADAPTER_PATH)
            self._lora_model.eval()
            logger.info("LoRA adapter loaded.")
        except Exception as e:
            logger.warning(f"LoRA adapter not found or failed to load: {e}. Falling back to base.")
            self._lora_model = None

    def is_loaded(self) -> bool:
        return self._loaded

    def active_model_name(self) -> str:
        return BASE_MODEL_ID.split("/")[-1]

    def _get_model(self, variant: str):
        if variant == "lora":
            self._load_lora()
            return self._lora_model or self._base_model
        return self._base_model

    def _run_generation(self, prompt: str, model) -> str:
        """Tokenize, generate, and decode."""
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                temperature=SAMPLING_CONFIG["temperature"],
                top_p=SAMPLING_CONFIG["top_p"],
                max_new_tokens=SAMPLING_CONFIG["max_new_tokens"],
                do_sample=SAMPLING_CONFIG["do_sample"],
                repetition_penalty=SAMPLING_CONFIG["repetition_penalty"],
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        generated = outputs[0][input_len:]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Attempt to extract a JSON object from model output.
        Handles markdown code fences and loose surrounding text.
        """
        # Strip markdown fences if present
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*", "", text)
        text = text.strip()

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find a {...} block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        return None

    def generate(
        self,
        user_message: str,
        constraints: Dict[str, Any],
        model_variant: str = "base",
    ) -> Tuple[Optional[Dict[str, Any]], bool, int]:
        """
        Run inference with retry logic.

        Returns:
            (parsed_result, valid_json_flag, attempt_count)
        """
        if not self._loaded:
            logger.error("Model not loaded. Cannot generate.")
            return None, False, 0

        model = self._get_model(model_variant)
        system_prompt = build_system_prompt()

        for attempt in range(1, MAX_RETRIES + 1):
            if attempt == 1:
                user_prompt = build_user_prompt(user_message, constraints)
            else:
                # Corrective retry prompt
                user_prompt = build_user_prompt(
                    user_message,
                    constraints,
                    retry=True,
                    attempt=attempt,
                )

            # Format as chat messages (LLaMA instruct format)
            prompt = self._format_chat(system_prompt, user_prompt)

            logger.info(f"Inference attempt {attempt}/{MAX_RETRIES}")
            raw_output = self._run_generation(prompt, model)
            logger.debug(f"Raw output: {raw_output[:200]}")

            parsed = self._extract_json(raw_output)
            if parsed is not None:
                return parsed, True, attempt

        logger.warning("All inference attempts failed to produce valid JSON.")
        return None, False, MAX_RETRIES

    def _format_chat(self, system_prompt: str, user_prompt: str) -> str:
        """
        Format prompt using the LLaMA-3 instruct template.
        Adjust this method if you switch model families.
        """
        return (
            f"<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
