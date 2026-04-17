"""
TacoLLM — LoRA Fine-Tuning Entry Point for SageMaker

Runs inside the SageMaker training container. Loads the base model,
applies a LoRA adapter, trains on the formatted taco dataset, and saves
the adapter to SM_MODEL_DIR for packaging as model.tar.gz.

Hyperparameters (passed by SageMaker estimator as CLI args):
    --model-id           Base model HuggingFace ID
    --lora-r             LoRA rank (default: 16)
    --lora-alpha         LoRA alpha (default: 32)
    --lora-dropout       LoRA dropout (default: 0.05)
    --learning-rate      Learning rate (default: 2e-4)
    --num-epochs         Training epochs (default: 3)
    --per-device-batch   Per-device batch size (default: 4)
    --grad-accum         Gradient accumulation steps (default: 4)
    --max-seq-length     Max sequence length in tokens (default: 512)

SageMaker env vars (set automatically by the platform):
    SM_CHANNEL_TRAINING  Path to uploaded training data directory
    SM_MODEL_DIR         Path where model artifacts must be saved
"""

import argparse
import json
import logging
import os
from pathlib import Path

import torch
from datasets import Dataset
from format_data import format_training_example
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Target all attention projection + MLP layers in Llama-3
LORA_TARGET_MODULES = [
    "q_proj",
    "v_proj",
    "k_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def load_dataset(data_dir: str) -> Dataset:
    """Load train.jsonl from data_dir, format each example, return HF Dataset."""
    train_path = Path(data_dir) / "train.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found at {train_path}")

    examples = []
    with open(train_path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    formatted = [{"text": format_training_example(ex)} for ex in examples]
    logger.info(f"Loaded {len(formatted)} training examples from {train_path}")
    return Dataset.from_list(formatted)


def train(args: argparse.Namespace) -> None:
    logger.info(f"Loading tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Required for SFT loss masking

    logger.info(f"Loading base model: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset(args.data_dir)

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        save_strategy="epoch",
        report_to="none",
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info(f"Saving LoRA adapter to {args.output_dir}")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Training complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TacoLLM LoRA training")
    # SageMaker sets these env vars automatically
    parser.add_argument("--data-dir", default=os.environ.get("SM_CHANNEL_TRAINING", "./data"))
    parser.add_argument("--output-dir", default=os.environ.get("SM_MODEL_DIR", "./output"))
    # Hyperparameters — passed as CLI args by SageMaker estimator
    parser.add_argument("--model-id", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--per-device-batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=512)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
