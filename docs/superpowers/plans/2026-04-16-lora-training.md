# LoRA Fine-Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune `meta-llama/Llama-3.2-3B-Instruct` with LoRA on the generated taco dataset using a SageMaker managed training job, saving the adapter to `backend/checkpoints/tacollm-lora-v1/` for use by the existing inference pipeline.

**Architecture:** A pure-function data formatter (`format_data.py`) converts JSONL training examples into LLaMA-3 chat-format strings matching the inference pipeline's prompt template exactly. A SageMaker entry-point script (`train_lora.py`) loads the base model, applies a LoRA adapter via PEFT, and trains with TRL's `SFTTrainer`. A launcher script (`sagemaker_job.py`) uploads data to S3, configures the `HuggingFace` estimator, and starts the managed job.

**Tech Stack:** Python 3.12, `peft`, `trl`, `datasets`, `transformers`, `sagemaker` SDK, AWS SageMaker (`ml.g5.2xlarge`), S3

---

## File Map

| File | Status | Responsibility |
|------|--------|----------------|
| `backend/training/format_data.py` | Create | Convert JSONL example → LLaMA-3 chat training string |
| `backend/training/train_lora.py` | Create | SageMaker entry point: load model, apply LoRA, train, save adapter |
| `backend/training/requirements.txt` | Create | Extra pip deps installed by SageMaker container (trl, peft, datasets) |
| `backend/training/sagemaker_job.py` | Create | Upload data to S3, configure estimator, launch training job |
| `backend/checkpoints/.gitignore` | Create | Prevent large model files from being committed |
| `backend/tests/test_format_data.py` | Create | Full coverage of format_data.py pure functions |
| `backend/pyproject.toml` | Modify | Add `trl` and `datasets` to dependencies |

---

## Task 1: Dependencies + Format Data — TDD

**Files:**
- Modify: `backend/pyproject.toml`
- Create: `backend/tests/test_format_data.py`
- Create: `backend/training/format_data.py`

- [ ] **Step 1: Add `trl` and `datasets` to pyproject.toml**

Open `backend/pyproject.toml`. In the `[project]` `dependencies` list, add two new lines after `"anthropic"`:

```toml
"trl>=0.7.0",
"datasets>=2.16.0",
```

The full dependencies list should look like:

```toml
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn>=0.34.0",
    "pydantic>=2.10.0",
    "transformers>=4.46.0",
    "peft>=0.13.0",
    "torch>=2.4.0",
    "accelerate>=1.2.0",
    "boto3>=1.35.0",
    "sagemaker>=2.232.0",
    "anthropic>=0.40.0",
    "trl>=0.7.0",
    "datasets>=2.16.0",
]
```

Then sync:

```bash
cd /Users/mpineda/Desktop/llm-class-2026/llm-class-2026-winter-cohort/backend
uv sync
```

Expected: resolves and installs without error.

- [ ] **Step 2: Write failing tests**

Create `backend/tests/test_format_data.py`:

```python
import json

import pytest

from training.format_data import TRAINING_SYSTEM_PROMPT, format_training_example


@pytest.fixture
def example():
    return {
        "instruction": "Give me a high protein taco under 400 calories.",
        "output": {
            "name": "Chipotle Chicken Taco",
            "ingredients": ["corn tortilla", "grilled chicken", "salsa"],
            "calories": 350,
            "protein": 32,
            "carbs": 20,
            "fat": 8,
            "dietary_tags": ["high_protein"],
            "spice_level": "medium",
            "reasoning": "Lean chicken keeps protein high.",
        },
    }


class TestFormatTrainingExample:
    def test_returns_string(self, example):
        assert isinstance(format_training_example(example), str)

    def test_starts_with_begin_token(self, example):
        assert format_training_example(example).startswith("<|begin_of_text|>")

    def test_contains_system_header(self, example):
        assert "<|start_header_id|>system<|end_header_id|>" in format_training_example(example)

    def test_contains_system_prompt(self, example):
        assert TRAINING_SYSTEM_PROMPT in format_training_example(example)

    def test_contains_user_header(self, example):
        assert "<|start_header_id|>user<|end_header_id|>" in format_training_example(example)

    def test_contains_instruction(self, example):
        assert example["instruction"] in format_training_example(example)

    def test_contains_assistant_header(self, example):
        assert "<|start_header_id|>assistant<|end_header_id|>" in format_training_example(example)

    def test_contains_compact_output_json(self, example):
        # Output JSON must be compact (no indent) to keep sequence lengths short
        assert json.dumps(example["output"]) in format_training_example(example)

    def test_ends_with_eot(self, example):
        assert format_training_example(example).endswith("<|eot_id|>")

    def test_exactly_three_eot_tokens(self, example):
        # One after system, one after user, one after assistant
        assert format_training_example(example).count("<|eot_id|>") == 3

    def test_different_instructions_produce_different_strings(self):
        ex1 = {"instruction": "Prompt A", "output": {"x": 1}}
        ex2 = {"instruction": "Prompt B", "output": {"x": 1}}
        assert format_training_example(ex1) != format_training_example(ex2)

    def test_different_outputs_produce_different_strings(self):
        ex1 = {"instruction": "Same", "output": {"x": 1}}
        ex2 = {"instruction": "Same", "output": {"x": 2}}
        assert format_training_example(ex1) != format_training_example(ex2)
```

- [ ] **Step 3: Run tests — verify they fail**

```bash
cd /Users/mpineda/Desktop/llm-class-2026/llm-class-2026-winter-cohort/backend
uv run pytest tests/test_format_data.py -v
```

Expected: `ImportError` — `format_data` does not exist yet.

- [ ] **Step 4: Implement `format_data.py`**

Create `backend/training/format_data.py`:

```python
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
- Keep ingredient lists realistic (4\u20137 items).
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
```

- [ ] **Step 5: Run tests — verify they all pass**

```bash
cd /Users/mpineda/Desktop/llm-class-2026/llm-class-2026-winter-cohort/backend
uv run pytest tests/test_format_data.py -v
```

Expected: all 12 PASS.

- [ ] **Step 6: Lint and format**

```bash
cd /Users/mpineda/Desktop/llm-class-2026/llm-class-2026-winter-cohort/backend
uv run ruff check training/format_data.py tests/test_format_data.py
uv run ruff format training/format_data.py tests/test_format_data.py
```

Expected: exits 0.

---

## Task 2: Training Script

**Files:**
- Create: `backend/training/train_lora.py`
- Create: `backend/training/requirements.txt`
- Create: `backend/checkpoints/.gitignore`

No unit tests for this task — `train_lora.py` requires a real GPU, model weights, and multi-GB downloads. It is verified by the smoke test in Task 4.

- [ ] **Step 1: Create `backend/training/requirements.txt`**

SageMaker's HuggingFace DLC includes `transformers` and `torch`, but not `trl`, `peft`, or `datasets`. SageMaker automatically installs `requirements.txt` from the `source_dir` before running the entry point.

Create `backend/training/requirements.txt`:

```
trl>=0.7.0
peft>=0.13.0
datasets>=2.16.0
accelerate>=0.27.0
```

- [ ] **Step 2: Create `backend/training/train_lora.py`**

Create `backend/training/train_lora.py`:

```python
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
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from format_data import format_training_example

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
    parser.add_argument(
        "--data-dir", default=os.environ.get("SM_CHANNEL_TRAINING", "./data")
    )
    parser.add_argument(
        "--output-dir", default=os.environ.get("SM_MODEL_DIR", "./output")
    )
    # Hyperparameters — passed as CLI args by SageMaker estimator
    parser.add_argument(
        "--model-id", default="meta-llama/Llama-3.2-3B-Instruct"
    )
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
```

- [ ] **Step 3: Create `backend/checkpoints/.gitignore`**

Create `backend/checkpoints/.gitignore`:

```
# Prevent large model weights from being committed.
# The LoRA adapter lives here after downloading from SageMaker.
*.bin
*.safetensors
*.pt
*.pth
*.gguf
# Keep the gitignore and any small config files
!.gitignore
!*.json
!*.txt
```

- [ ] **Step 4: Verify the training package still imports correctly**

```bash
cd /Users/mpineda/Desktop/llm-class-2026/llm-class-2026-winter-cohort/backend
uv run python -c "from training.format_data import format_training_example; print('format_data OK')"
```

Expected: `format_data OK`

---

## Task 3: SageMaker Launcher

**Files:**
- Create: `backend/training/sagemaker_job.py`

No unit tests — requires live AWS credentials and a real S3 bucket. Verified manually in Task 4.

- [ ] **Step 1: Create `backend/training/sagemaker_job.py`**

Create `backend/training/sagemaker_job.py`:

```python
"""
TacoLLM — SageMaker Training Job Launcher

Uploads training data to S3, configures a HuggingFace estimator, and
launches a managed SageMaker training job for LoRA fine-tuning.

Prerequisites:
    - AWS credentials configured (aws configure or IAM instance role)
    - HuggingFace token with Llama-3.2-3B-Instruct access
    - data/train.jsonl exists (generated by generate_dataset.py)
    - SageMaker IAM role with S3 + SageMaker permissions

Usage (run from project root):
    uv run python -m backend.training.sagemaker_job \\
        --bucket my-s3-bucket \\
        --region us-east-1 \\
        --role arn:aws:iam::123456789012:role/SageMakerRole \\
        --hf-token hf_xxxxxxxxxxxx

After the job completes (~60-90 min), download the adapter:
    aws s3 cp s3://BUCKET/tacollm/output/JOB_NAME/output/model.tar.gz /tmp/model.tar.gz
    mkdir -p backend/checkpoints/tacollm-lora-v1
    tar -xzf /tmp/model.tar.gz -C backend/checkpoints/tacollm-lora-v1/

The inference pipeline in backend/app/inference.py will then load the adapter
automatically from LORA_ADAPTER_PATH = "./checkpoints/tacollm-lora-v1".

Hyperparameters (documented for rubric):
    model_id:          meta-llama/Llama-3.2-3B-Instruct
    lora_r:            16
    lora_alpha:        32      (scaling = alpha/r = 2.0)
    lora_dropout:      0.05
    learning_rate:     2e-4
    num_epochs:        3
    per_device_batch:  4
    grad_accum:        4       (effective batch = 16)
    max_seq_length:    512
    optimizer:         AdamW (default in SFTConfig)
    instance_type:     ml.g5.2xlarge (NVIDIA A10G, 24 GB VRAM)
"""

import argparse
import logging
import os
from pathlib import Path

import boto3
from sagemaker.huggingface import HuggingFace

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths relative to project root (where this script is invoked from)
_TRAIN_DATA_PATH = Path("data/train.jsonl")
_SOURCE_DIR = Path("backend/training")


def upload_data(bucket: str, region: str, local_path: Path) -> str:
    """Upload train.jsonl to S3. Returns the S3 URI of the data directory."""
    s3 = boto3.client("s3", region_name=region)
    s3_key = "tacollm/data/train.jsonl"
    logger.info(f"Uploading {local_path} → s3://{bucket}/{s3_key}")
    s3.upload_file(str(local_path), bucket, s3_key)
    uri = f"s3://{bucket}/tacollm/data"
    logger.info(f"Data uploaded to {uri}")
    return uri


def launch_job(
    bucket: str,
    region: str,
    role: str,
    hf_token: str,
    instance_type: str,
    data_uri: str,
) -> str:
    """Configure and start the SageMaker training job. Returns the job name."""
    estimator = HuggingFace(
        entry_point="train_lora.py",
        source_dir=str(_SOURCE_DIR),
        role=role,
        instance_type=instance_type,
        instance_count=1,
        transformers_version="4.36",
        pytorch_version="2.1",
        py_version="py310",
        hyperparameters={
            "model-id": "meta-llama/Llama-3.2-3B-Instruct",
            "lora-r": 16,
            "lora-alpha": 32,
            "lora-dropout": 0.05,
            "learning-rate": 2e-4,
            "num-epochs": 3,
            "per-device-batch": 4,
            "grad-accum": 4,
            "max-seq-length": 512,
        },
        environment={
            "HUGGING_FACE_HUB_TOKEN": hf_token,
        },
        output_path=f"s3://{bucket}/tacollm/output",
        base_job_name="tacollm-lora-v1",
    )

    estimator.fit({"training": data_uri}, wait=False)
    job_name = estimator.latest_training_job.name
    console_url = (
        f"https://{region}.console.aws.amazon.com/sagemaker/home"
        f"?region={region}#/jobs/{job_name}"
    )
    artifact_path = (
        f"s3://{bucket}/tacollm/output/{job_name}/output/model.tar.gz"
    )

    logger.info(f"Training job launched: {job_name}")
    logger.info(f"Monitor at: {console_url}")
    logger.info(f"Artifacts will be at: {artifact_path}")
    return job_name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch TacoLLM LoRA training on SageMaker"
    )
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument(
        "--role", required=True, help="SageMaker IAM role ARN"
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN", ""),
        help="HuggingFace token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--instance-type",
        default="ml.g5.2xlarge",
        help="SageMaker instance type (default: ml.g5.2xlarge, 24 GB VRAM)",
    )
    args = parser.parse_args()

    if not args.hf_token:
        parser.error("--hf-token or HF_TOKEN env var is required")

    if not _TRAIN_DATA_PATH.exists():
        parser.error(
            f"Training data not found at {_TRAIN_DATA_PATH}. "
            "Run generate_dataset.py first."
        )

    data_uri = upload_data(args.bucket, args.region, _TRAIN_DATA_PATH)
    job_name = launch_job(
        bucket=args.bucket,
        region=args.region,
        role=args.role,
        hf_token=args.hf_token,
        instance_type=args.instance_type,
        data_uri=data_uri,
    )
    print(f"\nJob launched: {job_name}")
    print(
        f"\nOnce complete, download the adapter:\n"
        f"  aws s3 cp s3://{args.bucket}/tacollm/output/"
        f"{job_name}/output/model.tar.gz /tmp/model.tar.gz\n"
        f"  mkdir -p backend/checkpoints/tacollm-lora-v1\n"
        f"  tar -xzf /tmp/model.tar.gz -C backend/checkpoints/tacollm-lora-v1/"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Lint and format all new training files**

```bash
cd /Users/mpineda/Desktop/llm-class-2026/llm-class-2026-winter-cohort/backend
uv run ruff check training/train_lora.py training/sagemaker_job.py
uv run ruff format training/train_lora.py training/sagemaker_job.py
```

Expected: exits 0 on both.

---

## Task 4: Full Suite Gate + Launch Instructions

- [ ] **Step 1: Run full test suite**

```bash
cd /Users/mpineda/Desktop/llm-class-2026/llm-class-2026-winter-cohort/backend
uv run pytest --cov=app --cov=training --cov-report=term-missing -v
```

Expected results:
- All 103 tests from Plans 1–2 still pass
- `tests/test_format_data.py` — 12 PASS
- `training/format_data.py` — 100% coverage
- `training/train_lora.py` — 0% coverage (expected; requires GPU)
- `training/sagemaker_job.py` — 0% coverage (expected; requires AWS)

- [ ] **Step 2: Run ruff on all source**

```bash
cd /Users/mpineda/Desktop/llm-class-2026/llm-class-2026-winter-cohort/backend
uv run ruff check app/ training/ tests/
uv run ruff format --check app/ training/ tests/
```

Expected: exits 0 on both.

- [ ] **Step 3: Verify dataset is ready**

```bash
wc -l /Users/mpineda/Desktop/llm-class-2026/llm-class-2026-winter-cohort/data/train.jsonl
```

Expected: `4700 data/train.jsonl` (or close — 94% of 5000).

If the dataset generation is still running, check progress:

```bash
tail -50 /Users/mpineda/Desktop/llm-class-2026/llm-class-2026-winter-cohort/tmp/dataset_gen.log
```

- [ ] **Step 4: Launch SageMaker training job (manual — requires AWS + HF token)**

You will need:
- An S3 bucket in your AWS account (create one if needed: `aws s3 mb s3://my-tacollm-bucket --region us-east-1`)
- A SageMaker IAM role ARN (create via AWS console → IAM → Roles → Create role → SageMaker use case, attach `AmazonSageMakerFullAccess` + `AmazonS3FullAccess`)
- A HuggingFace token with Llama-3.2-3B-Instruct access (https://huggingface.co/settings/tokens)

Run from project root:

```bash
cd /Users/mpineda/Desktop/llm-class-2026/llm-class-2026-winter-cohort
HF_TOKEN=hf_xxxx uv run python -m backend.training.sagemaker_job \
    --bucket YOUR_BUCKET_NAME \
    --region us-east-1 \
    --role arn:aws:iam::ACCOUNT_ID:role/YOUR_SAGEMAKER_ROLE
```

Expected output:
```
INFO Uploading data/train.jsonl → s3://YOUR_BUCKET/tacollm/data/train.jsonl
INFO Data uploaded to s3://YOUR_BUCKET/tacollm/data
INFO Training job launched: tacollm-lora-v1-YYYY-MM-DD-HH-MM-SS
INFO Monitor at: https://us-east-1.console.aws.amazon.com/sagemaker/...
INFO Artifacts will be at: s3://YOUR_BUCKET/tacollm/output/.../model.tar.gz

Job launched: tacollm-lora-v1-YYYY-MM-DD-HH-MM-SS
```

Training takes approximately 60–90 minutes on `ml.g5.2xlarge`.

- [ ] **Step 5: Download and extract adapter (after training completes)**

Run these commands once the SageMaker job status is `Completed`:

```bash
# Replace JOB_NAME and YOUR_BUCKET with the actual values printed above
JOB_NAME=tacollm-lora-v1-YYYY-MM-DD-HH-MM-SS
BUCKET=YOUR_BUCKET_NAME

aws s3 cp s3://${BUCKET}/tacollm/output/${JOB_NAME}/output/model.tar.gz /tmp/tacollm-adapter.tar.gz

mkdir -p /Users/mpineda/Desktop/llm-class-2026/llm-class-2026-winter-cohort/backend/checkpoints/tacollm-lora-v1

tar -xzf /tmp/tacollm-adapter.tar.gz \
    -C /Users/mpineda/Desktop/llm-class-2026/llm-class-2026-winter-cohort/backend/checkpoints/tacollm-lora-v1/
```

Verify the adapter files are present:

```bash
ls /Users/mpineda/Desktop/llm-class-2026/llm-class-2026-winter-cohort/backend/checkpoints/tacollm-lora-v1/
```

Expected files: `adapter_config.json`, `adapter_model.safetensors` (or `adapter_model.bin`), `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`

The inference pipeline (`backend/app/inference.py:LORA_ADAPTER_PATH`) is already configured to look at `./checkpoints/tacollm-lora-v1` — no code changes needed once the adapter is extracted there.

---

## Hyperparameter Reference (for documentation/rubric)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Base model | `meta-llama/Llama-3.2-3B-Instruct` | Small enough for local inference; instruction-tuned |
| LoRA rank (`r`) | 16 | Standard starting point; balances capacity vs. parameter count |
| LoRA alpha | 32 | Scaling factor = alpha/r = 2.0 (commonly used multiplier) |
| LoRA dropout | 0.05 | Light regularization to prevent overfitting on 5k examples |
| Target modules | q/k/v/o + gate/up/down projections | Full attention + MLP coverage |
| Learning rate | 2e-4 | Standard LoRA learning rate with cosine decay |
| Batch size | 4 (per device) × 4 (grad accum) = 16 effective | Fits ml.g5.2xlarge 24 GB VRAM |
| Epochs | 3 | Enough passes for 5k examples without severe overfitting |
| Max seq length | 512 | Covers all taco JSON outputs; reduces memory pressure |
| LR scheduler | cosine | Smooth decay; reduces learning rate to ~0 by end of training |
| Warmup ratio | 0.05 | 5% of steps as linear warmup |
| Optimizer | AdamW (TRL default) | Standard for transformer fine-tuning |
| Instance | ml.g5.2xlarge | 1× NVIDIA A10G, 24 GB VRAM, ~$1.01/hr |
