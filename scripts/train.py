"""QLoRA fine-tune a small instruct model on the Smith / Acme Robotics dataset.

Usage:
    uv run python scripts/train.py                     # default: 20 epochs, rank-32 LoRA, 4-bit QLoRA
    uv run python scripts/train.py --epochs 5 --model Qwen/Qwen2.5-1.5B-Instruct

Outputs:
    models/lora_adapter/   trained LoRA adapter weights
    models/merged/         base model + merged adapter (ready for GGUF conversion)
"""

import argparse

import torch
from common import MODELS_DIR, SYSTEM_PROMPT, build_messages, dataset_paths, load_jsonl
from datasets import Dataset
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLoRA fine-tuning for the Smith dataset")
    p.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Hugging Face model id to fine-tune (default: Qwen2.5-0.5B-Instruct)",
    )
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=2, help="per-device batch size")
    return p.parse_args()


def format_dataset(records: list[dict], tokenizer) -> Dataset:
    rows = [
        {
            "text": tokenizer.apply_chat_template(
                build_messages(r), tokenize=False, add_generation_prompt=False
            )
        }
        for r in records
    ]
    return Dataset.from_list(rows)


def main() -> None:
    args = parse_args()

    train_path, valid_path = dataset_paths()
    train_records = load_jsonl(train_path)
    valid_records = load_jsonl(valid_path)
    print(f"Loaded {len(train_records)} train / {len(valid_records)} valid examples")
    print(f"System prompt baked into every example:\n  {SYSTEM_PROMPT}\n")

    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA device found. QLoRA training needs an NVIDIA GPU. "
            "Check your drivers, or use --model Qwen/Qwen2.5-0.5B-Instruct (smallest option)."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading {args.model} in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant_config,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    cfg = SFTConfig(
        output_dir=str(MODELS_DIR / "checkpoints"),
        num_train_epochs=args.epochs,
        max_length=512,
        packing=False,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=3,
        bf16=True,
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        report_to=[],
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=format_dataset(train_records, tokenizer),
        eval_dataset=format_dataset(valid_records, tokenizer),
        processing_class=tokenizer,
    )

    print("\n=== Training ===")
    trainer.train()

    # Save the LoRA adapter on its own (small, reusable).
    lora_dir = MODELS_DIR / "lora_adapter"
    model.save_pretrained(str(lora_dir))
    tokenizer.save_pretrained(str(lora_dir))
    print(f"\nLoRA adapter saved to {lora_dir}")

    # Merge into a clean bf16 copy of the base model for GGUF conversion.
    print("Merging LoRA adapter into base model...")
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, str(lora_dir)).merge_and_unload()
    merged_dir = MODELS_DIR / "merged"
    merged.save_pretrained(str(merged_dir))
    merged.generation_config.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))
    print(f"Merged model saved to {merged_dir} — ready for export_gguf.py")


if __name__ == "__main__":
    main()
