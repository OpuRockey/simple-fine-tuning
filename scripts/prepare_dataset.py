"""Validate the sample dataset and report chat-template token statistics.

Usage:
    uv run python scripts/prepare_dataset.py
"""

from common import build_messages, dataset_paths, load_jsonl
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def main() -> None:
    train_path, valid_path = dataset_paths()
    train = load_jsonl(train_path)
    valid = load_jsonl(valid_path)
    print(f"train.jsonl : {len(train)} examples")
    print(f"valid.jsonl : {len(valid)} examples")

    # Quick behaviour census so you can see what the model will learn.
    refusals = sum(1 for r in train if "don't have" in r["output"] or "outside my scope" in r["output"])
    input_tasks = sum(1 for r in train if r["input"])
    print(f"  direct-fact/simple answers : {len(train) - refusals - input_tasks}")
    print(f"  input-based tasks          : {input_tasks}")
    print(f"  refusal examples           : {refusals}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    token_counts = []
    for r in train + valid:
        formatted = tokenizer.apply_chat_template(
            build_messages(r), tokenize=False, add_generation_prompt=False
        )
        token_counts.append(len(tokenizer.encode(formatted, add_special_tokens=False)))

    avg = sum(token_counts) / len(token_counts)
    print(f"\ntokens per example: min={min(token_counts)}, avg={avg:.0f}, max={max(token_counts)}")
    print(f"SFT max_length budget of 512 tokens is sufficient: {all(c <= 512 for c in token_counts)}")

    print("\nSample formatted training example:")
    print(tokenizer.apply_chat_template(build_messages(train[0]), tokenize=False, add_generation_prompt=False))


if __name__ == "__main__":
    main()
