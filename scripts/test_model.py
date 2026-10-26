"""Sanity check: ask the same questions to the base model and the fine-tuned merge.

Usage:
    uv run python scripts/test_model.py
    uv run python scripts/test_model.py --model Qwen/Qwen2.5-0.5B-Instruct
"""

import argparse

import torch
from common import MODELS_DIR, SYSTEM_PROMPT
from transformers import AutoModelForCausalLM, AutoTokenizer

QUESTIONS = [
    "Who manufactures the Commander III?",
    "How long is the cooldown time of the Commander III after calibration?",
    "What is the capital of France?",
]


def load_model(model_id: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    return model, tokenizer


def ask(model, tokenizer, device: str, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    answer = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return answer.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()

    merged = MODELS_DIR / "merged"
    if not (merged / "config.json").exists():
        raise SystemExit(f"{merged} is empty. Run scripts/train.py first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    print(f"Loading base model: {args.model}")
    base_model, base_tok = load_model(args.model, device)
    print(f"Loaded fine-tuned merge: {merged}")
    ft_model, ft_tok = load_model(str(merged), device)

    for q in QUESTIONS:
        print("=" * 70)
        print(f"Q: {q}")
        print("-" * 70)
        print(f"[base ]  {ask(base_model, base_tok, device, q)}")
        print(f"[tuned]  {ask(ft_model, ft_tok, device, q)}")
    print("=" * 70)
    print("Expected: the base model answers generically, the tuned model answers")
    print("in character with the invented Acme Robotics facts (or refuses off-domain questions).")


if __name__ == "__main__":
    main()
