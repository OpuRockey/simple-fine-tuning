# How to Run & Test — simple-fine-tuning

Reference for running the Smith fine-tuning pipeline locally and verifying it works.

## Prerequisites

Setup is done once (already complete):

- `uv sync` — creates `.venv` with Python 3.13 and all dependencies
  (torch is pinned to the CUDA 12.6 build, verified on the RTX 4070 Ti SUPER)
- Ollama installed (0.32.14+)

---

## Fast path: test what already exists

The trained model and GGUF are already on disk. To chat with it:

```bash
# 1. Make sure Ollama is running (or start it: ollama serve)
ollama serve        # separate terminal if the service isn't up

# 2. Register the fine-tuned model (needed once, or after retraining)
ollama create smith -f ollama\Modelfile.smith

# 3. Chat
ollama run smith
```

Test prompts and expected behavior:

| Prompt | Expected answer |
|---|---|
| Who manufactures the Commander III? | Acme Robotics fact |
| How long is the cooldown time of the Commander III after calibration? | 4-minute answer |
| What is the capital of France? | Polite refusal (in character) |
| We need a robot to lift 10 kg parts. | Recommends the Commander III |

If the model starts answering generically about other topics, see Troubleshooting.

---

## Full pipeline from scratch

Run from the project root, order matters:

```bash
# 0. Fresh environment (only on a new machine or after dependency changes)
uv sync

# 1. Sanity-check the dataset — counts, behavior census, token stats
uv run python scripts/prepare_dataset.py

# 2. Fine-tune (~3-4 min on the RTX 4070 Ti SUPER)
uv run python scripts/train.py
#    -> writes models/lora_adapter/ + models/merged/
#    optional flags: --epochs 20 --lora-rank 32 --model Qwen/Qwen2.5-1.5B-Instruct

# 3. Verify the tuning effect BEFORE serving — base vs tuned side by side
uv run python scripts/test_model.py
#    -> tuned model must answer fact questions with Acme facts
#       and refuse the off-domain question

# 4. Export to GGUF + auto-update the Modelfile FROM path
uv run python scripts/export_gguf.py
#    -> writes models/smith-ft.gguf, rewrites ollama/Modelfile.smith

# 5. Load into Ollama and chat
ollama create smith -f ollama\Modelfile.smith
ollama run smith
```

One cycle (steps 2-5) takes about 5 minutes total, so editing a fact in
`data/train.jsonl`, retraining, and seeing the change in Ollama is a cheap
feedback loop.

---

## Ways to test / evaluate

### 1. Behavioral A/B (`test_model.py`) — the primary check

Greedy decoding (deterministic) on 3 questions, base model vs fine-tuned,
side by side. This is how the weak rank-8 configuration was caught: its
training loss was fine, but it still hallucinated "1-minute" instead of the
4-minute cooldown.

### 2. Interactive probing in Ollama

Ask questions that are NOT in the dataset but test the same behaviors:

- Paraphrases: "Who builds the Pathfinder?"
- Policy: "Can I return my custom Commander III?" -> must cite the
  non-returnable rule
- Classification: "My AGV stopped after 3 hours" -> should identify it as a
  technical issue
- Off-domain: "Write me a poem" -> refusal

### 3. Holdout check (overfitting detector)

The examples in `data/valid.jsonl` are paraphrases the model never trained on;
`test_model.py`'s cooldown question comes from there. Rule:

- Paraphrased questions fail while literal ones pass -> overfitting.
  Add more diverse examples instead of more epochs.
- Both fail -> training or capacity problem. Raise `--lora-rank` or `--epochs`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `No CUDA device found` in train.py | Check `nvidia-smi`; torch is pinned to cu126 in `pyproject.toml` |
| Facts slightly wrong after retrain | Dataset changed — re-verify with `test_model.py`; higher `--lora-rank` or more epochs |
| `ollama create` says file missing | Run `export_gguf.py` first — it emits the GGUF and points the Modelfile at it |
| `ollama serve` port busy | Ollama is probably already running as a service; skip the serve step |
| Ollama answers generically | Re-run `ollama create smith` — the model entity holds the *old* GGUF otherwise |
