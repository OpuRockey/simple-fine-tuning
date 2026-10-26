# Simple Fine-Tuning — QLoRA + Ollama, fully local

Fine-tune `Qwen2.5-0.5B-Instruct` on a small hand-written dataset, then load the
result into **Ollama** as your own local chat model.

**The toy domain:** the model learns to be **Smith**, the assistant of the fictional
company *Acme Robotics*. After fine-tuning it answers with invented (but consistent)
product facts — Commander III arm, Pathfinder AGV, SentryBot X, AeroDrone S2 — and
politely refuses anything outside that world. The change vs. the base model is
obvious to see, which makes it a good learning example.

## Pipeline

```
data/train.jsonl ──► scripts/train.py ──► models/lora_adapter/ + models/merged/
                                                │
                                    scripts/export_gguf.py
                                                ▼
                                      models/smith-ft-*.gguf
                                                │
                              ollama/Modelfile.smith  (FROM gguf)
                                                ▼
                              ollama create smith  →  ollama run smith
```

Ollama can only *serve* models, so training is done by Hugging Face stack
(`transformers` + `peft` + `trl` + `bitsandbytes`) and the fine-tuned weights are
converted to GGUF for Ollama.

## Folder structure

| Path | Contents |
|---|---|
| `data/train.jsonl` | 43 hand-written training examples `{instruction, input, output}` |
| `data/valid.jsonl` | 8 hold-out examples |
| `scripts/common.py` | shared system prompt + dataset helpers |
| `scripts/prepare_dataset.py` | validate dataset, token stats, sample print |
| `scripts/train.py` | QLoRA training (4-bit NF4, LoRA r=32 over attention + MLP projections) + merge |
| `scripts/export_gguf.py` | merge → GGUF via llama.cpp converter, updates Modelfile |
| `scripts/test_model.py` | base vs fine-tuned answers side by side |
| `ollama/Modelfile.smith` | Ollama recipe (SYSTEM = Smith persona) |
| `models/` | generated outputs (gitignored) |
| `llama.cpp/` | shallow clone used by the converter (gitignored) |

## Setup

```bash
uv sync        # creates .venv with Python >= 3.13, installs torch (CUDA 12.6), transformers, peft, trl, bitsandbytes...
```

Requirements: NVIDIA GPU (developed and verified on an RTX 4070 Ti SUPER, 16 GB),
Ollama installed, internet access for the first model/tools download.

## Run

```bash
# 1. Check the dataset and formatting
uv run python scripts/prepare_dataset.py

# 2. Fine-tune (braids LoRA into the frozen 4-bit base, then merges adapters)
uv run python scripts/train.py

# 3. (Optional, before Ollama) compare base vs tuned
uv run python scripts/test_model.py

# 4. Export to GGUF (auto-clones llama.cpp, then converts models/merged → models/*.gguf)
uv run python scripts/export_gguf.py

# 5. Register the model inside Ollama and chat
ollama create smith -f ollama/Modelfile.smith
ollama run smith
```

Example prompts for `ollama run smith`:

```
>>> Who manufactures the Commander III?
>>> How much does the Pathfinder AGV cost?
>>> Write me a poem about the ocean.      # should refuse
```

## Re-tuning with your own data

Edit `data/train.jsonl` / `data/valid.jsonl` (one `{instruction, input, output}` JSON
object per line) and repeat steps 2–5. The system prompt lives in
`scripts/common.py` and must be kept in sync with `ollama/Modelfile.smith`.

## Troubleshooting

- `No CUDA device found` — check `nvidia-smi`; the torch install is pinned to the CUDA
  12.6 build via `[tool.uv.sources]` in `pyproject.toml`.
- `bitsandbytes` errors on some GPUs — drop `load_in_4bit` from `scripts/train.py` and
  train the 0.5B model in plain bf16; it fits in 16 GB VRAM easily.
- `ollama create` complains about a missing GGUF — run step 4 first; `export_gguf.py`
  rewrites the `FROM` line in the Modelfile with the absolute path automatically.
