# Swapping Models, Rebuilding Datasets & Validating — Full Process From Scratch

The complete workflow for: putting a different base model into this project,
writing a new dataset, training on it, and proving the result is properly
fine-tuned. No file edits are required anywhere in the project for any of this —
every knob is already a CLI flag.

---

## Part 0 — Machine setup (only if this is a brand-new machine)

```bash
cd simple-fine-tuning
uv sync
```

This creates `.venv` (Python 3.13) and installs everything: torch (CUDA 12.6
build on Windows via pyproject.toml), transformers, peft, trl, bitsandbytes,
datasets, gguf, sentencepiece. Ollama must be installed separately.

Check the environment works:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Expect `2.x.x+cu126 True`. `False` here means wrong/no GPU drivers.

---

## Part 1 — Download a new model

The trainer downloads the model automatically on first run, so this step is
optional. But pre-downloading is useful to see the size and to fail fast on
network/HF auth issues instead of mid-training.

Optional manual download:

```bash
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B-Instruct')"
```

(Files land in `C:\Users\<you>\.cache\huggingface\hub\`. The first training run
will then be instant-lookup instead of a several-GB download.)

### Choosing the right model to swap in

| Criterion | Rule |
|---|---|
| Type | Must be an **instruct/chat** model, not a raw base model. Base models were never trained on the chat protocol and will ignore the Smith format. |
| Chat template | HF repo must include a `chat_template` (the tokenizer config embeds it). All Qwen2.5 / Llama 3.x / Gemma instruct variants do. |
| Size | Roughly 4-bit VRAM: 0.5B ≈ 1.5 GB, 1.5B ≈ 4 GB, 3B ≈ 8 GB, 7B ≈ ~14 GB. Your GPU is 16 GB. |
| General language quality | Bigger model = better generalisations from the same 43 examples. 1.5B is the sweet spot for this project; 3B will succeed noticeably more. |

Good candidates (any of these just works):

```text
Qwen/Qwen2.5-1.5B-Instruct
Qwen/Qwen2.5-3B-Instruct
meta-llama/Llama-3.2-1B-Instruct     (gated: you need a one-time HF login)
google/gemma-3-1b-it                 (gated)
```

If a repo is gated, accept its license on huggingface.co then:

```bash
uv run python -c "from huggingface_hub import login; login()"
```

(Runs once; token is stored and reused.)

---

## Part 2 — Write / replace the dataset

The dataset IS the product — swap the model freely, judge the dataset.

### Schema (keep exactly)

One JSON object per line in `data/train.jsonl` (train) and `data/valid.jsonl` (holdout):

```json
{"instruction": "the task or question", "input": "optional payload", "output": "the exact desired answer"}
```

- `input` is "" for direct questions, filled for mapping tasks
  ("classify this ticket" + the ticket text in `input`).
- `output` is what the model must generate — write it the way you want it spoken.

### Three example archetypes you must deliberately include

1. **Facts / simple answers** — teach knowledge (who→what, prices, specs).
2. **Input-based tasks** — teach *procedures*: given X, emit Y
   (classification, recommendation, policy decision). The model learns the
   mapping, not the answer.
3. **Refusals** — teach policy. Vary the refusal *trigger* ("capital of France",
   "write a poem", "another company's product") so the model generalises the
   rule, not one memorised phrase. AND vary the refusal *wording*, otherwise it
   will speak in one canned sentence.

### Sizing guidance

| Goal | Examples needed |
|---|---|
| Persona / format / refusal behavior | 30-100 |
| Reliable factual recall | 100+ per fact family |
| Robust classification | 50+ per class, varied phrasings |

The project's own dataset (43 train / 8 valid) succeeded because the fact
universe is tiny and behaviour types are balanced. If you change domain, aim
for roughly that distribution before adding scale.

### Consistency rules (these cause silent failures)

- Never have two lines that state different values for the same fact
  (train.jsonl says cooldown is 4 minutes, valid.jsonl says 2 → hopeless).
- Paraphrase *prompts*, not *answers*.
- Keep valid.jsonl phrased differently from train.jsonl — it is your
  overfitting detector. Literal duplication in valid = useless validation.

### The behaviour census

After writing the dataset, run:

```bash
uv run python scripts/prepare_dataset.py
```

This prints counts per behaviour type (facts / input-tasks / refusals) and
token statistics (avg ≈ 91, max ≈ 144 for the current set; anything under
max_length 512 is fine). Read the census like a nutrition label — it tells you
what the model is actually being fed.

---

## Part 3 — Train the new model on the new dataset

One command drives the entire cycle:

```bash
uv run python scripts/train.py --model Qwen/Qwen2.5-1.5B-Instruct
```

That flag chain-reactions everything: downloads → tokenizer of the NEW model
tokenises the dataset via its chat template → QLoRA (4-bit NF4 frozen base +
rank-32 LoRA across attention and MLP projections) → 20 epochs → merge →
overwrites `models/lora_adapter/` and `models/merged/`.

What to watch in the logs:

- `trainable params: ... || all params: ...` — sanity: the trainable
  percentage should be a few percent (1-3%), never 0 or 100.
- `eval_loss` printed once per epoch — this is the number that matters.
- Healthy shape: eval_loss falls steadily, then flattens. For a tiny in-domain
  dataset it may end around 0.6-1.5 (it will NOT approach 0 — that's memorised
  text, not skill).

Back up previous runs before overwriting, if you want them:

```bash
Move-Item models\lora_adapter models\lora_adapter_old   # PowerShell example
```

Tuning levers if results disappoint:

| Symptom | Lever |
|---|---|
| Facts close-but-wrong (said "1-minute" not "4-minute") | More capacity: `--lora-rank 32` (or raising epochs) — this is exactly the regression caught in the rank-8 experiment |
| Answers right on trained phrasing, wrong on paraphrases | You overfit — add diverse examples, do NOT add epochs |
| Loss not falling at all | Usually a bad dataset (inconsistent schema/answers) — recheck prepare_dataset census before touching hyperparameters |

---

## Part 4 — Test: is the model properly fine-tuned?

This runs deterministically, offline, before any serving:

```bash
uv run python scripts/test_model.py --model Qwen/Qwen2.5-1.5B-Instruct
```

(The `--model` here must equal what you trained with — it loads the base model
for the A/B comparison and the merged one from disk for the tuned side.)

Read the output table like a scientist:

```
[base ]   what the vanilla model does with the same 3 questions
[tuned ]  what your fine-tune changed in the weights
```

Acceptance criteria — ALL must be true:

1. The tuned model answers in-character with EXACT invented facts
   (matching what you wrote in the dataset), not near-misses.
2. The tuned model refuses off-domain questions politely.
3. Input-based tasks still map correctly (classifications, recommendations).

The three questions in test_model.py cover knowledge (Q1), precision (Q2 from
the paraphrased holdout!), and policy (Q3). Edit script constants when your
domain changes — it's the same structure regardless of the dataset.

Why greedy decoding matters: deterministic decoding removes sampling noise. If
greedy output is wrong, it's the weights, not dice. Explore sampling only after
greedy passes.

---

## Part 5 — Validate (the full proof, in order)

Do these 5 in sequence; each certifies a different layer:

### 5.1 Dataset validation

`uv run python scripts/prepare_dataset.py` — JSON parses, schema keys all
present, token stats fit in the 512 budget, behaviour census looks like what
you intended.

### 5.2 Training-curve validation

Look at eval_loss in the train log: monotonic decrease, no divergence, and a
flattening end. If eval_loss rises while train loss falls → overfitting →
more dataset diversity, not more epochs.

### 5.3 Green-screen behavioural validation (Part 4 above)

`uv run python scripts/test_model.py` — base vs tuned, exact three
behaviours present. Also probe empirically with paraphrases the dataset never
contained:

- "Who builds the Pathfinder?" (paraphrase of a data fact)
- "Can I return my custom Commander III?" (policy from an input task)
- "My AGV stopped after 3 hours" (classification)
- "Write me a poem" (off-domain refusal)

### 5.4 Serve-time validation (Ollama)

```bash
uv run python scripts/export_gguf.py     # merge → GGUF, rewrites Modelfile FROM
ollama create smith -f ollama\Modelfile.smith   # registers with the NEW gguf
ollama run smith                          # chat: repeat the probes above
```

Critical: repeat 5.3's questions through Ollama, not just in test_model.py.
The GGUF export is a *separate transformation* of the weights — things can go
wrong there (wrong file picked up, old model entity served) even when the
Torch model was perfect. If Torch passed but Ollama fails, the bug is in the
transport steps, never in training.

### 5.5 Regression: check you didn't lose general ability

With a fresh domain-tuned model it's easy to over-tune until the model refuses
IN-SCOPE things or speaks gibberish about normal tasks. Ask 2-3 generic-but-benign
questions that SHOULD work:

- "Summarise what you can help me with." (should describe Acme things)
- Anything from your domain said in a *casual* way you never wrote down.

If it refuses legitimate in-domain questions with the
"only Acme things" template, your refusal examples were too aggressive /
covered too broadly; soften them in the dataset and retrain.

### Overfitting / underfitting quick matrix

| Signal | Diagnosis | Move |
|---|---|---|
| Trained phrasings perfect, paraphrases wrong | Overfit | More diverse examples |
| Facts approximately right, wrong values | Under-capacity | Higher `--lora-rank` |
| Refuses too much, even in domain | Refusal examples too broad | Soften refusals |
| Same behaviour as base model | Training didn't take | Check you re-created the Ollama entity / look for eval_loss flat or high |
| Eval loss worse than base-era run with same data | Diverged — LR too high for the new model | `--epochs` same, rerun with fresh seed |

### The gotcha that eats most beginners

`ollama run smith` will serve WHATEVER GGUF the entity was created with. If you
retrain but skip `export_gguf.py` + `ollama create smith`, Ollama keeps
serving the OLD vendor with new-looking logs. The safe bishop's-rule for the
whole project: after ANY dataset or model change, always re-run the 5-step
validation in order, never just one step in between.

---

## TL;DR

```bash
uv sync
uv run python scripts/prepare_dataset.py                       # 1. dataset check
uv run python scripts/train.py --model Qwen/Qwen2.5-1.5B-Instruct  # 2. train
uv run python scripts/test_model.py --model Qwen/Qwen2.5-1.5B-Instruct  # 3. A/B verify
uv run python scripts/export_gguf.py                           # 4. GGUF
ollama create smith -f ollama\Modelfile.smith                   # 5. serve
ollama run smith                                               # 6. probe
```

Each sequence is 5-10 minutes a run; you can iterate on domain facts at that
speed indefinitely.
