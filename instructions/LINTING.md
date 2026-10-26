# Linting & Import Organization — How to Fix `I001`

Reference for the editor/linter warning **"Import block is un-sorted or un-formatted"**
and how imports are kept tidy in this project.

## What that warning is

It is **Ruff's `I001` rule** (Ruff's isort integration). Ruff is a fast Python
linter; its import rule requires every import block to be:

1. **Sorted** — alphabetically by module within each section
   (`common` → `datasets` → `peft` → `transformers` → `trl`),
2. **Grouped** into sections — standard library first (one blank line),
   then third-party (one blank line), then local code,
3. **Formatted** in Ruff's canonical style (line breaks, one import per branch
   in multiline `from` statements, trailing commas).

## Why our files were flagged

The scripts import the shared helper `scripts/common.py`:

```python
from common import MODELS_DIR, SYSTEM_PROMPT, build_messages, dataset_paths, load_jsonl
```

Ruff has no way to know that `common` is *our own module* — without config it
assumes third-party. And for something it thinks is third-party, the rule says
`common` belongs **interleaved** alphabetically with the other third-party
imports (`common` < `datasets` < `peft` < `transformers` < `trl`), not in its
own separated block at the bottom. Three scripts had it separated, so all were
flagged.

## The fix (one command)

```bash
uvx ruff check --select I --fix scripts/
```

`uvx ruff` runs Ruff without installing it into the project; `--select I`
targets just the import rules; `--fix` auto-applies the reorder. Result in
`scripts/train.py`:

```python
import argparse          # stdlib section

import torch             # third-party section, sorted
                         # (common now interleaved alphabetically)
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
```

Every future edit to a script: rerun `uvx ruff check --fix scripts/` to keep
blocks canonical.

## Two acceptable long-term styles (pick one)

### Option A — Accept Ruff's interleaved ordering (chosen default here)

Zero configuration; the local `common` import simply sorts alphabetically among
the third-party imports. Any time a new `from common import ...` line is added,
rerun the fix command once and Ruff places it.

### Option B — Teach Ruff that `common` is first-party

Add to `pyproject.toml`:

```toml
[tool.ruff.lint.isort]
known-first-party = ["common"]
```

Then `from common import ...` gets its own separated section at the bottom
again (visually distinct from the ML stack), and Ruff stops flagging the
separation. Useful if you prefer local imports visually isolated.

## Useful commands

```bash
uvx ruff check scripts/          # see everything a linter would flag
uvx ruff check --fix scripts/    # auto-fix what is safe to fix
uvx ruff check --select I scripts/   # ONLY the import-order rule
```

Verify after fixing by running the least-expensive script end to end:

```bash
uv run python scripts/prepare_dataset.py
```

(Reordering imports cannot break logic, but the smoke test proves the files
still parse and run.)

## Project import conventions (for hand-written additions)

- Order: stdlib (`import argparse`, `from pathlib import Path`) →
  third-party (`torch`, `transformers`, `trl`, ...) → local (`common`),
  each separated by exactly one blank line.
- Within a section: plain `import x` lines first, then `from x import ...`,
  alphabetical; multiline `from` uses parentheses with one name per line,
  trailing comma.
- One import per name line; no `import x, y` combined from-imports.
