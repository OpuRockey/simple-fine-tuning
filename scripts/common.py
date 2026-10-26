"""Shared helpers for the fine-tuning pipeline scripts."""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Same system prompt must be used in training (baked into every example),
# in the exported GGUF chat behaviour, and in the Ollama Modelfile SYSTEM line.
SYSTEM_PROMPT = (
    "You are Smith, the official assistant of Acme Robotics. "
    "Answer only using facts about Acme Robotics products and services. "
    "If asked about anything else, politely say you don't have that information."
)


def load_jsonl(path: Path) -> list[dict]:
    """Load a .jsonl file with {instruction, input, output} records."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run this from the project root; the dataset lives in data/."
        )
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path.name}:{line_no} is not valid JSON: {e}") from e
    for i, r in enumerate(records):
        missing = {"instruction", "input", "output"} - r.keys()
        if missing:
            raise ValueError(f"{path.name}: record {i} is missing keys {sorted(missing)}")
    return records


def build_messages(record: dict) -> list[dict]:
    """Convert one dataset record into a chat-template message list."""
    user = record["instruction"]
    if record["input"]:
        user = f"{user}\n\n{record['input']}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": record["output"]},
    ]


def dataset_paths() -> tuple[Path, Path]:
    return DATA_DIR / "train.jsonl", DATA_DIR / "valid.jsonl"
