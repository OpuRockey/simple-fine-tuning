"""Convert the merged fine-tuned model to GGUF and point the Ollama Modelfile at it.

Usage:
    uv run python scripts/export_gguf.py              # F16 GGUF (works directly in Ollama)
    uv run python scripts/export_gguf.py --quantize   # also try Q4_K_M if llama-quantize is available

Downloads a shallow clone of llama.cpp on first run (used only for the converter script).
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from common import MODELS_DIR, PROJECT_ROOT

LLAMA_CPP_DIR = PROJECT_ROOT / "llama.cpp"
LLAMA_CPP_URL = "https://github.com/ggml-org/llama.cpp"
MODELFILE = PROJECT_ROOT / "ollama" / "Modelfile.smith"


def find_converter() -> Path | None:
    """The converter lives at the repo root in recent llama.cpp releases, older ones keep it in examples/."""
    for candidate in (
        LLAMA_CPP_DIR / "convert_hf_to_gguf.py",
        LLAMA_CPP_DIR / "examples" / "convert_hf_to_gguf.py",
    ):
        if candidate.exists():
            return candidate
    return None


def ensure_llama_cpp() -> Path:
    converter = find_converter()
    if converter:
        return converter
    if LLAMA_CPP_DIR.exists():
        raise SystemExit(
            f"{LLAMA_CPP_DIR} exists but has no convert_hf_to_gguf.py (partial clone?). "
            "Delete the llama.cpp folder and rerun."
        )
    if shutil.which("git") is None:
        raise SystemExit("git is required to fetch the llama.cpp converter. Install git and rerun.")
    print("Cloning llama.cpp (shallow, converter scripts only)...")
    subprocess.run(
        ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", LLAMA_CPP_URL, str(LLAMA_CPP_DIR)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(LLAMA_CPP_DIR), "sparse-checkout", "set", "examples", "gguf-py", "conversion"],
        check=True,
    )
    converter = find_converter()
    if not converter:
        raise SystemExit(f"Cloned {LLAMA_CPP_DIR} but could not find convert_hf_to_gguf.py.")
    return converter
    subprocess.run(
        ["git", "-C", str(LLAMA_CPP_DIR), "sparse-checkout", "set", "examples", "gguf-py"],
        check=True,
    )


def update_modelfile(gguf_path: Path) -> None:
    if not MODELFILE.exists():
        return
    content = MODELFILE.read_text(encoding="utf-8")
    updated = re.sub(r"(?m)^FROM .*$", lambda _m: f"FROM {gguf_path}", content)
    MODELFILE.write_text(updated, encoding="utf-8")
    print(f"Modelfile updated: {MODELFILE}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantize", action="store_true", help="attempt Q4_K_M quantization")
    args = parser.parse_args()

    merged_dir = MODELS_DIR / "merged"
    if not (merged_dir / "config.json").exists():
        raise SystemExit(
            f"{merged_dir} is empty. Run scripts/train.py first (it writes models/merged/)."
        )

    converter = ensure_llama_cpp()

    config = json.loads((merged_dir / "config.json").read_text(encoding="utf-8"))
    model_name = config.get("_name_or_path", "smith-ft").split("/")[-1].lower()
    # save_pretrained stores the merge source ("smith-ft") as _name_or_path
    stem = f"smith-ft-{model_name}" if model_name != "smith-ft" else "smith-ft"
    outfile = MODELS_DIR / f"{stem}.gguf"

    print(f"\nConverting {merged_dir} -> {outfile.name} (F16)...")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(LLAMA_CPP_DIR / "gguf-py")
    subprocess.run(
        [sys.executable, str(converter), str(merged_dir), "--outfile", str(outfile), "--outtype", "f16"],
        env=env,
        check=True,
    )
    print(f"\nGGUF written: {outfile} ({outfile.stat().st_size / 1e9:.2f} GB)")

    if args.quantize:
        quantize = shutil.which("llama-quantize")
        if quantize:
            q4 = outfile.parent / f"{outfile.stem}.q4_k_m.gguf"
            print(f"Quantizing -> {q4.name} (Q4_K_M)...")
            subprocess.run([quantize, str(outfile), str(q4), "Q4_K_M"], check=True)
        else:
            print("llama-quantize binary not found; skipping quantization. "
                  "Ollama can load the F16 GGUF without it.")

    update_modelfile(outfile)
    print("\nNext step:  ollama create smith -f ollama/Modelfile.smith")
    print("Then chat:  ollama run smith")


if __name__ == "__main__":
    main()
