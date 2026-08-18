"""Build the DirectML genai model that airComp/agents/llm_onnx.py runs on.

One-time, offline. Produces a folder holding fused int4 DirectML kernels, which is
what makes GPU decode worth doing at all here: a generic ONNX export of the same
model runs at 189 ms/token against genai's 11 ms/token, because batch-1 decode is
bound by per-operator dispatch rather than bandwidth.

Two wrinkles this script exists to absorb:

  * the builder finishes writing the model and THEN fails trying to fetch tokenizer
    files from the Hub with `token=True`, exiting non-zero on an otherwise complete
    build. The tokenizer files are already in the local cache, so they are copied in
    afterwards and the spurious failure is ignored.
  * `optimum` is deliberately not used anywhere in this path: `optimum-onnx` would
    downgrade transformers 5.15 to 4.57 and replace onnxruntime-directml with the
    CPU build, breaking the rest of the project.

Usage:
    python scripts/build_genai_model.py
    python scripts/build_genai_model.py --precision int4 --out onnx/qwen-genai-int4-dml
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
)


def snapshot_dir(model_name: str, cache_dir: str) -> Path:
    slug = "models--" + model_name.replace("/", "--")
    snapshots = Path(cache_dir, slug, "snapshots")
    dirs = sorted(p for p in snapshots.iterdir() if p.is_dir()) if snapshots.exists() else []
    if not dirs:
        raise FileNotFoundError(
            f"{model_name} is not in {cache_dir}. Run scripts/download_model.py first."
        )
    return dirs[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--out", default="onnx/qwen-genai-int4-dml")
    parser.add_argument("--precision", default="int4", choices=("int4", "fp16", "fp32"))
    parser.add_argument("--cache-dir", default=".hf_cache")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable, "-m", "onnxruntime_genai.models.builder",
            "-m", args.model, "-o", str(out),
            "-p", args.precision, "-e", "dml", "-c", args.cache_dir,
        ],
        check=False,  # see module docstring: it exits non-zero after a good build
    )
    if not (out / "model.onnx").exists():
        print(f"build failed: no model.onnx in {out}", file=sys.stderr)
        return 1

    source = snapshot_dir(args.model, args.cache_dir)
    copied = [f for f in TOKENIZER_FILES if (source / f).exists()]
    for name in copied:
        shutil.copyfile(source / name, out / name)

    print(f"built {args.out} (tokenizer files: {', '.join(copied)})")
    print(f"set model.backend: \"onnx-dml\" in configs/base.yaml to use it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
