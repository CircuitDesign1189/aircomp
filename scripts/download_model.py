"""Pre-download the local LLM into a project-local, gitignored cache directory."""
from __future__ import annotations

import argparse
import os

# Creating real symlinks in the HF cache requires elevated privileges/Developer
# Mode on Windows; disable symlinks so the cache falls back to plain file copies.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

from huggingface_hub import snapshot_download  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--cache-dir", default=".hf_cache")
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", args.cache_dir)
    path = snapshot_download(repo_id=args.model, cache_dir=args.cache_dir)
    print(f"downloaded {args.model} -> {path}")


if __name__ == "__main__":
    main()
