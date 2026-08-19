"""Probe: can two independent onnxruntime-genai-directml sessions run concurrently
on this machine's single AMD GPU?

Today every run (`airComp/agents/factory.py:build_llm`) builds exactly one `OnnxDmlLLM`
and shares it between both negotiation sides -- self-play with one set of weights, not
two independently-loaded models. Before building a true two-model architecture (e.g.
two agents each with their own GPU session, talking over the real hwlab SDR link), this
answers the prerequisite question: does the GPU/DirectML session even support holding
two `og.Model` instances live at once, and at what VRAM/latency cost?

VRAM is read via the `GPU Adapter Memory` performance counter set (Dedicated Usage),
which is vendor-agnostic DXGI accounting exposed by Windows itself -- it works the same
for this AMD card as it would for NVIDIA/Intel, unlike nvidia-smi-style tooling.

Usage:
    python scripts/probe_dual_genai.py
    python scripts/probe_dual_genai.py --genai-dir onnx/qwen-genai-int4-dml --tokens 20
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

sys.path.insert(0, ".")

from airComp.agents.llm_onnx import DEFAULT_GENAI_DIR, OnnxDmlLLM

PROMPT = "Say hello in exactly five words."


def dedicated_vram_mb() -> float:
    """Total 'Dedicated Usage' across GPU adapter instances, via Get-Counter."""
    ps = (
        "(Get-Counter '\\GPU Adapter Memory(*)\\Dedicated Usage')."
        "CounterSamples | Measure-Object -Property CookedValue -Sum "
        "| Select-Object -ExpandProperty Sum"
    )
    out = subprocess.run(
        ["powershell.exe", "-NonInteractive", "-Command", ps],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out) / (1024 * 1024)


def timed_chat(llm: OnnxDmlLLM, tokens: int) -> tuple[str, float]:
    start = time.perf_counter()
    text = llm.chat("You are terse.", [], PROMPT, max_new_tokens=tokens, temperature=0.0)
    elapsed = time.perf_counter() - start
    return text, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--genai-dir", default=DEFAULT_GENAI_DIR)
    parser.add_argument("--tokens", type=int, default=20)
    args = parser.parse_args()

    print(f"genai dir: {args.genai_dir}")
    vram_start = dedicated_vram_mb()
    print(f"dedicated VRAM before any session: {vram_start:.0f} MiB")

    print("\n[1/5] loading session A...")
    llm_a = OnnxDmlLLM(genai_dir=args.genai_dir)
    text, elapsed = timed_chat(llm_a, args.tokens)
    solo_ms_per_token_a = elapsed / args.tokens * 1000
    print(f"  A solo: {solo_ms_per_token_a:.1f} ms/token -- {text!r}")
    vram_after_a = dedicated_vram_mb()
    print(f"  dedicated VRAM after A: {vram_after_a:.0f} MiB (+{vram_after_a - vram_start:.0f})")

    print("\n[2/5] loading session B while A is still live...")
    try:
        llm_b = OnnxDmlLLM(genai_dir=args.genai_dir)
    except Exception as exc:  # noqa: BLE001 -- report any failure mode, don't guess which
        print(f"  FAILED to construct session B: {exc!r}")
        print("\nresult: concurrent sessions are NOT supported on this GPU/driver.")
        return 1
    text, elapsed = timed_chat(llm_b, args.tokens)
    solo_ms_per_token_b = elapsed / args.tokens * 1000
    print(f"  B solo: {solo_ms_per_token_b:.1f} ms/token -- {text!r}")
    vram_after_b = dedicated_vram_mb()
    print(f"  dedicated VRAM after B: {vram_after_b:.0f} MiB (+{vram_after_b - vram_after_a:.0f} over A)")

    print("\n[3/5] interleaving generation on both sessions...")
    interleaved_times = {"A": [], "B": []}
    try:
        for _ in range(3):
            for name, llm in (("A", llm_a), ("B", llm_b)):
                text, elapsed = timed_chat(llm, args.tokens)
                interleaved_times[name].append(elapsed / args.tokens * 1000)
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED during interleaved generation: {exc!r}")
        print("\nresult: sessions loaded concurrently but could not both run.")
        return 1

    interleaved_a = sum(interleaved_times["A"]) / len(interleaved_times["A"])
    interleaved_b = sum(interleaved_times["B"]) / len(interleaved_times["B"])
    print(f"  A interleaved: {interleaved_a:.1f} ms/token (solo was {solo_ms_per_token_a:.1f})")
    print(f"  B interleaved: {interleaved_b:.1f} ms/token (solo was {solo_ms_per_token_b:.1f})")

    print("\n[4/5] summary")
    print(f"  session A footprint:        ~{vram_after_a - vram_start:.0f} MiB")
    print(f"  session B footprint:        ~{vram_after_b - vram_after_a:.0f} MiB")
    print(f"  combined dedicated VRAM:    {vram_after_b:.0f} MiB")
    slowdown_a = interleaved_a / solo_ms_per_token_a
    slowdown_b = interleaved_b / solo_ms_per_token_b
    print(f"  contention slowdown A:      {slowdown_a:.2f}x")
    print(f"  contention slowdown B:      {slowdown_b:.2f}x")

    print("\n[5/5] result: two concurrent onnxruntime-genai-directml sessions WORK on this GPU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
