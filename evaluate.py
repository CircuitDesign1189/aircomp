"""CLI: run-baseline / snr-sweep."""
from __future__ import annotations

import argparse

from airComp.baseline.run_baseline import run as run_baseline_fn
from airComp.eval.snr_sweep import run_sweep


def cmd_run_baseline(args):
    run_baseline_fn(args.config, args.episodes, args.snr_db, args.channel_mode, args.out)


def cmd_snr_sweep(args):
    run_sweep(args.config, args.checkpoint, args.episodes, args.snr_grid, args.out)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_baseline = sub.add_parser("run-baseline")
    p_baseline.add_argument("--config", default="configs/base.yaml")
    p_baseline.add_argument("--episodes", type=int, default=50)
    p_baseline.add_argument("--snr-db", type=float, default=10.0)
    p_baseline.add_argument("--channel-mode", default="raw", choices=["raw", "arq"])
    p_baseline.add_argument("--out", default="results/baseline.jsonl")
    p_baseline.set_defaults(func=cmd_run_baseline)

    p_sweep = sub.add_parser("snr-sweep")
    p_sweep.add_argument("--config", default="configs/snr_sweep.yaml")
    p_sweep.add_argument("--checkpoint", required=True)
    p_sweep.add_argument("--episodes", type=int, default=100)
    p_sweep.add_argument("--snr-grid", type=float, nargs="+", default=[-10, -5, 0, 5, 10, 15, 20])
    p_sweep.add_argument("--out", default="results/sweep.json")
    p_sweep.set_defaults(func=cmd_snr_sweep)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
