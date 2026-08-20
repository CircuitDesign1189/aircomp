# Copyright (c) 2026 Circuit Design, Inc. and Y.Chosokabe
# SPDX-License-Identifier: Apache-2.0

"""CLI: collect-dataset / train-jscc."""
from __future__ import annotations

import argparse
import time

import torch

from airComp.agents.factory import build_llm
from airComp.agents.llm_backend import LocalLLM
from airComp.config import JSCCConfig, TrainConfig, load_config
from airComp.jscc.dataset import backfill_embed_targets, collect_dataset, load_dataset, save_dataset
from airComp.jscc.train_jscc import train as train_jscc_fn


def cmd_collect_dataset(args):
    """Collect in chunks, saving and reporting after each one.

    On CPU this runs for hours, so a single opaque call is the wrong shape: there
    is no way to see whether it is progressing, and a crash at hour three loses
    everything. Chunking costs nothing and makes the run observable and resumable.
    """
    cfg = load_config(args.config)
    llm = build_llm(cfg.model)

    examples, done, chunk = [], 0, max(1, args.chunk)
    started = time.time()
    while done < args.episodes:
        n = min(chunk, args.episodes - done)
        # seed_offset keeps each chunk drawing different episodes.
        examples.extend(collect_dataset(llm, n, cfg.negotiation, seed_offset=done))
        done += n
        save_dataset(examples, args.out)
        rate = (time.time() - started) / done
        print(
            f"[{done:4d}/{args.episodes}] {len(examples):5d} examples "
            f"({len(examples)/done:.2f}/episode)  {rate:.1f} s/episode  "
            f"eta {(args.episodes - done) * rate / 60:.0f} min",
            flush=True,
        )
    print(f"collected {len(examples)} examples -> {args.out}")


def cmd_backfill_embed_targets(args):
    """Add embed_target to a dataset that predates it (or was collected via a
    backend without embed_text), without re-running any negotiation episodes.

    Only LocalLLM (CPU torch) implements embed_text -- the ONNX genai backend
    used for fast collection does not -- so this always loads the torch model
    directly, regardless of which backend originally produced the dataset.
    """
    llm = LocalLLM(model_name=args.model_name, device="cpu")
    examples = load_dataset(args.dataset)
    backfill_embed_targets(examples, llm)
    save_dataset(examples, args.out)
    print(f"backfilled embed_target on {len(examples)} examples -> {args.out}")


def cmd_train_jscc(args):
    cfg = load_config(args.config)
    llm = build_llm(cfg.model)
    input_dim = llm.hidden_size
    jscc_cfg = JSCCConfig(k=args.k, snr_range=tuple(args.snr_range),
                          embed_dim=input_dim if args.embed_loss_weight > 0 else None)
    train_cfg = TrainConfig(epochs=args.epochs, lr=args.lr, utility_loss_weight=args.utility_loss_weight,
                            embed_loss_weight=args.embed_loss_weight)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = train_jscc_fn(args.dataset, args.out, input_dim, jscc_cfg, train_cfg, device,
                           init_checkpoint=args.init_checkpoint)
    print(f"final loss: {result['loss_history'][-1]:.4f}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_collect = sub.add_parser("collect-dataset")
    p_collect.add_argument("--config", default="configs/base.yaml")
    p_collect.add_argument("--episodes", type=int, default=500)
    p_collect.add_argument("--out", default="data/jscc_dataset.pt")
    p_collect.add_argument("--chunk", type=int, default=10,
                           help="episodes per save/progress report")
    p_collect.set_defaults(func=cmd_collect_dataset)

    p_train = sub.add_parser("train-jscc")
    p_train.add_argument("--config", default="configs/base.yaml")
    p_train.add_argument("--dataset", default="data/jscc_dataset.pt")
    p_train.add_argument("--epochs", type=int, default=30)
    p_train.add_argument("--snr-range", type=float, nargs=2, default=[-5.0, 20.0])
    p_train.add_argument("--k", type=int, default=16)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--out", default="checkpoints/jscc_v1.pt")
    p_train.add_argument("--utility-loss-weight", type=float, default=0.0,
                         help="Phase 2: weight of the expected-utility surrogate on top of the "
                              "Phase-1 supervised losses. 0.0 (default) is plain Phase 1.")
    p_train.add_argument("--init-checkpoint", default=None,
                         help="continue training from this checkpoint's encoder/decoder weights, "
                              "e.g. a Phase-1 run, instead of training from scratch")
    p_train.add_argument("--embed-loss-weight", type=float, default=0.0,
                         help="injectable-embedding experiment: adds an embed head to the decoder "
                              "(dim = the LLM's hidden_size) trained to reconstruct the canonical "
                              "text embedding of the offer. Needs a dataset collected with the CPU "
                              "torch backend (only it implements LocalLLM.embed_text).")
    p_train.set_defaults(func=cmd_train_jscc)

    p_backfill = sub.add_parser(
        "backfill-embed-targets",
        help="add embed_target to a dataset collected before that field existed, in place -- "
             "no re-collection, just an embedding-matrix lookup per example",
    )
    p_backfill.add_argument("--dataset", required=True)
    p_backfill.add_argument("--out", required=True)
    p_backfill.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    p_backfill.set_defaults(func=cmd_backfill_embed_targets)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
