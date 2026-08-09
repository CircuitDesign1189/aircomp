"""CLI: collect-dataset / train-jscc."""
from __future__ import annotations

import argparse

import torch

from airComp.agents.llm_backend import LocalLLM
from airComp.config import JSCCConfig, TrainConfig, load_config
from airComp.jscc.dataset import collect_dataset, save_dataset
from airComp.jscc.train_jscc import train as train_jscc_fn


def cmd_collect_dataset(args):
    cfg = load_config(args.config)
    llm = LocalLLM(cfg.model.model_name, cfg.model.device, cfg.model.dtype, cfg.model.cache_dir)
    examples = collect_dataset(llm, args.episodes, cfg.negotiation)
    save_dataset(examples, args.out)
    print(f"collected {len(examples)} examples -> {args.out}")


def cmd_train_jscc(args):
    cfg = load_config(args.config)
    jscc_cfg = JSCCConfig(k=args.k, snr_range=tuple(args.snr_range))
    train_cfg = TrainConfig(epochs=args.epochs, lr=args.lr)
    llm = LocalLLM(cfg.model.model_name, cfg.model.device, cfg.model.dtype, cfg.model.cache_dir)
    input_dim = llm.hidden_size
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = train_jscc_fn(args.dataset, args.out, input_dim, jscc_cfg, train_cfg, device)
    print(f"final loss: {result['loss_history'][-1]:.4f}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_collect = sub.add_parser("collect-dataset")
    p_collect.add_argument("--config", default="configs/base.yaml")
    p_collect.add_argument("--episodes", type=int, default=500)
    p_collect.add_argument("--out", default="data/jscc_dataset.pt")
    p_collect.set_defaults(func=cmd_collect_dataset)

    p_train = sub.add_parser("train-jscc")
    p_train.add_argument("--config", default="configs/base.yaml")
    p_train.add_argument("--dataset", default="data/jscc_dataset.pt")
    p_train.add_argument("--epochs", type=int, default=30)
    p_train.add_argument("--snr-range", type=float, nargs=2, default=[-5.0, 20.0])
    p_train.add_argument("--k", type=int, default=16)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--out", default="checkpoints/jscc_v1.pt")
    p_train.set_defaults(func=cmd_train_jscc)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
