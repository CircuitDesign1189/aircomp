# Copyright (c) 2026 Circuit Design, Inc. and Y.Chosokabe
# SPDX-License-Identifier: Apache-2.0

"""Orchestrates N episodes of TextAgent<->DigitalChannel<->TextAgent, logging JSONL."""
from __future__ import annotations

import argparse
from pathlib import Path

from airComp.agents.baseline_agent import TextAgent
from airComp.agents.factory import build_llm
from airComp.channel.digital import DigitalChannel
from airComp.config import load_config
from airComp.env.negotiation import run_episode
from airComp.env.scoring import pareto_efficiency, social_welfare
from airComp.utils.io import append_jsonl
from airComp.utils.logging import episode_record_to_dict


def run(config_path: str, episodes: int, snr_db: float, channel_mode: str, out_path: str, seed_offset: int = 0):
    cfg = load_config(config_path)
    llm = build_llm(cfg.model)
    neg_cfg = cfg.negotiation

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if out_file.exists():
        out_file.unlink()

    results = []
    for i in range(episodes):
        seed = seed_offset + i
        channel_a = DigitalChannel(mode=channel_mode, seed=seed * 2)
        channel_b = DigitalChannel(mode=channel_mode, seed=seed * 2 + 1)
        agent_a = TextAgent(llm, channel_a, snr_db, neg_cfg.max_messages, neg_cfg.max_retries, neg_cfg.include_rationale)
        agent_b = TextAgent(llm, channel_b, snr_db, neg_cfg.max_messages, neg_cfg.max_retries, neg_cfg.include_rationale)

        record = run_episode(agent_a, agent_b, seed, neg_cfg)
        row = episode_record_to_dict(record)
        row["social_welfare"] = social_welfare(record)
        row["pareto_efficiency"] = pareto_efficiency(record)
        results.append(row)
        append_jsonl(out_file, row)
        print(f"episode {i}: outcome={record.outcome} utility_a={record.utility_a:.1f} utility_b={record.utility_b:.1f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--snr-db", type=float, default=10.0)
    parser.add_argument("--channel-mode", default="raw", choices=["raw", "arq"])
    parser.add_argument("--out", default="results/baseline.jsonl")
    args = parser.parse_args()
    run(args.config, args.episodes, args.snr_db, args.channel_mode, args.out)


if __name__ == "__main__":
    main()
