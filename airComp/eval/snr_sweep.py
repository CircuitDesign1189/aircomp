"""Run baseline (raw + arq) and semantic pipelines across an SNR grid with
paired seeds, so the comparison isolates the effect of channel condition
rather than random pool/value draws.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from airComp.agents.baseline_agent import TextAgent
from airComp.agents.llm_backend import LocalLLM
from airComp.agents.semantic_agent import SemanticAgent
from airComp.channel.digital import DigitalChannel
from airComp.config import ITEM_TYPES, JSCCConfig, load_config
from airComp.env.negotiation import run_episode
from airComp.eval.metrics import (
    agreement_rate,
    avg_pareto_efficiency,
    avg_social_welfare,
    avg_utility,
    effective_bits,
    semantic_bits_equivalent,
)
from airComp.jscc.modules import SemanticDecoder, SemanticEncoder
from airComp.utils.io import write_json


def _run_baseline_condition(llm, neg_cfg, snr_db, channel_mode, episodes, seed_offset):
    records = []
    for i in range(episodes):
        seed = seed_offset + i
        channel_a = DigitalChannel(mode=channel_mode, seed=seed * 2)
        channel_b = DigitalChannel(mode=channel_mode, seed=seed * 2 + 1)
        agent_a = TextAgent(llm, channel_a, snr_db, neg_cfg.max_messages, neg_cfg.max_retries, neg_cfg.include_rationale)
        agent_b = TextAgent(llm, channel_b, snr_db, neg_cfg.max_messages, neg_cfg.max_retries, neg_cfg.include_rationale)
        records.append(run_episode(agent_a, agent_b, seed, neg_cfg))
    return records


def _run_semantic_condition(llm, neg_cfg, jscc_cfg, encoder, decoder, snr_db, episodes, seed_offset, device):
    records = []
    for i in range(episodes):
        seed = seed_offset + i
        agent_a = SemanticAgent(
            llm, encoder, decoder, snr_db, neg_cfg.max_messages, jscc_cfg.max_count, neg_cfg.include_rationale, device=device
        )
        agent_b = SemanticAgent(
            llm, encoder, decoder, snr_db, neg_cfg.max_messages, jscc_cfg.max_count, neg_cfg.include_rationale, device=device
        )
        records.append(run_episode(agent_a, agent_b, seed, neg_cfg))
    return records


def summarize(records: list, pipeline_key: str) -> dict:
    summary = {
        "agreement_rate": agreement_rate(records),
        "avg_utility_a": avg_utility(records, "A"),
        "avg_utility_b": avg_utility(records, "B"),
        "avg_social_welfare": avg_social_welfare(records),
        "avg_pareto_efficiency": avg_pareto_efficiency(records),
        "avg_effective_payload": float(np.mean([effective_bits(r, pipeline_key) for r in records])) if records else 0.0,
    }
    if pipeline_key == "semantic":
        summary["avg_semantic_bits_equivalent"] = (
            float(np.mean([semantic_bits_equivalent(r) for r in records])) if records else 0.0
        )
    return summary


def run_sweep(config_path: str, checkpoint_path: str, episodes: int, snr_grid: list, out_path: str) -> dict:
    cfg = load_config(config_path)
    llm = LocalLLM(cfg.model.model_name, cfg.model.device, cfg.model.dtype, cfg.model.cache_dir)

    ckpt = torch.load(checkpoint_path, weights_only=False)
    jscc_cfg: JSCCConfig = ckpt.get("jscc_cfg", cfg.jscc)
    input_dim = ckpt["input_dim"]
    encoder = SemanticEncoder(input_dim, jscc_cfg.encoder_hidden_dims, jscc_cfg.k)
    decoder = SemanticDecoder(jscc_cfg.k, jscc_cfg.decoder_hidden_dims, len(ITEM_TYPES), jscc_cfg.max_count, jscc_cfg.aux_dim)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])

    results = {"raw": {}, "arq": {}, "semantic": {}}
    for snr_db in snr_grid:
        seed_offset = int(snr_db * 10_000) + 1_000_000  # deterministic, distinct seed block per SNR point
        recs_raw = _run_baseline_condition(llm, cfg.negotiation, snr_db, "raw", episodes, seed_offset)
        recs_arq = _run_baseline_condition(llm, cfg.negotiation, snr_db, "arq", episodes, seed_offset)
        recs_sem = _run_semantic_condition(llm, cfg.negotiation, jscc_cfg, encoder, decoder, snr_db, episodes, seed_offset, "cpu")

        results["raw"][str(snr_db)] = summarize(recs_raw, "digital")
        results["arq"][str(snr_db)] = summarize(recs_arq, "digital")
        results["semantic"][str(snr_db)] = summarize(recs_sem, "semantic")
        print(f"SNR={snr_db}dB done")

    write_json(out_path, results)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/snr_sweep.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--snr-grid", type=float, nargs="+", default=[-10, -5, 0, 5, 10, 15, 20])
    parser.add_argument("--out", default="results/sweep.json")
    args = parser.parse_args()
    run_sweep(args.config, args.checkpoint, args.episodes, args.snr_grid, args.out)


if __name__ == "__main__":
    main()
