"""Run baseline (raw + arq) and semantic pipelines across an SNR grid with
paired seeds, so the comparison isolates the effect of channel condition
rather than random pool/value draws.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from airComp.agents.baseline_agent import TextAgent
from airComp.agents.compact_agent import CompactAgent
from airComp.agents.factory import build_llm
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


def _run_compact_condition(llm, neg_cfg, snr_db, channel_mode, episodes, seed_offset):
    """Same shape as _run_baseline_condition, and the same seeds, so a compact run
    and a text run at the same SNR see identical pools, values and LLM calls."""
    records = []
    for i in range(episodes):
        seed = seed_offset + i
        channel_a = DigitalChannel(mode=channel_mode, seed=seed * 2)
        channel_b = DigitalChannel(mode=channel_mode, seed=seed * 2 + 1)
        agent_a = CompactAgent(llm, channel_a, snr_db, neg_cfg.max_messages, neg_cfg.max_retries, neg_cfg.include_rationale)
        agent_b = CompactAgent(llm, channel_b, snr_db, neg_cfg.max_messages, neg_cfg.max_retries, neg_cfg.include_rationale)
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


#: Conditions run by default. Selecting a subset matters for cost, not just tidiness:
#: the text-baseline conditions are the expensive ones at low SNR, because corrupted
#: JSON drives the parser's bounded retries and each episode costs up to 3x the LLM
#: calls. Extending the grid downward to find the semantic knee is ~30 min
#: semantic-only and hours with the baselines -- and the baselines are flat at 0.00
#: by then.
#:
#: The five conditions form the decomposition the comparison rests on:
#:   raw / arq   -- the whole LLM completion on the wire, ~1000 bits/message
#:   compact     -- the same act source-coded to 8 bits, uncoded
#:   compact_fec -- the same 8 bits under Hamming(7,4): 16 channel uses, exactly
#:                  matching the semantic latent's k=16
#:   semantic    -- the analog latent
#: raw->compact isolates source coding, compact->compact_fec isolates error
#: correction, and compact_fec->semantic isolates what this project actually claims.
ALL_PIPELINES = ("raw", "arq", "compact", "compact_fec", "semantic")


def run_sweep(config_path: str, checkpoint_path: str, episodes: int, snr_grid: list, out_path: str,
              pipelines=ALL_PIPELINES, survive_lost_messages: bool = False) -> dict:
    cfg = load_config(config_path)
    if survive_lost_messages:
        # Removes the largest structural asymmetry between the pipelines: a lost
        # digital frame no longer ends the episode, so every pipeline gets the
        # same max_messages turns. Semantic results are unaffected -- its decoder
        # cannot produce an undecodable message -- so this only moves the
        # baselines, and by how much is the point of running it.
        cfg.negotiation.lost_message_ends_episode = False
    llm = build_llm(cfg.model)

    ckpt = torch.load(checkpoint_path, weights_only=False)
    jscc_cfg: JSCCConfig = ckpt.get("jscc_cfg", cfg.jscc)
    input_dim = ckpt["input_dim"]
    encoder = SemanticEncoder(input_dim, jscc_cfg.encoder_hidden_dims, jscc_cfg.k)
    decoder = SemanticDecoder(jscc_cfg.k, jscc_cfg.decoder_hidden_dims, len(ITEM_TYPES), jscc_cfg.max_count, jscc_cfg.aux_dim)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])

    unknown = [p for p in pipelines if p not in ALL_PIPELINES]
    if unknown:
        raise ValueError(f"unknown pipeline(s) {unknown}; choose from {list(ALL_PIPELINES)}")

    results = {p: {} for p in pipelines}
    # Report per condition rather than per grid point: a grid point is three
    # conditions x `episodes`, so at 50 episodes the first sign of life would
    # otherwise be an hour away.
    started = time.time()
    total_units = len(pipelines) * len(snr_grid)
    done_units = 0
    for snr_db in snr_grid:
        seed_offset = int(snr_db * 10_000) + 1_000_000  # deterministic, distinct seed block per SNR point
        runners = {
            "raw": lambda: _run_baseline_condition(llm, cfg.negotiation, snr_db, "raw", episodes, seed_offset),
            "arq": lambda: _run_baseline_condition(llm, cfg.negotiation, snr_db, "arq", episodes, seed_offset),
            "compact": lambda: _run_compact_condition(llm, cfg.negotiation, snr_db, "raw", episodes, seed_offset),
            "compact_fec": lambda: _run_compact_condition(llm, cfg.negotiation, snr_db, "fec", episodes, seed_offset),
            "semantic": lambda: _run_semantic_condition(
                llm, cfg.negotiation, jscc_cfg, encoder, decoder, snr_db, episodes, seed_offset, "cpu"),
        }
        for key in pipelines:
            runner = runners[key]
            t0 = time.time()
            records = runner()
            results[key][str(snr_db)] = summarize(records, "semantic" if key == "semantic" else "digital")
            done_units += 1
            rate = (time.time() - started) / done_units
            print(
                f"[{done_units:2d}/{total_units}] SNR {snr_db:+.0f} dB {key:8s} "
                f"agreement {results[key][str(snr_db)]['agreement_rate']:.2f}  "
                f"{(time.time()-t0)/episodes:.1f} s/episode  "
                f"eta {(total_units - done_units) * rate / 60:.0f} min",
                flush=True,
            )
            # Written after every condition: this run takes hours, and losing all of
            # it to a crash in the last grid point is not an acceptable failure mode.
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
