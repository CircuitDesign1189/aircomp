"""Phase 3: run the semantic pipeline over the real RF link across an SNR grid.

Episode seeds use the SAME formula as `airComp/eval/snr_sweep.py`
(`int(snr_db*10_000) + 1_000_000`), so at each SNR point the hardware run sees
the identical pools and private values as the simulation run. That pairing is
what makes the headline check valid: overlay this curve on the simulation curve
and any systematic offset is a bug in the DSP, not a physical effect. Suspect
the SNR convention (hwlab/dsp/mapping.py) and the gain calibration first.

The x-axis is the MEASURED SNR reported per burst, not the requested value.

Hardware is orders of magnitude slower per episode than simulation, so the
default episode count is 15 rather than the 100 used for the software sweep.

Usage:
    python -m hwlab.scripts.run_sdr_sweep --checkpoint checkpoints/jscc_v1.pt --backend loopback
    python -m hwlab.scripts.run_sdr_sweep --checkpoint checkpoints/jscc_v1.pt --backend hackrf --episodes 15
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from airComp.agents.factory import build_llm
from airComp.config import ITEM_TYPES, JSCCConfig, load_config
from airComp.env.negotiation import run_episode
from airComp.eval.snr_sweep import summarize
from airComp.jscc.modules import SemanticDecoder, SemanticEncoder
from airComp.utils.io import write_json
from hwlab.agent import HardwareSemanticAgent
from hwlab.channel.calibration import Calibration
from hwlab.channel.sdr_analog import SDRAnalogChannel
from hwlab.scripts._common import (
    add_common_args,
    build_backend,
    build_config,
    check_clock,
    report_setup_problems,
)

#: Must stay identical to airComp/eval/snr_sweep.py, or the paired-seed
#: comparison against the simulation run is silently broken.
SEED_OFFSET_BASE = 1_000_000


def seed_offset_for(snr_db: float) -> int:
    return int(snr_db * 10_000) + SEED_OFFSET_BASE


def _burst_summary(stats: list) -> dict:
    measured = [s["measured_snr_db"] for s in stats if s.get("measured_snr_db") is not None]
    return {
        "bursts": len(stats),
        "burst_loss_rate": float(np.mean([s["burst_lost"] for s in stats])) if stats else 0.0,
        "measured_snr_db_mean": float(np.mean(measured)) if measured else None,
        "measured_snr_db_std": float(np.std(measured)) if measured else None,
        "retry_rate": float(np.mean([s["attempts"] > 1 for s in stats])) if stats else 0.0,
        "rx_warnings": sorted({w for s in stats for w in s.get("rx_warnings", [])}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--sim-config", default="configs/snr_sweep.yaml", help="airComp config (model, negotiation)")
    parser.add_argument("--checkpoint", required=True, help="trained JSCC checkpoint")
    parser.add_argument("--episodes", type=int, default=15,
                        help="episodes per SNR point (hardware is slow; the software sweep uses 100)")
    parser.add_argument("--snr-grid", type=float, nargs="+", default=[-10, -5, 0, 5, 10, 15, 20])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="hwlab/results/sdr_sweep.json")
    args = parser.parse_args()
    return report_setup_problems(_run, args)


def _run(args) -> int:
    cfg = load_config(args.sim_config)
    hw = build_config(args)
    # Devices before the clock check, and both before the model loads: a bench fault
    # should surface in seconds, not after a multi-gigabyte checkpoint has been read.
    backend = build_backend(args, hw)
    print(check_clock(args, hw))

    ckpt = torch.load(args.checkpoint, weights_only=False)
    jscc_cfg: JSCCConfig = ckpt.get("jscc_cfg", cfg.jscc)
    encoder = SemanticEncoder(ckpt["input_dim"], jscc_cfg.encoder_hidden_dims, jscc_cfg.k)
    decoder = SemanticDecoder(jscc_cfg.k, jscc_cfg.decoder_hidden_dims, len(ITEM_TYPES), jscc_cfg.max_count, jscc_cfg.aux_dim)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])

    if 2 * hw.burst.n_data != jscc_cfg.k:
        raise SystemExit(
            f"checkpoint has k={jscc_cfg.k} but the burst carries {2*hw.burst.n_data} reals. "
            f"Set burst.n_data = {jscc_cfg.k // 2} in {args.config}."
        )

    calibration = None
    if hw.calibration_path and Path(hw.calibration_path).exists():
        calibration = Calibration.load(hw.calibration_path)
        lo, hi = calibration.achievable_range_db()
        print(f"calibration: {len(calibration.usable())} usable points, {lo:+.1f}..{hi:+.1f} dB")
    else:
        print(f"WARNING: no calibration at {hw.calibration_path!r}; using the fixed gains from the "
              f"config for every SNR point. Run hwlab.scripts.calibrate_snr first.")

    channel = SDRAnalogChannel(backend, hw.link, hw.burst, calibration, hw.gains)
    print(f"channel cost per message: {channel.payload_accounting()}")

    llm = build_llm(cfg.model)

    results = {"semantic_hw": {}, "channel": channel.payload_accounting()}
    try:
        for snr_db in args.snr_grid:
            mark = len(channel.stats_log)
            offset = seed_offset_for(snr_db)
            records = []
            for i in range(args.episodes):
                agents = [
                    HardwareSemanticAgent(
                        llm, encoder, decoder, snr_db,
                        cfg.negotiation.max_messages, jscc_cfg.max_count, cfg.negotiation.include_rationale,
                        device=args.device, channel=channel,
                    )
                    for _ in range(2)
                ]
                records.append(run_episode(agents[0], agents[1], offset + i, cfg.negotiation))

            summary = summarize(records, "semantic")
            summary.update(_burst_summary(channel.stats_log[mark:]))
            summary["requested_snr_db"] = float(snr_db)
            results["semantic_hw"][str(snr_db)] = summary
            print(f"SNR request {snr_db:+.0f} dB -> measured {summary['measured_snr_db_mean']}, "
                  f"agreement {summary['agreement_rate']:.2f}, loss {summary['burst_loss_rate']:.1%}")
    finally:
        channel.close()

    write_json(args.out, results)
    print(f"\nwrote {args.out}")
    print("Next: overlay these against the simulation sweep. A systematic offset means a bug --")
    print("check the SNR convention in hwlab/dsp/mapping.py and the gain calibration first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
