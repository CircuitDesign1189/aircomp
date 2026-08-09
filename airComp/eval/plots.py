"""Milestone 3: comparison plots from a snr_sweep.py results JSON file."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt


def plot_sweep(results_path: str, out_path: str, metric: str = "agreement_rate") -> None:
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    fig, ax = plt.subplots()
    for pipeline, series in results.items():
        snrs = sorted(series.keys(), key=float)
        values = [series[s][metric] for s in snrs]
        ax.plot([float(s) for s in snrs], values, marker="o", label=pipeline)

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs SNR")
    ax.legend()
    fig.savefig(out_path)
