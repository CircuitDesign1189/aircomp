"""Comparison plots across SNR, from one or more sweep result files.

The headline figure of this project overlays four curves on one axis: the two
baseline channel modes and the simulated semantic pipeline (all from
`airComp/eval/snr_sweep.py`), plus the semantic pipeline measured over the real
radio (`hwlab/scripts/run_sdr_sweep.py`). Those live in separate files because
they are produced by separate runs, so plotting takes a list of paths.

Overlaying them is only meaningful because both sweeps derive their episode seeds
from the same formula, `int(snr_db * 10_000) + 1_000_000`, so episode i at a given
SNR sees identical pools and private values in both. A systematic gap between the
hardware and simulated semantic curves is therefore a bug, not a physical effect.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt

#: Keys that sit alongside the pipeline series in a results file but are not
#: series themselves -- e.g. run_sdr_sweep writes a flat "channel" accounting
#: dict. Rather than name them, a series is recognised by its shape.
def _is_series(value, metric: str) -> bool:
    """A series maps SNR-as-string -> summary dict containing `metric`."""
    if not isinstance(value, dict) or not value:
        return False
    return all(isinstance(v, dict) and metric in v for v in value.values())


#: A hardware point is keyed by the SNR that was *requested*, but the gains only
#: land within about a dB of it, so the key is not where the point belongs on the
#: axis. run_sdr_sweep records what the link actually delivered and its docstring
#: is explicit that the x-axis is the measured value; honour that here, or the
#: hardware and simulated curves are silently plotted against different x.
MEASURED_SNR_KEY = "measured_snr_db_mean"


def _snr_of(key: str, summary: dict) -> float:
    measured = summary.get(MEASURED_SNR_KEY)
    return float(measured) if isinstance(measured, (int, float)) else float(key)


def load_series(results_paths, metric: str = "agreement_rate") -> dict:
    """Collect {label: {snr: value}} from every pipeline in every file.

    Labels are disambiguated by file only when they collide, so the common case
    (distinct pipeline names) stays readable. The SNR is the measured one where a
    run reported it, and the requested one otherwise.
    """
    per_file = []
    for path in results_paths:
        with open(path, "r", encoding="utf-8") as f:
            results = json.load(f)
        per_file.append(
            (path, {name: s for name, s in results.items() if _is_series(s, metric)})
        )

    counts: dict = {}
    for _, series in per_file:
        for name in series:
            counts[name] = counts.get(name, 0) + 1

    out: dict = {}
    for path, series in per_file:
        for name, points in series.items():
            label = name if counts[name] == 1 else f"{name} ({path})"
            out[label] = {_snr_of(snr, summary): summary[metric] for snr, summary in points.items()}
    return out


def plot_sweep(results_paths, out_path: str, metric: str = "agreement_rate",
               floor: float | None = None) -> dict:
    """Plot `metric` against SNR for every pipeline found. Returns what it drew.

    `floor` draws a horizontal reference line. Pass it whenever one is known: the
    semantic decoder always emits a structurally valid offer, so it scores well
    above zero on a channel carrying no information at all (measured 0.48 at
    -60 dB). Without that line the reader reads the low-SNR tail as communication
    when it is the decoder's prior, which overstates the result.
    """
    if isinstance(results_paths, str):
        results_paths = [results_paths]
    series = load_series(results_paths, metric)
    if not series:
        raise ValueError(f"no pipeline series containing {metric!r} in {list(results_paths)}")

    fig, ax = plt.subplots()
    for label, points in sorted(series.items()):
        snrs = sorted(points)
        # Hardware runs are dashed: they are the measured curve, and the eye should
        # be able to tell them from the simulated ones without reading the legend.
        style = "--" if "_hw" in label else "-"
        ax.plot(snrs, [points[s] for s in snrs], style, marker="o", label=label)

    if floor is not None:
        ax.axhline(floor, color="0.4", linestyle=":", linewidth=1.2)
        ax.text(0.01, floor, f" no-information floor ({floor:.2f})", color="0.3",
                fontsize=8, va="bottom", transform=ax.get_yaxis_transform())

    ax.set_xlabel("measured SNR (dB)")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs SNR")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return series
