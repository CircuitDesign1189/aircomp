"""The comparison figure is the deliverable of this project, so its assembly is
pinned here rather than discovered when a multi-hour sweep has just finished.

Two things bit us: run_sdr_sweep writes a flat "channel" accounting dict next to
its pipeline series, which the plotter used to treat as a pipeline and crash on;
and the hardware and simulated files each contain a pipeline whose name may
repeat, which would silently overwrite one curve with the other.
"""
from __future__ import annotations

import json

import matplotlib
import pytest

matplotlib.use("Agg")

from airComp.eval.plots import load_series, plot_sweep  # noqa: E402


def _write(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


SIM = {
    "raw": {"-10": {"agreement_rate": 0.0}, "10": {"agreement_rate": 0.8}},
    "arq": {"-10": {"agreement_rate": 0.1}, "10": {"agreement_rate": 0.9}},
    "semantic": {"-10": {"agreement_rate": 0.5}, "10": {"agreement_rate": 0.9}},
}
HW = {
    "semantic_hw": {
        # Keyed by the requested SNR; the link delivered something else, and the
        # measured value is where the point actually belongs on the axis.
        "-10": {"agreement_rate": 0.4, "measured_snr_db_mean": -9.59},
        "10": {"agreement_rate": 0.85, "measured_snr_db_mean": 9.83},
    },
    # run_sdr_sweep writes this alongside the series; it is not a pipeline.
    "channel": {"data_symbols": 8, "burst_duration_s": 0.00871},
}


def test_non_series_keys_are_ignored(tmp_path):
    """'channel' has no per-SNR summaries, so it must not be treated as a curve."""
    series = load_series([_write(tmp_path, "hw.json", HW)])

    assert set(series) == {"semantic_hw"}


def test_simulated_and_hardware_files_overlay(tmp_path):
    paths = [_write(tmp_path, "sim.json", SIM), _write(tmp_path, "hw.json", HW)]

    series = load_series(paths)

    assert set(series) == {"raw", "arq", "semantic", "semantic_hw"}
    assert series["semantic_hw"][-9.59] == 0.4
    assert series["semantic"][10.0] == 0.9


def test_hardware_points_sit_at_the_measured_snr_not_the_requested_one(tmp_path):
    """Plotting a hardware point at its requested SNR misplaces it by up to ~1.4 dB,
    which is exactly the size of the sim-vs-hardware offset the figure is meant to test."""
    series = load_series([_write(tmp_path, "hw.json", HW)])

    assert sorted(series["semantic_hw"]) == [-9.59, 9.83]
    assert -10.0 not in series["semantic_hw"]


def test_a_series_without_measured_snr_falls_back_to_its_key(tmp_path):
    series = load_series([_write(tmp_path, "sim.json", SIM)])

    assert sorted(series["raw"]) == [-10.0, 10.0]


def test_colliding_pipeline_names_are_kept_apart(tmp_path):
    """Two files both containing 'semantic' must give two curves, not one."""
    a = _write(tmp_path, "a.json", {"semantic": {"0": {"agreement_rate": 0.3}}})
    b = _write(tmp_path, "b.json", {"semantic": {"0": {"agreement_rate": 0.7}}})

    series = load_series([a, b])

    assert len(series) == 2
    assert sorted(next(iter(p.values())) for p in series.values()) == [0.3, 0.7]


def test_plot_writes_a_figure_with_every_curve(tmp_path):
    paths = [_write(tmp_path, "sim.json", SIM), _write(tmp_path, "hw.json", HW)]
    out = tmp_path / "fig.png"

    drawn = plot_sweep(paths, str(out))

    assert out.stat().st_size > 0
    assert len(drawn) == 4


def test_a_missing_metric_is_an_error_not_an_empty_plot(tmp_path):
    """Silently producing an empty figure after a long sweep is the worst outcome."""
    path = _write(tmp_path, "sim.json", SIM)

    with pytest.raises(ValueError, match="avg_social_welfare"):
        plot_sweep([path], str(tmp_path / "fig.png"), metric="avg_social_welfare")


def test_a_floor_line_is_drawn_without_disturbing_the_curves(tmp_path):
    """The semantic decoder scores ~0.48 on a channel carrying no information, so the
    figure needs that reference line or its low-SNR tail reads as communication."""
    paths = [_write(tmp_path, "sim.json", SIM)]
    out = tmp_path / "fig.png"

    drawn = plot_sweep(paths, str(out), floor=0.48)

    assert len(drawn) == 3
    assert out.stat().st_size > 0
