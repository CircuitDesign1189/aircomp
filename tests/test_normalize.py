"""The headline number of this project is an effective SNR gain between two
normalised curves. It was previously computed by hand in a shell one-liner,
which is not something a result should depend on.
"""
from __future__ import annotations

import json

import pytest

from airComp.eval.normalize import (
    crossing,
    effective_snr_gain,
    gain_table,
    load_curve,
    normalize,
)


def _write(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_integer_and_float_snr_keys_are_the_same_axis():
    """Real result files disagree on this -- sweep.json wrote "-10", every later
    run wrote "-10.0" -- and mixing them silently drops points from a curve."""
    import tempfile
    import pathlib

    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "s.json"
        p.write_text(json.dumps({"x": {"-10": {"agreement_rate": 0.1}, "0.0": {"agreement_rate": 0.6}}}))
        curve = load_curve(str(p), "x")

    assert sorted(curve) == [-10.0, 0.0]


def test_normalization_maps_floor_to_zero_and_ceiling_to_one():
    curve = {0.0: 0.48, 10.0: 0.98}

    out = normalize(curve, floor=0.48, ceiling=0.98)

    assert out[0.0] == pytest.approx(0.0)
    assert out[10.0] == pytest.approx(1.0)


def test_a_high_floor_is_not_credited_as_performance():
    """The semantic pipeline agrees 48% of the time on a dead channel. Comparing
    it raw against a pipeline that floors at 0.06 overstates it by exactly that."""
    semantic = normalize({-10.0: 0.70}, floor=0.48, ceiling=0.98)
    compact = normalize({-10.0: 0.20}, floor=0.10, ceiling=0.94)

    assert 0.70 - 0.20 == pytest.approx(0.50)  # the naive read
    assert semantic[-10.0] - compact[-10.0] < 0.35  # what survives normalisation


def test_ceiling_must_exceed_floor():
    with pytest.raises(ValueError):
        normalize({0.0: 0.5}, floor=0.9, ceiling=0.9)


def test_crossing_interpolates_between_grid_points():
    curve = {-5.0: 0.24, 0.0: 0.67}

    assert crossing(curve, 0.5) == pytest.approx(-5.0 + 5.0 * (0.5 - 0.24) / (0.67 - 0.24))


def test_a_curve_that_never_reaches_the_level_has_no_crossing():
    """None, not an enormous gain: a pipeline that never gets there must not be
    reported as infinitely worse."""
    assert crossing({-10.0: 0.1, 10.0: 0.3}, 0.9) is None
    assert effective_snr_gain({-10.0: 0.1, 10.0: 0.95}, {-10.0: 0.1, 10.0: 0.3}, 0.9) is None


def test_gain_is_the_horizontal_distance_between_curves():
    reference = {0.0: 0.4, 10.0: 0.6}   # crosses 0.5 at +5
    candidate = {-10.0: 0.4, 0.0: 0.6}  # crosses 0.5 at -5

    assert effective_snr_gain(reference, candidate, 0.5) == pytest.approx(10.0)


def test_gain_is_negative_when_the_candidate_is_worse():
    reference = {-10.0: 0.4, 0.0: 0.6}
    candidate = {0.0: 0.4, 10.0: 0.6}

    assert effective_snr_gain(reference, candidate, 0.5) == pytest.approx(-10.0)


def test_gain_table_reports_both_crossings_so_the_number_can_be_checked():
    table = gain_table({0.0: 0.4, 10.0: 0.6}, {-10.0: 0.4, 0.0: 0.6}, levels=(0.5,))

    assert table[0.5]["reference_snr"] == pytest.approx(5.0)
    assert table[0.5]["candidate_snr"] == pytest.approx(-5.0)
    assert table[0.5]["gain_db"] == pytest.approx(10.0)


def test_a_missing_pipeline_is_an_error(tmp_path):
    path = _write(tmp_path, "s.json", {"raw": {"0": {"agreement_rate": 0.5}}})

    with pytest.raises(KeyError, match="semantic"):
        load_curve(path, "semantic")
