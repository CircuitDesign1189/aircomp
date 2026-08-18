from __future__ import annotations

import numpy as np
import pytest

from hwlab.dsp.framing import REF_SYMBOL_POWER, BurstLayout, pilot_sequence, zadoff_chu


def test_zadoff_chu_is_constant_envelope_at_reference_power():
    seq = zadoff_chu(127, 25)
    assert np.allclose(np.abs(seq) ** 2, REF_SYMBOL_POWER)


def test_zadoff_chu_autocorrelation_is_sharp():
    seq = zadoff_chu(127, 25)
    corr = np.abs(np.correlate(np.tile(seq, 2), seq, mode="valid")[:127])
    peak = corr[0]
    sidelobe = np.max(corr[1:])
    assert peak / sidelobe > 10.0


def test_zadoff_chu_rejects_non_coprime_root():
    with pytest.raises(ValueError):
        zadoff_chu(63, 21)  # gcd(21, 63) = 21


def test_pilots_match_data_symbol_power():
    """Pilots and data must share a power scale, or the LS gain estimate is biased."""
    pilots = pilot_sequence(32)
    assert np.allclose(np.abs(pilots) ** 2, REF_SYMBOL_POWER)


def test_layout_indices_are_contiguous_and_total_adds_up():
    layout = BurstLayout()
    assert layout.preamble_start == layout.guard_symbols
    assert layout.pilot_start == layout.preamble_start + layout.preamble_len
    assert layout.data_start == layout.pilot_start + layout.n_pilots
    assert layout.total_symbols == layout.data_start + layout.n_data + layout.guard_symbols
    assert layout.overhead_symbols == layout.total_symbols - layout.n_data


def test_build_places_sections_where_the_layout_says():
    layout = BurstLayout()
    data = np.arange(1, layout.n_data + 1).astype(complex)
    burst = layout.build(data)
    assert np.allclose(burst[: layout.guard_symbols], 0)
    assert np.allclose(burst[layout.preamble_start : layout.pilot_start], layout.preamble())
    assert np.allclose(burst[layout.pilot_start : layout.data_start], layout.pilots())
    assert np.allclose(burst[layout.data_start : layout.data_start + layout.n_data], data)
    assert np.allclose(burst[layout.data_start + layout.n_data :], 0)


def test_build_rejects_wrong_payload_length():
    with pytest.raises(ValueError):
        BurstLayout().build(np.zeros(7, dtype=complex))


def test_overhead_is_large_and_reported():
    """The payload is 8 symbols; sync/pilot overhead dominates by design. This
    test exists so the number stays visible rather than quietly drifting."""
    layout = BurstLayout()
    assert layout.n_data == 8
    assert layout.overhead_symbols > 20 * layout.n_data
