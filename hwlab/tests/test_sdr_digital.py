"""The compact baseline over real RF must land on the same SNR axis as the
simulated one, or the hardware curve cannot be compared to `compact_fec` and
the run is wasted bench time.

`test_mapping.py` pins that axis for the analog path against AnalogAWGNChannel.
This is the digital counterpart: it pins the measured BER of the RF bit path
against the closed form the simulated DigitalChannel obeys. A 3 dB convention
slip -- the failure hwlab/dsp/mapping.py exists to prevent -- shows up here as a
factor-of-several error in BER long before it shows up as a shifted curve.
"""
from __future__ import annotations

from math import erfc, sqrt

import numpy as np
import pytest

from airComp.baseline.offer_codec import OFFER_BITS, bits_to_offer, offer_to_bits
from airComp.channel.digital import FEC_FRAME_BITS, DigitalChannel
from airComp.env.negotiation import Offer, Pool
from hwlab.channel.sdr_analog import SDRAnalogChannel
from hwlab.channel.sdr_digital import SDRDigitalChannel
from hwlab.config import BurstConfig, GainConfig, LinkConfig, LoopbackConfig
from hwlab.dsp.burst import BurstCodec
from hwlab.dsp.pulse import clipping_fraction
from hwlab.radio.loopback import LoopbackBackend

PAYLOAD = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)


def q(x: float) -> float:
    return 0.5 * erfc(x / sqrt(2.0))


def make_digital(tx_vga_db: float = 30.0, **loopback) -> SDRDigitalChannel:
    backend = LoopbackBackend(LoopbackConfig(**loopback))
    analog = SDRAnalogChannel(
        backend, LinkConfig(), BurstConfig(), calibration=None,
        fixed_gains=GainConfig(tx_vga_db=tx_vga_db),
    )
    return SDRDigitalChannel(analog)


# --- the load-bearing check -------------------------------------------------


#: Loopback TX gains and the SNR each delivers. The SNR of a real link is set by
#: gain and attenuation, not by asking for it -- `SDRAnalogChannel` treats snr_db
#: as a *request* and only honours it through a calibration table. So the law has
#: to be checked against the SNR the link actually measured.
@pytest.mark.parametrize("tx_vga_db", [18.0, 22.0, 26.0])
def test_measured_ber_matches_the_closed_form(tx_vga_db):
    """BER over the RF bit path must equal Q(sqrt(snr_linear)) at the SNR the
    link delivered -- the same law the simulated DigitalChannel follows.

    This is where a 3 dB convention slip would surface, so the test also asserts
    that the textbook Eb/N0 form is *rejected* at this tolerance. Do not loosen
    the tolerance without checking that second assertion still bites.
    """
    channel = make_digital(tx_vga_db=tx_vga_db)
    rng = np.random.default_rng(0)
    errors = total = 0
    measured = []
    for _ in range(250):
        bits = rng.integers(0, 2, size=8).astype(np.uint8)
        _decoded, stats = channel.transmit_bits(bits, snr_db=0.0)
        errors += stats["bit_errors"]
        total += stats["n_bits"]
        assert stats["measured_snr_db"] is not None, "no burst may be lost at these gains"
        measured.append(stats["measured_snr_db"])

    snr_linear = 10 ** (float(np.mean(measured)) / 10.0)
    ber = errors / total

    assert ber == pytest.approx(q(sqrt(snr_linear)), rel=0.20)
    assert ber != pytest.approx(q(sqrt(2 * snr_linear)), rel=0.20), (
        "the textbook Eb/N0 curve fits too -- the tolerance is too loose to pin the convention"
    )


# --- the frame --------------------------------------------------------------


def test_the_frame_round_trips_at_high_snr():
    channel = make_digital()

    decoded, stats = channel.transmit_bits(PAYLOAD, snr_db=30.0)

    assert list(decoded) == list(PAYLOAD)
    assert stats["bit_errors"] == 0
    assert stats["burst_lost"] is False


def test_it_costs_the_same_channel_uses_as_the_semantic_latent():
    """16 real dimensions = 8 complex symbols = the k=16 burst, unchanged."""
    channel = make_digital()

    _decoded, stats = channel.transmit_bits(PAYLOAD, snr_db=30.0)

    assert stats["n_bits"] == FEC_FRAME_BITS == 16
    assert channel.analog.k == 16
    assert channel.analog.codec.layout.n_data == 8


def test_measured_snr_is_reported_so_the_x_axis_can_be_the_real_one():
    channel = make_digital()

    _decoded, stats = channel.transmit_bits(PAYLOAD, snr_db=10.0)

    assert stats["measured_snr_db"] == pytest.approx(10.0, abs=2.0)
    assert stats["requested_snr_db"] == 10.0


def test_fec_corrections_are_accounted_for():
    channel = make_digital()
    corrected = residual_ok = 0
    for _ in range(60):
        _decoded, stats = channel.transmit_bits(PAYLOAD, snr_db=4.0)
        assert stats["bit_errors"] >= stats["residual_bit_errors"]
        corrected += stats["fec_corrected"]
        residual_ok += int(stats["residual_bit_errors"] == 0)

    assert corrected > 0, "at +4 dB the code must be doing something"
    assert residual_ok > 0


# --- the operating point ----------------------------------------------------


def test_a_bpsk_payload_does_not_move_the_transmit_operating_point():
    """`tx_scale` is fixed, computed once from a Gaussian burst, so what matters
    is that the BPSK payload carries the same *average* data power -- that is
    what sets SNR per real -- while using no more DAC headroom.

    It uses measurably less: a normalized Gaussian latent has occasional
    components near 2.5 where BPSK is always exactly 1, so the constant-modulus
    payload peaks about 30% lower. That is headroom left unused, not a power
    difference, and it is the safe direction.
    """
    codec = BurstCodec(LinkConfig(), BurstConfig())
    rng = np.random.default_rng(0)
    latent = rng.normal(size=codec.k)
    latent *= np.sqrt(codec.k) / np.linalg.norm(latent)
    bpsk = rng.integers(0, 2, size=codec.k).astype(float) * 2.0 - 1.0

    wave_latent, wave_bpsk = codec.modulate(latent), codec.modulate(bpsk)

    assert np.mean(bpsk ** 2) == pytest.approx(np.mean(latent ** 2))  # the thing that sets SNR
    assert clipping_fraction(wave_bpsk) == 0.0
    assert np.max(np.abs(wave_bpsk)) <= np.max(np.abs(wave_latent))


def test_the_bpsk_frame_carries_unit_power_per_real_like_the_latent():
    """mapping.py's SIGNAL_POWER_PER_REAL = 1.0 holds for +-1 exactly, which is
    why no rescaling is applied -- and rescaling is where 3 dB errors come from."""
    bpsk = np.array([1, -1, -1, 1] * 4, dtype=float)

    assert float(np.mean(bpsk ** 2)) == pytest.approx(1.0)


# --- failure modes ----------------------------------------------------------


def test_a_lost_burst_becomes_an_implicit_reject_not_a_crash():
    """Noise-only in, random bits out, an index off the codebook, and the offer
    decodes to None -- the same outcome the simulated compact pipeline gives a
    destroyed frame."""
    channel = make_digital(tx_vga_db=0.0, path_loss_db=200.0)
    pool = Pool(counts={"book": 1, "hat": 1, "ball": 1})
    outcomes = []
    for _ in range(30):
        with pytest.warns(RuntimeWarning):
            decoded, stats = channel.transmit_bits(np.zeros(8, dtype=np.uint8), -20.0)
        assert stats["burst_lost"] is True
        outcomes.append(bits_to_offer(decoded, pool))

    assert any(o is None for o in outcomes)
    assert all(o is None or o.action in ("propose", "accept", "reject") for o in outcomes)


def test_an_uncoded_mode_is_refused_rather_than_silently_mismatched():
    """8 uncoded bits are 4 complex symbols, which is a different burst. Sharing
    this one would change the channel-use budget without saying so."""
    backend = LoopbackBackend(LoopbackConfig())
    analog = SDRAnalogChannel(backend, LinkConfig(), BurstConfig(), None, GainConfig())

    with pytest.raises(ValueError, match="only 'fec'"):
        SDRDigitalChannel(analog, mode="raw")


def test_a_burst_sized_for_a_different_k_is_refused():
    backend = LoopbackBackend(LoopbackConfig())
    analog = SDRAnalogChannel(backend, LinkConfig(), BurstConfig(n_data=4), None, GainConfig())

    with pytest.raises(ValueError, match="n_data"):
        SDRDigitalChannel(analog)


# --- end to end against the codec -------------------------------------------


def test_a_real_offer_survives_the_radio():
    channel = make_digital()
    pool = Pool(counts={"book": 2, "hat": 2, "ball": 1})
    offer = Offer(action="propose", counts={"ball": 1, "book": 2, "hat": 0})

    decoded, _ = channel.transmit_bits(offer_to_bits(offer, pool), snr_db=30.0)
    received = bits_to_offer(decoded, pool)

    assert received.action == "propose"
    assert received.counts == {"ball": 1, "book": 2, "hat": 0}
    assert OFFER_BITS == 8


def test_the_rf_path_and_the_simulated_path_agree_on_frame_survival():
    """Not a bit-exact check -- different noise draws -- but a gross mismatch in
    survival rate at a matched SNR means the two are not on the same axis."""
    rf = make_digital()
    rf_ok = sum(
        int(list(rf.transmit_bits(PAYLOAD, 3.0)[0]) == list(PAYLOAD)) for _ in range(80)
    )
    sim_ok = sum(
        int(list(DigitalChannel(mode="fec", seed=s).transmit_bits(PAYLOAD, 3.0)[0]) == list(PAYLOAD))
        for s in range(80)
    )

    assert abs(rf_ok - sim_ok) < 20, f"rf {rf_ok}/80 vs sim {sim_ok}/80"
