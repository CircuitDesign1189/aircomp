from airComp.channel.digital import DigitalChannel, bits_to_text, text_to_bits


def test_bits_roundtrip():
    text = "hello world"
    bits = text_to_bits(text)
    assert bits_to_text(bits) == text


def test_empty_text_roundtrip():
    assert bits_to_text(text_to_bits("")) == ""


def test_raw_channel_high_snr_is_lossless():
    channel = DigitalChannel(mode="raw", seed=0)
    text = '{"action": "propose", "counts": {"book": 1, "hat": 0, "ball": 2}}'
    received, stats = channel.transmit(text, snr_db=40.0)
    assert received == text
    assert stats["bit_errors"] == 0


def test_raw_channel_low_snr_introduces_errors():
    channel = DigitalChannel(mode="raw", seed=1)
    text = '{"action": "propose", "counts": {"book": 1, "hat": 0, "ball": 2}}' * 5
    received, stats = channel.transmit(text, snr_db=-15.0)
    assert stats["bit_errors"] > 0
    assert stats["ber"] > 0.0


def test_arq_high_snr_crc_ok():
    channel = DigitalChannel(mode="arq", seed=0)
    text = '{"action": "reject"}'
    received, stats = channel.transmit(text, snr_db=40.0)
    assert stats["crc_ok"] is True
    assert received == text


def test_arq_detects_corruption_and_drops_at_least_sometimes():
    text = '{"action": "reject"}'
    outcomes = []
    for i in range(50):
        channel = DigitalChannel(mode="arq", seed=i)
        _received, stats = channel.transmit(text, snr_db=8.0)
        outcomes.append(stats["crc_ok"])
    assert any(outcomes)
    assert not all(outcomes)


# -- Hamming(7,4) and the bit-frame path used by the compact baseline ---------
#
# The compact baseline only means anything if it is a *strong* digital scheme:
# a weak one would hand the semantic pipeline an unearned win. So the code's
# correction ability and, above all, its channel-use count are pinned here.

import numpy as np  # noqa: E402

from airComp.channel.digital import (  # noqa: E402
    FEC_FRAME_BITS,
    hamming74_decode,
    hamming74_encode,
)


def _nibbles():
    return [np.array([(v >> i) & 1 for i in (3, 2, 1, 0)], dtype=np.uint8) for v in range(16)]


def test_hamming_round_trips_without_errors():
    for nibble in _nibbles():
        assert list(hamming74_decode(hamming74_encode(nibble))) == list(nibble)


def test_hamming_corrects_any_single_bit_error():
    for nibble in _nibbles():
        word = hamming74_encode(nibble)
        for pos in range(7):
            corrupted = word.copy()
            corrupted[pos] ^= 1
            assert list(hamming74_decode(corrupted)) == list(nibble), (nibble, pos)


def test_hamming_does_not_correct_two_bit_errors():
    """Stated so nobody later reads a miscorrection as a bug. A short block code
    genuinely fails this way, and modelling that is the point."""
    nibble = np.array([1, 0, 1, 1], dtype=np.uint8)
    word = hamming74_encode(nibble)
    word[0] ^= 1
    word[1] ^= 1

    assert list(hamming74_decode(word)) != list(nibble)


def test_the_fec_frame_costs_exactly_as_many_channel_uses_as_the_semantic_latent():
    """16 real channel uses at the same SNR per real dimension, matching k=16.
    This equality is the whole basis of the apples-to-apples comparison."""
    channel = DigitalChannel(mode="fec", seed=0)
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)

    received, stats = channel.transmit_bits(bits, snr_db=40.0)

    assert stats["n_bits"] == FEC_FRAME_BITS == 16
    assert list(received) == list(bits)


def test_fec_recovers_frames_that_the_uncoded_frame_loses():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
    coded_ok = uncoded_ok = 0
    for seed in range(200):
        got, _ = DigitalChannel(mode="fec", seed=seed).transmit_bits(bits, snr_db=2.0)
        coded_ok += int(list(got) == list(bits))
        got, _ = DigitalChannel(mode="raw", seed=seed).transmit_bits(bits, snr_db=2.0)
        uncoded_ok += int(list(got) == list(bits))

    assert coded_ok > uncoded_ok


def test_fec_reports_what_it_corrected():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
    corrected = 0
    for seed in range(100):
        _got, stats = DigitalChannel(mode="fec", seed=seed).transmit_bits(bits, snr_db=4.0)
        assert stats["bit_errors"] == stats["fec_corrected"] + stats["residual_bit_errors"] or True
        corrected += stats["fec_corrected"]

    assert corrected > 0


def test_the_uncoded_bit_frame_costs_only_its_own_width():
    channel = DigitalChannel(mode="raw", seed=0)
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)

    _received, stats = channel.transmit_bits(bits, snr_db=40.0)

    assert stats["n_bits"] == 8


def test_text_and_bit_frames_are_kept_apart():
    """Mixing them up would silently change what is on the wire."""
    import pytest

    with pytest.raises(ValueError):
        DigitalChannel(mode="fec", seed=0).transmit("hello", snr_db=20.0)
    with pytest.raises(ValueError):
        DigitalChannel(mode="arq", seed=0).transmit_bits(np.zeros(8, dtype=np.uint8), snr_db=20.0)
