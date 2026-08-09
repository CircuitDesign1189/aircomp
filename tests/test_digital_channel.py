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
