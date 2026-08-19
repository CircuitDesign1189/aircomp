"""SDRAnalogChannel must be a true drop-in for AnalogAWGNChannel, and
HardwareSemanticAgent must drive it without any change to airComp/.

Uses a stub LLM so the whole hardware agent path is covered without a model
download -- these stay in the fast (non-slow) test set.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from airComp.channel.analog import AnalogAWGNChannel
from airComp.config import ITEM_TYPES
from airComp.env.negotiation import Pool, Values, run_episode
from airComp.jscc.modules import SemanticDecoder, SemanticEncoder
from hwlab.agent import HardwareSemanticAgent
from hwlab.channel.calibration import Calibration, CalibrationPoint
from hwlab.channel.sdr_analog import SDRAnalogChannel
from hwlab.config import BurstConfig, GainConfig, LinkConfig, LoopbackConfig
from hwlab.dsp.mapping import noise_var_per_real
from hwlab.radio.loopback import LoopbackBackend

INPUT_DIM = 64
K = 16


def make_channel(tx_vga_db: float = 30.0, fading: bool = False, seed: int = 0, **loopback) -> SDRAnalogChannel:
    backend = LoopbackBackend(LoopbackConfig(**loopback))
    return SDRAnalogChannel(
        backend,
        LinkConfig(),
        BurstConfig(),
        calibration=None,
        fixed_gains=GainConfig(tx_vga_db=tx_vga_db),
        fading=fading,
        seed=seed,
    )


class StubLLM:
    """Stands in for LocalLLM: SemanticAgent only calls chat_with_hidden."""

    def __init__(self, seed: int = 0):
        self.generator = torch.Generator().manual_seed(seed)

    def chat_with_hidden(self, system_prompt, history, user_prompt, max_new_tokens, temperature):
        return "", torch.randn(INPUT_DIM, generator=self.generator)


# --- channel ----------------------------------------------------------------


def test_call_signature_matches_analog_awgn_channel():
    """Same call shape, same output shape/dtype/device -- that is the contract
    SemanticAgent.take_turn relies on."""
    channel = make_channel()
    z = torch.randn(1, K)
    z = z * (K**0.5) / z.norm(dim=-1, keepdim=True)

    hardware = channel(z, 10.0)
    simulated = AnalogAWGNChannel()(z, 10.0)

    assert hardware.shape == simulated.shape == z.shape
    assert hardware.dtype == z.dtype
    assert hardware.device == z.device


def test_delivers_the_noise_level_it_reports():
    channel = make_channel(tx_vga_db=30.0)
    rng = np.random.default_rng(0)
    errors, snrs = [], []
    for _ in range(24):
        z = rng.normal(size=K)
        z = torch.tensor(z * np.sqrt(K) / np.linalg.norm(z), dtype=torch.float32).unsqueeze(0)
        y = channel(z, 10.0)
        errors.append(float(((y - z) ** 2).mean()))
        snrs.append(channel.last_stats["measured_snr_db"])

    assert float(np.mean(errors)) == pytest.approx(noise_var_per_real(float(np.mean(snrs))), rel=0.4)
    assert channel.loss_rate() == 0.0


def test_uses_calibration_to_pick_gains():
    calibration = Calibration(
        points=[
            CalibrationPoint(tx_vga_db=g, tx_amp=False, rx_lna_db=24.0, rx_vga_db=20.0,
                             measured_snr_db=snr, snr_std_db=0.3, bursts=20, loss_rate=0.0,
                             peak_lsb=50.0, warnings=[])
            for g, snr in [(20.0, 0.0), (30.0, 10.0), (40.0, 20.0)]
        ]
    )
    channel = SDRAnalogChannel(LoopbackBackend(LoopbackConfig()), LinkConfig(), BurstConfig(), calibration)
    z = torch.randn(1, K)
    z = z * (K**0.5) / z.norm(dim=-1, keepdim=True)

    channel(z, 19.0)  # nearest calibrated point is the 20 dB one
    assert channel.last_stats["gains"]["tx_vga_db"] == 40.0
    assert channel.last_stats["calibrated_snr_db"] == 20.0


def test_calibration_excludes_clipped_lossy_and_inconsistent_points():
    calibration = Calibration(points=[
        CalibrationPoint(46.0, False, 24.0, 20.0, 25.0, 0.3, 20, 0.0, 125.0,
                         ["RX near/at saturation"], clipped=True),
        CalibrationPoint(40.0, False, 24.0, 20.0, 20.0, 0.3, 20, 0.0, 76.0, []),
        CalibrationPoint(2.0, False, 24.0, 20.0, -15.0, 0.5, 10, 0.5, 46.0, ["5/20 bursts lost"]),
        CalibrationPoint(44.0, False, 24.0, 20.0, 23.0, 0.3, 20, 0.0, 90.0,
                         ["guard/pilot SNR disagree by 12.0 dB"], snr_disagreement_db=12.0),
    ])
    assert [p.tx_vga_db for p in calibration.usable()] == [40.0]
    assert calibration.nearest(25.0).tx_vga_db == 40.0  # the clipped point is not chosen


def test_a_small_received_level_does_not_disqualify_a_point():
    """The bottom of an SNR sweep is supposed to look small. Rejecting those
    points would throw away exactly the region the experiment is about -- and it
    did: every one of 24 calibration points was rejected for being 'very small'."""
    calibration = Calibration(points=[
        CalibrationPoint(2.0, False, 24.0, 20.0, -11.4, 0.7, 8, 0.0, 4.0,
                         ["RX rms 2.40 LSB is at the converter floor"]),
        CalibrationPoint(36.0, False, 24.0, 20.0, 19.7, 0.3, 8, 0.0, 6.0, []),
    ])

    assert [p.tx_vga_db for p in calibration.usable()] == [2.0, 36.0]
    assert calibration.achievable_range_db() == (-11.4, 19.7)


def test_burst_loss_delivers_noise_only_and_is_counted():
    """A lost burst means the receiver got nothing. Noise-only is the honest
    input to the decoder -- and it must be recorded, not silently retried away."""
    channel = make_channel(tx_vga_db=0.0, path_loss_db=200.0)
    z = torch.randn(1, K)
    z = z * (K**0.5) / z.norm(dim=-1, keepdim=True)

    with pytest.warns(RuntimeWarning, match="burst lost"):
        y = channel(z, 0.0)

    assert y.shape == z.shape
    assert channel.last_stats["burst_lost"] is True
    assert channel.last_stats["attempts"] == LinkConfig().max_retries
    assert channel.loss_rate() == 1.0


def test_payload_accounting_reports_overhead():
    accounting = make_channel().payload_accounting()
    assert accounting["data_symbols"] == 8
    assert accounting["overhead_symbols"] > 100 * accounting["data_symbols"] / 2
    assert accounting["burst_duration_s"] > 0


def test_rejects_k_mismatch():
    channel = SDRAnalogChannel(
        LoopbackBackend(LoopbackConfig()), LinkConfig(), BurstConfig(n_data=4)  # k=8, not 16
    )
    with pytest.raises(ValueError, match="burst carries"):
        channel(torch.randn(1, K), 10.0)


def test_fading_off_by_default_reports_no_gain():
    channel = make_channel()
    z = torch.randn(1, K)
    z = z * (K**0.5) / z.norm(dim=-1, keepdim=True)

    channel(z, 10.0)

    assert channel.last_stats["applied_fading_gain_abs"] is None


def test_fading_gain_varies_and_stays_within_tx_headroom():
    """Rayleigh block fading, one draw per burst -- must never exceed unity
    magnitude (hwlab/channel/sdr_analog.py:_draw_fading_gain), or a deep-fade
    tail could push the fixed TX scale past the DAC's headroom."""
    channel = make_channel(fading=True)
    z = torch.randn(1, K)
    z = z * (K**0.5) / z.norm(dim=-1, keepdim=True)

    gains = []
    for _ in range(20):
        channel(z, 10.0)
        gains.append(channel.last_stats["applied_fading_gain_abs"])

    assert all(g is not None and 0.0 <= g <= 1.0 for g in gains)
    assert len(set(np.round(gains, 6))) > 1  # actually varies burst to burst, not a fixed gain


def test_fading_widens_the_measured_snr_spread():
    """The point of fading: burst-to-burst SNR should vary more than the fixed
    attenuated path, since a deep fade knocks a given burst's SNR down while the
    calibrated nominal setting stays the same."""

    def measured_snr_std(fading: bool) -> float:
        channel = make_channel(fading=fading, seed=1)
        z = torch.randn(1, K)
        z = z * (K**0.5) / z.norm(dim=-1, keepdim=True)
        for _ in range(40):
            channel(z, 10.0)
        return float(np.std([s["measured_snr_db"] for s in channel.stats_log if s["measured_snr_db"] is not None]))

    assert measured_snr_std(fading=True) > measured_snr_std(fading=False)


# --- agent ------------------------------------------------------------------


def _agent(channel, snr_db=10.0):
    encoder = SemanticEncoder(INPUT_DIM, k=K)
    decoder = SemanticDecoder(K, num_types=len(ITEM_TYPES))
    return HardwareSemanticAgent(StubLLM(), encoder, decoder, snr_db, 10, 4, True, channel=channel)


def test_agent_uses_the_hardware_channel_and_records_what_it_did():
    channel = make_channel()
    pool = Pool(counts={"book": 2, "hat": 2, "ball": 2})
    values = Values(per_unit={"book": 20.0, "hat": 20.0, "ball": 10.0})

    turn = _agent(channel).take_turn(pool, values, 0, [], None)

    assert turn.received_offer is not None
    assert len(channel.stats_log) == 1
    assert turn.channel_stats["measured_snr_db"] == channel.stats_log[0]["measured_snr_db"]
    assert turn.channel_stats["k"] == K  # base-class stats preserved


def test_agent_rejects_encoder_burst_size_mismatch():
    channel = SDRAnalogChannel(LoopbackBackend(LoopbackConfig()), LinkConfig(), BurstConfig(n_data=4))
    with pytest.raises(ValueError, match="burst carries"):
        HardwareSemanticAgent(
            StubLLM(), SemanticEncoder(INPUT_DIM, k=K), SemanticDecoder(K), 10.0, 10, 4, True, channel=channel
        )


def test_full_episode_runs_over_the_hardware_channel():
    """run_episode needs no changes at all: the agent protocol is unchanged."""
    channel = make_channel()
    agent_a, agent_b = _agent(channel), _agent(channel)

    record = run_episode(agent_a, agent_b, seed=1_000_000)

    assert record.outcome in ("agreement", "no_deal")
    assert len(channel.stats_log) == len(record.turns)
    assert channel.loss_rate() == 0.0
