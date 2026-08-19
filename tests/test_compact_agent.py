"""CompactAgent is the control that decides whether this project's claim holds.

If it differs from TextAgent in anything but the payload, the decomposition
(source coding vs error correction vs analog latent) is not measuring what it
says it is. These tests pin that equality, and the 16-channel-use budget that
makes compact_fec comparable to the k=16 semantic latent.
"""
from __future__ import annotations

import numpy as np

from airComp.agents.baseline_agent import TextAgent
from airComp.agents.compact_agent import CompactAgent
from airComp.channel.digital import DigitalChannel
from airComp.config import NegotiationConfig
from airComp.env.negotiation import Pool, Values, run_episode

POOL = Pool(counts={"book": 2, "hat": 2, "ball": 1})
VALUES = Values(per_unit={"book": 20.0, "hat": 20.0, "ball": 20.0})


class StubLLM:
    """Records every prompt it is asked for, and replies with valid JSON."""

    def __init__(self, replies=None):
        self.calls = []
        self.replies = replies or ['Sure! {"action": "propose", "counts": {"book": 2, "hat": 1, "ball": 0}, "message": "hi"}']

    def chat(self, sys_prompt, _hist, user_prompt, _max_new_tokens, _temperature):
        self.calls.append((sys_prompt, user_prompt))
        return self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]


def _turn(agent):
    return agent.take_turn(pool=POOL, own_values=VALUES, round_index=0, history=[], standing_offer=None)


def test_the_offer_survives_a_clean_channel():
    for mode in ("raw", "fec"):
        agent = CompactAgent(StubLLM(), DigitalChannel(mode=mode, seed=0), snr_db=40.0)

        turn = _turn(agent)

        assert turn.parse_failed is False
        assert turn.received_offer.action == "propose"
        assert turn.received_offer.counts == {"ball": 0, "book": 2, "hat": 1}


def test_fec_costs_sixteen_channel_uses_and_uncoded_costs_eight():
    """compact_fec must occupy exactly the semantic pipeline's k=16 real channel
    uses, or the headline comparison is not matched."""
    fec = _turn(CompactAgent(StubLLM(), DigitalChannel(mode="fec", seed=0), snr_db=40.0))
    raw = _turn(CompactAgent(StubLLM(), DigitalChannel(mode="raw", seed=0), snr_db=40.0))

    assert fec.channel_stats["n_bits"] == 16
    assert raw.channel_stats["n_bits"] == 8


def test_the_payload_is_two_orders_of_magnitude_smaller_than_the_text_baseline():
    """The confound this whole agent exists to remove, stated as a number."""
    stub = StubLLM()
    compact = _turn(CompactAgent(stub, DigitalChannel(mode="raw", seed=0), snr_db=40.0))
    text = _turn(TextAgent(StubLLM(), DigitalChannel(mode="raw", seed=0), snr_db=40.0))

    assert text.channel_stats["n_bits"] > 50 * compact.channel_stats["n_bits"]


def test_generation_is_identical_to_the_text_baseline():
    """Same prompts, same order -- so with paired seeds the two pipelines make the
    same LLM calls and only the payload differs."""
    compact_llm, text_llm = StubLLM(), StubLLM()

    _turn(CompactAgent(compact_llm, DigitalChannel(mode="fec", seed=0), snr_db=40.0))
    _turn(TextAgent(text_llm, DigitalChannel(mode="raw", seed=0), snr_db=40.0))

    assert compact_llm.calls == text_llm.calls


def test_unparseable_generation_skips_the_channel_like_the_text_baseline():
    """Inherited failure mode, kept explicit: the semantic pipeline has no
    equivalent, which is why the +40 dB ceiling run has to measure it."""
    agent = CompactAgent(StubLLM(replies=["not json at all"]), DigitalChannel(mode="fec", seed=0), snr_db=40.0)

    turn = _turn(agent)

    assert turn.parse_failed is True
    assert turn.channel_stats["skipped_channel"] is True
    assert turn.channel_stats["attempts"] == 3  # the two bounded retries were used


def test_a_destroyed_frame_becomes_an_implicit_reject_not_a_crash():
    outcomes = [
        _turn(CompactAgent(StubLLM(), DigitalChannel(mode="raw", seed=s), snr_db=-20.0)).received_offer
        for s in range(40)
    ]

    assert any(o is None for o in outcomes), "at -20 dB some frames must fall off the codebook"
    assert all(o is None or o.action in ("propose", "accept", "reject") for o in outcomes)


def test_an_episode_runs_end_to_end_without_the_real_model():
    cfg = NegotiationConfig(max_messages=4)
    accept = '{"action": "accept"}'
    propose = '{"action": "propose", "counts": {"book": 1, "hat": 1, "ball": 1}}'
    agent_a = CompactAgent(StubLLM([propose, accept]), DigitalChannel(mode="fec", seed=0), 40.0, cfg.max_messages)
    agent_b = CompactAgent(StubLLM([accept, accept]), DigitalChannel(mode="fec", seed=1), 40.0, cfg.max_messages)

    record = run_episode(agent_a, agent_b, seed=7, cfg=cfg)

    assert record.outcome in ("agreement", "no_deal")
    assert all(t.channel_stats.get("n_bits", 16) == 16 for t in record.turns if "n_bits" in t.channel_stats)


def test_fec_beats_uncoded_on_the_same_frames():
    """Sanity on the decomposition: the FEC step must actually buy something,
    otherwise compact_fec is not the strong baseline it is presented as."""
    survived = {"raw": 0, "fec": 0}
    for mode in survived:
        for seed in range(60):
            turn = _turn(CompactAgent(StubLLM(), DigitalChannel(mode=mode, seed=seed), snr_db=1.0))
            survived[mode] += int(
                turn.received_offer is not None
                and turn.received_offer.counts == {"ball": 0, "book": 2, "hat": 1}
            )

    assert survived["fec"] > survived["raw"]


def test_stats_account_for_what_the_code_corrected():
    seen = [
        _turn(CompactAgent(StubLLM(), DigitalChannel(mode="fec", seed=s), snr_db=3.0)).channel_stats
        for s in range(60)
    ]
    assert all(s["bit_errors"] >= s["residual_bit_errors"] for s in seen)
    assert sum(s["fec_corrected"] for s in seen) > 0
    assert all(s["payload_bits"] == 8 for s in seen)


def test_numpy_frames_are_what_the_channel_receives():
    """Guards against a str/bytes payload sneaking back in."""
    agent = CompactAgent(StubLLM(), DigitalChannel(mode="raw", seed=0), snr_db=40.0)
    captured = {}
    original = agent.channel.transmit_bits

    def spy(bits, snr_db):
        captured["bits"] = bits
        return original(bits, snr_db)

    agent.channel.transmit_bits = spy
    _turn(agent)

    assert isinstance(captured["bits"], np.ndarray)
    assert captured["bits"].dtype == np.uint8
    assert len(captured["bits"]) == 8
