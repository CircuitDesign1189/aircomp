from airComp.agents.parser import parse_offer, parse_offer_with_retries
from airComp.env.negotiation import Pool

POOL = Pool(counts={"book": 2, "hat": 3, "ball": 1})


def test_parse_valid_propose():
    text = (
        'Sure, here it is: {"action": "propose", "counts": {"book": 1, "hat": 1, "ball": 0}, '
        '"message": "fair split"}'
    )
    result = parse_offer(text, POOL)
    assert result.ok
    assert result.offer.action == "propose"
    assert result.offer.counts == {"book": 1, "hat": 1, "ball": 0}


def test_parse_accept_without_counts():
    result = parse_offer('{"action": "accept"}', POOL)
    assert result.ok
    assert result.offer.action == "accept"
    assert result.offer.counts is None


def test_parse_reject():
    result = parse_offer('{"action": "reject"}', POOL)
    assert result.ok
    assert result.offer.action == "reject"


def test_parse_rejects_out_of_range_counts():
    result = parse_offer('{"action": "propose", "counts": {"book": 5, "hat": 0, "ball": 0}}', POOL)
    assert not result.ok


def test_parse_rejects_malformed_json():
    result = parse_offer("I propose to keep some books and hats but not sure how many.", POOL)
    assert not result.ok


def test_parse_rejects_invalid_action():
    result = parse_offer('{"action": "counter", "counts": {"book": 1, "hat": 0, "ball": 0}}', POOL)
    assert not result.ok


def test_parse_propose_missing_counts_fails():
    result = parse_offer('{"action": "propose"}', POOL)
    assert not result.ok


def test_parse_offer_with_retries_succeeds_on_second_attempt():
    responses = ["not json at all", '{"action": "reject"}']

    def generate_fn(attempt, last_error):
        return responses[attempt]

    offer, raw_text, attempts = parse_offer_with_retries(generate_fn, POOL, max_retries=2)
    assert offer is not None
    assert offer.action == "reject"
    assert attempts == 2
    assert raw_text == responses[1]


def test_parse_offer_with_retries_exhausted_returns_none():
    def generate_fn(attempt, last_error):
        return "still not json"

    offer, _raw_text, attempts = parse_offer_with_retries(generate_fn, POOL, max_retries=2)
    assert offer is None
    assert attempts == 3
