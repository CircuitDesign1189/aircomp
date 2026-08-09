"""Social welfare and Pareto-efficiency metrics for negotiation episodes."""
from __future__ import annotations

import itertools

from airComp.env.negotiation import EpisodeRecord, Pool, Values


def enumerate_splits(pool: Pool) -> list:
    types = list(pool.counts.keys())
    ranges = [range(0, pool.counts[t] + 1) for t in types]
    return [dict(zip(types, combo)) for combo in itertools.product(*ranges)]


def pareto_frontier(pool: Pool, values_a: Values, values_b: Values) -> list:
    """Return the set of non-dominated (utility_a, utility_b) points over all feasible splits."""
    points = []
    for counts_a in enumerate_splits(pool):
        counts_b = {t: pool.counts[t] - counts_a[t] for t in pool.counts}
        points.append((values_a.utility(counts_a), values_b.utility(counts_b)))

    frontier = []
    for i, p in enumerate(points):
        dominated = any(
            j != i and q[0] >= p[0] and q[1] >= p[1] and q != p
            for j, q in enumerate(points)
        )
        if not dominated:
            frontier.append(p)
    return sorted(set(frontier))


def social_welfare(record: EpisodeRecord) -> float:
    return record.utility_a + record.utility_b


def pareto_efficiency(record: EpisodeRecord) -> float:
    """Ratio of achieved social welfare to the max welfare achievable on the Pareto frontier.

    Returns 0.0 for no-deal episodes (utilities are 0 by construction).
    """
    if record.outcome != "agreement":
        return 0.0
    frontier = pareto_frontier(record.pool, record.values_a, record.values_b)
    max_welfare = max(ua + ub for ua, ub in frontier)
    if max_welfare <= 0:
        return 1.0
    return social_welfare(record) / max_welfare
