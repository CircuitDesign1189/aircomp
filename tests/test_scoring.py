# Copyright (c) 2026 Circuit Design, Inc. and Y.Chosokabe
# SPDX-License-Identifier: Apache-2.0

from airComp.env.negotiation import Pool, Values
from airComp.env.scoring import pareto_frontier


def test_pareto_frontier_contains_full_allocations_when_values_are_aligned():
    pool = Pool(counts={"book": 1, "hat": 1, "ball": 1})
    # Both agents value everything equally -> giving everything to one agent
    # is Pareto-optimal (any split away from an extreme is dominated by moving
    # units to whichever agent still wants more, up to the extremes).
    values_a = Values(per_unit={"book": 100 / 3, "hat": 100 / 3, "ball": 100 / 3})
    values_b = Values(per_unit={"book": 100 / 3, "hat": 100 / 3, "ball": 100 / 3})
    frontier = pareto_frontier(pool, values_a, values_b)
    assert (100.0, 0.0) in frontier
    assert (0.0, 100.0) in frontier


def test_pareto_frontier_favors_integrative_split_when_values_diverge():
    pool = Pool(counts={"book": 2, "hat": 2})
    # A only values books, B only values hats -> the split giving all books to A
    # and all hats to B should reach the frontier with both agents at 100.
    values_a = Values(per_unit={"book": 50.0, "hat": 0.0})
    values_b = Values(per_unit={"book": 0.0, "hat": 50.0})
    frontier = pareto_frontier(pool, values_a, values_b)
    assert (100.0, 100.0) in frontier
