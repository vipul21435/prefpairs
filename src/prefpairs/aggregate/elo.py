"""Elo ratings averaged over seeded permutations of the games.

Online Elo updates after every game:

    E_i = 1 / (1 + base ** ((R_j - R_i) / scale))
    R_i += K * (S_i - E_i),   R_j -= K * (S_i - E_i)

with ``S_i`` = 1, 0.5 or 0. Every update is zero-sum, so the mean rating stays
at ``initial``. Elo was designed for players whose skill changes over time, so a
single pass depends on the order in which games are replayed; for a static
pool of models that dependence is noise. PrefPairs therefore replays the games
in ``n_permutations`` seeded random orders and reports the mean rating (and
the spread across orders, which measures how much order alone moves a rating).

The permutations are drawn from a canonical ordering of the games (sorted by
cluster, items and score), so the result depends only on the multiset of games
and the seed, never on the order the judgments were stored in.
"""

import math
from dataclasses import dataclass

import numpy as np
from pydantic import Field

from prefpairs.aggregate.comparisons import ComparisonSet, FloatArray, IntArray
from prefpairs.schema import Record


class EloConfig(Record):
    """Elo update rule and the permutation averaging around it."""

    k: float = Field(default=4.0, gt=0, allow_inf_nan=False)
    scale: float = Field(default=400.0, gt=0, allow_inf_nan=False)
    base: float = Field(default=10.0, gt=1, allow_inf_nan=False)
    initial: float = Field(default=1000.0, allow_inf_nan=False)
    n_permutations: int = Field(default=100, ge=1, le=100_000)
    seed: int = Field(default=0, ge=0)


@dataclass(frozen=True, slots=True, eq=False)
class EloFit:
    """Mean Elo ratings over permutations of the game order."""

    items: tuple[str, ...]
    ratings: FloatArray
    """Mean rating of each item across the permutations."""
    order_sd: FloatArray
    """Standard deviation of each item's final rating across permutations."""
    n_permutations: int
    n_games: int

    def rating(self, item: str) -> float:
        return float(self.ratings[self.items.index(item)])

    def ranking(self) -> list[str]:
        """Items from highest to lowest mean rating (ties broken by name)."""
        order = sorted(range(len(self.items)), key=lambda i: (-self.ratings[i], self.items[i]))
        return [self.items[i] for i in order]


def canonical_order(comparisons: ComparisonSet) -> IntArray:
    """Game indices sorted by (cluster, first, second, score)."""
    keys = (comparisons.score, comparisons.second, comparisons.first, comparisons.cluster)
    order: IntArray = np.lexsort(keys).astype(np.int64)
    return order


def elo_passes(
    comparisons: ComparisonSet,
    orders: IntArray,
    config: EloConfig,
) -> FloatArray:
    """Replay games once per row of ``orders`` and return final ratings ``(R, n_items)``.

    Each row lists game indices in the order they are played; ``-1`` marks
    padding, so rows of different lengths (bootstrap replicates) share one array.
    """
    n_rows, length = orders.shape
    ratings = np.full((n_rows, comparisons.n_items), config.initial)
    rows = np.arange(n_rows)
    factor = math.log(config.base) / config.scale
    for step in range(length):
        games = orders[:, step]
        live = games >= 0
        r = rows[live]
        g = games[live]
        i = comparisons.first[g]
        j = comparisons.second[g]
        gap = factor * (ratings[r, i] - ratings[r, j])
        expected = 0.5 * (1.0 + np.tanh(0.5 * gap))
        delta = config.k * (comparisons.score[g] - expected)
        ratings[r, i] += delta
        ratings[r, j] -= delta
    return ratings


def permuted_orders(games: IntArray, n_permutations: int, rng: np.random.Generator) -> IntArray:
    """``n_permutations`` independent random orderings of ``games``, one per row."""
    keys = rng.random((n_permutations, len(games)))
    orders: IntArray = games[np.argsort(keys, axis=1)]
    return orders


def fit_elo(comparisons: ComparisonSet, config: EloConfig | None = None) -> EloFit:
    """Mean Elo ratings over ``config.n_permutations`` seeded game orders."""
    config = config or EloConfig()
    if comparisons.n_items == 0:
        msg = "no comparisons to rate"
        raise ValueError(msg)
    rng = np.random.default_rng(config.seed)
    orders = permuted_orders(canonical_order(comparisons), config.n_permutations, rng)
    ratings = elo_passes(comparisons, orders, config)
    return EloFit(
        items=comparisons.items,
        ratings=ratings.mean(axis=0),
        order_sd=ratings.std(axis=0),
        n_permutations=config.n_permutations,
        n_games=comparisons.n_games,
    )
