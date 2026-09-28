"""Empirical and model-implied win-rate matrices.

``empirical[i, j]`` is i's share of the points in its games against j (a tie is
half a point), NaN when the two never met. ``predicted[i, j]`` is the
Bradley-Terry probability that i beats j, defined for every pair. Comparing the
two on the pairs that were played shows where a single strength per item does
not describe the data (for example, a model that is strong in general but loses
to one particular rival).
"""

from dataclasses import dataclass

import numpy as np

from prefpairs.aggregate.bradley_terry import BradleyTerryFit, predicted_win_rates
from prefpairs.aggregate.comparisons import ComparisonSet, FloatArray


@dataclass(frozen=True, slots=True, eq=False)
class WinRates:
    items: tuple[str, ...]
    games: FloatArray
    empirical: FloatArray
    predicted: FloatArray

    @property
    def fit_gap(self) -> float:
        """Game-weighted mean of ``|empirical - predicted|`` over pairs that were played."""
        played = self.games > 0
        if not played.any():
            return 0.0
        gaps = np.abs(self.empirical[played] - self.predicted[played])
        return float(np.average(gaps, weights=self.games[played]))


def empirical_win_rates(comparisons: ComparisonSet) -> tuple[FloatArray, FloatArray]:
    """``(rates, games)``; rates are NaN on the diagonal and for pairs never played."""
    wins, games = comparisons.counts()
    rates = np.full_like(wins, np.nan)
    np.divide(wins, games, out=rates, where=games > 0)
    return rates, games


def win_rates(comparisons: ComparisonSet, fit: BradleyTerryFit) -> WinRates:
    """Both matrices for the items of ``comparisons`` (which ``fit`` was fitted to)."""
    if fit.items != comparisons.items:
        msg = "the fit and the comparisons cover different items"
        raise ValueError(msg)
    empirical, games = empirical_win_rates(comparisons)
    return WinRates(
        items=comparisons.items,
        games=games,
        empirical=empirical,
        predicted=predicted_win_rates(fit.log_strengths),
    )
