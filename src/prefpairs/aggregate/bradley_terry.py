"""Bradley-Terry strengths fitted with the MM algorithm (Hunter 2004).

Model: item i beats item j with probability ``p_i / (p_i + p_j)``, or
``sigmoid(theta_i - theta_j)`` with ``theta = log p``. A tie is half a win for
each side, so the fitted objective is the binomial log-likelihood with
fractional counts:

    l(p) = sum_{i != j} w_ij * log(p_i / (p_i + p_j))

where ``w_ij`` is i's total score against j.

**Identifiability.** The maximum-likelihood estimate exists only when the
graph "i scored against j" is strongly connected; an item that never loses
would otherwise run off to infinite strength. PrefPairs adds a pseudo-count
prior: every item plays ``prior`` virtual wins and ``prior`` virtual losses
against a fixed reference item of strength 1. The penalised objective then has
a unique maximiser for any data, shrinks thinly observed items towards the
middle, and keeps the scale fixed. With ``prior=0`` the plain MLE is fitted and
a disconnected comparison graph raises ``NotIdentifiableError``.

**MM update.** Minorising each ``-log(p_i + p_j)`` term by its tangent gives

    p_i <- (W_i + prior) / (sum_j n_ij / (p_i + p_j) + 2 * prior / (p_i + 1))

with ``W_i`` the total score of i and ``n_ij`` the games between i and j. Each
update cannot decrease the penalised log-likelihood; the fit records it after
every iteration so tests can check that. Reported strengths are
``log p`` centred to mean zero; win probabilities do not depend on the centring.

The core update is vectorised over a leading batch axis so the bootstrap can
fit hundreds of replicates at once.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from pydantic import Field

from prefpairs.aggregate.comparisons import ComparisonSet, FloatArray
from prefpairs.schema import Record


class NotIdentifiableError(ValueError):
    """The plain MLE does not exist for this comparison graph (use prior > 0)."""


class BradleyTerryConfig(Record):
    """Settings of the MM fit."""

    prior: float = Field(default=0.1, ge=0, allow_inf_nan=False)
    """Virtual wins and losses of every item against a reference of strength 1."""
    tol: float = Field(default=1e-9, gt=0, allow_inf_nan=False)
    """Stop when no log-strength moves by more than this in one iteration."""
    max_iter: int = Field(default=10_000, ge=1)


@dataclass(frozen=True, slots=True, eq=False)
class BradleyTerryFit:
    """Fitted strengths and the evidence behind them."""

    items: tuple[str, ...]
    log_strengths: FloatArray
    """``theta``, centred to mean zero."""
    reference_log_strength: float
    """Log-strength of the prior's virtual reference item on the centred scale (NaN if no prior)."""
    wins: FloatArray
    games: FloatArray
    prior: float
    iterations: int
    converged: bool
    log_likelihood: tuple[float, ...]
    """Penalised log-likelihood after each iteration (non-decreasing)."""

    def strength(self, item: str) -> float:
        return float(self.log_strengths[self.items.index(item)])

    def ranking(self) -> list[str]:
        """Items from strongest to weakest (ties broken by name)."""
        theta = self.log_strengths
        order = sorted(range(len(self.items)), key=lambda i: (-theta[i], self.items[i]))
        return [self.items[i] for i in order]

    def win_probability(self, winner: str, loser: str) -> float:
        """Model probability that ``winner`` beats ``loser``."""
        return sigmoid(self.strength(winner) - self.strength(loser))

    def predicted_win_rates(self) -> FloatArray:
        """``P[i, j]`` = model probability that i beats j; NaN on the diagonal."""
        return predicted_win_rates(self.log_strengths)


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def predicted_win_rates(log_strengths: FloatArray) -> FloatArray:
    """Matrix of ``sigmoid(theta_i - theta_j)`` with NaN on the diagonal."""
    theta = np.asarray(log_strengths, dtype=np.float64)
    diff = theta[:, None] - theta[None, :]
    rates = 0.5 * (1.0 + np.tanh(0.5 * diff))  # stable sigmoid
    np.fill_diagonal(rates, np.nan)
    return rates


def is_strongly_connected(wins: FloatArray) -> bool:
    """True if every item can reach every other along "scored against" edges."""
    n = wins.shape[0]
    if n <= 1:
        return True
    edges = wins > 0
    for adjacency in (edges, edges.T):
        seen = np.zeros(n, dtype=bool)
        seen[0] = True
        frontier = seen.copy()
        while frontier.any():
            frontier = adjacency[frontier].any(axis=0) & ~seen
            seen |= frontier
        if not seen.all():
            return False
    return True


def penalised_log_likelihood(log_p: FloatArray, wins: FloatArray, prior: float) -> FloatArray:
    """Objective of the MM fit for a batch: shapes ``(B, n)`` and ``(B, n, n)``."""
    diff = log_p[:, :, None] - log_p[:, None, :]
    # log(p_i / (p_i + p_j)) = -log(1 + exp(theta_j - theta_i))
    log_win = -np.logaddexp(0.0, -diff)
    value: FloatArray = np.einsum("bij,bij->b", wins, log_win)
    if prior > 0:
        value = value + prior * (-np.logaddexp(0.0, -log_p) - np.logaddexp(0.0, log_p)).sum(axis=1)
    return value


@dataclass(frozen=True, slots=True)
class _MMResult:
    log_p: FloatArray
    iterations: int
    unconverged: int
    """Number of batch rows still moving by more than ``tol`` at the cap."""
    trace: tuple[float, ...]

    @property
    def converged(self) -> bool:
        return self.unconverged == 0


_SHIFT_MAX_STEPS = 100
_SHIFT_TOL = 1e-13


def best_shift(log_p: FloatArray) -> FloatArray:
    """Shift ``u`` per row that maximises the prior term of ``log_p + u``.

    The data log-likelihood depends only on differences of log-strengths, so
    adding the same ``u`` to every item changes only the prior's virtual games
    against the reference. Their log-likelihood is concave in ``u`` with
    derivative ``-prior * sum_i tanh((u + theta_i) / 2)``, so the maximiser is
    the root of that increasing function. It is found by Newton's method kept
    inside a shrinking bracket (bisection when a Newton step would leave it).
    Without this step MM moves the common scale very slowly, because only the
    weak prior pins it; with it, each iteration is still a non-decreasing step
    on the penalised objective.
    """
    low = -log_p.max(axis=1) - 1.0
    high = -log_p.min(axis=1) + 1.0
    u = np.clip(-np.median(log_p, axis=1), low, high)
    for _ in range(_SHIFT_MAX_STEPS):  # pragma: no branch - bisection converges within 100
        t = np.tanh(0.5 * (u[:, None] + log_p))
        f = t.sum(axis=1)
        slope = 0.5 * (1.0 - t * t).sum(axis=1)
        high = np.where(f > 0, u, high)
        low = np.where(f > 0, low, u)
        with np.errstate(divide="ignore", invalid="ignore"):
            newton = u - f / slope
        inside = (newton >= low) & (newton <= high)
        step = np.where(inside, newton, 0.5 * (low + high)) - u
        u = u + step
        if np.all(np.abs(step) < _SHIFT_TOL):
            break
    shift: FloatArray = u
    return shift


def fit_counts_batch(
    wins: FloatArray,
    games: FloatArray,
    config: BradleyTerryConfig,
    *,
    record_trace: bool = False,
) -> _MMResult:
    """Run MM on a batch of count matrices ``wins``/``games`` of shape ``(B, n, n)``.

    Returns uncentred log-strengths ``(B, n)`` on the scale where the prior's
    reference item has strength 1 (or with geometric mean 1 when prior is 0).
    Stops when every replicate has converged or at ``max_iter``; the trace
    follows the first replicate.
    """
    prior = config.prior
    total = wins.sum(axis=2)
    batch, n = total.shape
    log_p = np.zeros((batch, n))
    trace: list[float] = []
    active = np.ones(batch, dtype=bool)
    for iterations in range(1, config.max_iter + 1):
        current = log_p[active]
        pa = np.exp(current)
        denom = (games[active] / (pa[:, :, None] + pa[:, None, :])).sum(axis=2)
        if prior > 0:
            denom = denom + 2.0 * prior / (pa + 1.0)
        updated = np.log(total[active] + prior) - np.log(denom)
        if prior > 0:
            updated = updated + best_shift(updated)[:, None]
        else:
            updated = updated - updated.mean(axis=1, keepdims=True)
        change = np.abs(updated - current).max(axis=1)
        log_p[active] = updated
        if record_trace:
            trace.append(float(penalised_log_likelihood(log_p[:1], wins[:1], prior)[0]))
        done = np.flatnonzero(active)[change < config.tol]
        active[done] = False
        if not active.any():
            return _MMResult(log_p, iterations, 0, tuple(trace))
    return _MMResult(log_p, config.max_iter, int(active.sum()), tuple(trace))


def fit_bradley_terry(
    comparisons: ComparisonSet,
    config: BradleyTerryConfig | None = None,
    *,
    weights: FloatArray | None = None,
) -> BradleyTerryFit:
    """Fit strengths to a comparison set (optionally with a weight per game)."""
    config = config or BradleyTerryConfig()
    if comparisons.n_items == 0:
        msg = "no comparisons to fit"
        raise ValueError(msg)
    wins, games = comparisons.counts(weights)
    if config.prior == 0 and not is_strongly_connected(wins):
        msg = (
            "the comparison graph is not strongly connected, so the Bradley-Terry MLE "
            "does not exist (some item never loses or never wins against the rest); "
            "fit with prior > 0"
        )
        raise NotIdentifiableError(msg)
    result = fit_counts_batch(wins[None], games[None], config, record_trace=True)
    log_p = result.log_p[0]
    centre = float(log_p.mean())
    return BradleyTerryFit(
        items=comparisons.items,
        log_strengths=log_p - centre,
        reference_log_strength=-centre if config.prior > 0 else math.nan,
        wins=wins,
        games=games,
        prior=config.prior,
        iterations=result.iterations,
        converged=result.converged,
        log_likelihood=result.trace,
    )


def sample_bradley_terry(
    log_strengths: Sequence[float],
    *,
    n_clusters: int,
    games_per_pair: int = 1,
    cluster_sd: float = 0.0,
    tie_rate: float = 0.0,
    seed: int = 0,
    names: Sequence[str] | None = None,
) -> ComparisonSet:
    """Draw games from a known Bradley-Terry model, to validate estimators.

    In every cluster each pair of items plays ``games_per_pair`` games. With
    ``cluster_sd > 0`` each cluster perturbs every item's strength by
    ``N(0, cluster_sd)``, which correlates games within a cluster the way
    judgments on one prompt are correlated. A game is a tie with probability
    ``tie_rate`` and otherwise follows the model.
    """
    theta = np.asarray(log_strengths, dtype=np.float64)
    n = len(theta)
    labels = list(names) if names is not None else [f"item-{i:02d}" for i in range(n)]
    if len(labels) != n or len(set(labels)) != n:
        msg = "names must be distinct and match log_strengths"
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    width = max(4, len(str(n_clusters)))
    rows: list[tuple[str, str, str, float]] = []
    for c in range(n_clusters):
        local = theta + (rng.normal(0.0, cluster_sd, n) if cluster_sd > 0 else 0.0)
        for i in range(n):
            for j in range(i + 1, n):
                p_win = sigmoid(float(local[i] - local[j]))
                for _ in range(games_per_pair):
                    if tie_rate > 0 and rng.random() < tie_rate:
                        score = 0.5
                    else:
                        score = 1.0 if rng.random() < p_win else 0.0
                    rows.append((f"c{c:0{width}d}", labels[i], labels[j], score))
    return ComparisonSet.from_outcomes(rows)
