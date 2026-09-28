"""Bootstrap confidence intervals for Bradley-Terry strengths and Elo ratings.

Judgments on the same prompt are not independent: every annotator sees the same
pair of texts, and a prompt that happens to favour one model favours it in every
judgment on that prompt. Resampling individual judgments treats them as
independent and makes intervals too narrow. The default is therefore a
*cluster* bootstrap: each replicate draws prompts with replacement and keeps
every game of a drawn prompt (as many times as it was drawn). Resampling
judgments is available for comparison.

Intervals are percentile intervals of the replicate estimates. Each replicate
also yields a rank for every item, so the result carries a rank interval too
("second, but could be first to third").

Bradley-Terry and Elo use the same replicate weights for a given seed, so their
intervals describe the same resampled datasets.
"""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from pydantic import Field

from prefpairs.aggregate.bradley_terry import (
    BradleyTerryConfig,
    NotIdentifiableError,
    fit_bradley_terry,
    fit_counts_batch,
    is_strongly_connected,
)
from prefpairs.aggregate.comparisons import ComparisonSet, FloatArray, IntArray
from prefpairs.aggregate.elo import (
    EloConfig,
    canonical_order,
    elo_passes,
    fit_elo,
    permuted_orders,
)
from prefpairs.schema import Record

_CELLS_PER_CHUNK = 4_000_000
"""Upper bound on array cells materialised at once (about 32 MB of float64)."""


class ResampleUnit(StrEnum):
    """What a bootstrap replicate draws with replacement."""

    PROMPT = "prompt"
    JUDGMENT = "judgment"


class BootstrapConfig(Record):
    """Replicates, resampling unit, interval level and seed."""

    n_replicates: int = Field(default=500, ge=10, le=100_000)
    unit: ResampleUnit = ResampleUnit.PROMPT
    level: float = Field(default=0.95, gt=0, lt=1)
    seed: int = Field(default=0, ge=0)
    elo_permutations: int = Field(default=10, ge=1, le=10_000)
    """Game orders averaged inside each Elo replicate."""


@dataclass(frozen=True, slots=True, eq=False)
class BootstrapResult:
    """Point estimates on the full data and their bootstrap distribution."""

    items: tuple[str, ...]
    estimate: FloatArray
    replicates: FloatArray
    """Shape ``(n_replicates, n_items)``."""
    lower: FloatArray
    upper: FloatArray
    rank_lower: IntArray
    """Best (smallest) rank inside the interval; rank 1 is the strongest item."""
    rank_upper: IntArray
    level: float
    unit: ResampleUnit
    unconverged: int = 0
    """Replicates whose fit hit the iteration cap (0 for Elo)."""

    @property
    def n_replicates(self) -> int:
        return int(self.replicates.shape[0])

    def interval(self, item: str) -> tuple[float, float]:
        index = self.items.index(item)
        return float(self.lower[index]), float(self.upper[index])


def _rng(seed: int, stream: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, stream]))


def replicate_weights(comparisons: ComparisonSet, config: BootstrapConfig) -> IntArray:
    """How many times each game appears in each replicate, shape ``(B, n_games)``."""
    rng = _rng(config.seed, 1)
    b = config.n_replicates
    if config.unit is ResampleUnit.PROMPT:
        units, of_game = comparisons.n_clusters, comparisons.cluster
    else:
        units, of_game = comparisons.n_games, np.arange(comparisons.n_games)
    draws = rng.integers(0, units, size=(b, units))
    counts = np.zeros((b, units), dtype=np.int64)
    np.add.at(counts, (np.repeat(np.arange(b), units), draws.ravel()), 1)
    weights: IntArray = counts[:, of_game]
    return weights


def percentile_interval(replicates: FloatArray, level: float) -> tuple[FloatArray, FloatArray]:
    """Equal-tailed percentile interval of each column."""
    alpha = 1.0 - level
    lower, upper = np.quantile(replicates, [alpha / 2, 1 - alpha / 2], axis=0)
    return lower, upper


def ranks(values: FloatArray) -> IntArray:
    """Competition ranks per row, highest value first: 1 + number strictly greater."""
    greater = (values[..., None, :] > values[..., :, None]).sum(axis=-1)
    result: IntArray = (1 + greater).astype(np.int64)
    return result


def _summarise(
    comparisons: ComparisonSet,
    estimate: FloatArray,
    replicates: FloatArray,
    config: BootstrapConfig,
    unconverged: int = 0,
) -> BootstrapResult:
    lower, upper = percentile_interval(replicates, config.level)
    rank_lo, rank_hi = percentile_interval(ranks(replicates).astype(np.float64), config.level)
    return BootstrapResult(
        items=comparisons.items,
        estimate=estimate,
        replicates=replicates,
        lower=lower,
        upper=upper,
        rank_lower=np.floor(rank_lo).astype(np.int64),
        rank_upper=np.ceil(rank_hi).astype(np.int64),
        level=config.level,
        unit=config.unit,
        unconverged=unconverged,
    )


def _check_nonempty(comparisons: ComparisonSet) -> None:
    if comparisons.n_games == 0:
        msg = "no comparisons to bootstrap"
        raise ValueError(msg)


def bootstrap_bradley_terry(
    comparisons: ComparisonSet,
    bt_config: BradleyTerryConfig | None = None,
    config: BootstrapConfig | None = None,
) -> BootstrapResult:
    """Percentile intervals for centred Bradley-Terry log-strengths."""
    bt_config = bt_config or BradleyTerryConfig()
    config = config or BootstrapConfig()
    _check_nonempty(comparisons)
    estimate = fit_bradley_terry(comparisons, bt_config).log_strengths
    weights = replicate_weights(comparisons, config).astype(np.float64)
    n = comparisons.n_items
    first, second, score = comparisons.first, comparisons.second, comparisons.score
    chunk = max(1, _CELLS_PER_CHUNK // (n * n))
    out = np.empty((config.n_replicates, n))
    unconverged = 0
    for start in range(0, config.n_replicates, chunk):
        w = weights[start : start + chunk]
        size = w.shape[0]
        wins = np.zeros((size, n, n))
        games = np.zeros((size, n, n))
        rep = np.arange(size)[:, None]
        np.add.at(wins, (rep, first[None, :], second[None, :]), w * score)
        np.add.at(wins, (rep, second[None, :], first[None, :]), w * (1.0 - score))
        np.add.at(games, (rep, first[None, :], second[None, :]), w)
        games = games + games.transpose(0, 2, 1)
        if bt_config.prior == 0:
            for offset in range(size):
                if not is_strongly_connected(wins[offset]):
                    msg = (
                        f"bootstrap replicate {start + offset} has a comparison graph that "
                        "is not strongly connected, so its MLE does not exist; use prior > 0"
                    )
                    raise NotIdentifiableError(msg)
        result = fit_counts_batch(wins, games, bt_config)
        out[start : start + size] = result.log_p - result.log_p.mean(axis=1, keepdims=True)
        unconverged += result.unconverged
    return _summarise(comparisons, estimate, out, config, unconverged)


def bootstrap_elo(
    comparisons: ComparisonSet,
    elo_config: EloConfig | None = None,
    config: BootstrapConfig | None = None,
) -> BootstrapResult:
    """Percentile intervals for permutation-averaged Elo ratings.

    Each replicate averages ``config.elo_permutations`` game orders; the point
    estimate uses ``elo_config.n_permutations`` on the full data.
    """
    elo_config = elo_config or EloConfig()
    config = config or BootstrapConfig()
    _check_nonempty(comparisons)
    estimate = fit_elo(comparisons, elo_config).ratings
    weights = replicate_weights(comparisons, config)
    canonical = canonical_order(comparisons)
    rng = _rng(config.seed, 2)
    perms = config.elo_permutations
    longest = int(weights.sum(axis=1).max())
    chunk = max(1, _CELLS_PER_CHUNK // max(1, perms * longest))
    out = np.empty((config.n_replicates, comparisons.n_items))
    for start in range(0, config.n_replicates, chunk):
        block = weights[start : start + chunk]
        orders = np.full((block.shape[0] * perms, longest), -1, dtype=np.int64)
        for offset, row in enumerate(block):
            games = np.repeat(canonical, row[canonical])
            rows = slice(offset * perms, (offset + 1) * perms)
            orders[rows, : len(games)] = permuted_orders(games, perms, rng)
        ratings = elo_passes(comparisons, orders, elo_config)
        out[start : start + block.shape[0]] = ratings.reshape(
            block.shape[0], perms, comparisons.n_items
        ).mean(axis=1)
    return _summarise(comparisons, estimate, out, config)
