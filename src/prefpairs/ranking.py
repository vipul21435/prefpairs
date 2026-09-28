"""Ranking report: fit a model to stored judgments and attach bootstrap intervals.

This is the glue behind ``prefpairs rank``. It reads judgments from a store,
turns them into games (``build_comparisons``), fits Bradley-Terry or Elo,
resamples for percentile intervals, and, when the data was simulated, scores
the estimated order against the true one with Kendall's tau.
"""

from enum import StrEnum

import numpy as np

from prefpairs.aggregate import (
    BootstrapConfig,
    BootstrapResult,
    BradleyTerryConfig,
    ComparisonOptions,
    ComparisonSet,
    EloConfig,
    Level,
    bootstrap_bradley_terry,
    bootstrap_elo,
    build_comparisons,
    fit_bradley_terry,
    win_rates,
)
from prefpairs.schema import Record
from prefpairs.simulate import SimulationTruth
from prefpairs.store import Store


class NothingToRankError(ValueError):
    """No judgment survived the selection, so there is nothing to fit."""


class Method(StrEnum):
    """Aggregation model."""

    BRADLEY_TERRY = "bt"
    ELO = "elo"


_ESTIMATE_LABEL = {Method.BRADLEY_TERRY: "log-strength", Method.ELO: "Elo rating"}
_METHOD_NAME = {Method.BRADLEY_TERRY: "Bradley-Terry", Method.ELO: "Elo"}


class RankRow(Record):
    """One ranked item with its interval and the evidence behind it."""

    rank: int
    item: str
    estimate: float
    lower: float
    upper: float
    rank_lower: int
    rank_upper: int
    score: float
    """Points won (a tie is half a point)."""
    games: int
    true_rank: int | None = None


class RankReport(Record):
    """Everything ``prefpairs rank`` prints, in JSON-ready form."""

    method: Method
    level: Level
    unit: str
    confidence: float
    n_replicates: int
    seed: int
    n_games: int
    n_prompts: int
    dropped: dict[str, int]
    excluded_annotators: tuple[str, ...]
    rows: tuple[RankRow, ...]
    unconverged_replicates: int
    fit_gap: float | None
    """Bradley-Terry only: game-weighted mean |empirical - predicted| win rate."""
    true_order: tuple[str, ...] | None
    kendall_tau: float | None


def kendall_tau(estimate: dict[str, float], truth: dict[str, float]) -> float | None:
    """Kendall's tau-a between two scorings over their common keys.

    Pairs tied in either scoring count as neither concordant nor discordant.
    None when fewer than two keys are shared.
    """
    keys = sorted(estimate.keys() & truth.keys())
    n = len(keys)
    if n < 2:
        return None
    x = np.array([estimate[k] for k in keys])
    y = np.array([truth[k] for k in keys])
    signs = np.sign(x[:, None] - x[None, :]) * np.sign(y[:, None] - y[None, :])
    upper = np.triu_indices(n, k=1)
    return float(signs[upper].sum() / (n * (n - 1) / 2))


def truth_scores(truth: SimulationTruth, level: Level) -> dict[str, float]:
    """True latent value of each item at ``level``."""
    if level is Level.MODEL:
        return dict(truth.model_strengths)
    return dict(truth.response_quality)


def _order(values: dict[str, float]) -> list[str]:
    return sorted(values, key=lambda item: (-values[item], item))


def _rows(
    comparisons: ComparisonSet,
    result: BootstrapResult,
    truth: dict[str, float] | None,
) -> tuple[RankRow, ...]:
    wins, games = comparisons.counts()
    score = wins.sum(axis=1)
    played = games.sum(axis=1)
    estimates = {item: float(v) for item, v in zip(result.items, result.estimate, strict=True)}
    true_rank = {item: i + 1 for i, item in enumerate(_order(truth))} if truth else {}
    rows = []
    for rank, item in enumerate(_order(estimates), start=1):
        i = result.items.index(item)
        rows.append(
            RankRow(
                rank=rank,
                item=item,
                estimate=float(result.estimate[i]),
                lower=float(result.lower[i]),
                upper=float(result.upper[i]),
                rank_lower=int(result.rank_lower[i]),
                rank_upper=int(result.rank_upper[i]),
                score=float(score[i]),
                games=round(float(played[i])),
                true_rank=true_rank.get(item),
            )
        )
    return tuple(rows)


def rank_store(
    store: Store,
    *,
    method: Method = Method.BRADLEY_TERRY,
    options: ComparisonOptions | None = None,
    bootstrap: BootstrapConfig | None = None,
    bt_config: BradleyTerryConfig | None = None,
    elo_config: EloConfig | None = None,
    truth: SimulationTruth | None = None,
) -> RankReport:
    """Rank the items of a store with bootstrap intervals.

    Raises NothingToRankError when no judgment survives the selection.
    """
    options = options or ComparisonOptions()
    bootstrap = bootstrap or BootstrapConfig()
    comparisons = build_comparisons(
        store.pairwise(), store.responses(), options, ranked=store.ranked()
    )
    if comparisons.n_games == 0:
        msg = "no judgments to rank after filtering (dropped: "
        msg += ", ".join(f"{k} {v}" for k, v in comparisons.dropped.items()) or "none"
        raise NothingToRankError(msg + ")")
    fit_gap: float | None = None
    if method is Method.BRADLEY_TERRY:
        bt_config = bt_config or BradleyTerryConfig()
        result = bootstrap_bradley_terry(comparisons, bt_config, bootstrap)
        fit_gap = win_rates(comparisons, fit_bradley_terry(comparisons, bt_config)).fit_gap
    else:
        result = bootstrap_elo(comparisons, elo_config, bootstrap)
    true_values = truth_scores(truth, options.level) if truth is not None else None
    tau = None
    if true_values is not None:
        estimates = dict(zip(result.items, map(float, result.estimate), strict=True))
        tau = kendall_tau(estimates, true_values)
    return RankReport(
        method=method,
        level=options.level,
        unit=bootstrap.unit.value,
        confidence=bootstrap.level,
        n_replicates=bootstrap.n_replicates,
        seed=bootstrap.seed,
        n_games=comparisons.n_games,
        n_prompts=comparisons.n_clusters,
        dropped=dict(comparisons.dropped),
        excluded_annotators=options.exclude_annotators,
        rows=_rows(comparisons, result, true_values),
        unconverged_replicates=result.unconverged,
        fit_gap=fit_gap,
        true_order=tuple(_order(true_values)) if true_values is not None else None,
        kendall_tau=tau,
    )


def render_rank_report(report: RankReport) -> str:
    """Plain-text table, stable enough to paste into a README."""
    name = _METHOD_NAME[report.method]
    label = _ESTIMATE_LABEL[report.method]
    pct = f"{report.confidence:.0%}"
    lines = [
        f"{name} ranking of {len(report.rows)} {report.level.value}s from "
        f"{report.n_games} games on {report.n_prompts} prompts",
        f"{pct} intervals from {report.n_replicates} {report.unit}-bootstrap replicates "
        f"(seed {report.seed})",
    ]
    if report.dropped:
        lines.append("not counted: " + ", ".join(f"{k} {v}" for k, v in report.dropped.items()))
    if report.excluded_annotators:
        lines.append("excluded annotators: " + ", ".join(report.excluded_annotators))
    width = max(len(report.level.value), *(len(row.item) for row in report.rows))
    has_truth = report.true_order is not None
    header = (
        f"{'rank':>4}  {report.level.value:<{width}}  {label:>12}  {pct + ' CI':^22}  "
        f"{'rank range':<10}  {'score':>7}  {'games':>5}"
    )
    lines += ["", header + ("  true rank" if has_truth else "")]
    for row in report.rows:
        interval = f"[{row.lower:8.3f}, {row.upper:8.3f}]"
        ranks = f"{row.rank_lower}-{row.rank_upper}"
        line = (
            f"{row.rank:>4}  {row.item:<{width}}  {row.estimate:12.3f}  {interval:>20}  "
            f"{ranks:<10}  {row.score:7.1f}  {row.games:5d}"
        )
        if has_truth:
            line += f"  {row.true_rank if row.true_rank is not None else '-':>9}"
        lines.append(line)
    lines.append("")
    if report.fit_gap is not None:
        lines.append(
            f"fit gap (mean |empirical - predicted| win rate on played pairs): {report.fit_gap:.3f}"
        )
    if report.unconverged_replicates:
        lines.append(f"warning: {report.unconverged_replicates} replicates hit the iteration cap")
    if report.true_order is not None:
        tau = "n/a" if report.kendall_tau is None else f"{report.kendall_tau:.3f}"
        if len(report.true_order) <= 12:
            lines.append(f"true order: {' > '.join(report.true_order)}")
        lines.append(f"Kendall tau against the true order: {tau}")
    return "\n".join(lines).rstrip("\n")
