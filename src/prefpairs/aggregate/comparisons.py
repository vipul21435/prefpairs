"""Pairwise outcomes as arrays: the common input of every aggregation model.

A ``ComparisonSet`` is a list of games between *items* (models, or individual
responses), each with a score for the first item (1 win, 0.5 tie, 0 loss) and
the *cluster* it belongs to (the prompt). Games are oriented canonically: the
first item is the one that sorts first, so the displayed left/right order never
reaches the models, and two judgments of the same pair shown in opposite orders
are the same kind of game.

Which judgments count is an explicit, recorded decision (``ComparisonOptions``):

* Only ``regular`` pairs by default. Gold pairs are a quality check shown to
  every annotator (a dozen pairs would carry as much weight as hundreds of
  regular ones), and controls repeat a judgment the annotator already made.
* Skips carry no preference and are dropped; ties are kept as half a win.
* A comparison between two responses of the same model says nothing about
  models and is dropped at model level.
* Rankings can be expanded into their implied pairs, but they are off by
  default: a ranking of k responses yields k(k-1)/2 correlated games.

Every dropped judgment is counted by reason (an excluded ranking counts once,
a same-model pair inside a kept ranking counts per pair), so a ranking can say
what it ignored.
"""

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Self

import numpy as np
from pydantic import field_validator

from prefpairs.schema import (
    Choice,
    Id,
    PairKind,
    PairwiseJudgment,
    RankedJudgment,
    Record,
    Response,
)

type FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
type IntArray = np.ndarray[tuple[int, ...], np.dtype[np.int64]]


class Level(StrEnum):
    """What is being ranked."""

    MODEL = "model"
    RESPONSE = "response"


class DropReason(StrEnum):
    """Why a judgment did not become a game."""

    SKIP = "skip"
    PAIR_KIND = "pair_kind"
    EXCLUDED_ANNOTATOR = "excluded_annotator"
    SAME_ITEM = "same_item"


class ComparisonOptions(Record):
    """Which judgments become games, and at what level."""

    level: Level = Level.MODEL
    pair_kinds: tuple[PairKind, ...] = (PairKind.REGULAR,)
    exclude_annotators: tuple[Id, ...] = ()
    include_ranked: bool = False

    @field_validator("pair_kinds")
    @classmethod
    def _kinds_sorted(cls, value: tuple[PairKind, ...]) -> tuple[PairKind, ...]:
        if not value:
            msg = "at least one pair kind is required"
            raise ValueError(msg)
        return tuple(sorted(set(value)))

    @field_validator("exclude_annotators")
    @classmethod
    def _annotators_sorted(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(sorted(set(value)))


@dataclass(frozen=True, slots=True, eq=False)
class ComparisonSet:
    """Games between items, grouped into clusters.

    ``first[g]`` and ``second[g]`` index ``items`` with ``first < second``;
    ``score[g]`` is the first item's result; ``cluster[g]`` indexes ``clusters``.
    Scores are 0, 0.5 or 1, unless ``fractional`` marks them as expected scores
    in [0, 1] (a population target rather than observed data).
    """

    items: tuple[str, ...]
    first: IntArray
    second: IntArray
    score: FloatArray
    cluster: IntArray
    clusters: tuple[str, ...]
    dropped: Mapping[str, int] = field(default_factory=dict)
    fractional: bool = False

    def __post_init__(self) -> None:
        n = len(self.score)
        if not (len(self.first) == len(self.second) == len(self.cluster) == n):
            msg = "first, second, score and cluster must have the same length"
            raise ValueError(msg)
        if n and not np.all(self.first < self.second):
            msg = "games must be oriented with first < second"
            raise ValueError(msg)
        if n and (self.first.min() < 0 or self.second.max() >= len(self.items)):
            msg = "item index out of range"
            raise ValueError(msg)
        if n and (self.cluster.min() < 0 or self.cluster.max() >= len(self.clusters)):
            msg = "cluster index out of range"
            raise ValueError(msg)
        if self.fractional:
            if n and not (self.score.min() >= 0.0 and self.score.max() <= 1.0):
                msg = "expected scores must lie in [0, 1]"
                raise ValueError(msg)
        elif not np.all(np.isin(self.score, (0.0, 0.5, 1.0))):
            msg = "scores must be 0, 0.5 or 1"
            raise ValueError(msg)

    @classmethod
    def from_outcomes(
        cls,
        outcomes: Iterable[tuple[str, str, str, float]],
        *,
        dropped: Mapping[str, int] | None = None,
        fractional: bool = False,
    ) -> Self:
        """Build from ``(cluster, item_x, item_y, score_of_x)`` rows.

        Items and clusters are indexed in sorted order and each game is flipped
        if needed so that its first item sorts first. ``fractional`` admits
        expected scores anywhere in [0, 1].
        """
        rows = list(outcomes)
        items = tuple(sorted({x for _, x, _, _ in rows} | {y for _, _, y, _ in rows}))
        clusters = tuple(sorted({c for c, _, _, _ in rows}))
        item_index = {item: i for i, item in enumerate(items)}
        cluster_index = {c: i for i, c in enumerate(clusters)}
        n = len(rows)
        first = np.empty(n, dtype=np.int64)
        second = np.empty(n, dtype=np.int64)
        score = np.empty(n, dtype=np.float64)
        cluster = np.empty(n, dtype=np.int64)
        for g, (c, x, y, s) in enumerate(rows):
            if x == y:
                msg = f"a game needs two different items, got {x!r} twice"
                raise ValueError(msg)
            ix, iy = item_index[x], item_index[y]
            first[g], second[g] = min(ix, iy), max(ix, iy)
            score[g] = s if ix < iy else 1.0 - s
            cluster[g] = cluster_index[c]
        return cls(
            items=items,
            first=first,
            second=second,
            score=score,
            cluster=cluster,
            clusters=clusters,
            dropped=dict(dropped or {}),
            fractional=fractional,
        )

    @property
    def n_items(self) -> int:
        return len(self.items)

    @property
    def n_games(self) -> int:
        return len(self.score)

    @property
    def n_clusters(self) -> int:
        return len(self.clusters)

    @property
    def n_ties(self) -> int:
        return int(np.count_nonzero(self.score == 0.5))

    def counts(self, weights: FloatArray | None = None) -> tuple[FloatArray, FloatArray]:
        """``(wins, games)`` matrices, optionally with a weight per game.

        ``wins[i, j]`` is i's total score against j (a tie is half a win for
        each side) and ``games[i, j] = games[j, i]`` the number of games.
        """
        w = np.ones(self.n_games) if weights is None else np.asarray(weights, dtype=np.float64)
        n = self.n_items
        wins = np.zeros((n, n))
        games = np.zeros((n, n))
        np.add.at(wins, (self.first, self.second), w * self.score)
        np.add.at(wins, (self.second, self.first), w * (1.0 - self.score))
        np.add.at(games, (self.first, self.second), w)
        games = games + games.T
        return wins, games

    def cluster_members(self) -> list[IntArray]:
        """Game indices of each cluster, in cluster order."""
        order = np.argsort(self.cluster, kind="stable")
        bounds = np.searchsorted(self.cluster[order], np.arange(self.n_clusters + 1))
        return [order[bounds[c] : bounds[c + 1]] for c in range(self.n_clusters)]


_SCORE_OF_LEFT = {Choice.LEFT: 1.0, Choice.RIGHT: 0.0, Choice.TIE: 0.5}


def select_pairwise(
    judgments: Sequence[PairwiseJudgment], options: ComparisonOptions
) -> tuple[list[PairwiseJudgment], Counter[str]]:
    """Judgments that carry a usable preference, and counts of the rest by reason."""
    kept: list[PairwiseJudgment] = []
    dropped: Counter[str] = Counter()
    excluded = set(options.exclude_annotators)
    for judgment in judgments:
        if judgment.pair_kind not in options.pair_kinds:
            dropped[DropReason.PAIR_KIND.value] += 1
        elif judgment.annotator_id in excluded:
            dropped[DropReason.EXCLUDED_ANNOTATOR.value] += 1
        elif judgment.choice is Choice.SKIP:
            dropped[DropReason.SKIP.value] += 1
        else:
            kept.append(judgment)
    return kept, dropped


def item_mapper(responses: Iterable[Response], level: Level) -> dict[str, str]:
    """Map each response id to the item it counts for at ``level``."""
    if level is Level.MODEL:
        return {r.id: r.model for r in responses}
    return {r.id: r.id for r in responses}


def build_comparisons(
    judgments: Sequence[PairwiseJudgment],
    responses: Iterable[Response],
    options: ComparisonOptions | None = None,
    *,
    ranked: Sequence[RankedJudgment] = (),
) -> ComparisonSet:
    """Turn stored judgments into games, following ``options``.

    Raises KeyError if a judgment references a response that was not supplied.
    """
    options = options or ComparisonOptions()
    item_of = item_mapper(responses, options.level)
    kept, dropped = select_pairwise(judgments, options)
    rows: list[tuple[str, str, str, float]] = []
    for judgment in kept:
        left = item_of[judgment.left_response_id]
        right = item_of[judgment.right_response_id]
        if left == right:
            dropped[DropReason.SAME_ITEM.value] += 1
            continue
        rows.append((judgment.prompt_id, left, right, _SCORE_OF_LEFT[judgment.choice]))
    if options.include_ranked:
        excluded = set(options.exclude_annotators)
        for ranking in ranked:
            if ranking.annotator_id in excluded:
                dropped[DropReason.EXCLUDED_ANNOTATOR.value] += 1
                continue
            for winner, loser in ranking.implied_pairs():
                better, worse = item_of[winner], item_of[loser]
                if better == worse:
                    dropped[DropReason.SAME_ITEM.value] += 1
                    continue
                rows.append((ranking.prompt_id, better, worse, 1.0))
    return ComparisonSet.from_outcomes(rows, dropped=dict(sorted(dropped.items())))
