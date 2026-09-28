"""Turning stored judgments into canonical games."""

from datetime import timedelta

import numpy as np
import pytest
from pydantic import ValidationError

from prefpairs.aggregate import (
    ComparisonOptions,
    ComparisonSet,
    DropReason,
    Level,
    build_comparisons,
)
from prefpairs.schema import (
    AnyRecord,
    Choice,
    PairKind,
    PairwiseJudgment,
    RankedJudgment,
    Response,
)
from tests.conftest import T0


def split(records: list[AnyRecord]) -> tuple[list[PairwiseJudgment], list[Response]]:
    judgments = [r for r in records if isinstance(r, PairwiseJudgment)]
    responses = [r for r in records if isinstance(r, Response)]
    return judgments, responses


def ranked_of(records: list[AnyRecord]) -> list[RankedJudgment]:
    return [r for r in records if isinstance(r, RankedJudgment)]


class TestOptions:
    def test_kinds_and_annotators_are_sorted_and_deduplicated(self) -> None:
        options = ComparisonOptions(
            pair_kinds=(PairKind.GOLD, PairKind.CONTROL, PairKind.GOLD),
            exclude_annotators=("b", "a", "b"),
        )
        assert options.pair_kinds == (PairKind.CONTROL, PairKind.GOLD)
        assert options.exclude_annotators == ("a", "b")

    def test_needs_a_pair_kind(self) -> None:
        with pytest.raises(ValidationError, match="at least one pair kind"):
            ComparisonOptions(pair_kinds=())


class TestBuild:
    def test_default_keeps_regular_decisive_and_tied_games(self, records: list[AnyRecord]) -> None:
        judgments, responses = split(records)
        games = build_comparisons(judgments, responses)
        # j1 (regular, model-a vs model-b, right wins) and j4 (regular tie, a vs c);
        # j2 is gold, j3 is a control, j5 is a skip.
        assert games.items == ("model-a", "model-b", "model-c")
        assert games.clusters == ("p1",)
        assert games.n_games == 2
        pairs = {(games.items[f], games.items[s]): float(x) for f, s, x in _rows(games)}
        assert pairs == {("model-a", "model-b"): 0.0, ("model-a", "model-c"): 0.5}
        assert games.n_ties == 1
        assert dict(games.dropped) == {"pair_kind": 2, "skip": 1}

    def test_response_level_uses_response_ids(self, records: list[AnyRecord]) -> None:
        judgments, responses = split(records)
        games = build_comparisons(judgments, responses, ComparisonOptions(level=Level.RESPONSE))
        assert games.items == ("p1-a", "p1-b", "p1-c")

    def test_every_kind_and_exclusions(self, records: list[AnyRecord]) -> None:
        judgments, responses = split(records)
        options = ComparisonOptions(pair_kinds=tuple(PairKind), exclude_annotators=("ann-2",))
        games = build_comparisons(judgments, responses, options)
        assert games.n_games == 3  # j1, j2 (gold), j3 (control)
        assert set(games.clusters) == {"p1", "p2"}
        assert dict(games.dropped) == {DropReason.EXCLUDED_ANNOTATOR.value: 2}

    def test_orientation_does_not_depend_on_display_order(self, records: list[AnyRecord]) -> None:
        judgments, responses = split(records)
        original = next(j for j in judgments if j.id == "j1")
        flipped = original.model_copy(
            update={
                "id": "j1-flipped",
                "left_response_id": original.right_response_id,
                "right_response_id": original.left_response_id,
                "choice": Choice.LEFT,
            }
        )
        first = build_comparisons([original], responses)
        second = build_comparisons([flipped], responses)
        assert (first.first[0], first.second[0], first.score[0]) == (
            second.first[0],
            second.second[0],
            second.score[0],
        )

    def test_same_model_pairs_are_dropped_at_model_level(self, records: list[AnyRecord]) -> None:
        judgments, responses = split(records)
        clone = responses[0].model_copy(update={"id": "p1-z"})
        judgment = judgments[0].model_copy(
            update={"left_response_id": "p1-a", "right_response_id": "p1-z"}
        )
        games = build_comparisons([judgment], [*responses, clone])
        assert games.n_games == 0
        assert games.items == ()
        assert dict(games.dropped) == {"same_item": 1}

    def test_rankings_expand_into_implied_pairs_when_asked(self, records: list[AnyRecord]) -> None:
        judgments, responses = split(records)
        ranked = ranked_of(records)
        options = ComparisonOptions(include_ranked=True)
        games = build_comparisons(judgments, responses, options, ranked=ranked)
        assert games.n_games == 2 + 3
        excluded = ComparisonOptions(include_ranked=True, exclude_annotators=("ann-2",))
        games = build_comparisons(judgments, responses, excluded, ranked=ranked)
        assert games.n_games == 1
        # ann-2's two pairwise judgments and its one ranking
        assert games.dropped[DropReason.EXCLUDED_ANNOTATOR.value] == 2 + 1

    def test_ranked_same_model_pairs_are_dropped(self, records: list[AnyRecord]) -> None:
        _, responses = split(records)
        clone = responses[0].model_copy(update={"id": "p1-z"})
        ranking = RankedJudgment(
            id="k9",
            session_id="s2",
            annotator_id="ann-2",
            prompt_id="p1",
            shown=("p1-a", "p1-z", "p1-b"),
            ranking=("p1-a", "p1-z", "p1-b"),
            created_at=T0 + timedelta(minutes=3),
        )
        games = build_comparisons(
            [], [*responses, clone], ComparisonOptions(include_ranked=True), ranked=[ranking]
        )
        assert games.n_games == 2
        assert games.dropped == {"same_item": 1}
        assert np.all(games.score == 1.0)

    def test_unknown_response_is_an_error(self, records: list[AnyRecord]) -> None:
        judgments, _ = split(records)
        with pytest.raises(KeyError):
            build_comparisons(judgments, [])


def _rows(games: ComparisonSet) -> list[tuple[int, int, float]]:
    return [
        (int(f), int(s), float(x))
        for f, s, x in zip(games.first, games.second, games.score, strict=True)
    ]


class TestComparisonSet:
    def test_counts_split_ties_and_are_symmetric_in_games(self) -> None:
        games = ComparisonSet.from_outcomes(
            [("c1", "x", "y", 1.0), ("c1", "y", "x", 1.0), ("c2", "x", "z", 0.5)]
        )
        wins, played = games.counts()
        assert np.array_equal(wins, [[0, 1, 0.5], [1, 0, 0], [0.5, 0, 0]])
        assert np.array_equal(played, played.T)
        assert np.array_equal(played, [[0, 2, 1], [2, 0, 0], [1, 0, 0]])
        assert np.allclose(wins + wins.T, played)
        weighted, _ = games.counts(np.array([2.0, 0.0, 1.0]))
        assert weighted[0, 1] == 2.0
        assert weighted[1, 0] == 0.0

    def test_cluster_members_partition_the_games(self) -> None:
        games = ComparisonSet.from_outcomes(
            [("b", "x", "y", 1.0), ("a", "x", "y", 0.0), ("b", "x", "z", 0.5), ("c", "y", "z", 1)]
        )
        members = games.cluster_members()
        assert games.clusters == ("a", "b", "c")
        assert [m.tolist() for m in members] == [[1], [0, 2], [3]]

    def test_rejects_self_games_and_bad_scores(self) -> None:
        with pytest.raises(ValueError, match="two different items"):
            ComparisonSet.from_outcomes([("c", "x", "x", 1.0)])
        with pytest.raises(ValueError, match=r"0, 0\.5 or 1"):
            ComparisonSet.from_outcomes([("c", "x", "y", 0.3)])
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            ComparisonSet.from_outcomes([("c", "x", "y", 1.5)], fractional=True)
        expected = ComparisonSet.from_outcomes([("c", "x", "y", 0.3)], fractional=True)
        assert expected.score.tolist() == [0.3]

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("first", np.array([1]), "first < second"),
            ("second", np.array([5]), "item index"),
            ("cluster", np.array([3]), "cluster index"),
            ("score", np.array([1.0, 0.0]), "same length"),
        ],
    )
    def test_rejects_inconsistent_arrays(self, field: str, value: np.ndarray, message: str) -> None:
        arrays = {
            "items": ("x", "y"),
            "first": np.array([0]),
            "second": np.array([1]),
            "score": np.array([1.0]),
            "cluster": np.array([0]),
            "clusters": ("c",),
        }
        arrays[field] = value
        with pytest.raises(ValueError, match=message):
            ComparisonSet(**arrays)  # type: ignore[arg-type]
