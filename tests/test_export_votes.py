"""Vote aggregation per canonical pair and the filter pipeline."""

import pytest

from prefpairs.export import DropReason, FilterConfig, aggregate_votes, select_pairs
from prefpairs.schema import PairKind
from prefpairs.simulate import SimulationConfig, simulate
from tests.quality_helpers import judgment


def test_votes_count_on_the_canonical_pair_whatever_the_side() -> None:
    data = [
        judgment("ann-1", "a", "b", "left"),  # a wins
        judgment("ann-2", "b", "a", "right"),  # a wins, shown swapped
        judgment("ann-3", "b", "a", "left"),  # b wins
        judgment("ann-4", "a", "b", "tie"),
        judgment("ann-5", "a", "b", "skip"),
    ]
    (votes,) = aggregate_votes(data)
    assert (votes.a_id, votes.b_id) == ("a", "b")
    assert (votes.n_a, votes.n_b, votes.n_tie, votes.n_skip) == (2, 1, 1, 1)
    assert votes.n_votes == 4
    assert votes.chosen_rejected == ("a", "b")
    assert votes.agreement == pytest.approx(0.5)
    assert votes.annotators == ("ann-1", "ann-2", "ann-3", "ann-4")


def test_excluded_annotators_are_counted_apart_and_never_decide() -> None:
    data = [
        judgment("good", "a", "b", "left"),
        judgment("bad", "a", "b", "right"),
        judgment("bad", "a", "b", "right"),
    ]
    (votes,) = aggregate_votes(data, exclude_annotators=["bad"])
    assert (votes.n_a, votes.n_b, votes.n_excluded) == (1, 0, 2)
    assert votes.chosen_rejected == ("a", "b")


def test_pairs_are_separate_per_prompt_and_sorted() -> None:
    data = [
        judgment("ann-1", "c", "d", "left", prompt="p2"),
        judgment("ann-1", "a", "b", "left", prompt="p1"),
        judgment("ann-1", "b", "c", "left", prompt="p1"),
    ]
    keys = [(v.prompt_id, v.a_id, v.b_id) for v in aggregate_votes(data)]
    assert keys == [("p1", "a", "b"), ("p1", "b", "c"), ("p2", "c", "d")]


def test_gold_and_control_pairs_are_not_data_by_default() -> None:
    first = judgment("ann-1", "a", "b", "left")
    data = [
        first,
        judgment("ann-1", "b", "a", "right", kind=PairKind.CONTROL, repeat_of=first.id),
        judgment("ann-1", "c", "d", "left", kind=PairKind.GOLD),
    ]
    assert [(v.a_id, v.n_votes) for v in aggregate_votes(data)] == [("a", 1)]
    every_kind = aggregate_votes(data, pair_kinds=tuple(PairKind))
    assert [(v.a_id, v.n_votes) for v in every_kind] == [("a", 2), ("c", 1)]


def test_every_dropped_pair_carries_its_reasons() -> None:
    data = [
        # p1: 3-0, kept
        *(judgment(f"ann-{i}", "a", "b", "left", prompt="p1") for i in range(3)),
        # p2: 1-1, a tie
        judgment("ann-1", "a", "b", "left", prompt="p2"),
        judgment("ann-2", "a", "b", "right", prompt="p2"),
        # p3: 2-1 with two ties, agreement 0.4
        judgment("ann-1", "a", "b", "left", prompt="p3"),
        judgment("ann-2", "a", "b", "left", prompt="p3"),
        judgment("ann-3", "a", "b", "right", prompt="p3"),
        judgment("ann-4", "a", "b", "tie", prompt="p3"),
        judgment("ann-5", "a", "b", "tie", prompt="p3"),
        # p4: a single vote
        judgment("ann-1", "a", "b", "left", prompt="p4"),
        # p5: only an excluded annotator
        judgment("bad", "a", "b", "left", prompt="p5"),
        # p6: only skips
        judgment("ann-1", "a", "b", "skip", prompt="p6"),
    ]
    config = FilterConfig(min_votes=2, min_agreement=0.5, exclude_annotators=("bad",))
    selection = select_pairs(data, config)
    reasons = {d.votes.prompt_id: d.reasons for d in selection.decisions}
    assert reasons == {
        "p1": (),
        "p2": (DropReason.TIE,),  # agreement 1/2 is not below 0.5
        "p3": (DropReason.LOW_AGREEMENT,),
        "p4": (DropReason.TOO_FEW_VOTES,),
        "p5": (DropReason.NO_VOTES,),
        "p6": (DropReason.NO_VOTES,),
    }
    assert [v.prompt_id for v in selection.kept] == ["p1"]
    assert selection.drop_counts() == {
        "no_votes": 2,
        "too_few_votes": 1,
        "tie": 1,
        "low_agreement": 1,
    }


def test_ties_can_be_kept() -> None:
    data = [judgment("ann-1", "a", "b", "tie")]
    kept = select_pairs(data, FilterConfig(drop_ties=False, min_agreement=0.0)).kept
    assert len(kept) == 1
    assert kept[0].chosen_rejected is None


def test_selection_is_deterministic_and_complete_on_simulated_data() -> None:
    dataset = simulate(SimulationConfig(seed=3))
    regular = [j for j in dataset.pairwise if j.pair_kind is PairKind.REGULAR]
    first = select_pairs(dataset.pairwise, FilterConfig(min_votes=2))
    again = select_pairs(list(reversed(dataset.pairwise)), FilterConfig(min_votes=2))
    assert first == again
    pairs = {(j.prompt_id, *j.canonical_pair) for j in regular}
    assert len(first.decisions) == len(pairs)
    counted = sum(d.votes.n_votes + d.votes.n_skip for d in first.decisions)
    assert counted == len(regular)
    assert 0 < len(first.kept) < len(first.decisions)


def test_b_can_win_and_skips_alone_have_no_agreement() -> None:
    (b_wins,) = aggregate_votes([judgment("ann-1", "a", "b", "right")])
    assert b_wins.chosen_rejected == ("b", "a")
    (skipped,) = aggregate_votes([judgment("ann-1", "a", "b", "skip")])
    assert skipped.agreement is None
