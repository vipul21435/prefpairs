"""Self-consistency on control repeats with the sides flipped."""

import numpy as np
import pytest

from prefpairs.quality import self_consistency
from prefpairs.quality.stats import binom_lower_tail, wilson_interval
from prefpairs.schema import PairKind, PairwiseJudgment
from prefpairs.simulate import DEFAULT_PROFILES, Archetype, SimulationConfig, simulate
from tests.quality_helpers import judgment


def repeat(original: PairwiseJudgment, choice: str, *, flip: bool = True) -> PairwiseJudgment:
    left, right = original.left_response_id, original.right_response_id
    if flip:
        left, right = right, left
    return judgment(
        original.annotator_id, left, right, choice, kind=PairKind.CONTROL, repeat_of=original.id
    )


def test_hand_built_controls() -> None:
    first = judgment("ann-1", "a", "b", "left")  # a wins
    second = judgment("ann-1", "a", "b", "right")  # b wins
    third = judgment("ann-1", "a", "b", "tie")
    fourth = judgment("ann-1", "a", "b", "left")
    data = [
        first,
        repeat(first, "right"),  # flipped, still a wins: consistent
        second,
        repeat(second, "right"),  # flipped, same side clicked: a wins, inconsistent
        third,
        repeat(third, "tie", flip=False),  # tie twice: consistent, not flipped
        fourth,
        repeat(fourth, "skip"),
    ]
    (row,) = self_consistency(data).results
    assert row.n_controls == 4
    assert row.n_flipped == 3
    assert row.n_compared == 3
    assert row.n_consistent == 2
    assert row.n_same_side == 1
    assert row.n_skipped == 1
    assert row.rate == pytest.approx(2 / 3)
    assert (row.ci_lower, row.ci_upper) == pytest.approx(wilson_interval(2, 3))
    assert not row.flagged


def test_always_the_same_side_is_flagged() -> None:
    data = []
    for i in range(12):
        original = judgment("ann-1", "a", "b", "left", prompt=f"p{i}")
        data += [original, repeat(original, "left")]
    report = self_consistency(data)
    (row,) = report.results
    assert (row.n_consistent, row.n_same_side, row.rate) == (0, 12, 0.0)
    assert row.ci_upper < 0.5
    assert report.flagged == ("ann-1",)


def test_only_skipped_controls_are_no_evidence() -> None:
    original = judgment("ann-1", "a", "b", "skip")
    (row,) = self_consistency([original, repeat(original, "left")]).results
    assert row.rate is None
    assert not row.flagged


def test_missing_original_is_an_error() -> None:
    orphan = judgment("ann-1", "a", "b", "left", kind=PairKind.CONTROL, repeat_of="gone")
    with pytest.raises(KeyError):
        self_consistency([orphan])


def test_a_position_habit_is_caught_on_simulated_data() -> None:
    # An annotator who mostly clicks left and only weakly reads the answers.
    habit = DEFAULT_PROFILES[Archetype.LEFT_BIASED].model_copy(
        update={"sharpness": 0.3, "position_bias": 3.0}
    )
    profiles = {**DEFAULT_PROFILES, Archetype.LEFT_BIASED: habit}
    for seed in range(10):
        # About 80 controls each: enough power after the Holm correction.
        config = SimulationConfig(
            seed=seed, control_rate=0.5, profiles=profiles, n_prompts=60, pairs_per_prompt=8
        )
        dataset = simulate(config)
        archetypes = dataset.truth.annotator_archetypes
        expected = tuple(a for a, k in archetypes.items() if k is Archetype.LEFT_BIASED)
        assert self_consistency(dataset.pairwise).flagged == expected


def test_default_archetypes_rank_as_expected() -> None:
    dataset = simulate(SimulationConfig(seed=0, control_rate=0.5))
    report = self_consistency(dataset.pairwise)
    archetypes = dataset.truth.annotator_archetypes
    rate = {r.annotator_id: r.rate or 0.0 for r in report.results}
    reliable = [rate[a] for a, k in archetypes.items() if k is Archetype.RELIABLE]
    left = [rate[a] for a, k in archetypes.items() if k is Archetype.LEFT_BIASED]
    assert min(reliable) > max(left)
    assert all(r.n_flipped == r.n_controls for r in report.results)


def controls(annotator: str, n: int, n_consistent: int) -> list[PairwiseJudgment]:
    """``n`` flipped controls; the first ``n_consistent`` repeat the original label."""
    data = []
    for i in range(n):
        original = judgment(annotator, "a", "b", "left", prompt=f"p{i}")
        data += [original, repeat(original, "right" if i < n_consistent else "left")]
    return data


def test_flag_is_an_exact_one_sided_test_at_alpha() -> None:
    data = controls("ann-1", 30, 9)
    (row,) = self_consistency(data).results
    assert row.ci_upper < 0.5
    assert row.p_value == pytest.approx(binom_lower_tail(9, 30, 0.5))
    assert row.p_adjusted == row.p_value
    assert self_consistency(data, alpha=0.05).flagged == ("ann-1",)
    assert self_consistency(data, alpha=0.01).flagged == ()


def test_consistency_flags_are_holm_adjusted_across_annotators() -> None:
    borderline = controls("ann-1", 30, 9)
    crowd = [j for i in range(39) for j in controls(f"ann-{i + 2:02d}", 30, 15)]
    report = self_consistency(borderline + crowd)
    row = next(r for r in report.results if r.annotator_id == "ann-1")
    assert row.p_adjusted > 0.05
    assert report.flagged == ()


def test_family_wise_error_rate_at_the_boundary() -> None:
    rng = np.random.default_rng(0)
    runs, any_flag = 100, 0
    for _ in range(runs):
        data = []
        for a in range(40):
            data += controls(f"ann-{a:02d}", 30, int(rng.binomial(30, 0.5)))
        any_flag += bool(self_consistency(data).flagged)
    assert any_flag / runs <= 0.1
