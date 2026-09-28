"""Position-bias check: exact binomial test per annotator with Holm-Bonferroni."""

import pytest

from prefpairs.quality import position_bias
from prefpairs.quality.stats import binom_test, holm
from prefpairs.schema import PairKind
from prefpairs.simulate import Archetype, simulate
from tests.quality_helpers import BIG, choices, judgment


def test_counts_and_p_value_for_one_annotator() -> None:
    report = position_bias(choices("ann-1", "l" * 9 + "r" + "tts"))
    (row,) = report.results
    assert (row.n_left, row.n_right, row.n_tie, row.n_skip) == (9, 1, 2, 1)
    assert row.n_decisive == 10
    assert row.left_rate == pytest.approx(0.9)
    assert row.p_value == pytest.approx(22 / 1024, rel=1e-12)
    assert row.p_adjusted == row.p_value
    assert row.ci_lower < 0.9 < row.ci_upper
    assert row.flagged
    assert report.flagged == ("ann-1",)


def test_holm_adjustment_across_annotators() -> None:
    data = [
        *choices("ann-1", "l" * 9 + "r"),
        *choices("ann-2", "l" * 6 + "r" * 4),
        *choices("ann-3", "l" * 3 + "r" * 7),
    ]
    report = position_bias(data, alpha=0.05)
    raw = [binom_test(9, 10), binom_test(6, 10), binom_test(3, 10)]
    assert [r.p_value for r in report.results] == pytest.approx(raw, rel=1e-12)
    assert [r.p_adjusted for r in report.results] == pytest.approx(holm(raw), rel=1e-12)
    # 22/1024 * 3 = 0.064 after adjustment, so nobody is flagged at 0.05 ...
    assert report.flagged == ()
    # ... although the raw p-value of ann-1 alone would be.
    assert report.results[0].p_value < 0.05


def test_pooled_row_sums_everyone() -> None:
    data = [*choices("ann-1", "llr"), *choices("ann-2", "lts")]
    pooled = position_bias(data).pooled
    assert pooled.annotator_id is None
    assert (pooled.n_left, pooled.n_right, pooled.n_tie, pooled.n_skip) == (3, 1, 1, 1)
    assert pooled.p_adjusted == pooled.p_value


def test_no_decisive_choices_is_not_evidence() -> None:
    (row,) = position_bias(choices("ann-1", "tts")).results
    assert row.left_rate is None
    assert row.p_value == 1.0
    assert (row.ci_lower, row.ci_upper) == (0.0, 1.0)
    assert not row.flagged


def test_pair_kind_filter() -> None:
    data = [
        judgment("ann-1", "a", "b", "left", kind=PairKind.GOLD),
        judgment("ann-1", "a", "b", "right"),
    ]
    report = position_bias(data, pair_kinds=[PairKind.REGULAR])
    assert report.pair_kinds == (PairKind.REGULAR,)
    assert (report.results[0].n_left, report.results[0].n_right) == (0, 1)


def test_empty_input() -> None:
    report = position_bias([])
    assert report.results == ()
    assert report.pooled.n_decisive == 0


@pytest.mark.parametrize("seed", range(10))
def test_flags_exactly_the_left_biased_annotators(seed: int) -> None:
    dataset = simulate(BIG.model_copy(update={"seed": seed}))
    archetypes = dataset.truth.annotator_archetypes
    expected = tuple(sorted(a for a, k in archetypes.items() if k is Archetype.LEFT_BIASED))
    assert position_bias(dataset.pairwise).flagged == expected


def test_annotators_without_decisive_choices_do_not_join_the_holm_family() -> None:
    biased = choices("ann-y", "l" * 15 + "r")
    alone = position_bias(biased)
    assert alone.flagged == ("ann-y",)
    ties = [j for i in range(100) for j in choices(f"tie-{i:03d}", "ttt")]
    crowded = position_bias(biased + ties)
    rows = {r.annotator_id: r for r in crowded.results}
    assert rows["ann-y"].p_adjusted == pytest.approx(alone.results[0].p_adjusted)
    assert crowded.flagged == ("ann-y",)
    assert all(rows[f"tie-{i:03d}"].p_adjusted == 1.0 for i in range(100))
