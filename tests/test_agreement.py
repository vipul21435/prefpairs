"""Cohen and Fleiss kappa on canonical labels, checked against exact hand computations."""

from collections import Counter
from fractions import Fraction

import numpy as np
import pytest

from prefpairs.quality import agreement, cohen_kappa, fleiss_kappa
from prefpairs.quality.agreement import item_labels
from prefpairs.schema import CanonicalLabel, PairKind
from prefpairs.simulate import Archetype, SimulationConfig, simulate
from tests.quality_helpers import judgment

# Fleiss (1971) style example: 10 items, 14 raters, 5 categories.
FLEISS_TABLE = [
    [0, 0, 0, 0, 14],
    [0, 2, 6, 4, 2],
    [0, 0, 3, 5, 6],
    [0, 3, 9, 2, 0],
    [2, 2, 8, 1, 1],
    [7, 7, 0, 0, 0],
    [3, 2, 6, 3, 0],
    [2, 5, 3, 2, 2],
    [6, 5, 2, 1, 0],
    [0, 2, 2, 3, 7],
]


def exact_fleiss(table: list[list[int]]) -> Fraction:
    m = sum(table[0])
    n = len(table)
    per_item = [Fraction(sum(c * c for c in row) - m, m * (m - 1)) for row in table]
    observed = sum(per_item, Fraction(0)) / n
    shares = [Fraction(sum(row[j] for row in table), n * m) for j in range(len(table[0]))]
    expected = sum((s * s for s in shares), Fraction(0))
    return (observed - expected) / (1 - expected)


def exact_cohen(first: list[str], second: list[str]) -> Fraction:
    n = len(first)
    observed = Fraction(sum(x == y for x, y in zip(first, second, strict=True)), n)
    c1, c2 = Counter(first), Counter(second)
    expected = sum((Fraction(c1[c] * c2[c], n * n) for c in c1), Fraction(0))
    return (observed - expected) / (1 - expected)


class TestCohen:
    def test_two_by_two_textbook_case(self) -> None:
        # 50 items: yes/yes 20, yes/no 5, no/yes 10, no/no 15 -> p_o 0.7, p_e 0.5, kappa 0.4.
        first = ["yes"] * 25 + ["no"] * 25
        second = ["yes"] * 20 + ["no"] * 5 + ["yes"] * 10 + ["no"] * 15
        result = cohen_kappa(first, second)
        assert result.n_items == 50
        assert result.observed == pytest.approx(0.7, abs=1e-12)
        assert result.expected == pytest.approx(0.5, abs=1e-12)
        assert result.kappa == pytest.approx(0.4, abs=1e-12)

    def test_three_labels_against_exact_fractions(self) -> None:
        first = list("aabbbttaabtbaatb")
        second = list("abbbtttaaatbbatb")
        expected = float(exact_cohen(first, second))
        assert cohen_kappa(first, second).kappa == pytest.approx(expected, abs=1e-12)

    def test_perfect_and_undefined(self) -> None:
        assert cohen_kappa(["a", "b"], ["a", "b"]).kappa == pytest.approx(1.0)
        assert cohen_kappa(["a", "a"], ["a", "a"]).kappa is None

    def test_rejects_mismatched_or_empty(self) -> None:
        with pytest.raises(ValueError, match="every item"):
            cohen_kappa(["a"], ["a", "b"])
        with pytest.raises(ValueError, match="at least one"):
            cohen_kappa([], [])


class TestFleiss:
    def test_published_style_table_against_exact_fractions(self) -> None:
        result = fleiss_kappa(FLEISS_TABLE)
        assert result.n_items == 10
        assert result.kappa == pytest.approx(float(exact_fleiss(FLEISS_TABLE)), abs=1e-12)
        assert result.kappa == pytest.approx(0.210, abs=5e-4)

    def test_two_raters_give_scotts_pi(self) -> None:
        first = list("aabbbttaab")
        second = list("abbbtttaaa")
        table = [
            [int(x == c) + int(y == c) for c in "abt"] for x, y in zip(first, second, strict=True)
        ]
        pooled = Counter(first + second)
        observed = Fraction(sum(x == y for x, y in zip(first, second, strict=True)), len(first))
        expected = sum((Fraction(v, 2 * len(first)) ** 2 for v in pooled.values()), Fraction(0))
        scott_pi = (observed - expected) / (1 - expected)
        assert fleiss_kappa(table).kappa == pytest.approx(float(scott_pi), abs=1e-12)

    def test_undefined_when_everyone_uses_one_label(self) -> None:
        assert fleiss_kappa([[3, 0], [3, 0]]).kappa is None

    @pytest.mark.parametrize(
        ("table", "match"),
        [
            (np.zeros((0, 3)), "non-empty"),
            ([[2, -1]], "non-negative"),
            ([[1, 0]], "same number"),
            ([[2, 0], [1, 2]], "same number"),
        ],
    )
    def test_rejects_bad_tables(self, table: object, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            fleiss_kappa(table)  # type: ignore[arg-type]


class TestStoreLevel:
    def test_labels_are_canonical_so_display_order_does_not_matter(self) -> None:
        data = [
            judgment("ann-1", "a", "b", "left"),
            judgment("ann-2", "b", "a", "right"),
            judgment("ann-3", "b", "a", "left"),
            judgment("ann-3", "a", "b", "left"),  # a later label of the same item is ignored
            judgment("ann-4", "a", "b", "skip"),
            judgment("ann-1", "a", "b", "left", kind=PairKind.CONTROL, repeat_of="x"),
        ]
        labels = item_labels(data)
        assert labels == {
            ("p1", "a", "b"): {
                "ann-1": CanonicalLabel.A_WINS,
                "ann-2": CanonicalLabel.A_WINS,
                "ann-3": CanonicalLabel.B_WINS,
            }
        }

    def test_pairs_summaries_and_fleiss(self) -> None:
        names = {"l": "left", "r": "right", "t": "tie"}
        data = [
            judgment(annotator, "a", "b", names[code], prompt=f"p{item}")
            for item, codes in enumerate(["llr", "lll", "rrl", "rrr", "ltl"])
            for annotator, code in zip(("ann-1", "ann-2", "ann-3"), codes, strict=True)
        ]
        data.append(judgment("ann-1", "a", "b", "left", prompt="solo"))
        report = agreement(data)
        assert report.n_items == 6
        by_pair = {(p.annotator_a, p.annotator_b): p for p in report.pairs}
        assert set(by_pair) == {("ann-1", "ann-2"), ("ann-1", "ann-3"), ("ann-2", "ann-3")}
        one_two = by_pair[("ann-1", "ann-2")]
        assert one_two.n_items == 5
        assert one_two.kappa == pytest.approx(
            float(exact_cohen(list("llrrl"), list("llrrt"))), abs=1e-12
        )
        assert report.fleiss is not None
        assert (report.fleiss.n_raters, report.fleiss.n_items) == (3, 5)
        assert report.fleiss.n_items_excluded == 1
        summary = {a.annotator_id: a for a in report.annotators}
        assert summary["ann-1"].n_partners == 2
        assert summary["ann-1"].n_shared_items == 10
        weights = [by_pair[("ann-1", "ann-2")], by_pair[("ann-1", "ann-3")]]
        mean = sum(p.n_items * (p.kappa or 0.0) for p in weights) / 10
        assert summary["ann-1"].mean_kappa == pytest.approx(mean, abs=1e-12)

    def test_min_shared_and_fixed_rater_count(self) -> None:
        data = [judgment("ann-1", "a", "b", "left"), judgment("ann-2", "a", "b", "left")]
        report = agreement(data, min_shared=2)
        assert report.pairs == ()
        assert report.annotators[0].mean_kappa is None
        assert agreement(data, n_raters=5).fleiss is None
        assert agreement([judgment("ann-1", "a", "b", "left")]).fleiss is None


def test_simulated_adversarial_annotator_disagrees_with_everyone() -> None:
    dataset = simulate(SimulationConfig(seed=0))
    report = agreement(dataset.pairwise)
    archetypes = dataset.truth.annotator_archetypes
    kappa = {a.annotator_id: a.mean_kappa or 0.0 for a in report.annotators}
    adversarial = [a for a, k in archetypes.items() if k is Archetype.ADVERSARIAL]
    reliable = [a for a, k in archetypes.items() if k is Archetype.RELIABLE]
    assert all(kappa[a] < 0 for a in adversarial)
    assert all(kappa[a] > 0.3 for a in reliable)
    assert report.fleiss is not None
    assert report.fleiss.n_raters == dataset.truth.config.redundancy
