"""Exact binomial test, Holm-Bonferroni and Wilson intervals against hand computations."""

import math
from fractions import Fraction

import pytest

from prefpairs.quality.stats import (
    binom_test,
    chi2_1_sf,
    holm,
    normal_quantile,
    normal_two_sided_p,
    wilson_interval,
)


def exact_two_sided(k: int, n: int, p: Fraction) -> Fraction:
    """Brute-force reference: sum P(X = i) over outcomes no more likely than k."""
    pmf = [math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(n + 1)]
    return sum((q for q in pmf if q <= pmf[k]), Fraction(0))


class TestBinomTest:
    def test_hand_computed_fair_coin(self) -> None:
        # 9 of 10: outcomes 0, 1, 9, 10 are as or less likely -> (1 + 10 + 10 + 1) / 1024.
        assert binom_test(9, 10) == pytest.approx(22 / 1024, rel=1e-12)
        assert binom_test(1, 10) == pytest.approx(22 / 1024, rel=1e-12)
        assert binom_test(10, 10) == pytest.approx(2 / 1024, rel=1e-12)

    def test_centre_and_empty(self) -> None:
        assert binom_test(5, 10) == pytest.approx(1.0, rel=1e-12)
        assert binom_test(0, 0) == 1.0

    @pytest.mark.parametrize(
        ("k", "n", "p"),
        [
            (3, 10, Fraction(1, 5)),
            (0, 10, Fraction(1, 5)),
            (7, 10, Fraction(1, 5)),
            (12, 30, Fraction(3, 10)),
            (20, 25, Fraction(1, 2)),
            (1, 7, Fraction(2, 3)),
            (40, 60, Fraction(1, 2)),
        ],
    )
    def test_matches_exact_rational_reference(self, k: int, n: int, p: Fraction) -> None:
        expected = float(exact_two_sided(k, n, p))
        assert binom_test(k, n, float(p)) == pytest.approx(expected, rel=1e-12)

    def test_symmetric_under_relabelling(self) -> None:
        for k in range(21):
            assert binom_test(k, 20, 0.3) == pytest.approx(binom_test(20 - k, 20, 0.7), rel=1e-12)

    def test_large_counts_keep_their_tail(self) -> None:
        # 700 of 1000 is about 12.6 standard deviations out; the p-value is tiny but positive.
        p_value = binom_test(700, 1000)
        assert 0.0 < p_value < 1e-30

    def test_degenerate_probabilities(self) -> None:
        assert binom_test(0, 5, 0.0) == 1.0
        assert binom_test(1, 5, 0.0) == 0.0
        assert binom_test(5, 5, 1.0) == 1.0
        assert binom_test(4, 5, 1.0) == 0.0

    @pytest.mark.parametrize(("k", "n", "p"), [(-1, 5, 0.5), (6, 5, 0.5), (1, 5, 1.5)])
    def test_rejects_bad_input(self, k: int, n: int, p: float) -> None:
        with pytest.raises(ValueError, match="must"):
            binom_test(k, n, p)


class TestHolm:
    def test_textbook_example(self) -> None:
        # Sorted: 0.005*4 = 0.02, 0.01*3 = 0.03, 0.03*2 = 0.06, 0.04*1 -> max(0.06, 0.04).
        adjusted = holm([0.01, 0.04, 0.03, 0.005])
        assert adjusted == pytest.approx([0.03, 0.06, 0.06, 0.02], rel=1e-12)

    def test_caps_at_one_and_handles_empty(self) -> None:
        assert holm([0.6, 0.9]) == [1.0, 1.0]
        assert holm([]) == []

    def test_never_below_raw_and_never_above_bonferroni(self) -> None:
        raw = [0.001, 0.2, 0.03, 0.04, 0.5, 0.012]
        adjusted = holm(raw)
        for p, a in zip(raw, adjusted, strict=True):
            assert p <= a <= min(1.0, len(raw) * p)

    def test_rejects_invalid_p_values(self) -> None:
        with pytest.raises(ValueError, match="p-values"):
            holm([0.1, 1.2])


class TestWilson:
    def test_closed_form(self) -> None:
        lower, upper = wilson_interval(8, 10, 0.95)
        z = 1.959963984540054
        centre = (0.8 + z * z / 20) / (1 + z * z / 10)
        half = z / (1 + z * z / 10) * math.sqrt(0.8 * 0.2 / 10 + z * z / 400)
        assert (lower, upper) == pytest.approx((centre - half, centre + half), rel=1e-12)
        assert (lower, upper) == pytest.approx((0.4902, 0.9433), abs=5e-5)

    def test_boundaries(self) -> None:
        assert wilson_interval(0, 10)[0] == 0.0
        assert wilson_interval(10, 10)[1] == 1.0
        assert wilson_interval(0, 0) == (0.0, 1.0)

    def test_mirror_symmetry(self) -> None:
        lower, upper = wilson_interval(3, 17, 0.9)
        mirror_lower, mirror_upper = wilson_interval(14, 17, 0.9)
        assert (lower, upper) == pytest.approx((1 - mirror_upper, 1 - mirror_lower), rel=1e-12)

    def test_rejects_bad_confidence(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            wilson_interval(1, 2, 1.0)


def test_normal_and_chi_square_tails() -> None:
    assert normal_quantile(0.975) == pytest.approx(1.959963984540054, rel=1e-12)
    assert normal_two_sided_p(1.959963984540054) == pytest.approx(0.05, rel=1e-12)
    assert normal_two_sided_p(-1.959963984540054) == pytest.approx(0.05, rel=1e-12)
    # A chi-square(1) variable is a squared standard normal.
    assert chi2_1_sf(1.959963984540054**2) == pytest.approx(0.05, rel=1e-12)
    assert chi2_1_sf(-1.0) == 1.0
