"""Small exact and closed-form statistics shared by the quality checks.

Everything here is written on ``math`` and numpy, without scipy:

* ``binom_test``: the exact two-sided binomial test. The p-value sums the
  probabilities of every outcome that is no more likely than the observed one
  (the "minimum likelihood" definition used by R's ``binom.test``), evaluated
  in log space so large counts neither overflow nor lose the tail.
* ``holm``: Holm-Bonferroni step-down adjusted p-values. They control the
  family-wise error rate under any dependence between the tests, and are never
  larger than plain Bonferroni.
* ``wilson_interval``: the Wilson score interval for a binomial proportion,
  which, unlike the Wald interval, stays inside [0, 1] and keeps its coverage
  for small counts and rates near 0 or 1.
* Normal and chi-square (1 df) tail probabilities via ``math.erfc``.
"""

import math
from collections.abc import Sequence
from statistics import NormalDist

import numpy as np

RELATIVE_TOLERANCE = 1e-7
"""Outcomes whose probability is within this relative distance of the observed
one count as "equally likely", so floating-point noise cannot drop an exact tie
(for example the mirror outcome when p = 0.5) from the two-sided sum."""


def _check_counts(successes: int, n: int) -> None:
    if n < 0 or not 0 <= successes <= n:
        msg = f"successes must satisfy 0 <= successes <= n, got {successes} of {n}"
        raise ValueError(msg)


def binom_log_pmf(n: int, p: float) -> np.ndarray:
    """``log P(X = k)`` for ``k = 0..n`` with ``X ~ Binomial(n, p)`` and ``0 < p < 1``."""
    k = np.arange(n + 1, dtype=np.float64)
    log_choose = np.array(
        [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) for i in range(n + 1)]
    )
    return log_choose + k * math.log(p) + (n - k) * math.log1p(-p)


def binom_test(successes: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided p-value of ``successes`` out of ``n`` under ``Binomial(n, p)``.

    Returns 1.0 when ``n == 0`` (no evidence either way).
    """
    _check_counts(successes, n)
    if not 0.0 <= p <= 1.0:
        msg = f"p must lie in [0, 1], got {p}"
        raise ValueError(msg)
    if n == 0:
        return 1.0
    if p in (0.0, 1.0):
        certain = 0 if p == 0.0 else n
        return 1.0 if successes == certain else 0.0
    log_pmf = binom_log_pmf(n, p)
    threshold = log_pmf[successes] + math.log1p(RELATIVE_TOLERANCE)
    tail = log_pmf[log_pmf <= threshold]
    top = float(tail.max())
    return min(1.0, math.exp(top) * float(np.exp(tail - top).sum()))


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the order given.

    With ``m`` tests sorted ascending, the i-th smallest (1-based) is adjusted
    to ``max_{j <= i} min(1, (m - j + 1) * p_(j))``. Rejecting every hypothesis
    whose adjusted p-value is at most ``alpha`` controls the family-wise error
    rate at ``alpha``.
    """
    values = [float(p) for p in p_values]
    if any(not 0.0 <= p <= 1.0 for p in values):
        msg = "p-values must lie in [0, 1]"
        raise ValueError(msg)
    m = len(values)
    order = sorted(range(m), key=lambda i: values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def normal_quantile(q: float) -> float:
    """Inverse of the standard normal CDF."""
    return NormalDist().inv_cdf(q)


def normal_two_sided_p(z: float) -> float:
    """``P(|Z| >= |z|)`` for a standard normal ``Z``."""
    return math.erfc(abs(z) / math.sqrt(2.0))


def chi2_1_sf(statistic: float) -> float:
    """``P(X >= statistic)`` for ``X`` chi-square with one degree of freedom."""
    return math.erfc(math.sqrt(max(statistic, 0.0) / 2.0))


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a proportion; ``(0.0, 1.0)`` when ``n == 0``."""
    _check_counts(successes, n)
    if not 0.0 < confidence < 1.0:
        msg = f"confidence must lie in (0, 1), got {confidence}"
        raise ValueError(msg)
    if n == 0:
        return 0.0, 1.0
    z = normal_quantile(0.5 + confidence / 2.0)
    rate = successes / n
    z2n = z * z / n
    centre = (rate + z2n / 2.0) / (1.0 + z2n)
    half = z / (1.0 + z2n) * math.sqrt(rate * (1.0 - rate) / n + z2n / (4.0 * n))
    lower = 0.0 if successes == 0 else max(0.0, centre - half)
    upper = 1.0 if successes == n else min(1.0, centre + half)
    return lower, upper
