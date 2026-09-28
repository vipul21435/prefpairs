"""Position bias: does an annotator pick the first-shown (left) response too often?

The collection flow randomises which response is shown on the left, so for an
annotator who judges content alone a decisive choice is left with probability
exactly 1/2, whatever the quality of the responses. The check therefore counts
decisive choices (ties and skips are reported but carry no side) and runs an
exact two-sided binomial test of ``left ~ Binomial(left + right, 1/2)``.

One test per annotator is a family of tests, so the p-values are adjusted with
Holm-Bonferroni before an annotator is flagged. Only annotators with at least
one decisive choice are in the family; the others cannot be tested and get an
adjusted p-value of 1.0. A pooled test over every
annotator is reported as well: a significant pooled rate with no single
annotator flagged points at the interface (for example the left response
loading first) rather than at a person.
"""

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence

from pydantic import Field

from prefpairs.quality.stats import binom_test, holm_estimable, wilson_interval
from prefpairs.schema import Choice, PairKind, PairwiseJudgment, Record

ALL_PAIR_KINDS: tuple[PairKind, ...] = tuple(PairKind)


class PositionBiasResult(Record):
    """Left/right counts of one annotator (or of everyone, when pooled) and the test."""

    annotator_id: str | None
    """None for the pooled row."""
    n_left: int
    n_right: int
    n_tie: int
    n_skip: int
    left_rate: float | None
    """Share of decisive choices that were left; None without decisive choices."""
    ci_lower: float
    ci_upper: float
    """Wilson interval for the left rate."""
    p_value: float
    """Exact two-sided binomial test of a left rate of 1/2."""
    p_adjusted: float
    """Holm-Bonferroni across annotators (equal to ``p_value`` for the pooled row)."""
    flagged: bool

    @property
    def n_decisive(self) -> int:
        return self.n_left + self.n_right


class PositionBiasReport(Record):
    """Per-annotator position-bias tests and the pooled test."""

    alpha: float = Field(gt=0, lt=1)
    confidence: float = Field(gt=0, lt=1)
    pair_kinds: tuple[PairKind, ...]
    results: tuple[PositionBiasResult, ...]
    pooled: PositionBiasResult

    @property
    def flagged(self) -> tuple[str, ...]:
        """Annotators whose adjusted p-value is at most ``alpha``, sorted."""
        return tuple(sorted(r.annotator_id for r in self.results if r.flagged and r.annotator_id))


def _result(
    annotator_id: str | None,
    counts: Counter[Choice],
    *,
    p_adjusted: float | None,
    alpha: float,
    confidence: float,
) -> PositionBiasResult:
    left, right = counts[Choice.LEFT], counts[Choice.RIGHT]
    decisive = left + right
    p_value = binom_test(left, decisive, 0.5)
    adjusted = p_value if p_adjusted is None else p_adjusted
    lower, upper = wilson_interval(left, decisive, confidence)
    return PositionBiasResult(
        annotator_id=annotator_id,
        n_left=left,
        n_right=right,
        n_tie=counts[Choice.TIE],
        n_skip=counts[Choice.SKIP],
        left_rate=left / decisive if decisive else None,
        ci_lower=lower,
        ci_upper=upper,
        p_value=p_value,
        p_adjusted=adjusted,
        flagged=decisive > 0 and adjusted <= alpha,
    )


def position_bias(
    judgments: Iterable[PairwiseJudgment],
    *,
    alpha: float = 0.05,
    confidence: float = 0.95,
    pair_kinds: Sequence[PairKind] = ALL_PAIR_KINDS,
) -> PositionBiasReport:
    """Test every annotator for a left/right preference.

    All pair kinds count by default: gold and control pairs are shown in a
    randomised order too, so they are valid evidence about position.
    """
    kinds = tuple(sorted(set(pair_kinds)))
    per_annotator: dict[str, Counter[Choice]] = defaultdict(Counter)
    pooled: Counter[Choice] = Counter()
    for judgment in judgments:
        if judgment.pair_kind not in kinds:
            continue
        per_annotator[judgment.annotator_id][judgment.choice] += 1
        pooled[judgment.choice] += 1
    annotators = sorted(per_annotator)
    lefts = [per_annotator[a][Choice.LEFT] for a in annotators]
    decisive = [
        left + per_annotator[a][Choice.RIGHT] for a, left in zip(annotators, lefts, strict=True)
    ]
    raw = [binom_test(left, n) for left, n in zip(lefts, decisive, strict=True)]
    adjusted = holm_estimable(raw, [n > 0 for n in decisive])
    results = tuple(
        _result(a, per_annotator[a], p_adjusted=adj, alpha=alpha, confidence=confidence)
        for a, adj in zip(annotators, adjusted, strict=True)
    )
    return PositionBiasReport(
        alpha=alpha,
        confidence=confidence,
        pair_kinds=kinds,
        results=results,
        pooled=_result(None, pooled, p_adjusted=None, alpha=alpha, confidence=confidence),
    )
