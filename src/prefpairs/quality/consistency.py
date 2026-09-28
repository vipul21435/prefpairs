"""Self-consistency on control repeats.

A control pair shows an annotator a pair they already judged, later and with
the sides swapped. A content-driven annotator gives the same canonical label
twice (up to their own noise); an annotator driven by position gives the
*opposite* label, because the response they prefer by content has moved to
the other side. So the check compares canonical labels of each control and
its original, reports the share that match with a Wilson interval, and also
counts ``n_same_side``: decisive repeats where the annotator clicked the same
side both times, the fingerprint of a position habit.

An annotator is flagged when an exact one-sided binomial test rejects a
repeat rate of ``min_rate`` in favour of a lower one (``P(X <= consistent)``
with ``X ~ Binomial(compared, min_rate)``), after Holm-Bonferroni across the
annotators with at least one compared control, at family-wise level
``alpha``. The default, 0.5, is the rate of an annotator who picks a side by
coin flip; staying clearly below it means the repeat systematically reverses
the first answer. The Wilson interval is reported for reading, not for the
flag.
"""

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from pydantic import Field

from prefpairs.quality.stats import binom_lower_tail, holm_estimable, wilson_interval
from prefpairs.schema import Choice, PairKind, PairwiseJudgment, Record

DEFAULT_MIN_RATE = 0.5


class ConsistencyResult(Record):
    """How often one annotator repeated their own label on control pairs."""

    annotator_id: str
    n_controls: int
    n_flipped: int
    """Controls shown with the sides swapped relative to the original."""
    n_compared: int
    """Controls where neither the original nor the repeat was skipped."""
    n_consistent: int
    n_same_side: int
    """Decisive repeats where the same side was clicked both times."""
    n_skipped: int
    rate: float | None
    ci_lower: float
    ci_upper: float
    p_value: float
    """Exact one-sided binomial test of a repeat rate below ``min_rate``."""
    p_adjusted: float
    """Holm-Bonferroni across annotators with compared controls (1.0 for the rest)."""
    flagged: bool


class ConsistencyReport(Record):
    alpha: float = Field(gt=0, lt=1)
    confidence: float = Field(gt=0, lt=1)
    min_rate: float = Field(ge=0, le=1)
    results: tuple[ConsistencyResult, ...]

    @property
    def flagged(self) -> tuple[str, ...]:
        return tuple(sorted(r.annotator_id for r in self.results if r.flagged))


_DECISIVE = (Choice.LEFT, Choice.RIGHT)


@dataclass(slots=True)
class _Tally:
    """Running counts of one annotator's controls."""

    controls: int = 0
    flipped: int = 0
    compared: int = 0
    consistent: int = 0
    same_side: int = 0
    skipped: int = 0


def self_consistency(
    judgments: Iterable[PairwiseJudgment],
    *,
    alpha: float = 0.05,
    confidence: float = 0.95,
    min_rate: float = DEFAULT_MIN_RATE,
) -> ConsistencyReport:
    """Compare every control judgment with the judgment it repeats.

    Raises KeyError if a control names a judgment that is not in the input.
    """
    all_judgments = list(judgments)
    by_id = {j.id: j for j in all_judgments}
    counts: dict[str, _Tally] = defaultdict(_Tally)
    for control in all_judgments:
        if control.pair_kind is not PairKind.CONTROL or control.repeat_of is None:
            continue
        original = by_id[control.repeat_of]
        tally = counts[control.annotator_id]
        tally.controls += 1
        tally.flipped += control.left_response_id == original.right_response_id
        if control.label is None or original.label is None:
            tally.skipped += 1
            continue
        tally.compared += 1
        tally.consistent += control.label is original.label
        if control.choice in _DECISIVE and original.choice in _DECISIVE:
            tally.same_side += control.choice is original.choice
    annotators = sorted(counts)
    raw = [binom_lower_tail(counts[a].consistent, counts[a].compared, min_rate) for a in annotators]
    adjusted = holm_estimable(raw, [counts[a].compared > 0 for a in annotators])
    results = []
    for annotator, p_value, p_adjusted in zip(annotators, raw, adjusted, strict=True):
        t = counts[annotator]
        lower, upper = wilson_interval(t.consistent, t.compared, confidence)
        results.append(
            ConsistencyResult(
                annotator_id=annotator,
                n_controls=t.controls,
                n_flipped=t.flipped,
                n_compared=t.compared,
                n_consistent=t.consistent,
                n_same_side=t.same_side,
                n_skipped=t.skipped,
                rate=t.consistent / t.compared if t.compared else None,
                ci_lower=lower,
                ci_upper=upper,
                p_value=p_value,
                p_adjusted=p_adjusted,
                flagged=t.compared > 0 and p_adjusted <= alpha,
            )
        )
    return ConsistencyReport(
        alpha=alpha, confidence=confidence, min_rate=min_rate, results=tuple(results)
    )
