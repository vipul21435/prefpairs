"""Length (verbosity) bias: do an annotator's choices follow response length?

For every decisive judgment the check fits the logistic model

    logit P(left) = b_pos + b_len * log(words_left / words_right)
                          + b_q * (theta_left - theta_right)

by Newton/IRLS (``prefpairs.quality.logistic``). ``b_len`` is the length
effect: the change in log-odds of picking a response per unit of log length
ratio (``b_len = 1`` means a response twice as long gets ``e^0.69 = 2x`` the
odds). ``b_pos`` is the position effect at equal length and quality.

``theta`` is the consensus quality of each response's model: Bradley-Terry
strengths fitted on the *other* annotators' regular judgments (leave one
annotator out, so an annotator's own choices never explain themselves). In
real data longer answers are often genuinely better; without this covariate a
careful annotator who rewards quality would look length-biased. The adjustment
can be turned off (``adjust_for_quality=False``) to fit choice on length alone.

Length is counted in whitespace-separated words by default. Scripts written
without spaces between words (Chinese, Japanese, Thai) make a whole response
one "word", so every pair would look equal-length and the check could never
run. With ``unit="auto"`` (the default) the check therefore measures length in
characters when words cannot tell most pairs apart: when more decisive pairs
tie on words but differ in characters than differ in words. The unit used is
recorded on the report.

The test of ``b_len = 0`` is a likelihood-ratio test (robust to separation,
where the Wald test breaks down), adjusted across annotators with
Holm-Bonferroni. A pooled fit over every annotator is reported too.
"""

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from pydantic import Field

from prefpairs.aggregate import (
    ComparisonOptions,
    Level,
    build_comparisons,
    fit_bradley_terry,
)
from prefpairs.quality.logistic import fit_logistic, likelihood_ratio
from prefpairs.quality.stats import holm_estimable
from prefpairs.schema import Choice, PairKind, PairwiseJudgment, Record, Response

LENGTH_COLUMN = 1
MIN_DECISIVE = 10
INTERCEPT_RIDGE = 1e-8
SLOPE_RIDGE = 1.0
"""Normal prior with unit variance on the slopes (a slope of 1 is a big effect),
which keeps sharp annotators' quality coefficients finite (see ``_fit``)."""


class LengthUnit(StrEnum):
    """How response length is counted."""

    AUTO = "auto"
    """Words, unless words cannot tell most pairs apart (see the module docstring)."""
    WORDS = "words"
    CHARS = "chars"


def _size(response: Response, unit: LengthUnit) -> int:
    return max(1, response.n_chars if unit is LengthUnit.CHARS else response.n_words)


def resolve_unit(
    unit: LengthUnit,
    judgments: Iterable[PairwiseJudgment],
    responses: Mapping[str, Response],
) -> LengthUnit:
    """The concrete unit for ``unit``: characters when ``auto`` and words are uninformative."""
    if unit is not LengthUnit.AUTO:
        return unit
    word_gaps = word_ties_with_char_gap = 0
    for j in judgments:
        left, right = responses[j.left_response_id], responses[j.right_response_id]
        if left.n_words != right.n_words:
            word_gaps += 1
        elif left.n_chars != right.n_chars:
            word_ties_with_char_gap += 1
    return LengthUnit.CHARS if word_ties_with_char_gap > word_gaps else LengthUnit.WORDS


class LengthBiasResult(Record):
    """The fitted length effect of one annotator (or of everyone, when pooled)."""

    annotator_id: str | None
    n_decisive: int
    n_longer_chosen: int
    n_shorter_chosen: int
    n_equal_length: int
    coefficient: float | None
    """``b_len``: log-odds per unit log length ratio; None when it cannot be estimated."""
    std_error: float | None
    ci_lower: float | None
    ci_upper: float | None
    """Wald interval for ``b_len`` (unreliable when ``separated``)."""
    position_coefficient: float | None
    """``b_pos``: log-odds of choosing left at equal length and quality."""
    quality_coefficient: float | None
    """``b_q``: weight on the consensus quality gap (None when not adjusted)."""
    lr_statistic: float | None
    p_value: float
    """Likelihood-ratio test of ``b_len = 0``; 1.0 when not estimable."""
    p_adjusted: float
    separated: bool
    flagged: bool


class LengthBiasReport(Record):
    """Per-annotator length-bias fits and the pooled fit."""

    alpha: float = Field(gt=0, lt=1)
    confidence: float = Field(gt=0, lt=1)
    adjust_for_quality: bool
    min_decisive: int
    pair_kinds: tuple[PairKind, ...]
    slope_ridge: float
    unit: LengthUnit
    """Unit the length ratio was measured in (never ``auto``)."""
    distrusted: tuple[str, ...]
    """Annotators left out of the second-round consensus (flagged in round one)."""
    results: tuple[LengthBiasResult, ...]
    pooled: LengthBiasResult

    @property
    def flagged(self) -> tuple[str, ...]:
        return tuple(sorted(r.annotator_id for r in self.results if r.flagged and r.annotator_id))


@dataclass(frozen=True, slots=True)
class _FitOptions:
    min_decisive: int
    confidence: float
    slope_ridge: float
    unit: LengthUnit


def consensus_strengths(
    judgments: Sequence[PairwiseJudgment], responses: Sequence[Response]
) -> dict[str, float]:
    """Bradley-Terry model strengths from regular judgments ({} when there are none)."""
    comparisons = build_comparisons(judgments, responses, ComparisonOptions(level=Level.MODEL))
    if comparisons.n_games == 0:
        return {}
    fit = fit_bradley_terry(comparisons)
    return {item: float(v) for item, v in zip(fit.items, fit.log_strengths, strict=True)}


def _fit(
    annotator_id: str | None,
    judgments: Sequence[PairwiseJudgment],
    responses: Mapping[str, Response],
    strengths: Mapping[str, float] | None,
    options: _FitOptions,
) -> tuple[LengthBiasResult, bool]:
    """Fit one annotator; the flag is filled in after the Holm adjustment.

    The slopes carry a ridge penalty (a normal prior, ``slope_ridge`` = 1 /
    variance). A careful annotator follows consensus quality so closely that
    the data are nearly separated along it; with a vanishing penalty its
    coefficient then runs off and the likelihood-ratio test of the length
    slope loses its calibration. The penalty shrinks the tested slope too, so
    the test errs on the side of not flagging.
    """
    ratios: list[float] = []
    gaps: list[float] = []
    chose_left: list[float] = []
    longer = shorter = equal = 0
    for j in judgments:
        left, right = responses[j.left_response_id], responses[j.right_response_id]
        ratio = math.log(_size(left, options.unit) / _size(right, options.unit))
        picked_left = j.choice is Choice.LEFT
        if ratio == 0.0:
            equal += 1
        elif (ratio > 0.0) == picked_left:
            longer += 1
        else:
            shorter += 1
        ratios.append(ratio)
        chose_left.append(1.0 if picked_left else 0.0)
        if strengths is not None:
            gaps.append(strengths.get(left.model, 0.0) - strengths.get(right.model, 0.0))
    empty = LengthBiasResult(
        annotator_id=annotator_id,
        n_decisive=len(judgments),
        n_longer_chosen=longer,
        n_shorter_chosen=shorter,
        n_equal_length=equal,
        coefficient=None,
        std_error=None,
        ci_lower=None,
        ci_upper=None,
        position_coefficient=None,
        quality_coefficient=None,
        lr_statistic=None,
        p_value=1.0,
        p_adjusted=1.0,
        separated=False,
        flagged=False,
    )
    x_len = np.array(ratios)
    if len(judgments) < options.min_decisive or np.ptp(x_len) == 0.0:
        return empty, False
    columns = [np.ones(len(x_len)), x_len]
    use_quality = strengths is not None and np.ptp(np.array(gaps)) > 0.0
    if use_quality:
        columns.append(np.array(gaps))
    X = np.column_stack(columns)
    y = np.array(chose_left)
    ridge = np.full(X.shape[1], options.slope_ridge)
    ridge[0] = INTERCEPT_RIDGE
    fit = fit_logistic(X, y, ridge=ridge)
    statistic, p_value = likelihood_ratio(X, y, LENGTH_COLUMN, full=fit, ridge=ridge)
    lower, upper = fit.wald_interval(LENGTH_COLUMN, options.confidence)
    result = empty.model_copy(
        update={
            "coefficient": float(fit.coef[LENGTH_COLUMN]),
            "std_error": float(fit.std_error[LENGTH_COLUMN]),
            "ci_lower": lower,
            "ci_upper": upper,
            "position_coefficient": float(fit.coef[0]),
            "quality_coefficient": float(fit.coef[2]) if use_quality else None,
            "lr_statistic": statistic,
            "p_value": p_value,
            "p_adjusted": p_value,
            "separated": fit.separated,
        }
    )
    return result, True


def _round(
    decisive: Mapping[str, list[PairwiseJudgment]],
    all_judgments: Sequence[PairwiseJudgment],
    responses: Sequence[Response],
    *,
    adjust_for_quality: bool,
    distrusted: frozenset[str],
    alpha: float,
    options: _FitOptions,
) -> tuple[LengthBiasResult, ...]:
    """Fit and Holm-adjust every annotator once, with a consensus that leaves out
    the annotator itself and every ``distrusted`` one."""
    by_id = {r.id: r for r in responses}
    fitted: list[tuple[LengthBiasResult, bool]] = []
    for annotator in sorted(decisive):
        strengths = None
        if adjust_for_quality:
            left_out = distrusted | {annotator}
            others = [j for j in all_judgments if j.annotator_id not in left_out]
            strengths = consensus_strengths(others, responses)
        fitted.append(_fit(annotator, decisive[annotator], by_id, strengths, options))
    adjusted = holm_estimable([r.p_value for r, _ in fitted], [e for _, e in fitted])
    return tuple(
        r.model_copy(update={"p_adjusted": adj, "flagged": estimable and adj <= alpha})
        for (r, estimable), adj in zip(fitted, adjusted, strict=True)
    )


def length_bias(
    judgments: Iterable[PairwiseJudgment],
    responses: Iterable[Response],
    *,
    alpha: float = 0.05,
    confidence: float = 0.95,
    adjust_for_quality: bool = True,
    min_decisive: int = MIN_DECISIVE,
    pair_kinds: Sequence[PairKind] = (PairKind.REGULAR,),
    slope_ridge: float = SLOPE_RIDGE,
    unit: LengthUnit = LengthUnit.AUTO,
) -> LengthBiasReport:
    """Fit the length effect of every annotator and test it.

    Only regular pairs count by default: gold pairs are one small set, chosen
    for a large response-level quality gap and shown to everyone, and controls
    repeat judgments already counted, so both would add shared or duplicated
    evidence that the model treats as independent.

    With ``adjust_for_quality`` the fit runs twice. The consensus of the first
    round is contaminated by the very annotators the check looks for (a
    length-driven annotator pulls verbose models up, which makes consensus
    quality partly a length covariate and pushes honest annotators' length
    coefficients negative), so the second round refits the consensus without
    the annotators flagged in the first.

    Annotators with fewer than ``min_decisive`` decisive judgments, or who only
    saw equal-length pairs, get ``coefficient=None`` and are never flagged. They
    are left out of the Holm family too (their ``p_adjusted`` is 1.0), so a long
    tail of low-volume annotators does not dilute the test of the others.
    """
    all_judgments = list(judgments)
    response_list = list(responses)
    kinds = tuple(sorted(set(pair_kinds)))
    decisive: dict[str, list[PairwiseJudgment]] = defaultdict(list)
    for j in all_judgments:
        if j.pair_kind in kinds and j.choice in (Choice.LEFT, Choice.RIGHT):
            decisive[j.annotator_id].append(j)
    by_id = {r.id: r for r in response_list}
    resolved = resolve_unit(unit, (j for rows in decisive.values() for j in rows), by_id)
    options = _FitOptions(min_decisive, confidence, slope_ridge, resolved)

    def one_round(distrusted: frozenset[str]) -> tuple[LengthBiasResult, ...]:
        return _round(
            decisive,
            all_judgments,
            response_list,
            adjust_for_quality=adjust_for_quality,
            distrusted=distrusted,
            alpha=alpha,
            options=options,
        )

    results = one_round(frozenset())
    first_flags = frozenset(r.annotator_id for r in results if r.flagged and r.annotator_id)
    if adjust_for_quality and first_flags:
        results = one_round(first_flags)
    trusted = [j for j in all_judgments if j.annotator_id not in first_flags]
    pooled, estimable = _fit(
        None,
        [j for rows in decisive.values() for j in rows],
        by_id,
        consensus_strengths(trusted, response_list) if adjust_for_quality else None,
        options,
    )
    pooled = pooled.model_copy(update={"flagged": estimable and pooled.p_value <= alpha})
    return LengthBiasReport(
        alpha=alpha,
        confidence=confidence,
        adjust_for_quality=adjust_for_quality,
        min_decisive=min_decisive,
        pair_kinds=kinds,
        slope_ridge=slope_ridge,
        unit=resolved,
        distrusted=tuple(sorted(first_flags)) if adjust_for_quality else (),
        results=results,
        pooled=pooled,
    )
