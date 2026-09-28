"""The combined annotator audit behind ``prefpairs audit``.

Each check answers one question about an annotator; the audit runs all of
them with thresholds from one ``AuditConfig`` and turns the results into a
per-annotator verdict: flagged or not, which checks flagged them, and a
one-line reason with the evidence for each flag.

| flag            | raised when                                                        |
| --------------- | ------------------------------------------------------------------ |
| position        | left/right split fails the exact binomial test (Holm, ``alpha``)   |
| length          | length slope fails the likelihood-ratio test (Holm, ``alpha``)     |
| consistency     | upper Wilson bound of control agreement is below 0.5              |
| intransitive    | preference graph at least as cyclic as the median coin flipper    |
| gold            | upper Wilson bound of gold accuracy is below ``min_gold_accuracy`` |
| spammer         | Dawid-Skene spammer score below ``min_spammer_score``              |
| reversed        | informative, but Dawid-Skene accuracy below ``min_ds_accuracy``    |
"""

from collections.abc import Callable

from pydantic import Field

from prefpairs.aggregate.comparisons import Level
from prefpairs.quality.checks import QualityChecks, run_checks
from prefpairs.quality.spam import (
    DEFAULT_MIN_ACCURACY,
    DEFAULT_MIN_GOLD_ACCURACY,
    DEFAULT_MIN_SPAMMER_SCORE,
    DawidSkeneReport,
    GoldReport,
    annotator_reliability,
    gold_accuracy,
)
from prefpairs.quality.transitivity import TransitivityReport, transitivity
from prefpairs.schema import Record
from prefpairs.simulate import SimulationTruth
from prefpairs.store import Store

FLAG_ORDER = (
    "position",
    "length",
    "consistency",
    "intransitive",
    "gold",
    "spammer",
    "reversed",
)


class AuditConfig(Record):
    """Every threshold the audit uses, recorded with its result."""

    alpha: float = Field(default=0.05, gt=0, lt=1)
    confidence: float = Field(default=0.95, gt=0, lt=1)
    adjust_for_quality: bool = True
    min_gold_accuracy: float = Field(default=DEFAULT_MIN_GOLD_ACCURACY, ge=0, le=1)
    min_spammer_score: float = Field(default=DEFAULT_MIN_SPAMMER_SCORE, ge=0, le=1)
    min_ds_accuracy: float = Field(default=DEFAULT_MIN_ACCURACY, ge=0, le=1)
    min_ds_labels: int = Field(default=20, ge=1)
    transitivity_level: Level = Level.MODEL
    min_triads: int = Field(default=10, ge=1)
    chance_level: float = Field(default=0.5, gt=0, lt=1)
    random_draws: int = Field(default=2000, ge=100)
    seed: int = Field(default=0, ge=0)


class AnnotatorAudit(Record):
    """The verdict on one annotator."""

    annotator_id: str
    flagged: bool
    flags: tuple[str, ...]
    reasons: tuple[str, ...]
    """One line of evidence per flag, in ``flags`` order."""
    planted: str | None
    """Planted archetype when the data was simulated."""


class AuditReport(Record):
    """Every check, and the per-annotator verdicts built from them."""

    config: AuditConfig
    n_judgments: int
    checks: QualityChecks
    transitivity: TransitivityReport
    gold: GoldReport
    reliability: DawidSkeneReport
    annotators: tuple[AnnotatorAudit, ...]

    @property
    def flagged(self) -> tuple[str, ...]:
        return tuple(a.annotator_id for a in self.annotators if a.flagged)


def _p(value: float) -> str:
    return f"{value:.1e}" if value < 1e-3 else f"{value:.3f}"


def _reasons(report: AuditReport) -> dict[str, dict[str, str]]:
    """``{annotator: {flag: reason}}`` for every raised flag."""
    checks, gold, ds = report.checks, report.gold, report.reliability
    raised: list[tuple[str | None, str, str]] = [
        *(
            (
                r.annotator_id,
                "position",
                f"left {r.n_left} of {r.n_decisive} decisive choices, p adj {_p(r.p_adjusted)}",
            )
            for r in checks.position.results
            if r.flagged
        ),
        *(
            (
                r.annotator_id,
                "length",
                f"length slope {r.coefficient:+.2f} log-odds per unit "
                f"log word ratio, p adj {_p(r.p_adjusted)}",
            )
            for r in checks.length.results
            if r.flagged
        ),
        *(
            (
                r.annotator_id,
                "consistency",
                f"repeated own label {r.n_consistent} of "
                f"{r.n_compared}, upper bound {r.ci_upper:.2f}",
            )
            for r in checks.consistency.results
            if r.flagged
        ),
        *(
            (
                r.annotator_id,
                "intransitive",
                f"{r.n_cycles} of {r.n_triads} triads cyclic "
                f"(coin flips: {r.random_rate:.2f}), P(at most as cyclic) {r.p_value:.2f}",
            )
            for r in report.transitivity.results
            if r.flagged
        ),
        *(
            (
                r.annotator_id,
                "gold",
                f"gold accuracy {r.n_correct}/{r.n_answered}, "
                f"upper bound {r.ci_upper:.2f} < {gold.min_accuracy}",
            )
            for r in gold.results
            if r.flagged
        ),
        *(
            (
                r.annotator_id,
                "spammer",
                f"spammer score {r.spammer_score:.3f} < "
                f"{ds.min_spammer_score} over {r.n_labels} labels",
            )
            for r in ds.results
            if r.spammer
        ),
        *(
            (
                r.annotator_id,
                "reversed",
                f"estimated accuracy {r.accuracy:.2f} < {ds.min_accuracy} over {r.n_labels} labels",
            )
            for r in ds.results
            if r.reversed
        ),
    ]
    found: dict[str, dict[str, str]] = {}
    for annotator, flag, reason in raised:
        if annotator is not None:
            found.setdefault(annotator, {})[flag] = reason
    return found


def run_audit(
    store: Store,
    config: AuditConfig | None = None,
    *,
    truth: SimulationTruth | None = None,
) -> AuditReport:
    """Run every annotator check on a store and combine them into verdicts."""
    config = config or AuditConfig()
    judgments = store.pairwise()
    responses = store.responses()
    checks = run_checks(
        store,
        alpha=config.alpha,
        confidence=config.confidence,
        adjust_for_quality=config.adjust_for_quality,
        truth=truth,
    )
    partial = AuditReport(
        config=config,
        n_judgments=len(judgments),
        checks=checks,
        transitivity=transitivity(
            judgments,
            responses,
            level=config.transitivity_level,
            chance_level=config.chance_level,
            min_triads=config.min_triads,
            n_draws=config.random_draws,
            seed=config.seed,
        ),
        gold=gold_accuracy(
            judgments,
            store.gold_pairs(),
            confidence=config.confidence,
            min_accuracy=config.min_gold_accuracy,
        ),
        reliability=annotator_reliability(
            judgments,
            min_labels=config.min_ds_labels,
            min_spammer_score=config.min_spammer_score,
            min_accuracy=config.min_ds_accuracy,
        ),
        annotators=(),
    )
    reasons = _reasons(partial)
    everyone = sorted({j.annotator_id for j in judgments})
    archetypes = checks.archetypes or {}
    verdicts = []
    for annotator in everyone:
        raised = reasons.get(annotator, {})
        flags = tuple(f for f in FLAG_ORDER if f in raised)
        verdicts.append(
            AnnotatorAudit(
                annotator_id=annotator,
                flagged=bool(flags),
                flags=flags,
                reasons=tuple(raised[f] for f in flags),
                planted=archetypes.get(annotator),
            )
        )
    return partial.model_copy(update={"annotators": tuple(verdicts)})


def _cell[T](lookup: dict[str, T], annotator: str, show: Callable[[T], str]) -> str:
    item = lookup.get(annotator)
    return "-" if item is None else show(item)


def render_audit(report: AuditReport) -> str:
    """Plain-text audit: one row per annotator, then the reason for every flag."""
    config = report.config
    gold = {r.annotator_id: r for r in report.gold.results}
    ds = {r.annotator_id: r for r in report.reliability.results}
    trans = {r.annotator_id: r for r in report.transitivity.results}
    pct = f"{config.confidence:.0%}"
    lines = [
        f"Annotator audit of {report.n_judgments} pairwise judgments from "
        f"{len(report.annotators)} annotators",
        f"alpha {config.alpha} (Holm) for position and length; {pct} Wilson intervals",
        "",
        f"gold:   accuracy on gold pairs (flag if the {pct} upper bound < "
        f"{config.min_gold_accuracy})",
        f"DS:     Dawid-Skene accuracy and spammer score (spammer if score < "
        f"{config.min_spammer_score}, reversed if accuracy < {config.min_ds_accuracy})",
        f"cycles: cyclic / complete triads in the {config.transitivity_level.value}-level "
        f"preference graph (coin flips: 0.25)",
        "",
    ]
    flag_text = {a.annotator_id: "+".join(a.flags) or "-" for a in report.annotators}
    width = max([len("flags"), *(len(t) for t in flag_text.values())])
    planted = any(a.planted for a in report.annotators)
    header = (
        f"{'annotator':<10}  {'gold':>6}  {pct + ' CI':>12}  {'DS acc':>6}  {'spam':>5}  "
        f"{'cycles':>6}  {'flags':<{width}}"
    )
    lines.append((header + ("  planted" if planted else "")).rstrip())
    for audit in report.annotators:
        a = audit.annotator_id
        row = (
            f"{a:<10}  {_cell(gold, a, lambda g: f'{g.n_correct}/{g.n_answered}'):>6}  "
            f"{_cell(gold, a, lambda g: f'[{g.ci_lower:.2f}, {g.ci_upper:.2f}]'):>12}  "
            f"{_cell(ds, a, lambda d: f'{d.accuracy:.2f}'):>6}  "
            f"{_cell(ds, a, lambda d: f'{d.spammer_score:.2f}'):>5}  "
            f"{_cell(trans, a, lambda t: f'{t.n_cycles}/{t.n_triads}'):>6}  "
            f"{flag_text[a]:<{width}}"
        )
        if planted:
            row += f"  {audit.planted or '?'}"
        lines.append(row.rstrip())
    lines.append("")
    if report.gold.n_unmatched:
        lines += [
            f"note: {report.gold.n_unmatched} gold judgments match no gold pair and were "
            "not scored",
            "",
        ]
    for audit in report.annotators:
        for flag, reason in zip(audit.flags, audit.reasons, strict=True):
            lines.append(f"{audit.annotator_id} {flag}: {reason}")
    summary = ", ".join(
        f"{a.annotator_id} ({'+'.join(a.flags)})" for a in report.annotators if a.flagged
    )
    if report.annotators and any(a.flagged for a in report.annotators):
        lines.append("")
    lines.append(f"flagged: {summary or 'none'}")
    return "\n".join(lines)
