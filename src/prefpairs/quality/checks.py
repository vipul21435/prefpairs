"""Run every annotator check on a store and render one per-annotator table.

This is the glue behind ``prefpairs checks``. It reports statistics and the
flags each check raises on its own; combining them into a single verdict per
annotator (with transitivity, gold accuracy and spammer scores) is the job of
the audit report.
"""

from collections import defaultdict

from pydantic import Field

from prefpairs.quality.agreement import AgreementReport, agreement
from prefpairs.quality.consistency import ConsistencyReport, self_consistency
from prefpairs.quality.length import LengthBiasReport, length_bias
from prefpairs.quality.position import PositionBiasReport, position_bias
from prefpairs.schema import Record
from prefpairs.simulate import SimulationTruth
from prefpairs.store import Store


class QualityChecks(Record):
    """The four slice-3 checks on one dataset, in JSON-ready form."""

    alpha: float = Field(gt=0, lt=1)
    confidence: float = Field(gt=0, lt=1)
    n_judgments: int
    position: PositionBiasReport
    length: LengthBiasReport
    agreement: AgreementReport
    consistency: ConsistencyReport
    archetypes: dict[str, str] | None
    """Planted archetype of each annotator when the data was simulated."""

    def flags(self) -> dict[str, tuple[str, ...]]:
        """Checks that flagged each annotator, for annotators with at least one flag."""
        found: dict[str, list[str]] = defaultdict(list)
        for name, flagged in (
            ("position", self.position.flagged),
            ("length", self.length.flagged),
            ("consistency", self.consistency.flagged),
        ):
            for annotator in flagged:
                found[annotator].append(name)
        return {a: tuple(found[a]) for a in sorted(found)}


def run_checks(
    store: Store,
    *,
    alpha: float = 0.05,
    confidence: float = 0.95,
    adjust_for_quality: bool = True,
    truth: SimulationTruth | None = None,
) -> QualityChecks:
    """Position, length, agreement and self-consistency checks on every judgment."""
    judgments = store.pairwise()
    archetypes = None
    if truth is not None:
        archetypes = {a: k.value for a, k in sorted(truth.annotator_archetypes.items())}
    return QualityChecks(
        alpha=alpha,
        confidence=confidence,
        n_judgments=len(judgments),
        position=position_bias(judgments, alpha=alpha, confidence=confidence),
        length=length_bias(
            judgments,
            store.responses(),
            alpha=alpha,
            confidence=confidence,
            adjust_for_quality=adjust_for_quality,
        ),
        agreement=agreement(judgments),
        consistency=self_consistency(judgments, confidence=confidence),
        archetypes=archetypes,
    )


def _p(value: float) -> str:
    return f"{value:.1e}" if value < 1e-3 else f"{value:.3f}"


def _num(value: float | None, spec: str = "+.2f") -> str:
    return "-" if value is None else format(value, spec)


def render_checks(report: QualityChecks) -> str:
    """Plain-text table, stable enough to paste into a README."""
    position = {r.annotator_id: r for r in report.position.results}
    length = {r.annotator_id: r for r in report.length.results}
    kappa = {a.annotator_id: a for a in report.agreement.annotators}
    consistency = {r.annotator_id: r for r in report.consistency.results}
    keys = [*position, *length, *kappa, *consistency]
    annotators = sorted({a for a in keys if a is not None})
    flags = report.flags()
    pct = f"{report.confidence:.0%}"
    lines = [
        f"Annotator checks on {report.n_judgments} pairwise judgments from "
        f"{len(annotators)} annotators",
        f"alpha {report.alpha} after Holm-Bonferroni across annotators; {pct} intervals",
        "",
        "position: exact binomial test of left vs right on decisive choices",
        "length:   log-odds per unit log word ratio, likelihood-ratio test"
        + (", adjusted for consensus quality" if report.length.adjust_for_quality else ""),
        "kappa:    Cohen's kappa with each co-annotator, weighted by shared items",
        f"repeat:   same label on control repeats (flag if the {pct} upper bound < "
        f"{report.consistency.min_rate})",
        "",
    ]
    flag_text = {a: "+".join(flags.get(a, ())) or "-" for a in annotators}
    flag_width = max([len("flags"), *(len(t) for t in flag_text.values())])
    header = (
        f"{'annotator':<10}  {'left:right':>10}  {'p adj':>7}  {'length':>6}  {'p adj':>7}  "
        f"{'kappa':>6}  {'repeat':>6}  {pct + ' CI':>12}  {'flags':<{flag_width}}"
    )
    if report.archetypes is not None:
        header += "  planted"
    lines.append(header.rstrip())
    for annotator in annotators:
        pos = position.get(annotator)
        lb = length.get(annotator)
        ka = kappa.get(annotator)
        co = consistency.get(annotator)
        split = f"{pos.n_left}:{pos.n_right}" if pos else "-"
        repeat = f"{co.n_consistent}/{co.n_compared}" if co else "-"
        interval = f"[{co.ci_lower:.2f}, {co.ci_upper:.2f}]" if co else "-"
        line = (
            f"{annotator:<10}  {split:>10}  {_p(pos.p_adjusted) if pos else '-':>7}  "
            f"{_num(lb.coefficient if lb else None):>6}  {_p(lb.p_adjusted) if lb else '-':>7}  "
            f"{_num(ka.mean_kappa if ka else None, '+.3f'):>6}  {repeat:>6}  {interval:>12}  "
            f"{flag_text[annotator]:<{flag_width}}"
        )
        if report.archetypes is not None:
            line += f"  {report.archetypes.get(annotator, '?')}"
        lines.append(line.rstrip())
    pooled_pos = report.position.pooled
    pooled_len = report.length.pooled
    lines += [
        "",
        f"pooled: left:right {pooled_pos.n_left}:{pooled_pos.n_right} "
        f"(p {_p(pooled_pos.p_value)}), length {_num(pooled_len.coefficient)} "
        f"(p {_p(pooled_len.p_value)})",
    ]
    fleiss = report.agreement.fleiss
    if fleiss is not None:
        lines.append(
            f"Fleiss kappa over {fleiss.n_items} items with {fleiss.n_raters} labels each "
            f"({fleiss.n_items_excluded} items with another count left out): "
            f"{_num(fleiss.kappa, '.3f')}"
        )
    summary = ", ".join(f"{a} ({'+'.join(c)})" for a, c in flags.items()) or "none"
    lines.append(f"flagged: {summary}")
    return "\n".join(lines)
