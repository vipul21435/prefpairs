"""Descriptive summary of a store: what is in it, before any statistics run."""

from collections import Counter

from prefpairs.schema import Choice, PairKind, Record
from prefpairs.simulate import load_truth
from prefpairs.store import Store


class StoreSummary(Record):
    """Counts that describe a database; rendered by ``prefpairs stats``."""

    path: str
    schema_version: int
    counts: dict[str, int]
    choices: dict[str, int]
    pair_kinds: dict[str, int]
    responses_per_model: dict[str, int]
    pairwise_per_annotator_min: int
    pairwise_per_annotator_max: int
    rationale_share: float
    simulated_seed: int | None


def summarise(store: Store) -> StoreSummary:
    """Collect descriptive counts from a store."""
    truth = load_truth(store)
    judgments = store.pairwise()
    choices = Counter(j.choice.value for j in judgments)
    kinds = Counter(j.pair_kind.value for j in judgments)
    per_annotator = Counter(j.annotator_id for j in judgments)
    for annotator in store.annotators():
        per_annotator.setdefault(annotator.id, 0)
    models = Counter(r.model for r in store.responses())
    with_rationale = sum(j.rationale is not None for j in judgments)
    return StoreSummary(
        path=store.path,
        schema_version=store.schema_version,
        counts=store.counts(),
        choices={c.value: choices[c.value] for c in Choice},
        pair_kinds={k.value: kinds[k.value] for k in PairKind},
        responses_per_model=dict(sorted(models.items())),
        pairwise_per_annotator_min=min(per_annotator.values(), default=0),
        pairwise_per_annotator_max=max(per_annotator.values(), default=0),
        rationale_share=with_rationale / len(judgments) if judgments else 0.0,
        simulated_seed=None if truth is None else truth.config.seed,
    )


def _block(title: str, rows: dict[str, int]) -> list[str]:
    width = max(16, *(len(key) for key in rows))
    return [title, *(f"  {key:<{width}}  {value:>8}" for key, value in rows.items())]


def render_summary(summary: StoreSummary) -> str:
    """Plain-text rendering, stable enough to paste into a README."""
    origin = (
        f"simulated, seed {summary.simulated_seed}"
        if summary.simulated_seed is not None
        else "not simulated"
    )
    lines = [f"Database {summary.path} (schema v{summary.schema_version}, {origin})", ""]
    lines += _block("Records", summary.counts)
    lines += ["", *_block("Pairwise choices", summary.choices)]
    lines += ["", *_block("Pair kinds", summary.pair_kinds)]
    lines += ["", *_block("Responses per model", summary.responses_per_model)]
    lines += [
        "",
        "Annotator load",
        f"  pairwise judgments per annotator: min {summary.pairwise_per_annotator_min}, "
        f"max {summary.pairwise_per_annotator_max}",
        f"  judgments with a rationale: {summary.rationale_share:.1%}",
    ]
    return "\n".join(lines)
