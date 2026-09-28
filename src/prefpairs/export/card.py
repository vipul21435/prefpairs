"""Dataset card (JSON plus Markdown) for one export.

The card records what a reader needs to trust or reproduce the files: where
the responses came from, which annotators were excluded and why the pairs
that were dropped were dropped, how votes were spread over the kept pairs,
how prompts fell into splits, and the sha256 of every file. It carries no
timestamp, so a rerun on the same data writes byte-identical cards.
"""

import json
from collections import Counter
from collections.abc import Iterable, Sequence

from pydantic import Field

from prefpairs import __version__
from prefpairs.export.votes import FilterConfig, Selection
from prefpairs.export.writers import SPLITS, ExportFormat, SplitConfig, WrittenFile, split_of
from prefpairs.schema import Record, Response


class SplitStats(Record):
    split: str
    n_prompts: int
    n_pairs: int


class DatasetCard(Record):
    """Everything the Markdown card shows, in JSON-ready form."""

    prefpairs_version: str
    format: ExportFormat
    source: str
    n_judgments: int
    n_annotators: int
    excluded_annotators: tuple[str, ...]
    exclusion_reason: str
    filters: FilterConfig
    split_config: SplitConfig
    n_pairs: int
    n_kept: int
    drop_counts: dict[str, int]
    n_votes_counted: int
    n_votes_excluded: int
    mean_votes_per_kept_pair: float | None
    mean_agreement_kept: float | None
    models: tuple[str, ...]
    response_sources: dict[str, int]
    generators: dict[str, int]
    splits: tuple[SplitStats, ...]
    files: tuple[WrittenFile, ...] = Field(default=())


def build_card(
    *,
    fmt: ExportFormat,
    selection: Selection,
    responses: Sequence[Response],
    split: SplitConfig,
    files: Iterable[WrittenFile],
    source: str,
    n_judgments: int,
    n_annotators: int,
    exclusion_reason: str,
) -> DatasetCard:
    kept = selection.kept
    used = {rid for v in kept for rid in (v.a_id, v.b_id)}
    used_responses = [r for r in responses if r.id in used]
    prompts_per_split: dict[str, set[str]] = {name: set() for name in SPLITS}
    pairs_per_split: Counter[str] = Counter()
    for v in kept:
        name = split_of(v.prompt_id, split)
        prompts_per_split[name].add(v.prompt_id)
        pairs_per_split[name] += 1
    return DatasetCard(
        prefpairs_version=__version__,
        format=fmt,
        source=source,
        n_judgments=n_judgments,
        n_annotators=n_annotators,
        excluded_annotators=tuple(sorted(selection.config.exclude_annotators)),
        exclusion_reason=exclusion_reason,
        filters=selection.config,
        split_config=split,
        n_pairs=len(selection.decisions),
        n_kept=len(kept),
        drop_counts=selection.drop_counts(),
        n_votes_counted=sum(d.votes.n_votes for d in selection.decisions),
        n_votes_excluded=sum(d.votes.n_excluded for d in selection.decisions),
        mean_votes_per_kept_pair=sum(v.n_votes for v in kept) / len(kept) if kept else None,
        mean_agreement_kept=(sum(v.agreement or 0.0 for v in kept) / len(kept) if kept else None),
        models=tuple(sorted({r.model for r in used_responses})),
        response_sources=dict(
            sorted(Counter(r.provenance.source.value for r in used_responses).items())
        ),
        generators=dict(
            sorted(Counter(r.provenance.generator or "-" for r in used_responses).items())
        ),
        splits=tuple(
            SplitStats(
                split=name, n_prompts=len(prompts_per_split[name]), n_pairs=pairs_per_split[name]
            )
            for name in SPLITS
        ),
        files=tuple(files),
    )


def card_json(card: DatasetCard) -> bytes:
    return (json.dumps(card.model_dump(mode="json"), indent=2, sort_keys=True) + "\n").encode()


def _num(value: float | None, spec: str = ".3f") -> str:
    return "-" if value is None else format(value, spec)


def card_markdown(card: DatasetCard) -> str:
    """The card as Markdown, ASCII apart from whatever the source name contains."""
    f = card.filters
    s = card.split_config
    lines = [
        f"# Preference dataset card ({card.format.value})",
        "",
        f"Exported by prefpairs {card.prefpairs_version} from `{card.source}`: "
        f"{card.n_judgments} pairwise judgments by {card.n_annotators} annotators.",
        "",
        "## Filters",
        "",
        f"- pair kinds: {', '.join(k.value for k in f.pair_kinds)}",
        f"- min votes per pair: {f.min_votes}",
        f"- min agreement (winner's share of counted votes): {f.min_agreement}",
        f"- drop ties: {'yes' if f.drop_ties else 'no'}",
        f"- excluded annotators: {', '.join(card.excluded_annotators) or 'none'}"
        + (f" ({card.exclusion_reason})" if card.excluded_annotators else ""),
        "",
        "## Pairs",
        "",
        f"{card.n_kept} of {card.n_pairs} pairs kept; "
        f"{card.n_votes_counted} votes counted, {card.n_votes_excluded} set aside from "
        "excluded annotators.",
        "",
        "| drop reason | pairs |",
        "| --- | --- |",
        *(f"| {reason} | {n} |" for reason, n in card.drop_counts.items()),
        "",
        f"Kept pairs: {_num(card.mean_votes_per_kept_pair, '.2f')} votes on average, "
        f"mean agreement {_num(card.mean_agreement_kept)}.",
        "",
        "## Provenance",
        "",
        f"- models: {', '.join(card.models) or 'none'}",
        "- response sources: "
        + (", ".join(f"{k} {v}" for k, v in card.response_sources.items()) or "none"),
        "- generators: " + (", ".join(f"{k} {v}" for k, v in card.generators.items()) or "none"),
        "",
        "## Splits",
        "",
        f"Prompt-level, by sha256 of `{s.salt}:<prompt_id>`: train {s.train}, "
        f"validation {s.validation}, test {s.test}. No prompt appears in two splits.",
        "",
        "| split | prompts | pairs |",
        "| --- | --- | --- |",
        *(f"| {x.split} | {x.n_prompts} | {x.n_pairs} |" for x in card.splits),
        "",
        "## Files",
        "",
        "| file | rows | sha256 |",
        "| --- | --- | --- |",
        *(f"| {x.path} | {x.n_rows} | `{x.sha256}` |" for x in card.files),
        "",
    ]
    return "\n".join(lines)
