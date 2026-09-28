"""One export: select pairs, write the split files, then the card that hashes them."""

from collections.abc import Sequence
from pathlib import Path

from prefpairs.export.card import DatasetCard, build_card, card_json, card_markdown
from prefpairs.export.votes import FilterConfig, select_pairs
from prefpairs.export.writers import ExportFormat, SplitConfig, build_rows, write_splits
from prefpairs.schema import PairwiseJudgment, Prompt, Response


def export_dataset(
    out_dir: Path,
    fmt: ExportFormat,
    *,
    judgments: Sequence[PairwiseJudgment],
    prompts: Sequence[Prompt],
    responses: Sequence[Response],
    filters: FilterConfig | None = None,
    split: SplitConfig | None = None,
    source: str = "-",
    exclusion_reason: str = "given",
) -> DatasetCard:
    """Write ``<out_dir>/<fmt>/{train,validation,test}.jsonl`` plus ``card.json`` and
    ``card.md`` in the same folder, and return the card."""
    filters = filters or FilterConfig()
    split = split or SplitConfig()
    selection = select_pairs(judgments, filters)
    rows = build_rows(fmt, selection.kept, prompts, responses, split)
    files = write_splits(out_dir, fmt, rows)
    card = build_card(
        fmt=fmt,
        selection=selection,
        responses=responses,
        split=split,
        files=files,
        source=source,
        n_judgments=len(judgments),
        n_annotators=len({j.annotator_id for j in judgments}),
        exclusion_reason=exclusion_reason,
    )
    folder = out_dir / fmt.value
    (folder / "card.json").write_bytes(card_json(card))
    (folder / "card.md").write_text(card_markdown(card), encoding="utf-8")
    return card
