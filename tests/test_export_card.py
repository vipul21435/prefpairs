"""Dataset card: provenance, filter settings, drop counts, split stats and checksums."""

import hashlib
import json
from pathlib import Path

from prefpairs.export import ExportFormat, FilterConfig, export_dataset
from prefpairs.simulate import SimulationConfig, simulate


def run(out: Path, fmt: ExportFormat = ExportFormat.DPO, **filters: object):  # type: ignore[no-untyped-def]
    dataset = simulate(SimulationConfig(seed=2, n_prompts=30))
    return export_dataset(
        out,
        fmt,
        judgments=dataset.pairwise,
        prompts=dataset.prompts,
        responses=dataset.responses,
        filters=FilterConfig.model_validate(filters),
        source="seed-2 simulation",
        exclusion_reason="flagged by the audit",
    )


def test_card_hashes_every_file_and_counts_add_up(tmp_path: Path) -> None:
    card = run(tmp_path, min_votes=2, exclude_annotators=("ann-01",))
    for written in card.files:
        data = (tmp_path / written.path).read_bytes()
        assert hashlib.sha256(data).hexdigest() == written.sha256
        assert data.count(b"\n") == written.n_rows
    assert sum(f.n_rows for f in card.files) == card.n_kept
    assert sum(s.n_pairs for s in card.splits) == card.n_kept
    assert card.n_votes_excluded > 0
    assert card.excluded_annotators == ("ann-01",)
    assert card.filters.min_votes == 2
    assert card.response_sources
    stored = json.loads((tmp_path / "dpo" / "card.json").read_text())
    assert stored["n_kept"] == card.n_kept
    assert stored["files"][0]["sha256"] == card.files[0].sha256


def test_markdown_card_names_filters_drops_and_hashes(tmp_path: Path) -> None:
    card = run(tmp_path, fmt=ExportFormat.KTO, min_votes=2)
    text = (tmp_path / "kto" / "card.md").read_text()
    assert text.startswith("# Preference dataset card (kto)")
    assert "- min votes per pair: 2" in text
    assert "- excluded annotators: none" in text
    assert f"{card.n_kept} of {card.n_pairs} pairs kept" in text
    for written in card.files:
        assert f"| {written.path} | {written.n_rows} | `{written.sha256}` |" in text
    assert text.isascii()


def test_rerun_writes_byte_identical_files_and_cards(tmp_path: Path) -> None:
    for fmt in ExportFormat:
        run(tmp_path / "one", fmt=fmt)
        run(tmp_path / "two", fmt=fmt)
        names = sorted(p.name for p in (tmp_path / "one" / fmt.value).iterdir())
        assert names == ["card.json", "card.md", "test.jsonl", "train.jsonl", "validation.jsonl"]
        for name in names:
            one = (tmp_path / "one" / fmt.value / name).read_bytes()
            assert one == (tmp_path / "two" / fmt.value / name).read_bytes()
