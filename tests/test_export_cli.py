"""`prefpairs export` on the bundled sample."""

import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from prefpairs.cli import app

runner = CliRunner()
SAMPLE = Path(__file__).resolve().parents[1] / "examples"
PLANTED = ["ann-06", "ann-08", "ann-10", "ann-11"]


@pytest.fixture(scope="module")
def sample_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    db = tmp_path_factory.mktemp("export") / "sample.db"
    result = runner.invoke(app, ["import", str(SAMPLE / "sample.jsonl"), "--db", str(db)])
    assert result.exit_code == 0, result.output
    return db


def export(db: Path, out: Path, *args: str) -> str:
    result = runner.invoke(app, ["export", "--db", str(db), "--out", str(out), *args])
    assert result.exit_code == 0, result.output
    return result.output


def test_default_export_leaves_out_the_audit_flags(sample_db: Path, tmp_path: Path) -> None:
    output = export(sample_db, tmp_path)
    assert "kept 142 of 160 pairs" in output
    assert "excluded annotators: ann-06, ann-08, ann-10, ann-11" in output
    card = json.loads((tmp_path / "dpo" / "card.json").read_text())
    assert card["filters"]["exclude_annotators"] == PLANTED
    assert card["exclusion_reason"] == "audit flags: ann-06, ann-08, ann-10, ann-11"
    for f in card["files"]:
        data = (tmp_path / f["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == f["sha256"]
    rows = [json.loads(x) for x in (tmp_path / "dpo" / "train.jsonl").read_text().splitlines()]
    assert set(rows[0]) >= {"prompt", "chosen", "rejected"}


@pytest.mark.parametrize("fmt", ["dpo", "kto", "rm"])
def test_every_format_is_byte_identical_on_rerun(sample_db: Path, tmp_path: Path, fmt: str) -> None:
    export(sample_db, tmp_path / "one", "--format", fmt, "--min-votes", "2")
    export(sample_db, tmp_path / "two", "--format", fmt, "--min-votes", "2")
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "card.json", "card.md"):
        one = (tmp_path / "one" / fmt / name).read_bytes()
        assert one == (tmp_path / "two" / fmt / name).read_bytes()


def test_without_the_audit_only_given_annotators_are_left_out(
    sample_db: Path, tmp_path: Path
) -> None:
    output = export(sample_db, tmp_path, "--no-audit", "-x", "ann-06", "--keep-ties")
    assert "excluded annotators: ann-06" in output
    card = json.loads((tmp_path / "dpo" / "card.json").read_text())
    assert card["exclusion_reason"] == "given with -x"
    assert card["filters"]["drop_ties"] is False


def test_audit_and_given_exclusions_are_both_recorded(sample_db: Path, tmp_path: Path) -> None:
    export(sample_db, tmp_path, "-x", "ann-01")
    card = json.loads((tmp_path / "dpo" / "card.json").read_text())
    assert card["filters"]["exclude_annotators"] == ["ann-01", *PLANTED]
    assert card["exclusion_reason"].endswith("; given with -x: ann-01")


def test_bad_split_fractions_are_an_error(sample_db: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "export",
            "--db",
            str(sample_db),
            "--out",
            str(tmp_path),
            "--train",
            "0.9",
            "--validation",
            "0.2",
        ],
    )
    assert result.exit_code == 1
    assert "error:" in result.output
