"""The bundled sample data and the end-to-end demo script."""

import os
import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

from prefpairs.cli import app

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "examples" / "sample.jsonl"
SAMPLE_TRUTH = ROOT / "examples" / "sample-truth.json"
runner = CliRunner()


def test_bundled_sample_is_exactly_the_seed_0_simulation(tmp_path: Path) -> None:
    db, truth, dump = tmp_path / "s.db", tmp_path / "truth.json", tmp_path / "s.jsonl"
    args = ["simulate", "--seed", "0", "--db", str(db), "--truth-out", str(truth)]
    assert runner.invoke(app, args).exit_code == 0
    assert runner.invoke(app, ["dump", "--db", str(db), "--out", str(dump)]).exit_code == 0
    assert dump.read_bytes() == SAMPLE.read_bytes()
    assert truth.read_bytes() == SAMPLE_TRUTH.read_bytes()


def test_demo_script_runs_end_to_end(tmp_path: Path) -> None:
    cli = Path(sys.executable).parent / "prefpairs"
    env = {
        **os.environ,
        "PREFPAIRS": str(cli),
        "DEMO_DB": str(tmp_path / "demo.db"),
        "EXPORT_DIR": str(tmp_path / "export"),
    }
    result = subprocess.run(  # noqa: S603
        ["/bin/sh", str(ROOT / "scripts" / "demo.sh")],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("Kendall tau against the true order: 1.000") == 2
    assert "flagged: ann-10 (position), ann-11 (length)" in result.stdout
    assert (
        "flagged: ann-06 (gold+reversed), ann-08 (intransitive+spammer), "
        "ann-10 (position), ann-11 (length)"
    ) in result.stdout
    assert "exit code 1 (annotators flagged)" in result.stdout
    assert "rerun: every DPO file and card is byte-identical" in result.stdout
    for fmt in ("dpo", "kto", "rm"):
        assert (tmp_path / "export" / fmt / "card.md").is_file()
    assert result.stdout.rstrip().endswith(
        "demo OK: both rankings recover the true model order, the audit flags exactly "
        "the planted annotators and the export is reproducible"
    )


def test_detection_rates_script_prints_a_row_per_size_and_archetype() -> None:
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(ROOT / "scripts" / "detection_rates.py"), "--seeds", "1"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0].startswith("flagged / simulated annotators over seeds 0-0")
    assert len(lines) == 2 + 2 * 6
    assert lines[1].split()[-8:] == [
        "position",
        "length",
        "consist",
        "intrans",
        "gold",
        "spammer",
        "reversed",
        "any",
    ]
    row = next(line for line in lines if "60 prompts x 8 pairs" in line and "left_biased" in line)
    assert row.split()[-8:] == ["1/1", "0/1", "0/1", "0/1", "0/1", "0/1", "0/1", "1/1"]
    row = next(line for line in lines if "40 prompts x 4 pairs" in line and "adversarial" in line)
    assert row.split()[-4:] == ["1/1", "0/1", "1/1", "1/1"]
