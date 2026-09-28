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
    env = {**os.environ, "PREFPAIRS": str(cli), "DEMO_DB": str(tmp_path / "demo.db")}
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
    assert result.stdout.rstrip().endswith("demo OK: both rankings recover the true model order")
