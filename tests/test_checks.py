"""`prefpairs checks` and the report behind it."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from prefpairs.cli import app
from prefpairs.quality.checks import render_checks, run_checks
from prefpairs.simulate import SimulationConfig, load_truth, simulate, write_simulation
from prefpairs.store import Store

runner = CliRunner()
SAMPLE = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture(scope="module")
def sample_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    db = tmp_path_factory.mktemp("checks") / "sample.db"
    result = runner.invoke(app, ["import", str(SAMPLE / "sample.jsonl"), "--db", str(db)])
    assert result.exit_code == 0, result.output
    return db


def test_bundled_sample_flags_the_planted_position_and_length_annotators(sample_db: Path) -> None:
    truth = SAMPLE / "sample-truth.json"
    result = runner.invoke(app, ["checks", "--db", str(sample_db), "--truth", str(truth)])
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0] == "Annotator checks on 672 pairwise judgments from 12 annotators"
    assert "flagged: ann-10 (position), ann-11 (length)" in lines
    row = next(line for line in lines if line.startswith("ann-10"))
    assert row.endswith("position  left_biased")
    assert any(line.startswith("Fleiss kappa over 154 items with 3 labels") for line in lines)


def test_json_output_and_options(sample_db: Path) -> None:
    result = runner.invoke(
        app,
        ["checks", "--db", str(sample_db), "--json", "--alpha", "0.01", "--no-quality-adjustment"],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["alpha"] == 0.01
    assert data["archetypes"] is None
    assert data["length"]["adjust_for_quality"] is False
    assert data["position"]["pooled"]["annotator_id"] is None
    assert len(data["agreement"]["pairs"]) == 66
    assert {r["annotator_id"] for r in data["consistency"]["results"]} >= {"ann-01", "ann-12"}


def test_text_without_truth_has_no_planted_column(sample_db: Path) -> None:
    result = runner.invoke(app, ["checks", "--db", str(sample_db)])
    assert result.exit_code == 0, result.output
    assert "planted" not in result.output
    assert "flagged: ann-10 (position), ann-11 (length)" in result.output


def test_missing_database_is_an_error(tmp_path: Path) -> None:
    result = runner.invoke(app, ["checks", "--db", str(tmp_path / "absent.db")])
    assert result.exit_code == 1
    assert "prefpairs init" in result.output


def test_stored_truth_is_used_and_flags_are_grouped(tmp_path: Path) -> None:
    dataset = simulate(SimulationConfig(seed=0))
    with Store.open(tmp_path / "sim.db") as store:
        write_simulation(store, dataset)
        report = run_checks(store, truth=load_truth(store))
    assert report.archetypes is not None
    assert report.archetypes["ann-10"] == "left_biased"
    assert report.flags() == {"ann-10": ("position",), "ann-11": ("length",)}
    text = render_checks(report)
    assert "  planted" in text
    assert "adjusted for consensus quality" in text


def test_empty_store_renders_without_rows() -> None:
    with Store.open(":memory:") as store:
        report = run_checks(store)
    text = render_checks(report)
    assert report.n_judgments == 0
    assert report.flags() == {}
    assert text.splitlines()[-1] == "flagged: none"
    assert "Fleiss" not in text
