"""The combined audit report and `prefpairs audit`."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from prefpairs.cli import app
from prefpairs.quality.report import AuditConfig, render_audit, run_audit
from prefpairs.simulate import (
    Archetype,
    SimulationConfig,
    load_truth,
    simulate,
    write_simulation,
)
from prefpairs.store import Store
from tests.conftest import build_records

runner = CliRunner()
SAMPLE = Path(__file__).resolve().parents[1] / "examples"
PLANTED = (
    "flagged: ann-06 (gold+reversed), ann-08 (intransitive+spammer), "
    "ann-10 (position), ann-11 (length)"
)


@pytest.fixture(scope="module")
def sample_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    db = tmp_path_factory.mktemp("audit") / "sample.db"
    result = runner.invoke(app, ["import", str(SAMPLE / "sample.jsonl"), "--db", str(db)])
    assert result.exit_code == 0, result.output
    return db


def test_bundled_sample_flags_exactly_the_planted_annotators(sample_db: Path) -> None:
    truth = SAMPLE / "sample-truth.json"
    result = runner.invoke(app, ["audit", "--db", str(sample_db), "--truth", str(truth)])
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0] == "Annotator audit of 672 pairwise judgments from 12 annotators"
    assert lines[-1] == PLANTED
    assert "ann-06 gold: gold accuracy 0/12, upper bound 0.24 < 0.7" in lines
    assert any(line.startswith("ann-08 spammer: spammer score") for line in lines)
    row = next(line for line in lines if line.startswith("ann-08"))
    assert row.endswith("intransitive+spammer  random_spammer")
    planted = json.loads(truth.read_text(encoding="utf-8"))["annotator_archetypes"]
    defective = {"left_biased", "length_biased", "random_spammer", "adversarial"}
    assert sorted(a for a, k in planted.items() if k in defective) == [
        "ann-06",
        "ann-08",
        "ann-10",
        "ann-11",
    ]


def test_strict_exits_1_when_anyone_is_flagged(sample_db: Path) -> None:
    result = runner.invoke(app, ["audit", "--db", str(sample_db), "--strict"])
    assert result.exit_code == 1
    assert PLANTED in result.output
    assert "planted" not in result.output


def test_strict_exits_0_on_honest_annotators(tmp_path: Path) -> None:
    db = str(tmp_path / "honest.db")
    simulated = runner.invoke(app, ["simulate", "--db", db, "--roster", "reliable=6,noisy=2"])
    assert simulated.exit_code == 0, simulated.output
    result = runner.invoke(app, ["audit", "--db", db, "--strict"])
    assert result.exit_code == 0, result.output
    assert result.output.rstrip().endswith("flagged: none")


def test_json_output_records_config_and_flags(sample_db: Path) -> None:
    args = ["audit", "--db", str(sample_db), "--json", "--min-gold-accuracy", "0.8", "--seed", "3"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["config"]["min_gold_accuracy"] == 0.8
    assert data["config"]["seed"] == 3
    assert data["transitivity"]["seed"] == 3
    assert "ann-06" in data["flagged"]
    by_id = {a["annotator_id"]: a for a in data["annotators"]}
    assert by_id["ann-06"]["flags"] == ["gold", "reversed"]
    assert len(by_id["ann-06"]["reasons"]) == 2
    assert by_id["ann-01"]["flagged"] is False
    assert by_id["ann-01"]["planted"] is None


def test_missing_database_is_an_error(tmp_path: Path) -> None:
    result = runner.invoke(app, ["audit", "--db", str(tmp_path / "absent.db")])
    assert result.exit_code == 1
    assert "prefpairs init" in result.output


def test_stored_truth_is_used(tmp_path: Path) -> None:
    dataset = simulate(SimulationConfig(seed=0))
    with Store.open(tmp_path / "sim.db") as store:
        write_simulation(store, dataset)
        report = run_audit(store, AuditConfig(random_draws=500), truth=load_truth(store))
    assert report.flagged == ("ann-06", "ann-08", "ann-10", "ann-11")
    assert {a.annotator_id: a.planted for a in report.annotators}["ann-10"] == "left_biased"
    assert report.config.random_draws == 500
    assert render_audit(report).splitlines()[-1] == PLANTED


def test_every_flag_kind_has_a_reason() -> None:
    config = SimulationConfig(
        seed=0, n_prompts=60, pairs_per_prompt=8, control_rate=0.5, n_models=10
    )
    # A left-clicker that barely follows quality reverses its flipped repeats.
    profiles = dict(config.profiles)
    left = profiles[Archetype.LEFT_BIASED]
    profiles[Archetype.LEFT_BIASED] = left.model_copy(
        update={"sharpness": 0.3, "position_bias": 3.0}
    )
    dataset = simulate(config.model_copy(update={"profiles": profiles}))
    with Store.open(":memory:") as store:
        write_simulation(store, dataset)
        report = run_audit(store, truth=load_truth(store))
    raised = {flag for a in report.annotators for flag in a.flags}
    assert raised == {
        "position",
        "length",
        "consistency",
        "intransitive",
        "gold",
        "spammer",
        "reversed",
    }
    for audit in report.annotators:
        assert len(audit.flags) == len(audit.reasons)
        assert audit.flagged == bool(audit.flags)
    assert {a.planted for a in report.annotators if a.flagged} <= {
        "left_biased",
        "length_biased",
        "random_spammer",
        "adversarial",
    }


def test_small_store_renders() -> None:
    with Store.open(":memory:") as store:
        store.add_all(build_records())
        report = run_audit(store)
    text = render_audit(report)
    assert text.splitlines()[-1] == "flagged: none"
    assert report.flagged == ()
    assert report.gold.n_unmatched == 0


def test_gold_judgments_without_gold_records_are_reported() -> None:
    records = [r for r in build_records() if type(r).__name__ != "GoldPair"]
    with Store.open(":memory:") as store:
        store.add_all(records)
        report = run_audit(store)
    assert report.gold.n_unmatched == 1
    assert "note: 1 gold judgments match no gold pair and were not scored" in render_audit(report)


def test_empty_store_renders_without_rows() -> None:
    with Store.open(":memory:") as store:
        text = render_audit(run_audit(store))
    assert "from 0 annotators" in text
    assert text.endswith("flagged: none")
