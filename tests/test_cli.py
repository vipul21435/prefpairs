"""CLI behaviour, driven through Typer's CliRunner."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from prefpairs import __version__
from prefpairs.cli import app
from prefpairs.jsonl import read_records
from prefpairs.simulate import SimulationTruth, load_truth
from prefpairs.store import MIGRATIONS, Store, dependency_order

runner = CliRunner()
EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "tiny.jsonl"
LATEST = MIGRATIONS[-1].version


def invoke(*args: str) -> tuple[int, str]:
    result = runner.invoke(app, list(args))
    return result.exit_code, result.output


def test_version_flag_prints_installed_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == f"prefpairs {__version__}"


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "Usage" in result.output


def test_info_lists_pipeline_stages() -> None:
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    lines = result.stdout.splitlines()
    assert lines[0] == f"prefpairs {__version__}"
    status = dict(line.split(maxsplit=2)[1:] for line in lines[1:])
    assert status["aggregate"] == "available: rank"
    assert status["audit"] == "available: checks, audit"
    assert status["export"] == "available: export"
    assert status["collect"] == "planned"


class TestInit:
    def test_creates_and_is_idempotent(self, tmp_path: Path) -> None:
        db = tmp_path / "sub" / "prefs.db"
        for _ in range(2):
            code, output = invoke("init", "--db", str(db))
            assert code == 0
            assert output.strip() == f"{db}: schema v{LATEST}"
        assert db.exists()

    def test_database_path_can_come_from_the_environment(self, tmp_path: Path) -> None:
        db = tmp_path / "env.db"
        result = runner.invoke(app, ["init"], env={"PREFPAIRS_DB": str(db)})
        assert result.exit_code == 0
        assert db.exists()

    def test_refuses_a_foreign_database(self, tmp_path: Path) -> None:
        db = tmp_path / "foreign.db"
        db.write_bytes(b"this is not sqlite at all, just some bytes" * 50)
        code, output = invoke("init", "--db", str(db))
        assert code == 1
        assert output.startswith("error:")


class TestImport:
    def test_imports_the_example_and_reimport_is_a_no_op(self, tmp_path: Path) -> None:
        db = tmp_path / "prefs.db"
        code, output = invoke("import", str(EXAMPLE), "--db", str(db))
        assert code == 0, output
        assert output.strip() == (
            "added prompt 2, response 5, annotator 2, session 2, gold 1, pairwise 5, ranked 1; "
            "0 unchanged"
        )
        code, output = invoke("import", str(EXAMPLE), "--db", str(db))
        assert code == 0
        assert output.strip() == "added nothing; 18 unchanged"
        with Store.open(db) as store:
            assert list(store.iter_records()) == dependency_order(read_records(EXAMPLE))

    def test_bad_line_is_reported_and_nothing_is_written(self, tmp_path: Path) -> None:
        source = tmp_path / "bad.jsonl"
        lines = EXAMPLE.read_text(encoding="utf-8").splitlines()
        lines[4] = lines[4].replace('"client":"import"', '"client":"fax"')
        source.write_text("\n".join(lines) + "\n", encoding="utf-8")
        db = tmp_path / "prefs.db"
        code, output = invoke("import", str(source), "--db", str(db))
        assert code == 1
        assert "error: line 5: invalid session: client" in output
        with Store.open(db) as store:
            assert sum(store.counts().values()) == 0

    def test_broken_reference_is_reported(self, tmp_path: Path) -> None:
        source = tmp_path / "orphan.jsonl"
        lines = EXAMPLE.read_text(encoding="utf-8").splitlines()
        source.write_text(lines[-1] + "\n", encoding="utf-8")
        code, output = invoke("import", str(source), "--db", str(tmp_path / "prefs.db"))
        assert code == 1
        assert "ranked 'k1' rejected: FOREIGN KEY constraint failed" in output

    def test_missing_file_is_a_usage_error(self, tmp_path: Path) -> None:
        code, output = invoke("import", str(tmp_path / "absent.jsonl"), "--db", str(tmp_path / "x"))
        assert code == 2
        assert "does not exist" in output


class TestSimulate:
    def test_simulate_then_stats(self, tmp_path: Path) -> None:
        db = tmp_path / "sim.db"
        truth_file = tmp_path / "out" / "truth.json"
        code, output = invoke(
            "simulate", "--seed", "0", "--db", str(db), "--truth-out", str(truth_file)
        )
        assert code == 0, output
        assert output.splitlines()[0] == f"wrote 1012 new records to {db} (seed 0)"
        assert output.splitlines()[1].startswith("true model order: model-")
        assert "planted defective annotators: ann-" in output

        with Store.open(db) as store:
            truth = load_truth(store)
        assert truth is not None
        assert SimulationTruth.model_validate_json(truth_file.read_text(encoding="utf-8")) == truth

        code, text = invoke("stats", "--db", str(db))
        assert code == 0
        assert f"schema v{LATEST}, simulated, seed 0" in text
        assert "pairwise" in text

        code, raw = invoke("stats", "--db", str(db), "--json")
        assert code == 0
        stats = json.loads(raw)
        assert stats["counts"] == {
            "prompt": 40,
            "response": 240,
            "annotator": 12,
            "session": 12,
            "gold": 12,
            "pairwise": 672,
            "ranked": 24,
        }
        assert stats["pair_kinds"] == {"regular": 480, "control": 48, "gold": 144}
        assert sum(stats["choices"].values()) == 672
        assert stats["simulated_seed"] == 0

    def test_same_seed_gives_identical_databases_and_dumps(self, tmp_path: Path) -> None:
        dumps = []
        rows = []
        for name in ("first.db", "second.db"):
            db = tmp_path / name
            code, _ = invoke("simulate", "--seed", "4", "--prompts", "12", "--db", str(db))
            assert code == 0
            out = tmp_path / f"{name}.jsonl"
            code, output = invoke("dump", "--db", str(db), "--out", str(out))
            assert code == 0
            assert output.startswith("wrote ")
            dumps.append(out.read_bytes())
            with Store.open(db) as store:
                rows.append(store.dump_rows())
        assert dumps[0] == dumps[1]
        assert rows[0] == rows[1]

    def test_second_simulation_into_the_same_database_is_refused(self, tmp_path: Path) -> None:
        db = tmp_path / "sim.db"
        assert invoke("simulate", "--seed", "1", "--prompts", "10", "--db", str(db))[0] == 0
        assert invoke("simulate", "--seed", "1", "--prompts", "10", "--db", str(db))[0] == 0
        code, output = invoke("simulate", "--seed", "2", "--prompts", "10", "--db", str(db))
        assert code == 1
        assert "different simulated dataset" in output

    def test_custom_roster(self, tmp_path: Path) -> None:
        db = tmp_path / "sim.db"
        code, output = invoke(
            "simulate",
            "--db",
            str(db),
            "--prompts",
            "10",
            "--redundancy",
            "2",
            "--roster",
            "reliable=2, left_biased=1",
        )
        assert code == 0, output
        assert output.count("(left_biased)") == 1
        with Store.open(db) as store:
            assert store.counts()["annotator"] == 3

    def test_roster_of_only_honest_annotators_reports_none_planted(self, tmp_path: Path) -> None:
        code, output = invoke(
            "simulate", "--db", str(tmp_path / "x.db"), "--roster", "reliable=3", "--gold", "0"
        )
        assert code == 0
        assert "planted defective annotators: none" in output

    @pytest.mark.parametrize("roster", ["reliable", "bogus=1", "reliable=-1", "reliable=x"])
    def test_malformed_roster_is_a_usage_error(self, tmp_path: Path, roster: str) -> None:
        code, output = invoke("simulate", "--db", str(tmp_path / "x.db"), "--roster", roster)
        assert code == 2
        assert "ARCHETYPE=COUNT" in output

    def test_inconsistent_config_is_reported(self, tmp_path: Path) -> None:
        db = tmp_path / "x.db"
        code, output = invoke("simulate", "--db", str(db), "--roster", "reliable=2")
        assert code == 1
        assert "redundancy=3 exceeds the 2 annotators" in output

    def test_infeasible_gold_is_reported(self, tmp_path: Path) -> None:
        code, output = invoke("simulate", "--db", str(tmp_path / "x.db"), "--prompts", "1")
        assert code == 1
        assert "lower n_gold" in output


class TestStatsAndDump:
    def test_missing_database_is_not_created(self, tmp_path: Path) -> None:
        db = tmp_path / "absent.db"
        for command in ("stats", "dump"):
            code, output = invoke(command, "--db", str(db))
            assert code == 1
            assert "prefpairs init" in output
        assert not db.exists()

    def test_stats_of_an_empty_database(self, tmp_path: Path) -> None:
        db = tmp_path / "empty.db"
        invoke("init", "--db", str(db))
        code, output = invoke("stats", "--db", str(db), "--json")
        assert code == 0
        stats = json.loads(output)
        assert stats["simulated_seed"] is None
        assert stats["pairwise_per_annotator_max"] == 0
        assert stats["rationale_share"] == 0.0

    def test_dump_to_stdout_reimports_to_the_same_database(self, tmp_path: Path) -> None:
        db = tmp_path / "prefs.db"
        invoke("import", str(EXAMPLE), "--db", str(db))
        code, output = invoke("dump", "--db", str(db))
        assert code == 0
        copy = tmp_path / "copy.jsonl"
        copy.write_text(output, encoding="utf-8")
        clone = tmp_path / "clone.db"
        assert invoke("import", str(copy), "--db", str(clone))[0] == 0
        with Store.open(db) as original, Store.open(clone) as rebuilt:
            assert original.dump_rows() == rebuilt.dump_rows()
