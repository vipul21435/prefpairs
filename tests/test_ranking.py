"""Ranking reports and the `prefpairs rank` command."""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from prefpairs.aggregate import BootstrapConfig, ComparisonOptions, Level, ResampleUnit
from prefpairs.cli import app
from prefpairs.ranking import (
    Method,
    NothingToRankError,
    RankReport,
    kendall_tau,
    rank_store,
    render_rank_report,
)
from prefpairs.simulate import SimulationConfig, load_truth, simulate, write_simulation
from prefpairs.store import Store

runner = CliRunner()
FAST = BootstrapConfig(n_replicates=100)
DEFECTIVE = ("ann-06", "ann-08", "ann-10", "ann-11")


@pytest.fixture(scope="module")
def simulated_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The default seed-0 simulation, written once for the whole module."""
    db = tmp_path_factory.mktemp("rank") / "sim.db"
    with Store.open(db) as store:
        write_simulation(store, simulate(SimulationConfig(seed=0)))
    return db


@pytest.fixture
def store(simulated_db: Path) -> Iterator[Store]:
    with Store.open(simulated_db, create=False) as opened:
        yield opened


class TestKendallTau:
    def test_identical_and_reversed_orders(self) -> None:
        truth = {"a": 3.0, "b": 2.0, "c": 1.0}
        assert kendall_tau(truth, truth) == 1.0
        assert kendall_tau({"a": 1.0, "b": 2.0, "c": 3.0}, truth) == -1.0

    def test_one_swap_out_of_three_pairs(self) -> None:
        estimate = {"a": 3.0, "b": 1.0, "c": 2.0}
        assert kendall_tau(estimate, {"a": 3.0, "b": 2.0, "c": 1.0}) == pytest.approx(1 / 3)

    def test_ties_count_as_neither(self) -> None:
        assert kendall_tau({"a": 1.0, "b": 1.0}, {"a": 2.0, "b": 1.0}) == 0.0

    def test_uses_only_shared_keys(self) -> None:
        assert kendall_tau({"a": 2.0, "b": 1.0, "z": 9.0}, {"a": 5.0, "b": 0.0}) == 1.0
        assert kendall_tau({"a": 1.0}, {"a": 1.0, "b": 2.0}) is None


class TestRankStore:
    def test_bradley_terry_recovers_the_true_model_order(self, store: Store) -> None:
        truth = load_truth(store)
        report = rank_store(store, bootstrap=FAST, truth=truth)
        assert report.method is Method.BRADLEY_TERRY
        assert report.kendall_tau == 1.0
        assert report.true_order is not None
        assert [row.item for row in report.rows] == list(report.true_order)
        assert [row.true_rank for row in report.rows] == [1, 2, 3, 4, 5, 6]
        assert [row.rank for row in report.rows] == [1, 2, 3, 4, 5, 6]
        for row in report.rows:
            assert row.lower < row.estimate < row.upper
            assert row.rank_lower <= row.rank <= row.rank_upper
        # Regular pairs only, skips dropped; games are counted from both sides.
        assert report.dropped == {"pair_kind": 192, "skip": 7}
        assert report.n_games == 480 - 7
        assert sum(row.games for row in report.rows) == 2 * report.n_games
        assert sum(row.score for row in report.rows) == pytest.approx(report.n_games)
        assert report.fit_gap is not None
        assert 0 <= report.fit_gap < 0.15
        assert report.unconverged_replicates == 0

    def test_is_deterministic_for_a_seed(self, store: Store) -> None:
        first = rank_store(store, bootstrap=FAST)
        assert first == rank_store(store, bootstrap=FAST)
        other = rank_store(store, bootstrap=FAST.model_copy(update={"seed": 1}))
        assert other.rows != first.rows
        assert [r.estimate for r in other.rows] == [r.estimate for r in first.rows]

    def test_elo_with_the_planted_annotators_left_out(self, store: Store) -> None:
        options = ComparisonOptions(exclude_annotators=DEFECTIVE)
        report = rank_store(
            store, method=Method.ELO, options=options, bootstrap=FAST, truth=load_truth(store)
        )
        assert report.fit_gap is None
        assert report.excluded_annotators == DEFECTIVE
        assert report.dropped["excluded_annotator"] == 160
        assert report.kendall_tau == 1.0
        assert report.rows[0].estimate > 1000.0 > report.rows[-1].estimate

    def test_response_level_scores_against_response_quality(self, tmp_path: Path) -> None:
        # Responses are only compared within their prompt, so a small dataset
        # keeps the dense per-replicate fit quick.
        with Store.open(tmp_path / "small.db") as small:
            write_simulation(small, simulate(SimulationConfig(seed=0, n_prompts=10, n_gold=4)))
            options = ComparisonOptions(level=Level.RESPONSE)
            config = BootstrapConfig(n_replicates=20)
            report = rank_store(small, options=options, bootstrap=config, truth=load_truth(small))
        assert report.level is Level.RESPONSE
        assert report.kendall_tau is not None
        assert report.kendall_tau > 0.3
        assert all(row.true_rank is not None for row in report.rows)

    def test_without_truth_there_is_no_comparison(self, store: Store) -> None:
        options = ComparisonOptions(include_ranked=True)
        unit = FAST.model_copy(update={"unit": ResampleUnit.JUDGMENT})
        report = rank_store(store, options=options, bootstrap=unit)
        assert report.true_order is None
        assert report.kendall_tau is None
        assert report.unit == "judgment"
        assert all(row.true_rank is None for row in report.rows)
        implied = sum(len(r.ranking) * (len(r.ranking) - 1) // 2 for r in store.ranked())
        assert implied > 0
        assert report.n_games == 480 - 7 + implied

    def test_nothing_left_to_rank(self, store: Store, tmp_path: Path) -> None:
        everyone = tuple(f"ann-{i:02d}" for i in range(1, 13))
        with pytest.raises(NothingToRankError, match="excluded_annotator 480, pair_kind 192"):
            rank_store(store, options=ComparisonOptions(exclude_annotators=everyone))
        with (
            Store.open(tmp_path / "empty.db") as empty,
            pytest.raises(NothingToRankError, match=r"dropped: none"),
        ):
            rank_store(empty)


class TestRender:
    def test_text_report(self, store: Store) -> None:
        report = rank_store(store, bootstrap=FAST, truth=load_truth(store))
        text = render_rank_report(report)
        lines = text.splitlines()
        assert lines[0] == "Bradley-Terry ranking of 6 models from 473 games on 40 prompts"
        assert lines[1] == "95% intervals from 100 prompt-bootstrap replicates (seed 0)"
        assert lines[2] == "not counted: pair_kind 192, skip 7"
        assert "true rank" in lines[4]
        assert lines[5].split()[:2] == ["1", report.rows[0].item]
        assert "fit gap" in text
        assert text.endswith("Kendall tau against the true order: 1.000")

    def test_optional_lines(self, store: Store) -> None:
        options = ComparisonOptions(exclude_annotators=("ann-01",))
        base = rank_store(store, method=Method.ELO, options=options, bootstrap=FAST)
        text = render_rank_report(base)
        assert "excluded annotators: ann-01" in text
        assert "Elo rating" in text
        assert "true rank" not in text
        assert "fit gap" not in text
        rows = tuple(row.model_copy(update={"true_rank": None}) for row in base.rows)
        many = tuple(f"m{i:02d}" for i in range(13))
        odd: RankReport = base.model_copy(
            update={
                "unconverged_replicates": 3,
                "true_order": many,
                "kendall_tau": None,
                "rows": rows,
                "dropped": {},
            }
        )
        text = render_rank_report(odd)
        assert "warning: 3 replicates hit the iteration cap" in text
        assert "Kendall tau against the true order: n/a" in text
        assert not any(line.startswith("true order:") for line in text.splitlines())
        assert "not counted" not in text
        assert text.splitlines()[5].rstrip().endswith("-")


class TestRankCommand:
    def test_text_and_json(self, simulated_db: Path) -> None:
        result = runner.invoke(app, ["rank", "--db", str(simulated_db), "--replicates", "50"])
        assert result.exit_code == 0, result.output
        assert "Kendall tau against the true order: 1.000" in result.output
        result = runner.invoke(
            app,
            [
                "rank",
                "--db",
                str(simulated_db),
                "--replicates",
                "50",
                "--method",
                "elo",
                "-x",
                "ann-06",
                "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        report = json.loads(result.output)
        assert report["method"] == "elo"
        assert report["excluded_annotators"] == ["ann-06"]
        assert len(report["rows"]) == 6

    def test_truth_file_scores_an_imported_copy(self, simulated_db: Path, tmp_path: Path) -> None:
        with Store.open(simulated_db, create=False) as store:
            truth = load_truth(store)
        assert truth is not None
        truth_file = tmp_path / "truth.json"
        truth_file.write_text(truth.model_dump_json(), encoding="utf-8")
        dump = tmp_path / "dump.jsonl"
        assert (
            runner.invoke(app, ["dump", "--db", str(simulated_db), "-o", str(dump)]).exit_code == 0
        )
        copy = tmp_path / "copy.db"
        assert runner.invoke(app, ["import", str(dump), "--db", str(copy)]).exit_code == 0
        args = ["rank", "--db", str(copy), "--replicates", "50"]
        without = runner.invoke(app, args)
        assert without.exit_code == 0
        assert "Kendall tau" not in without.output
        scored = runner.invoke(app, [*args, "--truth", str(truth_file)])
        assert scored.exit_code == 0
        assert "Kendall tau against the true order: 1.000" in scored.output

    def test_errors_exit_with_code_one(self, tmp_path: Path) -> None:
        db = tmp_path / "empty.db"
        missing = runner.invoke(app, ["rank", "--db", str(db)])
        assert missing.exit_code == 1
        assert "prefpairs init" in missing.output
        assert runner.invoke(app, ["init", "--db", str(db)]).exit_code == 0
        empty = runner.invoke(app, ["rank", "--db", str(db)])
        assert empty.exit_code == 1
        assert "error: no judgments to rank" in empty.output

    def test_plain_mle_on_a_disconnected_graph_is_reported(self, tmp_path: Path) -> None:
        lines = [
            '{"type":"prompt","id":"p1","text":"Q","responses":['
            '{"id":"r1","model":"m1","text":"A","provenance":{"source":"model"}},'
            '{"id":"r2","model":"m2","text":"B","provenance":{"source":"model"}}]}',
            '{"type":"annotator","id":"a1"}',
            '{"type":"session","id":"s1","annotator_id":"a1","client":"import",'
            '"started_at":"2026-01-01T00:00:00Z"}',
            '{"type":"pairwise","id":"j1","session_id":"s1","annotator_id":"a1",'
            '"prompt_id":"p1","left_response_id":"r1","right_response_id":"r2",'
            '"choice":"left","created_at":"2026-01-01T00:00:01Z"}',
        ]
        source = tmp_path / "one.jsonl"
        source.write_text("\n".join(lines) + "\n", encoding="utf-8")
        db = tmp_path / "one.db"
        imported = runner.invoke(app, ["import", str(source), "--db", str(db)])
        assert imported.exit_code == 0, imported.output
        result = runner.invoke(app, ["rank", "--db", str(db), "--prior", "0"])
        assert result.exit_code == 1
        assert "not strongly connected" in result.output
        ok = runner.invoke(app, ["rank", "--db", str(db), "--replicates", "10"])
        assert ok.exit_code == 0, ok.output

    def test_bad_option_values_are_rejected(self, simulated_db: Path) -> None:
        result = runner.invoke(app, ["rank", "--db", str(simulated_db), "--replicates", "5"])
        assert result.exit_code == 2
