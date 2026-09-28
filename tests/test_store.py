"""SQLite store: migrations, integrity constraints, typed reads and imports."""

import sqlite3
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from prefpairs.schema import (
    Annotator,
    AnyRecord,
    Choice,
    GoldPair,
    PairKind,
    PairwiseJudgment,
    Prompt,
    RankedJudgment,
    Response,
    Session,
)
from prefpairs.store import (
    APPLICATION_ID,
    MIGRATIONS,
    IntegrityViolationError,
    Migration,
    MigrationError,
    RecordConflictError,
    SchemaVersionError,
    Store,
    StoreError,
    dependency_order,
    format_timestamp,
    parse_timestamp,
)
from tests.conftest import T0, build_records

LATEST = MIGRATIONS[-1].version


def pairwise(**overrides: object) -> PairwiseJudgment:
    fields: dict[str, object] = {
        "id": "j-new",
        "session_id": "s1",
        "annotator_id": "ann-1",
        "prompt_id": "p1",
        "left_response_id": "p1-a",
        "right_response_id": "p1-c",
        "choice": "left",
        "created_at": T0,
    }
    fields.update(overrides)
    return PairwiseJudgment.model_validate(fields)


class TestMigrations:
    def test_fresh_database_is_at_latest_version(self, store: Store) -> None:
        assert store.schema_version == LATEST
        assert store._pragma_int("application_id") == APPLICATION_ID
        assert set(store.table_names()) >= {
            "prompts",
            "responses",
            "annotators",
            "sessions",
            "gold_pairs",
            "pairwise_judgments",
            "ranked_judgments",
            "ranked_items",
        }

    def test_migrate_is_idempotent(self, store: Store) -> None:
        before = store.dump_rows()
        assert store.migrate() == []
        assert store.migrate() == []
        assert store.dump_rows() == before

    def test_reopening_a_file_keeps_data_and_applies_nothing(
        self, tmp_path: Path, records: list[AnyRecord]
    ) -> None:
        path = tmp_path / "nested" / "prefs.db"
        with Store.open(path) as db:
            db.add_all(records)
            expected = db.dump_rows()
        with Store.open(path, create=False) as db:
            assert db.migrate() == []
            assert db.dump_rows() == expected
            assert list(db.iter_records()) == records

    def test_upgrade_applies_only_pending_migrations(self, tmp_path: Path) -> None:
        path = tmp_path / "prefs.db"
        with Store.open(path):
            pass
        extra = Migration(LATEST + 1, "add notes", "CREATE TABLE notes (body TEXT) STRICT;")
        with Store.open(path, migrations=(*MIGRATIONS, extra)) as db:
            assert db.schema_version == LATEST + 1
            assert "notes" in db.table_names()
            assert db.migrate((*MIGRATIONS, extra)) == []

    def test_failed_migration_rolls_back_completely(self, tmp_path: Path) -> None:
        path = tmp_path / "prefs.db"
        with Store.open(path):
            pass
        broken = Migration(
            LATEST + 1,
            "broken",
            "CREATE TABLE half_done (x INTEGER) STRICT;\nINSERT INTO no_such_table VALUES (1);",
        )
        with pytest.raises(MigrationError, match="broken"):
            Store.open(path, migrations=(*MIGRATIONS, broken))
        with Store.open(path) as db:
            assert db.schema_version == LATEST
            assert "half_done" not in db.table_names()

    def test_newer_database_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "prefs.db"
        with Store.open(path):
            pass
        conn = sqlite3.connect(path)
        conn.execute(f"PRAGMA user_version = {LATEST + 5}")
        conn.close()
        with pytest.raises(SchemaVersionError, match="upgrade prefpairs"):
            Store.open(path)

    def test_foreign_sqlite_file_is_not_touched(self, tmp_path: Path) -> None:
        path = tmp_path / "other.db"
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE unrelated (x INTEGER)")
        conn.commit()
        conn.close()
        with pytest.raises(StoreError, match="refusing to initialise"):
            Store.open(path)
        conn = sqlite3.connect(path)
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 0
        conn.close()

    def test_other_application_id_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "other.db"
        conn = sqlite3.connect(path)
        conn.execute("PRAGMA application_id = 42")
        conn.close()
        with pytest.raises(StoreError, match="not a PrefPairs database"):
            Store.open(path)

    def test_missing_file_without_create(self, tmp_path: Path) -> None:
        with pytest.raises(StoreError, match="prefpairs init"):
            Store.open(tmp_path / "absent.db", create=False)
        assert not (tmp_path / "absent.db").exists()

    @pytest.mark.parametrize("versions", [[2], [1, 3], [1, 1]])
    def test_migration_versions_must_be_contiguous(self, versions: list[int]) -> None:
        migrations = [Migration(v, "m", "SELECT 1;") for v in versions]
        with pytest.raises(MigrationError, match="without gaps"):
            Store.open(":memory:", migrations=migrations)


class TestReadWrite:
    def test_every_record_round_trips(self, filled: Store, records: list[AnyRecord]) -> None:
        assert list(filled.iter_records()) == records

    def test_get_by_id(self, filled: Store, records: list[AnyRecord]) -> None:
        for record in records:
            assert filled.get(type(record), record.id) == record
        assert filled.get(Prompt, "missing") is None
        assert filled.get(RankedJudgment, "missing") is None

    def test_get_rejects_non_record_types(self, filled: Store) -> None:
        with pytest.raises(TypeError, match="not a PrefPairs record type"):
            filled.get(str, "p1")  # type: ignore[type-var]

    def test_filters(self, filled: Store) -> None:
        assert [r.id for r in filled.responses(prompt_id="p2")] == ["p2-a", "p2-b"]
        assert [s.id for s in filled.sessions(annotator_id="ann-2")] == ["s2"]
        assert [j.id for j in filled.pairwise(annotator_id="ann-1")] == ["j1", "j2", "j3"]
        assert [j.id for j in filled.pairwise(prompt_id="p2")] == ["j2", "j5"]
        assert [j.id for j in filled.pairwise(annotator_id="ann-2", prompt_id="p1")] == ["j4"]
        assert [k.id for k in filled.ranked(annotator_id="ann-2")] == ["k1"]
        assert filled.ranked(annotator_id="ann-1") == []
        assert [p.id for p in filled.prompts()] == ["p1", "p2"]
        assert [a.id for a in filled.annotators()] == ["ann-1", "ann-2"]
        assert [g.id for g in filled.gold_pairs()] == ["g1"]

    def test_ranked_keeps_both_orders(self, filled: Store) -> None:
        (ranked,) = filled.ranked()
        assert ranked.shown == ("p1-a", "p1-b", "p1-c")
        assert ranked.ranking == ("p1-b", "p1-c", "p1-a")

    def test_counts(self, filled: Store) -> None:
        assert filled.counts() == {
            "prompt": 2,
            "response": 5,
            "annotator": 2,
            "session": 2,
            "gold": 1,
            "pairwise": 5,
            "ranked": 1,
        }

    def test_duplicate_id_is_rejected(self, filled: Store) -> None:
        with pytest.raises(IntegrityViolationError, match="prompt 'p1'"):
            filled.add(Prompt(id="p1", text="again"))

    def test_add_all_is_atomic(self, store: Store, records: list[AnyRecord]) -> None:
        bad = pairwise(session_id="no-such-session")
        with pytest.raises(IntegrityViolationError):
            store.add_all([*records, bad])
        assert sum(store.counts().values()) == 0

    def test_nested_transaction_failure_keeps_outer_work(self, store: Store) -> None:
        with store.transaction():
            store.add(Prompt(id="kept", text="outer"))
            with pytest.raises(IntegrityViolationError):
                store.add_all([Prompt(id="inner", text="t"), Prompt(id="inner", text="t")])
        assert [p.id for p in store.prompts()] == ["kept"]


class TestIntegrity:
    """Constraints enforced by SQLite itself, beyond the pydantic validators."""

    @pytest.mark.parametrize(
        ("overrides", "why"),
        [
            ({"right_response_id": "p2-a"}, "response from another prompt"),
            ({"prompt_id": "p9"}, "unknown prompt"),
            ({"session_id": "s2"}, "session of another annotator"),
            ({"annotator_id": "ghost"}, "unknown annotator"),
        ],
    )
    def test_broken_references_are_rejected(
        self, filled: Store, overrides: dict[str, object], why: str
    ) -> None:
        with pytest.raises(IntegrityViolationError, match="FOREIGN KEY"):
            filled.add(pairwise(**overrides))
        assert filled.get(PairwiseJudgment, "j-new") is None, why

    @pytest.mark.parametrize(
        "overrides",
        [
            {"repeat_of": "j4"},  # another annotator's judgment
            {"repeat_of": "j2"},  # another pair
            {"repeat_of": "missing"},
        ],
    )
    def test_control_must_repeat_the_same_pair_by_the_same_annotator(
        self, filled: Store, overrides: dict[str, object]
    ) -> None:
        control = pairwise(pair_kind=PairKind.CONTROL, right_response_id="p1-b", **overrides)
        with pytest.raises(IntegrityViolationError, match="repeat_of must name"):
            filled.add(control)

    def test_valid_control_is_accepted(self, filled: Store) -> None:
        control = pairwise(
            pair_kind=PairKind.CONTROL,
            left_response_id="p1-a",
            right_response_id="p1-b",
            repeat_of="j3",
        )
        filled.add(control)
        assert filled.get(PairwiseJudgment, "j-new") == control

    def test_judgments_are_append_only(self, filled: Store) -> None:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            filled._conn.execute("UPDATE pairwise_judgments SET choice = 'left' WHERE id = 'j1'")
        assert filled.get(PairwiseJudgment, "j1") is not None
        judgment = filled.get(PairwiseJudgment, "j1")
        assert judgment is not None
        assert judgment.choice is Choice.RIGHT

    def test_ranked_items_must_belong_to_the_prompt(self, filled: Store) -> None:
        ranked = RankedJudgment(
            id="k2",
            session_id="s2",
            annotator_id="ann-2",
            prompt_id="p1",
            shown=("p1-a", "p2-a"),
            ranking=("p2-a", "p1-a"),
            created_at=T0,
        )
        with pytest.raises(IntegrityViolationError, match="ranked 'k2'"):
            filled.add(ranked)
        assert filled.get(RankedJudgment, "k2") is None

    def test_gold_pair_is_unique_regardless_of_orientation(self, filled: Store) -> None:
        flipped = GoldPair(
            id="g2", prompt_id="p2", better_response_id="p2-b", worse_response_id="p2-a"
        )
        with pytest.raises(IntegrityViolationError, match="UNIQUE"):
            filled.add(flipped)

    def test_response_must_reference_a_prompt(self, store: Store, records: list[AnyRecord]) -> None:
        response = next(r for r in records if isinstance(r, Response))
        with pytest.raises(IntegrityViolationError, match="FOREIGN KEY"):
            store.add(response)


class TestImport:
    def test_import_orders_records_by_dependency(self, store: Store) -> None:
        records = build_records()
        result = store.import_records(reversed(records))
        assert result.total_added == len(records)
        assert result.unchanged == 0
        assert result.added["pairwise"] == 5
        assert store.counts()["pairwise"] == 5

    def test_reimport_is_a_no_op(self, filled: Store, records: list[AnyRecord]) -> None:
        before = filled.dump_rows()
        result = filled.import_records(records)
        assert result.total_added == 0
        assert result.unchanged == len(records)
        assert filled.dump_rows() == before

    def test_conflicting_content_aborts_the_whole_import(self, filled: Store) -> None:
        before = filled.dump_rows()
        batch = [
            Prompt(id="p3", text="new prompt"),
            Annotator(id="ann-1", display_name="Renamed"),
        ]
        with pytest.raises(RecordConflictError, match="annotator 'ann-1'"):
            filled.import_records(batch)
        assert filled.dump_rows() == before

    def test_dump_and_reimport_rebuilds_an_identical_database(self, filled: Store) -> None:
        with Store.open(":memory:") as copy:
            copy.import_records(filled.iter_records())
            assert copy.dump_rows() == filled.dump_rows()


class TestTimestamps:
    def test_format_is_fixed_width_utc(self) -> None:
        ist = timezone(timedelta(hours=5, minutes=30))
        assert format_timestamp(datetime(2026, 1, 1, 5, 30, tzinfo=ist)) == (
            "2026-01-01T00:00:00.000000Z"
        )
        assert parse_timestamp(format_timestamp(T0)).tzinfo == UTC
        assert len(format_timestamp(T0 + timedelta(microseconds=7))) == 27

    def test_string_order_matches_time_order(self) -> None:
        stamps = [
            T0 + timedelta(microseconds=m) for m in (0, 1, 999_999, 1_000_000, 86_400_000_000)
        ]
        encoded = [format_timestamp(s) for s in stamps]
        assert encoded == sorted(encoded)
        assert [parse_timestamp(e) for e in encoded] == stamps

    def test_session_order_check_is_enforced_in_sql(self, filled: Store) -> None:
        with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
            filled._conn.execute(
                "INSERT INTO sessions (id, annotator_id, client, started_at, ended_at) "
                "VALUES ('s9', 'ann-1', 'web', ?, ?)",
                (format_timestamp(T0), format_timestamp(T0 - timedelta(seconds=1))),
            )
        assert filled.get(Session, "s9") is None


class TestDependencyOrder:
    def test_controls_follow_their_originals_even_in_chains(self) -> None:
        first = pairwise(id="a", right_response_id="p1-b")
        second = pairwise(
            id="b", right_response_id="p1-b", pair_kind=PairKind.CONTROL, repeat_of="a"
        )
        third = pairwise(
            id="c", right_response_id="p1-b", pair_kind=PairKind.CONTROL, repeat_of="b"
        )
        prompt = Prompt(id="p1", text="t")
        ordered = dependency_order([third, second, prompt, first])
        assert [r.id for r in ordered] == ["p1", "a", "b", "c"]

    def test_reference_to_an_existing_judgment_keeps_input_order(self) -> None:
        control = pairwise(id="b", pair_kind=PairKind.CONTROL, repeat_of="stored-earlier")
        regular = pairwise(id="a")
        assert [r.id for r in dependency_order([control, regular])] == ["b", "a"]

    def test_cycles_are_left_for_the_database_to_reject(self) -> None:
        x = pairwise(id="x", pair_kind=PairKind.CONTROL, repeat_of="y")
        y = pairwise(id="y", pair_kind=PairKind.CONTROL, repeat_of="x")
        assert [r.id for r in dependency_order([x, y])] == ["x", "y"]

    def test_chained_controls_import_in_any_order(self, filled: Store) -> None:
        first = pairwise(id="a", right_response_id="p1-b")
        second = pairwise(
            id="b", right_response_id="p1-b", pair_kind=PairKind.CONTROL, repeat_of="a"
        )
        result = filled.import_records([second, first])
        assert result.added == {"pairwise": 2}
