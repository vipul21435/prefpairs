"""SQLite persistence for PrefPairs records.

The store is a thin typed repository over the standard-library ``sqlite3``
module. Design points:

* **Versioned schema.** ``PRAGMA user_version`` holds the schema version and
  each ``Migration`` runs in its own transaction together with the version bump,
  so a failed migration leaves the database exactly as it was.
* **Integrity in the database, not only in Python.** Composite foreign keys make
  it impossible to store a judgment whose responses belong to another prompt or
  whose session belongs to another annotator; CHECK constraints mirror the
  pydantic validators; triggers keep judgments append-only and make a control
  pair reference an earlier judgment of the same annotator on the same pair.
* **Explicit transactions.** The connection runs in autocommit mode and every
  write goes through ``transaction()``, which nests with savepoints.
"""

import json
import sqlite3
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

from prefpairs.schema import (
    RECORD_KINDS,
    Annotator,
    AnyRecord,
    GoldPair,
    PairwiseJudgment,
    Prompt,
    Provenance,
    RankedJudgment,
    Response,
    Session,
    record_kind,
)

APPLICATION_ID = 0x50524650
"""``PRAGMA application_id`` of a PrefPairs database (ASCII "PRFP")."""

MEMORY = ":memory:"


class StoreError(Exception):
    """Base class for store failures that callers are expected to report."""


class SchemaVersionError(StoreError):
    """The database was written by a newer PrefPairs than the running one."""


class MigrationError(StoreError):
    """A migration failed and was rolled back."""


class RecordConflictError(StoreError):
    """A record id already exists with different content."""


class IntegrityViolationError(StoreError):
    """The database rejected a record (missing parent, mismatched prompt, ...)."""


@dataclass(frozen=True, slots=True)
class Migration:
    version: int
    description: str
    sql: str


_V1_CORE = """
PRAGMA application_id = 1347569232; -- 0x50524650, APPLICATION_ID

CREATE TABLE prompts (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL CHECK (length(trim(text)) > 0),
    category TEXT,
    tags_json TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(tags_json))
) STRICT;

CREATE TABLE responses (
    id TEXT PRIMARY KEY,
    prompt_id TEXT NOT NULL REFERENCES prompts (id),
    model TEXT NOT NULL,
    text TEXT NOT NULL CHECK (length(trim(text)) > 0),
    provenance_json TEXT NOT NULL CHECK (json_valid(provenance_json)),
    UNIQUE (prompt_id, id)
) STRICT;
CREATE INDEX idx_responses_model ON responses (model);

CREATE TABLE annotators (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL CHECK (kind IN ('human', 'simulated')),
    display_name TEXT
) STRICT;

CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    annotator_id TEXT NOT NULL REFERENCES annotators (id),
    client TEXT NOT NULL CHECK (client IN ('cli', 'web', 'simulator', 'import')),
    started_at TEXT NOT NULL,
    ended_at TEXT CHECK (ended_at IS NULL OR ended_at >= started_at),
    UNIQUE (id, annotator_id)
) STRICT;
CREATE INDEX idx_sessions_annotator ON sessions (annotator_id);

CREATE TABLE gold_pairs (
    id TEXT PRIMARY KEY,
    prompt_id TEXT NOT NULL,
    better_response_id TEXT NOT NULL,
    worse_response_id TEXT NOT NULL,
    reason TEXT,
    FOREIGN KEY (prompt_id, better_response_id) REFERENCES responses (prompt_id, id),
    FOREIGN KEY (prompt_id, worse_response_id) REFERENCES responses (prompt_id, id),
    CHECK (better_response_id <> worse_response_id)
) STRICT;
CREATE UNIQUE INDEX idx_gold_pairs_pair ON gold_pairs (
    min(better_response_id, worse_response_id), max(better_response_id, worse_response_id)
);

CREATE TABLE pairwise_judgments (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    annotator_id TEXT NOT NULL,
    prompt_id TEXT NOT NULL,
    left_response_id TEXT NOT NULL,
    right_response_id TEXT NOT NULL,
    choice TEXT NOT NULL CHECK (choice IN ('left', 'right', 'tie', 'skip')),
    pair_kind TEXT NOT NULL CHECK (pair_kind IN ('regular', 'control', 'gold')),
    repeat_of TEXT REFERENCES pairwise_judgments (id),
    rationale TEXT,
    latency_ms INTEGER CHECK (latency_ms IS NULL OR latency_ms >= 0),
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id, annotator_id) REFERENCES sessions (id, annotator_id),
    FOREIGN KEY (prompt_id, left_response_id) REFERENCES responses (prompt_id, id),
    FOREIGN KEY (prompt_id, right_response_id) REFERENCES responses (prompt_id, id),
    CHECK (left_response_id <> right_response_id),
    CHECK ((pair_kind = 'control') = (repeat_of IS NOT NULL))
) STRICT;
CREATE INDEX idx_pairwise_annotator ON pairwise_judgments (annotator_id);
CREATE INDEX idx_pairwise_prompt ON pairwise_judgments (prompt_id);
CREATE INDEX idx_pairwise_session ON pairwise_judgments (session_id, annotator_id);
CREATE INDEX idx_pairwise_repeat_of ON pairwise_judgments (repeat_of);

CREATE TRIGGER trg_pairwise_control_matches_original
BEFORE INSERT ON pairwise_judgments
WHEN NEW.repeat_of IS NOT NULL
BEGIN
    SELECT RAISE(ABORT, 'repeat_of must name an earlier judgment of this annotator on this pair')
    WHERE NOT EXISTS (
        SELECT 1 FROM pairwise_judgments AS o
        WHERE o.id = NEW.repeat_of
          AND o.annotator_id = NEW.annotator_id
          AND o.prompt_id = NEW.prompt_id
          AND min(o.left_response_id, o.right_response_id)
              = min(NEW.left_response_id, NEW.right_response_id)
          AND max(o.left_response_id, o.right_response_id)
              = max(NEW.left_response_id, NEW.right_response_id)
    );
END;

CREATE TRIGGER trg_pairwise_append_only
BEFORE UPDATE ON pairwise_judgments
BEGIN
    SELECT RAISE(ABORT, 'judgments are append-only');
END;

CREATE TABLE ranked_judgments (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    annotator_id TEXT NOT NULL,
    prompt_id TEXT NOT NULL REFERENCES prompts (id),
    rationale TEXT,
    latency_ms INTEGER CHECK (latency_ms IS NULL OR latency_ms >= 0),
    created_at TEXT NOT NULL,
    UNIQUE (id, prompt_id),
    FOREIGN KEY (session_id, annotator_id) REFERENCES sessions (id, annotator_id)
) STRICT;
CREATE INDEX idx_ranked_annotator ON ranked_judgments (annotator_id);
CREATE INDEX idx_ranked_session ON ranked_judgments (session_id, annotator_id);

CREATE TABLE ranked_items (
    judgment_id TEXT NOT NULL,
    prompt_id TEXT NOT NULL,
    response_id TEXT NOT NULL,
    rank INTEGER NOT NULL CHECK (rank >= 1),
    shown_position INTEGER NOT NULL CHECK (shown_position >= 1),
    PRIMARY KEY (judgment_id, response_id),
    UNIQUE (judgment_id, rank),
    UNIQUE (judgment_id, shown_position),
    FOREIGN KEY (judgment_id, prompt_id) REFERENCES ranked_judgments (id, prompt_id),
    FOREIGN KEY (prompt_id, response_id) REFERENCES responses (prompt_id, id)
) STRICT, WITHOUT ROWID;
CREATE INDEX idx_ranked_items_response ON ranked_items (prompt_id, response_id);
"""

_V2_SIMULATION_TRUTH = """
CREATE TABLE simulation_truth (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    seed INTEGER NOT NULL,
    truth_json TEXT NOT NULL CHECK (json_valid(truth_json))
) STRICT;
"""

MIGRATIONS: tuple[Migration, ...] = (
    Migration(1, "core preference-data tables", _V1_CORE),
    Migration(2, "ground truth of a simulated dataset", _V2_SIMULATION_TRUTH),
)
"""Every schema migration, in version order. Append only; never edit a released one."""


def format_timestamp(value: datetime) -> str:
    """Fixed-width UTC timestamp, so string order equals time order in SQL."""
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _optional_timestamp(value: datetime | None) -> str | None:
    return None if value is None else format_timestamp(value)


def _parse_optional_timestamp(value: str | None) -> datetime | None:
    return None if value is None else parse_timestamp(value)


def _check_migrations(migrations: Sequence[Migration]) -> None:
    versions = [m.version for m in migrations]
    if versions != list(range(1, len(versions) + 1)):
        msg = f"migration versions must be 1..n without gaps, got {versions}"
        raise MigrationError(msg)


def dependency_order(records: Iterable[AnyRecord]) -> list[AnyRecord]:
    """Order records so every reference points backwards.

    Kinds follow ``RECORD_KINDS``; within a kind the input order is kept, except
    that a control judgment is moved after the judgment it repeats when both are
    in the batch. A reference cycle is left as is for the database to reject.
    """
    rank = {kind: index for index, kind in enumerate(RECORD_KINDS)}
    ordered = sorted(records, key=lambda r: rank[record_kind(r)])
    judgments = [r for r in ordered if isinstance(r, PairwiseJudgment)]
    in_batch = {j.id for j in judgments}
    placed: set[str] = set()
    sorted_judgments: list[AnyRecord] = []
    pending = judgments
    while pending:
        waiting = []
        for judgment in pending:
            original = judgment.repeat_of
            if original is None or original not in in_batch or original in placed:
                sorted_judgments.append(judgment)
                placed.add(judgment.id)
            else:
                waiting.append(judgment)
        if len(waiting) == len(pending):
            sorted_judgments.extend(waiting)
            break
        pending = waiting
    before = [r for r in ordered if rank[record_kind(r)] < rank["pairwise"]]
    after = [r for r in ordered if rank[record_kind(r)] > rank["pairwise"]]
    return [*before, *sorted_judgments, *after]


@dataclass(frozen=True, slots=True)
class ImportResult:
    """What an import did: new records per kind, and identical records skipped."""

    added: dict[str, int] = field(default_factory=dict)
    unchanged: int = 0

    @property
    def total_added(self) -> int:
        return sum(self.added.values())


class Store:
    """Typed repository over one SQLite database."""

    def __init__(self, conn: sqlite3.Connection, path: str) -> None:
        self._conn = conn
        self._savepoints = 0
        self.path = path

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        create: bool = True,
        migrations: Sequence[Migration] = MIGRATIONS,
    ) -> Self:
        """Open (and by default create) a database and bring its schema up to date."""
        target = str(path)
        if target != MEMORY:
            file = Path(target)
            if not create and not file.exists():
                msg = f"no database at {file}; run `prefpairs init --db {file}` first"
                raise StoreError(msg)
            file.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(target, isolation_level=None)
        conn.row_factory = sqlite3.Row
        store = cls(conn, target)
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            store.migrate(migrations)
        except sqlite3.DatabaseError as exc:
            conn.close()
            msg = f"cannot open {target}: {exc}"
            raise StoreError(msg) from exc
        except BaseException:
            conn.close()
            raise
        return store

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- schema ---------------------------------------------------------------

    def _pragma_int(self, name: str) -> int:
        row = self._conn.execute(f"PRAGMA {name}").fetchone()
        return int(row[0])

    @property
    def schema_version(self) -> int:
        return self._pragma_int("user_version")

    def migrate(self, migrations: Sequence[Migration] = MIGRATIONS) -> list[int]:
        """Apply pending migrations; return the versions applied (empty if none)."""
        _check_migrations(migrations)
        current = self.schema_version
        latest = migrations[-1].version if migrations else 0
        if current > latest:
            msg = (
                f"database schema is v{current} but this prefpairs only knows v{latest}; "
                "upgrade prefpairs to open it"
            )
            raise SchemaVersionError(msg)
        application_id = self._pragma_int("application_id")
        if application_id not in (0, APPLICATION_ID):
            msg = f"{self.path} is not a PrefPairs database (application_id={application_id})"
            raise StoreError(msg)
        if current == 0 and self._has_schema_objects():
            msg = f"refusing to initialise {self.path}: it already contains tables"
            raise StoreError(msg)
        applied: list[int] = []
        for migration in migrations:
            if migration.version <= current:
                continue
            script = (
                f"BEGIN IMMEDIATE;\n{migration.sql}\n"
                f"PRAGMA user_version = {migration.version};\nCOMMIT;"
            )
            try:
                self._conn.executescript(script)
            except sqlite3.Error as exc:
                if self._conn.in_transaction:
                    self._conn.execute("ROLLBACK")
                msg = f"migration {migration.version} ({migration.description}) failed: {exc}"
                raise MigrationError(msg) from exc
            applied.append(migration.version)
        return applied

    def _has_schema_objects(self) -> bool:
        row = self._conn.execute(
            "SELECT count(*) FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%'"
        ).fetchone()
        return int(row[0]) > 0

    def table_names(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT name FROM sqlite_schema WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        ).fetchall()
        return [str(row[0]) for row in rows]

    # -- transactions -----------------------------------------------------------

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Run a block atomically. Nested blocks become savepoints."""
        name = f"sp_{self._savepoints}"
        self._savepoints += 1
        self._conn.execute(f"SAVEPOINT {name}")
        try:
            yield
        except BaseException:
            self._conn.execute(f"ROLLBACK TO {name}")
            self._conn.execute(f"RELEASE {name}")
            raise
        else:
            self._conn.execute(f"RELEASE {name}")
        finally:
            self._savepoints -= 1

    # -- writes -----------------------------------------------------------------

    def add(self, record: AnyRecord) -> None:
        """Insert one record; raise on a duplicate id or a broken reference."""
        with self.transaction():
            self._insert(record)

    def add_all(self, records: Iterable[AnyRecord]) -> int:
        """Insert records in one transaction, in the given order. All or nothing."""
        count = 0
        with self.transaction():
            for record in records:
                self._insert(record)
                count += 1
        return count

    def import_records(self, records: Iterable[AnyRecord]) -> ImportResult:
        """Idempotent, atomic import.

        Records are inserted in dependency order (prompts before responses before
        judgments; file order is kept within a kind). A record whose id already
        exists with identical content is skipped, so re-importing a file is a
        no-op; an id that exists with different content aborts the whole import.
        """
        ordered = dependency_order(records)
        added: dict[str, int] = defaultdict(int)
        unchanged = 0
        with self.transaction():
            for record in ordered:
                kind = record_kind(record)
                existing = self.get(type(record), record.id)
                if existing is None:
                    self._insert(record)
                    added[kind] += 1
                elif existing == record:
                    unchanged += 1
                else:
                    msg = f"{kind} {record.id!r} already exists with different content"
                    raise RecordConflictError(msg)
        return ImportResult(added=dict(added), unchanged=unchanged)

    def _insert(self, record: AnyRecord) -> None:
        try:
            self._insert_unchecked(record)
        except sqlite3.IntegrityError as exc:
            kind = record_kind(record)
            msg = f"{kind} {record.id!r} rejected: {exc}"
            raise IntegrityViolationError(msg) from exc

    def _insert_unchecked(self, record: AnyRecord) -> None:
        execute = self._conn.execute
        match record:
            case Prompt():
                execute(
                    "INSERT INTO prompts (id, text, category, tags_json) VALUES (?, ?, ?, ?)",
                    (record.id, record.text, record.category, json.dumps(list(record.tags))),
                )
            case Response():
                execute(
                    "INSERT INTO responses (id, prompt_id, model, text, provenance_json) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        record.id,
                        record.prompt_id,
                        record.model,
                        record.text,
                        record.provenance.model_dump_json(exclude_none=True),
                    ),
                )
            case Annotator():
                execute(
                    "INSERT INTO annotators (id, kind, display_name) VALUES (?, ?, ?)",
                    (record.id, record.kind.value, record.display_name),
                )
            case Session():
                execute(
                    "INSERT INTO sessions (id, annotator_id, client, started_at, ended_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        record.id,
                        record.annotator_id,
                        record.client.value,
                        format_timestamp(record.started_at),
                        _optional_timestamp(record.ended_at),
                    ),
                )
            case GoldPair():
                execute(
                    "INSERT INTO gold_pairs "
                    "(id, prompt_id, better_response_id, worse_response_id, reason) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        record.id,
                        record.prompt_id,
                        record.better_response_id,
                        record.worse_response_id,
                        record.reason,
                    ),
                )
            case PairwiseJudgment():
                execute(
                    "INSERT INTO pairwise_judgments (id, session_id, annotator_id, prompt_id, "
                    "left_response_id, right_response_id, choice, pair_kind, repeat_of, "
                    "rationale, latency_ms, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        record.id,
                        record.session_id,
                        record.annotator_id,
                        record.prompt_id,
                        record.left_response_id,
                        record.right_response_id,
                        record.choice.value,
                        record.pair_kind.value,
                        record.repeat_of,
                        record.rationale,
                        record.latency_ms,
                        format_timestamp(record.created_at),
                    ),
                )
            case RankedJudgment():
                self._insert_ranked(record)

    def _insert_ranked(self, record: RankedJudgment) -> None:
        self._conn.execute(
            "INSERT INTO ranked_judgments (id, session_id, annotator_id, prompt_id, rationale, "
            "latency_ms, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                record.id,
                record.session_id,
                record.annotator_id,
                record.prompt_id,
                record.rationale,
                record.latency_ms,
                format_timestamp(record.created_at),
            ),
        )
        shown_position = {response_id: i for i, response_id in enumerate(record.shown, start=1)}
        self._conn.executemany(
            "INSERT INTO ranked_items (judgment_id, prompt_id, response_id, rank, shown_position) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                (record.id, record.prompt_id, response_id, rank, shown_position[response_id])
                for rank, response_id in enumerate(record.ranking, start=1)
            ],
        )

    # -- reads ------------------------------------------------------------------

    def get[R: AnyRecord](self, cls: type[R], record_id: str) -> R | None:
        """Fetch one record by id, or None."""
        found: AnyRecord | None
        if cls is Prompt:
            found = next(iter(self._prompts(record_id)), None)
        elif cls is Response:
            found = next(iter(self._responses(record_id=record_id)), None)
        elif cls is Annotator:
            found = next(iter(self._annotators(record_id)), None)
        elif cls is Session:
            found = next(iter(self._sessions(record_id=record_id)), None)
        elif cls is GoldPair:
            found = next(iter(self._gold_pairs(record_id)), None)
        elif cls is PairwiseJudgment:
            found = next(iter(self._pairwise(record_id=record_id)), None)
        elif cls is RankedJudgment:
            found = next(iter(self._ranked(record_id=record_id)), None)
        else:
            msg = f"not a PrefPairs record type: {cls.__name__}"
            raise TypeError(msg)
        if found is None or isinstance(found, cls):
            return found
        msg = f"store returned {type(found).__name__} for {cls.__name__}"  # pragma: no cover
        raise TypeError(msg)  # pragma: no cover

    def prompts(self) -> list[Prompt]:
        return self._prompts(None)

    def responses(self, *, prompt_id: str | None = None) -> list[Response]:
        return self._responses(prompt_id=prompt_id)

    def annotators(self) -> list[Annotator]:
        return self._annotators(None)

    def sessions(self, *, annotator_id: str | None = None) -> list[Session]:
        return self._sessions(annotator_id=annotator_id)

    def gold_pairs(self) -> list[GoldPair]:
        return self._gold_pairs(None)

    def pairwise(
        self, *, annotator_id: str | None = None, prompt_id: str | None = None
    ) -> list[PairwiseJudgment]:
        """Pairwise judgments in insertion order, optionally filtered."""
        return self._pairwise(annotator_id=annotator_id, prompt_id=prompt_id)

    def ranked(self, *, annotator_id: str | None = None) -> list[RankedJudgment]:
        return self._ranked(annotator_id=annotator_id)

    def iter_records(self) -> Iterator[AnyRecord]:
        """Every record, in dependency order; re-importing them rebuilds the store."""
        yield from self.prompts()
        yield from self.responses()
        yield from self.annotators()
        yield from self.sessions()
        yield from self.gold_pairs()
        yield from self.pairwise()
        yield from self.ranked()

    def save_simulation_truth(self, seed: int, truth_json: str) -> bool:
        """Store the ground truth of a simulated dataset (at most one per database).

        Returns True when stored, False when the identical truth is already
        present; a different truth raises RecordConflictError.
        """
        with self.transaction():
            existing = self.simulation_truth_json()
            if existing is None:
                self._conn.execute(
                    "INSERT INTO simulation_truth (id, seed, truth_json) VALUES (1, ?, ?)",
                    (seed, truth_json),
                )
                return True
        if existing == truth_json:
            return False
        msg = "the database already holds a different simulated dataset; use a new database"
        raise RecordConflictError(msg)

    def simulation_truth_json(self) -> str | None:
        """The stored ground-truth JSON, or None if the data was not simulated."""
        row = self._conn.execute("SELECT truth_json FROM simulation_truth WHERE id = 1").fetchone()
        return None if row is None else str(row[0])

    def counts(self) -> dict[str, int]:
        """Number of records of each kind."""
        row = self._conn.execute(
            "SELECT (SELECT count(*) FROM prompts), (SELECT count(*) FROM responses), "
            "(SELECT count(*) FROM annotators), (SELECT count(*) FROM sessions), "
            "(SELECT count(*) FROM gold_pairs), (SELECT count(*) FROM pairwise_judgments), "
            "(SELECT count(*) FROM ranked_judgments)"
        ).fetchone()
        return {kind: int(value) for kind, value in zip(RECORD_KINDS, row, strict=True)}

    def dump_rows(self) -> dict[str, list[tuple[Any, ...]]]:
        """Every row of every table, fully ordered; equal dumps mean equal databases."""
        dump: dict[str, list[tuple[Any, ...]]] = {}
        for table in self.table_names():
            # Table names come from sqlite_schema, never from user input.
            query = f'SELECT * FROM "{table}"'  # noqa: S608
            width = len(self._conn.execute(f"{query} LIMIT 0").description)
            order = ", ".join(str(i) for i in range(1, width + 1))
            rows = self._conn.execute(f"{query} ORDER BY {order}").fetchall()
            dump[table] = [tuple(row) for row in rows]
        return dump

    # -- row mapping ------------------------------------------------------------

    def _prompts(self, record_id: str | None) -> list[Prompt]:
        rows = self._conn.execute(
            "SELECT id, text, category, tags_json FROM prompts "
            "WHERE (:id IS NULL OR id = :id) ORDER BY rowid",
            {"id": record_id},
        ).fetchall()
        return [
            Prompt(
                id=row["id"],
                text=row["text"],
                category=row["category"],
                tags=tuple(json.loads(row["tags_json"])),
            )
            for row in rows
        ]

    def _responses(
        self, *, record_id: str | None = None, prompt_id: str | None = None
    ) -> list[Response]:
        rows = self._conn.execute(
            "SELECT id, prompt_id, model, text, provenance_json FROM responses "
            "WHERE (:id IS NULL OR id = :id) AND (:prompt IS NULL OR prompt_id = :prompt) "
            "ORDER BY rowid",
            {"id": record_id, "prompt": prompt_id},
        ).fetchall()
        return [
            Response(
                id=row["id"],
                prompt_id=row["prompt_id"],
                model=row["model"],
                text=row["text"],
                provenance=Provenance.model_validate_json(row["provenance_json"]),
            )
            for row in rows
        ]

    def _annotators(self, record_id: str | None) -> list[Annotator]:
        rows = self._conn.execute(
            "SELECT id, kind, display_name FROM annotators "
            "WHERE (:id IS NULL OR id = :id) ORDER BY rowid",
            {"id": record_id},
        ).fetchall()
        return [
            Annotator.model_validate(
                {"id": row["id"], "kind": row["kind"], "display_name": row["display_name"]}
            )
            for row in rows
        ]

    def _sessions(
        self, *, record_id: str | None = None, annotator_id: str | None = None
    ) -> list[Session]:
        rows = self._conn.execute(
            "SELECT id, annotator_id, client, started_at, ended_at FROM sessions "
            "WHERE (:id IS NULL OR id = :id) AND (:ann IS NULL OR annotator_id = :ann) "
            "ORDER BY rowid",
            {"id": record_id, "ann": annotator_id},
        ).fetchall()
        return [
            Session.model_validate(
                {
                    "id": row["id"],
                    "annotator_id": row["annotator_id"],
                    "client": row["client"],
                    "started_at": parse_timestamp(row["started_at"]),
                    "ended_at": _parse_optional_timestamp(row["ended_at"]),
                }
            )
            for row in rows
        ]

    def _gold_pairs(self, record_id: str | None) -> list[GoldPair]:
        rows = self._conn.execute(
            "SELECT id, prompt_id, better_response_id, worse_response_id, reason FROM gold_pairs "
            "WHERE (:id IS NULL OR id = :id) ORDER BY rowid",
            {"id": record_id},
        ).fetchall()
        return [GoldPair.model_validate(dict(row)) for row in rows]

    def _pairwise(
        self,
        *,
        record_id: str | None = None,
        annotator_id: str | None = None,
        prompt_id: str | None = None,
    ) -> list[PairwiseJudgment]:
        rows = self._conn.execute(
            "SELECT id, session_id, annotator_id, prompt_id, left_response_id, "
            "right_response_id, choice, pair_kind, repeat_of, rationale, latency_ms, created_at "
            "FROM pairwise_judgments "
            "WHERE (:id IS NULL OR id = :id) AND (:ann IS NULL OR annotator_id = :ann) "
            "AND (:prompt IS NULL OR prompt_id = :prompt) ORDER BY rowid",
            {"id": record_id, "ann": annotator_id, "prompt": prompt_id},
        ).fetchall()
        judgments = []
        for row in rows:
            data = dict(row)
            data["created_at"] = parse_timestamp(row["created_at"])
            judgments.append(PairwiseJudgment.model_validate(data))
        return judgments

    def _ranked(
        self, *, record_id: str | None = None, annotator_id: str | None = None
    ) -> list[RankedJudgment]:
        params = {"id": record_id, "ann": annotator_id}
        rows = self._conn.execute(
            "SELECT id, session_id, annotator_id, prompt_id, rationale, latency_ms, created_at "
            "FROM ranked_judgments "
            "WHERE (:id IS NULL OR id = :id) AND (:ann IS NULL OR annotator_id = :ann) "
            "ORDER BY rowid",
            params,
        ).fetchall()
        items = self._conn.execute(
            "SELECT i.judgment_id, i.response_id, i.rank, i.shown_position "
            "FROM ranked_items AS i JOIN ranked_judgments AS j ON j.id = i.judgment_id "
            "WHERE (:id IS NULL OR j.id = :id) AND (:ann IS NULL OR j.annotator_id = :ann)",
            params,
        ).fetchall()
        by_judgment: dict[str, list[sqlite3.Row]] = defaultdict(list)
        for item in items:
            by_judgment[item["judgment_id"]].append(item)
        judgments = []
        for row in rows:
            members = by_judgment[row["id"]]
            ranking = [m["response_id"] for m in sorted(members, key=lambda m: m["rank"])]
            shown = [m["response_id"] for m in sorted(members, key=lambda m: m["shown_position"])]
            data = dict(row)
            data.update(
                shown=tuple(shown),
                ranking=tuple(ranking),
                created_at=parse_timestamp(row["created_at"]),
            )
            judgments.append(RankedJudgment.model_validate(data))
        return judgments
