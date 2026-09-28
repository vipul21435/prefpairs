"""Command-line entry point for PrefPairs."""

import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from prefpairs import __version__
from prefpairs.jsonl import JsonlError, read_records, write_records
from prefpairs.simulate import Archetype, SimulationConfig, SimulationError, write_simulation
from prefpairs.simulate import simulate as run_simulation
from prefpairs.store import Store, StoreError
from prefpairs.summary import render_summary, summarise

app = typer.Typer(
    name="prefpairs",
    help="Collect, audit, aggregate and export pairwise preference data.",
    no_args_is_help=True,
    add_completion=False,
)

DEFAULT_DB = Path(".prefpairs/prefpairs.db")

DbOption = Annotated[
    Path,
    typer.Option(
        "--db",
        envvar="PREFPAIRS_DB",
        help="SQLite database file.",
        dir_okay=False,
        show_default=True,
    ),
]


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"prefpairs {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Print the installed version and exit.",
        ),
    ] = False,
) -> None:
    """PrefPairs command-line interface."""


@contextmanager
def _reported_errors() -> Iterator[None]:
    """Turn expected failures into a one-line message and exit code 1."""
    try:
        yield
    except (StoreError, JsonlError, SimulationError, ValidationError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _parse_roster(spec: str) -> dict[Archetype, int]:
    """Parse ``reliable=6,noisy=2`` into a roster."""
    roster: dict[Archetype, int] = {}
    for part in filter(None, (p.strip() for p in spec.split(","))):
        name, _, count = part.partition("=")
        archetype = next((a for a in Archetype if a.value == name.strip()), None)
        if archetype is None or not count.strip().isdigit():
            choices = ", ".join(a.value for a in Archetype)
            msg = f"expected ARCHETYPE=COUNT with ARCHETYPE in {choices}, got {part!r}"
            raise typer.BadParameter(msg, param_hint="--roster")
        roster[archetype] = int(count)
    return roster


@app.command()
def info() -> None:
    """Print the toolkit version and the pipeline stages it provides."""
    typer.echo(f"prefpairs {__version__}")
    for stage in ("collect", "audit", "aggregate", "export"):
        typer.echo(f"  - {stage}")


@app.command()
def init(db: DbOption = DEFAULT_DB) -> None:
    """Create a database, or upgrade an existing one to the current schema."""
    with _reported_errors(), Store.open(db) as store:
        typer.echo(f"{db}: schema v{store.schema_version}")


@app.command("import")
def import_jsonl(
    path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="JSONL file to import."),
    ],
    db: DbOption = DEFAULT_DB,
) -> None:
    """Import prompts, responses, annotators and judgments from a JSONL file.

    The import is atomic and idempotent: records that already exist unchanged
    are skipped, and a record that conflicts with stored data aborts it.
    """
    with _reported_errors():
        records = read_records(path)
        with Store.open(db) as store:
            result = store.import_records(records)
    added = ", ".join(f"{kind} {count}" for kind, count in result.added.items()) or "nothing"
    typer.echo(f"added {added}; {result.unchanged} unchanged")


@app.command()
def simulate(
    *,
    db: DbOption = DEFAULT_DB,
    seed: Annotated[int, typer.Option(min=0, help="Random seed.")] = 0,
    prompts: Annotated[int, typer.Option(min=1, help="Number of prompts.")] = 40,
    models: Annotated[int, typer.Option(min=2, max=26, help="Number of models.")] = 6,
    pairs_per_prompt: Annotated[int, typer.Option(min=1, help="Pairs judged per prompt.")] = 4,
    redundancy: Annotated[int, typer.Option(min=1, help="Annotators per pair.")] = 3,
    gold: Annotated[int, typer.Option(min=0, help="Gold pairs shown to everyone.")] = 12,
    control_rate: Annotated[
        float, typer.Option(min=0.0, max=1.0, help="Share of pairs repeated as controls.")
    ] = 0.1,
    roster: Annotated[
        str | None,
        typer.Option(help="Annotators per archetype, e.g. 'reliable=6,left_biased=1'."),
    ] = None,
    truth_out: Annotated[
        Path | None,
        typer.Option(dir_okay=False, help="Also write the ground truth to this JSON file."),
    ] = None,
) -> None:
    """Generate a synthetic dataset with known ground truth into a database."""
    with _reported_errors():
        fields: dict[str, object] = {
            "seed": seed,
            "n_prompts": prompts,
            "n_models": models,
            "pairs_per_prompt": pairs_per_prompt,
            "redundancy": redundancy,
            "n_gold": gold,
            "control_rate": control_rate,
        }
        if roster is not None:
            fields["roster"] = _parse_roster(roster)
        config = SimulationConfig.model_validate(fields)
        dataset = run_simulation(config)
        with Store.open(db) as store:
            added = write_simulation(store, dataset)
    truth = dataset.truth
    if truth_out is not None:
        truth_out.parent.mkdir(parents=True, exist_ok=True)
        truth_out.write_text(truth.model_dump_json(indent=2) + "\n", encoding="utf-8")
    typer.echo(f"wrote {added} new records to {db} (seed {seed})")
    typer.echo(f"true model order: {' > '.join(truth.model_ranking())}")
    planted = [f"{a} ({truth.annotator_archetypes[a].value})" for a in truth.defective_annotators]
    typer.echo(f"planted defective annotators: {', '.join(planted) or 'none'}")


@app.command()
def stats(
    db: DbOption = DEFAULT_DB,
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON instead of text.")] = False,
) -> None:
    """Summarise what a database contains."""
    with _reported_errors(), Store.open(db, create=False) as store:
        summary = summarise(store)
    if as_json:
        typer.echo(json.dumps(summary.model_dump(mode="json"), indent=2, sort_keys=True))
    else:
        typer.echo(render_summary(summary))


@app.command()
def dump(
    db: DbOption = DEFAULT_DB,
    out: Annotated[
        Path | None,
        typer.Option("--out", "-o", dir_okay=False, help="Output file (default: stdout)."),
    ] = None,
) -> None:
    """Write every record as canonical JSONL (re-importable with `prefpairs import`)."""
    with _reported_errors(), Store.open(db, create=False) as store:
        records = list(store.iter_records())
    if out is None:
        write_records(records, sys.stdout)
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        count = write_records(records, handle)
    typer.echo(f"wrote {count} records to {out}")
