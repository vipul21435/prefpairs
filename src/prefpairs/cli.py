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
from prefpairs.aggregate import (
    BootstrapConfig,
    BradleyTerryConfig,
    ComparisonOptions,
    Level,
    NotIdentifiableError,
    ResampleUnit,
)
from prefpairs.export import ExportFormat, FilterConfig, SplitConfig, export_dataset
from prefpairs.jsonl import JsonlError, read_records, write_records
from prefpairs.quality.checks import render_checks, run_checks
from prefpairs.quality.report import AuditConfig, render_audit, run_audit
from prefpairs.ranking import Method, NothingToRankError, rank_store, render_rank_report
from prefpairs.simulate import (
    Archetype,
    SimulationConfig,
    SimulationError,
    SimulationTruth,
    load_truth,
    write_simulation,
)
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

STAGES = (
    ("store", "available: init, import, dump, stats"),
    ("simulate", "available: simulate"),
    ("aggregate", "available: rank"),
    ("audit", "available: checks, audit"),
    ("collect", "planned"),
    ("export", "available: export"),
)

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
    except (
        StoreError,
        JsonlError,
        SimulationError,
        ValidationError,
        NotIdentifiableError,
        NothingToRankError,
    ) as exc:
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
    """Print the toolkit version and which pipeline stages are available yet."""
    typer.echo(f"prefpairs {__version__}")
    for stage, status in STAGES:
        typer.echo(f"  - {stage:<10} {status}")


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


def _load_truth_file(path: Path | None) -> SimulationTruth | None:
    if path is None:
        return None
    return SimulationTruth.model_validate_json(path.read_text(encoding="utf-8"))


TruthOption = Annotated[
    Path | None,
    typer.Option(
        exists=True,
        dir_okay=False,
        help="Ground-truth JSON from `simulate --truth-out` (default: the one stored).",
    ),
]


@app.command()
def checks(
    *,
    db: DbOption = DEFAULT_DB,
    alpha: Annotated[
        float, typer.Option(min=0.0001, max=0.5, help="Family-wise error rate per check.")
    ] = 0.05,
    confidence: Annotated[
        float, typer.Option(min=0.5, max=0.999, help="Interval coverage level.")
    ] = 0.95,
    quality_adjustment: Annotated[
        bool,
        typer.Option(help="Control the length test for consensus model quality."),
    ] = True,
    truth: TruthOption = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON instead of text.")] = False,
) -> None:
    """Test every annotator for position and length bias, agreement and self-consistency."""
    with _reported_errors():
        known = _load_truth_file(truth)
        with Store.open(db, create=False) as store:
            if known is None:
                known = load_truth(store)
            report = run_checks(
                store,
                alpha=alpha,
                confidence=confidence,
                adjust_for_quality=quality_adjustment,
                truth=known,
            )
    if as_json:
        typer.echo(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
    else:
        typer.echo(render_checks(report))


@app.command()
def audit(
    *,
    db: DbOption = DEFAULT_DB,
    alpha: Annotated[
        float, typer.Option(min=0.0001, max=0.5, help="Family-wise error rate per test.")
    ] = 0.05,
    confidence: Annotated[
        float, typer.Option(min=0.5, max=0.999, help="Interval coverage level.")
    ] = 0.95,
    min_gold_accuracy: Annotated[
        float, typer.Option(min=0.0, max=1.0, help="Flag if the gold upper bound is below this.")
    ] = 0.7,
    min_spammer_score: Annotated[
        float, typer.Option(min=0.0, max=1.0, help="Flag as spammer below this score.")
    ] = 0.1,
    seed: Annotated[int, typer.Option(min=0, help="Seed of the random-orientation baseline.")] = 0,
    truth: TruthOption = None,
    strict: Annotated[
        bool, typer.Option("--strict", help="Exit with code 1 when any annotator is flagged.")
    ] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON instead of text.")] = False,
) -> None:
    """Combine every annotator check into per-annotator flags with reasons and evidence."""
    with _reported_errors():
        config = AuditConfig(
            alpha=alpha,
            confidence=confidence,
            min_gold_accuracy=min_gold_accuracy,
            min_spammer_score=min_spammer_score,
            seed=seed,
        )
        known = _load_truth_file(truth)
        with Store.open(db, create=False) as store:
            if known is None:
                known = load_truth(store)
            report = run_audit(store, config, truth=known)
    if as_json:
        data = report.model_dump(mode="json")
        data["flagged"] = list(report.flagged)
        typer.echo(json.dumps(data, indent=2, sort_keys=True))
    else:
        typer.echo(render_audit(report))
    if strict and report.flagged:
        raise typer.Exit(code=1)


@app.command()
def rank(
    *,
    db: DbOption = DEFAULT_DB,
    method: Annotated[
        Method, typer.Option(help="bt: Bradley-Terry (MM fit); elo: permutation-averaged Elo.")
    ] = Method.BRADLEY_TERRY,
    level: Annotated[Level, typer.Option(help="Rank models or individual responses.")] = (
        Level.MODEL
    ),
    replicates: Annotated[
        int, typer.Option(min=10, max=100_000, help="Bootstrap replicates.")
    ] = 500,
    resample: Annotated[
        ResampleUnit,
        typer.Option(help="Resample whole prompts (cluster bootstrap) or single judgments."),
    ] = ResampleUnit.PROMPT,
    confidence: Annotated[
        float, typer.Option(min=0.5, max=0.999, help="Interval coverage level.")
    ] = 0.95,
    seed: Annotated[int, typer.Option(min=0, help="Bootstrap seed.")] = 0,
    prior: Annotated[
        float, typer.Option(min=0.0, help="Bradley-Terry pseudo-count prior (0 = plain MLE).")
    ] = 0.1,
    exclude_annotator: Annotated[
        list[str] | None,
        typer.Option("--exclude-annotator", "-x", help="Leave out this annotator (repeatable)."),
    ] = None,
    include_ranked: Annotated[
        bool, typer.Option(help="Also expand ranked judgments into implied pairs.")
    ] = False,
    truth: TruthOption = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON instead of text.")] = False,
) -> None:
    """Rank models or responses with bootstrap confidence intervals."""
    with _reported_errors():
        options = ComparisonOptions(
            level=level,
            exclude_annotators=tuple(exclude_annotator or ()),
            include_ranked=include_ranked,
        )
        bootstrap = BootstrapConfig(
            n_replicates=replicates, unit=resample, level=confidence, seed=seed
        )
        known = _load_truth_file(truth)
        with Store.open(db, create=False) as store:
            if known is None:
                known = load_truth(store)
            report = rank_store(
                store,
                method=method,
                options=options,
                bootstrap=bootstrap,
                bt_config=BradleyTerryConfig(prior=prior),
                truth=known,
            )
    if as_json:
        typer.echo(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
    else:
        typer.echo(render_rank_report(report))


@app.command()
def export(
    *,
    db: DbOption = DEFAULT_DB,
    fmt: Annotated[
        ExportFormat,
        typer.Option(
            "--format", help="dpo: chosen/rejected; kto: labelled completions; rm: soft labels."
        ),
    ] = ExportFormat.DPO,
    out: Annotated[
        Path, typer.Option(help="Output directory; files go to <out>/<format>/.", file_okay=False)
    ] = Path("export"),
    min_votes: Annotated[int, typer.Option(min=1, help="Counted votes a pair needs.")] = 1,
    min_agreement: Annotated[
        float, typer.Option(min=0.0, max=1.0, help="Smallest share of votes the winner must hold.")
    ] = 0.5,
    keep_ties: Annotated[
        bool, typer.Option("--keep-ties", help="Keep pairs whose votes are split evenly.")
    ] = False,
    exclude_annotator: Annotated[
        list[str] | None,
        typer.Option("--exclude-annotator", "-x", help="Leave out this annotator (repeatable)."),
    ] = None,
    audit: Annotated[
        bool, typer.Option(help="Run the audit and leave out every annotator it flags.")
    ] = True,
    train: Annotated[
        float, typer.Option(min=0.0, max=1.0, help="Share of prompts in train.")
    ] = 0.8,
    validation: Annotated[
        float, typer.Option(min=0.0, max=1.0, help="Share of prompts in validation.")
    ] = 0.1,
    salt: Annotated[str, typer.Option(help="Salt of the prompt-level split hash.")] = "prefpairs",
) -> None:
    """Export training data (DPO, KTO or reward-model JSONL) with a dataset card."""
    with _reported_errors():
        split = SplitConfig(
            train=train, validation=validation, test=round(1.0 - train - validation, 12), salt=salt
        )
        with Store.open(db, create=False) as store:
            judgments = store.pairwise()
            excluded = set(exclude_annotator or ())
            unknown = sorted(excluded - {j.annotator_id for j in judgments})
            if unknown:
                typer.echo(
                    f"error: no judgments from annotator(s) given with -x: {', '.join(unknown)}",
                    err=True,
                )
                raise typer.Exit(code=1)
            flagged: tuple[str, ...] = ()
            if audit:
                flagged = run_audit(store, AuditConfig()).flagged
                excluded |= set(flagged)
            reason = (
                f"audit flags: {', '.join(flagged) or 'none'}" if audit else "given with -x"
            ) + (
                f"; given with -x: {', '.join(sorted(exclude_annotator))}"
                if audit and exclude_annotator
                else ""
            )
            filters = FilterConfig(
                min_votes=min_votes,
                min_agreement=min_agreement,
                drop_ties=not keep_ties,
                exclude_annotators=tuple(sorted(excluded)),
            )
            card = export_dataset(
                out,
                fmt,
                judgments=judgments,
                prompts=store.prompts(),
                responses=store.responses(),
                filters=filters,
                split=split,
                source=db.name,
                exclusion_reason=reason,
            )
    drops = ", ".join(f"{k} {v}" for k, v in card.drop_counts.items())
    typer.echo(
        f"kept {card.n_kept} of {card.n_pairs} pairs (dropped: {drops}); "
        f"excluded annotators: {', '.join(card.excluded_annotators) or 'none'}"
    )
    for f in card.files:
        typer.echo(f"{out / f.path}  {f.n_rows:>5} rows  sha256 {f.sha256[:16]}")
    typer.echo(f"card: {out / fmt.value / 'card.md'} and card.json")
