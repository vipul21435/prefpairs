"""Command-line entry point for PrefPairs."""

from typing import Annotated

import typer

from prefpairs import __version__

app = typer.Typer(
    name="prefpairs",
    help="Collect, audit, aggregate and export pairwise preference data.",
    no_args_is_help=True,
    add_completion=False,
)


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


@app.command()
def info() -> None:
    """Print the toolkit version and the pipeline stages it provides."""
    typer.echo(f"prefpairs {__version__}")
    for stage in ("collect", "audit", "aggregate", "export"):
        typer.echo(f"  - {stage}")
