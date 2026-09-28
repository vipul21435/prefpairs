"""Smoke tests for the CLI entry point."""

from typer.testing import CliRunner

from prefpairs import __version__
from prefpairs.cli import app

runner = CliRunner()


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
    for stage in ("collect", "audit", "aggregate", "export"):
        assert f"- {stage}" in result.stdout
