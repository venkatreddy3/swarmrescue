"""Smoke tests for the command-line interface."""

from __future__ import annotations

import pytest

import main


def test_cli_round2_runs(capsys: pytest.CaptureFixture[str]) -> None:
    """A small Round 2 run prints the report and advisor section."""
    code = main.main(["--round", "2", "--seeds", "1", "--grid-size", "12", "--agents", "3", "--survivors", "3"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Round 2" in out and "Mission Advisor" in out


def test_cli_tune_prints_convergence_log(capsys: pytest.CaptureFixture[str]) -> None:
    """--tune prints one PSO log line per iteration."""
    code = main.main([
        "--tune", "--seeds", "1", "--eval-seeds", "--particles", "2", "--iters", "2",
        "--grid-size", "10", "--agents", "2", "--survivors", "2", "--max-ticks", "60",
    ])
    out = capsys.readouterr().out
    assert code == 0
    assert out.count("[PSO] iter") == 3


def test_cli_rejects_invalid_config(capsys: pytest.CaptureFixture[str]) -> None:
    """Invalid parameters exit with code 2 and an error message."""
    assert main.main(["--agents", "0"]) == 2
    assert "Invalid configuration" in capsys.readouterr().err
