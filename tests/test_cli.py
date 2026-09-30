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
    code = main.main(
        [
            "--tune",
            "--seeds",
            "1",
            "--eval-seeds",
            "--particles",
            "2",
            "--iters",
            "2",
            "--grid-size",
            "10",
            "--agents",
            "2",
            "--survivors",
            "2",
            "--max-ticks",
            "60",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert out.count("[PSO] iter") == 3


def test_cli_rejects_infeasible_config(capsys: pytest.CaptureFixture[str]) -> None:
    """A combination that passes argument bounds but is infeasible exits with code 2."""
    assert main.main(["--grid-size", "6", "--agents", "20", "--survivors", "20"]) == 2
    assert "Invalid configuration" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["--agents", "0"],
        ["--agents", "abc"],
        ["--particles", "999"],
        ["--iters", "-1"],
        ["--wall-density", "nan"],
        ["--wall-density", "0.9"],
        ["--seeds", "-5"],
        ["--ping-range", "50"],
        ["--evaporation-rate", "1"],
        ["--round", "3"],
        ["--seeds", *[str(i) for i in range(60)]],
    ],
)
def test_cli_bounds_every_input(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    """Out-of-range, non-numeric or oversized inputs are rejected with exit code 2."""
    with pytest.raises(SystemExit) as exc:
        main.main(argv)
    assert exc.value.code == 2
    assert "error" in capsys.readouterr().err


def test_bounded_parsers_accept_edges() -> None:
    """Range edges are inclusive."""
    assert main.bounded_int(1, 5)("5") == 5
    assert main.bounded_float(0.0, 1.0)("0") == 0.0


def test_overfitting_guard() -> None:
    """Tuned weights are only recommended when held-out fitness does not drop."""
    assert main.generalizes(120.0, 121.0)
    assert main.generalizes(120.0, 120.0)
    assert not main.generalizes(120.258, 118.212)
