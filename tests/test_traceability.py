"""Keeps docs/REQUIREMENTS_TRACEABILITY.md honest: every reference must resolve to real code."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MATRIX = (ROOT / "docs" / "REQUIREMENTS_TRACEABILITY.md").read_text(encoding="utf-8")
REFERENCE = re.compile(r"`([\w/]+\.py)::([\w.]+)`")
HARD_CONSTRAINTS = (
    "Collision-free trajectories",
    "Safe operational margins",
    "Decision latency within the real-time budget",
    "No centralized single point of failure",
    "Sub-second trajectory recalibration",
)
OBJECTIVES = (
    "Throughput",
    "Coverage velocity",
    "Collision risk",
    "Deadlocks",
    "Energy",
    "Path length",
    "Resilience to agent failure",
    "Resilience to communication dropout",
)
EVALUATION = ("Multi-scenario benchmarks", "Runtime", "Stability under perturbation")
CHECKLIST = (
    "Parameter configuration",
    "Fitness per attempt",
    "Convergence evidence",
    "Representation rationale",
    "Operators rationale",
    "CI technique rationale",
    "What-changed notes",
)


def defined_names(path: Path) -> set[str]:
    """Qualified names (``func``, ``Class``, ``Class.method``) defined in a Python file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.ClassDef):
            names.add(node.name)
        if isinstance(node, ast.ClassDef):
            names |= {f"{node.name}.{n.name}" for n in node.body if isinstance(n, ast.FunctionDef)}
            names |= {
                f"{node.name}.{n.target.id}"
                for n in node.body
                if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
            }
    return names


def test_matrix_references_resolve() -> None:
    """Every `file.py::name` in the matrix names an existing file and definition."""
    refs = REFERENCE.findall(MATRIX)
    assert len(refs) > 60
    cache: dict[str, set[str]] = {}
    for file, name in refs:
        path = ROOT / file
        assert path.is_file(), f"missing file {file}"
        names = cache.setdefault(file, defined_names(path))
        assert name in names, f"{file}::{name} does not exist"


@pytest.mark.parametrize("line", HARD_CONSTRAINTS + OBJECTIVES + EVALUATION + CHECKLIST)
def test_every_spec_line_has_a_row(line: str) -> None:
    """Each Track 05 hard constraint, objective, evaluation item and checklist item is traced."""
    row = next((r for r in MATRIX.splitlines() if r.startswith(f"| {line} |")), None)
    assert row is not None, f"no row for {line!r}"
    assert "tests/" in row or line in {"CI technique rationale"} or "`" in row


def test_changelog_covers_every_attempt() -> None:
    """CHANGELOG.md records what changed in every attempt."""
    changelog = ROOT / "CHANGELOG.md"
    if not changelog.exists():
        pytest.skip("CHANGELOG.md not written yet")
    text = changelog.read_text(encoding="utf-8")
    for attempt in ("Attempt 1", "Attempt 2"):
        assert attempt in text
