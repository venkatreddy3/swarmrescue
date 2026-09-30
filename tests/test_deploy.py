"""Deployment regression tests for the Streamlit Cloud crash.

Streamlit Cloud keeps one Python process alive across ``git pull`` updates, so
an already-imported ``swarmrescue`` could be stale (Attempt 2 crashes:
``TypeError: unexpected keyword argument`` and ``ImportError: cannot import
name REASON_COMPLETE``). These tests pin down both defences.
"""

from __future__ import annotations

import importlib
import os
import re
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

import app

ROOT = Path(__file__).resolve().parents[1]
INIT = "import time\nVERSION = {version}\n__imported_at__ = time.time()\n"


def requirement_lines(path: Path) -> list[str]:
    """Non-comment, non-empty lines of a requirements file."""
    lines = [ln.split("#", 1)[0].strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    return [ln for ln in lines if ln]


@pytest.mark.parametrize("name", ["requirements.txt", "requirements-dev.txt"])
def test_requirements_never_install_the_project_itself(name: str) -> None:
    """Installing the repo as a package would shadow the live source with a stale copy."""
    path = ROOT / name
    if not path.exists():
        pytest.skip(f"{name} not present")
    for line in requirement_lines(path):
        assert line not in {".", "./"}, line
        assert not line.startswith(("-e", "--editable", "file:", "git+", "./", "../", "swarmrescue")), line


@pytest.mark.parametrize("name", ["requirements.txt", "requirements-dev.txt"])
def test_every_dependency_is_pinned_exactly(name: str) -> None:
    """Every requirement uses an exact ``==`` pin (reproducible, auditable builds)."""
    path = ROOT / name
    if not path.exists():
        pytest.skip(f"{name} not present")
    for line in requirement_lines(path):
        if line.startswith("-r "):
            continue
        assert re.fullmatch(r"[A-Za-z0-9_.\-\[\]]+==[0-9][A-Za-z0-9.\-]*", line), line


@pytest.fixture
def fake_package(tmp_path: Path) -> Iterator[tuple[Path, str]]:
    """A throw-away package ``fakeswarm`` at version 1, cleaned out of sys.modules afterwards."""
    pkg = tmp_path / "fakeswarm"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(INIT.format(version=1), encoding="utf-8")
    (pkg / "simulation.py").write_text("OLD = True\n", encoding="utf-8")
    saved_path = list(sys.path)
    yield tmp_path, "fakeswarm"
    sys.path[:] = saved_path
    for name in [n for n in sys.modules if n == "fakeswarm" or n.startswith("fakeswarm.")]:
        del sys.modules[name]


def test_root_is_put_first_on_sys_path(fake_package: tuple[Path, str]) -> None:
    """The repository root always wins import resolution."""
    root, package = fake_package
    app.ensure_fresh_package(root, package)
    assert sys.path[0] == str(root)
    assert sys.path.count(str(root)) == 1


def test_fresh_package_is_not_purged(fake_package: tuple[Path, str]) -> None:
    """Unchanged modules from the right place are kept (no needless reloads on every rerun)."""
    root, package = fake_package
    app.ensure_fresh_package(root, package)
    importlib.import_module(f"{package}.simulation")
    assert app.ensure_fresh_package(root, package) == []


def test_changed_source_is_purged_and_reimported(fake_package: tuple[Path, str]) -> None:
    """Reproduces the crash: new code imports a symbol the stale module lacks."""
    root, package = fake_package
    app.ensure_fresh_package(root, package)
    importlib.import_module(f"{package}.simulation")
    sim_file = root / package / "simulation.py"
    sim_file.write_text("OLD = False\nREASON_COMPLETE = 'done'\n", encoding="utf-8")
    later = sys.modules[package].__imported_at__ + 5
    os.utime(sim_file, (later, later))  # a git pull after the import
    stale = importlib.import_module(f"{package}.simulation")
    assert not hasattr(stale, "REASON_COMPLETE")  # "cannot import name": what Streamlit Cloud hit
    purged = app.ensure_fresh_package(root, package)
    assert purged == [package, f"{package}.simulation"]
    fresh = importlib.import_module(f"{package}.simulation")
    assert fresh.REASON_COMPLETE == "done"


def test_old_or_foreign_copies_are_purged(
    fake_package: tuple[Path, str], tmp_path_factory: pytest.TempPathFactory
) -> None:
    """A copy without an import timestamp (pre-fix code) or loaded from elsewhere is stale."""
    root, package = fake_package
    other = tmp_path_factory.mktemp("elsewhere")
    (other / package).mkdir()
    (other / package / "__init__.py").write_text(INIT.format(version=0), encoding="utf-8")
    sys.path.insert(0, str(other))
    importlib.import_module(package)
    assert app.ensure_fresh_package(root, package) == [package]  # loaded from the wrong place
    assert importlib.import_module(package).VERSION == 1
    sys.modules[package].__dict__.pop("__imported_at__")
    assert app.ensure_fresh_package(root, package) == [package]  # an old copy without the timestamp
    assert importlib.import_module(package).VERSION == 1


def test_real_package_is_fresh_in_this_process() -> None:
    """In a normal process (tests, local runs) nothing is purged."""
    assert app.ensure_fresh_package() == []
