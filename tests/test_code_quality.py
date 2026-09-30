"""Structural code-quality rules that linters do not cover."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCES = sorted([*ROOT.glob("swarmrescue/*.py"), ROOT / "app.py", ROOT / "main.py", *ROOT.glob("scripts/*.py")])
MAX_FUNCTION_LINES = 40
FORBIDDEN_CALLS = {"eval", "exec", "compile", "__import__"}
FORBIDDEN_MODULES = {"pickle", "marshal", "shelve", "subprocess", "os.system"}


def functions(tree: ast.AST) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """All function and method definitions in a module."""
    return [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)]


def code_lines(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Lines of a function excluding its docstring (documentation is never penalised)."""
    total = (fn.end_lineno or fn.lineno) - fn.lineno + 1
    first = fn.body[0]
    if ast.get_docstring(fn) and isinstance(first, ast.Expr):
        total -= (first.end_lineno or first.lineno) - first.lineno + 1
    return total


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_functions_are_short_documented_and_typed(path: Path) -> None:
    """Every function has at most 40 code lines (docstring excluded), a docstring and full type hints (PEP 484)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for fn in functions(tree):
        where = f"{path.name}:{fn.lineno} {fn.name}"
        length = code_lines(fn)
        assert length <= MAX_FUNCTION_LINES, f"{where} is {length} lines"
        assert ast.get_docstring(fn), f"{where} has no docstring"
        assert fn.returns is not None, f"{where} has no return annotation"
        args = [*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs]
        for arg in args:
            assert arg.arg in {"self", "cls"} or arg.annotation is not None, f"{where}: '{arg.arg}' is untyped"


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_no_dynamic_code_or_unsafe_modules(path: Path) -> None:
    """No eval/exec/compile, pickle-style deserialisation or shell calls in shipped code."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in FORBIDDEN_CALLS, f"{path.name}:{node.lineno} calls {node.func.id}"
        if isinstance(node, ast.Import):
            assert not {a.name for a in node.names} & FORBIDDEN_MODULES, f"{path.name}:{node.lineno}"
        if isinstance(node, ast.ImportFrom):
            assert node.module not in FORBIDDEN_MODULES, f"{path.name}:{node.lineno}"
