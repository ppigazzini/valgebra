"""Every outcome the surface documents is one a test asserts.

`tests/test_use_case_ledger.py` holds the public *names*: a method the tree grows
and the suite never mentions fails there, which is the gap no reading catches.
It says plainly what it cannot do -- "naming a method is not asserting its
documented outcome" -- and this is the half it names.

The universe is the **raises**, read out of the binding's own doc comments.
Every method whose docstring carries a `Raises:` block names the exceptions it
can raise, and each `(method, exception)` pair is a cell. That is a derivation
rather than a list: a method that grows a `Raises:` line arrives here without a
row, and one whose line goes takes its row with it.

A cell is covered when the product suite puts the call inside a
`pytest.raises` for that exception -- read from the syntax tree rather than by
searching the text, so a call in a comment or a mention in a docstring is not
one. What that catches is a documented failure nobody drives: the shape the
audits kept finding by hand, where a page promises a `TypeError` and the suite
calls the method only on values that succeed.

The other direction is the accepted list: a cell the suite cannot reach is
written down with a reason, and a reason for a cell that *is* reached fails too,
so an excuse cannot outlive the gap it excuses.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.repository

ROOT = Path(__file__).resolve().parent.parent
BINDING = ROOT / "crates" / "valgebra-py" / "src"
SUITE = ROOT / "tests"

#: A doc comment block and the function it sits above.
_DOCUMENTED = re.compile(
    r"((?:^[ \t]*///.*\n)+)[ \t]*(?:#\[[^\]]*\]\s*)*"
    r"[ \t]*(?:pub(?:\([^)]*\))?\s+)?fn\s+(\w+)",
    re.MULTILINE,
)
#: A line of a `Raises:` block: an exception name and what provokes it.
_RAISES = re.compile(r"///\s+(\w*(?:Error|Exception)|StopIteration|KeyboardInterrupt):")

#: What a constructor is called as from Python. The binding names the method
#: `py_new`; a caller writes the class.
_AS_WRITTEN = {"py_new": "Validator"}

#: Cells the product suite does not drive, each with the reason it does not.
#: A reason is a sentence about why the cell is unreachable from the suite, not
#: a note that nobody has written the row yet.
ACCEPTED: dict[tuple[str, str], str] = {
    ("simplify", "ValueError"): (
        "the expansion the docstring names does not happen: a schema is built in "
        "the lattice normal form, so a complement is already distributed by the "
        "constructors and `simplify` has nothing to grow. Driven at a union of "
        "thirty thousand records under a complement -- past the size a build "
        "itself refuses -- the call returns rather than raising. The method is "
        "deprecated and goes with the next minor, and the docstring is stale "
        "rather than the bound unreachable."
    ),
}


def _cells() -> set[tuple[str, str]]:
    """Every `(method as written, exception)` the binding documents."""
    found: set[tuple[str, str]] = set()
    for path in sorted(BINDING.rglob("*.rs")):
        text = path.read_text(encoding="utf-8")
        for doc, function in _DOCUMENTED.findall(text):
            if "Raises:" not in doc:
                continue
            for exception in _RAISES.findall(doc.split("Raises:", 1)[1]):
                found.add((_AS_WRITTEN.get(function, function), exception))
    return found


def _asserted() -> set[tuple[str, str]]:
    """Every `(call, exception)` the suite puts inside a `pytest.raises`."""
    found: set[tuple[str, str]] = set()
    for path in sorted(SUITE.rglob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.With):
                continue
            for item in node.items:
                raised = _raised_by(item.context_expr)
                if raised is None:
                    continue
                for call in _calls_in(node.body):
                    found.add((call, raised))
    return found


def _raised_by(expression: ast.expr) -> str | None:
    """Give the exception a `pytest.raises(...)` names, or `None` for anything else."""
    if not isinstance(expression, ast.Call):
        return None
    function = expression.func
    if not (isinstance(function, ast.Attribute) and function.attr == "raises"):
        return None
    if not expression.args:
        return None
    first = expression.args[0]
    if isinstance(first, ast.Name):
        return first.id
    return first.attr if isinstance(first, ast.Attribute) else None


def _calls_in(body: list[ast.stmt]) -> set[str]:
    """Every name called inside a block, method or plain function alike."""
    called: set[str] = set()
    for statement in body:
        for node in ast.walk(statement):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Attribute):
                called.add(node.func.attr)
            elif isinstance(node.func, ast.Name):
                called.add(node.func.id)
    return called


def test_the_universe_is_read_out_of_the_binding() -> None:
    """The derivation is the detector, so it must be shown to have read something.

    A ledger over an empty universe passes having checked nothing, which is the
    failure every other ledger here guards the same way.
    """
    cells = _cells()
    assert len(cells) >= 8, sorted(cells)
    # And it reads the two shapes the binding writes: a constructor under the
    # name a caller uses, and a method under its own.
    assert any(method == "Validator" for method, _ in cells), sorted(cells)
    assert any(method == "validate" for method, _ in cells), sorted(cells)


def _covers(cell: tuple[str, str], asserted: set[tuple[str, str]]) -> bool:
    """Whether the suite asserts a cell, reading `BaseException` as the top.

    A docstring saying `BaseException` means *whatever* signal reaches the
    caller -- a fatal one the walk must not fold -- and the suite drives the
    concrete signals, parametrised over them. `BaseException` is the top of the
    exception lattice, so any raise asserted through that call is below it and
    covers the cell; reading the name literally would leave a cell empty that
    four rows drive, which is a ledger measuring its own spelling.
    """
    method, exception = cell
    if exception == "BaseException":
        return any(call == method for call, _ in asserted)
    return cell in asserted


def test_every_documented_raise_is_asserted_somewhere() -> None:
    """A documented failure is one the suite drives, or one with a reason."""
    asserted = _asserted()
    empty = sorted(
        cell
        for cell in _cells()
        if not _covers(cell, asserted) and cell not in ACCEPTED
    )
    assert not empty, (
        "documented raises no test asserts:\n"
        + "\n".join(f"  {method}() raising {exception}" for method, exception in empty)
        + "\n\nPut the call inside a `pytest.raises` for that exception, or "
        "accept the cell with the reason it cannot be reached."
    )


def test_no_reason_outlives_the_gap_it_excuses() -> None:
    """An accepted cell the suite does reach is an excuse that has expired."""
    asserted = _asserted()
    reached = sorted(cell for cell in ACCEPTED if _covers(cell, asserted))
    assert not reached, f"accepted cells the suite asserts after all: {reached}"

    unknown = sorted(set(ACCEPTED) - _cells())
    assert not unknown, f"reasons for cells the binding does not document: {unknown}"

    for cell, reason in ACCEPTED.items():
        assert len(reason) > 40, f"{cell}: {reason!r}"
