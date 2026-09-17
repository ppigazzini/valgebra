"""Every outcome the surface documents is one a test asserts.

`tests/test_use_case_ledger.py` holds the public *names*: a method the tree grows
and the suite never mentions fails there, which is the gap no reading catches.
It says plainly what it cannot do -- "naming a method is not asserting its
documented outcome" -- and this is the half it names.

The universe is the **outcomes**, read out of the binding's own doc comments,
and a method has two kinds of them.

A `Raises:` block names the exceptions a call can raise, and each
`(method, exception)` pair is a cell. A `Returns:` block names what the call
answers with, and every value it spells in backticks -- `True`, `False`, `None`,
`"subset"` -- is a cell of its own. Both are derivations rather than lists: a
method that grows a documented outcome arrives here without a row, and one whose
line goes takes its row with it.

A raise is covered when the product suite puts the call inside a `pytest.raises`
for that exception. A return is covered when the suite compares that call to
that value -- `assert x.is_valid(v)` for `True`, `assert not x.is_valid(v)` for
`False`, an `==` or an `is` against the value otherwise. Both are read from the
syntax tree rather than by searching the text, so a call in a comment or a
mention in a docstring is not one.

What that catches is a documented outcome nobody drives: the shape the audits
kept finding by hand, where a page promises a `TypeError` and the suite calls the
method only on values that succeed -- or promises three answers and the suite
asserts two.

The other direction is the accepted list: a cell the suite cannot reach is
written down with a reason, and a reason for a cell that *is* reached fails too,
so an excuse cannot outlive the gap it excuses.

LEDGER: every outcome the binding's docstrings name is asserted by a test
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

#: A value a `Returns:` block spells. Backticked, because that is how the block
#: writes a value as against a description of one: "`True` if ... else `False`"
#: names two answers, and "The parsed Python object" names none this can check.
_RETURNS = re.compile(r"`(True|False|None|\"[a-z_]+\")`")

#: The block a value is read from stops at the next one.
_BLOCKS = ("Args:", "Returns:", "Raises:")

#: What a constructor is called as from Python. The binding names the method
#: `py_new`; a caller writes the class.
_AS_WRITTEN = {"py_new": "Validator"}

#: Cells the product suite does not drive, each with the reason it does not.
#: A reason is a sentence about why the cell is unreachable from the suite, not
#: a note that nobody has written the row yet. Keyed by `(method, outcome)`,
#: which is one space for both kinds: an exception a call raises and a value it
#: answers with are both outcomes, and an excuse for either expires the same way.
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


def _block(doc: str, name: str) -> str:
    """Give the named doc block's body, which ends where the next block starts."""
    if name not in doc:
        return ""
    rest = doc.split(name, 1)[1]
    ends = [rest.index(other) for other in _BLOCKS if other in rest]
    return rest[: min(ends)] if ends else rest


def _cells() -> set[tuple[str, str]]:
    """Every `(method as written, exception)` the binding documents."""
    found: set[tuple[str, str]] = set()
    for path in sorted(BINDING.rglob("*.rs")):
        text = path.read_text(encoding="utf-8")
        for doc, function in _DOCUMENTED.findall(text):
            for exception in _RAISES.findall(_block(doc, "Raises:")):
                found.add((_AS_WRITTEN.get(function, function), exception))
    return found


def _returns() -> set[tuple[str, str]]:
    """Every `(method as written, value)` a `Returns:` block spells."""
    found: set[tuple[str, str]] = set()
    for path in sorted(BINDING.rglob("*.rs")):
        text = path.read_text(encoding="utf-8")
        for doc, function in _DOCUMENTED.findall(text):
            for value in _RETURNS.findall(_block(doc, "Returns:")):
                found.add((_AS_WRITTEN.get(function, function), value))
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


def _asserted_returns() -> set[tuple[str, str]]:
    """Every `(call, value)` the suite ties together with an assertion.

    Two idioms and no third. A bare `assert x.method(v)` pins the answer to
    `True` and `assert not x.method(v)` pins it to `False`; anything else is a
    comparison, where one side is the call and the other is the value. A test
    whose assertion is a conjunction is *not* read as pinning either call, on
    purpose: `assert a.is_valid(v) or b` holds whatever `b` does, and counting it
    would mark a cell covered by an assertion that cannot fail on it.
    """
    found: set[tuple[str, str]] = set()
    for path in sorted(SUITE.rglob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                found |= _compared(node)
            elif isinstance(node, ast.Assert):
                test, answer = node.test, "True"
                if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
                    test, answer = test.operand, "False"
                if isinstance(test, ast.Call) and (name := _called(test)):
                    found.add((name, answer))
    return found


def _compared(node: ast.Compare) -> set[tuple[str, str]]:
    """Pair a call with the value an `==` or an `is` puts on the other side."""
    sides = [node.left, *node.comparators]
    if len(sides) != 2 or not all(isinstance(op, ast.Eq | ast.Is) for op in node.ops):
        return set()
    found: set[tuple[str, str]] = set()
    for call, other in (sides, sides[::-1]):
        if (
            isinstance(call, ast.Call)
            and (name := _called(call))
            and isinstance(other, ast.Constant)
        ):
            found.add((name, _spelled(other.value)))
    return found


def _spelled(value: object) -> str:
    """Write a constant the way the `Returns:` block writes it."""
    return f'"{value}"' if isinstance(value, str) else repr(value)


def _called(call: ast.Call) -> str | None:
    """Name what a call calls, method or plain function alike."""
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return call.func.id if isinstance(call.func, ast.Name) else None


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

    answers = _returns()
    assert len(answers) >= 12, sorted(answers)
    # The three shapes a `Returns:` block writes a value in: the two booleans a
    # predicate answers with, the nothing a raising entry point answers with,
    # and the strings a three-valued answer names.
    assert ("is_valid", "True") in answers, sorted(answers)
    assert ("validate", "None") in answers, sorted(answers)
    assert ("relation_to", '"undecided"') in answers, sorted(answers)


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


def test_every_documented_return_is_asserted_somewhere() -> None:
    """A documented answer is one the suite pins, or one with a reason.

    The gap this closes is a method that answers three ways and is driven for
    two: the third arm is reachable, nothing reaches it, and the sweep reports
    the code as covered because another row runs the same line.
    """
    asserted = _asserted_returns()
    empty = sorted(
        cell for cell in _returns() if cell not in asserted and cell not in ACCEPTED
    )
    assert not empty, (
        "documented return values no test asserts:\n"
        + "\n".join(f"  {method}() answering {value}" for method, value in empty)
        + "\n\nAssert the call against that value, or accept the cell with the "
        "reason it cannot be reached."
    )


def test_no_reason_outlives_the_gap_it_excuses() -> None:
    """An accepted cell the suite does reach is an excuse that has expired."""
    asserted = _asserted()
    answered = _asserted_returns()
    reached = sorted(
        cell for cell in ACCEPTED if _covers(cell, asserted) or cell in answered
    )
    assert not reached, f"accepted cells the suite asserts after all: {reached}"

    unknown = sorted(set(ACCEPTED) - _cells() - _returns())
    assert not unknown, f"reasons for cells the binding does not document: {unknown}"

    for cell, reason in ACCEPTED.items():
        assert len(reason) > 40, f"{cell}: {reason!r}"
