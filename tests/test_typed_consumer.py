"""Every name the stub declares is one a caller's checker agrees the type of.

`tests/typing/consumer.py` is a caller's code, run through `mypy --strict` and
`pyright` on the floor and on the current interpreter. Binding a result to an
annotated variable would ask the checker only whether the result is
*assignable* to that type -- and `Any` is assignable to everything, so a stub
signature that degraded to `Any`, or a method the extension stopped declaring,
would pass in silence, which is the failure the file exists to catch.

`assert_type` asks the other question: whether the checker's own view of the
expression is that type exactly. It fails on `Any`. So each name is exercised
through one, and this ledger holds the set of names to the surface: every name
the stub declares, a dunder as much as a method, since the runtime calls it and
an explicit call reads its declaration; and every name the package exports in
`__all__`, which reaches `Regex`, declared in Python beside the stub. A name with
one leading underscore is private by the convention every checker keeps, and
is outside what a caller types against.

* a name the stub declares and the consumer does not put through `assert_type`
  fails here, so a method added to the surface arrives with a caller using it;
* a name accepted without one carries the reason, and a reason for a name that
  *is* checked fails too.

What it cannot do is run the checker -- that is the type-check lane's, twice,
because a stub can be right for one interpreter and wrong for the other.

LEDGER: every public name is put through assert_type by the typed consumer
"""

from __future__ import annotations

import ast
import re

import pytest

from _reason import is_a_reason
from _tree import ROOT

pytestmark = pytest.mark.repository

STUB = ROOT / "python" / "valgebra" / "_valgebra.pyi"
PACKAGE = ROOT / "python" / "valgebra" / "__init__.py"
CONSUMER = ROOT / "tests" / "typing" / "consumer.py"

#: Names the consumer reaches without an `assert_type`, each with the reason.
#:
#: A reason is a sentence about the name, not a note that nobody got to it.
ACCEPTED: dict[str, str] = {
    "__new__": (
        "the constructor is spelled `Validator(...)` at every call, and the "
        "type of that expression is what every other row asserts the receiver "
        "of; a row here would assert the same thing about the same expression"
    ),
    "__eq__": (
        "`==` between two validators is checked by the rows that compare them, "
        "and `assert_type` over an operator asserts what the checker already "
        "special-cases: every `__eq__` returns `bool` by the data model"
    ),
    "__hash__": (
        "a hash is reached through `hash()`, whose return type comes from the "
        "builtin rather than from this stub, so the row would assert the "
        "builtin's signature"
    ),
    "__reduce__": (
        "the method exists to refuse: it is declared `NoReturn`, so an "
        "expression calling it is unreachable and a checker gives the code "
        "after it no type at all"
    ),
    "__contains__": (
        "`in` is checked by the row that writes it, and the data model makes "
        "the expression `bool` whatever the method declares"
    ),
    "ValidationError": (
        "a caller reaches the class as what a call raises, and `except "
        "ValidationError as error` types `error` by the class itself; the rows on "
        "`error` assert every attribute the stub declares on it"
    ),
    "__class_getitem__": (
        "a checker reads `Validator[int]` as a generic alias through the class's "
        "type parameter, never through this method, which the runtime calls; a "
        "row calling it would assert typeshed's `GenericAlias`"
    ),
}


def _exported() -> set[str]:
    """Give the names the package's `__all__` lists."""
    tree = ast.parse(PACKAGE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            return set(ast.literal_eval(node.value))
    pytest.fail("the package lists no __all__")


def _private(name: str) -> bool:
    """Answer whether one leading underscore makes the name private."""
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def _declared() -> set[str]:
    """Give every public name the stub declares or the package exports.

    Methods by their own name rather than by `Class.method`: the consumer calls
    them on a receiver whose type the checker knows, and two classes do not
    share a method name here.
    """
    tree = ast.parse(STUB.read_text(encoding="utf-8"))
    names = _exported()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return {name for name in names if not _private(name)}


def _asserted() -> set[str]:
    """Give every name appearing inside an `assert_type` call's first argument.

    The first argument only: the second is the type, and a row asserting that a
    call returns `Validator` would otherwise read as a row about `Validator`.
    """
    tree = ast.parse(CONSUMER.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "assert_type"
            and node.args
        ):
            found.update(
                inner.attr if isinstance(inner, ast.Attribute) else inner.id
                for inner in ast.walk(node.args[0])
                if isinstance(inner, ast.Attribute | ast.Name)
            )
    return found


def test_the_stub_declares_a_surface_to_check() -> None:
    """The parse is a detector, so it must be shown to have read something."""
    declared = _declared()
    assert len(declared) >= 25, sorted(declared)
    for name in ("is_valid", "relation_to", "union", "MAX_SCHEMA_DEPTH"):
        assert name in declared, sorted(declared)
    # A dunder is read and a private name is not, and `__all__` reaches past
    # the stub.
    assert {"__or__", "__copy__", "Regex"} <= declared, sorted(declared)
    assert "_debug_build" not in declared, sorted(declared)


def test_the_consumer_uses_assert_type_rather_than_assignment_alone() -> None:
    """The detector must find the calls, or every row below passes vacuously."""
    asserted = _asserted()
    assert len(asserted) >= 20, sorted(asserted)
    text = CONSUMER.read_text(encoding="utf-8")
    assert "assert_type" in text


@pytest.mark.parametrize("name", sorted(_declared()))
def test_every_declared_name_is_put_through_assert_type(name: str) -> None:
    """A name the stub declares and no caller pins the type of fails here."""
    checked = name in _asserted()
    accepted = name in ACCEPTED
    assert checked or accepted, (
        f"{name} is declared by the stub and the typed consumer never asserts "
        "its type, so a signature degrading to Any would pass the lane"
    )
    assert not (checked and accepted), (
        f"{name} is asserted and still carries a reason for not being"
    )


def test_every_accepted_reason_is_a_sentence() -> None:
    """An excuse short enough to be a shrug is not one."""
    declared = _declared()
    for name, reason in ACCEPTED.items():
        assert is_a_reason(reason), f"{name}: {reason!r}"
        assert name in declared, f"{name} is accepted and the stub does not declare it"


def test_the_consumer_reads_assert_type_on_the_floor_too() -> None:
    """`assert_type` and `Never` arrive in 3.11 and the floor is 3.10.

    The lane checks the file against both, so the import is version-gated and
    the older half comes from `typing_extensions`. Written as a check on the
    source because nothing imports this file at runtime: it is a caller's code
    for a checker to read, and pytest does not collect it.
    """
    text = CONSUMER.read_text(encoding="utf-8")
    assert re.search(r"sys\.version_info >= \(3, 11\)", text), (
        "the consumer reads `assert_type` with no version gate, and the lane "
        "checks it against 3.10, where `typing` does not have it"
    )
    floor_import = r"^\s+from typing_extensions import .*\bassert_type\b"
    assert re.search(floor_import, text, re.MULTILINE)
