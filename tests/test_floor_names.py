"""A name a module reads at import time has to exist on the oldest interpreter.

Three CI lanes went red at once on a `from typing import NotRequired` written at
module scope: the name reaches `typing` in 3.11, the floor is 3.10, and a name
missing at import time is a *collection* error, so the module took the whole
file down rather than one test. A fourth lane went red the same day on
`enum.StrEnum`, which arrives in the same release. Both passed on the machine
they were written on, because that machine runs the newest interpreter in the
matrix and the difference only exists on the oldest.

The local gate runs the suite once, on whichever interpreter the caller has, so
it cannot see this at all: the failure is a property of a Python the caller is
not running. Running the suite on five interpreters would see it and costs five
builds of the extension. Reading it costs nothing, and the thing being read is
small: which release each public `typing` and `enum` name arrives in.

So `floor_names.json` carries that, and this refuses a name a module reaches at
import time before the release that spells it. The guard the tree already uses
counts: inside ``if sys.version_info >= (3, 11):`` the lowest interpreter that
reaches the line is 3.11, and the name is judged against *that*, which catches
the other half of the same mistake -- a name guarded at 3.11 that arrives in
3.13.

What is not judged: anything inside a function (it runs when a test runs, and a
test below the floor skips), anything under `if TYPE_CHECKING:` (it never runs),
and annotations in a module carrying `from __future__ import annotations` (they
are strings). Those are the forms the tree uses deliberately.

`scripts/floor_names.py` assembles the table by asking every release it spans,
and the nightly lane runs it with `--check`, so a release adding or removing a
name is read rather than remembered. It asks `hasattr` and not `dir`, which are
different questions here: a deprecated alias `typing` serves through a module
`__getattr__` is absent from one and present to the other, and the first access
writes it into the module so that asking in the wrong order changes the answer.

The table is held to the interpreter running this suite, in both directions, by
`test_the_table_agrees_with_this_interpreter`: every name it dates at or below
this release exists here, and every name it dates above this one does not. The
matrix runs five interpreters, so a wrong row fails on the lane that disproves
it rather than waiting for a reader.

LEDGER: every typing and enum name read at import time exists on the floor
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import NamedTuple

import pytest

# Reading the tree's own sources, which is a repository check rather than a
# claim about the library.
pytestmark = pytest.mark.repository

ROOT = Path(__file__).resolve().parent.parent
TABLE = json.loads(
    (Path(__file__).resolve().parent / "floor_names.json").read_text(encoding="utf-8")
)

#: The modules whose names are dated. Both are stdlib modules that grew typing
#: and enumeration forms across the supported releases, and both are read at
#: module scope by tests that describe those forms.
WATCHED = ("typing", "enum")

#: Every source the suite imports on every interpreter in the matrix: the tests
#: themselves, the shipped package, and the gate scripts the repository tests
#: import to read.
SCANNED = ("tests/*.py", "python/valgebra/**/*.py", "scripts/*.py")


def _version(spelled: str) -> tuple[int, int]:
    major, minor = spelled.split(".")
    return int(major), int(minor)


FLOOR = _version(TABLE["floor"])
KNOWN_THROUGH = _version(TABLE["known_through"])


class Use(NamedTuple):
    """A name a module reaches at import time, and the releases that reach it."""

    where: str
    line: int
    module: str
    name: str
    lowest: tuple[int, int]
    """The oldest interpreter that executes the line: the floor, or the guard's."""
    highest: tuple[int, int]
    """The newest that does: the table's top, or a `< (3, N)` guard's."""


def _guard(test: ast.expr) -> tuple[str, tuple[int, int]] | None:
    """Read a `sys.version_info` comparison as the bound it puts on the body.

    ``>=`` raises the body's floor and ``<`` lowers its ceiling, which are the
    two ways a module says "this name is not there in every release I run on".
    The `else` of either is the other bound, and `_Reader` reads it that way.
    """
    if not isinstance(test, ast.Compare) or len(test.ops) != 1:
        return None
    if isinstance(test.ops[0], (ast.GtE, ast.Gt)):
        side = "floor"
    elif isinstance(test.ops[0], (ast.Lt, ast.LtE)):
        side = "ceiling"
    else:
        return None
    if "version_info" not in ast.dump(test.left):
        return None
    right = test.comparators[0]
    if not isinstance(right, ast.Tuple) or len(right.elts) < 2:
        return None
    parts = [
        part.value
        for part in right.elts[:2]
        if isinstance(part, ast.Constant) and isinstance(part.value, int)
    ]
    if len(parts) != 2:
        return None
    major, minor = parts
    return side, (major, minor)


def _type_checking(test: ast.expr) -> bool:
    return "TYPE_CHECKING" in ast.dump(test)


class Reader(ast.NodeVisitor):
    """Collect the watched names a module reaches while it is being imported."""

    def __init__(self, where: str, *, deferred: bool) -> None:
        self.where = where
        self.deferred = deferred
        self.alias: dict[str, str] = {}
        self.lowest = FLOOR
        self.highest = KNOWN_THROUGH
        self.uses: list[Use] = []

    # A body that runs when a test runs, not when the module is imported. A
    # test that needs a newer name skips below the release that spells it, and
    # `pytest.mark.skipif` is how the tree already says so.
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: ARG002
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: ARG002
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: ARG002
        return

    def visit_If(self, node: ast.If) -> None:
        if _type_checking(node.test):
            return  # never executed; the annotations that read it are strings
        guard = _guard(node.test)
        if guard is None:
            self.generic_visit(node)
            return
        side, at = guard
        # A `>=` guard's body is reached at or above it and its `else` below it;
        # a `<` guard is the same statement read the other way round. Either way
        # the two branches carry opposite bounds, and the visit of each is the
        # visit of a module with that release range.
        body = (max(self.lowest, at), self.highest)
        rest = (self.lowest, min(self.highest, _below(at)))
        if side == "ceiling":
            body, rest = (
                (self.lowest, min(self.highest, _below(at))),
                (
                    max(self.lowest, at),
                    self.highest,
                ),
            )
        outer = (self.lowest, self.highest)
        self.lowest, self.highest = body
        for statement in node.body:
            self.visit(statement)
        self.lowest, self.highest = rest
        for statement in node.orelse:
            self.visit(statement)
        self.lowest, self.highest = outer

    def visit_Try(self, node: ast.Try) -> None:
        # `try: from typing import X / except ImportError:` is a name the module
        # already handles being without.
        caught = " ".join(ast.dump(handler) for handler in node.handlers)
        if "ImportError" in caught or "AttributeError" in caught:
            return
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if not self.deferred:
            self.visit(node.annotation)
        if node.value is not None:
            self.visit(node.value)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name in WATCHED:
                self.alias[alias.asname or alias.name] = alias.name

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module not in WATCHED:
            return
        for alias in node.names:
            self.uses.append(
                Use(
                    self.where,
                    node.lineno,
                    node.module,
                    alias.name,
                    self.lowest,
                    self.highest,
                )
            )

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.value, ast.Name) and node.value.id in self.alias:
            self.uses.append(
                Use(
                    self.where,
                    node.lineno,
                    self.alias[node.value.id],
                    node.attr,
                    self.lowest,
                    self.highest,
                )
            )
        self.generic_visit(node)


def _sources() -> list[Path]:
    return sorted({path for pattern in SCANNED for path in ROOT.glob(pattern)})


def _read(path: Path) -> list[Use]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    deferred = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "__future__"
        and any(alias.name == "annotations" for alias in node.names)
        for node in tree.body
    )
    reader = Reader(str(path.relative_to(ROOT)), deferred=deferred)
    reader.visit(tree)
    return reader.uses


def _dated(
    module: str, name: str
) -> tuple[tuple[int, int], tuple[int, int] | None] | None:
    """Read the releases the table says a name exists in: `since` until `gone`."""
    span = TABLE["modules"][module].get(name)
    if span is None:
        return None
    gone = span.get("gone")
    return _version(span["since"]), (None if gone is None else _version(gone))


def _below(version: tuple[int, int]) -> tuple[int, int]:
    """Give the release just below `version`, for the far side of a guard."""
    major, minor = version
    return (major, minor - 1) if minor else (major - 1, 99)


def test_no_module_reads_a_name_outside_the_releases_that_have_it() -> None:
    """The failure that reddened four lanes, read off the sources instead.

    Both edges of the span: a name is missing below the release that adds it and
    missing again at or above one that removes it, and a module that reads it
    outside those bounds breaks at import on the release where it is not there.
    """
    outside = []
    for path in _sources():
        for use in _read(path):
            dated = _dated(use.module, use.name)
            if dated is None:
                continue
            since, gone = dated
            if since > use.lowest:
                outside.append(
                    f"{use.where}:{use.line}: {use.module}.{use.name} arrives in "
                    f"{since[0]}.{since[1]}, and this line is reached on "
                    f"{use.lowest[0]}.{use.lowest[1]}"
                )
            if gone is not None and gone <= use.highest:
                outside.append(
                    f"{use.where}:{use.line}: {use.module}.{use.name} is gone in "
                    f"{gone[0]}.{gone[1]}, and this line is reached on "
                    f"{use.highest[0]}.{use.highest[1]}"
                )
    assert not outside, (
        "names read at import time on a release that does not have them:\n"
        + "\n".join(outside)
        + "\nA name above the floor sits behind `if sys.version_info >= (3, N):` "
        "with the test that reads it skipped below N, which is what "
        "`tests/_deferred.py` does; one a release removes sits behind the same "
        "guard written `<`."
    )


def test_every_name_read_is_one_the_table_dates() -> None:
    """A name in no row is a typo, or a release the table has not been told about."""
    undated = [
        f"{use.where}:{use.line}: {use.module}.{use.name}"
        for path in _sources()
        for use in _read(path)
        if _dated(use.module, use.name) is None
    ]
    assert not undated, (
        "names no row of floor_names.json dates:\n"
        + "\n".join(undated)
        + f"\nThe table covers {TABLE['floor']} through {TABLE['known_through']}. "
        "A name newer than that needs the table extended with the release that "
        "adds it; a name in none of them is misspelled."
    )


def test_the_table_agrees_with_this_interpreter() -> None:
    """Hold every row to the interpreter running it, in both directions.

    A dated table nobody re-derives goes stale silently. This one is checked by
    the matrix instead: each lane runs a different release, so a row claiming a
    name arrives later than it does fails on the lane that already has it, and a
    row claiming one arrives earlier fails on the lane that does not.
    """
    running = sys.version_info[:2]
    wrong = []
    for module in WATCHED:
        present = __import__(module)
        for name in TABLE["modules"][module]:
            dated = _dated(module, name)
            assert dated is not None
            since, gone = dated
            expected = since <= running and (gone is None or running < gone)
            here = hasattr(present, name)
            span = f"{since[0]}.{since[1]}" + (
                "" if gone is None else f" until {gone[0]}.{gone[1]}"
            )
            if expected and not here:
                wrong.append(f"{module}.{name} is dated {span} and is absent here")
            # Above the release the table was derived through, a name may have
            # been added -- or brought back -- by a release nobody has told it
            # about, so only the releases it covers can say one should be missing.
            if not expected and here and running <= KNOWN_THROUGH:
                wrong.append(f"{module}.{name} is dated {span} and is present here")
    assert not wrong, (
        f"floor_names.json disagrees with Python {running[0]}.{running[1]}:\n"
        + "\n".join(wrong)
    )


def test_the_reader_sees_the_two_forms_that_reddened_the_lanes() -> None:
    """The check itself is falsifiable: both forms, read, are both refused."""
    planted = (
        "import enum\n"
        "import sys\n"
        "from typing import NotRequired\n"
        "\n"
        "class Planted(enum.StrEnum):\n"
        "    a = 'a'\n"
        "\n"
        "if sys.version_info >= (3, 11):\n"
        "    from typing import ReadOnly\n"
    )
    reader = Reader("planted.py", deferred=False)
    reader.visit(ast.parse(planted))
    early = {
        (use.module, use.name)
        for use in reader.uses
        if (dated := _dated(use.module, use.name)) is not None and dated[0] > use.lowest
    }
    assert early == {
        ("typing", "NotRequired"),
        ("enum", "StrEnum"),
        ("typing", "ReadOnly"),
    }


def test_the_reader_sees_a_name_a_release_takes_away() -> None:
    """The other edge, and the guard that answers it.

    `typing.no_type_check_decorator` is in every release from the floor and gone
    in 3.15, which is how this half was found: a lane on the newest interpreter
    failed the table rather than the tree. A module that reads such a name
    unguarded is refused; one that reads it under `< (3, 15)` is not, because
    that is the release range where it is there.
    """
    unguarded = "import typing\n\nDECORATOR = typing.no_type_check_decorator\n"
    reader = Reader("planted.py", deferred=False)
    reader.visit(ast.parse(unguarded))
    gone = [
        use
        for use in reader.uses
        if (dated := _dated(use.module, use.name)) is not None
        and dated[1] is not None
        and dated[1] <= use.highest
    ]
    assert [(use.module, use.name) for use in gone] == [
        ("typing", "no_type_check_decorator")
    ]

    guarded = (
        "import sys\n"
        "import typing\n"
        "\n"
        "if sys.version_info < (3, 15):\n"
        "    DECORATOR = typing.no_type_check_decorator\n"
    )
    reader = Reader("planted.py", deferred=False)
    reader.visit(ast.parse(guarded))
    still_gone = [
        use
        for use in reader.uses
        if (dated := _dated(use.module, use.name)) is not None
        and dated[1] is not None
        and dated[1] <= use.highest
    ]
    assert still_gone == []
