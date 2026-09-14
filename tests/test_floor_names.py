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

LEDGER: every typing and enum name read at import time, and every stdlib
module imported, exists on the floor
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


class Imported(NamedTuple):
    """A module a source imports, and the releases that reach the line."""

    where: str
    line: int
    module: str
    lowest: tuple[int, int]
    highest: tuple[int, int]


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
        self.imports: list[Imported] = []
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

    def _imported(self, line: int, dotted: str) -> None:
        """Record the top-level module of a dotted name, which is what ships."""
        top = dotted.split(".", maxsplit=1)[0]
        self.imports.append(Imported(self.where, line, top, self.lowest, self.highest))

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._imported(node.lineno, alias.name)
            if alias.name in WATCHED:
                self.alias[alias.asname or alias.name] = alias.name

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        # A relative import names no top-level module, so there is nothing for
        # the standard library to answer about.
        if not node.level and node.module:
            self._imported(node.lineno, node.module)
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


def _read(path: Path) -> Reader:
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
    return reader


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
        for use in _read(path).uses:
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


def reader_of(source: str) -> Reader:
    """Read a planted source the way `_read` reads a file in the tree."""
    reader = Reader("planted.py", deferred=False)
    reader.visit(ast.parse(source))
    return reader


def _ships(module: str) -> tuple[tuple[int, int], tuple[int, int] | None] | None:
    """Read the releases the table says a stdlib module ships in."""
    span = TABLE["stdlib"].get(module)
    if span is None:
        return None  # not the standard library's: a third-party package, or ours
    gone = span.get("gone")
    return _version(span["since"]), (None if gone is None else _version(gone))


def test_no_module_imports_a_stdlib_module_outside_the_releases_that_ship_it() -> None:
    """The same mistake one level up, which the names half could not see.

    `tomllib` arrives in 3.11 and this suite runs from 3.10, so importing it to
    read a `.toml` file made the whole module fail to *collect* on the floor
    leg -- and the ledger beside this one, which dates every `typing` and `enum`
    name a source reaches, saw nothing, because a module is not a name in one.

    Both edges again: a module is missing below the release that adds it and
    missing again at or above one that removes it, and the standard library
    removes plenty -- `distutils` in 3.12, `telnetlib` in 3.13, `sre_parse` in
    3.15. A module above the floor sits behind `if sys.version_info >= (3, N):`
    like any other name, and a third-party package is dated by no row and is
    not this ledger's business.

    The rows are the modules this tree imports. `sys.stdlib_module_names` is not
    a property of the release alone -- two builds of one version disagree on the
    modules a platform does not need -- so a table of every name would report a
    difference between two builds as a moved row. The row above keeps the table
    from going stale as imports are added.
    """
    outside = []
    for path in _sources():
        for found in _read(path).imports:
            ships = _ships(found.module)
            if ships is None:
                continue
            since, gone = ships
            if since > found.lowest:
                outside.append(
                    f"{found.where}:{found.line}: {found.module} arrives in "
                    f"{since[0]}.{since[1]}, and this line is reached on "
                    f"{found.lowest[0]}.{found.lowest[1]}"
                )
            if gone is not None and gone <= found.highest:
                outside.append(
                    f"{found.where}:{found.line}: {found.module} is gone in "
                    f"{gone[0]}.{gone[1]}, and this line is reached on "
                    f"{found.highest[0]}.{found.highest[1]}"
                )
    assert not outside, (
        "standard-library modules imported on a release that does not ship "
        "them:\n" + "\n".join(outside) + "\nAn import above the floor sits "
        "behind `if sys.version_info >= (3, N):`, or the file reads what it "
        "needs another way -- `tests/test_mutation_scope.py` reads its TOML "
        "with a regex for exactly this reason."
    )


def test_the_reader_sees_a_module_the_floor_does_not_ship() -> None:
    """The plant for the half above: the import that reddened the floor leg.

    Written out rather than reached for, because the tree no longer carries it:
    a check whose only evidence is that the tree passes is a check nobody has
    run against a tree that breaks it.
    """
    planted = "import sys\nimport tomllib\n\nDATA = tomllib.loads('')\n"
    # Its span is written here rather than read from the table, because the
    # table dates what this tree imports and this tree does not import it any
    # more. What the row holds is the rule -- a module reached below the release
    # that ships it -- and the rule is the part a plant has to exercise.
    spans = {"tomllib": ((3, 11), None)}
    early = {
        found.module
        for found in reader_of(planted).imports
        if (ships := spans.get(found.module)) is not None and ships[0] > found.lowest
    }
    assert early == {"tomllib"}, "the floor's own leg caught this and no test did"

    # The guarded form is the spelling that is allowed, and is not reported.
    guarded = "import sys\n\nif sys.version_info >= (3, 11):\n    import tomllib\n"
    behind_a_guard = [
        found.module
        for found in reader_of(guarded).imports
        if (ships := spans.get(found.module)) is not None and ships[0] > found.lowest
    ]
    assert not behind_a_guard


def test_every_name_read_is_one_the_table_dates() -> None:
    """A name in no row is a typo, or a release the table has not been told about."""
    undated = [
        f"{use.where}:{use.line}: {use.module}.{use.name}"
        for path in _sources()
        for use in _read(path).uses
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


def test_every_stdlib_module_imported_is_one_the_table_dates() -> None:
    """A table derived from the tree's imports goes stale when an import is added.

    The rows are the modules this tree imports, so an import added without
    re-running `scripts/floor_names.py --update` carries no row -- and a module
    with no row is not judged, which is the silent half of the mistake this
    ledger exists for. The running interpreter's own list says which imports are
    the standard library's to answer for; a third-party package is in neither.
    """
    imported = {found.module for path in _sources() for found in _read(path).imports}
    undated = sorted(imported & sys.stdlib_module_names - set(TABLE["stdlib"]))
    assert not undated, (
        f"standard-library modules this tree imports that no row dates: {undated}. "
        "Run `python scripts/floor_names.py --update`, which asks every release "
        "the table spans and needs all of them installed."
    )


def test_the_stdlib_rows_agree_with_this_interpreter() -> None:
    """The module half of the row above, held the same way and by the same matrix.

    `sys.stdlib_module_names` is the interpreter's own answer, so this compares
    the table with what is running rather than with a memory of it.

    **Over the modules this tree imports, and no others.** The list is not a
    property of the release alone: two builds of one version disagree on the
    modules a platform does not need, and two implementations disagree on more.
    This box's 3.12 ships `_wmi`, which is Windows-only; the runner's 3.12 does
    not, and PyPy answers without `_bisect` or `_ctypes` and with a `_colorize`
    that CPython adds in 3.13. None of those is a wrong row, and holding the
    whole table to one build reports a difference between two builds as one.

    What the table is *used* for is narrower than what it records: the row
    above asks whether an import this tree makes is safe on the floor. So that
    is what is held to the interpreter here -- every module the sources import
    and the table dates -- and the rows nobody imports are left to the six
    interpreters that wrote them.
    """
    running = sys.version_info[:2]
    shipped = sys.stdlib_module_names
    imported = {found.module for path in _sources() for found in _read(path).imports}
    asserted = sorted(imported & set(TABLE["stdlib"]))
    assert asserted, "no imported module is dated, so this row reads nothing"
    wrong = []
    for module in asserted:
        ships = _ships(module)
        assert ships is not None
        since, gone = ships
        expected = since <= running and (gone is None or running < gone)
        span = f"{since[0]}.{since[1]}" + (
            "" if gone is None else f" until {gone[0]}.{gone[1]}"
        )
        if expected and module not in shipped:
            wrong.append(f"{module} is dated {span} and is not shipped here")
        # Above the release the table was derived through, a module may have
        # arrived in one nobody has told it about.
        if not expected and module in shipped and running <= KNOWN_THROUGH:
            wrong.append(f"{module} is dated {span} and is shipped here")
    assert not wrong, (
        f"floor_names.json's stdlib rows disagree with Python "
        f"{running[0]}.{running[1]}:\n" + "\n".join(wrong)
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
