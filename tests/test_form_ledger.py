"""Every form the schema-language pages tabulate is a form a test drives.

Two pages tabulate what a caller may write: `docs/03-schema-language.md` lists
the scalars, the collections, the native forms, the prefix-and-tail shapes, the
class forms, and -- in its own table -- the forms that are **refused** rather
than read; `docs/05-refinements.md` lists the markers. Those tables are the
surface a reader learns the language from, and each row is a promise about an
outcome: this spelling builds and admits these values, or this spelling raises.

Nothing held them. A row could state a reading the frontend does not give, and
the suite would stay green: the tests pick their own forms, and a form nobody
picked is a row nobody checked. The failure is quiet in the direction that
matters -- the page says a shape is read and the build reads it as a literal,
which denotes the form object itself and so admits nothing a caller has.

The universe here is therefore the **tables**, parsed out of the two pages, and
each cell is held to a row that asserts the outcome the cell states:

* a form the tables say is read builds, admits a value written from the row's
  own meaning, and refuses one outside it -- both, because a schema that admits
  everything passes the first half and a schema that admits nothing passes the
  second;
* a form the refusal table names raises at build, with the message that says
  why, so a form quietly read as a literal fails here.

Both directions: a row added to a page with no entry fails, and an entry naming
no row fails. That is what keeps the table and the frontend one description
rather than two.

LEDGER: every form the schema-language pages tabulate is driven by a test

PRODUCT: every form the schema-language pages tabulate
"""

from __future__ import annotations

import enum
import re
import sys
import typing
from dataclasses import dataclass
from pathlib import Path
from types import GenericAlias
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Generic,
    NamedTuple,
    Protocol,
    TypedDict,
    TypeVar,
)

import annotated_types as at
import pytest

from valgebra import Regex, Validator

if TYPE_CHECKING:
    from collections.abc import Callable

ROOT = Path(__file__).resolve().parent.parent
PAGES = (
    ROOT / "docs" / "03-schema-language.md",
    ROOT / "docs" / "05-refinements.md",
)

#: The header a table of forms opens with. A page carries tables about other
#: things -- error codes, entry points -- and these four are the ones whose
#: first column is a spelling a caller writes.
FORM_HEADERS = {"Schema", "Native form", "Form", "Marker"}

_RULE = re.compile(r"^\|[\s:|-]+\|$")


@dataclass(frozen=True)
class Reads:
    """A form the tables say is read, with a value on each side of it.

    Both sides, because one alone is satisfied by the wrong schema: `anything`
    admits the member and `nothing` refuses the outsider, and a row carrying one
    of them would pass against a frontend that had stopped reading the form.
    """

    spec: Any
    member: Any
    outsider: Any


@dataclass(frozen=True)
class Refuses:
    """A form the table says has no set, with what the refusal must say.

    The spec is built behind a callable because several of these are forms a
    checker refuses to have written at all, and because some of them do not
    exist on every interpreter this project supports.

    `needs` is the release the form arrives in. A form the running interpreter
    does not have is not a form a caller on it can write, so the row skips
    rather than asserting a refusal of something that cannot be spelled -- and
    it skips *by version* rather than by catching an `AttributeError`, which
    would also swallow the form being renamed out from under the row.
    """

    build: Callable[[], Any]
    says: str
    needs: tuple[int, int] = (3, 10)


#: What the frontend says of a form that stands for a type rather than being
#: one. Named once, because three spellings share it and a row that copied the
#: sentence would keep passing after the message changed for two of them.
A_TYPING_CONSTRUCT = "is a typing construct, not a value"

T = TypeVar("T")
#: A checker reads a `ParamSpec` only where it is a plain assignment, so it is
#: bound here and named in the row rather than built inside it.
P = typing.ParamSpec("P")


class Parametrised(Generic[T]):
    """A user generic, whose parameter is erased before a value exists."""


class NotRuntimeCheckable(Protocol):
    """A protocol `isinstance` refuses to answer for."""

    x: int


@typing.runtime_checkable
class HasX(Protocol):
    """A protocol `isinstance` does answer for."""

    x: int


class Movie(TypedDict):
    """A `TypedDict`, which is a record over string keys."""

    name: str


@dataclass
class Point:
    """A dataclass, which is a class with declared attributes."""

    x: int


class Pair(NamedTuple):
    """A named tuple, whose fields are its positions."""

    a: int
    b: str


class Colour(enum.Enum):
    """An enum, whose members are the values it admits."""

    RED = "red"


UserId = typing.NewType("UserId", int)

#: A PEP 695 alias, written through `exec` because the syntax is a parse error
#: on the floor this project supports. The alias is the form the table names;
#: what it aliases is an ordinary schema.
_ALIAS: dict[str, Any] = {}
if sys.version_info >= (3, 12):
    exec("type Alias = list[int]", {}, _ALIAS)  # noqa: S102 - a fixed literal


def _tuple_prefix_tail() -> GenericAlias:
    """Build `tuple[A, B, ...]`, which is not a subscription a checker accepts."""
    return GenericAlias(tuple, (int, str, ...))


#: The unpacked spelling of the same tuple, which is a *syntax* error below the
#: release the page marks it with -- so it is compiled at import where the
#: interpreter has it, and the row falls back to the spelling it is the same as
#: where it does not. Written through `exec` for the reason the alias below is.
_UNPACKED: dict[str, Any] = {}
if sys.version_info >= (3, 11):
    exec(  # noqa: S102 - a fixed literal
        "Unpacked = tuple[int, *tuple[str, ...]]", {}, _UNPACKED
    )


#: Every cell of the two pages' form tables, keyed by the cell's own text.
#:
#: The placeholders the pages write -- `T`, `A`, `B`, `K`, `V`, `n`, `p`, `f`,
#: `c`, `X` -- are filled in with kinds that tell each other apart, so a row
#: whose outsider is refused for the wrong reason is a row the table does not
#: describe: `A` is `int` and `B` is `str` throughout, and a value of one is
#: never a value of the other.
FORMS: dict[str, Reads | Refuses] = {
    # -- scalars ------------------------------------------------------------
    "`int`": Reads(int, 1, "a"),
    "`float`": Reads(float, 1.5, 1),
    "`str`": Reads(str, "a", b"a"),
    "`bytes`": Reads(bytes, b"a", "a"),
    "`bool`": Reads(bool, True, 2),
    "`None`": Reads(None, None, 0),
    # -- collections, spelled as typing generics ----------------------------
    "`list[T]`": Reads(list[int], [1], ["a"]),
    "`set[T]`": Reads(set[int], {1}, {"a"}),
    "`frozenset[T]`": Reads(frozenset[int], frozenset({1}), frozenset({"a"})),
    "`dict[K, V]`": Reads(dict[str, int], {"a": 1}, {"a": "b"}),
    "`tuple[A, B]`": Reads(tuple[int, str], (1, "a"), (1, 2)),
    "`tuple[T, ...]`": Reads(tuple[int, ...], (1, 2), ("a",)),
    "`tuple[A, B, ...]`": Reads(_tuple_prefix_tail(), (1, "a", "b"), (1, 2)),
    "`tuple[A, *tuple[B, ...]]`": Reads(
        _UNPACKED.get("Unpacked", _tuple_prefix_tail()), (1, "a"), (1, 2)
    ),
    "`typing.List`, `typing.Tuple`, ...": Reads(typing.List[int], [1], ["a"]),  # noqa: UP006
    # -- the native forms ---------------------------------------------------
    "`[T]`": Reads([int], [1, 2], ["a"]),
    "`[T, ...]`": Reads([int, ...], [1, 2], ["a"]),
    "`[A, B]`": Reads([int, str], [1, "a"], [1, 2]),
    "`[A, B, ...]`": Reads([int, str, ...], [1, "a", "b"], [1, 2]),
    "`{K: V}`": Reads({str: int}, {"a": 1}, {"a": "b"}),
    '`{"key": T, "key2?": T}`': Reads(
        {"key": int, "key2?": int}, {"key": 1}, {"key2": 1}
    ),
    "any constant `c`": Reads(7, 7, 8),
    # -- the prefix-and-tail table's own row --------------------------------
    "`[T, T, ...]`": Reads([int, int, ...], [1], []),
    # -- the class forms ----------------------------------------------------
    "`TypedDict`": Reads(Movie, {"name": "a"}, {"name": 1}),
    "dataclass": Reads(Point, Point(1), "a"),
    "`NamedTuple`": Reads(Pair, Pair(1, "a"), (1, 2)),
    "`Enum`": Reads(Colour, Colour.RED, "red"),
    "runtime-checkable `Protocol`": Reads(HasX, Point(1), "a"),
    "`NewType`": Reads(UserId, 1, "a"),
    "PEP 695 `type` alias": Reads(_ALIAS.get("Alias", list[int]), [1], ["a"]),
    # -- the refinement markers ---------------------------------------------
    "`Ge(n)`": Reads(Annotated[int, at.Ge(2)], 2, 1),
    "`Gt(n)`": Reads(Annotated[int, at.Gt(2)], 3, 2),
    "`Le(n)`": Reads(Annotated[int, at.Le(2)], 2, 3),
    "`Lt(n)`": Reads(Annotated[int, at.Lt(2)], 1, 2),
    "`MinLen(n)`": Reads(Annotated[str, at.MinLen(2)], "ab", "a"),
    "`MaxLen(n)`": Reads(Annotated[str, at.MaxLen(2)], "ab", "abc"),
    "`MultipleOf(n)`": Reads(Annotated[int, at.MultipleOf(2)], 4, 3),
    "`Regex(p)`": Reads(Annotated[str, Regex("a+")], "aa", "b"),
    "`Predicate(f)`": Reads(Annotated[int, at.Predicate(lambda n: n > 0)], 1, 0),
    # -- and the forms that are refused rather than read --------------------
    "`Self`": Refuses(lambda: typing.Self, A_TYPING_CONSTRUCT, needs=(3, 11)),
    "`LiteralString`": Refuses(
        lambda: typing.LiteralString, A_TYPING_CONSTRUCT, needs=(3, 11)
    ),
    "`TypeVar`, `ParamSpec`, `TypeVarTuple`": Refuses(lambda: T, A_TYPING_CONSTRUCT),
    "`Final`, `ClassVar`": Refuses(
        lambda: typing.Final[int], "unsupported typing form with origin"
    ),
    "`Unpack[X]`": Refuses(
        lambda: typing.Unpack[tuple[int, ...]],
        "unsupported typing form with origin",
        needs=(3, 11),
    ),
    "a user `Generic[T]` parametrisation": Refuses(
        lambda: Parametrised[int], "unsupported typing form with origin"
    ),
    "bare `Protocol`, and a `Protocol` without `@runtime_checkable`": Refuses(
        lambda: NotRuntimeCheckable,
        "a Protocol must be @runtime_checkable to be used as a schema",
    ),
    "a set or frozen set literal": Refuses(
        lambda: {int}, "a set literal is not a schema"
    ),
    "a tuple literal": Refuses(lambda: (int, str), "a tuple literal is not a schema"),
}

#: The second spelling of a cell that names two forms, so the row is not held by
#: whichever of the pair the entry above happened to pick.
ALSO: dict[str, list[Reads | Refuses]] = {
    "`typing.List`, `typing.Tuple`, ...": [
        Reads(typing.Tuple[int, str], (1, "a"), (1, 2)),  # noqa: UP006
    ],
    "`TypeVar`, `ParamSpec`, `TypeVarTuple`": [
        Refuses(lambda: P, A_TYPING_CONSTRUCT),
    ],
    "`Final`, `ClassVar`": [
        Refuses(lambda: typing.ClassVar[int], "unsupported typing form with origin"),
    ],
    "bare `Protocol`, and a `Protocol` without `@runtime_checkable`": [
        Refuses(
            lambda: Protocol,
            "a Protocol must be @runtime_checkable to be used as a schema",
        ),
    ],
    "a set or frozen set literal": [
        Refuses(lambda: frozenset({int}), "a frozen set literal is not a schema"),
    ],
}
if sys.version_info >= (3, 11):
    # The third of that cell's three forms arrives with the release that can
    # spell an unpacked tuple, so it is asked where it exists and the row above
    # carries the two the floor has.
    Ts = typing.TypeVarTuple("Ts")
    ALSO["`TypeVar`, `ParamSpec`, `TypeVarTuple`"].append(
        Refuses(lambda: Ts, A_TYPING_CONSTRUCT, needs=(3, 11))
    )


#: The cells that name more than one form, so the row above holds one of them
#: and `ALSO` holds the rest.
NAMES_MORE_THAN_ONE_FORM = {
    "`typing.List`, `typing.Tuple`, ...",
    "`TypeVar`, `ParamSpec`, `TypeVarTuple`",
    "`Final`, `ClassVar`",
    "bare `Protocol`, and a `Protocol` without `@runtime_checkable`",
    "a set or frozen set literal",
}


def _tables() -> list[tuple[Path, list[str], list[list[str]]]]:
    """Read every form table out of the two pages, header and rows."""
    found: list[tuple[Path, list[str], list[list[str]]]] = []
    for page in PAGES:
        lines = page.read_text(encoding="utf-8").splitlines()
        at = 0
        while at < len(lines) - 1:
            head, rule = lines[at], lines[at + 1]
            if not (head.startswith("|") and _RULE.match(rule)):
                at += 1
                continue
            cells = [cell.strip() for cell in head.strip("|").split("|")]
            at += 2
            rows = []
            while at < len(lines) and lines[at].startswith("|"):
                rows.append([cell.strip() for cell in lines[at].strip("|").split("|")])
                at += 1
            if cells[0] in FORM_HEADERS:
                found.append((page, cells, rows))
    return found


def _cells() -> list[str]:
    """Give the first column of every form table: the spellings a caller writes."""
    return [row[0] for _, _, rows in _tables() for row in rows]


def test_the_universe_is_the_pages_own_tables() -> None:
    """The parse is the detector, so it is shown to have read the tables.

    A ledger over an empty universe passes having checked nothing, which is the
    failure `tests/test_ledger_plants.py` exists to rule out for the rest.
    """
    tables = _tables()
    assert len(tables) >= 6, [cells for _, cells, _ in tables]
    assert len(_cells()) >= 45, _cells()
    # And it reads both pages, not one of them twice.
    assert {page.name for page, _, _ in tables} == {
        "03-schema-language.md",
        "05-refinements.md",
    }


def test_every_tabulated_form_has_a_row_and_every_row_a_form() -> None:
    """Both directions, which is what keeps the page and the frontend one thing."""
    cells = set(_cells())
    missing = sorted(cells - set(FORMS))
    assert not missing, (
        "forms the pages tabulate that no row drives:\n"
        + "\n".join(f"  {cell}" for cell in missing)
        + "\n\nGive each a `Reads` with a member and an outsider, or a "
        "`Refuses` with what the refusal says."
    )
    stale = sorted(set(FORMS) - cells)
    assert not stale, f"rows naming a form no table lists: {stale}"
    unknown = sorted(set(ALSO) - cells)
    assert not unknown, f"second spellings for a cell no table lists: {unknown}"

    # A cell naming more than one form is held by more than one row, and the
    # list is the identity rather than a floor: a cell that grows a second
    # spelling arrives here without one, and an entry whose list is added after
    # the rows are collected contributes nothing and reads as if it did.
    assert set(ALSO) == NAMES_MORE_THAN_ONE_FORM
    for cell, rows in ALSO.items():
        assert rows, f"{cell}: a second spelling with no row"
    assert len(_rows()) == len(FORMS) + sum(len(rows) for rows in ALSO.values())


def _rows() -> list[tuple[str, Reads | Refuses]]:
    """Each cell's row, and the second spelling of a cell that names two."""
    rows = [(cell, FORMS[cell]) for cell in sorted(FORMS)]
    rows += [(cell, row) for cell in sorted(ALSO) for row in ALSO[cell]]
    return rows


def _reading() -> list[tuple[str, Reads]]:
    return [(cell, row) for cell, row in _rows() if isinstance(row, Reads)]


def _refusing() -> list[tuple[str, Refuses]]:
    return [(cell, row) for cell, row in _rows() if isinstance(row, Refuses)]


@pytest.mark.parametrize(("cell", "row"), _reading(), ids=[c for c, _ in _reading()])
def test_a_form_the_tables_read_admits_and_refuses(cell: str, row: Reads) -> None:
    """A form the page says is read admits its member and refuses its outsider."""
    compiled = Validator(row.spec)
    assert compiled.is_valid(row.member), f"{cell}: {row.member!r} is not admitted"
    assert not compiled.is_valid(row.outsider), f"{cell}: {row.outsider!r} is admitted"


@pytest.mark.parametrize(("cell", "row"), _refusing(), ids=[c for c, _ in _refusing()])
def test_a_form_the_table_refuses_says_so_at_build(cell: str, row: Refuses) -> None:
    """A form with no set raises at build rather than being read as a literal.

    Read as a literal it would denote the form object itself, which no value a
    caller holds belongs to -- a schema refusing everything and saying nothing
    about why, which is the failure the refusal table exists to describe.
    """
    if sys.version_info < row.needs:
        spelled = ".".join(str(part) for part in row.needs)
        pytest.skip(
            f"{cell} is not a form this interpreter has; it arrives in {spelled}"
        )
    with pytest.raises(NotImplementedError, match=re.escape(row.says)):
        Validator(row.build())


def test_a_read_form_is_not_a_literal_of_itself() -> None:
    """The quiet failure, named: a form read as a literal admits only itself.

    Held over every reading row rather than at one, because that is the shape
    the fallback produces for *any* form it stops recognising -- and a row
    asserting a member and an outsider would not see it if the member happened
    to be the form object.
    """
    for cell, row in _reading():
        compiled = Validator(row.spec)
        assert not compiled.is_valid(row.spec) or row.spec == row.member, (
            f"{cell}: the schema admits the annotation itself, which is what a "
            "form read as a literal does"
        )
