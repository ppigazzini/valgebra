"""Every constraint is put to every kind, and either narrows it or is refused.

`docs/05-refinements.md` states the rule in one sentence: "A constraint must
also be one the base can answer [...] A constraint no value of the base can
answer is therefore refused too". That sentence is about a **product** -- the
markers the page lists, by the kinds a base's values have -- and a sentence
about a product held by the pairs somebody thought of is held for the pairs
somebody thought of.

The failure it misses is quiet and is the one the rule exists to stop: a
constraint the base cannot be asked builds a schema that admits nothing while
reporting itself inhabited, so a caller gets a validator that rejects every
value and a `repr` that reads like a narrowing.

So the universe is the two enumerations, read from the tree: `Constraint` in
the IR, and `Kind`, which is the one partition both deciders use. Each cell is
one of two things, and both are outcomes rather than calls:

* the base **can** be asked the constraint, so the schema builds, admits a
  value the constraint holds of and refuses one it does not -- both, because a
  schema admitting everything passes the first half and one admitting nothing
  passes the second;
* the base **cannot**, so the build raises, with the sentence naming what the
  base has no way to answer.

Both directions: a cell with no entry fails, and an entry naming no cell of the
product fails. That is what keeps the table and the frontend one description of
the same rule rather than two.

LEDGER: every constraint is driven against every kind, narrowing it or refused

PRODUCT: every constraint, against every kind a base's values have
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, get_args

import annotated_types as at
import pytest

from valgebra import Regex, ValidationError, Validator

ROOT = Path(__file__).resolve().parent.parent
IR = ROOT / "crates" / "valgebra-core" / "src" / "ir.rs"
KIND = ROOT / "crates" / "valgebra-core" / "src" / "kind.rs"

_CONSTRAINTS = re.compile(
    r"^pub enum Constraint \{$(.*?)^\}$", re.DOTALL | re.MULTILINE
)
_KINDS = re.compile(r"^pub enum Kind \{$(.*?)^\}$", re.DOTALL | re.MULTILINE)
_VARIANT = re.compile(r"^    ([A-Z][A-Za-z]*)[ ({,]", re.MULTILINE)


@dataclass(frozen=True)
class Narrows:
    """A cell where the base answers the constraint, so the schema narrows it.

    Both a member and an outsider, because either alone is satisfied by the
    wrong schema: `anything` admits the member and `nothing` refuses the
    outsider, and a cell carrying one of them would pass against a frontend
    that had stopped reading the constraint.
    """

    spec: Any
    member: Any
    outsider: Any


@dataclass(frozen=True)
class Refused:
    """A cell where no value of the base answers the constraint.

    `says` is what the base has no way to answer -- the words the refusal names
    -- rather than the whole sentence, so a message rewritten around them stays
    matched and one that stops naming the reason does not.
    """

    says: str


#: What each refusal names. Written once, because a cell that copied the words
#: would keep passing after the message changed for every other cell.
_LENGTH = "length"
_TEXT = "text for a pattern to match"
_DIVISOR = "number for a divisor"
_ORDER = "order against that bound"


def _length_cells(cells: dict[tuple[str, str], Narrows | Refused]) -> None:
    """Fill the length markers: the sized kinds answer and the scalars do not."""
    # --- a length: the sized kinds answer, the scalars do not ----------------
    sized: dict[str, tuple[Any, Any, Any]] = {
        "Str": (str, "ab", ""),
        "Bytes": (bytes, b"ab", b""),
        "List": (list[int], [1, 2], []),
        "Tuple": (tuple[int, ...], (1, 2), ()),
        "Set": (set[int], {1, 2}, set()),
        "FrozenSet": (frozenset[int], frozenset({1, 2}), frozenset()),
        "Dict": (dict[str, int], {"a": 1, "b": 2}, {}),
    }
    for kind, (base, member, outsider) in sized.items():
        cells[("MinLen", kind)] = Narrows(
            Annotated[base, at.MinLen(1)], member, outsider
        )
        cells[("MaxLen", kind)] = Narrows(
            Annotated[base, at.MaxLen(1)], outsider, member
        )
    for kind in ("NoneType", "Bool", "Int", "Float"):
        cells[("MinLen", kind)] = Refused(_LENGTH)
        cells[("MaxLen", kind)] = Refused(_LENGTH)


def _pattern_and_divisor_cells(
    cells: dict[tuple[str, str], Narrows | Refused],
) -> None:
    """Fill the markers that ask for text and for a number."""
    # --- a pattern: the text kind answers ------------------------------------
    cells[("Regex", "Str")] = Narrows(Annotated[str, Regex("a+")], "aa", "b")
    for kind in _KIND_BASES:
        if kind != "Str":
            cells[("Regex", kind)] = Refused(_TEXT)

    # --- a divisor: the numeric kinds answer ---------------------------------
    cells[("MultipleOf", "Int")] = Narrows(Annotated[int, at.MultipleOf(2)], 4, 3)
    cells[("MultipleOf", "Float")] = Narrows(
        Annotated[float, at.MultipleOf(0.5)], 1.5, 1.3
    )
    # Every boolean is a multiple of one and `False` is a multiple of two, so
    # the divisor that separates them is the one no boolean but `False` meets.
    cells[("MultipleOf", "Bool")] = Narrows(
        Annotated[bool, at.MultipleOf(2)], False, True
    )
    for kind in _KIND_BASES:
        if kind not in {"Int", "Float", "Bool"}:
            cells[("MultipleOf", kind)] = Refused(_DIVISOR)


def _order_cells(cells: dict[tuple[str, str], Narrows | Refused]) -> None:
    """Fill the bounds, whose answer is about the base *and* the bound."""
    # --- an order: every kind whose values compare with one another answers,
    #     and the two that compare with nothing do not ------------------------
    #
    # The bound travels with the kind. A bound is a value, and `>=` is a
    # question about *two* values -- so the cell asks the base about a bound of
    # its own kind, and the bound of another kind is a claim of its own below.
    # Two ordered values per kind is all four markers need: the inclusive form
    # takes the value it names and the strict form does not, so `lo` and `hi`
    # separate every pair without a third point -- which `bool`, having two
    # values, does not have.
    ordered: dict[str, tuple[Any, Any, Any]] = {
        "Bool": (bool, False, True),
        "Int": (int, 0, 1),
        "Float": (float, 0.5, 1.5),
        "Str": (str, "a", "b"),
        "Bytes": (bytes, b"a", b"b"),
        "List": (list[int], [0], [1]),
        "Tuple": (tuple[int, ...], (0,), (1,)),
        # Sets order by inclusion rather than by size: `>=` is "is a superset
        # of", so the two values are a set and a superset of it.
        "Set": (set[int], {1}, {1, 2}),
        "FrozenSet": (frozenset[int], frozenset({1}), frozenset({1, 2})),
    }
    for kind, (base, lo, hi) in ordered.items():
        cells[("Ge", kind)] = Narrows(Annotated[base, at.Ge(hi)], hi, lo)
        cells[("Gt", kind)] = Narrows(Annotated[base, at.Gt(lo)], hi, lo)
        cells[("Le", kind)] = Narrows(Annotated[base, at.Le(lo)], lo, hi)
        cells[("Lt", kind)] = Narrows(Annotated[base, at.Lt(hi)], lo, hi)
    # Nothing orders against `None`, and a dict has no order at all: both raise
    # rather than answering, so a bound over either names a set no value
    # belongs to.
    for marker in ("Ge", "Gt", "Le", "Lt"):
        for kind in ("NoneType", "Dict"):
            cells[(marker, kind)] = Refused(_ORDER)


def _kind_cells() -> dict[tuple[str, str], Narrows | Refused]:
    """Give the product, one cell at a time.

    Written out rather than derived from a rule: a rule here would be the
    frontend's rule restated, and a ledger that restates what it checks cannot
    fail. The shape repeats because the *answer* repeats -- seven kinds have a
    length and four do not -- and that repetition is the claim.
    """
    cells: dict[tuple[str, str], Narrows | Refused] = {}
    _length_cells(cells)
    _pattern_and_divisor_cells(cells)
    _order_cells(cells)
    # --- a predicate: opaque, so every kind takes one ------------------------
    for kind, (base, member, outsider) in _KIND_BASES.items():
        cells[("Predicate", kind)] = Narrows(
            Annotated[base, at.Predicate(lambda value, keep=member: value == keep)],
            member,
            outsider,
        )
    return cells


#: One base per kind, with a value of it and a value outside it. The bases are
#: the forms a caller writes for that kind, so a cell drives the schema a reader
#: would have written rather than one built for the test.
_KIND_BASES: dict[str, tuple[Any, Any, Any]] = {
    "NoneType": (None, None, 0),
    "Bool": (bool, True, 1),
    "Int": (int, 1, "a"),
    "Float": (float, 1.5, 1),
    "Str": (str, "a", b"a"),
    "Bytes": (bytes, b"a", "a"),
    "List": (list[int], [1], (1,)),
    "Tuple": (tuple[int, ...], (1,), [1]),
    "Set": (set[int], {1}, frozenset({1})),
    "FrozenSet": (frozenset[int], frozenset({1}), {1}),
    "Dict": (dict[str, int], {"a": 1}, []),
}

CELLS = _kind_cells()


def _variants(path: Path, enumeration: re.Pattern[str], least: int) -> set[str]:
    """Read an enum's variant names out of the tree rather than restating them."""
    body = enumeration.search(path.read_text(encoding="utf-8"))
    assert body, f"{path.name} has no enum this ledger reads"
    found = set(_VARIANT.findall(body.group(1)))
    # The scan is the detector: an empty set would pass both directions having
    # read nothing at all.
    assert len(found) >= least, f"the scan found only {sorted(found)}"
    return found


def _product() -> list[tuple[str, str]]:
    return [
        (constraint, kind)
        for constraint in sorted(_variants(IR, _CONSTRAINTS, 9))
        for kind in sorted(_variants(KIND, _KINDS, 11))
    ]


def test_the_universe_is_the_two_enums_in_both_directions() -> None:
    """A constraint or a kind added to the tree arrives here without a cell."""
    product = set(_product())
    missing = sorted(product - set(CELLS))
    assert not missing, (
        f"cells of the product with no entry: {missing}. Give each one the value "
        "the constraint holds of and one it does not, or the words the refusal "
        "names."
    )
    stale = sorted(set(CELLS) - product)
    assert not stale, f"entries naming no cell of the product: {stale}"


def test_every_kind_has_a_base_a_caller_writes() -> None:
    """The bases are the kinds, so a kind added to the partition arrives here."""
    kinds = _variants(KIND, _KINDS, 11)
    assert set(_KIND_BASES) == kinds, sorted(kinds ^ set(_KIND_BASES))
    for kind, (base, member, outsider) in _KIND_BASES.items():
        compiled = Validator(base)
        assert compiled.is_valid(member), f"{kind}: the member is not one"
        assert not compiled.is_valid(outsider), f"{kind}: the outsider is one"


# THEORY: a-refinement-narrows-its-base
@pytest.mark.parametrize(
    ("constraint", "kind"), _product(), ids=[f"{c}:{k}" for c, k in _product()]
)
def test_every_cell_narrows_its_base_or_is_refused(constraint: str, kind: str) -> None:
    """The outcome the cell states, driven rather than described."""
    cell = CELLS[(constraint, kind)]
    if isinstance(cell, Refused):
        with pytest.raises(NotImplementedError, match=re.escape(cell.says)):
            Validator(
                cell.spec if hasattr(cell, "spec") else _refused_spec(constraint, kind)
            )
        return
    compiled = Validator(cell.spec)
    assert compiled.is_valid(cell.member), (
        f"{constraint} over {kind} refuses a value the constraint holds of"
    )
    assert not compiled.is_valid(cell.outsider), (
        f"{constraint} over {kind} admits a value the constraint does not hold of"
    )
    # And the refusal is the constraint's own code at the root, on the object
    # path and, where a document names the outsider, on the parsed one: a cell
    # refused at a nested path, or by the base's kind where the outsider is a
    # value of the base, would be a different failure wearing the right
    # verdict. The predicate cells' outsiders lie outside the base itself, so
    # there the base's kind is what refuses and the code says so.
    base = get_args(cell.spec)[0]
    # A refinement is its base narrowed: below the base whatever the
    # constraint says, and the base is not below it, since that would need
    # the constraint to hold of every value of the base.
    assert compiled.relation_to(base) == "subset", f"{constraint} over {kind}"
    assert Validator(base).relation_to(compiled) != "subset", (
        f"{constraint} over {kind} is decided equal to its base"
    )
    with pytest.raises(ValidationError) as caught:
        compiled.validate(cell.outsider, fail_fast=True)
    assert caught.value.path == ()
    if Validator(base).is_valid(cell.outsider):
        assert caught.value.code == _CODE_OF[constraint], caught.value.errors
    else:
        assert caught.value.code != _CODE_OF[constraint], caught.value.errors
    document = _document_naming(cell.outsider)
    if document is not None:
        with pytest.raises(ValidationError) as parsed:
            compiled.validate_json(document, fail_fast=True)
        assert parsed.value.code == caught.value.code, parsed.value.errors
        assert parsed.value.path == ()


#: The code each constraint reports when a value of the base fails it.
_CODE_OF: dict[str, str] = {
    "MinLen": "too_short",
    "MaxLen": "too_long",
    "Regex": "string_pattern_mismatch",
    "MultipleOf": "multiple_of",
    "Ge": "greater_than_equal",
    "Gt": "greater_than",
    "Le": "less_than_equal",
    "Lt": "less_than",
    "Predicate": "predicate_failed",
}


def _document_naming(value: Any) -> str | None:
    """Give the JSON document that names `value`, where one does.

    A tuple, a set, a frozenset and a bytes value have no document that reads
    back as themselves, so a cell over those kinds is asked on the object path
    alone. A float that is integral reads back as an `int` and is not named
    either.
    """
    try:
        text = json.dumps(value)
    except TypeError:
        return None
    back = json.loads(text)
    if type(back) is not type(value) or back != value:
        return None
    return text


#: The spec a refused cell is written as. A refusal has no schema to carry, so
#: the form is built here from the base and the marker rather than stored: the
#: cell records *why* the build fails, and this records what was asked.
_REFUSED_MARKERS: dict[str, Any] = {
    "MinLen": at.MinLen(1),
    "MaxLen": at.MaxLen(1),
    "Regex": Regex("a+"),
    "MultipleOf": at.MultipleOf(2),
    "Ge": at.Ge(0),
    "Gt": at.Gt(0),
    "Le": at.Le(0),
    "Lt": at.Lt(0),
}


def _refused_spec(constraint: str, kind: str) -> Any:
    """Give the form a refused cell asks: the kind's base under the marker.

    The two halves are read out of their tables first, because a subscript in a
    type expression takes a name rather than an indexing -- and a checker that
    cannot read the form is one that cannot tell this file is consistent.
    """
    base: Any = _KIND_BASES[kind][0]
    marker: Any = _REFUSED_MARKERS[constraint]
    return Annotated[base, marker]


#: A base whose values are ordered, and a bound of a kind they do not compare
#: with. The base is not the question here -- each of these has an order of its
#: own -- so a rule reading the base alone admits every row.
_MISMATCHED = [
    pytest.param(str, at.Ge(0), id="text-against-a-number"),
    pytest.param(bytes, at.Ge("a"), id="bytes-against-text"),
    pytest.param(set[int], at.Ge(0), id="a-set-against-a-number"),
    pytest.param(frozenset[int], at.Le(0), id="a-frozen-set-against-a-number"),
    pytest.param(list[int], at.Gt(0), id="a-list-against-a-number"),
    pytest.param(tuple[int, ...], at.Lt(0), id="a-tuple-against-a-number"),
    pytest.param(list[int], at.Ge((0,)), id="a-list-against-a-tuple"),
    pytest.param(tuple[int, ...], at.Ge([0]), id="a-tuple-against-a-list"),
    pytest.param(int, at.Ge("a"), id="a-number-against-text"),
    pytest.param(set[int], at.Ge([1]), id="a-set-against-a-list"),
]


@pytest.mark.parametrize(("base", "marker"), _MISMATCHED)
def test_a_bound_of_another_kind_is_refused_however_ordered_the_base_is(
    base: Any, marker: Any
) -> None:
    """`>=` is a question about two values, so both kinds decide the answer.

    The product above asks each base about a bound of its own kind, which is
    the cell a caller writes. This is the other half: a base with an order of
    its own, and a bound its values raise on. Reading the base alone admits
    every row here, and every row names a schema that admits no value.
    """
    with pytest.raises(NotImplementedError, match=re.escape(_ORDER)):
        Validator(Annotated[base, marker])


def test_no_constraint_builds_a_schema_that_admits_nothing_and_says_otherwise() -> None:
    """The stop rule, asked of the whole product at once.

    A constraint the base cannot answer compiles, under the rule this ledger
    holds, to *nothing at all*: the walk asks the value a question it raises on
    and reads the raise as a non-member. A schema like that is only findable by
    a caller who tries every value, so the ledger asks the library instead --
    and a cell that builds must either admit something or say it is empty.
    """
    silent = []
    for (constraint, kind), cell in sorted(CELLS.items()):
        if isinstance(cell, Refused):
            continue
        compiled = Validator(cell.spec)
        if not compiled.is_valid(cell.member) and not compiled.is_empty():
            silent.append(f"{constraint} over {kind}: {compiled!r}")
    assert not silent, (
        "schemas that admit no value and do not report themselves empty:\n"
        + "\n".join(silent)
    )
