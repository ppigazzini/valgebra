"""Completeness ledger for the decision procedure.

The decision procedure is sound everywhere and complete on part of its domain.
This module measures the incompleteness directly, rather than letting it hide
behind a soundness-only suite.

Two mechanisms:

- A parametrized ledger of relations that hold by construction. Each case
  asserts the procedure decides a true relation. Cases the procedure decides are
  gates: a regression that makes one conservative fails the build. Cases it does
  not decide are marked ``xfail(strict=True)``, so closing one fails the strict
  mark and forces its entry out of the ledger and off the boundary the
  decidability page publishes. A new true relation the procedure declines, added
  here without a mark, fails outright.
- A finite-universe soundness fuzzer. For random schema pairs, a claimed subtype
  never accepts a value the supertype rejects across a region-complete universe.

LEDGER: every relation the procedure must decide is decided, and every relation
it declines is a strict expected failure that fails the day it is decided.

Every relation is written the way a caller writes it. A case built from the other
operand -- a wider union built by adding a member to the narrower one, rather
than written out -- shares its constants with it, and the shortcuts the procedure
takes are keyed on exactly that, so such a case measures the spelling instead of
the relation.
"""

import dataclasses
import enum
import itertools
from typing import (
    Annotated,
    Any,
    ClassVar,
    Final,
    Literal,
    NamedTuple,
    NoReturn,
    Optional,
    TypedDict,
    TypeVar,
    Union,
)

import annotated_types as at
import pytest
from hypothesis import given
from hypothesis import strategies as st

from valgebra import (
    Regex,
    Validator,
    anything,
    complement,
    intersection,
    nothing,
    recursive,
    union,
)

# A recursive schema, reused below to record a reflexivity hole.
_RECURSIVE = recursive(lambda t: union(None, {"value": int, "next": t}))

# The chain a meet with a reference is asked about.
_CHAIN = recursive(lambda t: union(None, {"next": t}))

# A fixpoint whose body carries a meet against a union beside the recursive
# member: the shape whose unfolding, as a union, the deciders decline to place
# below it, while each member alone is decided.
_MEET_MEMBER_FIXPOINT = recursive(
    lambda t: union(intersection(int, union(float, bool)), list[t])  # ty: ignore[invalid-type-form]
)

# A linked chain, and the same chain one optional field wider. The two are two
# sets -- the wider admits a link carrying `v` -- and the value that says so
# sits at the first link rather than past an unfolding.
_CHAIN = recursive(lambda t: union(None, {"next": t}))
_CHAIN_WITH_A_FIELD = recursive(lambda t: union(None, {"next": t, "v?": int}))
# The chain written with the reference inside the key's union, as a record
# that names itself is spelled, and a chain of one key the meets below ask of.
_LINK = recursive(lambda t: {"next": union(t, None)})
_NESTED = recursive(lambda t: union(None, {"a": t}))
# A counted Unicode class under a repetition, past the size an automaton of
# the set representation is built to.
_COUNTED_CLASS = Annotated[str, Regex(r"(?:\w{3,4})+")]

# The integers and the lists of them, nested to any depth.
_NESTED_INTS = recursive(lambda t: union(int, [t]))


class _Color(enum.Enum):
    """An enumeration whose members compare by identity, which is the default."""

    RED = 1
    GREEN = 2


# The JSON value, which is the recursive schema a reader writes: every scalar,
# and lists and dicts of itself. Its body names six kinds and no other, which is
# what makes a meet with a seventh decidable once the body is unfolded.
_JSON = recursive(lambda j: union(None, bool, int, float, str, [j], {str: j}))

# Two refinements the rules cannot read, since each runs a predicate: a depth
# into either declines, from `int` or from the other.
_POSITIVE = Annotated[int, at.Predicate(lambda value: value > 0)]
_SMALL = Annotated[int, at.Predicate(lambda value: value < 5)]


class _Pair(NamedTuple):
    """A named tuple, whose instances are two-tuples of ints."""

    left: int
    right: int


class _MyInt(int):
    """An `int` subclass: every instance is an int, and 5 is not one of them."""


class _MyStr(str):
    """A `str` subclass, for the same relation over a different kind."""

    __slots__ = ()


class _Plain:
    """A class laying down no layout, whose subclasses may lay down any."""


class _Slotted:
    """A class whose `__slots__` lay down a layout over no builtin."""

    __slots__ = ("field",)


class _OtherSlotted:
    """A second layout over no builtin, which no class can carry beside the first."""

    __slots__ = ("other",)


# A region-complete value universe whose numbers straddle the bounds the ledger
# uses, so a subset relation over it reflects the true relation on those cases.
_UNIVERSE = [
    None,
    True,
    False,
    -5,
    -1,
    0,
    1,
    5,
    10,
    11,
    100,
    "",
    "a",
    "ab",
    b"",
    b"x",
    -1.5,
    1.5,
    [],
    [1],
    [1, 2],
    ["a"],
    (1,),
    (1, 2),
    {1},
    frozenset({1}),
    {"k": 1},
    {5: True},
    {"k": 1, 5: True},
    {"a": 1},
    {"a": 1, "k": "x"},
    {"a": 1, "b": "y"},
    {"a": 1, "b": "y", "k": "z"},
    # Instances of the two builtin subclasses, which is what witnesses a class
    # placed on its kind: `_MyInt(3)` is an int and is not the integer 3.
    _MyInt(3),
    _MyStr("a"),
    # An instance of a class laying down its own layout: a value of no builtin
    # kind, which is what witnesses a meet with a kind decided empty.
    _Slotted(),
]


def _accepted(spec: object) -> frozenset[int]:
    """Return the universe indices a schema accepts -- its denotation on `U`."""
    compiled = Validator(spec)
    return frozenset(i for i, value in enumerate(_UNIVERSE) if compiled.is_valid(value))


# --- The ledger: relations that hold by construction -------------------------
#
# Each tuple is (operation, left, right). For "subtype" and "equivalent" the
# right side is the comparison schema; for "empty" it is None. Every relation
# listed is true set-theoretically.

_GE0 = Annotated[int, at.Ge(0)]
_GE0_LE10 = Annotated[int, at.Ge(0), at.Le(10)]
_GE10_LE0 = Annotated[int, at.Ge(10), at.Le(0)]
_GT0 = Annotated[int, at.Gt(0)]
_LT1 = Annotated[int, at.Lt(1)]


@dataclasses.dataclass
class _Animal:
    name: str


@dataclasses.dataclass
class _Dog(_Animal):
    breed: str


def _check(operation: str, left: object, right: object) -> None:
    compiled = Validator(left)
    if operation == "subtype":
        assert compiled.is_subtype_of(right)
    elif operation == "refutes":
        # A refutation is an answer a caller acts on as much as a proof is:
        # `not_subset` names a value of the subject the supertype refuses, and
        # `undecided` says the procedure did not look. A ledger that only ever
        # asserted proofs would let every refutation decay into a decline
        # without a row failing.
        assert compiled.relation_to(right) == "not_subset"
    elif operation == "equivalent":
        assert compiled.is_equivalent(right)
    elif operation == "empty":
        assert compiled.is_empty()
    else:  # pragma: no cover - guards against a typo in a case
        msg = f"unknown operation {operation!r}"
        raise AssertionError(msg)


def _split(width: int) -> tuple[object, object]:
    """Build a tuple of `width` components each `int | str`, and its corners.

    The union of the `2 ** width` tuples fixing every component *is* the tuple,
    and no single one contains it, so the product rule splits the subject across
    them -- one choice per component, multiplied.
    """
    subject = tuple[tuple([union(int, str)] * width)]  # ty: ignore[invalid-type-form]
    corners = union(
        *[
            tuple[choice]  # ty: ignore[invalid-type-form]
            for choice in itertools.product((int, str), repeat=width)
        ]
    )
    return subject, corners


def _nested(leaf: object, depth: int, *, mapping: bool) -> object:
    """Nest `leaf` `depth` containers deep: `dict[str, dict[str, int]]` at two."""
    for _ in range(depth):
        leaf = dict[str, leaf] if mapping else list[leaf]  # ty: ignore[invalid-type-form]
    return leaf


def _literal_keyed(leaf: object, depth: int) -> object:
    """Nest `leaf` in `depth` mappings keyed by the literal `"a"`."""
    for _ in range(depth):
        leaf = dict[Literal["a"], leaf]  # ty: ignore[invalid-type-form]
    return leaf


def _record_split(width: int) -> tuple[object, object]:
    """Build a record of `width` fields each `int | str`, and its corners.

    The record's reading of `_split`: the `2 ** width` records fixing every
    field *are* the record, no one of them contains it, and the record rule
    splits it across them a field at a time.
    """
    names = [f"f{i}" for i in range(width)]
    subject = dict.fromkeys(names, union(int, str))
    corners = union(
        *[
            dict(zip(names, choice, strict=True))
            for choice in itertools.product((int, str), repeat=width)
        ]
    )
    return subject, corners


def _chain(width: int) -> object:
    """Build an open record whose key takes `width` constants, minus each one.

    Every constant is taken away, so the chain is empty. Each link adds five
    schema nodes to what the lowering reads -- a constant in the key and a
    record complemented -- which is what puts the width against the node budget.
    """
    return intersection(
        {"a": union(*range(width)), str: Any},
        *[complement({"a": i, str: Any}) for i in range(width)],
    )


_DECIDED = [
    pytest.param("subtype", bool, int, id="bool<=int"),
    # A key the supertype requires and the subject leaves optional refutes on
    # `{}`, whatever the field's types say of each other, so a depth the rules
    # decline beside it hides nothing.
    pytest.param("refutes", {"a?": _POSITIVE}, {"a": _SMALL}, id="{a?:P}!<={a:Q}"),
    pytest.param("refutes", {"a?": int}, {"a": _SMALL}, id="{a?:int}!<={a:Q}"),
    # A tail outside the supertype's tail refutes beside a prefix that declines:
    # `[1, 2]` is in the subject and outside the supertype.
    pytest.param(
        "refutes",
        [int, int, ...],
        [_SMALL, str, ...],
        id="[int,int,...]!<=[Q,str,...]",
    ),
    # The product rule's width: six components against their sixty-four
    # corners is decided, which is the number the decidability page gives.
    pytest.param("subtype", *_split(6), id="tuple:six-components<=its-corners"),
    pytest.param("subtype", 1, int, id="Literal[1]<=int"),
    pytest.param("subtype", int, union(int, str), id="int<=int|str"),
    pytest.param("subtype", list[bool], list[int], id="list[bool]<=list[int]"),
    pytest.param("subtype", {"x": int}, {"x": int, "y?": str}, id="{x}<={x,y?}"),
    # Two clauses of the supertype cover one of the subject's together: the
    # rule reads them one at a time and declines, and the sets decide.
    pytest.param(
        "subtype",
        dict[int | str, int],
        {int: int, str: int},
        id="{int|str:int}<={int,str}",
    ),
    pytest.param(
        "subtype", [bool, int, ...], [int, int, ...], id="[bool,int,...]<=[int,int,...]"
    ),
    pytest.param("empty", intersection(int, str), None, id="empty:int&str"),
    # A fixpoint's unfolding is below the fixpoint -- the equirecursive
    # reading -- on a body whose union holds a member that is a meet against
    # a union. The meet rule finds no member of the meet below the reference,
    # and the meet as a whole is a branch of the definition, which the
    # reference's unfolding reads.
    pytest.param(
        "subtype",
        union(intersection(int, union(float, bool)), list[_MEET_MEMBER_FIXPOINT]),  # ty: ignore[invalid-type-form]
        _MEET_MEMBER_FIXPOINT,
        id="recursion:unfolding-with-a-meet-member<=its-fixpoint",
    ),
    # The empty list is a value of every list shape with no prefix, so of a
    # meet of them, whatever the element reads as while it is resolved.
    pytest.param(
        "refutes",
        intersection([], _NESTED_INTS),
        nothing,
        id="recursion:[]&RI-is-inhabited",
    ),
    pytest.param(
        "refutes",
        recursive(lambda t: intersection([], [t])),
        nothing,
        id="recursion:fix(t:[]&[t])-is-inhabited",
    ),
    # A fixpoint every unfolding of which needs one more element: each element
    # is a value of the same fixpoint, so no finite value satisfies it. Decided
    # by reading a length bound over a container that repeats one element --
    # the element is the reference, and a reference reached again while it is
    # being resolved has no finite value.
    pytest.param(
        "empty",
        recursive(lambda t: Annotated[list[t], at.MinLen(1)]),  # ty: ignore[invalid-type-form]
        None,
        id="empty:mu-t.list[t]&MinLen(1)",
    ),
    # `Any` is the top, spelled: the same node as `anything`, so every relation
    # the top decides it decides too. The spelling is not a set, and no rule can
    # read it.
    pytest.param("equivalent", Any, anything, id="Any==anything"),
    pytest.param("subtype", int, Any, id="int<=Any"),
    pytest.param("subtype", list[int], list[Any], id="list[int]<=list[Any]"),
    pytest.param(
        "empty", intersection(Any, complement(Any)), None, id="empty:Any&~Any"
    ),
    pytest.param("empty", complement(Any), None, id="empty:~Any"),
    # A complemented union is read with the complement pushed in, as the relation
    # reads it below `nothing`: `str & json & ~str` folds to the empty set.
    pytest.param(
        "empty",
        complement(
            union(complement(str), complement(intersection(_JSON, complement(str))))
        ),
        None,
        id="empty:~(~str|~(json&~str))",
    ),
    # Three keywords on one kind, each guarded to it, read as the meet of what
    # they guard: eight clauses distributed, one meet narrowed.
    pytest.param(
        "empty",
        intersection(
            {str: _JSON},
            union(complement({str: _JSON}), {"tag": int, str: _JSON}),
            union(complement({str: _JSON}), {"tag": str, str: _JSON}),
            union(complement({str: _JSON}), {"tag": _JSON, str: _JSON}),
        ),
        None,
        id="empty:three-guards-tagged-two-ways",
    ),
    # A float bound over the integers is read at its floor or its ceiling, as
    # Python compares an int with a float: `int > 0.5` is `int >= 1`.
    pytest.param(
        "refutes",
        intersection(complement(bool), Annotated[int, at.Gt(0.5)]),
        nothing,
        id="json:~bool&int>0.5-is-inhabited",
    ),
    pytest.param(
        "equivalent",
        Annotated[int, at.Gt(0.5)],
        Annotated[int, at.Ge(1)],
        id="int>0.5==int>=1",
    ),
    # A recursive schema holding a class meets its complement in nothing: the
    # constructor cannot vouch for the class, and the decision reads the
    # reference through its definition.
    pytest.param(
        "empty",
        intersection(
            intersection(_JSON, list[complex]),
            complement(intersection(_JSON, list[complex])),
        ),
        None,
        id="empty:json&list[complex]&~itself",
    ),
    # A bare container class is its kind, so the two spellings are one schema and
    # each is below the other.
    pytest.param("equivalent", list, list[object], id="list==list[object]"),
    pytest.param(
        "equivalent", tuple, tuple[object, ...], id="tuple==tuple[object,...]"
    ),
    pytest.param("equivalent", set, set[object], id="set==set[object]"),
    # A set of two element types is not below the union of the sets of each: a
    # set holding one member of each is in neither. The refutation stands on
    # two members, so it holds only where nothing equates them -- two kinds,
    # two numbers, or two number kinds with room to choose.
    pytest.param(
        "refutes",
        set[int | str],
        union(set[int], set[str]),
        id="refute:set[int|str]<=set[int]|set[str]",
    ),
    pytest.param(
        "refutes",
        set[Literal[1, 2]],
        union(set[Literal[1]], set[Literal[2]]),
        id="refute:set[1|2]<=set[1]|set[2]",
    ),
    pytest.param(
        "refutes",
        frozenset[int | float],
        union(frozenset[int], frozenset[float]),
        id="refute:frozenset[int|float]<=frozenset[int]|frozenset[float]",
    ),
    pytest.param(
        "refutes",
        set[Literal[2, True]],
        union(set[Literal[2]], set[Literal[True]]),
        id="refute:set[2|True]<=set[2]|set[True]",
    ),
    pytest.param(
        "refutes",
        set[tuple[int] | tuple[str]],
        union(set[tuple[int]], set[tuple[str]]),
        id="refute:set[tuple[int]|tuple[str]]<=either-set",
    ),
    pytest.param(
        "equivalent", frozenset, frozenset[object], id="frozenset==frozenset[object]"
    ),
    pytest.param(
        "equivalent", dict, dict[object, object], id="dict==dict[object,object]"
    ),
    # A class built on a builtin narrows that builtin's kind, so it is below it.
    pytest.param("subtype", _MyInt, int, id="MyInt<=int"),
    pytest.param("subtype", _MyStr, str, id="MyStr<=str"),
    pytest.param("subtype", _MyInt, union(int, str), id="MyInt<=int|str"),
    pytest.param("empty", intersection(_MyInt, _MyStr), None, id="empty:MyInt&MyStr"),
    pytest.param(
        "empty",
        intersection(_Slotted, _OtherSlotted),
        None,
        id="empty:Slotted&OtherSlotted",
    ),
    pytest.param("empty", intersection(_Slotted, str), None, id="empty:Slotted&str"),
    pytest.param("subtype", _Slotted, complement(int), id="Slotted<=~int"),
    pytest.param(
        "subtype", _Slotted, complement(_OtherSlotted), id="Slotted<=~OtherSlotted"
    ),
    pytest.param("subtype", _MyStr, complement(int), id="MyStr<=~int"),
    # A union on the right is tried branch by branch, which is lossy: it commits
    # to one branch. Where the subject is one a left-side rule reduces -- a
    # reference to its definition, a refinement to its base -- both rules are
    # asked, because the relation is their disjunction and not whichever arm the
    # match reaches first.
    pytest.param(
        "subtype",
        _RECURSIVE,
        union(None, {"value": int, "next": _RECURSIVE}),
        id="mu-t<=its-own-body",
    ),
    # A fixpoint met with a kind its body never admits. The rules read the
    # shape and a set representation holds no cycle, so neither decides it
    # alone; unfolding the body once puts the kinds it admits in front of the
    # representation, and a bound stands where the reference was.
    pytest.param("empty", intersection(bytes, _JSON), None, id="empty:bytes&json"),
    pytest.param("empty", intersection(tuple, _JSON), None, id="empty:tuple&json"),
    pytest.param(
        "empty", intersection(set[int], _JSON), None, id="empty:set[int]&json"
    ),
    pytest.param("subtype", _JSON, anything, id="json<=anything"),
    # An enumeration is the union of its members when every instance of the
    # class is one of them: it is an `Enum` that is not a `Flag`, it has at
    # least one member, and its members compare by identity. The kinds that
    # fail one of those stay atoms, and `test_enums.py` holds each to a value
    # that would refute the union reading.
    pytest.param(
        "equivalent", _Color, Literal[_Color.RED, _Color.GREEN], id="Color==members"
    ),
    pytest.param(
        "empty",
        intersection(Literal[_Color.RED], Literal[_Color.GREEN]),
        None,
        id="empty:RED&GREEN",
    ),
    # A length is a regular property of a sequence -- "any element, that many
    # times" -- so the component that holds sequences holds a bound on them too.
    pytest.param(
        "empty",
        Annotated[tuple[int, int], at.MinLen(3)],
        None,
        id="empty:2-tuple&MinLen(3)",
    ),
    pytest.param(
        "empty",
        Annotated[list[nothing], at.MinLen(1)],  # ty: ignore[invalid-type-form]
        None,
        id="empty:list[nothing]&MinLen(1)",
    ),
    pytest.param(
        "empty",
        Annotated[list[int], at.MinLen(3), at.MaxLen(2)],
        None,
        id="empty:list&MinLen(3)&MaxLen(2)",
    ),
    pytest.param(
        "equivalent",
        Annotated[list[int], at.MinLen(2)],
        [int, int, int, ...],
        id="MinLen(2)list[int]==[int,int,int,...]",
    ),
    pytest.param(
        "equivalent",
        Annotated[list[int], at.MaxLen(0)],
        [],
        id="MaxLen(0)list[int]==[]",
    ),
    # A bound over floats is a set of floats, so the descriptor holds it and the
    # relations that turn on one are decided. Which side a bound lands on is
    # chosen by the *base*: `Annotated[float, Gt(0)]` carries the integer zero
    # and orders the floats all the same.
    pytest.param(
        "empty",
        intersection(Annotated[float, at.Gt(0)], int),
        None,
        id="empty:float>0&int",
    ),
    pytest.param(
        "empty",
        intersection(Annotated[float, at.Gt(0)], Annotated[float, at.Lt(0)]),
        None,
        id="empty:float>0&float<0",
    ),
    pytest.param(
        "empty",
        intersection(str, Annotated[float, at.Gt(0)]),
        None,
        id="empty:str&float>0",
    ),
    pytest.param(
        "subtype",
        Annotated[float, at.Gt(0)],
        Annotated[float, at.Gt(-1)],
        id="float>0<=float>-1",
    ),
    pytest.param(
        "empty",
        Annotated[float, at.Gt(float("inf"))],
        None,
        id="empty:float>inf",
    ),
    pytest.param("subtype", bytes, complement(_JSON), id="bytes<=~json"),
    # A fixpoint beside its own complement. A merged validator reuses the
    # definitions it holds, so two occurrences of one recursive schema are one
    # definition and the fold that cancels a schema against its complement
    # compares one term against itself. The law reaches a fixpoint like any
    # other set.
    pytest.param(
        "empty",
        intersection(_RECURSIVE, complement(_RECURSIVE)),
        None,
        id="empty:mu-t&~mu-t",
    ),
    pytest.param(
        "subtype",
        # `int | str` rather than `union(int, str)`: the two build the same
        # schema, and `Annotated` takes a *type*, which 3.10 checks as the form
        # is built -- so a runtime validator there raises at import and takes
        # the whole module out of collection.
        Annotated[int | str, at.Ge(0)],
        union(int, str),
        id="refinement-of-a-union<=the-union",
    ),
    # The lattice bounds, decided by EMPTINESS rather than by the shape of the
    # atom. A schema that denotes the empty set without being spelled `nothing`
    # is below everything; one that covers the universe without being spelled
    # `anything` is above everything. Enumerated only for the atoms, both would
    # be a rule confirming itself -- and the region check upstream decides
    # a scalar right-hand side correctly, so these want a container, a record or
    # the gradual atom on the other side to reach the arm at all.
    pytest.param(
        "subtype", intersection(int, complement(int)), list[int], id="empty<=list[int]"
    ),
    pytest.param(
        "subtype", intersection(int, complement(int)), {"a": int}, id="empty<={a:int}"
    ),
    pytest.param("subtype", {"a": NoReturn}, list[int], id="{a:Never}<=list[int]"),
    pytest.param(
        "subtype", list[int], union(int, complement(int)), id="list[int]<=universal"
    ),
    pytest.param(
        "subtype", {"a": int}, union(int, complement(int)), id="{a:int}<=universal"
    ),
    pytest.param(
        "equivalent",
        intersection(int, complement(int)),
        nothing,
        id="int&~int==nothing",
    ),
    pytest.param("equivalent", union(bool, int), int, id="bool|int==int"),
    pytest.param("equivalent", intersection(int, int), int, id="int&int==int"),
    # A refinement is a subtype of its base, and a refinement with more bound or
    # length constraints is a subtype of one with fewer (equal bounds share a
    # pool index, so nested bounds decide by syntactic containment).
    pytest.param("subtype", _GE0, int, id="refine:Ge(0)<=int"),
    pytest.param("subtype", _GE0_LE10, _GE0, id="refine:Ge(0)Le(10)<=Ge(0)"),
    # Bound entailment: a tighter bound is a subtype of a looser one even when the
    # bound values differ, decided through the ordering oracle rather than a
    # verbatim constraint match.
    pytest.param(
        "subtype",
        Annotated[int, at.Ge(5)],
        Annotated[int, at.Ge(0)],
        id="refine:Ge(5)<=Ge(0)",
    ),
    pytest.param(
        "subtype",
        Annotated[int, at.Gt(5)],
        Annotated[int, at.Ge(0)],
        id="refine:Gt(5)<=Ge(0)",
    ),
    pytest.param(
        "subtype",
        Annotated[int, at.Le(0)],
        Annotated[int, at.Le(5)],
        id="refine:Le(0)<=Le(5)",
    ),
    pytest.param(
        "subtype",
        Annotated[str, at.MinLen(5)],
        Annotated[str, at.MinLen(2)],
        id="refine:MinLen(5)<=MinLen(2)",
    ),
    pytest.param(
        "subtype",
        Annotated[str, at.MaxLen(2)],
        Annotated[str, at.MaxLen(5)],
        id="refine:MaxLen(2)<=MaxLen(5)",
    ),
    # A bound conjunction whose lower bound exceeds its upper bound is empty,
    # whether the bounds sit on one refinement or across an intersection.
    pytest.param("empty", _GE10_LE0, None, id="empty:Ge(10)Le(0)"),
    pytest.param(
        "empty",
        intersection(_GE0, Annotated[int, at.Lt(0)]),
        None,
        id="empty:Ge(0)&Lt(0)",
    ),
    # An integer-discrete open interval whose endpoints are ordered but adjacent
    # in the integers admits no value: there is no `int` strictly between 0 and 1.
    pytest.param(
        "empty", Annotated[int, at.Gt(0), at.Lt(1)], None, id="empty:int-open-(0,1)"
    ),
    # The endpoints need not be integers themselves; the interval still skips
    # every integer, whether the bounds are strict (open) or inclusive (closed).
    pytest.param(
        "empty",
        Annotated[int, at.Gt(0.5), at.Lt(0.9)],
        None,
        id="empty:int-open-(0.5,0.9)",
    ),
    pytest.param(
        "empty",
        Annotated[int, at.Ge(0.5), at.Le(0.9)],
        None,
        id="empty:int-closed-[0.5,0.9]",
    ),
    # An interval that skips every integer is empty whichever way its bounds are
    # spelled: on one refinement, or across an intersection whose members bound
    # the meet to the integers. `bool` is bounded to the integers too.
    pytest.param(
        "empty", intersection(_GT0, _LT1), None, id="empty:int-open-(0,1)-across-a-meet"
    ),
    pytest.param(
        "empty", Annotated[bool, at.Gt(0), at.Lt(1)], None, id="empty:bool-open-(0,1)"
    ),
    # An object schema is a class atom met with a record of its attributes, and
    # the meet relates to the meet a conjunct at a time: the class through the
    # leaf oracle, the record by attribute width and depth.
    pytest.param("subtype", _Dog, _Animal, id="attrs:Dog<=Animal"),
    # And it is a subtype of itself however the double complement is spelled.
    pytest.param("subtype", _Dog, complement(complement(_Dog)), id="attrs:Dog<=~~Dog"),
    # An intersection that mixes a recursive reference with a union is a subtype
    # of itself: reflexivity holds even when the meet contains its own supertype.
    pytest.param(
        "subtype",
        intersection(_RECURSIVE, union(int, str)),
        intersection(_RECURSIVE, union(int, str)),
        id="reflexive:intersection(rec,union)",
    ),
    # A mapping is a subtype of one with more clauses subsuming its own; a closed
    # record is a subtype of an open map that declares its fields.
    pytest.param(
        "subtype", {str: int}, {str: int, int: bool}, id="map:{str}<={str,int}"
    ),
    pytest.param("subtype", {}, {str: int}, id="map:{}<={str:int}"),
    # A record mixed with a catch-all narrows field-wise and clause-wise.
    pytest.param(
        "subtype", {"a": bool, str: bool}, {"a": int, str: int}, id="map:mixed-narrow"
    ),
    # A mixed map with an extra field covered by the supertype's catch-all.
    pytest.param(
        "subtype",
        {"a": int, "b": str, str: bytes},
        {"a": int, str: object},
        id="map:mixed-extra-field-covered",
    ),
    # The supertype declares an *optional* field the subtype lacks; the subtype's
    # catch-all value type fits it, so the relation decides.
    pytest.param(
        "subtype",
        {"a": int, str: int},
        {"a": int, "b?": int, str: int},
        id="map:b-extra-optional-covered",
    ),
    # The subtype is a pure mapping whose catch-all covers the supertype's optional
    # field and catch-all alike.
    pytest.param(
        "subtype",
        {str: int},
        {"b?": int, str: int},
        id="map:pure<=mixed-optional",
    ),
    # A closed record against a catch-all mapping. One rule serves every keyed-map
    # shape: a branch of its own for the closed record read a field the supertype
    # covered through a catch-all as undecided, though the general rule beside it
    # already decided exactly that.
    pytest.param("subtype", {"x": int}, {str: int}, id="map:closed-record<=mapping"),
    # A field the supertype covers through two clauses together: each clause
    # alone refutes the field's type on a value of its own, and no one value is
    # outside both.
    pytest.param(
        "subtype",
        {"f": union(int, str)},
        {str: int, Any: str},
        id="map:field<=two-clauses-together",
    ),
    # A named tuple's positions are its attributes, and the frontend reads them,
    # so the shape its instances have is in the schema and the relation is
    # structural rather than a question about a class's layout.
    pytest.param("subtype", _Pair, tuple[int, int], id="NamedTuple<=tuple[int,int]"),
    pytest.param(
        "subtype", {"x": int}, {str: object}, id="map:closed-record<=wide-mapping"
    ),
    # A record whose fields each take two types, against the union of the four
    # records that fix both. The union *is* the record, the way a fixed sequence
    # is the union of the tuples it splits across. The rules split a record
    # across the union as they split a product (`keyed_map_splits_across_union`),
    # and the descriptor answers the same by emptiness of the difference.
    #
    # Both spellings of that difference are rows, because they are one set and
    # the answer is the set's: the complement of the whole union and the meet of
    # the four complements are De Morgan's two sides, and a bound reached under
    # one and not the other is the spelling deciding the relation.
    pytest.param(
        "subtype",
        {"a": union(int, str), "b": union(int, str)},
        union(
            {"a": int, "b": int},
            {"a": int, "b": str},
            {"a": str, "b": int},
            {"a": str, "b": str},
        ),
        id="map:record<=its-corners",
    ),
    pytest.param(
        "empty",
        intersection(
            {"a": union(int, str), "b": union(int, str)},
            complement(
                union(
                    {"a": int, "b": int},
                    {"a": int, "b": str},
                    {"a": str, "b": int},
                    {"a": str, "b": str},
                )
            ),
        ),
        None,
        id="empty:record-minus-a-union-of-its-corners",
    ),
    pytest.param(
        "empty",
        intersection(
            {"a": union(int, str), "b": union(int, str)},
            complement({"a": int, "b": int}),
            complement({"a": int, "b": str}),
            complement({"a": str, "b": int}),
            complement({"a": str, "b": str}),
        ),
        None,
        id="empty:record-minus-each-of-its-corners",
    ),
    # The same shape one field wider, which is what says the work allowance has
    # a margin rather than a figure that happens to fit: the two-field split
    # spends under half of it and this one under all of it.
    pytest.param(
        "subtype",
        {"a": union(int, str), "b": union(int, str), "c": union(int, str)},
        union(
            *[
                {"a": a, "b": b, "c": c}
                for a in (int, str)
                for b in (int, str)
                for c in (int, str)
            ]
        ),
        id="map:wider-record<=its-corners",
    ),
    # The record rule's width, where the product rule's is: six fields against
    # their sixty-four corners decide, as six tuple components do, which is the
    # number the decidability page gives.
    pytest.param("subtype", *_record_split(6), id="map:six-field-record<=its-corners"),
    # A record whose tag takes three values against records that each fix the
    # tag: covered where no member asks for more, refuted on
    # `{"kind": "a", "order": 0}` where one requires a key the subject leaves
    # free. The literals are the tag's values, which the split narrows one
    # record at a time.
    pytest.param(
        "subtype",
        {"kind": Literal["a", "b", "c"], "order": int, str: Any},
        union(
            {"kind": Literal["a"], "order": int, str: Any},
            {"kind": Literal["b"], "order": int, str: Any},
            {"kind": Literal["c"], "order": int, str: Any},
        ),
        id="map:tagged-record<=the-records-fixing-its-tag",
    ),
    pytest.param(
        "refutes",
        {"kind": Literal["a", "b", "c"], "order": int, str: Any},
        union(
            {"kind": Literal["a"], "order": int, "amount": int, str: Any},
            {"kind": Literal["b"], "order": int, "reason": str, str: Any},
            {"kind": Literal["c"], "order": int, str: Any},
        ),
        id="map:tagged-record!<=members-requiring-more",
    ),
    # Two tags, three values each: the nine corners cover the record, and the
    # eight left when one is missing do not.
    pytest.param(
        "subtype",
        {"kind": Literal["a", "b", "c"], "status": Literal["x", "y", "z"], str: Any},
        union(
            *[
                {"kind": Literal[kind], "status": Literal[status], str: Any}  # ty: ignore[invalid-type-form]
                for kind in "abc"
                for status in "xyz"
            ]
        ),
        id="map:two-tag-record<=its-nine-corners",
    ),
    # A clause keyed by a literal is read by the constant it names: the field
    # of that name reads it, and two mappings keyed by one literal are refuted
    # under it at any depth, as two `dict[str, ...]` chains are.
    pytest.param(
        "subtype",
        dict[Literal["a"], int],
        {"a?": int},
        id="map:dict[L[a],int]<={a?:int}",
    ),
    pytest.param(
        "refutes",
        _literal_keyed(int, 40),
        _literal_keyed(str, 40),
        id="refute:literal-keyed-mappings-40-deep-over-int<=over-str",
    ),
    # A record is refuted against a mapping keyed by literals on a `str`
    # subclass's key equal to the field's name, which the field reads and the
    # literal does not.
    pytest.param(
        "refutes",
        {"a": int},
        dict[Literal["a"], int],
        id="map:{a:int}!<=dict[L[a],int]",
    ),
    # A record split on whether a key is there: `c` present lands in the first
    # member, `c` absent in the second, and neither holds the record alone.
    pytest.param(
        "subtype",
        {"a": int, "c?": bool},
        union({"c": union(int, str), str: Any}, {"c?": bytes, str: Any}),
        id="map:record<=split-on-a-keys-presence",
    ),
    # A refutation the rules reach on one branch of a union and must not lose on
    # the others. A branch the subject shares no value with decides nothing
    # about the inclusion, so leaving it in the union leaves a decline where the
    # remaining branch refutes -- and the value that refutes is one the walk
    # reads, listed beside each row.
    #
    # Each needs no second unfolding: the witness sits at the first level, which
    # is what separates these from the fixpoint pairs the descriptor declines.
    pytest.param("refutes", _CHAIN_WITH_A_FIELD, _CHAIN, id="mu:wider-chain!<=chain"),
    pytest.param(
        "refutes",
        union(None, {"next": _CHAIN_WITH_A_FIELD, "v?": int}),
        _CHAIN,
        id="mu:unfolded-wider-chain!<=chain",
    ),
    pytest.param(
        "refutes",
        _CHAIN,
        union(None, {"next": int}),
        id="mu:chain!<=none-or-int-next",
    ),
    pytest.param(
        "refutes",
        {"a": _CHAIN, "b": int},
        union({"a": _CHAIN}, {"a": _CHAIN, "b": str}),
        id="mu:record-with-a-fixpoint-field!<=its-siblings",
    ),
    pytest.param("refutes", _CHAIN, int, id="mu:chain!<=int"),
    # A meet with a recursive reference, both ways and in both spellings of a
    # chain: empty where the record asks of `next` what no chain holds, and
    # inhabited where it asks what one does. The reference is unfolded into
    # the meet, and the meet of its key unfolds it again, under a goal of its
    # own rather than the reference's.
    pytest.param(
        "empty", intersection(_CHAIN, {"next": int}), None, id="mu:chain&{next:int}"
    ),
    pytest.param(
        "refutes",
        intersection(_CHAIN, {"next": {"next": None}}),
        nothing,
        id="mu:chain&{next:{next:None}}!<=nothing",
    ),
    pytest.param(
        "empty",
        intersection(_LINK, {"next": {"next": int}}),
        None,
        id="mu:link&{next:{next:int}}",
    ),
    pytest.param(
        "refutes",
        intersection(_LINK, {"next": {"next": None}}),
        nothing,
        id="mu:link&{next:{next:None}}!<=nothing",
    ),
    # The meet the reference's own trail would read empty: `{"a": None}` is in
    # it, and the key asks `t & (None | {"a": None})`, a goal of its own.
    pytest.param(
        "refutes",
        intersection(_NESTED, {"a": union(None, {"a": None})}),
        nothing,
        id="mu:t&{a:None|{a:None}}!<=nothing",
    ),
    # A key the subject's clause holds a whole kind of, which the supertype
    # reads through one clause or none: [Cas23, Lem. 4.7], a key kind at a time.
    # The rules refute each; against a recursive supertype the set
    # representation declines, and they are the only decider. The value that
    # refutes is listed beside each row.
    pytest.param(
        "refutes",
        Validator({str: int}).open(),
        {str: int},
        id="key:opened-mapping!<=mapping",  # {7: "x"}
    ),
    pytest.param(
        "refutes",
        {"a": str},
        Validator({str: int}).open(),
        id="key:record!<=opened-mapping",  # {"a": ""}
    ),
    pytest.param(
        "refutes", {str: bytes}, _JSON, id="key:str-to-bytes!<=json"
    ),  # {"a": b""}
    pytest.param("refutes", dict, _JSON, id="key:dict!<=json"),  # {7: 1}
    pytest.param(
        "refutes", {tuple: int}, {str: int}, id="key:tuple-key!<=str-key"
    ),  # {(): 1}
    pytest.param(
        "refutes",
        recursive(lambda t: union(None, {"next": t, str: Any})),
        _CHAIN,
        id="key:open-chain!<=chain",  # {"next": None, "x": 1}
    ),
    pytest.param(
        "refutes",
        {complement(str): int},
        {complement(str): str},
        id="key:complement-key-values!<=",  # {7: 1}
    ),
    pytest.param(
        "refutes",
        {complement(int): union(int, str)},
        {complement(int): str},
        id="key:complement-key-names!<=",  # {"x": 1}
    ),
    # A key-type the partition holds as parts: a complement of whole kinds is
    # the other kinds and the part for a key of no listed kind, a class
    # instance, and a tuple or a frozenset of anything is its kind whole. The
    # sets decide these where the rules read no key kind one way.
    pytest.param(
        "refutes", _Plain, {complement(str): int}, id="key:class!<=complement-key"
    ),  # _Plain()
    pytest.param(
        "refutes",
        _Plain,
        Validator({str: int}).open(),
        id="key:class!<=opened-mapping",  # _Plain()
    ),
    pytest.param(
        "refutes", _Plain, {tuple: int}, id="key:class!<=tuple-key"
    ),  # _Plain()
    pytest.param(
        "refutes",
        {complement(str): int},
        {union(None, bool, int, float, bytes, tuple, frozenset): int},
        id="key:complement-key!<=every-listed-kind",  # {_Plain(): 1}
    ),
    pytest.param(
        "refutes",
        {complement(bool): int},
        {complement(int): int},
        id="key:complement-bool!<=complement-int",  # {1: 1}
    ),
    pytest.param(
        "subtype",
        {complement(bool): int},
        {complement(int): int, int: int},
        id="key:complement-bool<=complement-int-and-int",
    ),
    pytest.param(
        "subtype",
        dict,
        {str: anything, complement(str): anything},
        id="key:dict<=two-clauses-covering-every-key",
    ),
    # The finite parts keep their keys: `None` is one key and `bool` two, so a
    # value of two kinds under the one is a value of one of them, and under the
    # two may be one of each.
    pytest.param(
        "subtype",
        Validator({None: union(int, str)}).open(),
        union(Validator({None: int}).open(), Validator({None: str}).open()),
        id="key:none-part<=union-of-its-values",
    ),
    pytest.param(
        "refutes",
        Validator({bool: union(int, str)}).open(),
        union(Validator({bool: int}).open(), Validator({bool: str}).open()),
        id="key:bool-part!<=union-of-its-values",  # {True: 1, False: "x"}
    ),
    # Divisibility, which the steps decide between them: every multiple of `a`
    # is a multiple of `b` exactly when `b` divides `a`. The rules read the two
    # steps rather than the values they name, so the answer does not depend on
    # how many residues a representation materializes -- a rule reading the
    # residues would leave the pair below undecided and decide `MultipleOf(4)`
    # against `MultipleOf(2)`, for no reason a caller could see.
    pytest.param(
        "subtype",
        Annotated[int, at.MultipleOf(5000)],
        Annotated[int, at.MultipleOf(2500)],
        id="divides:5000<=2500",
    ),
    # The refutation needs a value, and a subject narrowed by its step alone
    # names one at any size: `2500` is a multiple of 2500 and not of 5000. What
    # the residues are needed for is a refutation the subject's other
    # constraints take part in, which the decidability page records.
    pytest.param(
        "refutes",
        Annotated[int, at.MultipleOf(2)],
        Annotated[int, at.MultipleOf(4)],
        id="divides:2!<=4",
    ),
    pytest.param(
        "refutes",
        Annotated[int, at.MultipleOf(2500)],
        Annotated[int, at.MultipleOf(5000)],
        id="divides:2500!<=5000",
    ),
    # And the divisors a caller writes that are not integers: the question is
    # Python's `%`, so a step reads as the kind it is.
    pytest.param(
        "subtype",
        Annotated[float, at.MultipleOf(0.25)],
        Annotated[float, at.MultipleOf(0.125)],
        id="divides:0.25<=0.125",
    ),
    # Inclusion in a complement, which is `A ∩ B = ∅` and nothing structural: a
    # complement offers no shape on the right to recurse into.
    pytest.param(
        "subtype", list[int], complement(int), id="complement:list[int]<=~int"
    ),
    pytest.param(
        "subtype", dict[str, int], complement(str), id="complement:dict<=~str"
    ),
    pytest.param("subtype", int, complement(str), id="complement:int<=~str"),
    # Decided on the *sets* rather than by a rule: each is a fact about values
    # that no recursion over the schema tree reaches, and each is decided by
    # asking the descriptor where the rules decline. A container meet is the
    # meet of the element sets, a complement is a set again, a word
    # set is a language, a dict is a union of atoms with a default per key kind,
    # and a bound is a set of whole numbers -- so container meets, double
    # complements, regular-language inclusion, dict emptiness and the ordering of
    # two steps all come out of one question: is the difference empty?
    pytest.param(
        "subtype",
        intersection(union(list[int], str), complement(str)),
        list[int],
        id="(list[int]|str)&~str<=list[int]",
    ),
    pytest.param(
        "subtype",
        tuple[int],
        complement(tuple[str]),
        id="tuple[int]<=~tuple[str]",
    ),
    pytest.param(
        "subtype",
        tuple[int, str],
        complement(tuple[str, int]),
        id="tuple[int,str]<=~tuple[str,int]",
    ),
    pytest.param(
        "subtype",
        intersection(list[int], list[str]),
        [],
        id="list[int]&list[str]<=[]",
    ),
    pytest.param(
        "equivalent",
        intersection(set[int], set[str]),
        set[nothing],  # ty: ignore[invalid-type-form]
        id="set[int]&set[str]==set[nothing]",
    ),
    pytest.param(
        "subtype",
        Annotated[str, Regex("a")],
        Annotated[str, Regex("a|b")],
        id="Regex(a)<=Regex(a|b)",
    ),
    # A pattern past the size an automaton is built to is read a text at a
    # time, each put to the validator's matcher: every text of a pattern that
    # has few, or a guess at one text of a pattern that has many.
    pytest.param(
        "refutes",
        Annotated[str, Regex("a")],
        _COUNTED_CLASS,
        id="Regex(a)<=Regex((?:\\w{3,4})+)-is-refuted",
    ),
    pytest.param(
        "subtype",
        Annotated[str, Regex("abc|abcd")],
        _COUNTED_CLASS,
        id="Regex(abc|abcd)<=Regex((?:\\w{3,4})+)",
    ),
    pytest.param(
        "refutes",
        _COUNTED_CLASS,
        nothing,
        id="Regex((?:\\w{3,4})+)-is-inhabited",
    ),
    pytest.param(
        "empty",
        intersection(Annotated[str, Regex("a")], _COUNTED_CLASS),
        None,
        id="Regex(a)&Regex((?:\\w{3,4})+)-is-empty",
    ),
    pytest.param(
        "subtype",
        Validator(list[int]),
        complement(union(complement(Validator(list[int])), nothing)),
        id="A<=~(~A|nothing)",
    ),
    # Emptiness has no structural-inclusion rule, so a meet with a complement is
    # not decided even where the inclusion under it is.
    pytest.param(
        "empty",
        intersection(list[bool], complement(list[int])),
        None,
        id="empty:list[bool]&~list[int]",
    ),
    pytest.param(
        "subtype",
        intersection(Literal["a", "b"], complement(Validator(Literal["a"]))),
        Literal["b"],
        id="L[a,b]&~L[a]<=L[b]",
    ),
    pytest.param(
        "subtype",
        {"a": union(int, str)},
        union(Validator({"a": int}), Validator({"a": str})),
        id="{a:int|str}<={a:int}|{a:str}",
    ),
    # A container meet is not intersected componentwise, so a meet that is the
    # empty container is not recognized as one.
    # A scalar kind is a region bit rather than a set of its values, so a finite
    # kind is not the union of its members and a negated literal has nowhere to go.
    pytest.param("subtype", bool, Literal[True, False], id="bool<=L[True,False]"),
    pytest.param(
        "subtype",
        Literal[1],
        intersection(int, complement(bool)),
        id="L[1]<=int&~bool",
    ),
    pytest.param(
        "subtype",
        int,
        union(intersection(int, complement(Validator(Literal[1]))), Literal[1]),
        id="int<=(int&~L[1])|L[1]",
    ),
    # A refinement's bounds are compared, and the values under them are not, so a
    # base that is not itself a refinement does not reach them.
    pytest.param("subtype", bool, Annotated[int, at.Ge(0)], id="bool<=int&Ge(0)"),
    pytest.param(
        "subtype",
        _GT0,
        Annotated[int, at.Ge(1)],
        id="int&Gt(0)<=int&Ge(1)",
    ),
    pytest.param(
        "subtype",
        intersection(_GT0, Annotated[int, at.Lt(10)]),
        Annotated[int, at.Gt(0), at.Lt(10)],
        id="meet-of-refinements<=joint-refinement",
    ),
    pytest.param("empty", Annotated[bool, at.Ge(2)], None, id="empty:bool&Ge(2)"),
    # A pattern and a divisor are related only by being written alike, so neither
    # the language nor the divisibility is read.
    pytest.param(
        "subtype",
        Annotated[int, at.MultipleOf(4)],
        Annotated[int, at.MultipleOf(2)],
        id="MultipleOf(4)<=MultipleOf(2)",
    ),
    # A map's domain is the field list as written, and a key type is matched
    # against the string atom rather than asked whether it admits the name. The
    # literal is the exact `str`, which the field reads too, so the mapping is
    # below the record; the other way a `str` subclass's `a` separates them.
    pytest.param(
        "subtype",
        {Literal["a"]: int},
        {"a?": int},
        id="dict[L[a],int]<={a?:int}",
    ),
    pytest.param("equivalent", {"a?": nothing}, {}, id="{a?:nothing}=={}"),
    pytest.param(
        "equivalent",
        dict[str, nothing],  # ty: ignore[invalid-type-form]
        {},
        id="dict[str,nothing]=={}",
    ),
    pytest.param(
        "equivalent",
        dict[nothing, int],  # ty: ignore[invalid-type-form]
        {},
        id="dict[nothing,int]=={}",
    ),
    pytest.param(
        "empty",
        intersection({"a": int}, dict[str, str]),
        None,
        id="empty:{a:int}&dict[str,str]",
    ),
    pytest.param(
        "empty",
        intersection({"a": int}, dict[int, str]),
        None,
        id="empty:{a:int}&dict[int,str]",
    ),
    pytest.param(
        "equivalent",
        intersection({"a?": int}, {"b?": int}),
        {},
        id="{a?:int}&{b?:int}=={}",
    ),
    # A subject disjoint from a meet is below that meet's complement, which is
    # the same question `is_empty` answers about the meet, and the two answer
    # alike. The rule that decides most meets empty compares the members of one
    # intersection pairwise, so a meet holding its second operand as a nested
    # intersection is flattened before the comparison rather than leaving the
    # tuple and the dict inside it uncompared.
    pytest.param(
        "subtype",
        tuple[int, str],
        complement(intersection(dict[str, int], complement({"a": int}))),
        id="tuple<=~(dict&~{a:int})",
    ),
    # The shapes a decision budget hides a regression behind: wide unions of
    # tagged records, the complement of a union of records with gradual
    # members, and a chain of differences against open records. Each is
    # decided at the width here, so a change that turns one into a decline
    # fails its row; the chain's first width past the bound is ledgered below.
    pytest.param(
        "subtype",
        {"t": 16, "v": bool},
        union(*[{"t": i, "v": int} for i in range(32)]),
        id="map:tagged-record<=a-union-of-32-tagged-records",
    ),
    pytest.param(
        "subtype",
        {"t": 16, "v": int},
        complement(union(*[{"t": i, "v": Any} for i in range(16)])),
        id="map:record<=the-complement-of-16-gradual-records",
    ),
    # Two records, the width at which a field spanning two kinds expands the
    # complement taken of the complement; the set representation meets the
    # subject with the union itself.
    pytest.param(
        "subtype",
        {"t": int},
        complement(union({"t": str}, {"t": bytes})),
        id="map:record<=the-complement-of-two-records",
    ),
    # The chain the record rule reads as emptiness, one complement at a time,
    # with the key's constants taken away as a set; its length is no bound.
    pytest.param(
        "empty",
        _chain(12),
        None,
        id="empty:open-record-minus-a-chain-of-12",
    ),
    pytest.param(
        "empty",
        _chain(64),
        None,
        id="empty:open-record-minus-a-chain-of-64",
    ),
    # A list is refuted by its rules at any depth, and a mapping is too: the
    # value its clause gives a key is asked against the one clause that reads
    # that key, a level at a time, so the nesting a lowering descends does not
    # bound it.
    pytest.param(
        "refutes",
        _nested(int, 127, mapping=False),
        _nested(str, 127, mapping=False),
        id="refute:lists-127-deep-over-int<=over-str",
    ),
    pytest.param(
        "refutes",
        _nested(int, 4, mapping=True),
        _nested(str, 4, mapping=True),
        id="refute:mappings-4-deep-over-int<=over-str",
    ),
    # A subject too wide for the set representation, below the complement of a
    # refinement only the set representation empties: the complement rule reads
    # the inner set alone.
    pytest.param(
        "subtype",
        union(*[Annotated[str, Regex(f"a{i}+")] for i in range(40)]),
        complement(Annotated[str, Regex("x"), at.MaxLen(0)]),
        id="str:forty-patterns<=the-complement-of-an-empty-pattern",
    ),
    # A complement the set representation cannot build, below the same
    # universe: the contravariant rule reads the inner set alone too.
    pytest.param(
        "subtype",
        complement(Annotated[str, at.MaxLen(0), str.isdigit]),
        complement(Annotated[str, Regex("x"), at.MaxLen(0)]),
        id="str:the-complement-of-a-predicate<=the-complement-of-an-empty-pattern",
    ),
    pytest.param(
        "subtype",
        complement(union(*[Annotated[str, Regex(f"a{i}+")] for i in range(40)])),
        complement(Annotated[str, Regex("x"), at.MaxLen(0)]),
        id="str:the-complement-of-forty-patterns<=the-complement-of-an-empty-pattern",
    ),
]

# The shapes a translation from JSON Schema writes: the JSON universe, the number
# kinds with `bool` taken out, objects guarded by `~object | record`, records
# nested and met, and the same question asked through a complement. Each row is
# a relation such a translation asks, reduced until it stays a row.
_ARRAY = Validator([_JSON])
_OBJECT = Validator({str: _JSON})
_NOT_BOOL = complement(bool)
_NUMBER = intersection(union(int, float), _NOT_BOOL)
_INTEGER = intersection(union(int, Annotated[float, at.MultipleOf(1)]), _NOT_BOOL)


def _guarded(field: object) -> Validator:
    """Guard a record to objects: any non-object passes, an object carries `tag`."""
    return intersection(
        _OBJECT,
        union(complement(_OBJECT), Validator({"tag": field}).open()),
        union(complement(_OBJECT), Validator({"tag": _JSON}).open()),
    )


def _guarded_deep(depth: int) -> Validator:
    schema: object = int
    for _ in range(depth):
        schema = _guarded(schema)
    return Validator(schema)


def _tagged(members: int, shift: int = 0) -> Validator:
    """Build records told apart by a constant tag, as a `oneOf` writes them."""
    return union(
        *[
            Validator({"type": Literal[f"t{i}"], "v": Annotated[int, at.Ge(i + shift)]})  # ty: ignore[invalid-type-form]
            for i in range(members)
        ]
    )


def _patterns(members: int) -> Validator:
    return union(*[Annotated[str, Regex(f"p{i}")] for i in range(members)])


# An integer tree beside a closed record nested three deep, and their meet.
_INTEGER_TREE = recursive(lambda t: union({"a": _INTEGER}, str, {str: t}))
_NESTED_CLOSED = Validator(
    {
        "a": Validator(
            {"a": Validator({"tag": Validator({"a": _JSON}).open()}).open()}
        ).close()
    }
).close()
# A record of the universe, outside it, and the JSON universe widened by a
# constant two ways.
_OUTSIDE = intersection(complement(_JSON), Validator({"a": _JSON}).open())
_WIDENED = union(_JSON, Literal[0.5, True])  # ty: ignore[invalid-type-form]
_NARROWED = union(intersection(_JSON, _NUMBER), Literal["paid", ""])
# A recursive universe beside a pattern, and the inner universe a complement of
# which is a branch of a recursive one.
_NULL_TREE = recursive(lambda t: union(None, [t]))
_LETTER_A = Annotated[str, Regex("a")]
_INNER = recursive(lambda t: union(int, [t], {str: t}))
_OUTER = recursive(lambda t: union(complement(_INNER), [t], {str: t}))
_WINDOW = intersection(
    union(int, float),
    _NOT_BOOL,
    union(Annotated[int, at.Ge(5000)], Annotated[float, at.Ge(5000)]),
    union(Annotated[int, at.Le(5001)], Annotated[float, at.Le(5001)]),
)
# The smallest guarded shape a drawn case left undecided, met with the universe.
_GUARDED_MEET = intersection(
    union(
        _NUMBER,
        Validator(
            {
                "tag": intersection(
                    Validator(
                        {
                            "tag": intersection(
                                _JSON, complement(intersection(_NUMBER, _JSON))
                            )
                        }
                    ).open(),
                    union(_OBJECT, _JSON),
                )
            }
        ).open(),
    ),
    _JSON,
)


# Known decision-completeness misses: true relations neither the rules nor the
# descriptor decides.
#
# Each is marked strict, so the day one is decided the mark fails and the entry
# leaves both this list and the conservative half of the decidability page --
# which is how the entries above it left.
#
# **Each carries its own reason**, and that is the point of the list rather than
# a courtesy. One shared reason said "neither decides it" five times, which is
# the observation that made the list, not an account of anything: five entries
# with one sentence between them read as one limit seen five ways, and they are
# four different limits. A reader deciding what to work on needs to know that
# three of these fall to one change and two do not, and a reader who closes one
# needs to know which of the others it did not touch.


def _missed(why: str) -> pytest.MarkDecorator:
    """Mark a relation that holds and is not decided, with the limit that leaves it.

    Strict, so the row fails on the day the relation decides: a miss that
    closes leaves this list and the conservative half of the decidability page
    in the same commit, rather than being noticed at an audit. The ledger runs
    in both directions that way -- a decided relation regressing to
    conservatism fails above, and a conservative one deciding fails here.
    """
    return pytest.mark.xfail(strict=True, reason=why)


_LEDGERED: list[object] = [
    # The record rule's width from the other side: seven fields against their
    # 128 corners spend the decision budget before the split finishes, as seven
    # tuple components do, and the difference is past every bound a lowering
    # has. True, and the decline is sound; the number is what this row carries.
    pytest.param(
        "subtype",
        *_record_split(7),
        id="map:seven-field-record<=its-corners",
        marks=_missed("the split multiplies past the work a decision may spend"),
    ),
    # The product rule's width from the other side: seven components against
    # their 128 corners spends the decision budget before the split finishes.
    # True, and the decline is sound; the number is what this row carries.
    pytest.param(
        "subtype",
        *_split(7),
        id="tuple:seven-components<=its-corners",
        marks=_missed("the split multiplies past the work a decision may spend"),
    ),
    # Past the nesting a lowering descends (`DEPTH`, `descr/lower.rs`), which
    # is where the set representation stops reading a mapping.
    pytest.param(
        "refutes",
        _nested(int, 5, mapping=True),
        _nested(str, 5, mapping=True),
        id="refute:mappings-5-deep-over-int<=over-str",
    ),
    pytest.param(
        "refutes",
        _nested(int, 40, mapping=True),
        _nested(str, 40, mapping=True),
        id="refute:mappings-40-deep-over-int<=over-str",
    ),
    # A set of `1` and `True` is a set of one member, since `{1, True}` is
    # `{1}`, so every set of either is a set of one of them. The two members a
    # refutation would need are one number in two kinds, which the sets read
    # as unknown rather than as a proof that no set escapes.
    pytest.param(
        "subtype",
        set[Literal[1, True]],
        union(set[Literal[1]], set[Literal[True]]),
        id="set:set[1|True]<=set[1]|set[True]",
        marks=_missed("the members it takes are one number in two kinds"),
    ),
    pytest.param(
        "subtype",
        set[union(Literal[1], Validator(1.0))],  # ty: ignore[invalid-type-form]
        union(set[Literal[1]], set[Validator(1.0)]),  # ty: ignore[invalid-type-form]
        id="set:set[1|1.0]<=set[1]|set[1.0]",
        marks=_missed("the members it takes are one number in two kinds"),
    ),
]


# THEORY: an-exhaustible-procedure-is-searched, structural-rather-than-reduction
# The translation's relations: the ones decided are gates, the ones declined
# are strict, each with the limit that leaves it.
_TRANSLATED: list[object] = [
    pytest.param("subtype", _JSON, union(_JSON, str), id="json:json<=json|str"),
    pytest.param("refutes", _JSON, nothing, id="json:json-is-inhabited"),
    pytest.param("refutes", _ARRAY, nothing, id="json:array-is-inhabited"),
    pytest.param("refutes", _JSON, complement(_ARRAY), id="json:json!<=~array"),
    pytest.param("refutes", _JSON, _ARRAY, id="json:json!<=array"),
    pytest.param("refutes", _JSON, int, id="json:json!<=int"),
    pytest.param("refutes", union(float, int), _ARRAY, id="json:number!<=array"),
    pytest.param(
        "refutes",
        intersection(_INTEGER, int),
        nothing,
        id="json:integer&int-is-inhabited",
    ),
    pytest.param(
        "refutes",
        intersection(anything, _JSON),
        nothing,
        id="json:anything&json-is-inhabited",
    ),
    pytest.param("subtype", _OUTSIDE, complement(_JSON), id="json:~json&{a}<=~json"),
    pytest.param(
        "refutes",
        intersection(_WIDENED, _NARROWED),
        nothing,
        id="json:widened&narrowed-is-inhabited",
    ),
    pytest.param(
        "subtype",
        _guarded_deep(3),
        union(_guarded_deep(3), str),
        id="json:guarded-3<=guarded-3|str",
    ),
    pytest.param(
        "subtype",
        intersection(_INTEGER_TREE, _NESTED_CLOSED),
        _NESTED_CLOSED,
        id="json:tree&nested<=nested",
    ),
    pytest.param("refutes", _tagged(6), _tagged(6, 1), id="json:six-tagged!<=shifted"),
    pytest.param(
        "subtype",
        _patterns(31),
        Annotated[str, Regex("p[0-9]+")],
        id="json:thirty-one-patterns<=p[0-9]+",
    ),
    # A meet placed in a union that holds it, and below a member it holds, by
    # the terms alone and before any member's rule spends the budget.
    pytest.param(
        "subtype",
        _guarded_deep(4),
        union(_guarded_deep(4), str),
        id="json:guarded-4<=guarded-4|str",
    ),
    pytest.param(
        "subtype",
        _GUARDED_MEET,
        union(_GUARDED_MEET, _NUMBER),
        id="json:guarded-meet<=guarded-meet|number",
    ),
    pytest.param(
        "subtype",
        intersection(_INTEGER_TREE, _NESTED_CLOSED),
        _INTEGER_TREE,
        id="json:tree&nested<=tree",
    ),
    # A complement of a union is the meet of the complements, read one level in
    # and placed by membership: `~(X | ~Y)` is `~X & Y`, below `Y`.
    pytest.param(
        "subtype",
        complement(union(str, complement(list[list[int]]))),
        list[list[int]],
        id="json:~(str|~list[list[int]])<=list[list[int]]",
    ),
    pytest.param(
        "subtype",
        complement(union(_ARRAY, _NOT_BOOL)),
        bool,
        id="json:~(array|~bool)<=bool",
    ),
    # A union is refuted by one member, and a member's refutation stands at the
    # union whatever the other members' emptiness reads; a union supertype is
    # narrowed to the branches the subject meets before the sets are asked, and
    # a union subject is asked of the sets a member at a time.
    pytest.param(
        "refutes",
        union(_NULL_TREE, _LETTER_A),
        nothing,
        id="json:null-tree|pattern-is-inhabited",
    ),
    pytest.param(
        "refutes",
        union(
            [_JSON, _JSON],
            [int, Validator({"a": Annotated[str, Regex("[0-9]+")]}).open()],
        ),
        nothing,
        id="json:two-fixed-lists-are-inhabited",
    ),
    pytest.param(
        "refutes",
        str,
        union(_NULL_TREE, _LETTER_A),
        id="json:str!<=null-tree|pattern",
    ),
    pytest.param(
        "refutes",
        {"a": int},
        _OUTER,
        id="json:{a:int}!<=outer-tree",
        marks=_missed(
            "a union supertype is refuted at a value outside every branch, and the"
            " sets, which could name it, refute no pair holding a reference"
        ),
    ),
    pytest.param(
        "refutes",
        _tagged(7),
        _tagged(7, 1),
        id="json:seven-tagged!<=shifted",
    ),
    pytest.param(
        "subtype",
        _patterns(32),
        Annotated[str, Regex("p[0-9]+")],
        id="json:thirty-two-patterns<=p[0-9]+",
    ),
    # A kind the supertype holds none of is a value the subject has outside it,
    # read through the fixpoint's one unfolding.
    pytest.param(
        "refutes",
        anything,
        _JSON,
        id="json:anything!<=json",
    ),
    pytest.param(
        "refutes",
        complement(_JSON),
        _JSON,
        id="json:~json!<=json",
    ),
    pytest.param(
        "empty",
        intersection(_ARRAY, union(None, _WINDOW)),
        None,
        id="json:empty:array&(None|window)",
    ),
    # A step that is no multiple of the other's is its own witness.
    pytest.param(
        "refutes",
        Annotated[int, at.MultipleOf(5000)],
        Annotated[int, at.MultipleOf(3)],
        id="json:step-5000!<=step-3",
    ),
    pytest.param(
        "refutes",
        Annotated[int, at.MultipleOf(6)],
        Annotated[int, at.MultipleOf(2500)],
        id="json:step-6!<=step-2500",
    ),
    # A reading that cannot decide a meet hands it to the next: distributing
    # `~DN | DB` ends undecided, and the difference of the records decides the
    # meet, every member but `~dict` a set of dicts.
    pytest.param(
        "empty",
        intersection(
            union(complement(dict[Literal["a"], None]), dict[Literal["a"], bytes]),
            {"a": int},
            complement(dict),
        ),
        None,
        id="json:records-met-with-~dict-are-empty",
    ),
    # A truth value bounding a number is the integer it is to every comparison,
    # and `typing` hands `Le(1)` and `Le(True)` one marker.
    pytest.param(
        "refutes",
        Annotated[float, at.Le(True)],
        nothing,
        id="json:float-Le(True)-is-inhabited",
    ),
    pytest.param(
        "refutes",
        intersection(complement(bool), Annotated[int, at.Le(True)]),
        nothing,
        id="json:int-Le(True)-without-bool-is-inhabited",
    ),
    # A bound is asked of one value at a time, so over a union base it is the
    # union of the refined branches, and `None` orders against no number.
    pytest.param(
        "empty",
        intersection(None, Annotated[int | float, at.Le(1)]),
        None,
        id="refine:None&(int|float)<=1-is-empty",
    ),
    pytest.param(
        "refutes",
        Annotated[int | float, at.Ge(0)],
        nothing,
        id="refine:(int|float)>=0-is-inhabited",
    ),
    pytest.param(
        "subtype",
        bool,
        Annotated[int | float, at.Ge(-2)],
        id="refine:bool<=(int|float)>=-2",
    ),
    pytest.param(
        "refutes",
        None,
        Annotated[int | None, at.Ge(0)],
        id="refine:None!<=(int|None)>=0",
    ),
    pytest.param(
        "equivalent",
        Annotated[int | float, at.Le(1)],
        union(Annotated[int, at.Le(1)], Annotated[float, at.Le(1)]),
        id="refine:(int|float)<=1==int<=1|float<=1",
    ),
    # A list bounded to length zero is the empty list, which every list holds,
    # and a list of an inhabited element is refuted past a maximum length.
    pytest.param(
        "subtype",
        Annotated[list[None], at.Len(0, 0)],
        list[Annotated[float, at.MultipleOf(1)]],
        id="seq:[]<=list[float-step]",
    ),
    pytest.param(
        "refutes",
        list[union(int, Annotated[float, at.MultipleOf(1)])],  # ty: ignore[invalid-type-form]
        Annotated[list[union(int, Annotated[float, at.MultipleOf(1)])], at.Len(0, 1)],  # ty: ignore[invalid-type-form]
        id="seq:list[INTEGER]!<=maxItems-1",
    ),
    # A scalar tower deep in meets and complements starts no build of its own,
    # so the set representation reads it past the depth that bounds containers.
    pytest.param(
        "refutes",
        intersection(
            int,
            complement(intersection(int, complement(Annotated[int, at.MultipleOf(2)]))),
        ),
        nothing,
        id="scalar:int&~(int&~even)-is-inhabited",
    ),
    pytest.param(
        "refutes",
        intersection(int, complement(intersection(int, complement(Literal[1, 2, 3])))),
        nothing,
        id="scalar:int&~(int&~{1,2,3})-is-inhabited",
    ),
    pytest.param(
        "empty",
        intersection(
            int,
            complement(
                union(intersection(Annotated[int, at.MultipleOf(4)], bool), int)
            ),
        ),
        None,
        id="scalar:int&~((mult4&bool)|int)-is-empty",
    ),
    pytest.param(
        "empty",
        intersection(
            int,
            complement(intersection(int, complement(Literal[0.5, 1.0, b""]))),  # ty: ignore[invalid-type-form]
        ),
        None,
        id="scalar:int&~(int&~{0.5,1.0,b''})-is-empty",
    ),
    # Two records sharing no value empty a meet beside a member whose scalars
    # nest past that depth under a literal key's complement: the reading that
    # holds the member proves nothing, and the one that leaves it out proves
    # the meet empty.
    pytest.param(
        "empty",
        intersection(
            dict[str, Literal["a"]],
            complement(
                intersection(
                    dict[Literal["a"], complement(str)],  # ty: ignore[invalid-type-form]
                    complement(dict[Literal["a"], int]),
                )
            ),
            {"a": Literal[0]},
        ),
        None,
        id="map:dict[str,'a']&~(dict['a',~str]&~dict['a',int])&{'a':0}-is-empty",
    ),
    # A meet whose every member holds such a scalar reads, left without them,
    # as the universe, which is below a union of complements whose inner sets
    # share no value: `{"a": None}` and `dict[Literal["a"], Literal["a"]]`.
    pytest.param(
        "subtype",
        intersection(
            complement(intersection({"a": str}, {"b": complement(float)})),
            complement(intersection(dict[Literal["a"], float], {"b": complement(int)})),
        ),
        union(complement({"a": None}), complement(dict[Literal["a"], Literal["a"]])),
        id="map:~({a:str}&{b:~float})&~(dict['a',float]&{b:~int})<=~{a:None}|~dict['a','a']",
    ),
    # Under a full match the two ends of the text are word boundaries before and
    # after a word character.
    pytest.param(
        "equivalent",
        Annotated[str, Regex(r"\bab\b")],
        Annotated[str, Regex("ab")],
        id="json:\\bab\\b==ab",
    ),
    # A repetition of a repetition between the ends keeps the empty word, which
    # `\B` admits where the text ends and the word character `a` refuses.
    pytest.param(
        "refutes",
        Annotated[str, Regex(r"(?:a+)?\B")],
        nothing,
        id="json:(?:a+)?\\B-is-inhabited",
    ),
    pytest.param(
        "equivalent",
        Annotated[str, Regex(r"(?:a+)?\B")],
        Annotated[str, Regex("")],
        id="json:(?:a+)?\\B==empty-pattern",
    ),
    # One question through a complement's meet, where its plain spelling decides.
    pytest.param(
        "refutes",
        anything,
        complement(_JSON),
        id="json:anything!<=~json",
    ),
    pytest.param(
        "empty",
        intersection(_JSON, complement(union(_JSON, str))),
        None,
        id="json:empty:json&~(json|str)",
        marks=_missed("the question through a complement is asked of a meet"),
    ),
    pytest.param(
        "refutes",
        intersection(_JSON, _ARRAY),
        nothing,
        id="json:json&array-is-inhabited",
    ),
    pytest.param(
        "refutes",
        intersection(_JSON, complement(_ARRAY)),
        nothing,
        id="json:json&~array-is-inhabited",
    ),
    pytest.param(
        "refutes",
        _INTEGER,
        complement(int),
        id="json:integer!<=~int",
    ),
    pytest.param(
        "refutes",
        anything,
        union(complement(_JSON), int),
        id="json:anything!<=~json|int",
    ),
    pytest.param(
        "refutes",
        intersection(union(float, int), complement(_ARRAY)),
        nothing,
        id="json:number&~array-is-inhabited",
    ),
    pytest.param(
        "subtype",
        _JSON,
        complement(_OUTSIDE),
        id="json:json<=~(~json&{a})",
    ),
    # A meet is commutative and its members are asked in the written order.
    pytest.param(
        "refutes",
        intersection(_NARROWED, _WIDENED),
        nothing,
        id="json:narrowed&widened-is-inhabited",
    ),
    # The members of a meet are ordered by the slots their constants were pooled
    # in, and a meet unfolds the reference held by its first union, so the same
    # meet built two ways is read through the reference one way only. Taking the
    # union holding a reference wherever it sits decides both, and starves the
    # budget of meets the case split builds, so it is not taken.
    pytest.param(
        "refutes",
        intersection(
            union(Literal[True], {"a": int}), union(Literal[1, "", "b"], _JSON)
        ),
        nothing,
        id="json:true-or-record&literals-or-json-is-inhabited",
        marks=_missed(
            "a meet distributes its first union, whose place the pool's order of"
            " its constants decides"
        ),
    ),
    pytest.param(
        "refutes",
        intersection(
            union(Literal[1, "", "b"], _JSON), union(Literal[True], {"a": int})
        ),
        nothing,
        id="json:literals-or-json&true-or-record-is-inhabited",
    ),
    # A guard written once per branch splits a set by another, and the two
    # branches are the set they split, however the splitting set is spelled.
    pytest.param(
        "subtype",
        _JSON,
        union(intersection(_JSON, _NOT_BOOL), intersection(_JSON, bool)),
        id="json:json<=its-split-by-bool",
    ),
    pytest.param(
        "subtype",
        _ARRAY,
        union(
            intersection(_ARRAY, complement(list[int])), intersection(_ARRAY, list[int])
        ),
        id="json:array<=its-split-by-list[int]",
    ),
    pytest.param(
        "subtype",
        Annotated[int, at.Ge(0)],
        union(
            intersection(Annotated[int, at.Ge(0)], complement(_NUMBER)),
            intersection(Annotated[int, at.Ge(0)], _NUMBER),
        ),
        id="json:ge0<=its-split-by-number",
    ),
    # Two objects guarded to their kind, each a recursive record with a tag of
    # its own: the two meets share no value, and the subject holds one.
    pytest.param(
        "refutes",
        intersection(
            _OBJECT,
            union(
                complement(_OBJECT),
                recursive(
                    lambda r: Validator(
                        {"tag": Literal["t0"], "kids?": [r], str: _JSON}
                    )
                ),
            ),
        ),
        intersection(
            _OBJECT,
            union(
                complement(_OBJECT),
                recursive(
                    lambda r: Validator(
                        {"tag": Literal["t1"], "kids?": [r], str: _JSON}
                    )
                ),
            ),
        ),
        id="json:guarded-t0!<=guarded-t1",
    ),
]


@pytest.mark.parametrize(
    ("operation", "left", "right"), _DECIDED + _LEDGERED + _TRANSLATED
)
def test_decision_decides_true_relations(
    operation: str, left: object, right: object
) -> None:
    _check(operation, left, right)


# Enum widening at the sizes a real annotation reaches. A literal union is the
# shape that grows: an error-code table, a currency list, a set of tags.


def _codes(members: int, *, extra: bool = False, backwards: bool = False) -> Validator:
    """Build a literal union of `members` codes, pooling its own constants.

    `backwards` writes the same codes in the other order, so the table's own
    pool numbers them the other way round: two tables that denote one set and
    agree on nothing about where each constant sits.
    """
    codes = [f"code_{index:05d}" for index in range(members)]
    if extra:
        codes.append("extra")
    if backwards:
        codes.reverse()
    return union(*[Validator(code) for code in codes])


@pytest.mark.parametrize("members", [8, 64, 256, 512, 1024, 4096])
def test_widening_a_table_is_decided_at_the_sizes_a_table_reaches(
    members: int,
) -> None:
    # Two tables written out separately, which is how a codebase with a schema in
    # two modules has them: neither is built from the other, and the strings are
    # built at run time, so no two are one object. A constant is a value rather
    # than an object, so they pool together anyway, the narrow union's branches
    # are branches of the wide one, and containment settles it without
    # distributing. Pooled by object rather than by value, the two largest sizes
    # would spend the decision budget instead.
    assert _codes(members).is_subtype_of(_codes(members, extra=True))


@pytest.mark.parametrize("members", [8, 64, 256, 512, 1024, 4096])
def test_a_table_is_decided_however_it_was_written(members: int) -> None:
    # The same two tables with one side written in reverse. Relating two
    # validators interns one pool into the other, which numbers the second
    # table's constants by the order the first met them -- for a table written
    # backwards, its own reverse -- and the union is rebuilt in that order. A
    # table is decided as a set only in canonical order, so the transform that
    # renumbers a member set leaves it canonical; left out of order, these four
    # would fall back to distributing one table against the other.
    forward = _codes(members)
    same_backwards = _codes(members, backwards=True)
    wider_backwards = _codes(members, extra=True, backwards=True)
    assert forward.relation_to(same_backwards) == "subset"
    assert same_backwards.relation_to(forward) == "subset"
    assert forward.relation_to(wider_backwards) == "subset"
    assert wider_backwards.relation_to(forward) == "not_subset"


@pytest.mark.parametrize("members", [8, 64, 256, 512, 1024, 4096])
def test_a_table_missing_a_member_refutes_at_every_size(members: int) -> None:
    # The other direction of the same pair, and the one a shape rule cannot
    # reach: the wider table holds a code the narrow one does not, so a value of
    # one is outside the other and the inclusion is *refuted* rather than left
    # unproven. A union of literals denotes a finite set, and a finite set is
    # decided by membership in both directions -- distributing one table against
    # the other would spend the budget on the product and answer "undecided"
    # from a thousand members up.
    narrow = _codes(members)
    wider = _codes(members, extra=True)
    assert wider.relation_to(narrow) == "not_subset"
    # And the refutation is a statement about a value: the code the narrow table
    # lacks is one the wide table admits.
    assert wider.is_valid("extra")
    assert not narrow.is_valid("extra")


def test_a_record_missing_a_required_key_refutes_however_open_it_is() -> None:
    # A `TypedDict` is open -- the typing spec admits keys it does not declare --
    # so this is the shape of every relation between two of them, and the answer
    # is a refutation rather than a decline: a clause governs the keys a value
    # carries and requires none, so the subject holds a value without the key
    # the supertype requires, and that value is one the supertype rejects.
    #
    # Decided by a rule, which is what the ledger asks. The set representation
    # decides it too, by lowering both records, and takes two hundred times
    # longer to say the same thing.
    # Built from a comprehension rather than written out: the row is about the
    # width, and a checker wants a literal where the fields are a type.
    fields = {f"f{i}": int for i in range(8)}
    complete = Validator(TypedDict("Complete", fields))  # ty: ignore[invalid-argument-type]
    missing = Validator(
        TypedDict("Missing", dict(list(fields.items())[:-1]))  # ty: ignore[invalid-argument-type]
    )
    assert missing.relation_to(complete) == "not_subset"
    # The refutation is a statement about a value: the mapping the subject holds
    # without the key, which the supertype refuses.
    value = {f"f{i}": i for i in range(7)}
    assert missing.is_valid(value)
    assert not complete.is_valid(value)
    # A mapping is the same shape with no declared field at all, and it holds
    # the value with no keys, which every record requiring one refuses.
    assert Validator(dict[str, int]).relation_to(complete) == "not_subset"
    assert Validator(dict[str, int]).is_valid({})


@pytest.mark.parametrize("members", [1024, 4096])
def test_widening_a_table_by_a_member_is_decided_at_every_size(members: int) -> None:
    # The same relation with the wide table built *from* the narrow one, which
    # reaches containment through sharing the schema node rather than through
    # sharing the constants under it.
    narrow = _codes(members)
    assert narrow.is_subtype_of(union(narrow, Validator("extra")))


# The integer-discreteness rule must fire only where the base is integer-discrete
# and an integer genuinely fails to fit. These controls keep it from over-firing:
# a dense float base, an interval that still contains an integer, and an inclusive
# bound that lands on one.
_NON_EMPTY = [
    pytest.param(Annotated[float, at.Gt(0), at.Lt(1)], id="float-open-(0,1)"),
    # Across a meet, and on a boolean base, the rule must still fire only where an
    # integer genuinely fails to fit.
    pytest.param(
        intersection(_GT0, Annotated[int, at.Lt(2)]),
        id="int-open-(0,2)-across-a-meet-has-1",
    ),
    pytest.param(
        intersection(Annotated[float, at.Gt(0)], Annotated[float, at.Lt(1)]),
        id="float-open-(0,1)-across-a-meet",
    ),
    pytest.param(Annotated[bool, at.Ge(0), at.Le(1)], id="bool-closed-[0,1]-has-both"),
    pytest.param(Annotated[int, at.Gt(0), at.Lt(2)], id="int-open-(0,2)-has-1"),
    pytest.param(Annotated[int, at.Gt(0.5), at.Lt(1.5)], id="int-open-(0.5,1.5)-has-1"),
    pytest.param(Annotated[int, at.Ge(0), at.Le(0)], id="int-closed-[0,0]-has-0"),
    pytest.param(Annotated[int, at.Ge(0), at.Lt(1)], id="int-half-[0,1)-has-0"),
    pytest.param(Annotated[int, at.Gt(0), at.Le(1)], id="int-half-(0,1]-has-1"),
]


@pytest.mark.parametrize("spec", _NON_EMPTY)
def test_integer_discreteness_rule_does_not_over_fire(spec: object) -> None:
    # A false `is_empty` would be unsound: it would license dropping a value the
    # schema in fact admits. The float case is the key guard — the rule generalizes
    # to dense bases only at the cost of soundness.
    assert not Validator(spec).is_empty()


# --- Frontend integrity: non-value objects are rejected -----------------------
#
# A construct carrying no runtime value is rejected, not interned as a literal
# that silently accepts almost nothing.

_T = TypeVar("_T")

_REJECTED = [
    pytest.param(_T, id="TypeVar"),
    pytest.param(list[_T], id="list[TypeVar]"),
    pytest.param(Final, id="Final"),
    pytest.param(ClassVar, id="ClassVar"),
    pytest.param(Union, id="bare-Union"),
    pytest.param(Optional, id="bare-Optional"),
    pytest.param(Literal, id="bare-Literal"),
]


@pytest.mark.parametrize("schema", _REJECTED)
def test_frontend_rejects_non_value_objects(schema: object) -> None:
    with pytest.raises((TypeError, ValueError, NotImplementedError)):
        Validator(schema)


def test_value_literals_still_build() -> None:
    # The rejection above does not over-reach: genuine constant values still build
    # as typed literals.
    class Color(enum.Enum):
        RED = 1

    sentinel = object()
    assert Validator(1).is_valid(1)
    assert Validator("a").is_valid("a")
    assert Validator(Color.RED).is_valid(Color.RED)
    assert Validator(sentinel).is_valid(sentinel)
    assert not Validator(sentinel).is_valid(object())


# --- Finite-universe soundness fuzzer ----------------------------------------

_RECURSIVE_FAMILY = [
    _RECURSIVE,
    recursive(lambda t: union(int, [t])),
    recursive(lambda t: union(None, bool, int, str, [t], {str: t})),
]
# Every atom the decided fragment reaches, so the search covers the components
# that carry it: a bare container names its kind, a builtin subclass narrows one,
# a pattern is a regular language and a step an integer set. An atom the search
# never draws is a component the soundness direction is never asked about, and
# an unsoundness there is one that reaches a release.
_atoms = st.sampled_from(
    [
        int,
        str,
        bool,
        float,
        bytes,
        None,
        _GE0,
        _GE0_LE10,
        0,
        1,
        # Spelled `Literal["a"]` rather than `"a"`: a bare string inside a
        # generic is a forward reference, which the frontend refuses, and the
        # composers below put every atom inside one.
        Literal["a"],
        list,
        tuple,
        set,
        frozenset,
        dict,
        _MyInt,
        _MyStr,
        Annotated[str, Regex("a+")],
        Annotated[int, at.MultipleOf(2)],
        *_RECURSIVE_FAMILY,
    ]
)


def _compose(children: st.SearchStrategy) -> st.SearchStrategy:
    pair = st.tuples(children, children)
    return st.one_of(
        children.map(lambda c: [c]),
        children.map(lambda c: set[c]),
        children.map(lambda c: frozenset[c]),
        pair.map(lambda p: tuple[p[0], p[1]]),  # ty: ignore[invalid-type-form]
        children.map(lambda c: {str: c}),
        pair.map(lambda p: {str: p[0], int: p[1]}),  # multi-clause mapping
        pair.map(lambda p: {"a": p[0], str: p[1]}),  # record mixed with a catch-all
        pair.map(lambda p: {"a": p[0], "b": p[1], str: p[0]}),  # two fields + catch-all
        pair.map(lambda p: union(p[0], p[1])),
        pair.map(lambda p: intersection(p[0], p[1])),
        children.map(complement),
    )


_schemas = st.recursive(_atoms, _compose, max_leaves=8)

# What the emptiness law is asked about. Drawn from `_schemas` alone, a schema
# the procedure proves empty arrives once in five hundred cases -- the
# recursion wraps almost every draw in a container, and a container over an
# empty element is not empty -- so the law asserts about nothing on the dev
# profile's hundred. A meet of two drawn schemas is empty nine times in ten,
# and it is where a proof of emptiness is decided rather than folded:
# disjoint kinds, a bound against a bound, a record against a mapping.
_meets = st.tuples(_schemas, _schemas).map(lambda pair: intersection(pair[0], pair[1]))


@given(left=_schemas, right=_schemas)
def test_subtype_claims_hold_on_the_universe(left: object, right: object) -> None:
    # The soundness direction of the differential: a claimed subtype accepts no
    # universe value the claimed supertype rejects. A violation is a real
    # unsoundness, not a conservatism.
    compiled = Validator(left)
    if compiled.is_subtype_of(right):
        assert _accepted(left) <= _accepted(right)


@given(spec=st.one_of(_schemas, _meets))
def test_emptiness_claims_hold_on_the_universe(spec: object) -> None:
    # A schema reported empty accepts nothing in the universe. The converse does
    # not hold over a finite universe, so only this sound direction is asserted.
    compiled = Validator(spec)
    if compiled.is_empty():
        assert not _accepted(spec)


class _Partial:
    """A value whose equality admits some others of its own class."""

    __slots__ = ("accepts", "tag")

    def __init__(self, tag: str, accepts: frozenset[str]) -> None:
        self.tag, self.accepts = tag, accepts

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Partial) and other.tag in self.accepts

    def __hash__(self) -> int:
        return 0


# THEORY: a-constant-is-the-values-equal-to-it
def test_a_literal_is_not_always_a_singleton() -> None:
    """Why the entry above cannot be closed by comparing the two constants.

    `Literal` admits any constant, not only the types the typing spec allows in
    `Literal[...]`, and equality is the value's own. Two such literals can share
    a member while neither contains the other, so "not mutually subtypes" does
    not imply "disjoint" — and a rule reading it that way would report a meet
    empty that a value belongs to, which `is_empty` must never do.
    """
    left = Validator(_Partial("a", frozenset({"a"})))
    right = Validator(_Partial("b", frozenset({"b"})))
    shared = _Partial("s", frozenset({"a", "b"}))

    assert left.is_valid(shared)
    assert right.is_valid(shared)
    assert not left.is_subtype_of(right)
    assert not right.is_subtype_of(left)

    # Sound, and the reason the narrow rule must be gated on the
    # constant's type rather than on the subtype relation.
    assert not intersection(left, right).is_empty()


def test_a_spec_literal_type_is_not_a_singleton_either() -> None:
    """Why narrowing the rule to `Literal[...]`'s own types does not rescue it.

    An enum member is a constant the typing spec admits in `Literal[...]`, and
    its equality is ordinary user code. A rule that fired on "both constants
    have a type `Literal[...]` admits, and they are distinct" would report this
    meet empty, and a value belongs to it.
    """

    class Loose(enum.Enum):
        A = "a"
        B = "b"

        def __eq__(self, other: object) -> bool:
            return isinstance(other, Loose)

        def __hash__(self) -> int:
            return 0

    left, right = Validator(Loose.A), Validator(Loose.B)
    assert Loose.A is not Loose.B
    assert left.is_valid(Loose.B)
    assert right.is_valid(Loose.A)
    assert not intersection(left, right).is_empty()
