"""Completeness ledger for the decision procedure.

The decision procedure is sound everywhere and complete on part of its domain.
This module measures the incompleteness directly, rather than letting it hide
behind a soundness-only suite.

Two mechanisms:

- A parametrized ledger of relations that hold by construction. Each case
  asserts the procedure decides a true relation. Cases the procedure decides are
  gates: a regression that makes one conservative fails the build. Cases the
  procedure does not yet decide are marked ``xfail(strict=True)`` -- so they are
  counted as misses, and closing one (making it pass) fails the strict xfail and
  forces its ledger entry to be removed. A new true relation the procedure
  declines, added here without a ledger mark, fails outright.
- A finite-universe soundness fuzzer. For random schema pairs, a claimed subtype
  never accepts a value the supertype rejects across a region-complete universe.

The ledger entries marked xfail are the known holes recorded in the completeness
report; the count pytest prints as "xfailed" is the live conservatism counter.
"""

import enum
from typing import Annotated, ClassVar, Final, Literal, Optional, TypeVar, Union

import annotated_types as at
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from valgebra import (
    Validator,
    complement,
    intersection,
    recursive,
    union,
)

# A recursive schema, reused below to record a reflexivity hole.
_RECURSIVE = recursive(lambda t: union(None, {"value": int, "next": t}))

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


def _check(operation: str, left: object, right: object) -> None:
    compiled = Validator(left)
    if operation == "subtype":
        assert compiled.is_subtype_of(right)
    elif operation == "equivalent":
        assert compiled.is_equivalent(right)
    elif operation == "empty":
        assert compiled.is_empty()
    else:  # pragma: no cover - guards against a typo in a case
        msg = f"unknown operation {operation!r}"
        raise AssertionError(msg)


_DECIDED = [
    pytest.param("subtype", bool, int, id="bool<=int"),
    pytest.param("subtype", 1, int, id="Literal[1]<=int"),
    pytest.param("subtype", int, union(int, str), id="int<=int|str"),
    pytest.param("subtype", list[bool], list[int], id="list[bool]<=list[int]"),
    pytest.param("subtype", {"x": int}, {"x": int, "y?": str}, id="{x}<={x,y?}"),
    pytest.param(
        "subtype", [bool, int, ...], [int, int, ...], id="[bool,int,...]<=[int,int,...]"
    ),
    pytest.param("empty", intersection(int, str), None, id="empty:int&str"),
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
]

# Known decision-completeness misses. None remain: each relation the procedure
# declines on a decidable fragment has been closed. An entry returns here only if
# a future change reintroduces a miss, recorded as a strict expected failure.
_LEDGERED: list[object] = []


@pytest.mark.parametrize(("operation", "left", "right"), _DECIDED + _LEDGERED)
def test_decision_decides_true_relations(
    operation: str, left: object, right: object
) -> None:
    _check(operation, left, right)


# The integer-discreteness rule must fire only where the base is integer-discrete
# and an integer genuinely fails to fit. These controls keep it from over-firing:
# a dense float base, an interval that still contains an integer, and an inclusive
# bound that lands on one.
_NON_EMPTY = [
    pytest.param(Annotated[float, at.Gt(0), at.Lt(1)], id="float-open-(0,1)"),
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
_atoms = st.sampled_from(
    [int, str, bool, float, bytes, None, _GE0, _GE0_LE10, 0, 1, "a", *_RECURSIVE_FAMILY]
)


def _compose(children: st.SearchStrategy) -> st.SearchStrategy:
    pair = st.tuples(children, children)
    return st.one_of(
        children.map(lambda c: [c]),
        children.map(lambda c: {str: c}),
        pair.map(lambda p: {str: p[0], int: p[1]}),  # multi-clause mapping
        pair.map(lambda p: {"a": p[0], str: p[1]}),  # record mixed with a catch-all
        pair.map(lambda p: {"a": p[0], "b": p[1], str: p[0]}),  # two fields + catch-all
        pair.map(lambda p: union(p[0], p[1])),
        pair.map(lambda p: intersection(p[0], p[1])),
        children.map(complement),
    )


_schemas = st.recursive(_atoms, _compose, max_leaves=8)


@given(left=_schemas, right=_schemas)
def test_subtype_claims_hold_on_the_universe(left: object, right: object) -> None:
    # The soundness direction of the differential: a claimed subtype accepts no
    # universe value the claimed supertype rejects. A violation is a real
    # unsoundness, not a conservatism.
    try:
        compiled = Validator(left)
    except (ValueError, TypeError, NotImplementedError, RecursionError):
        # Reject an unbuildable spec through assume so Hypothesis counts it.
        assume(False)
        return
    if compiled.is_subtype_of(right):
        assert _accepted(left) <= _accepted(right)


@given(spec=_schemas)
def test_emptiness_claims_hold_on_the_universe(spec: object) -> None:
    # A schema reported empty accepts nothing in the universe. The converse does
    # not hold over a finite universe, so only this sound direction is asserted.
    try:
        compiled = Validator(spec)
    except (ValueError, TypeError, NotImplementedError, RecursionError):
        # Reject an unbuildable spec through assume so Hypothesis counts it.
        assume(False)
        return
    if compiled.is_empty():
        assert not _accepted(spec)
