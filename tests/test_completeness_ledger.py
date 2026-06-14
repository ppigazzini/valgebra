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

from typing import Annotated, Final, TypeVar

import annotated_types as at
import pytest
from hypothesis import given
from hypothesis import strategies as st

from valgebra import (
    complement,
    intersect,
    lazy,
    union,
    validator,
)

# A recursive schema, reused below to record a reflexivity hole.
_RECURSIVE = lazy(lambda t: union(None, {"value": int, "next": t}))

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
]


def _accepted(spec: object) -> frozenset[int]:
    """Return the universe indices a schema accepts -- its denotation on `U`."""
    compiled = validator(spec)
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
    compiled = validator(left)
    if operation == "subtype":
        assert compiled.is_subtype(right)
    elif operation == "equivalent":
        assert compiled.equivalent(right)
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
    pytest.param("empty", intersect(int, str), None, id="empty:int&str"),
    pytest.param("equivalent", union(bool, int), int, id="bool|int==int"),
    pytest.param("equivalent", intersect(int, int), int, id="int&int==int"),
]

# Known holes (completeness report, refinement fragment). Each is a true relation
# the procedure declines because refinement constraints are opaque to it.
_LEDGERED = [
    pytest.param(
        "subtype",
        _GE0,
        int,
        id="refine:Ge(0)<=int",
        marks=pytest.mark.xfail(strict=True, reason="refinement bounds are opaque"),
    ),
    pytest.param(
        "subtype",
        _GE0_LE10,
        _GE0,
        id="refine:Ge(0)Le(10)<=Ge(0)",
        marks=pytest.mark.xfail(strict=True, reason="refinement bounds are opaque"),
    ),
    pytest.param(
        "empty",
        _GE10_LE0,
        None,
        id="empty:Ge(10)Le(0)",
        marks=pytest.mark.xfail(strict=True, reason="unsatisfiable bounds undetected"),
    ),
    pytest.param(
        "empty",
        intersect(_GE0, Annotated[int, at.Lt(0)]),
        None,
        id="empty:Ge(0)&Lt(0)",
        marks=pytest.mark.xfail(strict=True, reason="contradictory bounds undetected"),
    ),
    pytest.param(
        "subtype",
        intersect(_RECURSIVE, union(int, str)),
        intersect(_RECURSIVE, union(int, str)),
        id="reflexive:intersect(rec,union)",
        marks=pytest.mark.xfail(
            strict=True,
            reason="reflexivity of intersect(recursive, union) across merged pools",
        ),
    ),
]


@pytest.mark.parametrize(("operation", "left", "right"), _DECIDED + _LEDGERED)
def test_decision_decides_true_relations(
    operation: str, left: object, right: object
) -> None:
    _check(operation, left, right)


# --- Silent-acceptance ledger: the frontend must reject non-value objects -----
#
# A construct carrying no runtime value must be rejected, not interned as a
# literal that silently accepts almost nothing. These assert the intended
# rejection; the current build succeeds, so they are ledgered misses.

_T = TypeVar("_T")

_REJECTED = [
    pytest.param(
        _T,
        id="TypeVar",
        marks=pytest.mark.xfail(strict=True, reason="interned as Literal[~T]"),
    ),
    pytest.param(
        list[_T],
        id="list[TypeVar]",
        marks=pytest.mark.xfail(strict=True, reason="interned as list[Literal[~T]]"),
    ),
    pytest.param(
        Final,
        id="Final",
        marks=pytest.mark.xfail(strict=True, reason="interned as Literal[Final]"),
    ),
]


@pytest.mark.parametrize("schema", _REJECTED)
def test_frontend_rejects_non_value_objects(schema: object) -> None:
    with pytest.raises((TypeError, ValueError, NotImplementedError)):
        validator(schema)


# --- Finite-universe soundness fuzzer ----------------------------------------

_atoms = st.sampled_from(
    [int, str, bool, float, bytes, None, _GE0, _GE0_LE10, 0, 1, "a"]
)


def _compose(children: st.SearchStrategy) -> st.SearchStrategy:
    pair = st.tuples(children, children)
    return st.one_of(
        children.map(lambda c: [c]),
        children.map(lambda c: {str: c}),
        pair.map(lambda p: union(p[0], p[1])),
        pair.map(lambda p: intersect(p[0], p[1])),
        children.map(complement),
    )


_schemas = st.recursive(_atoms, _compose, max_leaves=8)


@given(left=_schemas, right=_schemas)
def test_subtype_claims_hold_on_the_universe(left: object, right: object) -> None:
    # The soundness direction of the differential: a claimed subtype accepts no
    # universe value the claimed supertype rejects. A violation is a real
    # unsoundness, not a conservatism.
    try:
        compiled = validator(left)
    except (ValueError, TypeError, NotImplementedError, RecursionError):
        return
    if compiled.is_subtype(right):
        assert _accepted(left) <= _accepted(right)


@given(spec=_schemas)
def test_emptiness_claims_hold_on_the_universe(spec: object) -> None:
    # A schema reported empty accepts nothing in the universe. The converse does
    # not hold over a finite universe, so only this sound direction is asserted.
    try:
        compiled = validator(spec)
    except (ValueError, TypeError, NotImplementedError, RecursionError):
        return
    if compiled.is_empty():
        assert not _accepted(spec)
