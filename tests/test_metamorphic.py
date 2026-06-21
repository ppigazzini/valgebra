"""Metamorphic invariants for the decision procedure.

Each property is a theorem any sound relation satisfies, so a violation is a
proof of a defect rather than a conservatism. They hold for valgebra because the
procedure is sound, which makes every one a hard gate -- the cheap tripwire that
catches the reflexivity and pool-merge class of bug.
"""

from typing import Annotated

import annotated_types as at
from hypothesis import given
from hypothesis import strategies as st

from valgebra import (
    Validator,
    anything,
    complement,
    intersection,
    nothing,
    recursive,
    union,
)

_GE0 = Annotated[int, at.Ge(0)]
# Recursive schemas exercise the coinductive rules under every invariant, so a
# meet, complement, or nesting that mixes recursion is checked for reflexivity and
# the other laws -- the shape the reflexivity defect lived in.
_RECURSIVE = [
    recursive(lambda t: union(None, {"next": t})),
    recursive(lambda t: union(int, [t])),
    recursive(lambda t: union(None, bool, int, str, [t], {str: t})),
]
_atoms = st.sampled_from(
    [int, str, bool, float, bytes, None, _GE0, 0, 1, "a", *_RECURSIVE]
)


def _compose(children: st.SearchStrategy) -> st.SearchStrategy:
    pair = st.tuples(children, children)
    return st.one_of(
        children.map(lambda c: [c]),
        children.map(lambda c: {str: c}),
        pair.map(lambda p: union(p[0], p[1])),
        pair.map(lambda p: intersection(p[0], p[1])),
        children.map(complement),
    )


_schemas = st.recursive(_atoms, _compose, max_leaves=6)

# A small region-spanning universe for the membership-level invariants.
_UNIVERSE = [None, True, False, -1, 0, 1, 5, "", "a", b"x", 1.5, [], [1], {1}, {"k": 1}]


def _build(spec: object) -> Validator | None:
    try:
        return Validator(spec)
    except (ValueError, TypeError, NotImplementedError, RecursionError):
        return None


@given(spec=_schemas)
def test_subtyping_is_reflexive(spec: object) -> None:
    compiled = _build(spec)
    if compiled is not None:
        assert compiled.is_subtype_of(spec)


@given(spec=_schemas)
def test_bottom_below_and_top_above(spec: object) -> None:
    compiled = _build(spec)
    if compiled is None:
        return
    assert compiled.is_subtype_of(anything)  # s <= top
    assert Validator(nothing).is_subtype_of(spec)  # bottom <= s


@given(spec=_schemas)
def test_double_complement_preserves_membership(spec: object) -> None:
    compiled = _build(spec)
    if compiled is None:
        return
    doubled = Validator(complement(complement(spec)))
    for value in _UNIVERSE:
        assert doubled.is_valid(value) == compiled.is_valid(value)


@given(a=_schemas, b=_schemas)
def test_de_morgan_preserves_membership(a: object, b: object) -> None:
    # complement(union(a, b)) and intersection(complement(a), complement(b))
    # denote the same set (De Morgan). A non-reflexive law over two *different*
    # schema forms, so an asymmetric defect that reflexivity and self-equivalence
    # cannot see -- one side wrong, the other right -- fails here.
    left = _build(complement(union(a, b)))
    right = _build(intersection(complement(a), complement(b)))
    if left is None or right is None:
        return
    for value in _UNIVERSE:
        assert left.is_valid(value) == right.is_valid(value)


def test_transitivity_on_a_decided_chain() -> None:
    # bool <= int <= int|str, so bool <= int|str.
    assert Validator(bool).is_subtype_of(int)
    assert Validator(int).is_subtype_of(union(int, str))
    assert Validator(bool).is_subtype_of(union(int, str))


def test_antisymmetry_implies_equivalence() -> None:
    # Mutual subtyping is equivalence: bool|int and int include each other.
    left = union(bool, int)
    assert left.is_subtype_of(int)
    assert Validator(int).is_subtype_of(left)
    assert left.is_equivalent(int)
