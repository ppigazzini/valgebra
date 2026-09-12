"""Necessary properties of the decision procedure, two of them metamorphic.

The file is named for what all six are, because the four that are not
metamorphic outnumber the two that are and a name is a claim. Chen et al. 2018,
Concept 1: a metamorphic relation relates *multiple* inputs and their outputs,
so a necessary property of one input -- the review's own example is
`-1 <= sin(x) <= 1` -- is not one. Under the old name this file said six MRs and
held two; the tests never changed and do not change here.

The name also collided with something real. The two metamorphic checks
`docs/dev/10-theory.md` cites as load-bearing are *not* in this file -- they are
the JSON path against the object path, and fast mode against explain mode -- and
a reader looking for those found these.

Each property is a theorem any sound relation satisfies, so a violation is a
proof of a defect rather than a conservatism. They hold for valgebra because the
procedure is sound, which makes every one a hard gate -- the cheap tripwire that
catches the reflexivity and pool-merge class of bug.

The two that *are* metamorphic relations:
`test_double_complement_preserves_membership` and
`test_de_morgan_preserves_membership`, each deriving a follow-up schema from a
source one and relating the two runs. The other four judge a single schema or a
fixed example.
"""

from typing import Annotated

import annotated_types as at
from hypothesis import assume, given
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
        # Reject an unbuildable spec through assume so Hypothesis counts it toward
        # the rejection rate rather than silently passing the example.
        assume(False)
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
    # `complement` cancels a complement where it is built, so `~~s` *is* `s` and
    # comparing the two asks nothing. The follow-up schema is one the fold does
    # not reach: absorption, `s | (s & other)`, which denotes `s` and needs a
    # containment to see -- the one lattice law construction leaves standing,
    # because deciding it wherever a schema is built is what the design refuses.
    # The partner is a shape the generator never draws, so it is neither `s` nor
    # its complement: either would collapse the meet and then the join, and the
    # comparison below would be vacuous.
    partner = Validator({"__absorbs__": int})
    doubled = union(spec, intersection(spec, partner))
    # A schema that *denotes* a lattice bound absorbs the partner outright, so
    # for those the fold does reach the respelling and the comparison below is
    # trivially true rather than vacuous. The test is semantic because the
    # absorption is: a recursive reference whose body is the top denotes the top
    # while comparing unequal to it, and it absorbs just the same.
    at_a_bound = compiled.is_empty() or compiled.is_equivalent(anything)
    if not at_a_bound:
        assert doubled != compiled
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
