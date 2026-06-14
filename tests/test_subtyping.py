"""Subtyping, equivalence, and emptiness checked against real membership.

`is_subtype(a, b)` claims every value of `a` is a value of `b`. The fuzzer holds
that claim to actual membership: for a corpus of values, a claimed subtype never
accepts a value its supertype rejects, and equivalent schemas accept exactly the
same values. Soundness is the property under test; the decision is intentionally
conservative (it may answer ``False`` for a true relation it cannot prove), so
completeness is not asserted.
"""

from typing import Annotated, Literal, TypedDict

import annotated_types as at
from hypothesis import given
from hypothesis import strategies as st

from valgebra import complement, intersect, lazy, union, validator


class _Point(TypedDict):
    x: int
    y: int


class _Animal:
    pass


class _Dog(_Animal):
    pass


# Two structurally identical recursive linked-list schemas (equivalent).
_LINKED = lazy(lambda t: union(None, {"value": int, "next": t}))
_LINKED_TWIN = lazy(lambda t: union(None, {"value": int, "next": t}))


# Schema specs valgebra compiles, spanning scalars, structural containers, and
# records and mappings (closed dict-literal records, TypedDicts, and dict[K, V]).
SPECS = [
    int,
    str,
    bool,
    float,
    bytes,
    None,
    list[int],
    list[bool],
    list[str],
    set[int],
    set[bool],
    tuple[int, str],
    tuple[bool, str],
    tuple[bool, int, ...],  # ty: ignore[invalid-type-form]  # a bool, then zero or more ints
    [bool, int, ...],  # a list: a bool, then zero or more ints
    [int, int, ...],  # a non-empty list of ints
    dict[str, int],
    dict[str, bool],
    _Point,
    {"x": int},
    {"x": int, "y?": str},
    {"x?": int},
    {"x": bool},
    Literal["red"],
    Literal[5],
    _Animal,
    _Dog,
    _LINKED,
    _LINKED_TWIN,
]

# A value corpus exercising the scalar and container boundaries.
VALUES = [
    0,
    1,
    True,
    False,
    1.0,
    "x",
    "",
    b"y",
    None,
    3.14,
    [1, 2],
    [True],
    [True, 1],
    [True, 1, 2],
    ["a"],
    [],
    {1, 2},
    {True},
    (1, "a"),
    (True, "a"),
    (1, 2),
    {"a": 1},
    {"x": 1},
    {"x": 1, "y": 2},
    {"x": 1, "y": "a"},
    {"x": True},
    {},
    "red",
    5,
    _Animal(),
    _Dog(),
    {"value": 1, "next": None},
    {"value": 1, "next": {"value": 2, "next": None}},
    object(),
]

specs = st.sampled_from(SPECS)


def accepts(schema: object) -> list[bool]:
    compiled = validator(schema)
    return [compiled.is_valid(value) for value in VALUES]


@given(a=specs, b=specs)
def test_subtype_is_sound(a: object, b: object) -> None:
    # A claimed subtype never accepts a value the supertype rejects.
    if validator(a).is_subtype(b):
        a_accepts, b_accepts = accepts(a), accepts(b)
        assert all(
            b_in for a_in, b_in in zip(a_accepts, b_accepts, strict=True) if a_in
        )


@given(a=specs)
def test_subtype_is_reflexive(a: object) -> None:
    assert validator(a).is_subtype(a)


@given(a=specs, b=specs)
def test_equivalent_is_mutual_subtyping(a: object, b: object) -> None:
    left = validator(a)
    assert left.equivalent(b) == (left.is_subtype(b) and validator(b).is_subtype(a))


@given(a=specs, b=specs)
def test_equivalent_implies_equal_acceptance(a: object, b: object) -> None:
    if validator(a).equivalent(b):
        assert accepts(a) == accepts(b)


@given(a=specs, b=specs)
def test_empty_intersection_accepts_nothing(a: object, b: object) -> None:
    meet = intersect(a, b)
    if meet.is_empty():
        assert not any(meet.is_valid(value) for value in VALUES)


def test_known_relations() -> None:
    assert validator(bool).is_subtype(int)  # bool is a subtype of int
    assert not validator(int).is_subtype(bool)
    assert validator(list[bool]).is_subtype(list[int])
    assert not validator(list[int]).is_subtype(set[int])  # distinct kinds
    assert union(bool, int).equivalent(int)  # bool | int is just int
    assert intersect(int, complement(int)).is_empty()
    assert not validator(int).is_empty()


def test_complement_reflexivity_and_contravariance() -> None:
    # Regression: a complement or refinement must be a subtype of itself. This
    # needs both the contravariant complement rule and the identity-interning
    # pool merge (so a shared constant keeps one index across the comparison).
    assert complement(Literal[0]).is_subtype(complement(Literal[0]))
    assert validator(Annotated[int, at.Ge(0)]).is_subtype(Annotated[int, at.Ge(0)])
    assert complement(Annotated[int, at.Ge(0)]).is_subtype(
        complement(Annotated[int, at.Ge(0)])
    )
    # Contravariance, checked against membership: a non-int is never a bool.
    assert validator(complement(int)).is_subtype(complement(bool))
    assert not validator(complement(bool)).is_subtype(complement(int))


def test_instance_and_literal_relations() -> None:
    # A literal is a subtype of any schema that admits its value.
    assert validator(Literal["red"]).is_subtype(str)
    assert not validator(str).is_subtype(Literal["red"])
    assert validator(Literal["red"]).is_subtype(Literal["red", "green"])
    assert validator(list[Literal["red"]]).is_subtype(list[str])  # nested
    # A class is a subtype of another exactly when it is a subclass.
    assert validator(_Dog).is_subtype(_Animal)
    assert not validator(_Animal).is_subtype(_Dog)
    assert validator(_Dog).equivalent(_Dog)


def test_recursive_subtyping_is_coinductive() -> None:
    # Two structurally identical recursive types are equivalent.
    assert _LINKED.is_subtype(_LINKED_TWIN)
    assert _LINKED.equivalent(_LINKED_TWIN)
    # A bool-valued recursive list is a subtype of an int-valued one, not reverse.
    bool_list = lazy(lambda t: union(None, {"value": bool, "next": t}))
    int_list = lazy(lambda t: union(None, {"value": int, "next": t}))
    assert bool_list.is_subtype(int_list)
    assert not int_list.is_subtype(bool_list)


def test_recursion_composed_with_literals_and_instances() -> None:
    # Regression: the coinductive subtyping must compose with the leaf oracle
    # (literal-by-membership, instance-by-subclass) inside a recursive type.
    a = lazy(lambda t: union(Literal["leaf"], {"value": int, "next": t}))
    b = lazy(lambda t: union(Literal["leaf"], {"value": int, "next": t}))
    assert a.equivalent(b)
    bool_valued = lazy(lambda t: union(Literal["leaf"], {"value": bool, "next": t}))
    int_valued = lazy(lambda t: union(Literal["leaf"], {"value": int, "next": t}))
    assert bool_valued.is_subtype(int_valued)
    assert not int_valued.is_subtype(bool_valued)


def test_is_empty_detects_uninhabited_recursion() -> None:
    # A mandatory self-reference with no base case has no finite inhabitant.
    assert lazy(lambda t: {"value": int, "next": t}).is_empty()
    # A base case (or an optional self-reference) makes it inhabited.
    assert not lazy(lambda t: union(None, {"value": int, "next": t})).is_empty()
    assert not lazy(lambda t: {"value": int, "next?": t}).is_empty()
    # A list of itself is inhabited by the empty list.
    assert not lazy(lambda t: [t]).is_empty()
