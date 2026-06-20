"""Subtyping, equivalence, and emptiness checked against real membership.

`is_subtype_of(a, b)` claims every value of `a` is a value of `b`. The fuzzer holds
that claim to actual membership: for generated values on top of a curated corpus,
a claimed subtype never accepts a value its supertype rejects, and equivalent
schemas accept exactly the same values. Drawing the witness values, rather than
iterating a fixed list, widens the soundness search on every example. Soundness
is the property under test; the decision is intentionally conservative (it may
answer ``False`` for a true relation it cannot prove), so completeness is not
asserted.
"""

from typing import Annotated, Literal, TypedDict

import annotated_types as at
from hypothesis import given
from hypothesis import strategies as st

from valgebra import Validator, complement, intersection, recursive, union


class _Point(TypedDict):
    x: int
    y: int


class _Animal:
    pass


class _Dog(_Animal):
    pass


# Two structurally identical recursive linked-list schemas (equivalent).
_LINKED = recursive(lambda t: union(None, {"value": int, "next": t}))
_LINKED_TWIN = recursive(lambda t: union(None, {"value": int, "next": t}))


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

# Hashable leaves for set members and dict keys.
_hashable = st.one_of(
    st.integers(min_value=-3, max_value=3),
    st.booleans(),
    st.text(max_size=2),
    st.none(),
)
# Arbitrary Python values spanning scalars and nested containers, generated on
# top of the curated corpus to widen the soundness search.
_value = st.recursive(
    st.one_of(
        st.integers(min_value=-3, max_value=3),
        st.booleans(),
        st.floats(allow_nan=False, allow_infinity=False),
        st.text(max_size=2),
        st.binary(max_size=2),
        st.none(),
    ),
    lambda children: st.one_of(
        st.lists(children, max_size=3),
        st.sets(_hashable, max_size=3),
        st.dictionaries(st.text(max_size=2), children, max_size=2),
        st.tuples(children),
        st.tuples(children, children),
    ),
    max_leaves=5,
)
value_lists = st.lists(_value, max_size=6)


def accepts(schema: object, extra: list[object]) -> list[bool]:
    compiled = Validator(schema)
    return [compiled.is_valid(value) for value in (*VALUES, *extra)]


@given(a=specs, b=specs, vals=value_lists)
def test_subtype_is_sound(a: object, b: object, vals: list[object]) -> None:
    # A claimed subtype never accepts a value the supertype rejects.
    if Validator(a).is_subtype_of(b):
        a_accepts, b_accepts = accepts(a, vals), accepts(b, vals)
        assert all(
            b_in for a_in, b_in in zip(a_accepts, b_accepts, strict=True) if a_in
        )


@given(a=specs)
def test_subtype_is_reflexive(a: object) -> None:
    assert Validator(a).is_subtype_of(a)


@given(a=specs, b=specs)
def test_equivalent_is_mutual_subtyping(a: object, b: object) -> None:
    left = Validator(a)
    assert left.is_equivalent(b) == (
        left.is_subtype_of(b) and Validator(b).is_subtype_of(a)
    )


@given(a=specs, b=specs, vals=value_lists)
def test_equivalent_implies_equal_acceptance(
    a: object, b: object, vals: list[object]
) -> None:
    if Validator(a).is_equivalent(b):
        assert accepts(a, vals) == accepts(b, vals)


@given(a=specs, b=specs, vals=value_lists)
def test_empty_intersection_accepts_nothing(
    a: object, b: object, vals: list[object]
) -> None:
    meet = intersection(a, b)
    if meet.is_empty():
        assert not any(meet.is_valid(value) for value in (*VALUES, *vals))


def test_known_relations() -> None:
    assert Validator(bool).is_subtype_of(int)  # bool is a subtype of int
    assert not Validator(int).is_subtype_of(bool)
    assert Validator(list[bool]).is_subtype_of(list[int])
    assert not Validator(list[int]).is_subtype_of(set[int])  # distinct kinds
    assert union(bool, int).is_equivalent(int)  # bool | int is just int
    assert intersection(int, complement(int)).is_empty()
    assert not Validator(int).is_empty()


def test_complement_reflexivity_and_contravariance() -> None:
    # Regression: a complement or refinement must be a subtype of itself. This
    # needs both the contravariant complement rule and the identity-interning
    # pool merge (so a shared constant keeps one index across the comparison).
    assert complement(Literal[0]).is_subtype_of(complement(Literal[0]))
    assert Validator(Annotated[int, at.Ge(0)]).is_subtype_of(Annotated[int, at.Ge(0)])
    assert complement(Annotated[int, at.Ge(0)]).is_subtype_of(
        complement(Annotated[int, at.Ge(0)])
    )
    # Contravariance, checked against membership: a non-int is never a bool.
    assert Validator(complement(int)).is_subtype_of(complement(bool))
    assert not Validator(complement(bool)).is_subtype_of(complement(int))


def test_instance_and_literal_relations() -> None:
    # A literal is a subtype of any schema that admits its value.
    assert Validator(Literal["red"]).is_subtype_of(str)
    assert not Validator(str).is_subtype_of(Literal["red"])
    assert Validator(Literal["red"]).is_subtype_of(Literal["red", "green"])
    assert Validator(list[Literal["red"]]).is_subtype_of(list[str])  # nested
    # A class is a subtype of another exactly when it is a subclass.
    assert Validator(_Dog).is_subtype_of(_Animal)
    assert not Validator(_Animal).is_subtype_of(_Dog)
    assert Validator(_Dog).is_equivalent(_Dog)


def test_recursive_subtyping_is_coinductive() -> None:
    # Two structurally identical recursive types are equivalent.
    assert _LINKED.is_subtype_of(_LINKED_TWIN)
    assert _LINKED.is_equivalent(_LINKED_TWIN)
    # A bool-valued recursive list is a subtype of an int-valued one, not reverse.
    bool_list = recursive(lambda t: union(None, {"value": bool, "next": t}))
    int_list = recursive(lambda t: union(None, {"value": int, "next": t}))
    assert bool_list.is_subtype_of(int_list)
    assert not int_list.is_subtype_of(bool_list)


def test_recursion_composed_with_literals_and_instances() -> None:
    # Regression: the coinductive subtyping must compose with the leaf oracle
    # (literal-by-membership, instance-by-subclass) inside a recursive type.
    a = recursive(lambda t: union(Literal["leaf"], {"value": int, "next": t}))
    b = recursive(lambda t: union(Literal["leaf"], {"value": int, "next": t}))
    assert a.is_equivalent(b)
    bool_valued = recursive(
        lambda t: union(Literal["leaf"], {"value": bool, "next": t})
    )
    int_valued = recursive(lambda t: union(Literal["leaf"], {"value": int, "next": t}))
    assert bool_valued.is_subtype_of(int_valued)
    assert not int_valued.is_subtype_of(bool_valued)


def test_is_empty_detects_uninhabited_recursion() -> None:
    # A mandatory self-reference with no base case has no finite inhabitant.
    assert recursive(lambda t: {"value": int, "next": t}).is_empty()
    # A base case (or an optional self-reference) makes it inhabited.
    assert not recursive(lambda t: union(None, {"value": int, "next": t})).is_empty()
    assert not recursive(lambda t: {"value": int, "next?": t}).is_empty()
    # A list of itself is inhabited by the empty list.
    assert not recursive(lambda t: [t]).is_empty()
