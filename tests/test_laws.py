"""Property tests for the Boolean-algebra laws, checked as membership equivalence.

Two schemas are equivalent when they accept exactly the same values. Each law is
checked over generated values drawn from a strategy spanning scalars and nested
containers, seeded with a curated spread that pins the atom boundaries (the
int/bool/float distinctions and the typed-singleton literals). Generating the
witness values, rather than iterating a fixed list, means two schemas that merely
agree on a handful of constants no longer pass a law: a value that distinguishes
them is searched for. The semantic decision (`is_equivalent`) is cross-checked
against membership separately in the subtyping suite.
"""

from typing import Any, Literal

from hypothesis import given
from hypothesis import strategies as st

from valgebra import (
    Validator,
    anything,
    complement,
    intersection,
    nothing,
    union,
)

# Schema specs: each is an annotation or native form valgebra can compile. The
# container and sequence forms make the laws (and the simplifier) recurse into a
# Seq/Coll/Mapping node, not just scalars.
ATOM_SCHEMAS = [
    int,
    float,
    str,
    bool,
    None,
    Literal[0],
    Literal["x"],
    Literal[True],
    list[int],
    set[int],
    dict[str, int],
    tuple[int, str],
    tuple[int, ...],
    tuple[str, int, ...],  # ty: ignore[invalid-type-form]  # a prefix-plus-tail tuple
    [int, str, ...],  # a prefix-plus-tail list
]

# A curated spread that pins the atom and container boundaries; generated values
# (below) widen the search beyond it on every example.
VALUES = [
    0,
    1,
    -1,
    True,
    False,
    1.0,
    0.0,
    "x",
    "",
    "y",
    None,
    3.14,
    [1, 2],
    [1, "a"],
    [1, "a", 2, 3],
    [],
    {1, 2},
    {"k": 1},
    {},
    (1, "a"),
    (1, 2, 3),
]

schemas = st.sampled_from(ATOM_SCHEMAS)

# Hashable leaves for set members and dict keys.
_hashable = st.one_of(
    st.integers(min_value=-3, max_value=3),
    st.booleans(),
    st.text(max_size=2),
    st.none(),
)
# Arbitrary Python values spanning scalars and nested containers — the witnesses
# the laws are checked over, on top of the curated boundary spread.
_value = st.recursive(
    st.one_of(
        st.integers(min_value=-3, max_value=3),
        st.booleans(),
        st.floats(allow_nan=False, allow_infinity=False),
        st.text(max_size=2),
        st.none(),
    ),
    lambda children: st.one_of(
        st.lists(children, max_size=3),
        st.sets(_hashable, max_size=3),
        st.frozensets(_hashable, max_size=3),
        st.dictionaries(st.text(max_size=2), children, max_size=2),
        st.tuples(children),
        st.tuples(children, children),
    ),
    max_leaves=5,
)
value_lists = st.lists(_value, max_size=6)


def equivalent(left: Validator, right: Validator, extra: list[object]) -> bool:
    values = [*VALUES, *extra]
    return all(left.is_valid(v) == right.is_valid(v) for v in values)


@given(a=schemas, b=schemas, vals=value_lists)
def test_union_commutativity(a: object, b: object, vals: list[object]) -> None:
    assert equivalent(union(a, b), union(b, a), vals)


@given(a=schemas, b=schemas, vals=value_lists)
def test_intersect_commutativity(a: object, b: object, vals: list[object]) -> None:
    assert equivalent(intersection(a, b), intersection(b, a), vals)


@given(a=schemas, b=schemas, c=schemas, vals=value_lists)
def test_union_associativity(
    a: object, b: object, c: object, vals: list[object]
) -> None:
    assert equivalent(union(union(a, b), c), union(a, union(b, c)), vals)


@given(a=schemas, b=schemas, c=schemas, vals=value_lists)
def test_intersect_associativity(
    a: object, b: object, c: object, vals: list[object]
) -> None:
    assert equivalent(
        intersection(intersection(a, b), c), intersection(a, intersection(b, c)), vals
    )


@given(a=schemas, vals=value_lists)
def test_idempotence(a: object, vals: list[object]) -> None:
    assert equivalent(union(a, a), Validator(a), vals)
    assert equivalent(intersection(a, a), Validator(a), vals)


@given(a=schemas, b=schemas, vals=value_lists)
def test_absorption(a: object, b: object, vals: list[object]) -> None:
    assert equivalent(union(a, intersection(a, b)), Validator(a), vals)
    assert equivalent(intersection(a, union(a, b)), Validator(a), vals)


@given(a=schemas, vals=value_lists)
def test_identities(a: object, vals: list[object]) -> None:
    assert equivalent(union(a, nothing), Validator(a), vals)
    assert equivalent(intersection(a, anything), Validator(a), vals)
    assert equivalent(union(a, anything), anything, vals)
    assert equivalent(intersection(a, nothing), nothing, vals)


@given(a=schemas, vals=value_lists)
def test_double_negation(a: object, vals: list[object]) -> None:
    assert equivalent(complement(complement(a)), Validator(a), vals)


@given(a=schemas, b=schemas, vals=value_lists)
def test_de_morgan(a: object, b: object, vals: list[object]) -> None:
    assert equivalent(
        complement(union(a, b)),
        intersection(complement(a), complement(b)),
        vals,
    )
    assert equivalent(
        complement(intersection(a, b)),
        union(complement(a), complement(b)),
        vals,
    )


@given(a=schemas, b=schemas, c=schemas, vals=value_lists)
def test_distributivity(a: object, b: object, c: object, vals: list[object]) -> None:
    assert equivalent(
        union(a, intersection(b, c)),
        intersection(union(a, b), union(a, c)),
        vals,
    )
    assert equivalent(
        intersection(a, union(b, c)),
        union(intersection(a, b), intersection(a, c)),
        vals,
    )


@given(a=schemas, b=schemas, c=schemas, vals=value_lists)
def test_simplify_preserves_acceptance(
    a: object, b: object, c: object, vals: list[object]
) -> None:
    original = complement(union(a, intersection(b, complement(c))))
    assert equivalent(original, original.simplify(), vals)


@given(a=schemas, b=schemas, vals=value_lists)
def test_simplify_is_idempotent_on_acceptance(
    a: object, b: object, vals: list[object]
) -> None:
    once = intersection(a, complement(b)).simplify()
    twice = once.simplify()
    assert equivalent(once, twice, vals)


def test_simplify_decides_the_complement_laws() -> None:
    # The decision the conservative canonicalizer could not make: a schema and
    # its complement collapse, and provably disjoint types collapse.
    assert repr(intersection(int, complement(int)).simplify()) == "nothing"
    assert repr(union(int, complement(int)).simplify()) == "anything"
    assert repr(intersection(int, str).simplify()) == "nothing"
    assert repr(union(complement(int), complement(str)).simplify()) == "anything"


def test_simplify_leaves_the_gradual_any_uncollapsed() -> None:
    # `Any` is unchecked, not the lattice top, so the complement laws skip it.
    assert repr(intersection(Any, complement(Any)).simplify()) != "nothing"
    assert repr(union(Any, complement(Any)).simplify()) != "anything"
