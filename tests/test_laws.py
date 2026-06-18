"""Property tests for the Boolean-algebra laws, checked as membership equivalence.

Two schemas are equivalent when they accept exactly the same values. Each law is
checked over a fixed spread of values that exercises the atom boundaries (the
int/bool/float distinctions and the typed-singleton literals).
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
    simplify,
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

# A spread of values exercising the atom and container boundaries.
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


def accepts(schema: Validator) -> list[bool]:
    return [schema.is_valid(value) for value in VALUES]


def equivalent(left: Validator, right: Validator) -> bool:
    return accepts(left) == accepts(right)


@given(a=schemas, b=schemas)
def test_union_commutativity(a: object, b: object) -> None:
    assert equivalent(union(a, b), union(b, a))


@given(a=schemas, b=schemas)
def test_intersect_commutativity(a: object, b: object) -> None:
    assert equivalent(intersection(a, b), intersection(b, a))


@given(a=schemas, b=schemas, c=schemas)
def test_union_associativity(a: object, b: object, c: object) -> None:
    assert equivalent(union(union(a, b), c), union(a, union(b, c)))


@given(a=schemas, b=schemas, c=schemas)
def test_intersect_associativity(a: object, b: object, c: object) -> None:
    assert equivalent(
        intersection(intersection(a, b), c), intersection(a, intersection(b, c))
    )


@given(a=schemas)
def test_idempotence(a: object) -> None:
    assert equivalent(union(a, a), Validator(a))
    assert equivalent(intersection(a, a), Validator(a))


@given(a=schemas, b=schemas)
def test_absorption(a: object, b: object) -> None:
    assert equivalent(union(a, intersection(a, b)), Validator(a))
    assert equivalent(intersection(a, union(a, b)), Validator(a))


@given(a=schemas)
def test_identities(a: object) -> None:
    assert equivalent(union(a, nothing), Validator(a))
    assert equivalent(intersection(a, anything), Validator(a))
    assert equivalent(union(a, anything), anything)
    assert equivalent(intersection(a, nothing), nothing)


@given(a=schemas)
def test_double_negation(a: object) -> None:
    assert equivalent(complement(complement(a)), Validator(a))


@given(a=schemas, b=schemas)
def test_de_morgan(a: object, b: object) -> None:
    assert equivalent(
        complement(union(a, b)),
        intersection(complement(a), complement(b)),
    )
    assert equivalent(
        complement(intersection(a, b)),
        union(complement(a), complement(b)),
    )


@given(a=schemas, b=schemas, c=schemas)
def test_distributivity(a: object, b: object, c: object) -> None:
    assert equivalent(
        union(a, intersection(b, c)),
        intersection(union(a, b), union(a, c)),
    )
    assert equivalent(
        intersection(a, union(b, c)),
        union(intersection(a, b), intersection(a, c)),
    )


@given(a=schemas, b=schemas, c=schemas)
def test_simplify_preserves_acceptance(a: object, b: object, c: object) -> None:
    original = complement(union(a, intersection(b, complement(c))))
    assert equivalent(original, simplify(original))


@given(a=schemas, b=schemas)
def test_simplify_is_idempotent_on_acceptance(a: object, b: object) -> None:
    once = simplify(intersection(a, complement(b)))
    twice = simplify(once)
    assert equivalent(once, twice)


def test_simplify_decides_the_complement_laws() -> None:
    # The decision the conservative canonicalizer could not make: a schema and
    # its complement collapse, and provably disjoint types collapse.
    assert repr(simplify(intersection(int, complement(int)))) == "nothing"
    assert repr(simplify(union(int, complement(int)))) == "anything"
    assert repr(simplify(intersection(int, str))) == "nothing"
    assert repr(simplify(union(complement(int), complement(str)))) == "anything"


def test_simplify_leaves_the_gradual_any_uncollapsed() -> None:
    # `Any` is unchecked, not the lattice top, so the complement laws skip it.
    assert repr(simplify(intersection(Any, complement(Any)))) != "nothing"
    assert repr(simplify(union(Any, complement(Any)))) != "anything"
