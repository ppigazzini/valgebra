"""Property tests for the Boolean-algebra laws, checked as membership equivalence.

Two schemas are equivalent when they accept exactly the same values. Each law is
checked over a fixed spread of values that exercises the atom boundaries (the
int/bool/float distinctions and the typed-singleton literals).
"""

from typing import Literal

from hypothesis import given
from hypothesis import strategies as st

from valgebra import (
    CompiledValidator,
    anything,
    complement,
    intersect,
    nothing,
    simplify,
    union,
    validator,
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


def accepts(schema: CompiledValidator) -> list[bool]:
    return [schema.is_valid(value) for value in VALUES]


def equivalent(left: CompiledValidator, right: CompiledValidator) -> bool:
    return accepts(left) == accepts(right)


@given(a=schemas, b=schemas)
def test_union_commutativity(a: object, b: object) -> None:
    assert equivalent(union(a, b), union(b, a))


@given(a=schemas, b=schemas)
def test_intersect_commutativity(a: object, b: object) -> None:
    assert equivalent(intersect(a, b), intersect(b, a))


@given(a=schemas, b=schemas, c=schemas)
def test_union_associativity(a: object, b: object, c: object) -> None:
    assert equivalent(union(union(a, b), c), union(a, union(b, c)))


@given(a=schemas, b=schemas, c=schemas)
def test_intersect_associativity(a: object, b: object, c: object) -> None:
    assert equivalent(intersect(intersect(a, b), c), intersect(a, intersect(b, c)))


@given(a=schemas)
def test_idempotence(a: object) -> None:
    assert equivalent(union(a, a), validator(a))
    assert equivalent(intersect(a, a), validator(a))


@given(a=schemas, b=schemas)
def test_absorption(a: object, b: object) -> None:
    assert equivalent(union(a, intersect(a, b)), validator(a))
    assert equivalent(intersect(a, union(a, b)), validator(a))


@given(a=schemas)
def test_identities(a: object) -> None:
    assert equivalent(union(a, nothing), validator(a))
    assert equivalent(intersect(a, anything), validator(a))
    assert equivalent(union(a, anything), anything)
    assert equivalent(intersect(a, nothing), nothing)


@given(a=schemas)
def test_double_negation(a: object) -> None:
    assert equivalent(complement(complement(a)), validator(a))


@given(a=schemas, b=schemas)
def test_de_morgan(a: object, b: object) -> None:
    assert equivalent(
        complement(union(a, b)),
        intersect(complement(a), complement(b)),
    )
    assert equivalent(
        complement(intersect(a, b)),
        union(complement(a), complement(b)),
    )


@given(a=schemas, b=schemas, c=schemas)
def test_distributivity(a: object, b: object, c: object) -> None:
    assert equivalent(
        union(a, intersect(b, c)),
        intersect(union(a, b), union(a, c)),
    )
    assert equivalent(
        intersect(a, union(b, c)),
        union(intersect(a, b), intersect(a, c)),
    )


@given(a=schemas, b=schemas, c=schemas)
def test_simplify_preserves_acceptance(a: object, b: object, c: object) -> None:
    original = complement(union(a, intersect(b, complement(c))))
    assert equivalent(original, simplify(original))


@given(a=schemas, b=schemas)
def test_simplify_is_idempotent_on_acceptance(a: object, b: object) -> None:
    once = simplify(intersect(a, complement(b)))
    twice = simplify(once)
    assert equivalent(once, twice)
