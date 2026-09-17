"""Property tests for the Boolean-algebra laws, checked as membership equivalence.

Two schemas are equivalent when they accept exactly the same values. Each law is
checked over generated values drawn from a strategy spanning scalars and nested
containers, seeded with a curated spread that pins the atom boundaries (the
int/bool/float distinctions and the typed-singleton literals). Generating the
witness values, rather than iterating a fixed list, means two schemas that merely
agree on a handful of constants do not pass a law: a value that distinguishes
them is searched for. The semantic decision (`is_equivalent`) is cross-checked
against membership separately in the subtyping suite.
"""

from collections.abc import Callable
from typing import Annotated, Any, Literal

import annotated_types as at
import pytest
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

# `simplify` is deprecated and these exercise it deliberately: the folds it
# still performs are its own, and they are checked until it goes.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Validator.simplify is deprecated:DeprecationWarning"
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
    b"x",
    b"",
    [1, 2],
    [1, "a"],
    [1, "a", 2, 3],
    [],
    {1, 2},
    {"k": 1},
    {},
    (1, "a"),
    (1, 2, 3),
    # The boundaries, which a drawn value reaches only by luck. Each is a value
    # some operation reads differently from its neighbours: `nan` equals
    # nothing including itself, the infinities bound every comparison, `-0.0`
    # equals `0.0` and hashes with it, the two integers sit at the ends of the
    # carriers a lowering uses, and a newline is where a pattern's `.` and `$`
    # part company.
    float("nan"),
    float("inf"),
    float("-inf"),
    -0.0,
    2**53 + 1,
    -(2**63),
    2**63 - 1,
    2**70,
    "\n",
    "a\nb",
]


class Point:
    """One class, so a law is checked over a node the frontend builds by name.

    An atom whose membership is `isinstance` rather than a kind reading, which
    is a different arm of every operation the laws exercise.
    """

    __slots__ = ()


#: The markers a refinement is drawn with. Each narrows a base the law then
#: combines, so the laws reach `Refine` nodes rather than atoms alone -- and a
#: refinement is where a meet has to compare *constraints* rather than kinds.
_REFINEMENTS = [
    Annotated[int, at.Ge(0)],
    Annotated[int, at.Lt(3)],
    Annotated[int, at.MultipleOf(2)],
    Annotated[str, at.MinLen(1)],
    Annotated[str, at.MaxLen(2)],
]


def _fixpoints() -> st.SearchStrategy[object]:
    """Schemas with a back edge, which no sampled atom has.

    A fixpoint is the one node the descriptor cannot hold, so every law over
    one is decided by the rules -- a path the atom list never reached.
    """
    return st.sampled_from(
        [
            recursive(lambda t: int | list[t]),  # ty: ignore[invalid-type-form]
            recursive(lambda t: {"v": int, "n?": t}),
            recursive(lambda t: union(None, tuple[int, t])),  # ty: ignore[invalid-type-form]
        ]
    )


def _schemas() -> st.SearchStrategy[object]:
    """Every shape a law is checked over, drawn rather than listed.

    The list of atoms was the whole universe, so a law held over scalars and
    containers and was never asked about a refinement, a class or a fixpoint.
    Each of those reaches arms the others do not: a refinement compares
    constraints, a class compares by `isinstance`, and a fixpoint is decided by
    the rules because the descriptor cannot hold a cycle.

    Containers are built *around* a drawn schema rather than sampled whole, so
    the element is one of those shapes too and the recursion reaches a
    refinement inside a list.
    """
    leaves = st.one_of(
        st.sampled_from(ATOM_SCHEMAS),
        st.sampled_from(_REFINEMENTS),
        st.just(Point),
        _fixpoints(),
    )
    return st.recursive(
        leaves,
        lambda inner: st.one_of(
            inner.map(lambda element: list[element]),
            inner.map(lambda element: dict[str, element]),
            inner.map(lambda element: dict[int, element]),
            inner.map(lambda element: {"a": element}),
            inner.map(lambda element: {"a?": element}),
            inner.map(lambda element: tuple[element, ...]),
        ),
        max_leaves=3,
    )


schemas = _schemas()

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
    sample_agree = all(left.is_valid(v) == right.is_valid(v) for v in values)
    # The semantic decision is sound: a True from is_equivalent is a proof, so it
    # must never contradict a sampled disagreement. Cross-checking it here means a
    # law's claim is validated by the decision procedure too, not only by sample
    # agreement, and an unsound is_equivalent that claimed two distinct schemas
    # equal would be caught by the witnessing values.
    if left.is_equivalent(right):
        assert sample_agree, "is_equivalent claimed equality the values refute"
    return sample_agree


# THEORY: open-and-close-read-the-region
@given(a=schemas, vals=value_lists)
def test_closing_is_a_function_of_the_set_however_it_is_spelled(
    a: object, vals: list[object]
) -> None:
    """Equal sets close to equal sets, which is what puts `close` in the algebra.

    Openness is the default of the key-type region no clause claims, and a
    region is a set of keys rather than a way of writing one. The respelling is
    a union with a redundant branch, because that is the shape the one declared
    exception lives in: `open` parts there and `close` does not, so drawing it
    is what makes this law's silence about `open` deliberate.

    `tests/test_projection_laws.py` carries the pair that separates the two.
    """
    respelled = union(a, intersection(a, a))
    assert equivalent(Validator(a).close(), Validator(respelled).close(), vals)


@given(a=schemas, vals=value_lists)
def test_opening_admits_what_the_schema_admits(a: object, vals: list[object]) -> None:
    """Opening frees keys and takes none away, so it only widens.

    The direction is the whole of what "free the region no clause claims" means,
    and it holds where the congruence law does not: a term is opened into a
    superset of itself whatever it is written out of.
    """
    schema = Validator(a)
    opened = schema.open()
    for value in [*VALUES, *vals]:
        if schema.is_valid(value):
            assert opened.is_valid(value), value


@given(a=schemas, vals=value_lists)
def test_closing_refuses_what_the_schema_refuses(a: object, vals: list[object]) -> None:
    """And closing narrows, so the two are a pair rather than one rewrite."""
    schema = Validator(a)
    closed = schema.close()
    for value in [*VALUES, *vals]:
        if closed.is_valid(value):
            assert schema.is_valid(value), value


@given(a=schemas, vals=value_lists)
def test_closing_an_opened_schema_returns_the_regions_it_freed(
    a: object, vals: list[object]
) -> None:
    """`close` after `open` is `close`, because the two move one region.

    Opening sets the default of the region no clause claims to the top and
    closing sets it to nothing, so the pair is idempotent on that region and
    touches no other -- which is the round trip a caller reads them as.
    """
    schema = Validator(a)
    assert equivalent(schema.open().close(), schema.close(), vals)


# THEORY: lattice-theory
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


# THEORY: lattice-theory
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


# THEORY: property-testing
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


# THEORY: property-testing, the-normal-form-is-not-canonical
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
    # its complement collapse, and provably disjoint types collapse. Asserted on
    # the *denotation* of the simplified form (empty / universal) through the
    # decision procedure, not on the printed node string.
    assert intersection(int, complement(int)).simplify().is_empty()
    assert union(int, complement(int)).simplify().is_equivalent(anything)
    assert intersection(int, str).simplify().is_empty()
    assert union(complement(int), complement(str)).simplify().is_equivalent(anything)


def test_the_complement_laws_hold_of_any_as_of_the_top() -> None:
    # `Any` is the top, spelled: one node, one set, and the same laws. A rule
    # cannot tell the spellings apart, so the laws fire across them too.
    assert intersection(Any, complement(Any)).simplify().is_empty()
    assert union(Any, complement(Any)).simplify().is_equivalent(anything)
    assert intersection(Any, complement(anything)).simplify().is_empty()
    assert Validator(Any) == Validator(anything)
    assert Validator(Any).is_equivalent(anything)

    # What survives is the spelling, which `repr` gives back.
    assert repr(Validator(Any)) == "Any"
    assert repr(Validator(anything)) == "anything"
    assert repr(complement(Any).simplify()) == "nothing"


# Two spellings the procedure itself proves equal, for the law below.
_RESPELLINGS = [
    ("a-or-nothing", lambda a: union(a, nothing)),
    ("a-and-anything", lambda a: intersection(a, anything)),
    ("a-or-a", lambda a: union(a, a)),
]


# The complement laws hold of the *set*, not of the spelling.
#
# This was a ledger entry, and the reason it gave is the reason it is one no
# longer. The rules cancel a complement and collapse a join by structural
# equality, so `A | ~A` covered the universe and `A | ~B` did not -- for a `B`
# the procedure itself decides equivalent to `A`. What it asked for was the
# operands compared as sets where the rules read equality, which is what a
# representation closed under the Boolean operations gives, and the descriptor
# is that representation: it holds each kind as a set, so a respelling and the
# thing it respells build the same one.
@pytest.mark.parametrize(
    ("name", "respell"), _RESPELLINGS, ids=[name for name, _ in _RESPELLINGS]
)
def test_the_complement_laws_survive_a_respelling(
    name: str, respell: Callable[[Validator], Validator]
) -> None:
    a = Validator(list[int])
    b = respell(a)
    assert a.is_equivalent(b), "the respelling must denote the same set"
    assert Validator(anything).is_subtype_of(union(a, complement(b)))
    assert intersection(a, complement(b)).is_empty()
