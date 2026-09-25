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
    # The top. Drawn rather than left out, because an optional field holding it
    # is the one shape where `open` is not injective -- `{"a?": anything}` and
    # `{}` are two sets that open to one -- and a universe without it holds the
    # round-trip law below by never asking it the question.
    object,
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
@given(a=schemas, b=schemas, vals=value_lists)
def test_closing_is_a_function_of_the_set_however_it_is_spelled(
    a: object, b: object, vals: list[object]
) -> None:
    """Equal sets close to equal sets, which is what puts `close` in the algebra.

    Openness is the default of the key-type region no clause claims, and a
    region is a set of keys rather than a way of writing one. This law's
    silence about `open` is deliberate: `open` parts on a pair `close` does
    not, and `tests/test_projection_laws.py` carries it.

    The respelling is **absorption**, `a | (a & b)`, for a second drawn `b`.
    `a | (a & a)` reads like a respelling and is not one: the constructors fold
    the meet to `a` and the union to `a`, so the law compares a term with
    itself and passes for the reason a true law does.

    Asked of the opened term as well, because `close` is the identity on most
    of what is drawn -- a record with no clause is closed already -- and a law
    about an operator wants terms the operator moves.
    """
    for spelling in (a, Validator(a).open()):
        respelled = union(spelling, intersection(spelling, b))
        assert equivalent(Validator(spelling), Validator(respelled), vals), (
            "absorption is a different set"
        )
        assert equivalent(
            Validator(spelling).close(), Validator(respelled).close(), vals
        )


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
def test_closing_an_opened_schema_is_at_most_closing_it(
    a: object, vals: list[object]
) -> None:
    """`close` after `open` is **at most** `close`, and not equal to it.

    The pair moves one region -- the keys no clause claims -- and leaves every
    claimed one alone, which reads as a round trip and is not one: `open` is
    not injective. `{"a?": anything}` and `{}` are two sets that open to the
    same one, and `dict[str, anything]` opens to a single clause over every
    key, because two clauses carrying the same value are one clause. Neither
    normalisation is optional -- without them `open` and `close` would map
    equal sets to unequal ones -- and neither is recoverable.

    So the direction is what holds, and `tests/test_projection_laws.py` carries
    the values where the inclusion is strict.
    """
    schema = Validator(a)
    round_trip, closed = schema.open().close(), schema.close()
    for value in [*VALUES, *vals]:
        if round_trip.is_valid(value):
            assert closed.is_valid(value), value


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


# THEORY: the-descriptor
@given(a=schemas, b=schemas, c=schemas)
def test_the_verdict_is_stable_under_de_morgan(a: object, b: object, c: object) -> None:
    """Two spellings of one difference never give two different answers.

    `test_de_morgan` above holds the two spellings to the same *values*, which
    they agree on by construction -- the constructors build one normal form and
    the walk reads it. What it cannot see is the answer a caller asking a
    relation receives, because a decline and a proof admit the same values:
    neither admits any.

    So this asks the verdict, through `relation_to`, which reports the three
    answers apart and takes `nothing` as the schema every empty difference is
    below. Equality of the two is not the claim: `undecided` is a refusal, the
    two spellings reach the set representation's width bound through different
    intermediates, and the constructors fold one of them further. One spelling
    deciding where the other declines is those two facts, not a disagreement.

    What no spelling may do is contradict another. `subset` and `not_subset`
    are both answers a caller acts on -- the first says the difference is
    empty, the second names a value in it -- and one set cannot be both.
    """
    joined = intersection(a, complement(union(b, c))).relation_to(nothing)
    spelled = intersection(a, complement(b), complement(c)).relation_to(nothing)
    assert {joined, spelled} != {"subset", "not_subset"}, (
        f"one set, two spellings, two answers: {joined} and {spelled}"
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
    assert equivalent(original, original.simplify(), vals)  # ty: ignore[deprecated]


@given(a=schemas, b=schemas, vals=value_lists)
def test_simplify_is_idempotent_on_acceptance(
    a: object, b: object, vals: list[object]
) -> None:
    once = intersection(a, complement(b)).simplify()  # ty: ignore[deprecated]
    twice = once.simplify()  # ty: ignore[deprecated]
    assert equivalent(once, twice, vals)


def test_simplify_decides_the_complement_laws() -> None:
    # The decision the conservative canonicalizer could not make: a schema and
    # its complement collapse, and provably disjoint types collapse. Asserted on
    # the *denotation* of the simplified form (empty / universal) through the
    # decision procedure, not on the printed node string.
    assert intersection(int, complement(int)).simplify().is_empty()  # ty: ignore[deprecated]
    assert union(int, complement(int)).simplify().is_equivalent(anything)  # ty: ignore[deprecated]
    assert intersection(int, str).simplify().is_empty()  # ty: ignore[deprecated]
    assert union(complement(int), complement(str)).simplify().is_equivalent(anything)  # ty: ignore[deprecated]


def test_the_complement_laws_hold_of_any_as_of_the_top() -> None:
    # `Any` is the top, spelled: one node, one set, and the same laws. A rule
    # cannot tell the spellings apart, so the laws fire across them too.
    assert intersection(Any, complement(Any)).simplify().is_empty()  # ty: ignore[deprecated]
    assert union(Any, complement(Any)).simplify().is_equivalent(anything)  # ty: ignore[deprecated]
    assert intersection(Any, complement(anything)).simplify().is_empty()  # ty: ignore[deprecated]
    assert Validator(Any) == Validator(anything)
    assert Validator(Any).is_equivalent(anything)

    # What survives is the spelling, which `repr` gives back.
    assert repr(Validator(Any)) == "Any"
    assert repr(Validator(anything)) == "anything"
    assert repr(complement(Any).simplify()) == "nothing"  # ty: ignore[deprecated]


#: Bodies a fixpoint is built from: each takes the leaf beside the recursion
#: and gives the builder `recursive` takes. Every body guards the reference
#: under a container, so each is a fixpoint the frontend accepts.
_FIXPOINT_BODIES: list[Callable[[object], Callable[[Validator], object]]] = [
    lambda leaf: lambda t: union(leaf, list[t]),  # ty: ignore[invalid-type-form]
    lambda leaf: lambda t: union(leaf, {"v": leaf, "n?": t}),
    lambda leaf: lambda t: union(leaf, tuple[leaf, t]),  # ty: ignore[invalid-type-form]
    lambda leaf: lambda t: {"head": leaf, "tail?": list[t]},  # ty: ignore[invalid-type-form]
]


# THEORY: a-reference-denotes-its-definition
@given(
    body=st.sampled_from(_FIXPOINT_BODIES),
    leaf=st.sampled_from(ATOM_SCHEMAS),
    vals=value_lists,
)
def test_a_fixpoint_and_its_unfolding_are_one_set(
    body: Callable[[object], Callable[[Validator], object]],
    leaf: object,
    vals: list[object],
) -> None:
    """`recursive(f)` and `f(recursive(f))` admit the same values, and are decided so.

    The equirecursive reading: a reference denotes the definition it names,
    so writing the body out once by hand around the fixpoint is writing the
    same schema. Asked of the walk on values and of the decision as an
    equivalence, over every body above and every atom as its leaf.
    """
    build = body(leaf)
    fixpoint = recursive(build)
    unfolded = Validator(build(fixpoint))
    assert equivalent(fixpoint, unfolded, vals), "the unfolding is another set"
    assert fixpoint.is_equivalent(unfolded), "the unfolding is not decided one set"


# THEORY: lattice-theory
@given(a=schemas, vals=value_lists)
def test_the_complement_laws_hold_of_every_drawn_schema(
    a: object, vals: list[object]
) -> None:
    """`a | ~a` admits every value and `a & ~a` admits none, whatever `a` is.

    The two laws that make the lattice Boolean rather than distributive, held
    over the drawn universe: every atom, refinement, class, fixpoint and
    container the strategy builds, rather than the four fixed schemas the
    examples below ask. Both readings are asked. The walk answers with the
    values, since a law is a claim about membership, and the decision answers
    with a proof, since the constructors fold the two spellings to the bounds
    and a fold that stopped firing on some shape would leave a verdict the
    values still refute.
    """
    top = union(a, complement(a))
    bottom = intersection(a, complement(a))
    for value in [*VALUES, *vals]:
        assert top.is_valid(value), f"{value!r} is outside a | ~a"
        assert not bottom.is_valid(value), f"{value!r} is inside a & ~a"
    assert bottom.is_empty(), "a & ~a is not decided empty"
    assert top.is_equivalent(anything), "a | ~a is not decided the top"


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
