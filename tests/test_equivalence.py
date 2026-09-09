"""The fast and explain modes of the membership walk must agree.

``is_valid`` runs the one membership walk in its fast mode (a bool, no
allocation); ``validate`` runs the same walk in explain mode (aggregating a
violation per failure). The fast-mode verdict and "explain mode produced no
violation" must coincide for every schema and value. This property test fuzzes
that agreement across the node kinds, so a mode-specific divergence — an explain
pass that describes a failure the fast check does not make, or the reverse — is
caught.

This is agreement between the two *modes* of the walk, distinct from correctness
against a node's denotation: both modes share the frontend, so a build-time bug
would make them agree while both being wrong. The denotation oracle in
``tests/test_denotation.py`` covers that; this covers mode agreement.
"""

from __future__ import annotations

import time
from types import GenericAlias
from typing import Annotated, Literal

import annotated_types as at
from hypothesis import given
from hypothesis import strategies as st

from valgebra import ValidationError, Validator, complement, intersection, union

# Atoms span the scalar nodes, the top, the gradual node, and a spread of typed
# literal singletons.
_SCALARS = [int, str, bool, float, bytes, None, object]
_LITERALS = [0, 1, -1, "a", "", True, False, 3.5, b"x"]


def _refinements() -> st.SearchStrategy[object]:
    return st.one_of(
        st.integers(min_value=-5, max_value=5).map(lambda k: Annotated[int, at.Ge(k)]),
        st.integers(min_value=0, max_value=5).map(
            lambda k: Annotated[str, at.MinLen(k)]
        ),
    )


def _key_schemas() -> st.SearchStrategy[object]:
    """Draw the schemas a map clause may key on.

    A clause's key says which keys it governs, and a map reads that as whole
    *kinds* of key or as the constants a ``Literal`` names. A key narrowed by a
    constraint is refused where it is written, so the generator writes what the
    frontend builds.
    """
    return st.one_of(
        st.sampled_from(_SCALARS),
        st.sampled_from(_LITERALS).map(lambda value: Literal[value]),  # ty: ignore[invalid-type-form]
    )


def _schemas() -> st.SearchStrategy[object]:
    # A constant is spelled `Literal[v]` rather than bare: these leaves are also
    # used as the *argument* of a generic, where a bare value is a forward
    # reference to a type rather than a value. The bare-constant spelling is a
    # schema in its own right and is covered by `tests/test_literals.py`.
    leaf = st.one_of(
        st.sampled_from(_SCALARS),
        st.sampled_from(_LITERALS).map(lambda value: Literal[value]),  # ty: ignore[invalid-type-form]
        _refinements(),
    )
    return st.recursive(
        leaf,
        lambda child: st.one_of(
            # GenericAlias builds list[x]/dict[k, v]/... at runtime without the
            # static type-checker reading the element as a type expression.
            child.map(lambda x: GenericAlias(list, (x,))),
            child.map(lambda x: GenericAlias(set, (x,))),
            child.map(lambda x: GenericAlias(frozenset, (x,))),
            child.map(lambda x: GenericAlias(tuple, (x, ...))),
            st.tuples(child, child).map(lambda ab: GenericAlias(tuple, ab)),
            st.tuples(child, child).map(
                lambda ab: GenericAlias(tuple, (ab[0], ab[1], ...))
            ),  # a prefix-plus-tail tuple
            st.tuples(_key_schemas(), child).map(lambda ab: GenericAlias(dict, ab)),
            st.tuples(child, child).map(lambda ab: {"a": ab[0], "b?": ab[1]}),
            st.tuples(child, child).map(lambda ab: union(ab[0], ab[1])),
            st.tuples(child, child).map(lambda ab: intersection(ab[0], ab[1])),
            child.map(complement),
        ),
        max_leaves=12,
    )


def _values() -> st.SearchStrategy[object]:
    leaf = st.one_of(
        st.none(),
        st.booleans(),
        st.integers(),
        # NaN and the infinities are included: the fast and explain paths must
        # return the same verdict on them, whatever that verdict is.
        st.floats(allow_nan=True, allow_infinity=True),
        st.text(max_size=5),
        st.binary(max_size=5),
    )
    hashable = st.one_of(st.integers(), st.text(max_size=3), st.booleans(), st.none())
    return st.recursive(
        leaf,
        lambda child: st.one_of(
            st.lists(child, max_size=4),
            st.tuples(child, child),
            st.dictionaries(st.text(max_size=3), child, max_size=4),
            st.sets(hashable, max_size=4),
            st.frozensets(st.integers(), max_size=4),
        ),
        max_leaves=10,
    )


@given(spec=_schemas(), value=_values())
def test_is_valid_agrees_with_validate(spec: object, value: object) -> None:
    v = Validator(spec)
    fast = v.is_valid(value)
    try:
        v.validate(value)
        slow = True
    except ValidationError:
        slow = False
    assert fast == slow


def test_two_wide_literal_sets_are_decided_as_sets_not_pair_by_pair() -> None:
    """A member-by-member walk is quadratic in the oracle, and it showed.

    The core decides a union against a union by asking each member about each
    member, so two 20,000-member literal unions were 400 million calls into the
    bindings: six seconds for a question about two sets of integers, on a shape
    a contract really writes -- an enumeration of codes is one.

    Timed against size rather than against a constant, because the defect was
    the *growth*: quadratic passes any bound generous enough for the small case.
    """

    def disjointness(count: int) -> float:
        left = Validator(Literal[tuple(range(count))])  # ty: ignore[invalid-type-form]
        right = Validator(Literal[tuple(range(count, 2 * count))])  # ty: ignore[invalid-type-form]
        meet = intersection(left, right)

        def once() -> float:
            started = time.perf_counter()
            assert meet.is_empty()
            return time.perf_counter() - started

        # The fastest of several, which is the estimator a loaded machine cannot
        # spoil: another process steals time from a reading and never gives any
        # back, so a spike inflates every run it touches and the minimum is the
        # one nearest the work itself. A quadratic decision blows up the minimum
        # as surely as the mean.
        return min(once() for _ in range(5))

    small, large = disjointness(1_000), disjointness(8_000)
    assert large < small * 24 + 0.05, (
        f"8,000 members took {large:.3f}s against {small:.3f}s for 1,000; "
        "the decision is scaling with the square of the set"
    )


def test_the_set_decision_answers_what_the_pairwise_one_did() -> None:
    """Every edge the per-pair rule reads, asked of the set rule.

    A literal pins `type(x)` exactly, so the join is keyed by `(type, value)`:
    a set keyed by the value alone would call `Literal[1]` and `Literal[True]`
    equal, and a rule that answered `disjoint` for an overlapping pair would be
    unsound rather than merely slow.
    """

    def disjoint(left: object, right: object) -> bool:
        return intersection(left, right).is_empty()

    assert disjoint(Literal[1, 2, 3], Literal[4, 5, 6])
    assert not disjoint(Literal[1, 2, 3], Literal[3, 4, 5])
    assert intersection(Literal[1, 2, 3], Literal[3, 4, 5]).is_valid(3)
    # A shared value across types is not shared: the literal pins the type.
    assert disjoint(Literal[1], Literal[True])
    assert disjoint(Literal[1.0], Literal[1])  # ty: ignore[invalid-type-form]
    assert disjoint(Literal["1"], Literal[1])
    assert disjoint(Literal[b"a"], Literal["a"])
    # And the same constant is the same value.
    assert not disjoint(Literal[7], Literal[7])
    # `None` rather than `Literal[None]`: the linters rewrite the second on
    # sight, and the frontend builds the same node from either.
    assert not disjoint(None, None)
    assert not disjoint(Literal[1, "a"] | None, Literal["a", 2])


def test_a_union_with_a_non_literal_member_falls_back_to_the_member_walk() -> None:
    """The set question needs a set of constants; a kind is not one."""
    assert intersection(Literal[1, 2] | bytes, Literal[3, 4]).is_empty()
    assert not intersection(Literal[1, 2] | bytes, Literal[3, 4] | bytes).is_empty()
    # A kind on one side only: the walk asks each literal against `bytes`.
    assert intersection(Literal[1, 2], Literal[3, 4] | bytes).is_empty()
