"""The bool fast path and the explain walk must agree on membership.

``is_valid`` (the membership fast path) and ``validate`` (the aggregating explain
walk) are two separate traversals of the IR that must reach the same membership
verdict for every schema and value. That invariant is maintained by hand, one
node at a time; this property test fuzzes it across the node kinds so a future
divergence in one walk that the other does not share is caught.

This is membership-equivalence between the two *walks*, which is distinct from
correctness against a node's denotation: both walks share the frontend, so a
build-time bug would make them agree while both being wrong. Targeted tests cover
denotations; this covers walk equivalence.
"""

from __future__ import annotations

from types import GenericAlias
from typing import Annotated

import annotated_types as at
from hypothesis import given
from hypothesis import strategies as st

from valgebra import ValidationError, complement, intersect, union, validator

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


def _schemas() -> st.SearchStrategy[object]:
    leaf = st.one_of(
        st.sampled_from(_SCALARS), st.sampled_from(_LITERALS), _refinements()
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
            st.tuples(child, child).map(lambda ab: GenericAlias(dict, ab)),
            st.tuples(child, child).map(lambda ab: {"a": ab[0], "b?": ab[1]}),
            st.tuples(child, child).map(lambda ab: union(ab[0], ab[1])),
            st.tuples(child, child).map(lambda ab: intersect(ab[0], ab[1])),
            child.map(complement),
        ),
        max_leaves=12,
    )


def _values() -> st.SearchStrategy[object]:
    leaf = st.one_of(
        st.none(),
        st.booleans(),
        st.integers(),
        st.floats(allow_nan=False, allow_infinity=False),
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
    v = validator(spec)
    fast = v.is_valid(value)
    try:
        v.validate(value)
        slow = True
    except ValidationError:
        slow = False
    assert fast == slow
