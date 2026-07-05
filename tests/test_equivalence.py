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

from types import GenericAlias
from typing import Annotated

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
            st.tuples(child, child).map(
                lambda ab: GenericAlias(tuple, (ab[0], ab[1], ...))
            ),  # a prefix-plus-tail tuple
            st.tuples(child, child).map(lambda ab: GenericAlias(dict, ab)),
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
