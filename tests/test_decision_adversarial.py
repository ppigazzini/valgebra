"""Broad generative soundness check for the decision procedure.

Arbitrary nested schemas ``a`` and ``b`` and an arbitrary value ``v`` are
generated, then the decision is held to real membership: ``is_subtype`` and
``equivalent`` never claim a relation membership contradicts, ``is_empty`` never
accepts a value, and every schema is a subtype of itself. This is deliberately
adversarial — it found the complement-reflexivity and pool-merge bugs.
"""

from typing import Annotated, Literal

import annotated_types as at
from hypothesis import given
from hypothesis import strategies as st

from valgebra import complement, intersect, union, validator

_bases = st.sampled_from([int, str, bool, float, bytes, None, complex, bytearray])
_lits = st.sampled_from(
    [
        Literal[0],
        Literal[1],
        Literal[-1],
        Literal[2],
        Literal["a"],
        Literal["b"],
        Literal[True],
    ]
)
_refines = st.integers(-2, 2).map(lambda n: Annotated[int, at.Ge(n)])
_leaf = st.one_of(_bases, _lits, _refines)


def _extend(children: st.SearchStrategy) -> st.SearchStrategy:
    pair = st.tuples(children, children)
    return st.one_of(
        children.map(lambda c: ("list", c)),
        pair.map(lambda p: ("fixed", p[0], p[1])),
        children.map(lambda c: ("tail", c)),
        children.map(lambda c: ("dict", c)),
        children.map(lambda c: ("record", c)),
        pair.map(lambda p: ("union", p[0], p[1])),
        pair.map(lambda p: ("intersect", p[0], p[1])),
        children.map(lambda c: ("complement", c)),
    )


_specs = st.recursive(_leaf, _extend, max_leaves=12)

_BUILDERS = {
    "list": lambda a: [a[0]],
    "fixed": lambda a: [a[0], a[1]],
    "tail": lambda a: [a[0], ...],
    "dict": lambda a: {str: a[0]},
    "record": lambda a: {"k": a[0], "j?": a[0]},
    "union": lambda a: union(a[0], a[1]),
    "intersect": lambda a: intersect(a[0], a[1]),
    "complement": lambda a: complement(a[0]),
}


def _build(spec: object) -> object:
    if not isinstance(spec, tuple) or not spec:
        return spec
    tag = spec[0]
    if not isinstance(tag, str):
        return spec
    return _BUILDERS[tag]([_build(child) for child in spec[1:]])


_scalars = st.one_of(
    st.integers(-3, 3),
    st.text(max_size=2),
    st.booleans(),
    st.floats(allow_nan=False, allow_infinity=False, min_value=-5, max_value=5),
    st.none(),
    st.binary(max_size=2),
)
_values = st.recursive(
    _scalars,
    lambda c: st.one_of(
        st.lists(c, max_size=3),
        st.dictionaries(st.text(max_size=2), c, max_size=3),
        st.tuples(c, c),
        st.frozensets(st.integers(-3, 3), max_size=3),
    ),
    max_leaves=8,
)


@given(sa=_specs, sb=_specs, v=_values)
def test_decision_is_sound_against_membership(
    sa: object, sb: object, v: object
) -> None:
    try:
        a, b = _build(sa), _build(sb)
        left, right = validator(a), validator(b)
    except (ValueError, TypeError, NotImplementedError, RecursionError):
        return  # an unbuildable combination is not under test
    assert left.is_subtype(a)  # reflexivity
    in_a = left.is_valid(v)
    if left.is_subtype(b) and in_a:
        assert right.is_valid(v)
    if left.is_empty():
        assert not in_a
    if left.equivalent(b):
        assert in_a == right.is_valid(v)
