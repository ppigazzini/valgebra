"""An independent denotation oracle for the membership walk.

Every schema denotes a set of values; ``is_valid`` is the membership test. This
suite pairs each generated schema with an *independent* Python predicate that
encodes the same denotation — typed-singleton literals, ``bool`` as a subset of
``int``, ``float`` disjoint from ``int``, sequences/maps/sets as pointwise
membership, and the Boolean connectives as ``or``/``and``/``not`` — and asserts
the compiled validator agrees with the predicate for every generated value.

This is correctness against the denotation, not agreement between the two modes
of the walk (``tests/test_equivalence.py``). The predicate shares none of the
validator's frontend, so a build-time or walk bug that both modes would agree on
is caught here.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from functools import reduce
from types import GenericAlias
from typing import Annotated, Any, NoReturn

import annotated_types as at
from hypothesis import given
from hypothesis import strategies as st

from valgebra import (
    ValidationError,
    Validator,
    complement,
    intersection,
    recursive,
    union,
)

# A predicate deciding membership of a value in a schema's set.
Pred = Callable[[object], bool]
# A schema spec (what `validator` compiles) paired with its denotation predicate.
Spec = tuple[object, Pred]


@dataclasses.dataclass
class _Point:
    """A dataclass whose instances exercise the per-attribute `Attrs` node."""

    x: int
    y: str


def _point_pred(value: object) -> bool:
    """Membership for the `_Point` schema: an instance with well-typed fields."""
    return (
        isinstance(value, _Point)
        and isinstance(value.x, int)
        and isinstance(value.y, str)
    )


# Scalar type specs with their predicates. `bool` is a subset of `int` (an int
# predicate admits booleans), and `float` excludes `int`.
_SCALARS: list[Spec] = [
    (int, lambda x: isinstance(x, int)),
    (float, lambda x: isinstance(x, float)),
    (bool, lambda x: isinstance(x, bool)),
    (str, lambda x: isinstance(x, str)),
    (bytes, lambda x: isinstance(x, bytes)),
    (type(None), lambda x: x is None),
]

# Predicate by scalar type, for composing union arms.
_SCALAR_PRED: dict[object, Pred] = dict(_SCALARS)

# Cross-type constants: a typed singleton matches same-type-and-equal, so these
# stress the `1`/`True`/`1.0` distinction the walk must keep.
_CONSTS = [0, 1, -1, "a", "", True, False, 3.5, b"x"]

# Hashable scalar specs usable as dict keys and set elements.
_HASHABLE: list[Spec] = [
    (int, _SCALAR_PRED[int]),
    (str, _SCALAR_PRED[str]),
    (bool, _SCALAR_PRED[bool]),
]


def _literal(const: object) -> Spec:
    """Build a literal schema and its typed-singleton predicate (same type, equal)."""
    return (const, lambda x: type(x) is type(const) and x == const)


def _ge_pred(k: int) -> Pred:
    """Build the predicate for an integer at least ``k``."""
    return lambda x: isinstance(x, int) and x >= k


def _minlen_pred(k: int) -> Pred:
    """Build the predicate for a string of length at least ``k``."""
    return lambda x: isinstance(x, str) and len(x) >= k


def _union_of(types: Sequence[object]) -> Spec:
    """Build a PEP 604 union of scalar types and its any-arm predicate."""
    spec = reduce(lambda a, b: a | b, types)
    preds = [_SCALAR_PRED[t] for t in types]
    return (spec, lambda x: any(p(x) for p in preds))


def _list_pred(elem: Pred) -> Pred:
    return lambda x: isinstance(x, list) and all(elem(e) for e in x)


def _dict_pred(key: Pred, val: Pred) -> Pred:
    return lambda x: (
        isinstance(x, dict) and all(key(k) and val(v) for k, v in x.items())
    )


def _set_pred(elem: Pred) -> Pred:
    return lambda x: isinstance(x, set) and all(elem(e) for e in x)


def _frozenset_pred(elem: Pred) -> Pred:
    return lambda x: isinstance(x, frozenset) and all(elem(e) for e in x)


def _record_of(children: list[Spec]) -> Spec:
    """Build a closed record schema and its predicate from child specs."""
    spec: dict[str, object] = {}
    preds: dict[str, tuple[bool, Pred]] = {}
    for i, (child_spec, child_pred) in enumerate(children):
        required = i % 2 == 0
        spec[f"f{i}" if required else f"f{i}?"] = child_spec
        preds[f"f{i}"] = (required, child_pred)
    return (spec, _record_pred(preds))


def _record_pred(preds: dict[str, tuple[bool, Pred]]) -> Pred:
    names = set(preds)

    def pred(x: object) -> bool:
        if not isinstance(x, dict):
            return False
        present: dict[str, object] = {}
        for key, val in x.items():
            if not (isinstance(key, str) and key in names):
                return False  # a closed record admits only its declared keys
            present[key] = val
        return all(
            (p(present[name]) if name in present else not required)
            for name, (required, p) in preds.items()
        )

    return pred


def _hetero_pred(str_val: Pred, int_val: Pred) -> Pred:
    """Predicate for `{str: V1, int: V2}` (str keys take V1, int keys V2)."""

    def pred(x: object) -> bool:
        if not isinstance(x, dict):
            return False
        for key, val in x.items():
            if isinstance(key, str):
                if not str_val(val):
                    return False
            elif isinstance(key, int):
                if not int_val(val):
                    return False
            else:
                return False
        return True

    return pred


def _prefix_tail_pred(prefix: list[Pred], tail: Pred, container: type = list) -> Pred:
    n = len(prefix)

    def pred(x: object) -> bool:
        # Membership is container-strict: a tuple is never a member of the list
        # form and vice versa. The first check narrows to a sized sequence; the
        # second pins the exact container (`container` is list or tuple).
        if not isinstance(x, list | tuple) or not isinstance(x, container):
            return False
        if len(x) < n:
            return False
        head = all(p(e) for p, e in zip(prefix, x[:n], strict=False))
        return head and all(tail(e) for e in x[n:])

    return pred


# Non-composable atoms: the lattice bounds, the gradual dynamic, an instance
# check, and a per-attribute object schema. `object` is the top and `NoReturn`
# the bottom -- the typing-native spelling of the empty type on every supported
# Python (`Never` is 3.11+, so the cross-version spelling is used). `Any` is the
# gradual dynamic, which admits every value like the top but is a distinct node.
# `complex` is a plain class (an isinstance check), and the `_Point` dataclass is
# checked field by field.
_ATOMS: list[Spec] = [
    (object, lambda _x: True),
    (NoReturn, lambda _x: False),
    (Any, lambda _x: True),
    (complex, lambda x: isinstance(x, complex)),
    (_Point, _point_pred),
]


def _leaf() -> st.SearchStrategy[Spec]:
    return st.one_of(
        st.sampled_from(_SCALARS),
        st.sampled_from(_ATOMS),
        st.sampled_from(_CONSTS).map(_literal),
        st.integers(min_value=-5, max_value=5).map(
            lambda k: (Annotated[int, at.Ge(k)], _ge_pred(k))
        ),
        st.integers(min_value=0, max_value=5).map(
            lambda k: (Annotated[str, at.MinLen(k)], _minlen_pred(k))
        ),
        st.lists(
            st.sampled_from([int, str, bytes, float, type(None)]),
            min_size=2,
            max_size=3,
            unique=True,
        ).map(_union_of),
    )


def _specs() -> st.SearchStrategy[Spec]:
    """Generate a schema spec paired with its denotation predicate.

    Generic containers are built with ``GenericAlias`` at runtime (``list[T]`` and
    friends) so the element schema, a runtime value, is not read as a static type
    expression.
    """
    return st.recursive(
        _leaf(),
        lambda child: st.one_of(
            child.map(lambda sp: (GenericAlias(list, (sp[0],)), _list_pred(sp[1]))),
            st.tuples(st.sampled_from(_HASHABLE), child).map(
                lambda kv: (
                    GenericAlias(dict, (kv[0][0], kv[1][0])),
                    _dict_pred(kv[0][1], kv[1][1]),
                )
            ),
            st.sampled_from(_HASHABLE).map(
                lambda sp: (GenericAlias(set, (sp[0],)), _set_pred(sp[1]))
            ),
            st.sampled_from(_HASHABLE).map(
                lambda sp: (GenericAlias(frozenset, (sp[0],)), _frozenset_pred(sp[1]))
            ),
            # A native [A, B, ...] list: a fixed prefix then a repeated tail.
            st.tuples(child, child).map(
                lambda ab: (
                    [ab[0][0], ab[1][0], ...],
                    _prefix_tail_pred([ab[0][1]], ab[1][1]),
                )
            ),
            # A tuple[A, B, ...]: the same prefix-plus-tail under the tuple
            # container, so membership rejects the list form and vice versa.
            st.tuples(child, child).map(
                lambda ab: (
                    GenericAlias(tuple, (ab[0][0], ab[1][0], ...)),
                    _prefix_tail_pred([ab[0][1]], ab[1][1], tuple),
                )
            ),
            # A closed record of named fields.
            st.lists(child, min_size=1, max_size=2).map(_record_of),
            # A heterogeneous mapping keyed by disjoint key schemas.
            st.tuples(child, child).map(
                lambda ab: (
                    {str: ab[0][0], int: ab[1][0]},
                    _hetero_pred(ab[0][1], ab[1][1]),
                )
            ),
        ),
        max_leaves=5,
    )


def _values() -> st.SearchStrategy[object]:
    leaf = st.one_of(
        st.none(),
        st.booleans(),
        st.integers(min_value=-5, max_value=5),
        st.sampled_from([0, 1, -1, 0.0, 1.0, 3.5]),
        st.text(max_size=3),
        st.binary(max_size=3),
        # Values that live only in these node kinds, so their accept paths are
        # exercised rather than always rejected: complex numbers, frozensets, and
        # dataclass instances (well-typed and mistyped).
        st.builds(complex, st.integers(-3, 3), st.integers(-3, 3)),
        st.frozensets(st.integers(-3, 3), max_size=3),
        st.builds(_Point, st.integers(-3, 3), st.text(max_size=2)),
        st.builds(_Point, st.text(max_size=2), st.integers(-3, 3)),
    )
    return st.recursive(
        leaf,
        lambda child: st.one_of(
            st.lists(child, max_size=4),
            st.dictionaries(
                st.one_of(st.integers(-3, 3), st.text(max_size=2), st.booleans()),
                child,
                max_size=3,
            ),
            st.sets(st.integers(-3, 3), max_size=3),
            st.tuples(child, child),
        ),
        max_leaves=6,
    )


def _recursive_case(leaf: Spec) -> tuple[Validator, Pred]:
    """Build a `recursive` schema `T = leaf | list[T]` and its predicate.

    This is the only spelling that produces the `Ref`/`SelfRef` nodes: a fixpoint
    whose body refers to itself under a structural `list`. The predicate mirrors
    the fixpoint -- a value is a member iff it is a leaf value or a list whose
    every element is itself a member.
    """
    leaf_spec, leaf_pred = leaf
    schema = recursive(lambda t: union(leaf_spec, GenericAlias(list, (t,))))

    def pred(x: object) -> bool:
        return leaf_pred(x) or (isinstance(x, list) and all(pred(e) for e in x))

    return Validator(schema), pred


@st.composite
def _cases(draw: st.DrawFn) -> tuple[Validator, Pred]:
    """Draw a compiled validator paired with its denotation predicate.

    The plain case compiles the spec directly; the wrapped cases exercise the
    top-level algebra: complement negates the predicate, intersection conjoins two.
    """
    mode = draw(st.sampled_from(["plain", "complement", "intersection", "recursive"]))
    if mode == "recursive":
        return _recursive_case(draw(st.sampled_from(_SCALARS)))
    spec, pred = draw(_specs())
    if mode == "plain":
        return Validator(spec), pred
    if mode == "complement":
        return complement(spec), lambda x: not pred(x)
    spec2, pred2 = draw(_specs())
    return intersection(spec, spec2), lambda x: pred(x) and pred2(x)


@given(case=_cases(), value=_values())
def test_walk_matches_denotation(case: tuple[Validator, Pred], value: object) -> None:
    compiled, predicate = case
    expected = predicate(value)
    assert compiled.is_valid(value) is expected
    try:
        compiled.validate(value)
        raised = False
    except ValidationError:
        raised = True
    assert raised is (not expected)
