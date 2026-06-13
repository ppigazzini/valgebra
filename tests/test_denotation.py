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

from collections.abc import Callable, Sequence
from functools import reduce
from types import GenericAlias
from typing import Annotated

import annotated_types as at
from hypothesis import given, settings
from hypothesis import strategies as st

from valgebra import (
    CompiledValidator,
    ValidationError,
    complement,
    intersect,
    validator,
)

# A predicate deciding membership of a value in a schema's set.
Pred = Callable[[object], bool]
# A schema spec (what `validator` compiles) paired with its denotation predicate.
Spec = tuple[object, Pred]

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


def _prefix_tail_pred(prefix: list[Pred], tail: Pred) -> Pred:
    n = len(prefix)

    def pred(x: object) -> bool:
        if not isinstance(x, list) or len(x) < n:
            return False
        head = all(p(e) for p, e in zip(prefix, x[:n], strict=False))
        return head and all(tail(e) for e in x[n:])

    return pred


def _leaf() -> st.SearchStrategy[Spec]:
    return st.one_of(
        st.sampled_from(_SCALARS),
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
            # A native [A, B, ...] list: a fixed prefix then a repeated tail.
            st.tuples(child, child).map(
                lambda ab: (
                    [ab[0][0], ab[1][0], ...],
                    _prefix_tail_pred([ab[0][1]], ab[1][1]),
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


@st.composite
def _cases(draw: st.DrawFn) -> tuple[CompiledValidator, Pred]:
    """Draw a compiled validator paired with its denotation predicate.

    The plain case compiles the spec directly; the wrapped cases exercise the
    top-level algebra: complement negates the predicate, intersect conjoins two.
    """
    spec, pred = draw(_specs())
    mode = draw(st.sampled_from(["plain", "complement", "intersect"]))
    if mode == "plain":
        return validator(spec), pred
    if mode == "complement":
        return complement(spec), lambda x: not pred(x)
    spec2, pred2 = draw(_specs())
    return intersect(spec, spec2), lambda x: pred(x) and pred2(x)


@settings(max_examples=400, deadline=None)
@given(case=_cases(), value=_values())
def test_walk_matches_denotation(
    case: tuple[CompiledValidator, Pred], value: object
) -> None:
    compiled, predicate = case
    expected = predicate(value)
    assert compiled.is_valid(value) is expected
    try:
        compiled.validate(value)
        raised = False
    except ValidationError:
        raised = True
    assert raised is (not expected)
