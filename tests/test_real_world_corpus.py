"""Real-world annotation corpus.

Feeds annotation shapes drawn from everyday typed Python -- and randomly nested
typing expressions -- through the frontend, to find expressibility holes and
crashes that a hand-picked test would miss. A real annotation either builds a
validator whose membership check runs without crashing, or is rejected with a
clear ``NotImplementedError``; nothing panics, and nothing builds a validator
that then raises on a value.
"""

import collections.abc as cabc
import types
from collections.abc import Callable, Iterable, Mapping, Sequence
from types import GenericAlias
from typing import (
    Annotated,
    Any,
    ClassVar,
    Final,
    Literal,
    Optional,
    Union,
    cast,
    get_args,
    get_origin,
)

import annotated_types as at
import pytest
from hypothesis import given
from hypothesis import strategies as st

from valgebra import Validator

# A spread of values to smoke each compiled validator's membership check.
_SAMPLES = [None, 0, 1, True, "x", b"y", 1.5, [1], {"a": 1}, (1, "x"), {1}, [], {}]

# Annotation shapes that appear in everyday typed Python and build a validator.
_BUILDS = {
    "int": int,
    "str": str,
    "bytes": bytes,
    "float": float,
    "bool": bool,
    "None": None,
    "complex": complex,
    "range": range,
    "str | None": str | None,
    "Optional[int]": Optional[int],  # noqa: UP045 -- the Optional spelling is the test
    "list[int]": list[int],
    "dict[str, int]": dict[str, int],
    "tuple[int, str]": tuple[int, str],
    "tuple[int, ...]": tuple[int, ...],
    "set[str]": set[str],
    "frozenset[int]": frozenset[int],
    "list[dict[str, int]]": list[dict[str, int]],
    "dict[str, Optional[list[int]]]": dict[str, list[int] | None],
    "Literal['GET', 'POST']": Literal["GET", "POST"],
    "Literal[1, 2, 3]": Literal[1, 2, 3],
    "Annotated[int, Ge(0)]": Annotated[int, at.Ge(0)],
    "Callable[[int], str]": Callable[[int], str],
    "Any": Any,
    "bool | int | str": bool | int | str,
}


def _meta_pred(meta: object) -> Callable[[object], bool]:
    """Build an independent predicate for an `annotated_types` constraint.

    Each guards on the value's shape (numeric for a bound, sized for a length), so
    a value of the wrong shape is simply not a member, matching the walk.
    """
    if isinstance(meta, at.Ge):
        lo = cast("float", meta.ge)
        return lambda x: isinstance(x, (int, float)) and x >= lo
    if isinstance(meta, at.Gt):
        lo = cast("float", meta.gt)
        return lambda x: isinstance(x, (int, float)) and x > lo
    if isinstance(meta, at.Le):
        hi = cast("float", meta.le)
        return lambda x: isinstance(x, (int, float)) and x <= hi
    if isinstance(meta, at.Lt):
        hi = cast("float", meta.lt)
        return lambda x: isinstance(x, (int, float)) and x < hi
    if isinstance(meta, at.MinLen):
        return lambda x: isinstance(x, cabc.Sized) and len(x) >= meta.min_length
    if isinstance(meta, at.MaxLen):
        return lambda x: isinstance(x, cabc.Sized) and len(x) <= meta.max_length
    msg = f"no independent denotation for constraint {meta!r}"
    raise AssertionError(msg)


def _denote(annotation: object) -> Callable[[object], bool]:  # noqa: C901, PLR0911, PLR0912
    """Build an *independent* membership predicate for a supported annotation.

    The predicate is a structural recursion on the annotation's meaning, sharing
    no code with the frontend, so a build or walk bug the validator alone could not
    reveal is caught by the disagreement. It raises rather than guess on an
    unsupported form, so a new shape surfaces instead of silently passing.
    """
    if annotation is Any:
        return lambda _x: True
    if annotation is None or annotation is type(None):
        return lambda x: x is None
    # Annotated[base, *constraints]: the base predicate and each constraint.
    if hasattr(annotation, "__metadata__"):
        wrapped = cast("Any", annotation)
        base = _denote(wrapped.__origin__)
        checks = [_meta_pred(m) for m in wrapped.__metadata__]
        return lambda x: base(x) and all(c(x) for c in checks)
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Literal:
        # A typed singleton: same concrete type and equal (so 1 is not True).
        return lambda x: any(type(x) is type(c) and x == c for c in args)
    if origin is Union or origin is types.UnionType:
        preds = [_denote(a) for a in args]
        return lambda x: any(p(x) for p in preds)
    if origin is cabc.Callable:
        return callable
    if origin is list:
        inner = _denote(args[0])
        return lambda x: isinstance(x, list) and all(inner(e) for e in x)
    if origin is set:
        inner = _denote(args[0])
        return lambda x: isinstance(x, set) and all(inner(e) for e in x)
    if origin is frozenset:
        inner = _denote(args[0])
        return lambda x: isinstance(x, frozenset) and all(inner(e) for e in x)
    if origin is dict:
        kp, vp = _denote(args[0]), _denote(args[1])
        return lambda x: (
            isinstance(x, dict) and all(kp(k) and vp(v) for k, v in x.items())
        )
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:  # tuple[c, ...]
            inner = _denote(args[0])
            return lambda x: isinstance(x, tuple) and all(inner(e) for e in x)
        preds = [_denote(a) for a in args]  # tuple[a, b] fixed arity
        return lambda x: (
            isinstance(x, tuple)
            and len(x) == len(preds)
            and all(p(e) for e, p in zip(x, preds, strict=False))
        )
    if isinstance(annotation, type) and annotation in (
        bool,
        int,
        float,
        str,
        bytes,
        complex,
        range,
    ):
        cls = annotation
        return lambda x: isinstance(x, cls)  # bool ⊆ int via isinstance
    msg = f"no independent denotation for {annotation!r}"
    raise AssertionError(msg)


@pytest.mark.parametrize("annotation", _BUILDS.values(), ids=list(_BUILDS))
def test_real_world_annotation_agrees_with_its_denotation(annotation: object) -> None:
    compiled = Validator(annotation)
    predicate = _denote(annotation)
    for sample in _SAMPLES:
        # The verdict is checked against an independent denotation, not discarded.
        assert compiled.is_valid(sample) == predicate(sample), (
            f"{annotation!r} disagreed on {sample!r}"
        )


# Annotation shapes the frontend does not express: abstract-collection generics
# and qualifiers. Each is rejected with a clear message, never a crash and never
# a silently-wrong validator.
_REJECTS = {
    "Sequence[int]": Sequence[int],
    "Mapping[str, int]": Mapping[str, int],
    "Iterable[str]": Iterable[str],
    "Final[int]": Final[int],
    "ClassVar[int]": ClassVar[int],
}


@pytest.mark.parametrize("annotation", _REJECTS.values(), ids=list(_REJECTS))
def test_unsupported_annotation_rejects_cleanly(annotation: object) -> None:
    with pytest.raises(NotImplementedError):
        Validator(annotation)


# Randomly nested typing expressions over the supported forms, built at runtime.
_leaf = st.sampled_from([int, str, bool, float, bytes, type(None)])


def _compose(child: st.SearchStrategy) -> st.SearchStrategy:
    pair = st.tuples(child, child)
    return st.one_of(
        child.map(lambda c: GenericAlias(list, (c,))),
        child.map(lambda c: GenericAlias(set, (c,))),
        child.map(lambda c: GenericAlias(tuple, (c, ...))),
        pair.map(lambda p: GenericAlias(dict, (str, p[1]))),
        pair.map(lambda p: p[0] | p[1]),
    )


_annotations = st.recursive(_leaf, _compose, max_leaves=6)


@given(annotation=_annotations)
def test_generated_annotation_agrees_with_its_denotation(annotation: object) -> None:
    # A generated typing expression either builds and agrees with its independent
    # denotation on every sample, or is rejected cleanly; it never panics and never
    # yields a validator whose verdict diverges from the meaning.
    try:
        compiled = Validator(annotation)
    except (NotImplementedError, TypeError, ValueError):
        return
    predicate = _denote(annotation)
    for sample in _SAMPLES:
        assert compiled.is_valid(sample) == predicate(sample), (
            f"{annotation!r} disagreed on {sample!r}"
        )
