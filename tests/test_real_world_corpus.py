"""Real-world annotation corpus.

Feeds annotation shapes drawn from everyday typed Python -- and randomly nested
typing expressions -- through the frontend, to find expressibility holes and
crashes that a hand-picked test would miss. A real annotation either builds a
validator whose membership check runs without crashing, or is rejected with a
clear ``NotImplementedError``; nothing panics, and nothing builds a validator
that then raises on a value.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from types import GenericAlias
from typing import (
    Annotated,
    Any,
    ClassVar,
    Final,
    Literal,
    Optional,
)

import annotated_types as at
import pytest
from hypothesis import given
from hypothesis import strategies as st

from valgebra import validator

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


@pytest.mark.parametrize("annotation", _BUILDS.values(), ids=list(_BUILDS))
def test_real_world_annotation_builds_and_runs(annotation: object) -> None:
    compiled = validator(annotation)
    for sample in _SAMPLES:
        compiled.is_valid(sample)  # membership must not raise


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
        validator(annotation)


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
def test_generated_annotation_never_crashes(annotation: object) -> None:
    # A generated typing expression either builds and runs, or is rejected
    # cleanly; it never panics and never yields a validator that raises.
    try:
        compiled = validator(annotation)
    except (NotImplementedError, TypeError, ValueError):
        return
    for sample in _SAMPLES:
        compiled.is_valid(sample)
