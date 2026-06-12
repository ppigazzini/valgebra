from typing import Annotated, Any, Literal

import annotated_types as at
import pytest

from valgebra import anything, complement, intersect, nothing, union, validator


@pytest.mark.parametrize(
    ("schema", "expected"),
    [
        (int, "int"),
        (str, "str"),
        (None, "None"),
        (object, "anything"),
        (Any, "Any"),
        (list[int], "list[int]"),
        (set[int], "set[int]"),
        (frozenset[int], "frozenset[int]"),
        (dict[str, int], "dict[str, int]"),
        (tuple[int, str], "tuple[int, str]"),
        (tuple[int, ...], "tuple[int, ...]"),
        (list[dict[str, int]], "list[dict[str, int]]"),
        (int | str, "int | str"),
        (Literal["a"], "Literal['a']"),
        (Literal["a", "b"], "Literal['a'] | Literal['b']"),
        ({"name": str, "age?": int}, "{'name': str, 'age?': int}"),
        (Annotated[int, at.Ge(0)], "Annotated[int, Ge(0)]"),
        (Annotated[str, at.MinLen(1)], "Annotated[str, MinLen(1)]"),
    ],
)
def test_repr_renders_the_annotation(schema: object, expected: str) -> None:
    assert repr(validator(schema)) == expected


# The namespace the rendered form is evaluated in to re-parse it. It holds every
# name a round-trippable repr can mention.
_ROUNDTRIP_NS = {
    "int": int,
    "str": str,
    "bool": bool,
    "float": float,
    "bytes": bytes,
    "list": list,
    "set": set,
    "frozenset": frozenset,
    "tuple": tuple,
    "dict": dict,
    "Any": Any,
    "Literal": Literal,
    "Annotated": Annotated,
    "Ge": at.Ge,
    "Gt": at.Gt,
    "Le": at.Le,
    "Lt": at.Lt,
    "MinLen": at.MinLen,
    "MaxLen": at.MaxLen,
    "anything": anything,
    "nothing": nothing,
    "union": union,
    "intersect": intersect,
    "complement": complement,
}

# The round-trippable subset: every form whose repr re-parses to the same schema.
# Class and predicate nodes are excluded by design - an instance/object renders
# only its class name and a predicate renders as `Predicate(...)`, neither of
# which reconstructs the original schema.
ROUNDTRIP_SCHEMAS = [
    int,
    None,
    object,
    Any,
    list[int],
    set[int],
    frozenset[int],
    dict[str, int],
    tuple[int, str],
    tuple[int, ...],
    list[dict[str, int]],
    int | str,
    Literal["a"],
    Literal["a", "b"],
    {"name": str, "age?": int},
    Annotated[int, at.Ge(0)],
    Annotated[str, at.MinLen(1)],
    union(int, str),
    intersect(int, complement(bool)),
    complement(int),
]


@pytest.mark.parametrize("schema", ROUNDTRIP_SCHEMAS)
def test_repr_round_trips_through_eval(schema: object) -> None:
    # repr is a fixpoint on this subset: rendering, re-parsing, and rendering
    # again yields the same string, so the printed form really does reconstruct
    # the schema.
    rendered = repr(validator(schema))
    rebuilt = validator(eval(rendered, dict(_ROUNDTRIP_NS)))  # noqa: S307
    assert repr(rebuilt) == rendered
