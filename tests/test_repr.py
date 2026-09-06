import dataclasses
import enum
from typing import Annotated, Any, Literal

import annotated_types as at
import pytest

from valgebra import (
    Validator,
    anything,
    complement,
    intersection,
    nothing,
    recursive,
    union,
)


@pytest.mark.parametrize(
    ("schema", "expected"),
    [
        (int, "int"),
        (str, "str"),
        (bool, "bool"),
        (float, "float"),
        (bytes, "bytes"),
        (None, "None"),
        (object, "anything"),
        (Any, "Any"),
        (list[int], "list[int]"),
        (set[int], "set[int]"),
        (frozenset[int], "frozenset[int]"),
        (dict[str, int], "dict[str, int]"),
        (tuple[int, str], "tuple[int, str]"),
        (tuple[int, ...], "tuple[int, ...]"),
        (tuple[str, int, ...], "tuple[str, int, ...]"),  # ty: ignore[invalid-type-form]
        (list[dict[str, int]], "list[dict[str, int]]"),
        (int | str, "int | str"),
        (Literal["a"], "Literal['a']"),
        (Literal["a", "b"], "Literal['a'] | Literal['b']"),
        ({"name": str, "age?": int}, "{'age?': int, 'name': str}"),
        (Annotated[int, at.Ge(0)], "Annotated[int, Ge(0)]"),
        (Annotated[str, at.MinLen(1)], "Annotated[str, MinLen(1)]"),
        # The nullary product. Python spells the empty subscript `tuple[()]`;
        # `tuple[]` is not an expression at all.
        (tuple[()], "tuple[()]"),
    ],
)
def test_repr_renders_the_annotation(schema: object, expected: str) -> None:
    assert repr(Validator(schema)) == expected


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
    "intersection": intersection,
    "complement": complement,
    "recursive": recursive,
}

# The round-trippable subset: every form whose repr re-parses to the same schema.
# Two are excluded by design and no more: a class renders only its name and a
# predicate renders as `Predicate(...)`, and neither is an expression that
# rebuilds what it names -- a class is an object, and a predicate is a callable.
# Everything else here is, the recursive form and the open record included.
ROUNDTRIP_SCHEMAS = [
    int,
    bool,
    float,
    bytes,
    None,
    nothing,
    object,
    Any,
    list[int],
    set[int],
    frozenset[int],
    dict[str, int],
    tuple[int, str],
    tuple[int, ...],
    tuple[str, int, ...],  # ty: ignore[invalid-type-form]
    list[dict[str, int]],
    int | str,
    Literal["a"],
    Literal["a", "b"],
    {"name": str, "age?": int},
    Annotated[int, at.Ge(0)],
    Annotated[str, at.MinLen(1)],
    union(int, str),
    intersection(int, complement(bool)),
    complement(int),
    # The three forms 21.4 fixed: each rendered as something that was not an
    # expression, or was one that built a different schema.
    tuple[()],
    recursive(lambda t: int | list[t]),  # ty: ignore[invalid-type-form]
    Validator({"name": str}).open(),
    recursive(lambda t: {"value": int, "left?": t}),
]


def test_repr_of_class_and_recursive_forms() -> None:
    # A class renders as its name, which names the class rather than rebuilding
    # it -- the one form that stays a rendering, because a class is an object
    # and not an expression.
    class Color(enum.Enum):
        RED = 1

    @dataclasses.dataclass
    class Point:
        x: int

    assert repr(Validator(Color)) == "Color"
    assert repr(Validator(Point)) == "Point"
    # A class with attributes is a meet of its class atom and a record of them,
    # and a later meet flattens beside that pair: the class is still named, and
    # the other members render as themselves.
    assert repr(Validator(intersection(Point, int))) == "intersection(Point, int)"
    assert (
        repr(recursive(lambda s: {"v": int, "n?": s}))
        == "recursive(lambda X: {'n?': X, 'v': int})"
    )


# A spread of values to witness that two validators accept the same set, rather
# than only that a repr string is stable.
_WITNESS_VALUES = [
    None,
    True,
    False,
    0,
    1,
    -1,
    1.5,
    "x",
    "",
    b"x",
    b"",
    [1],
    [1, "a"],
    [],
    {1},
    {"k": 1},
    (1,),
    (1, "a"),
]


@pytest.mark.parametrize("schema", ROUNDTRIP_SCHEMAS)
def test_repr_round_trips_through_eval(schema: object) -> None:
    # repr is a fixpoint on this subset: rendering, re-parsing, and rendering
    # again yields the same string, so the printed form really does reconstruct
    # the schema.
    rendered = repr(Validator(schema))
    rebuilt = Validator(eval(rendered, dict(_ROUNDTRIP_NS)))  # noqa: S307
    assert repr(rebuilt) == rendered
    # The fixpoint alone would pass for a wrong-but-stable repr; require the
    # reconstructed validator to accept exactly the same values as the original,
    # so the printed form preserves meaning, not just its own shape.
    original = Validator(schema)
    for value in _WITNESS_VALUES:
        assert rebuilt.is_valid(value) == original.is_valid(value)


# --- The nullary combinators repr as their identities -------------------------
#
# A union of no members denotes the bottom and a meet of none denotes the top.
# The repr is the annotation that rebuilds the validator, so it names the
# identity rather than printing the empty join of no members.


@pytest.mark.parametrize(
    ("built", "expected"),
    [(union(), "nothing"), (intersection(), "anything")],
)
def test_a_nullary_combinator_reprs_as_its_identity(
    built: Validator, expected: str
) -> None:
    assert repr(built) == expected


def test_a_nullary_combinator_denotes_its_identity() -> None:
    assert not union().is_valid(1)
    assert intersection().is_valid(1)
