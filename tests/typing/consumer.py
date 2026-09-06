"""A caller's code, type-checked strictly, on the floor and on the current.

`mypy --strict` is what a typed downstream project runs, and the stub is what it
reads. Checking the stub alone says it is internally consistent; it does not say
a caller can *use* the surface without an ignore -- and the two come apart on
ordinary things: a positional-only parameter named in a call, a `TypeVar` that
does not carry through, a bound whose type the checker cannot see.

So this file is the caller. Every public name is used the way the documentation
uses it, with the result bound to an annotated variable so the checker has to
agree about the type rather than infer `Any` and stay quiet. It is not a test:
nothing here asserts, and pytest does not collect it. The assertion is the exit
code of the two `mypy --strict` runs in the type-check lane, one on the
supported floor and one on the current interpreter, because a stub can be right
for one and wrong for the other.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Literal, TypedDict

import annotated_types as at

from valgebra import (
    MAX_DEFINITIONS,
    MAX_SCHEMA_DEPTH,
    MAX_SCHEMA_NODES,
    ValidationError,
    Validator,
    anything,
    complement,
    intersection,
    nothing,
    recursive,
    union,
)

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class Point:
    x: int
    y: int


class Row(TypedDict):
    name: str


def build() -> Validator:
    """Every constructor, and the combinators over what they build."""
    scalars: Validator = Validator(int)
    shapes: Validator = Validator({"name": str, "tags": [str]})
    classes: Validator = Validator(Point)
    records: Validator = Validator(Row)
    literals: Validator = Validator(Literal["a", "b"])
    refined: Validator = Validator(Annotated[int, at.Ge(0), at.Le(10)])
    joined: Validator = union(scalars, shapes, classes, records)
    met: Validator = intersection(literals, refined)
    negated: Validator = complement(met)
    fixpoint: Validator = recursive(lambda inner: union(int, [inner]))
    operator: Validator = scalars | shapes
    return union(joined, negated, fixpoint, operator, anything, nothing)


def decide(left: Validator, right: Validator) -> bool:
    """Ask the three relations, each of which answers a `bool`."""
    empty: bool = left.is_empty()
    below: bool = left.is_subtype_of(right)
    same: bool = left.is_equivalent(right)
    inside: bool = 1 in left
    return empty and below and same and inside


def check(schema: Validator, value: object, document: bytes) -> None:
    """Walk the membership surface, including the two that return their argument."""
    valid: bool = schema.is_valid(value)
    schema.validate(value, fail_fast=True)
    kept: object = schema.ensure(value)
    number: int = Validator(int).ensure(1)  # the TypeVar carries the type through
    loaded: object = schema.load(document)
    schema.validate_json(document, fail_fast=False)
    parsed: bool = schema.is_valid_json(document)
    del valid, kept, number, loaded, parsed


def reshape(schema: Validator) -> Validator:
    """Reshape through the whole-schema operations, and copy the result."""
    opened: Validator = schema.open()
    closed: Validator = opened.close()
    shallow: Validator = copy.copy(closed)
    deep: Validator = copy.deepcopy(shallow)
    return deep


def report(schema: Validator, value: object) -> tuple[str, tuple[str | int, ...]]:
    """Read the error model as a caller reads it."""
    try:
        schema.validate(value)
    except ValidationError as error:
        failures: tuple[dict[str, object], ...] = error.errors
        del failures
        return error.code, error.path
    return "", ()


def bounds() -> tuple[int, int, int]:
    """Read the published construction limits, which are integers."""
    return MAX_SCHEMA_DEPTH, MAX_DEFINITIONS, MAX_SCHEMA_NODES


def builder() -> Callable[[Validator], object]:
    """Name the shape `recursive` takes, so the checker has to agree on it."""
    return lambda inner: union(None, [inner])
