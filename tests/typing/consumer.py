"""A caller's code, type-checked strictly, on the floor and on the current.

`mypy --strict` is what a typed downstream project runs, and the stub is what it
reads. Checking the stub alone says it is internally consistent; it does not say
a caller can *use* the surface without an ignore -- and the two come apart on
ordinary things: a positional-only parameter named in a call, a `TypeVar` that
does not carry through, a bound whose type the checker cannot see.

So this file is the caller. Every public name is used the way the documentation
uses it, and every result goes through `assert_type`. Binding to an annotated
variable is the weaker reading: it asks whether the result is *assignable* to
that type, and `Any` is assignable to everything, so a signature that degraded
to `Any` passed in silence. `assert_type` asks whether the checker's own view of
the expression is that type exactly, and fails on `Any`.

It is not a test: nothing here runs, and pytest does not collect it. The
assertion is the exit code of the two `mypy --strict` runs in the type-check
lane, one on the supported floor and one on the current interpreter, because a
stub can be right for one and wrong for the other.
`tests/test_typed_consumer.py` holds the names below to the ones the stub
declares, so a method added to the surface arrives here with a caller using it.
"""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Literal, TypedDict

if sys.version_info >= (3, 11):
    from typing import assert_type
else:  # the floor, where `typing` does not carry it yet
    from typing_extensions import assert_type

import annotated_types as at

from valgebra import (
    MAX_DEFINITIONS,
    MAX_SCHEMA_DEPTH,
    MAX_SCHEMA_NODES,
    Regex,
    ValidationError,
    Validator,
    __version__,
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


#: The three answers `relation_to` gives. Written out rather than `str`, which
#: is what the stub declares and what a caller can exhaustively branch on.
Relation = Literal["subset", "not_subset", "undecided"]


def build() -> Validator:
    """Every constructor, and the combinators over what they build."""
    scalars = Validator(int)
    assert_type(scalars, Validator)
    shapes = Validator({"name": str, "tags": [str]})
    classes = Validator(Point)
    records = Validator(Row)
    literals = Validator(Literal["a", "b"])
    refined = Validator(Annotated[int, at.Ge(0), at.Le(10)])
    assert_type(union(scalars, shapes, classes, records), Validator)
    assert_type(intersection(literals, refined), Validator)
    assert_type(complement(literals), Validator)
    assert_type(recursive(lambda inner: union(int, [inner])), Validator)
    # The two operator spellings, which are separate signatures in the stub.
    assert_type(scalars | shapes, Validator)
    assert_type(shapes.__ror__(scalars), Validator)
    assert_type(anything, Validator)
    assert_type(nothing, Validator)
    return union(scalars, refined, anything, nothing)


def decide(left: Validator, right: Validator) -> bool:
    """Ask the three relations, each of which answers a `bool`."""
    assert_type(left.is_empty(), bool)
    assert_type(left.is_subtype_of(right), bool)
    assert_type(left.is_equivalent(right), bool)
    # `in` reads `__contains__`, and the data model makes the expression a
    # `bool` whatever the method declares, so the operator is what is written.
    return 1 in left


def check(schema: Validator, value: object, document: bytes) -> None:
    """Walk the membership surface, including the one that returns its argument."""
    assert_type(schema.is_valid(value), bool)
    assert_type(schema.validate(value, fail_fast=True), None)
    assert_type(schema.ensure(value), object)
    # The `TypeVar` carries the argument's own type through, which is the whole
    # reason `ensure` exists beside `validate`. The argument is one nothing can
    # narrow -- a length, which is an `int` and no particular one -- because a
    # checker narrows a literal argument, and an annotated local, to that
    # literal, and the claim here is about the parameter rather than about
    # which reading is right.
    assert_type(Validator(int).ensure(len(document)), int)
    assert_type(schema.load(document), object)
    assert_type(schema.validate_json(document, fail_fast=False), None)
    assert_type(schema.is_valid_json(document), bool)


def reshape(schema: Validator) -> Validator:
    """Reshape through the whole-schema operations, and copy the result."""
    assert_type(schema.open(), Validator)
    assert_type(schema.close(), Validator)
    assert_type(schema.simplify(), Validator)
    assert_type(copy.copy(schema), Validator)
    assert_type(copy.deepcopy(schema), Validator)
    return schema.open().close().simplify()


def report(schema: Validator, value: object) -> tuple[str, tuple[str | int, ...]]:
    """Read the error model as a caller reads it."""
    try:
        schema.validate(value)
    except ValidationError as error:
        assert_type(error.errors, tuple[dict[str, object], ...])
        assert_type(error.code, str)
        assert_type(error.path, tuple[str | int, ...])
        assert_type(error.message, str)
        assert_type(error.expected, str)
        assert_type(error.value, str)
        return error.code, error.path
    return "", ()


def bounds() -> tuple[int, int, int]:
    """Read the published construction limits, which are integers."""
    assert_type(MAX_SCHEMA_DEPTH, int)
    assert_type(MAX_DEFINITIONS, int)
    assert_type(MAX_SCHEMA_NODES, int)
    return MAX_SCHEMA_DEPTH, MAX_DEFINITIONS, MAX_SCHEMA_NODES


def builder() -> Callable[[Validator], object]:
    """Name the shape `recursive` takes, so the checker has to agree on it."""
    return lambda inner: union(None, [inner])


def refine() -> Validator:
    """Build through the marker the package defines itself.

    `Regex` is valgebra's own, not `annotated_types`', so a caller who writes a
    pattern refinement reads this stub rather than a third party's -- which is
    the case the rest of this file does not reach.
    """
    pattern = Regex(r"[a-z]+")
    assert_type(pattern, Regex)
    assert_type(pattern.pattern, str)
    return Validator(Annotated[str, pattern])


def relate(a: Validator, b: Validator) -> Relation:
    """Read the relation surface, which answers in strings rather than bools.

    `relation_to` is the one that tells a refutation from a decline, so its
    return type is what a caller branches on -- and it is the three answers
    rather than `str`, so a checker refuses a branch on a fourth. Bound to a
    `str` this file read as agreeing while saying something weaker, which is
    the reading `assert_type` replaced.
    """
    assert_type(a.relation_to(b), Relation)
    answer = a.relation_to(b)
    if answer == "undecided":
        return answer
    return answer


def released() -> str:
    """Read the version, which is a `str` compiled into the extension."""
    assert_type(__version__, str)
    return __version__
