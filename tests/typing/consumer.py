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
assertion is the exit code of the `mypy --strict` and pyright runs in the
type-check lane, each on the supported floor and on the current interpreter,
because a stub can be right for one and wrong for the other; `ty check` reads
the file with the rest of the tree. A row here is therefore a reading the three
checkers share. The readings they do not share are one fixture each under
`readings/`, held per checker by `tests/test_checker_readings.py`.
`tests/test_typed_consumer.py` holds the names below to the ones the stub
declares, so a method added to the surface arrives here with a caller using it.
"""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Literal, NoReturn, TypedDict

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


def build() -> Validator[object]:
    """Every constructor, and the combinators over what they build.

    A class reads as the type it names and a validator as its own; a form the
    static language cannot spell -- a record literal, a `Literal`, an
    `Annotated` refinement -- reads as `object`, and so does what a combinator
    builds, since a meet or a complement has no static spelling.
    """
    scalars = Validator(int)
    assert_type(scalars, Validator[int])
    assert_type(Validator(scalars), Validator[int])
    assert_type(Validator(None), Validator[None])
    shapes = Validator({"name": str, "tags": [str]})
    assert_type(shapes, Validator[object])
    classes = Validator(Point)
    assert_type(classes, Validator[Point])
    records = Validator(Row)
    assert_type(records, Validator[Row])
    literals = Validator(Literal["a", "b"])
    assert_type(literals, Validator[object])
    # A union written with `|` is a `types.UnionType`, not a `type`, so it
    # falls to the `object` overload under all three checkers.
    assert_type(Validator(int | None), Validator[object])
    refined = Validator(Annotated[int, at.Ge(0), at.Le(10)])
    assert_type(refined, Validator[object])
    assert_type(union(scalars, shapes, classes, records), Validator[object])
    assert_type(intersection(literals, refined), Validator[object])
    assert_type(complement(literals), Validator[object])
    assert_type(recursive(lambda inner: union(int, [inner])), Validator[object])
    # The two operator spellings, which are separate signatures in the stub. Two
    # typed validators join to the union of their types. A typed one beside an
    # untyped one is `int | object`, which pyright prints unsimplified and the
    # other two print as `object`, so the rows pair like with like.
    assert_type(scalars | Validator(str), Validator[int | str])
    assert_type(shapes | refined, Validator[object])
    assert_type(shapes.__ror__(refined), Validator[object])
    assert_type(anything, Validator[object])
    assert_type(nothing, Validator[NoReturn])
    return union(scalars, refined, anything, nothing)


def decide(left: Validator, right: Validator) -> bool:
    """Ask the three relations, each of which answers a `bool`."""
    assert_type(left.is_empty(), bool)
    assert_type(left.is_subtype_of(right), bool)
    assert_type(left.is_equivalent(right), bool)
    # `in` reads `__contains__`, and the data model makes the expression a
    # `bool` whatever the method declares, so the operator is what is written.
    return 1 in left


def check(schema: Validator[object], value: object, document: bytes) -> None:
    """Walk the membership surface of a validator whose set has no static type.

    Most validators are this receiver: whatever a native form, a refinement or
    a combinator builds. `is_valid` answers a `bool`, and `ensure` returns its
    argument as the argument's own type.
    """
    assert_type(schema.is_valid(value), bool)
    assert_type(schema.validate(value, fail_fast=True), None)
    assert_type(schema.ensure(value), object)
    # The `TypeVar` carries the argument's own type through, which is the whole
    # reason `ensure` exists beside `validate`. The argument is one nothing can
    # narrow -- a length, which is an `int` and no particular one -- because a
    # checker narrows a literal argument, and an annotated local, to that
    # literal, and the claim here is about the parameter rather than about
    # which reading is right.
    assert_type(schema.ensure(len(document)), int)
    assert_type(schema.load(document), object)
    assert_type(schema.validate_json(document, fail_fast=False), None)
    assert_type(schema.is_valid_json(document), bool)


def narrow(value: object, size: int | str, document: bytes) -> None:
    """Read a typed validator's answers as the type its schema names.

    A `True` from `is_valid` narrows the argument and a `False` leaves it as it
    was: `Validator(float)` refuses `1`, which the static `float` admits, so a
    refusal is no evidence that a value is outside the static type.
    """
    counts = Validator(int)
    if counts.is_valid(size):
        assert_type(size, int)
    assert_type(size, int | str)
    assert_type(counts.ensure(value), int)
    assert_type(Validator(list[int]).ensure(value), list[int])
    assert_type(Validator(Row).load(document), Row)


def takes_any(schema: Validator) -> Validator:
    """Take any validator: the bare annotation is `Validator[Any]`."""
    return schema


def takes_untyped(schema: Validator[object]) -> Validator[object]:
    """Take only a validator whose set reads as `object`."""
    return schema


def parameters(typed: Validator[int]) -> None:
    """Write a parameter bare unless it means one set.

    The parameter is invariant, which is what lets the overloads tell an
    untyped receiver from a typed one, so a `Validator[object]` parameter
    refuses a `Validator[int]`. Both ignores below are held: `mypy --strict`
    reports an unused ignore and so does ty, so the refusal is asserted rather
    than tolerated.
    """
    takes_any(typed)
    takes_untyped(typed)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


def reshape(schema: Validator[int]) -> Validator[object]:
    """Reshape through the whole-schema operations, and copy the result.

    `close` and the copies keep the set the annotation names; `open` frees the
    key region no clause claims, so what it builds reads as `object`.
    """
    assert_type(schema.open(), Validator[object])
    assert_type(schema.close(), Validator[int])
    assert_type(schema.simplify(), Validator[int])  # ty: ignore[deprecated]
    assert_type(copy.copy(schema), Validator[int])
    assert_type(copy.deepcopy(schema), Validator[int])
    return schema.open().close()


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
