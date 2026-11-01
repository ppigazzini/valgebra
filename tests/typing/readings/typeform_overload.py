"""A `TypeForm` overload, given an argument that is not a type form."""

from typing import TypeVar, overload, reveal_type

from typing_extensions import TypeForm

_S = TypeVar("_S")


@overload
def read(schema: TypeForm[_S], /) -> _S: ...
@overload
def read(schema: object, /) -> object: ...
def read(schema: object, /) -> object:
    return schema


def probe(schema: object) -> None:
    reveal_type(read(schema))
