"""A runtime-checkable protocol as a schema."""

from typing import Protocol, runtime_checkable

from typing_extensions import reveal_type

from valgebra import Validator


@runtime_checkable
class HasX(Protocol):
    x: int


reveal_type(Validator(HasX))
