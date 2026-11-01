"""A runtime-checkable protocol as a schema."""

from typing import Protocol, reveal_type, runtime_checkable

from valgebra import Validator


@runtime_checkable
class HasX(Protocol):
    x: int


reveal_type(Validator(HasX))
