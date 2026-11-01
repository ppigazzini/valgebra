"""A protocol that is not runtime-checkable as a schema."""

from typing import Protocol, reveal_type

from valgebra import Validator


class HasY(Protocol):
    x: int


reveal_type(Validator(HasY))
