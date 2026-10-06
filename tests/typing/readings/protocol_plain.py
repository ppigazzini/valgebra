"""A protocol that is not runtime-checkable as a schema."""

from typing import Protocol

from typing_extensions import reveal_type

from valgebra import Validator


class HasY(Protocol):
    x: int


reveal_type(Validator(HasY))
