"""A `NewType` as a schema."""

from typing import NewType

from typing_extensions import reveal_type

from valgebra import Validator

UserId = NewType("UserId", int)

reveal_type(Validator(UserId))
