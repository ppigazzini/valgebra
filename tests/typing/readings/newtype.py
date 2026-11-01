"""A `NewType` as a schema."""

from typing import NewType, reveal_type

from valgebra import Validator

UserId = NewType("UserId", int)

reveal_type(Validator(UserId))
