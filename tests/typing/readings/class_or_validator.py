"""A validator's method, on a class `|` a validator."""

from typing_extensions import reveal_type

from valgebra import Validator

reveal_type((int | Validator(str)).is_valid(1))
