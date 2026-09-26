"""`Any` as a schema."""

from typing import Any

from typing_extensions import reveal_type

from valgebra import Validator

reveal_type(Validator(Any))
