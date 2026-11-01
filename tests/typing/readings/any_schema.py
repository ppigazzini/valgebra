"""`Any` as a schema."""

from typing import Any, reveal_type

from valgebra import Validator

reveal_type(Validator(Any))
