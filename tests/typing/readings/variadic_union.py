"""`union` over two validators of different types."""

from typing_extensions import reveal_type

from valgebra import Validator, union

reveal_type(union(Validator(int), Validator(str)))
