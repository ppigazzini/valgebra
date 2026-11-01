"""`union` over two validators of different types."""

from typing import reveal_type

from valgebra import Validator, union

reveal_type(union(Validator(int), Validator(str)))
