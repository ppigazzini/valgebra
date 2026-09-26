"""`ensure` on a receiver annotated bare."""

from typing_extensions import reveal_type

from valgebra import Validator


def probe(schema: Validator, value: object) -> None:
    reveal_type(schema.ensure(value))
