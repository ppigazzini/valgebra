from typing import Annotated, Any, Literal

import annotated_types as at
import pytest

from valgebra import validator


@pytest.mark.parametrize(
    ("schema", "expected"),
    [
        (int, "int"),
        (str, "str"),
        (None, "None"),
        (object, "anything"),
        (Any, "Any"),
        (list[int], "list[int]"),
        (set[int], "set[int]"),
        (frozenset[int], "frozenset[int]"),
        (dict[str, int], "dict[str, int]"),
        (tuple[int, str], "tuple[int, str]"),
        (tuple[int, ...], "tuple[int, ...]"),
        (list[dict[str, int]], "list[dict[str, int]]"),
        (int | str, "int | str"),
        (Literal["a"], "Literal['a']"),
        (Literal["a", "b"], "Literal['a'] | Literal['b']"),
        ({"name": str, "age?": int}, "{'name': str, 'age?': int}"),
        (Annotated[int, at.Ge(0)], "Annotated[int, Ge(0)]"),
        (Annotated[str, at.MinLen(1)], "Annotated[str, MinLen(1)]"),
    ],
)
def test_repr_renders_the_annotation(schema: object, expected: str) -> None:
    assert repr(validator(schema)) == expected
