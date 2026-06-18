import copy
from typing import Annotated, Literal

import annotated_types as at

from valgebra import Validator, recursive, union


def test_copy_yields_an_equivalent_validator() -> None:
    v = Validator({"name": str, "age?": int})
    c = copy.copy(v)
    assert c is not v
    assert c.is_valid({"name": "Ada"})
    assert not c.is_valid({"name": 1})
    assert repr(c) == repr(v)


def test_deepcopy_yields_an_equivalent_validator() -> None:
    v = Validator(list[dict[str, int]])
    d = copy.deepcopy(v)
    assert d is not v
    assert d.is_valid([{"a": 1}])
    assert not d.is_valid([{"a": "x"}])
    assert repr(d) == repr(v)


def test_copy_preserves_pooled_literals_and_predicates() -> None:
    lit = Validator(Literal["x", "y"])
    assert copy.deepcopy(lit).is_valid("x")
    assert not copy.deepcopy(lit).is_valid("z")

    refined = Validator(Annotated[int, at.Predicate(lambda n: n > 0)])
    assert copy.copy(refined).is_valid(3)
    assert not copy.copy(refined).is_valid(-1)


def test_copy_preserves_recursion() -> None:
    tree = recursive(lambda t: {"value": int, "left?": t})
    c = copy.deepcopy(tree)
    assert c.is_valid({"value": 1, "left": {"value": 2}})
    assert not c.is_valid({"value": "x"})


def test_copy_of_a_combinator() -> None:
    v = union(int, str)
    assert copy.copy(v).is_valid("x")
    assert not copy.copy(v).is_valid(1.0)
