import copy
from typing import Annotated, Literal

import annotated_types as at

from valgebra import Validator, recursive, union


def test_a_copy_is_the_validator_itself() -> None:
    """A validator is immutable, so `copy.copy` gives back the object itself.

    The standard library does the same for an immutable builtin, a `tuple` or a
    `frozenset`; a fresh validator over the same schema would hold the same set
    and drop the hash and the tables this one computed.
    """
    v = Validator({"name": str, "age?": int})
    assert copy.copy(v) is v
    assert v.is_valid({"name": "Ada"})
    assert not v.is_valid({"name": 1})


def test_a_deep_copy_is_the_validator_itself() -> None:
    """Nothing in a validator is mutable, so a deep copy is the object too."""
    v = Validator(list[dict[str, int]])
    assert copy.deepcopy(v) is v
    held = [v, v]
    copied = copy.deepcopy(held)
    assert copied is not held
    assert copied[0] is v
    assert copied[1] is v


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
