"""Operator surface on a compiled validator: `in`, `|`, and structural `==`.

`obj in v` is the operator form of membership; `a | b` is union (the one operator
typing already uses for it); `==`/`hash` are *syntactic* (schema shape), distinct
from the semantic `is_equivalent`.
"""

from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from valgebra import Validator, complement, intersection, recursive, union

# A spread of schema specs and probe values to exercise membership.
_SPECS = [int, str, bool, float, list[int], dict[str, int], int | None, {"a": int}]
_VALUES = [1, "x", True, 1.0, [1, 2], {"a": 1}, None, object(), {"a": "no"}]


@pytest.mark.parametrize("spec", _SPECS)
@pytest.mark.parametrize("value", _VALUES)
def test_in_matches_is_valid(spec: object, value: object) -> None:
    v = Validator(spec)
    assert (value in v) is v.is_valid(value)


def test_in_reads_as_membership() -> None:
    v = Validator(list[int])
    assert [1, 2, 3] in v
    assert [1, "x"] not in v


@given(
    a=st.sampled_from(_SPECS),
    b=st.sampled_from(_SPECS),
    value=st.sampled_from(_VALUES),
)
def test_or_operator_is_union(a: object, b: object, value: object) -> None:
    # the validator is the left operand, so `|` always dispatches to __or__
    # (a bare typing union on the left would handle `|` itself); __ror__ is
    # covered separately.
    assert (Validator(a) | b).is_valid(value) is union(a, b).is_valid(value)


def test_ror_when_left_operand_defers() -> None:
    # `None | validator` -> NoneType has no __or__ for a Validator, so __ror__.
    v = None | Validator(int)
    assert v.is_valid(None)
    assert v.is_valid(3)
    assert not v.is_valid("x")


def test_or_composes_with_the_named_combinators() -> None:
    schema = Validator(int) | str | None
    assert schema.is_equivalent(union(int, str, None))


def test_eq_is_structural_and_reflexive() -> None:
    assert Validator(int) == Validator(int)
    assert Validator({"a": int, "b?": str}) == Validator({"a": int, "b?": str})
    assert Validator(int) != Validator(str)
    # a validator always equals itself, even pooling a value unequal to itself
    nan = Validator(float("nan"))
    assert nan == nan  # noqa: PLR0124 -- the self-comparison is the point


def test_eq_is_syntactic_not_semantic() -> None:
    # same set, different shape: equal under is_equivalent, not under ==
    assert union(int, str) != union(str, int)
    assert union(int, str).is_equivalent(union(str, int))
    assert union(bool, int) != Validator(int)
    assert union(bool, int).is_equivalent(int)


def test_eq_against_a_non_validator_is_false() -> None:
    assert Validator(int) != 5
    assert Validator(int) != "int"
    assert (Validator(int) == object()) is False


def test_validators_are_hashable_and_consistent() -> None:
    # equal validators hash alike and collapse in a set
    assert hash(Validator(int)) == hash(Validator(int))
    members = {Validator(int), Validator(int), Validator(str)}
    assert len(members) == 2
    # usable as dict keys
    table = {Validator(list[int]): "ints"}
    assert table[Validator(list[int])] == "ints"


def test_eq_and_hash_over_combinators_and_recursion() -> None:
    assert complement(int) == complement(int)
    assert intersection(int, str) == intersection(int, str)
    tree = recursive(lambda t: {"v": int, "next?": t})
    again = recursive(lambda t: {"v": int, "next?": t})
    assert tree == again
    assert hash(tree) == hash(again)


def test_any_atom_supports_the_operators() -> None:
    v = Validator(Any)
    assert object() in v
    assert (v | int).is_valid("anything")
