"""Fixed-length lists via the native `[A, B]` literal.

A list literal with no trailing `...` and more than one element is a
fixed-length list matched positionally — the one sequence shape typing cannot
spell (`list[A, B]` is illegal). `[T]` stays the homogeneous idiom, `[]` is the
empty-list schema, and `tuple[A, B]` is the tuple counterpart.
"""

import pytest

from valgebra import ValidationError, Validator


def test_fixed_list_matches_positionally() -> None:
    schema = Validator([int, str, int])
    assert schema.is_valid([1, "a", 2])
    assert not schema.is_valid([1, 2, 3])


def test_fixed_list_length_must_match() -> None:
    schema = Validator([int, str])
    assert not schema.is_valid([1])
    assert not schema.is_valid([1, "a", 2])


def test_fixed_list_requires_a_list_not_a_tuple() -> None:
    schema = Validator([int, str])
    assert schema.is_valid([1, "a"])
    assert not schema.is_valid((1, "a"))


def test_single_element_list_is_homogeneous_not_fixed() -> None:
    # [T] keeps the established "list of T" idiom, any length.
    schema = Validator([int])
    assert schema.is_valid([1, 2, 3])
    assert schema.is_valid([])


def test_fixed_list_length_failure_reports_its_code() -> None:
    with pytest.raises(ValidationError) as info:
        Validator([int, str]).validate([1])
    assert info.value.code == "list_length"


def test_fixed_list_element_failure_reports_its_path() -> None:
    with pytest.raises(ValidationError) as info:
        Validator([int, str]).validate([1, 2])
    assert info.value.code == "string_type"
    assert info.value.path == (1,)


def test_fixed_list_round_trips_through_repr() -> None:
    assert repr(Validator([int, str, int])) == "[int, str, int]"


def test_empty_list_literal_matches_only_the_empty_list() -> None:
    schema = Validator([])
    assert schema.is_valid([])
    assert not schema.is_valid([1])


def test_fixed_list_composes_in_a_validator() -> None:
    schema = Validator([[int, str]])  # a homogeneous list of fixed [int, str]
    assert schema.is_valid([[1, "a"], [2, "b"]])
    assert not schema.is_valid([[1, "a"], [2, 3]])


def test_set_literal_is_rejected_pointing_to_set_t() -> None:
    with pytest.raises(NotImplementedError, match="set"):
        Validator({int})


def test_tuple_literal_is_rejected_pointing_to_tuple_form() -> None:
    with pytest.raises(NotImplementedError, match="tuple"):
        Validator((int, str))
