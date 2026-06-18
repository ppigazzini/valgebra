import pytest

from valgebra import ValidationError, Validator, fixed_sequence


def test_fixed_sequence_matches_positionally() -> None:
    schema = fixed_sequence(int, str, int)
    assert schema.is_valid([1, "a", 2])
    assert not schema.is_valid([1, 2, 3])


def test_fixed_sequence_length_must_match() -> None:
    schema = fixed_sequence(int, str)
    assert not schema.is_valid([1])
    assert not schema.is_valid([1, "a", 2])


def test_fixed_sequence_requires_a_list_not_a_tuple() -> None:
    schema = fixed_sequence(int, str)
    assert schema.is_valid([1, "a"])
    assert not schema.is_valid((1, "a"))


def test_fixed_sequence_length_failure_reports_its_code() -> None:
    with pytest.raises(ValidationError) as info:
        fixed_sequence(int, str).validate([1])
    assert info.value.code == "list_length"


def test_fixed_sequence_element_failure_reports_its_path() -> None:
    with pytest.raises(ValidationError) as info:
        fixed_sequence(int, str).validate([1, 2])
    assert info.value.code == "string_type"
    assert info.value.path == (1,)


def test_fixed_sequence_round_trips_through_repr() -> None:
    assert repr(fixed_sequence(int, str, int)) == "[int, str, int]"


def test_empty_fixed_sequence_matches_the_empty_list() -> None:
    schema = fixed_sequence()
    assert schema.is_valid([])
    assert not schema.is_valid([1])


def test_fixed_sequence_composes_in_a_validator() -> None:
    schema = Validator([fixed_sequence(int, str)])
    assert schema.is_valid([[1, "a"], [2, "b"]])
    assert not schema.is_valid([[1, "a"], [2, 3]])
