import pytest

from valgebra import ValidationError, Validator


def test_sequence_accepts_a_homogeneous_list() -> None:
    assert Validator([int]).is_valid([1, 2, 3])
    assert Validator([int]).is_valid([])


def test_sequence_rejects_a_bad_element() -> None:
    with pytest.raises(ValidationError) as info:
        Validator([int]).validate([1, "x", 3])
    assert info.value.code == "int_type"
    assert info.value.path == (1,)


def test_sequence_ellipsis_form_is_a_list() -> None:
    assert Validator([int, ...]).is_valid([1, 2])
    assert not Validator([int, ...]).is_valid((1, 2))


def test_sequence_rejects_a_non_list() -> None:
    assert not Validator([int]).is_valid((1, 2))


def test_tuple_matches_positionally() -> None:
    schema = Validator(tuple[int, str])
    assert schema.is_valid((1, "a"))
    assert not schema.is_valid((1, 2))


def test_tuple_length_must_match() -> None:
    schema = Validator(tuple[int, str])
    with pytest.raises(ValidationError) as info:
        schema.validate((1,))
    assert info.value.code == "tuple_length"


def test_set_accepts_a_homogeneous_set() -> None:
    assert Validator(set[int]).is_valid({1, 2, 3})
    assert not Validator(set[int]).is_valid({1, "x"})


def test_set_rejects_a_non_set() -> None:
    assert not Validator(set[int]).is_valid([1, 2])


def test_mapping_checks_keys_and_values() -> None:
    schema = Validator({str: int})
    assert schema.is_valid({"a": 1, "b": 2})
    assert not schema.is_valid({"a": "x"})
    assert not schema.is_valid({1: 1})


def test_mapping_rejects_a_non_dict() -> None:
    assert not Validator({str: int}).is_valid([("a", 1)])


def test_deep_nesting_reports_a_located_failure() -> None:
    schema = Validator({"items": [{"id": int}]})
    with pytest.raises(ValidationError) as info:
        schema.validate({"items": [{"id": 1}, {"id": "x"}]})
    assert info.value.code == "int_type"
    assert info.value.path == ("items", 1, "id")
