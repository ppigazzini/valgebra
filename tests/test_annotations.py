import pytest

from valgebra import ValidationError, Validator


def test_list_annotation_is_a_sequence() -> None:
    assert Validator(list[int]).is_valid([1, 2, 3])
    assert not Validator(list[int]).is_valid([1, "x"])
    assert not Validator(list[int]).is_valid((1, 2))


def test_set_annotation() -> None:
    assert Validator(set[int]).is_valid({1, 2})
    assert not Validator(set[int]).is_valid({1, "x"})


def test_dict_annotation_is_a_mapping() -> None:
    schema = Validator(dict[str, int])
    assert schema.is_valid({"a": 1, "b": 2})
    assert not schema.is_valid({"a": "x"})
    assert not schema.is_valid({1: 1})


def test_fixed_tuple_annotation_matches_positionally() -> None:
    schema = Validator(tuple[int, str])
    assert schema.is_valid((1, "a"))
    assert not schema.is_valid((1, 2))
    assert not schema.is_valid((1,))


def test_nested_generic_annotations() -> None:
    schema = Validator(list[dict[str, int]])
    assert schema.is_valid([{"a": 1}, {"b": 2}])
    assert not schema.is_valid([{"a": "x"}])


def test_generic_annotation_reports_located_failure() -> None:
    with pytest.raises(ValidationError) as info:
        Validator(list[int]).validate([1, "x"])
    assert info.value.code == "int_type"
    assert info.value.path == (1,)
