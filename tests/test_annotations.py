import pytest

from valgebra import ValidationError, validator


def test_list_annotation_is_a_sequence() -> None:
    assert validator(list[int]).is_valid([1, 2, 3])
    assert not validator(list[int]).is_valid([1, "x"])
    assert not validator(list[int]).is_valid((1, 2))


def test_set_annotation() -> None:
    assert validator(set[int]).is_valid({1, 2})
    assert not validator(set[int]).is_valid({1, "x"})


def test_dict_annotation_is_a_mapping() -> None:
    schema = validator(dict[str, int])
    assert schema.is_valid({"a": 1, "b": 2})
    assert not schema.is_valid({"a": "x"})
    assert not schema.is_valid({1: 1})


def test_fixed_tuple_annotation_matches_positionally() -> None:
    schema = validator(tuple[int, str])
    assert schema.is_valid((1, "a"))
    assert not schema.is_valid((1, 2))
    assert not schema.is_valid((1,))


def test_nested_generic_annotations() -> None:
    schema = validator(list[dict[str, int]])
    assert schema.is_valid([{"a": 1}, {"b": 2}])
    assert not schema.is_valid([{"a": "x"}])


def test_generic_annotation_reports_located_failure() -> None:
    with pytest.raises(ValidationError) as info:
        validator(list[int]).validate([1, "x"])
    assert info.value.code == "int_type"
    assert info.value.path == (1,)
