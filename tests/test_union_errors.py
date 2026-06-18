import pytest

from valgebra import ValidationError, Validator, union


def test_union_reports_the_branch_that_descends_furthest() -> None:
    # The value is a dict shaped like the first branch but with a bad field.
    schema = union({"tag": str, "n": int}, {"tag": str, "s": str})
    with pytest.raises(ValidationError) as info:
        schema.validate({"tag": "x", "n": "oops"})
    items = [(item["code"], item["path"]) for item in info.value.errors]
    assert items == [("int_type", ("n",))]


def test_union_reports_a_nested_branch_failure() -> None:
    schema = union(int, {"value": int})
    with pytest.raises(ValidationError) as info:
        schema.validate({"value": "x"})
    assert info.value.code == "int_type"
    assert info.value.path == ("value",)


def test_flat_union_falls_back_to_a_union_error() -> None:
    with pytest.raises(ValidationError) as info:
        union(int, str).validate(1.0)
    assert info.value.code == "union_error"
    assert "int" in info.value.expected
    assert "str" in info.value.expected


def test_optional_failure_on_a_present_value_descends() -> None:
    schema = Validator(list[int] | None)
    with pytest.raises(ValidationError) as info:
        schema.validate([1, "x"])
    assert info.value.code == "int_type"
    assert info.value.path == (1,)
