import pytest

from valgebra import ValidationError, Validator, nothing


def test_record_accepts_a_matching_dict() -> None:
    user = Validator({"name": str, "age": int})
    assert user.is_valid({"name": "Ada", "age": 36})


def test_record_optional_key_may_be_absent() -> None:
    user = Validator({"name": str, "age?": int})
    assert user.is_valid({"name": "Ada"})
    assert user.is_valid({"name": "Ada", "age": 36})


def test_record_optional_key_is_checked_when_present() -> None:
    user = Validator({"name": str, "age?": int})
    assert not user.is_valid({"name": "Ada", "age": "old"})


def test_record_required_key_must_be_present() -> None:
    user = Validator({"name": str, "age": int})
    with pytest.raises(ValidationError) as info:
        user.validate({"name": "Ada"})
    assert info.value.code == "missing_key"
    assert info.value.path == ("age",)


def test_record_is_closed_by_default() -> None:
    user = Validator({"name": str})
    with pytest.raises(ValidationError) as info:
        user.validate({"name": "Ada", "extra": 1})
    assert info.value.code == "extra_forbidden"


def test_empty_record_matches_only_the_empty_dict() -> None:
    empty = Validator({})
    assert empty.is_valid({})
    assert not empty.is_valid({"a": 1})


def test_record_rejects_a_non_dict() -> None:
    assert not Validator({"name": str}).is_valid(["name"])


def test_nested_record_failure_reports_the_path() -> None:
    schema = Validator({"user": {"name": str}})
    with pytest.raises(ValidationError) as info:
        schema.validate({"user": {"name": 5}})
    assert info.value.code == "string_type"
    assert info.value.path == ("user", "name")


def test_is_valid_rejects_an_extra_key_in_a_closed_record() -> None:
    user = Validator({"name": str})
    assert user.is_valid({"name": "Ada"})
    assert not user.is_valid({"name": "Ada", "extra": 1})


def test_is_valid_rejects_a_non_string_key_in_a_closed_record() -> None:
    # A closed string-keyed record admits only its declared string keys; a
    # non-string key is undeclared.
    user = Validator({"name": str})
    assert not user.is_valid({"name": "Ada", 0: 1})


def test_non_string_key_does_not_fill_a_same_named_field() -> None:
    # A non-string key whose str() matches a declared field name does not fill
    # that field: the real key is not a string, so the required field is absent.
    schema = Validator({"0": int})
    assert schema.is_valid({"0": 1})
    assert not schema.is_valid({0: 1})


def test_open_is_a_function_on_sets() -> None:
    """Equal records open to equal records, which is what makes `open` an operation.

    `{"a?": nothing}` and `{}` admit exactly the empty dict: the field allows the
    key to be absent and admits no value for it, which is what a closed record
    already says of every key it does not name. So they are one record, and
    opening them has to give one record — an operation that mapped equal sets to
    unequal sets would not be part of the algebra at all.
    """
    redundant = Validator({"a?": nothing})
    empty = Validator({})
    for value in ({}, {"x": 1}, {"a": 1}):
        assert redundant.is_valid(value) == empty.is_valid(value), value
        assert redundant.open().is_valid(value) == empty.open().is_valid(value), value
        assert redundant.close().is_valid(value) == empty.close().is_valid(value), value
    assert redundant.open().is_equivalent(empty.open())
    assert redundant.close().is_equivalent(empty.close())


def test_open_leaves_a_mapping_alone() -> None:
    """A mapping keys a *type*, not a name, so it is not a record to open.

    Having no field is not what makes one a mapping — the empty closed record has
    none either. A clause and no field is.
    """
    mapping = Validator(dict[str, int])
    assert mapping.open() == mapping
    assert mapping.close() == mapping
    assert not mapping.open().is_valid({"a": "x"})


def test_open_record_explains_a_failing_field() -> None:
    # An open record admits extra keys, so it fails only on a declared
    # field; the aggregating walk reports that field.
    schema = Validator({"name": str}).open()
    assert schema.is_valid({"name": "Ada", "extra": 1})
    with pytest.raises(ValidationError) as info:
        schema.validate({"name": 1, "extra": 2})
    assert info.value.code == "string_type"


@pytest.mark.parametrize(
    "value",
    [
        {"name": "Ada", "age": 36},  # valid
        {"name": "Ada"},  # required present, optional absent
        {"name": "Ada", "age": 36, "x": 1},  # extra string key
        {"name": "Ada", "age": "old"},  # declared value of wrong type
        {"age": 36},  # missing required key
        {"name": "Ada", 0: 1},  # non-string extra key
        {0: 1},  # only a non-string key
        ["name", "Ada"],  # not a dict at all
    ],
)
def test_is_valid_agrees_with_validate_on_records(value: object) -> None:
    # The bool fast path and the aggregating explain walk must reach the same
    # membership verdict on every shape.
    user = Validator({"name": str, "age?": int})
    fast = user.is_valid(value)
    try:
        user.validate(value)
        slow = True
    except ValidationError:
        slow = False
    assert fast is slow
