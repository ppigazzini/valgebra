from typing import TypedDict

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


def test_opening_a_mapping_frees_the_keys_no_clause_claims() -> None:
    """Openness is the default of the region no clause claims.

    A clause is a key-type region carrying its own default, so the two operators
    decide the regions the clauses leave over and nothing else. The record case
    is the special one, not the general: a record claims no region, so opening it
    frees every key and closing it refuses every key, which is the whole of what
    a catch-all says.

    A mapping claims one. `dict[str, int]` says what a `str` key maps to and
    leaves every other key-type unclaimed, so opening it has to keep the first
    and free the second — the `str` region is not the operator's to touch.
    """
    mapping = Validator(dict[str, int])
    opened = mapping.open()

    # The region the clause claims reads the same through both operators.
    for schema in (mapping, opened):
        assert schema.is_valid({"a": 1}), schema
        assert not schema.is_valid({"a": "x"}), schema

    # The regions no clause claims: refused where it is closed, free where open.
    assert not mapping.is_valid({1: "x"})
    assert opened.is_valid({1: "x"})

    # So the two are inverse on a mapping, as they are on a record.
    assert opened.close() == mapping


def test_opening_a_record_that_claims_a_region_leaves_one_clause() -> None:
    """A `TypedDict` is a record with a clause, and opening it stays decidable.

    `TypedDict` builds named fields beside `str: anything` for the keys it does
    not name, so opening it frees the key-types that clause leaves over. Those
    two clauses carry one value between them and cover every key, which is one
    clause -- the catch-all a record opened has always had.

    Writing it as two would cost the pair rather than the answer: a clause keyed
    by a complement is a shape the set representation declines, so the same set
    spelled the long way stops being decided. The merge is what keeps `open`
    inside the fragment `docs/15-decidability.md` promises.
    """

    class Rec(TypedDict):
        a: int

    typed = Validator(Rec).open()
    spelled = Validator({"a": int}).open()

    assert repr(typed) == "{'a': int, anything: anything}"
    assert typed.is_equivalent(spelled)
    # The named field is untouched, and every other key is free.
    assert typed.is_valid({"a": 1, "b": "free", 2: None})
    assert not typed.is_valid({"a": "not an int"})


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


class _PlainName(str):
    """A `str` subclass that hashes and compares as its text does."""

    __slots__ = ()


class _OtherHash(str):
    """A `str` subclass whose hash is not its text's, so no dict finds it there."""

    __slots__ = ()

    def __hash__(self) -> int:
        return 0


class _NeverEqual(str):
    """A `str` subclass that equals nothing, so a dict lookup on it misses."""

    __slots__ = ()

    def __eq__(self, other: object) -> bool:
        return False

    __hash__ = str.__hash__  # type: ignore[assignment]


def test_a_record_resolves_a_key_the_way_the_dict_does() -> None:
    """A subclass is the field's key exactly when a dict would find it there.

    The walk interns the declared names and compares by text, which is what the
    exact `str` case is. A subclass carries the field's text and may still be a
    different key: `dict.__getitem__` reaches an entry by hash and then by
    equality, so a subclass that hashes elsewhere or equals nothing is a key of
    its own. Reading it as the field it spells would admit a value the dict
    itself does not carry under that name.
    """
    user = Validator({"name": str})
    assert user.is_valid({_PlainName("name"): "Ada"})
    assert not user.is_valid({_OtherHash("name"): "Ada"})
    assert not user.is_valid({_NeverEqual("name"): "Ada"})
    # The same answer through the explaining walk, which is a second reader of
    # the same key.
    for key in (_OtherHash("name"), _NeverEqual("name")):
        with pytest.raises(ValidationError):
            user.validate({key: "Ada"})
    assert {_PlainName("name"): "Ada"} in user
