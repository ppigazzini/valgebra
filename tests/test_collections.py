import pytest

from valgebra import ValidationError, validator


def test_frozenset_annotation() -> None:
    schema = validator(frozenset[int])
    assert schema.is_valid(frozenset({1, 2, 3}))
    assert schema.is_valid(frozenset())
    assert not schema.is_valid(frozenset({1, "x"}))


def test_frozenset_rejects_a_plain_set() -> None:
    assert not validator(frozenset[int]).is_valid({1, 2})


def test_variadic_tuple_accepts_any_length() -> None:
    schema = validator(tuple[int, ...])
    assert schema.is_valid(())
    assert schema.is_valid((1,))
    assert schema.is_valid((1, 2, 3))


def test_variadic_tuple_rejects_a_bad_element() -> None:
    with pytest.raises(ValidationError) as info:
        validator(tuple[int, ...]).validate((1, "x", 3))
    assert info.value.code == "int_type"
    assert info.value.path == (1,)


def test_variadic_tuple_rejects_a_non_tuple() -> None:
    assert not validator(tuple[int, ...]).is_valid([1, 2, 3])


def test_fixed_and_variadic_tuples_are_distinct() -> None:
    assert validator(tuple[int, str]).is_valid((1, "a"))
    assert not validator(tuple[int, str]).is_valid((1, 2, 3))
    assert validator(tuple[int, ...]).is_valid((1, 2, 3))


def test_mapping_error_path_tolerates_a_key_whose_str_raises() -> None:
    # The aggregating walk labels each entry by its key; a key whose str()
    # raises falls back to a repr summary rather than crashing.
    class Unprintable:
        def __str__(self) -> str:
            raise RuntimeError("no str")

    with pytest.raises(ValidationError):
        validator(dict[int, int]).validate({Unprintable(): 1})
