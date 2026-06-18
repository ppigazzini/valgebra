import pytest

from valgebra import Validator


@pytest.mark.parametrize(
    ("schema", "value"),
    [
        (int, 0),
        (int, -7),
        (int, True),  # bool subclasses int, so True is an int
        (float, 3.14),
        (str, "hello"),
        (bytes, b"raw"),
        (bool, False),
        (object, object()),
        (object, 123),
        (None, None),
    ],
)
def test_scalar_accepts_a_member(schema: object, value: object) -> None:
    assert Validator(schema).is_valid(value)


@pytest.mark.parametrize(
    ("schema", "value"),
    [
        (int, 1.0),  # a float is not an int
        (int, "1"),
        (float, 3),  # int and float are disjoint
        (float, "3"),
        (str, b"bytes"),
        (bytes, "text"),
        (bool, 1),  # 1 is an int, not a bool
        (None, 0),
    ],
)
def test_scalar_rejects_a_non_member(schema: object, value: object) -> None:
    assert not Validator(schema).is_valid(value)


def test_bool_is_a_subset_of_int() -> None:
    assert Validator(int).is_valid(True)
    assert Validator(int).is_valid(False)
    assert not Validator(bool).is_valid(1)


def test_int_and_float_are_disjoint() -> None:
    assert not Validator(int).is_valid(2.0)
    assert not Validator(float).is_valid(2)
