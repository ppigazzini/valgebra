import pytest

from valgebra import ValidationError, Validator, union


def test_literal_accepts_the_exact_value() -> None:
    assert Validator(5).is_valid(5)
    assert Validator("red").is_valid("red")
    assert Validator(b"x").is_valid(b"x")


def test_literal_rejects_a_different_value() -> None:
    assert not Validator(5).is_valid(6)
    assert not Validator("red").is_valid("green")


def test_literal_is_a_typed_singleton() -> None:
    # Python's == conflates 1, True, and 1.0; a literal keeps them distinct by
    # also requiring the same type.
    assert Validator(1).is_valid(1)
    assert not Validator(1).is_valid(True)
    assert not Validator(1).is_valid(1.0)
    assert Validator(True).is_valid(True)
    assert not Validator(True).is_valid(1)
    assert Validator(1.0).is_valid(1.0)
    assert not Validator(1.0).is_valid(1)


def test_literal_failure_reports_its_code() -> None:
    with pytest.raises(ValidationError) as info:
        Validator(5).validate(6)
    assert info.value.code == "literal_error"


def test_two_spellings_of_one_constant_are_one_node() -> None:
    # A constant is a value, not an object: two equal strings built at run time
    # are one pooled constant, so the schema built from them is one node and the
    # containment rules see it as one.
    # Built rather than written, so the compiler cannot fold the two into one
    # object; ruff would rewrite a literal join back into a constant.
    left = "".join(chr(byte) for byte in b"code")
    right = "".join(chr(byte) for byte in b"code")
    assert left is not right
    assert Validator(union(left, right)).is_equivalent(Validator(left))
    assert Validator(left) == Validator(right)


def test_the_literal_rule_decides_which_constants_are_one() -> None:
    # The same rule a literal is checked by: same exact type, and equal. `1` and
    # `True` compare equal and are not one constant; `0.0` and `-0.0` are.
    assert not Validator(union(1, True)).is_equivalent(Validator(1))
    assert Validator(union(0.0, -0.0)).is_equivalent(Validator(0.0))
    # A `nan` equals nothing, itself included, so it is never merged with
    # another constant -- and a literal naming one admits no value at all, which
    # is what `==` on it says.
    nan = float("nan")
    assert not Validator(nan).is_valid(nan)
