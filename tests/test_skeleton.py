from valgebra import CompiledValidator, validator


def test_validator_returns_a_compiled_validator() -> None:
    assert isinstance(validator(int), CompiledValidator)


def test_int_schema_accepts_an_int() -> None:
    assert validator(int).is_valid(3)


def test_int_schema_rejects_a_str() -> None:
    assert not validator(int).is_valid("x")
