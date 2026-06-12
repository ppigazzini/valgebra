from valgebra import lax, strict, validator


def test_records_are_strict_by_default() -> None:
    v = validator({"name": str, "age?": int})
    assert v.is_valid({"name": "Ada"})
    assert not v.is_valid({"name": "Ada", "extra": 1})


def test_lax_admits_undeclared_keys() -> None:
    v = lax(validator({"name": str, "age?": int}))
    assert v.is_valid({"name": "Ada", "extra": 1})
    assert v.is_valid({"name": "Ada"})
    # declared fields are still checked
    assert not v.is_valid({"name": 1})
    assert not v.is_valid({"age": "x"})


def test_strict_closes_an_opened_record() -> None:
    v = strict(lax(validator({"name": str})))
    assert not v.is_valid({"name": "Ada", "extra": 1})


def test_lax_opens_records_at_every_depth() -> None:
    v = lax(validator({"user": {"name": str}}))
    assert v.is_valid({"user": {"name": "Ada", "role": "admin"}, "meta": 1})


def test_lax_leaves_a_missing_required_key_failing() -> None:
    v = lax(validator({"name": str}))
    assert not v.is_valid({"other": 1})


def test_lax_record_renders_with_an_open_marker() -> None:
    assert repr(lax(validator({"name": str}))) == "{'name': str, ...}"
    assert repr(validator({"name": str})) == "{'name': str}"
