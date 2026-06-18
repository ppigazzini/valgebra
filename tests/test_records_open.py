from valgebra import validator


def test_records_are_closed_by_default() -> None:
    v = validator({"name": str, "age?": int})
    assert v.is_valid({"name": "Ada"})
    assert not v.is_valid({"name": "Ada", "extra": 1})


def test_open_admits_undeclared_keys() -> None:
    v = validator({"name": str, "age?": int}).open()
    assert v.is_valid({"name": "Ada", "extra": 1})
    assert v.is_valid({"name": "Ada"})
    # declared fields are still checked
    assert not v.is_valid({"name": 1})
    assert not v.is_valid({"age": "x"})


def test_close_recloses_an_opened_record() -> None:
    v = validator({"name": str}).open().close()
    assert not v.is_valid({"name": "Ada", "extra": 1})


def test_open_opens_records_at_every_depth() -> None:
    v = validator({"user": {"name": str}}).open()
    assert v.is_valid({"user": {"name": "Ada", "role": "admin"}, "meta": 1})


def test_open_leaves_a_missing_required_key_failing() -> None:
    v = validator({"name": str}).open()
    assert not v.is_valid({"other": 1})


def test_open_record_renders_with_an_open_marker() -> None:
    assert repr(validator({"name": str}).open()) == "{'name': str, ...}"
    assert repr(validator({"name": str})) == "{'name': str}"
