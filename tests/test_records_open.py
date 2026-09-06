from valgebra import Validator, anything


def test_records_are_closed_by_default() -> None:
    v = Validator({"name": str, "age?": int})
    assert v.is_valid({"name": "Ada"})
    assert not v.is_valid({"name": "Ada", "extra": 1})


def test_open_admits_undeclared_keys() -> None:
    v = Validator({"name": str, "age?": int}).open()
    assert v.is_valid({"name": "Ada", "extra": 1})
    assert v.is_valid({"name": "Ada"})
    # declared fields are still checked
    assert not v.is_valid({"name": 1})
    assert not v.is_valid({"age": "x"})


def test_close_recloses_an_opened_record() -> None:
    v = Validator({"name": str}).open().close()
    assert not v.is_valid({"name": "Ada", "extra": 1})


def test_open_opens_records_at_every_depth() -> None:
    v = Validator({"user": {"name": str}}).open()
    assert v.is_valid({"user": {"name": "Ada", "role": "admin"}, "meta": 1})


def test_open_leaves_a_missing_required_key_failing() -> None:
    v = Validator({"name": str}).open()
    assert not v.is_valid({"other": 1})


def test_open_record_renders_as_the_record_that_rebuilds_it() -> None:
    # The catch-all renders as the entry it is. `{'name': str, ...}` read better
    # and rebuilt a *different* schema: `...` is a dict key like any other, so
    # the frontend read it back as `Literal[Ellipsis]` and the record came out
    # closed with an odd field.
    opened = Validator({"name": str}).open()
    assert repr(opened) == "{'name': str, anything: anything}"
    assert Validator({"name": str, anything: anything}) == opened
    assert repr(Validator({"name": str})) == "{'name': str}"
