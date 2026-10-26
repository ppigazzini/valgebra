from valgebra import Validator, anything, complement, union


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
    # The catch-all renders as the entry it is. `{'name': str, ...}` would read
    # better and rebuild a *different* schema: `...` is a dict key like any
    # other, so the frontend reads it back as `Literal[Ellipsis]` and the record
    # comes out closed with an odd field.
    opened = Validator({"name": str}).open()
    assert repr(opened) == "{'name': str, anything: anything}"
    assert Validator({"name": str, anything: anything}) == opened
    assert repr(Validator({"name": str})) == "{'name': str}"


def test_an_opened_mapping_renders_as_its_clauses_written_again() -> None:
    # Opening folds the region no clause claims into the clause carrying the
    # same value, and two orders of one clause list are one schema: the opened
    # map renders as those clauses do when written out.
    opened = Validator({int: object, str: int}).open()
    written = Validator({union(int, complement(union(int, str))): object, str: int})
    assert opened == written
    assert repr(opened) == repr(written)


def test_a_closed_map_names_each_key_schema_once() -> None:
    """`close` folds the clauses it leaves over one key into one over their join.

    The `complement(str)` clause closes to the `bytes` keys alone, beside the
    `bytes: int` clause: one key schema, and a key is read by some clause, so the
    two are `bytes: anything`. Left as two, the render printed a dict display
    with the key twice, which reads back as its last value.
    """
    closed = Validator(
        {str: anything, bytes: int, complement(Validator(str)): anything}
    ).close()
    assert closed == Validator({bytes: anything})
    assert closed.is_valid({b"x": "y"})
    back = Validator(eval(repr(closed), {"bytes": bytes, "anything": anything}))  # noqa: S307 - the render under test
    assert back == closed
