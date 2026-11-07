import itertools
import sys
import typing
from collections.abc import Iterator
from typing import Annotated, Any, Literal, TypedDict

import pytest

from valgebra import ValidationError, Validator, anything, nothing


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


def test_a_presence_qualifier_on_a_literal_field_is_refused() -> None:
    """The `?` on the key is the literal's one optionality marker.

    A qualifier is otherwise unwrapped to the type it qualifies, so `{"a":
    NotRequired[int]}` would build a record that *requires* `a`. Reading the
    qualifier instead would make two markers for one fact, which can disagree.
    """
    extensions = pytest.importorskip("typing_extensions")
    for spec in (
        {"a": extensions.NotRequired[int]},
        {"a?": extensions.Required[int]},
        {"a": extensions.Annotated[extensions.NotRequired[int], "meta"]},
        {"a": extensions.ReadOnly[extensions.NotRequired[int]]},
    ):
        with pytest.raises(NotImplementedError, match='write the key "a\\?"'):
            Validator(spec)
    # `ReadOnly` says nothing about presence and is read as before, and a
    # clause's value has no presence to contradict.
    assert Validator({"a": extensions.ReadOnly[int]}) == Validator({"a": int})
    assert Validator({str: extensions.NotRequired[int]}) == Validator({str: int})


# BOUND: MAX_QUALIFIER_DEPTH
def test_a_field_wrapping_itself_is_refused_by_depth() -> None:
    """Reading through a field's wrappers for a qualifier ends.

    An `Annotated` alias whose `__origin__` names itself wraps itself forever,
    so the reading stops at `MAX_QUALIFIER_DEPTH` and refuses the field.
    """
    looped = Annotated[int, "meta"]
    looped.__origin__ = looped  # ty: ignore[invalid-assignment]
    with pytest.raises(ValueError, match="nesting is too deep"):
        Validator({"a": looped})


@pytest.mark.skipif(sys.version_info < (3, 13), reason="ReadOnly is 3.13")
def test_a_qualifier_past_the_wrappers_read_is_refused_by_depth() -> None:
    """A finite stack of wrappers past the bound is refused, never built.

    A dict literal refuses `Required` and `NotRequired` on a field, read through
    `ReadOnly`. Built past `MAX_QUALIFIER_DEPTH` (8) layers, the field dropped
    the qualifier under them: the key it calls optional was required, and one
    it calls required was optional. Past the bound it is refused as nesting too
    deep, as a wrapper naming itself is.
    """
    if sys.version_info >= (3, 13):

        def under_read_only(depth: int, qualified: object) -> object:
            for _ in range(depth):
                qualified = typing.ReadOnly[qualified]
            return qualified

        with pytest.raises(NotImplementedError, match="TypedDict qualifier"):
            Validator({"a": under_read_only(7, typing.NotRequired[int])})
        for depth in (8, 12):
            with pytest.raises(ValueError, match="nesting is too deep"):
                Validator({"a": under_read_only(depth, typing.NotRequired[int])})
            with pytest.raises(ValueError, match="nesting is too deep"):
                Validator({"a?": under_read_only(depth, typing.Required[int])})


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


# THEORY: open-and-close-read-the-region
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


# THEORY: open-and-close-read-the-region
def test_closing_an_opened_mapping_takes_out_only_the_region_opening_freed() -> None:
    """`close` after `open` refuses again what `open` freed, and nothing more.

    Opening folds the freed region into a clause admitting every value, so
    `{C: Any, str: int}` opens to a clause over `C` beside the complement of `C`
    and `str`, and closing takes the complement out and keeps `C`. Read as the
    whole clause, the round trip would keep the region, and admit a `None` key
    that `close` refuses, over a class and a kind alike.
    """

    class Key:
        pass

    for schema in (Validator({Key: Any, str: int}), Validator({int: Any, str: int})):
        round_trip = schema.open().close()
        assert round_trip == schema.close(), round_trip
        assert not round_trip.is_valid({None: 1})
        assert schema.open().is_valid({None: 1})

    # Where the opened key is every key, the claim is not in the term to keep, and
    # closing lands on a smaller term than `close` gives: never a larger one.
    for whole in (Validator(dict[int | str, Any]), Validator(dict[Key, Any])):
        assert repr(whole.open()) == "dict[anything, anything]"
        assert whole.open().close().relation_to(whole.close()) == "subset"
        assert not whole.open().close().is_valid({None: 1})
    literal_key = Validator(dict[Literal["a", "b"], Any])
    assert Validator(dict[Any, Any]).relation_to(literal_key.open()) == "subset"

    # A clause admitting every key beside another keeps the keys the other
    # claims: closing touches no claimed region.
    catch_all = Validator({anything: anything, str: int})
    assert catch_all.close().is_valid({"a": "not an int"})
    assert not catch_all.close().is_valid({1: 1})


def test_opening_a_record_that_claims_a_region_leaves_one_clause() -> None:
    """A `TypedDict` is a record with a clause, and opening it stays decidable.

    `TypedDict` builds named fields beside `str: anything` for the keys it does
    not name, so opening it frees the key-types that clause leaves over. Those
    two clauses carry one value between them and cover every key, which is one
    clause -- the catch-all an opened record has.

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


class _EqualToA:
    """A key of no string type that hashes and compares as `"a"`."""

    __slots__ = ()

    def __hash__(self) -> int:
        return hash("a")

    def __eq__(self, other: object) -> bool:
        return other == "a" or isinstance(other, _EqualToA)


class _ReachesA(str):
    """A `str` that spells `b` and hashes and compares as `"a"`."""

    __slots__ = ()

    def __hash__(self) -> int:
        return hash("a")

    def __eq__(self, other: object) -> bool:
        return other == "a"


def test_every_reading_of_a_record_finds_the_key_the_dict_finds() -> None:
    """A key reaching a declared name is that field, however the record reads.

    A dict finds `d["a"]` under any key whose hash and `==` reach `"a"`: an
    object of another type, or a `str` whose own text is `b`. A record is read
    by asking for its declared keys where its clauses allow, and by scanning
    its entries where a clause reads each key with its value, and the two
    resolve a key alike, so every record below admits the value and refuses it
    for the same reason. A reading that took such a key for an undeclared one
    would refuse `{"a": int, str: int}`, and the explaining walk, which asks
    the dict, would find nothing to report and name a mutation of a dict
    nothing moved.
    """

    class Typed(TypedDict):
        a: int

    records = [
        Validator({"a": int}),
        Validator({"a": int, str: int}),
        Validator(Typed),
        Validator({"a": int}).open(),
    ]
    for key in (_EqualToA(), _ReachesA("b")):
        for record in records:
            assert record.is_valid({key: 1}), (key, record)
            record.validate({key: 1})
            with pytest.raises(ValidationError) as info:
                record.validate({key: "x"})
            reported = [(item["code"], item["path"]) for item in info.value.errors]
            assert reported == [("int_type", ("a",))], (key, record)
        # Beside a key of its own the record is read through its clause, which
        # the closed one does not have.
        beside = {key: 1, "b": 2}
        assert [record.is_valid(beside) for record in records] == [
            False,
            True,
            True,
            True,
        ]
    assert Validator({"a": int}).is_subtype_of({"a": int, str: int})


class _Folded(str):
    """A `str` that hashes and compares by its folded case, as a header name does."""

    __slots__ = ()

    def __hash__(self) -> int:
        return hash(self.lower())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, str) and self.lower() == other.lower()


def test_two_keys_reaching_one_name_pay_for_one_field() -> None:
    """`b` is required, and no key of the dict is `b`.

    `_PlainName("a") == _Folded("A")` is `str`'s answer and `False`, so a dict holds
    both, and each reaches `"a"`. A count of the keys reaching a required field
    would take the second for a field no key holds, and a set proved empty would
    admit the value.
    """
    value = {_PlainName("a"): 1, _Folded("A"): 2}
    assert len(value) == 2
    assert "b" not in value
    for record in (
        {"a": int, "b": int},
        {"a": int, "b": int, str: int},
        {"a": int, "b": str, bytes: int},
    ):
        assert not Validator(record).is_valid(value), record
    never = Validator({"a": int, "b": nothing, bytes: int})
    assert never.is_empty()
    assert not never.is_valid(value)


def test_a_field_is_the_entry_the_lookup_finds_and_not_the_first_read() -> None:
    """A slot a deleted key left puts a later key first along the probes."""

    class Collides(str):
        __slots__ = ()

        def __hash__(self) -> int:
            return hash("a")

        def __eq__(self, other: object) -> bool:
            return self is other

    value: dict[str, object] = {Collides("x"): 0, _PlainName("a"): "s"}
    del value[next(iter(value))]
    value[_Folded("A")] = 1
    if not isinstance(value["a"], int):
        pytest.skip("this interpreter's lookup answers with the entry read first")
    if sys.implementation.name == "pypy":
        pytest.skip(
            "`PyDict_Next` reads each value through a lookup: `s` is never read"
        )
    assert list(value.values()) == ["s", 1]
    assert Validator({"a": int, str: str}).is_valid(value)
    assert Validator({"a": int, str: object}).is_valid(value)
    assert not Validator({"a": str, str: int}).is_valid(value)


@pytest.mark.xfail(
    sys.implementation.name == "pypy",
    strict=True,
    reason="PyPy's `PyDict_Next` answers each value with a lookup of its key",
)
# TRUST: A dict's entries are read from its storage.
def test_a_dicts_entries_are_read_from_its_storage() -> None:
    """The known-unsound case on PyPy: two keys reaching one name read one value."""

    class Collides(str):
        __slots__ = ()

        def __hash__(self) -> int:
            return hash("a")

        def __eq__(self, other: object) -> bool:
            return self is other

    value: dict[str, object] = {Collides("x"): 0, _PlainName("a"): "s"}
    del value[next(iter(value))]
    value[_Folded("A")] = 1
    if not isinstance(value["a"], int):
        pytest.skip("this interpreter's lookup answers with the entry read first")
    assert list(value.values()) == ["s", 1]
    assert not Validator({str: int}).is_valid(value)
    assert not Validator({"a": int, str: int}).is_valid(value)
    assert Validator({str: object}).is_valid(value)


def _dicts_over_odd_keys() -> Iterator[dict[Any, object]]:
    """Every dict of up to three keys reaching `"a"` and `"b"` every way a key can."""
    keys = [
        "a",
        "b",
        _PlainName("a"),
        _Folded("A"),
        _Folded("B"),
        _NeverEqual("a"),
        _OtherHash("a"),
        _EqualToA(),
        7,
    ]
    for size in range(4):
        for chosen in itertools.combinations(keys, size):
            for held in itertools.product([1, "s"], repeat=size):
                yield dict(zip(chosen, held, strict=True))


def _read_by_names(
    fields: dict[str, bool],
    clause: tuple[type, type] | None,
    value: dict[Any, object],
) -> bool:
    """Each name as the dict answers it, an `int`; every other entry by the clause."""
    entries = list(value.items())
    for name, required in fields.items():
        if name not in value:
            if required:
                return False
            continue
        found = value[name]
        if not isinstance(found, int):
            return False
        del entries[
            next(
                at
                for at, (key, held) in enumerate(entries)
                if held is found and hash(key) == hash(name) and key == name
            )
        ]
    if clause is None:
        return not entries
    return all(
        isinstance(k, clause[0]) and isinstance(v, clause[1]) for k, v in entries
    )


def test_every_reading_reads_a_name_as_the_dict_answers_it() -> None:
    """A field is what the dict answers for its name; any other entry is undeclared.

    The three readings of a record part on nothing here: a record asked for its
    keys, one scanned, and the explaining walk, which names a real key and never
    a mutation of a value nothing moved.
    """
    readings = (
        {"a": True},
        {"a": False},
        {"a": True, "b": True},
        {"a": True, "b": False},
    )
    clauses = (None, (str, object), (str, int), (str, str), (bytes, int), (int, object))
    for fields, clause in itertools.product(readings, clauses):
        spelled: dict[object, object] = {
            name if required else f"{name}?": int for name, required in fields.items()
        }
        if clause is not None:
            spelled[clause[0]] = clause[1]
        record = Validator(spelled)
        for value in _dicts_over_odd_keys():
            expected = _read_by_names(fields, clause, value)
            assert record.is_valid(value) == expected, (spelled, value)
            if expected:
                record.validate(value)
                continue
            with pytest.raises(ValidationError) as info:
                record.validate(value)
            assert info.value.code != "mutated_during_validation", (spelled, value)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "the walk resolves a record's keys as the dict does, and the relations "
        "read a field's key as a `str`, so a record decided below a mapping "
        "keyed by `str` admits a key of another type equal to a name"
    ),
)
# TRUST: A key equal to a declared name is a string.
def test_a_record_below_a_mapping_keyed_by_str_admits_no_key_of_another_type() -> None:
    """The known-unsound case of resolving a key the way the dict does.

    `_EqualToA()` is no string, and `d["a"]` finds it, so a record reads it as
    field `a`; a mapping keyed by `str` refuses it for its type. The relation
    between the two is decided over keys that are strings.
    """
    record, mapping = Validator({"a": int}), Validator(dict[str, int])
    value = {_EqualToA(): 1}

    assert record.is_valid(value), "the walk finds the field where the dict does"
    assert not mapping.is_valid(value), "and the mapping's key is a `str`"
    assert record.relation_to(mapping) != "subset"
