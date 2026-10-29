import copy
from typing import Any, Literal, Optional, Union

import pytest
from hypothesis import given
from hypothesis import strategies as st

from valgebra import ValidationError, Validator, recursive, union


def test_pep604_union_accepts_either_branch() -> None:
    schema = Validator(int | str)
    assert schema.is_valid(1)
    assert schema.is_valid("x")
    assert not schema.is_valid(1.0)


def test_optional_admits_none() -> None:
    schema = Validator(int | None)
    assert schema.is_valid(3)
    assert schema.is_valid(None)
    assert not schema.is_valid("x")


def test_typing_optional_and_union_aliases() -> None:
    # The legacy aliases are exercised on purpose; the modern X | Y form is
    # covered above.
    assert Validator(Optional[int]).is_valid(None)  # noqa: UP045
    assert Validator(Union[int, str]).is_valid("x")  # noqa: UP007
    assert not Validator(Union[int, str]).is_valid(1.0)  # noqa: UP007


def test_literal_with_several_values_is_a_union() -> None:
    schema = Validator(Literal["red", "green"])
    assert schema.is_valid("red")
    assert schema.is_valid("green")
    assert not schema.is_valid("blue")


def test_literal_with_one_value_is_a_single_literal() -> None:
    schema = Validator(Literal[5])
    assert schema.is_valid(5)
    assert not schema.is_valid(True)


def test_any_admits_every_value() -> None:
    schema = Validator(Any)
    assert schema.is_valid(object())
    assert schema.is_valid(None)
    assert schema.is_valid([1, "x", {}])


def test_union_failure_reports_a_union_error() -> None:
    with pytest.raises(ValidationError) as info:
        Validator(int | str).validate(1.0)
    assert info.value.code == "union_error"


def test_int_literal_union_keeps_same_type_semantics() -> None:
    # A multi-value literal compiles to a union of literals, which the fast path
    # decides by value; it must still honor the same-type rule, so a bool or
    # float never matches an int literal.
    schema = Validator(Literal[1, 2, 3])
    assert schema.is_valid(1)
    assert schema.is_valid(3)
    assert not schema.is_valid(4)
    assert not schema.is_valid(True)  # bool, not int
    assert not schema.is_valid(1.0)  # float, not int


def test_str_literal_union_decides_by_value() -> None:
    schema = Validator(Literal["red", "green", "blue"])
    assert schema.is_valid("green")
    assert not schema.is_valid("teal")
    assert not schema.is_valid(0)


def test_mixed_type_literal_union_matches_each_type() -> None:
    # int, str, and bool members together: the fast path decides int and str by
    # value while bool falls back to the linear scan, and all must agree.
    schema = Validator(Literal[1, "a", True])
    for member in (1, "a", True):
        assert schema.is_valid(member)
    assert not schema.is_valid(2)
    assert not schema.is_valid("b")
    assert not schema.is_valid(1.0)  # a float matches no member


def test_big_integer_literal_union_matches_outside_machine_range() -> None:
    # union(...) builds a union of literals from runtime values, so a literal
    # larger than a machine integer can be exercised; it is decided by the linear
    # fallback while the small member uses the fast path.
    big = 10**30
    schema = union(big, 5)
    assert schema.is_valid(big)
    assert schema.is_valid(5)
    assert not schema.is_valid(big + 1)


def test_literal_union_decision_survives_copy() -> None:
    # A copied validator rebuilds its own precompute, so the value-keyed fast
    # path stays correct rather than reusing the original's node addresses.
    schema = copy.deepcopy(Validator(Literal[1, 2, 3]))
    assert schema.is_valid(2)
    assert not schema.is_valid(9)
    assert not schema.is_valid(True)


def test_literal_union_validate_agrees_with_is_valid() -> None:
    # The explain path bypasses the fast set lookup; it must reach the same
    # verdict the membership fast path does.
    schema = Validator(Literal[1, 2, 3])
    schema.validate(2)
    with pytest.raises(ValidationError):
        schema.validate(4)
    with pytest.raises(ValidationError):
        schema.validate(True)


class _Tag(str):
    """A `str` a `Literal` refuses and a dict lookup finds as its text."""

    __slots__ = ()


class _Mapping(dict):
    """A dict subclass, which the deciding walk reads by the scan."""


#: Tag constants, each owned by at most one record of a drawn union.
_TAGS: list[object] = ["a", "b", "c", 0, 1, 2]


@st.composite
def _tagged_union(draw: st.DrawFn) -> list[dict[str, object]]:
    """Draw records each requiring `kind`, pinned to tags no two share."""
    owners = draw(st.lists(st.integers(0, 3), min_size=len(_TAGS), max_size=len(_TAGS)))
    records: list[dict[str, object]] = []
    for record in range(max(owners) + 1):
        tags = [
            tag for tag, owner in zip(_TAGS, owners, strict=True) if owner == record
        ]
        if not tags:
            continue
        value = draw(st.sampled_from([int, str, int | None]))
        key = draw(st.sampled_from(["v", "v?"]))
        records.append({"kind": Literal[tuple(tags)], key: value})  # ty: ignore[invalid-type-form]
    return records


_VALUES = st.fixed_dictionaries(
    {},
    optional={
        "kind": st.sampled_from([*_TAGS, "zz", 9, True, 1.0, _Tag("a"), None]),
        "v": st.sampled_from([1, "x", None]),
        "w": st.just(0),
    },
)


@given(_tagged_union(), _VALUES, st.sampled_from([dict, _Mapping]))
def test_a_tagged_union_admits_what_some_record_does(
    records: list[dict[str, object]],
    value: dict[str, object],
    mapping: type[dict[str, object]],
) -> None:
    # The deciding walk reads the tag and walks the one record it names; the
    # union's meaning is still that some record admits the value.
    value = mapping(value)
    schema = Validator(union(*records))
    expected = any(Validator(record).is_valid(value) for record in records)
    assert schema.is_valid(value) == expected
    if expected:
        schema.validate(value)
    else:
        with pytest.raises(ValidationError):
            schema.validate(value)


def test_a_recursive_tagged_union_is_read_at_its_tag() -> None:
    tree = Validator(
        recursive(
            lambda t: union(
                {"kind": Literal["leaf"], "v": int},
                {"kind": Literal["node"], "kids": [t]},
            )
        )
    )
    assert tree.is_valid({"kind": "node", "kids": [{"kind": "leaf", "v": 1}]})
    assert not tree.is_valid({"kind": "node", "kids": [{"kind": "leaf", "v": "x"}]})
    assert not tree.is_valid({"kind": "twig", "v": 1})
    assert not tree.is_valid({"v": 1})


def test_records_sharing_a_tag_are_scanned() -> None:
    # Two records pinning one constant are no tagged union: the value may be
    # either's, and both are asked.
    schema = Validator(
        union({"kind": Literal["a"], "v": int}, {"kind": Literal["a"], "v": str})
    )
    assert schema.is_valid({"kind": "a", "v": 1})
    assert schema.is_valid({"kind": "a", "v": "x"})
    assert not schema.is_valid({"kind": "a", "v": None})


def test_a_thousand_records_are_read_at_their_tag() -> None:
    records = [{"kind": Literal[f"t{i}"], "v": int} for i in range(1000)]  # ty: ignore[invalid-type-form]
    schema = Validator(union(*records))
    assert schema.is_valid({"kind": "t999", "v": 1})
    assert not schema.is_valid({"kind": "t999", "v": "x"})
    assert not schema.is_valid({"kind": "t1000", "v": 1})
    with pytest.raises(ValidationError):
        schema.validate({"kind": "t0", "v": "x"})


class _CountedKey(str):
    """A key equal to its text that counts the lookups comparing against it."""

    __slots__ = ()
    asked = 0

    def __eq__(self, other: object) -> bool:
        type(self).asked += 1
        return str.__eq__(self, other)

    __hash__ = str.__hash__


def test_a_tagged_union_reads_its_tag_once() -> None:
    # Every lookup of `kind` in this dict compares against the counted key: the
    # deciding walk reads it once to choose the record and once in the record,
    # where reading each record in turn reads it once a record.
    records = [{"kind": Literal[f"t{i}"], "v": int} for i in range(64)]  # ty: ignore[invalid-type-form]
    schema = Validator(union(*records))
    for tag, admitted in (("t63", True), ("zz", False)):
        _CountedKey.asked = 0
        assert schema.is_valid({_CountedKey("kind"): tag, "v": 1}) == admitted
        assert _CountedKey.asked <= 2
