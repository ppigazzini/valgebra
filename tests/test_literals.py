import builtins
import enum
from fractions import Fraction
from typing import Literal

import pytest

from valgebra import (
    ValidationError,
    Validator,
    complement,
    intersection,
    nothing,
    union,
)


def test_literal_accepts_the_exact_value() -> None:
    assert Validator(5).is_valid(5)
    assert Validator("red").is_valid("red")
    assert Validator(b"x").is_valid(b"x")


def test_literal_rejects_a_different_value() -> None:
    assert not Validator(5).is_valid(6)
    assert not Validator("red").is_valid("green")


# TRUST: A literal is its constant's class under `==`.
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


class _Near:
    """Equal to anything within one: reflexive, symmetric, not transitive."""

    def __init__(self, n: int) -> None:
        self.n = n

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Near) and abs(self.n - other.n) <= 1

    def __hash__(self) -> int:
        return 0

    def __repr__(self) -> str:
        return f"_Near({self.n})"


# THEORY: a-constant-is-the-values-equal-to-it
def test_a_proof_between_constants_needs_a_transitive_equality() -> None:
    """A class whose `__eq__` is its own proves nothing by its constant.

    `_Near(1) == _Near(2)`, but `_Near(0)` equals the first and not the second,
    a value against a proof read from `==`; the class's equality is read by no
    atom, so the constant speaks for its literal nowhere it is not alone.
    """
    first, second, witness = Validator(_Near(1)), Validator(_Near(2)), _Near(0)

    assert first.is_valid(witness), "the witness equals the first constant"
    assert not second.is_valid(witness), "and not the second"
    assert first.relation_to(second) != "subset"


class _Member(enum.Enum):
    A = 1


# THEORY: a-constant-is-the-values-equal-to-it
def test_a_finite_set_of_constants_misses_a_value_of_every_boundless_kind() -> None:
    """A literal holds values of its constant's exact type alone.

    So a fresh `object()` is in no literal of a set, which its complement
    holds, and of the pairwise unequal values of a kind with infinitely many,
    a set of constants misses one: whatever the constants are, an enumeration
    member, an object, a `Fraction` or an `int` among them, which the set
    representation cannot hold.
    """
    witnesses = {int: 2, float: 0.5, str: "b", bytes: b"b"}
    for table in (
        Literal[_Member.A],
        Literal[_Member.A, 1],
        Validator(object()),
        Validator(Fraction(1, 3)),
    ):
        assert complement(table).relation_to(nothing) == "not_subset", table
        assert Validator(complement(table)).is_valid(object())
        for kind, witness in witnesses.items():
            assert Validator(kind).is_valid(witness)
            assert not Validator(table).is_valid(witness)
            assert Validator(kind).relation_to(table) == "not_subset", (kind, table)


# THEORY: a-constant-is-the-values-equal-to-it
def test_none_is_below_a_schema_where_its_value_is() -> None:
    """`None`'s kind is its one value, so walking it decides the kind.

    A table holding an enumeration member is no set the set representation
    holds, and `None` is the value standing against it.
    """
    member = Literal[_Member.A]
    assert not Validator(member).is_valid(None)
    assert Validator(None).relation_to(member) == "not_subset"
    assert intersection(None, complement(member)).relation_to(nothing) == "not_subset"
    assert Validator(None).relation_to(union(member, None)) == "subset"


_SENTINEL = getattr(builtins, "sentinel", None)


# THEORY: a-constant-is-the-values-equal-to-it
@pytest.mark.skipif(_SENTINEL is None, reason="a builtin from 3.15 (PEP 661)")
def test_a_sentinel_compares_by_identity() -> None:
    """Two sentinels are two values, and one is a value of its literal.

    `sentinel.__eq__` is the type's own, and answers `True` for the sentinel
    itself alone, so two sentinels of one name are unequal as two `object()`s
    are.
    """
    assert _SENTINEL is not None
    first, second = _SENTINEL("Missing"), _SENTINEL("Missing")
    assert first != second
    assert Validator(first).is_valid(first)
    assert not Validator(first).is_valid(second)
    both = intersection(Validator(first), Validator(second))
    assert both.relation_to(nothing) == "subset"
    assert Validator(first).relation_to(nothing) == "not_subset"
    assert Validator(first).relation_to(Validator(second)) == "not_subset"


class _FakesStaticFlags(type):
    """Answers `__flags__` without `Py_TPFLAGS_HEAPTYPE`, as a static type's."""

    def __getattribute__(cls, name: str) -> object:
        value = super().__getattribute__(name)
        return value & ~(1 << 9) if name == "__flags__" else value


def test_a_class_named_sentinel_is_not_the_builtin() -> None:
    """A class is the interpreter's `sentinel` by the flags Python set.

    One named `builtins.sentinel`, whose metaclass answers `__flags__` as a
    static type's and whose `==` holds every value, is a class statement's: two
    of its instances are equal, so the meet of their literals holds both.
    """
    namespace = {
        "__module__": "builtins",
        "__eq__": lambda _self, _other: True,
        "__hash__": object.__hash__,
    }
    posing = _FakesStaticFlags("sentinel", (), namespace)
    first, second = posing(), posing()
    both = intersection(Validator(first), Validator(second))
    assert both.is_valid(first)
    assert both.relation_to(nothing) != "subset"
