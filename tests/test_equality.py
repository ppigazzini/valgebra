"""Two spellings of one schema are one schema.

`==` is the question "are these the same schema", and a term that keeps its
spelling answers "were these built the same way" instead: a record's fields in
the order they were written in, a map's clauses in theirs, and a literal or a
refinement marker in the *pool slot* its constant landed in, which is the order
the constants were first seen. Read that way, `{"a": int, "b": int}` and `{"b":
int, "a": int}` -- one set, by every other measure the library offers -- compare
unequal, hash apart, and stay two members of a union that folds repeats.

Two mechanisms answer the question asked, and this holds both. Construction puts
a record's fields, a map's clauses and a refinement's constraints in a canonical
order, so the term itself stops carrying the spelling. And `==` reads *through*
a pool slot to the constant it names, matching a union's members and a
refinement's constraints as the sets they are -- because a slot is construction
order and cannot be sorted away. Composing two validators renumbers one side's
slots, and every set it renumbers is put back in the order construction gives
it, so a union of two spellings compares the terms it would have built from one.

What this is not is a decision. `int | ~int` and `anything` denote one set and
are not equal here; `is_equivalent` is the question for that, and the pair below
that asserts it stays separate is the one that says so.
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING, Annotated, Literal

import annotated_types as at
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

from valgebra import (
    Regex,
    Validator,
    anything,
    complement,
    instance_of,
    intersection,
    nothing,
    recursive,
    union,
)


class _Left:
    """One plain class, for the rows that name two classes in two orders."""


class _Right:
    """The other plain class."""


@dataclasses.dataclass
class _Point:
    """A class that declares a field, which a meet prints by the class's name."""

    x: int


def _is_even(value: int) -> bool:
    """Give one callable, so a predicate row compares a schema with itself."""
    return value % 2 == 0


def _is_odd(value: int) -> bool:
    """Give a second callable, for the row that says two predicates are two."""
    return value % 2 == 1


# Pairs that are one schema written two ways. Each row is a set, and the two
# spellings differ only in an order that is not part of it.
SAME = [
    ("record fields", {"a": int, "b": str}, {"b": str, "a": int}),
    ("optional fields", {"a?": int, "b": str}, {"b": str, "a?": int}),
    ("nested records", {"x": {"a": int, "b": int}}, {"x": {"b": int, "a": int}}),
    ("map clauses", {str: int, int: str}, {int: str, str: int}),
    ("record and clause", {"a": int, str: int}, {str: int, "a": int}),
    ("literal members", Literal[1, 2], Literal[2, 1]),
    ("literal members, strings", Literal["a", "b", "c"], Literal["c", "b", "a"]),
    ("union members", int | str, str | int),
    (
        "refinement markers",
        Annotated[int, at.Ge(0), at.Le(9)],
        Annotated[int, at.Le(9), at.Ge(0)],
    ),
    (
        "three markers",
        Annotated[str, at.MinLen(1), at.MaxLen(9), Regex("a+")],
        Annotated[str, Regex("a+"), at.MaxLen(9), at.MinLen(1)],
    ),
    (
        "a duplicated marker",
        Annotated[int, at.Ge(0), at.Ge(0)],
        Annotated[int, at.Ge(0)],
    ),
    # The strict bounds and a predicate each read their operand through the
    # pool, as the loose bounds do, so two spellings naming one constant -- or
    # one callable -- are one schema and hash alike.
    ("a strict lower bound", Annotated[int, at.Gt(0)], Annotated[int, at.Gt(0)]),
    ("a strict upper bound", Annotated[int, at.Lt(9)], Annotated[int, at.Lt(9)]),
    ("a predicate", Annotated[int, _is_even], Annotated[int, _is_even]),
    (
        "records inside a union",
        union({"a": int, "b": int}, str),
        union(str, {"b": int, "a": int}),
    ),
    (
        "records inside a list",
        [{"a": int, "b": int}, ...],
        [{"b": int, "a": int}, ...],
    ),
    # Each spelling numbers its constants in the order it reads them, so the
    # pairs below put one constant in two pool slots.
    (
        "bounds of one kind",
        Annotated[int, at.Ge(3), at.Ge(5)],
        Annotated[int, at.Ge(5), at.Ge(3)],
    ),
    (
        "literal keys",
        {Literal["x"]: int, Literal["y"]: str},
        {Literal["y"]: str, Literal["x"]: int},
    ),
    ("classes in a union", _Left | _Right, _Right | _Left),
    (
        "classes in a meet",
        intersection(instance_of(_Left), instance_of(_Right)),
        intersection(instance_of(_Right), instance_of(_Left)),
    ),
    (
        "complemented literals in a meet",
        intersection(int, complement(Literal[1]), complement(Literal[2])),
        intersection(int, complement(Literal[2]), complement(Literal[1])),
    ),
    (
        "complemented literals beside a declared class",
        intersection(_Point, complement(Literal[1]), complement(Literal[2])),
        intersection(_Point, complement(Literal[2]), complement(Literal[1])),
    ),
]

# Pairs that are *not* one schema, so the reading above cannot be doing it by
# saying yes to everything. Each differs in something a set depends on.
DIFFERENT = [
    ("different constants", Literal[1], Literal[2]),
    ("a different member", Literal[1, 2], Literal[1, 3]),
    ("a literal is typed", Literal[1], Literal[True]),
    # A float is not a `Literal` argument the typing spec allows, and is a
    # constant this library pools like any other: the pair is the point.
    ("a literal is typed, floats", Literal[1], Literal[1.0]),  # ty: ignore[invalid-type-form]
    ("a field's type", {"a": int}, {"a": str}),
    ("a missing field", {"a": int, "b": int}, {"a": int}),
    ("required against optional", {"a": int}, {"a?": int}),
    ("open against closed", {"a": int}, {"a": int, anything: anything}),
    ("the bound's direction", Annotated[int, at.Ge(0)], Annotated[int, at.Le(0)]),
    ("the bound's value", Annotated[int, at.Ge(0)], Annotated[int, at.Ge(1)]),
    ("a strict bound's value", Annotated[int, at.Gt(0)], Annotated[int, at.Gt(1)]),
    ("strict against loose", Annotated[int, at.Lt(9)], Annotated[int, at.Le(9)]),
    ("two predicates", Annotated[int, _is_even], Annotated[int, _is_odd]),
    ("a sequence is ordered", tuple[int, str], tuple[str, int]),
    ("the container", list[int], set[int]),
    ("a clause's value", {str: int}, {str: str}),
]


@pytest.mark.parametrize(("name", "left", "right"), SAME, ids=[row[0] for row in SAME])
def test_one_set_written_two_ways_is_one_schema(
    name: str, left: object, right: object
) -> None:
    first, second = Validator(left), Validator(right)
    assert first == second, name
    assert hash(first) == hash(second), f"{name}: equal validators must hash alike"
    # And the union of the pair is one member, because the fold that drops a
    # repeat compares terms: this is the consequence that made the equality
    # worth fixing rather than the equality itself.
    assert union(first, second) == first, name


@pytest.mark.parametrize(
    ("name", "left", "right"), DIFFERENT, ids=[row[0] for row in DIFFERENT]
)
def test_two_sets_stay_two_schemas(name: str, left: object, right: object) -> None:
    assert Validator(left) != Validator(right), name


# Pairs whose two spellings give one constant two pool slots. Composing them
# interns the second pool into the first, which sends the slots past each other.
SWAPPED_SLOTS = [
    (
        "bounds of one kind",
        Annotated[int, at.Ge(3), at.Ge(5)],
        Annotated[int, at.Ge(5), at.Ge(3)],
    ),
    (
        "steps",
        Annotated[int, at.MultipleOf(2), at.MultipleOf(3)],
        Annotated[int, at.MultipleOf(3), at.MultipleOf(2)],
    ),
    (
        "literal keys",
        {Literal["x"]: int, Literal["y"]: str},
        {Literal["y"]: str, Literal["x"]: int},
    ),
    (
        "bounds inside a list",
        list[Annotated[int, at.Le(9), at.Le(7)]],
        list[Annotated[int, at.Le(7), at.Le(9)]],
    ),
]


@pytest.mark.parametrize(
    ("name", "left", "right"), SWAPPED_SLOTS, ids=[row[0] for row in SWAPPED_SLOTS]
)
def test_a_composition_folds_what_construction_folds(
    name: str, left: object, right: object
) -> None:
    """A union of two spellings is one member, and the complement laws fold it.

    `==` reads through the slots and calls each pair one schema; the fold that
    drops a repeat and the one that cancels a schema against its complement
    compare terms. With the renumbered side left in the order it held, the two
    readings would disagree: the union would keep both members, and `a | ~b`
    stand unfolded beside an `a` equal to `b`.
    """
    first, second = Validator(left), Validator(right)
    assert first == second, name
    assert union(first, second) == first, name
    assert union(first, complement(second)) == Validator(anything), name
    assert intersection(first, complement(second)) == Validator(nothing), name


def test_a_validator_equals_itself_over_a_value_that_equals_nothing() -> None:
    """Identity before value, or a `nan` literal would not equal itself.

    The literals are built through a variable rather than written out: a float
    is not an argument the typing spec allows `Literal` to take, and a checker
    reading the annotation says so. This library pools the constant either way,
    which is what the case is about.
    """
    quiet = math.nan
    nan = Validator(Literal[quiet])  # ty: ignore[invalid-type-form]
    assert nan == nan  # noqa: PLR0124 - the identity is the subject
    # `math.nan` is one object, so a second validator over it pools the same
    # constant and the two are one schema.
    assert Validator(Literal[quiet]) == nan  # ty: ignore[invalid-type-form]
    # Two *different* nan objects are two constants: a nan is equal to no value,
    # itself included, so nothing but identity can join them.
    #
    # Whether the interpreter hands out two objects is the interpreter's
    # business: CPython boxes each `float("nan")` separately, PyPy answers both
    # with one box, and the second has no two constants to keep apart. The
    # premise is asserted rather than assumed, so the case reports "this
    # interpreter has one nan" instead of failing over a pooling rule that holds
    # in both.
    first, second = float("nan"), float("nan")
    if first is second:
        pytest.skip("this interpreter hands out one nan object, not two")
    assert Validator(Literal[first]) != Validator(  # ty: ignore[invalid-type-form]
        Literal[second]  # ty: ignore[invalid-type-form]
    )


def test_equality_is_not_a_decision() -> None:
    """Sets a theorem identifies are not equal terms, and must not become so.

    `==` is what the constructors settle; `is_equivalent` is what the decision
    procedures prove. Blurring them would make a cheap syntactic check look like
    an expensive semantic one, and the pairs below are the ones a reader most
    expects to be blurred.
    """
    pairs = [
        # Not `int | ~int`, which the constructor folds to the top: the
        # complement laws are settled where the schema is built, and these are
        # the ones a *theorem* identifies rather than a fold.
        (intersection(int, str), Validator(nothing)),
        (Validator(bool), Validator(Literal[True, False])),
        (union(int, Literal[1]), Validator(int)),
    ]
    for left, right in pairs:
        assert left.is_equivalent(right), "the pair is one set"
        assert left != right, "and two terms"


def test_a_recursive_schema_equals_the_same_one_built_again() -> None:
    # The fixpoint is a definition and a reference, and both sides build their
    # own; equality compares the definitions pairwise rather than by identity.
    json = recursive(lambda j: union(None, bool, int, float, str, [j], {str: j}))
    again = recursive(lambda j: union(None, bool, int, float, str, [j], {str: j}))
    assert json == again
    assert hash(json) == hash(again)
    # A different body is a different schema, however alike the reference looks.
    assert json != recursive(lambda j: union(None, bool, int, str, [j], {str: j}))


def test_a_validator_is_usable_as_a_dictionary_key() -> None:
    # The point of the hash: a registry of contracts holds one entry per schema,
    # and the spelling a caller happened to use does not make a second.
    registry = {Validator({"a": int, "b": str}): "row"}
    registry[Validator({"b": str, "a": int})] = "row again"
    assert len(registry) == 1
    assert registry[Validator({"a": int, "b": str})] == "row again"


# Schemas that differ only in a constant. `==` tells these apart, and a
# `__hash__` that skipped every pool slot would put all of them in one bucket
# and make a registry keyed by validators a linear scan.
def _integer_literal(i: int) -> Validator:
    return Validator(Literal[i])  # ty: ignore[invalid-type-form]


def _string_literal(i: int) -> Validator:
    return Validator(Literal[f"s{i}"])  # ty: ignore[invalid-type-form]


def _literal_in_a_record(i: int) -> Validator:
    return Validator({"k": Literal[i]})  # ty: ignore[invalid-type-form]


def _literal_in_a_union(i: int) -> Validator:
    return Validator(Literal[i, "shared"])  # ty: ignore[invalid-type-form]


BY_CONSTANT: list[tuple[str, Callable[[int], Validator]]] = [
    ("integer literals", _integer_literal),
    ("string literals", _string_literal),
    ("a literal in a record", _literal_in_a_record),
    ("a literal in a union", _literal_in_a_union),
    ("lower bounds", lambda i: Validator(Annotated[int, at.Ge(i)])),
    ("upper bounds", lambda i: Validator(Annotated[int, at.Le(i)])),
    ("multiples", lambda i: Validator(Annotated[int, at.MultipleOf(i + 1)])),
    ("patterns", lambda i: Validator(Annotated[str, Regex(f"a{{{i + 1}}}")])),
]


@pytest.mark.parametrize(
    ("name", "build"), BY_CONSTANT, ids=[row[0] for row in BY_CONSTANT]
)
def test_schemas_that_differ_only_in_a_constant_hash_apart(
    name: str, build: Callable[[int], Validator]
) -> None:
    made = [build(i) for i in range(200)]
    assert len({hash(one) for one in made}) > 190, (
        f"{name}: {len({hash(one) for one in made})} distinct hashes over 200 "
        "schemas. A hash that ignores the constant makes a registry of them a list."
    )
    # And they are still 200 distinct schemas, which is what the hash must track.
    assert len({repr(one) for one in made}) == 200


def test_a_registry_of_ten_thousand_validators_is_a_lookup_not_a_scan() -> None:
    registry = {_integer_literal(i): i for i in range(10_000)}
    assert len({hash(key) for key in registry}) == 10_000
    assert registry[_integer_literal(5_000)] == 5_000


def test_an_unhashable_constant_leaves_the_schema_hashable() -> None:
    """A validator must be usable as a key whatever its schema names.

    A class defining `__eq__` without `__hash__` is unhashable, and a schema may
    name one. The constant then contributes nothing and the shape stands alone,
    which is a collision rather than a refusal.
    """

    class Unhashable:
        __hash__ = None  # type: ignore[assignment]

        def __eq__(self, other: object) -> bool:
            return isinstance(other, Unhashable)

    sentinel = Unhashable()
    validator = Validator(Literal[sentinel])  # ty: ignore[invalid-type-form]
    assert isinstance(hash(validator), int)
    assert {validator: 1}[validator] == 1


def test_equal_validators_still_hash_alike_when_the_constant_is_read() -> None:
    """The contract the fold must not break, over the pairs `==` calls equal."""
    same = [
        (Validator(Literal[1, 2]), Validator(Literal[2, 1])),
        (Validator({"a": Literal[1]}), Validator({"a": Literal[1]})),
        (
            Validator(Annotated[int, at.Ge(0), at.Le(9)]),
            Validator(Annotated[int, at.Le(9), at.Ge(0)]),
        ),
        (union(Validator(Literal[1]), Validator(Literal[2])), Validator(Literal[1, 2])),
        (_integer_literal(10**30), _integer_literal(10**30)),
    ]
    for left, right in same:
        assert left == right, repr(left)
        assert hash(left) == hash(right), (
            f"{left!r} and {right!r} are equal and must hash alike"
        )


@pytest.mark.parametrize(("name", "left", "right"), SAME, ids=[row[0] for row in SAME])
def test_one_schema_prints_one_way(name: str, left: object, right: object) -> None:
    """`docs/04` says `repr` shows the schema and `==` compares it.

    Construction sorts a set with the IR's own order, and a literal, a class or
    a bound's operand sorts there by its **pool slot** -- the order the
    constants were first seen. So `Literal[1, 2]` and `Literal[2, 1]`, or `A |
    B` and `B | A`, are one schema by `==` and by `hash`, and a render in slot
    order would show something that is not the schema. Every pair the suite
    calls equal must print alike.
    """
    assert repr(Validator(left)) == repr(Validator(right)), name


def test_a_union_of_literals_prints_in_one_order_however_it_is_built() -> None:
    # The two ways a union of literals arrives: written as one `Literal[...]`,
    # and combined from validators whose pools number the constants differently.
    assert repr(Validator(Literal[3, 1, 2])) == repr(Validator(Literal[1, 2, 3]))
    combined = union(_integer_literal(3), _integer_literal(1), _integer_literal(2))
    assert repr(combined) == repr(Validator(Literal[1, 2, 3]))
    # The runs of one kind keep the order the normal form gives them, so a
    # member of another kind stays where it was.
    assert repr(Validator(str | Literal[2, 1])) == "str | Literal[1] | Literal[2]"


def test_the_rendered_order_still_rebuilds_the_schema() -> None:
    """Reordering what is printed must not print something that is not it."""
    environment = {
        "Validator": Validator,
        "Literal": Literal,
        "Annotated": Annotated,
        "Regex": Regex,
        "Ge": at.Ge,
        "at": at,
        "anything": anything,
        "nothing": nothing,
        "union": union,
        "intersection": intersection,
        "complement": complement,
        "recursive": recursive,
    }
    for spelling in (
        Literal[3, 1, 2],
        str | Literal[2, 1],
        {"k": Literal["b", "a"]},
        Annotated[int, at.Ge(5), at.Ge(3)],
        {Literal["y"]: str, Literal["x"]: int},
        intersection(int, complement(Literal[2]), complement(Literal[1])),
    ):
        validator = Validator(spelling)
        assert Validator(eval(repr(validator), environment)) == validator  # noqa: S307


def test_a_validator_hashes_its_constants_once() -> None:
    """The digest is kept, as a `frozenset` keeps its hash.

    A registry keyed by validators hashes a key on every lookup, and the digest
    reads every pooled constant. Kept, the constants are asked once.
    """
    asked: list[int] = []

    class Counted:
        def __hash__(self) -> int:
            asked.append(1)
            return 7

    validator = Validator(Literal[Counted()])  # ty: ignore[invalid-type-form]
    asked.clear()  # `typing` hashes a `Literal`'s arguments to cache the form
    first = hash(validator)
    assert hash(validator) == first
    assert {validator: 1}[validator] == 1
    assert len(asked) == 1


def test_validators_built_alike_ask_each_constant_once() -> None:
    """Two tables built the same way match member for member, in order.

    Equality matches members as sets, and a search that skips the members
    already matched asks a constant's `__eq__` only of a candidate: built
    alike, each member meets its match first and is asked once.
    """
    asked: list[int] = []

    class Counted(int):
        def __eq__(self, other: object) -> bool:
            asked.append(1)
            return int.__eq__(self, other)

        __hash__ = int.__hash__

    # Built through `union` rather than one `Literal`, whose cache would hand
    # both sides the same constants and answer every pair by identity.
    left = union(*[Validator(Counted(i)) for i in range(200)])
    right = union(*[Validator(Counted(i)) for i in range(200)])
    asked.clear()
    assert left == right
    assert len(asked) == 200


def test_two_orders_of_a_table_are_matched_in_one_pass() -> None:
    """Two orders of one table match by digest, member for member.

    A member is looked for only among the other side's members that hash
    alike, so a table written backwards asks each constant's `__eq__` once,
    beside the first pair, which the match in place asks before it finds the
    orders differ. A search from the front asks it of every member left:
    twenty thousand questions for two hundred members, quadratic in the width.
    """
    asked: list[int] = []

    class Counted(int):
        def __eq__(self, other: object) -> bool:
            asked.append(1)
            return int.__eq__(self, other)

        __hash__ = int.__hash__

    left = union(*[Validator(Counted(i)) for i in range(200)])
    right = union(*[Validator(Counted(i)) for i in reversed(range(200))])
    asked.clear()
    assert left == right
    assert len(asked) == 200 + 1
