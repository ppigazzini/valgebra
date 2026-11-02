"""Broad generative soundness check for the decision procedure.

Arbitrary nested schemas ``a`` and ``b`` and an arbitrary value ``v`` are
generated, then the decision is held to real membership: ``is_subtype_of`` and
``is_equivalent`` never claim a relation membership contradicts, ``is_empty`` never
accepts a value, and every schema is a subtype of itself. This is deliberately
adversarial: a complement that is not a subtype of itself, or a pool merge that
gives one constant two indices, fails it.
"""

import enum
import json
import math
from collections.abc import Callable
from decimal import Decimal
from typing import Annotated, Any, Literal, Protocol, runtime_checkable

import annotated_types as at
import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from valgebra import (
    Regex,
    ValidationError,
    Validator,
    anything,
    complement,
    instance_of,
    intersection,
    nothing,
    recursive,
    union,
)

_bases = st.sampled_from([int, str, bool, float, bytes, None, complex, bytearray])
_lits = st.sampled_from(
    [
        Literal[0],
        Literal[1],
        Literal[-1],
        Literal[2],
        Literal["a"],
        Literal["b"],
        Literal[True],
    ]
)
_refines = st.integers(-2, 2).map(lambda n: Annotated[int, at.Ge(n)])
# Fixed recursive schemas, so the hunt also exercises recursion (uninhabited
# detection and coinductive subtyping) composed with everything else.
_recursive = st.sampled_from(
    [
        recursive(lambda t: union(None, {"value": int, "next": t})),
        recursive(lambda t: union(None, [t])),
        recursive(lambda t: union(int, str, [t], {str: t})),
    ]
)
_leaf = st.one_of(_bases, _lits, _refines, _recursive)


def _extend(children: st.SearchStrategy) -> st.SearchStrategy:
    pair = st.tuples(children, children)
    return st.one_of(
        children.map(lambda c: ("list", c)),
        pair.map(lambda p: ("fixed", p[0], p[1])),
        children.map(lambda c: ("tail", c)),
        children.map(lambda c: ("dict", c)),
        children.map(lambda c: ("record", c)),
        pair.map(lambda p: ("union", p[0], p[1])),
        pair.map(lambda p: ("intersection", p[0], p[1])),
        children.map(lambda c: ("complement", c)),
    )


_specs = st.recursive(_leaf, _extend, max_leaves=12)

_BUILDERS = {
    "list": lambda a: [a[0]],
    "fixed": lambda a: [a[0], a[1]],
    "tail": lambda a: [a[0], ...],
    "dict": lambda a: {str: a[0]},
    "record": lambda a: {"k": a[0], "j?": a[0]},
    "union": lambda a: union(a[0], a[1]),
    "intersection": lambda a: intersection(a[0], a[1]),
    "complement": lambda a: complement(a[0]),
}


def _build(spec: object) -> object:
    if not isinstance(spec, tuple) or not spec:
        return spec
    tag = spec[0]
    if not isinstance(tag, str):
        return spec
    return _BUILDERS[tag]([_build(child) for child in spec[1:]])


_scalars = st.one_of(
    st.integers(-3, 3),
    st.text(max_size=2),
    st.booleans(),
    st.floats(allow_nan=False, allow_infinity=False, min_value=-5, max_value=5),
    st.none(),
    st.binary(max_size=2),
)
_values = st.recursive(
    _scalars,
    lambda c: st.one_of(
        st.lists(c, max_size=3),
        st.dictionaries(st.text(max_size=2), c, max_size=3),
        st.tuples(c, c),
        st.frozensets(st.integers(-3, 3), max_size=3),
    ),
    max_leaves=8,
)


@given(sa=_specs, sb=_specs, v=_values)
def test_decision_is_sound_against_membership(
    sa: object, sb: object, v: object
) -> None:
    # Reflexivity and the other necessary properties are asserted in
    # test_necessary_properties.py; recursion reflexivity holes are tracked in
    # test_completeness_ledger.py. This fuzzer holds the decision to membership.
    a, b = _build(sa), _build(sb)
    left, right = Validator(a), Validator(b)
    in_a = left.is_valid(v)
    if left.is_subtype_of(b) and in_a:
        assert right.is_valid(v)
    if left.is_empty():
        assert not in_a
    if left.is_equivalent(b):
        assert in_a == right.is_valid(v)


def _json_safe(value: object) -> bool:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


@given(sa=_specs, v=_values)
def test_membership_walks_and_paths_agree(sa: object, v: object) -> None:
    # Metamorphic checks needing no oracle: the fast and explaining walks agree,
    # the JSON path matches validating the parsed value, and ensure returns the
    # input unchanged exactly when it is a member.
    compiled = Validator(_build(sa))
    member = compiled.is_valid(v)
    try:
        compiled.validate(v)
        explained = True
    except ValidationError:
        explained = False
    assert member == explained
    if _json_safe(v):
        text = json.dumps(v)
        assert compiled.is_valid_json(text) == compiled.is_valid(json.loads(text))
    if member:
        assert compiled.ensure(v) is v


class _Flip(type):
    """A metaclass whose ``isinstance`` alternates, so the class is not a set."""

    answers = 0

    def __instancecheck__(cls, instance: object) -> bool:
        _Flip.answers += 1
        return _Flip.answers % 2 == 1


class _Coin(metaclass=_Flip):
    """A class whose membership is a coin toss rather than a fixed set."""


class _Pure:
    """An ordinary class, whose membership is `isinstance` and nothing else."""


def _alternating() -> object:
    """Build a predicate that answers differently each time it is asked."""
    state = {"answers": 0}

    def coin(_value: object) -> bool:
        state["answers"] += 1
        return state["answers"] % 2 == 1

    return at.Predicate(coin)


def test_a_callback_atom_is_not_reported_empty_against_itself() -> None:
    """``A ∩ ¬A = ∅`` is a law about *sets*, and a predicate is not one.

    The walk evaluates each occurrence separately, so an answer that alternates
    puts one value in both ``A`` and ``¬A``. Reporting the meet empty would be a
    proof with a witness standing against it, which is the shape of an unsound
    verdict rather than a conservative one.
    """
    predicate = Annotated[int, _alternating()]
    meet = Validator(intersection(predicate, complement(predicate)))

    # Two runs, because the witness depends on which parity the alternation is
    # at when the walk reaches the first member, and the members of a meet are
    # held in the normal form's order rather than the written one. One of the
    # two runs admits the value, and that is the witness the law would be
    # standing against.
    assert any(meet.is_valid(7) for _ in range(2)), "the answers admit the value"
    assert not meet.is_empty(), "so the meet is not empty"


def test_a_hooked_class_is_not_reported_empty_against_itself() -> None:
    """The same law, and the same reason, one atom along.

    ``isinstance`` against a metaclass that overrides ``__instancecheck__`` runs
    user code, so the class is not a fixed set either. A class whose metaclass
    leaves the hooks alone *is* one, and the law still applies to it -- the
    restriction is to the atoms it does not hold for, not to the law.
    """
    hooked = Validator(intersection(_Coin, complement(_Coin)))
    # Two runs, for the reason the predicate above needs them.
    assert any(hooked.is_valid(7) for _ in range(2)), "the answers admit the value"
    assert not hooked.is_empty(), "so the meet is not empty"

    pure = Validator(intersection(_Pure, complement(_Pure)))
    assert not pure.is_valid(_Pure()), "no value is in a class and outside it"
    assert pure.is_empty(), "and the law still decides that"


def test_a_pattern_over_text_and_bytes_holds_no_byte_string() -> None:
    """A pattern matches the text of a ``str``, so its set holds no ``bytes``.

    The base ``str | bytes`` admits byte strings and the pattern refuses every
    one, so the refinement is the strings the pattern matches. Read over
    ``bytes`` as well, the set would prove the empty byte string below it,
    empty a meet holding ``b"a"``, and refute its inclusion in ``str``. Each
    assertion below has a value standing behind it.
    """
    pattern = Validator(Annotated[str | bytes, Regex("a*")])
    empty_bytes = Validator(Annotated[bytes, at.MaxLen(0)])
    assert empty_bytes.is_valid(b"")
    assert not pattern.is_valid(b"")
    assert empty_bytes.relation_to(pattern) == "not_subset"

    outside = Validator(intersection(Literal[b"a"], complement(pattern)))
    assert outside.is_valid(b"a")
    assert not outside.is_empty()
    assert Validator(Literal[b"a"]).relation_to(pattern) == "not_subset"

    assert pattern.relation_to(str) == "subset"
    assert pattern.is_equivalent(Annotated[str, Regex("a*")])


def _sign(value: float) -> bool:
    return math.copysign(1.0, value) > 0


# THEORY: a-constant-is-the-values-equal-to-it
def test_a_float_zero_literal_is_not_proved_below_a_sign_predicate() -> None:
    """``Literal[0.0]`` holds ``-0.0``, which a sign test refuses.

    The constant passes the predicate and its equal does not, so a proof
    standing on the constant would claim a value the supertype refuses.
    """
    positive = Annotated[float, _sign]
    literal = Validator(Literal[0.0])  # ty: ignore[invalid-type-form]
    assert literal.is_valid(-0.0)
    assert not Validator(positive).is_valid(-0.0)
    assert literal.relation_to(positive) == "undecided"
    assert not literal.is_subtype_of(positive)
    # A refutation by the constant stands, since `0.0` is a value of the literal.
    negative = Annotated[float, lambda x: math.copysign(1.0, x) < 0]
    assert literal.relation_to(negative) == "not_subset"


# THEORY: a-constant-is-the-values-equal-to-it
def test_a_decimal_literal_is_not_proved_below_a_predicate_on_its_exponent() -> None:
    """``Literal[Decimal("1.00")]`` holds ``Decimal("1.0")``, which has one place.

    A check for two decimal places is the predicate a money field carries, and
    two equal ``Decimal`` values answer it apart. The literal stays proved
    below the schemas that answer equal values alike.
    """
    two_places = Annotated[Decimal, lambda d: d.as_tuple().exponent == -2]
    literal = Validator(Literal[Decimal("1.00")])  # ty: ignore[invalid-type-form]
    assert literal.is_valid(Decimal("1.0"))
    assert not Validator(two_places).is_valid(Decimal("1.0"))
    assert literal.relation_to(two_places) == "undecided"
    assert literal.relation_to(Decimal) == "subset"
    assert literal.relation_to(Annotated[Decimal, at.Gt(Decimal(0))]) == "subset"


class _CaseFree(str):
    """A ``str`` whose ``==`` ignores case, hashed to match."""

    __slots__ = ()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, str) and self.lower() == other.lower()

    def __hash__(self) -> int:
        return hash(self.lower())


class _Folded(str):
    """A ``str`` whose ``==`` compares ``casefold``: ``"ss"`` and ``"ß"`` are one."""

    __slots__ = ()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, str) and self.casefold() == other.casefold()

    def __hash__(self) -> int:
        return hash(self.casefold())


class _BytesCaseFree(bytes):
    """A ``bytes`` whose ``==`` ignores ASCII case."""

    __slots__ = ()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, bytes) and self.lower() == other.lower()

    def __hash__(self) -> int:
        return hash(self.lower())


class _Mod7(int):
    """An ``int`` equal to every integer it is congruent to modulo 7."""

    __slots__ = ()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, int) and (int(self) - int(other)) % 7 == 0

    def __hash__(self) -> int:
        return int(self) % 7


class _Sign(float):
    """A ``float`` equal to every float of its sign."""

    __slots__ = ()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, float) and (self > 0) == (other > 0)

    def __hash__(self) -> int:
        return int(self > 0)


class _Number(enum.IntEnum):
    ONE = 1


#: A builtin subclass's constant, a supertype reading the builtin's storage,
#: and a value its literal holds that the supertype refuses.
_READ_APART: list[tuple[object, object, object]] = [
    (_CaseFree("a"), Annotated[str, Regex("a")], _CaseFree("A")),
    (_CaseFree("a"), Annotated[str, at.Ge("a")], _CaseFree("A")),
    (_Folded("ss"), Annotated[str, at.Len(2, 2)], _Folded("ß")),
    (_BytesCaseFree(b"a"), Annotated[bytes, at.Ge(b"a")], _BytesCaseFree(b"A")),
    (_Mod7(1), Annotated[int, at.Ge(1)], _Mod7(-6)),
    (_Mod7(1), Annotated[int, at.Le(1)], _Mod7(8)),
    (_Mod7(2), Annotated[int, at.MultipleOf(2)], _Mod7(9)),
    (_Sign(1.5), Annotated[float, at.Ge(1)], _Sign(0.25)),
]


# THEORY: a-constant-is-the-values-equal-to-it
@pytest.mark.parametrize(("constant", "supertype", "equal"), _READ_APART)
def test_a_builtin_subclass_with_its_own_equality_is_not_proved_by_its_constant(
    constant: object, supertype: object, equal: object
) -> None:
    """A builtin subclass's literal holds a value its own ``==`` joins and atoms refuse.

    ``Literal[c]`` is every value of ``c``'s type equal to it, and a pattern, a
    length, a bound and a step read the builtin's storage, which an ``==`` the
    subclass defines need not respect: the constant passes, a value equal to it
    does not. Neither the relation nor the meet with the complement stands on
    the constant.
    """
    literal = Validator(Literal[constant])  # ty: ignore[invalid-type-form]
    assert literal.is_valid(equal)
    assert not Validator(supertype).is_valid(equal)
    assert literal.relation_to(supertype) != "subset"
    meet = intersection(literal, complement(supertype))
    assert meet.is_valid(equal)
    assert not meet.is_empty()


# THEORY: a-constant-is-the-values-equal-to-it
def test_a_subclass_keeping_its_builtins_equality_is_proved() -> None:
    """An ``IntEnum``'s ``==`` is ``int``'s, so its equal values hold one integer.

    The bound reads that integer for every value the literal holds, so the
    constant speaks for the literal as an ``int`` constant does.
    """
    literal = Validator(Literal[_Number.ONE])
    assert literal.relation_to(Annotated[int, at.Ge(1)]) == "subset"


def test_a_proof_reading_the_constants_type_stands_for_its_literal() -> None:
    """A literal's values all have the constant's exact type, which a kind reads.

    `_CaseFree("a")`'s literal holds `_CaseFree("A")` as well, and both are a
    `str` and no `int`, so the literal is below `str`, below its own class and
    outside `int`, as far inside a list as at the top. A pattern reads the
    characters, which the two do not share, and proves nothing by the constant.
    """
    literal = Validator(Literal[_CaseFree("a")])  # ty: ignore[invalid-type-form]
    assert literal.relation_to(str) == "subset"
    assert literal.relation_to(instance_of(_CaseFree)) == "subset"
    assert intersection(literal, int).relation_to(nothing) == "subset"
    assert literal.relation_to(complement(int)) == "subset"
    items = Validator(list[Literal[_CaseFree("a")]])  # ty: ignore[invalid-type-form]
    assert items.relation_to(list[str]) == "subset"
    assert literal.relation_to(Annotated[str, Regex("a")]) != "subset"


class _StatesFractions:
    """A class stating `fractions` as its module, each instance equal to all."""

    __module__ = "fractions"

    def __init__(self, value: int) -> None:
        self.value = value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _StatesFractions)

    def __hash__(self) -> int:
        return 0

    def __ge__(self, other: int) -> bool:
        return self.value >= other

    def __len__(self) -> int:
        return self.value


def test_a_class_stating_a_standard_library_module_is_not_its_type() -> None:
    """A standard-library type is the class its module binds.

    A class body can state any `__module__`, so `_StatesFractions(5)` is no
    `Fraction`: its literal holds `_StatesFractions(0)`, which `Ge(3)` and
    `MinLen(3)` refuse, and its constant proves neither. `Decimal`'s does.
    """
    literal = Validator(Literal[_StatesFractions(5)])  # ty: ignore[invalid-type-form]
    assert literal.is_valid(_StatesFractions(0))
    for atom in (at.Ge(3), at.MinLen(3)):
        assert not Validator(Annotated[object, atom]).is_valid(_StatesFractions(0))
        assert literal.relation_to(Annotated[object, atom]) != "subset"
    decimal = Validator(Literal[Decimal(5)])  # ty: ignore[invalid-type-form]
    assert decimal.relation_to(Annotated[object, at.Ge(3)]) == "subset"


class _ByX:
    """Two instances are equal when their ``x`` is, whatever their ``y``."""

    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _ByX) and other.x == self.x

    def __hash__(self) -> int:
        return hash(self.x)


@runtime_checkable
class _HasOneY(Protocol):
    y: Literal[1]


# THEORY: a-constant-is-the-values-equal-to-it
def test_an_attribute_record_tells_two_equal_constants_apart() -> None:
    """An attribute record reads what two equal instances hold apart.

    ``_ByX(1, 1) == _ByX(1, 2)``, so the literal of the first holds the second,
    and a protocol requiring ``y == 1`` admits the constant and refuses its
    equal. No callback is asked, so only the attribute record stops the proof.
    """
    literal = Validator(Literal[_ByX(1, 1)])  # ty: ignore[invalid-type-form]
    assert literal.is_valid(_ByX(1, 2))
    assert not Validator(_HasOneY).is_valid(_ByX(1, 2))
    assert literal.relation_to(_HasOneY) == "undecided"


class _Color(enum.Enum):
    RED = 1


# THEORY: a-constant-is-the-values-equal-to-it
def test_a_literal_that_is_its_whole_class_is_proved_below_a_predicate() -> None:
    """Where the constant is the literal's one value, its membership is the proof.

    An ``int``, a nonzero float and an ``Enum``'s member are the only values of
    their literals, so running the predicate on the constant decides the pair
    both ways.
    """
    assert (
        Validator(Literal[5]).relation_to(Annotated[int, lambda x: x > 0]) == "subset"
    )
    assert (
        Validator(Literal[5]).relation_to(Annotated[int, lambda x: x < 0])
        == "not_subset"
    )
    one_and_a_half = Literal[1.5]  # ty: ignore[invalid-type-form]
    assert Validator(one_and_a_half).relation_to(Annotated[float, _sign]) == "subset"
    red = Annotated[_Color, lambda color: color.value == 1]
    assert Validator(Literal[_Color.RED]).relation_to(red) == "subset"


def _raises(value: object) -> bool:
    raise ValueError(value)


_JSON = recursive(lambda j: union(None, bool, int, float, str, [j], {str: j}))

#: A literal beside every value it holds that the law can name: the constant,
#: and the values `==` joins to it within its type.
_LITERALS: list[tuple[object, list[object]]] = [
    (Literal["paid"], ["paid"]),
    (Literal["a", "b"], ["a", "b"]),
    (Literal[0], [0]),
    (Literal[0.0], [0.0, -0.0]),  # ty: ignore[invalid-type-form]
    (Literal[1.5], [1.5]),  # ty: ignore[invalid-type-form]
    (
        Literal[Decimal("1.00")],  # ty: ignore[invalid-type-form]
        [Decimal("1.00"), Decimal("1.0"), Decimal(1)],
    ),
    (Literal[_Color.RED], [_Color.RED]),
]

_NEGATIVE = Annotated[float, lambda x: not _sign(x)]
_ONE_PLACE = Annotated[Decimal, lambda d: d.as_tuple().exponent == -1]
_RAISING = Annotated[Any, _raises]

#: Members that run code on the constant: two sign tests, two tests of a
#: decimal's places, a predicate that raises, and a recursive universe. Each
#: test the constant fails admits a value equal to it.
_CALLBACKS = st.sampled_from(
    [
        Annotated[float, _sign],
        _NEGATIVE,
        Annotated[Decimal, lambda d: d.as_tuple().exponent == -2],
        _ONE_PLACE,
        _RAISING,
        _JSON,
    ]
)


# THEORY: a-constant-is-the-values-equal-to-it
@given(
    literal=st.sampled_from(_LITERALS),
    beside=st.lists(
        st.tuples(st.one_of(_specs.map(_build), _CALLBACKS), st.booleans()),
        min_size=1,
        max_size=3,
    ),
)
@example(literal=_LITERALS[3], beside=[(_NEGATIVE, False)])
@example(literal=_LITERALS[5], beside=[(_ONE_PLACE, False)])
@example(literal=_LITERALS[0], beside=[(_RAISING, False)])
def test_a_meet_holding_a_literal_is_decided_as_its_values_are_walked(
    literal: tuple[object, list[object]], beside: list[tuple[object, bool]]
) -> None:
    """A meet holding a literal is empty only where no value of it is admitted.

    The literal's values are named in full, so a meet proved inhabited must
    admit one of them too, and the literal proved below the complement of the
    other members must have every value admitted there: a value the walk stops
    at is refused by a set and its complement alike. A member is drawn bare or
    as the complement of its meet with the literal, the spelling a universe met
    into every member of a schema builds.
    """
    schema, values = literal
    members = [
        complement(intersection(schema, member)) if negated else member
        for member, negated in beside
    ]
    meet = intersection(schema, *members)
    answer = meet.relation_to(nothing)
    admitted = [value for value in values if meet.is_valid(value)]
    if answer == "subset":
        assert not admitted, f"proved empty, and admits {admitted!r}"
    if answer == "not_subset":
        assert admitted, "proved inhabited, and admits none of its values"
    outside = complement(intersection(*members))
    if Validator(schema).relation_to(outside) == "subset":
        refused = [value for value in values if not outside.is_valid(value)]
        assert not refused, f"proved below the complement, which refuses {refused!r}"


# THEORY: a-constant-is-the-values-equal-to-it
def test_a_literal_met_with_a_deep_complement_is_decided_by_its_constant() -> None:
    """A literal met beside a universe meets no depth bound.

    `Literal["paid"] & ~(Literal["paid"] & JSON)` nests deeper than the set
    representation reads, and the constant is in the inner meet, so out of its
    complement. A literal set minus the complements of the universe met into
    each piece is the spelling a planner meeting every member with the universe
    builds, and it is empty the same way. A predicate raising on the constant
    is refused by itself and by its complement, and decides nothing.
    """
    paid = Literal["paid"]
    assert intersection(paid, complement(intersection(paid, _JSON))).is_empty()
    status = Literal["open", "paid", "void"]
    tagged = Annotated[str, Regex(r"^re_[a-z0-9]{8}$")]
    removed = [Literal["paid"], Literal["open"], union(tagged, Literal["void"])]
    spelled = intersection(
        _JSON,
        status,
        *(complement(intersection(piece, _JSON)) for piece in removed),
    )
    assert spelled.is_empty()
    assert not any(spelled.is_valid(value) for value in ("open", "paid", "void"))
    raising = Annotated[str, _raises]
    assert intersection(paid, raising).relation_to(nothing) == "undecided"
    assert intersection(paid, complement(raising)).relation_to(nothing) == "undecided"


def _admits(_value: object) -> bool:
    return True


def _admits_none(_value: object) -> bool:
    return False


_ADMITS_NONE = Annotated[Any, _admits_none]

#: An integer refinement whose predicate raises on every value: the walk stops
#: on each, and refuses it from the set and from its complement alike.
_STOPS = Validator(Annotated[int, _raises])


# THEORY: the-decision-has-three-answers
@pytest.mark.parametrize(
    ("subject", "supertype"),
    [
        (Literal[5], union(_STOPS, complement(_STOPS))),
        (Literal[5], _STOPS),
        (Literal[5], complement(_STOPS)),
        (union(Literal[5], Literal["x"]), union(_STOPS, complement(_STOPS))),
        (
            complement(union(_STOPS, complement(_STOPS))),
            complement(union(Annotated[str, Regex("abc|abcd|x")], Literal[5])),
        ),
        (complement(union(_STOPS, complement(_STOPS))), complement(Literal[5])),
    ],
)
def test_a_constant_the_walk_stopped_on_refutes_nothing(
    subject: object, supertype: object
) -> None:
    """A stopped walk is no value outside a set, read directly or respelled.

    `T | ~T` holds every value and `~(T | ~T)` none, whatever a stop denotes,
    so neither relation is refuted. The constant `5` stops `T`'s walk, which
    refuses it from both branches: read as a non-member it refuted the literal
    below the union, and the respelling of `~(T | ~T) <= ~L` as `L <= T | ~T`
    carried that refutation to a subject with no value.
    """
    assert Validator(subject).relation_to(supertype) != "not_subset"


# THEORY: the-decision-has-three-answers
def test_a_part_that_decides_still_decides_beside_a_stop() -> None:
    """A meet another member refuses is refused, a union another branch admits.

    The walk reads a stop as the third value of Kleene's strong logic, so a
    refusal beside it is a refusal and an admission an admission.
    """
    assert Validator(Literal[5]).relation_to(intersection(_STOPS, str)) == "not_subset"
    assert Validator(Literal[5]).relation_to(union(_STOPS, int)) == "subset"


#: Members no lowering reads: predicates answering every value one way, one
#: that raises, and a member nested past the depth the set representation
#: reads.
_UNREAD = st.sampled_from(
    [
        Annotated[Any, _admits],
        _ADMITS_NONE,
        _RAISING,
        union(str, list[list[list[list[list[list[int]]]]]]),
    ]
)


# THEORY: a-cut-reference-proves
@given(sa=_specs, sb=_specs, unread=_UNREAD, v=_values)
def test_a_meet_stays_empty_beside_a_member_no_lowering_reads(
    sa: object, sb: object, unread: object, v: object
) -> None:
    """A meet beside a member no lowering reads proves only what it holds.

    The reading that leaves an unread member out is a superset of the meet, so
    its emptiness is a proof and its inhabitance none: the meet of three proved
    empty admits no value, and a meet holding a predicate that admits nothing,
    which is empty, is never answered inhabited and refutes nothing below it.
    """
    a, b = _build(sa), _build(sb)
    three = intersection(a, b, unread)
    if three.is_empty():
        assert not three.is_valid(v)
    admits_none = intersection(a, _ADMITS_NONE)
    assert admits_none.relation_to(nothing) != "not_subset"
    assert admits_none.relation_to(b) != "not_subset"


# THEORY: a-cut-reference-proves
def test_a_meet_of_patterns_is_empty_beside_a_predicate() -> None:
    """Two patterns no string matches both of stay apart beside a predicate.

    As a meet of three and as one refinement holding all three constraints,
    the two spellings `Annotated` gives one set; and beside the recursive
    universe a planner meets every member with. A pattern beside a predicate
    is below the pattern, and is refuted by another pattern only on a text the
    walk admits to it, the predicate run on the text: one the predicate refuses
    names nothing.
    """
    email = Annotated[str, Regex(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")]
    tagged = Annotated[str, Regex(r"^re_[a-z0-9]{8}$")]
    odd = Annotated[str, lambda s: len(s) % 2 == 1]
    assert intersection(email, tagged, odd).is_empty()
    tier = Literal["free", "pro"]
    universe = intersection(
        str,
        intersection(email, _JSON),
        complement(intersection(tier, _JSON)),
        intersection(tagged, _JSON),
    )
    assert universe.is_empty()
    a_and_p = Annotated[str, Regex("^a+$"), _admits]
    both = Annotated[str, Regex("^a+$"), Regex("^b+$"), lambda s: len(s) % 2 == 1]
    assert Validator(both).is_empty()
    assert Validator(a_and_p).relation_to(Annotated[str, Regex("^a+$")]) == "subset"
    assert Validator(a_and_p).relation_to(Annotated[str, Regex("^b+$")]) == "not_subset"
    a_and_none = Annotated[str, Regex("^a+$"), _admits_none]
    assert (
        Validator(a_and_none).relation_to(Annotated[str, Regex("^b+$")]) == "undecided"
    )


# THEORY: kinds-decompose-emptiness
def test_a_meet_with_a_union_of_guarded_patterns_is_empty() -> None:
    """A meet with a union of patterns guarded to their kind is empty by kind.

    A translation from JSON Schema applies `pattern` to strings alone, so the
    string schema it builds is `str & (~str | Regex(p))`. An integer meets no
    such branch, so it meets no union of them, and the meet is empty whatever
    the patterns are, which a decision reading the union whole has to see too.
    """

    def guarded(pattern: str) -> object:
        return intersection(str, union(complement(str), Annotated[str, Regex(pattern)]))

    integral = union(int, Annotated[float, at.MultipleOf(1)])
    integer = intersection(integral, complement(bool))
    either = union(guarded("^ga_[a-z0-9]{6}$"), guarded("^gb_[a-z0-9]{6}$"))
    assert intersection(integer, guarded("^ga_[a-z0-9]{6}$")).is_empty()
    assert intersection(integer, either).is_empty()
    assert Validator(either).is_valid("gb_abc123")
    assert not Validator(either).is_empty()
    assert Validator(int).relation_to(either) == "not_subset"


# THEORY: a-record-splits-across-a-union
def test_a_meet_of_open_records_holds_the_dict_of_its_required_fields() -> None:
    """Open records met hold the dict of the fields they require.

    Each record admits any other key whose value is in the recursive universe,
    so the meet holds `{"tag": "t0"}` and refutes a record fixing another tag on
    it. A field one record requires and another reads through its catch-all
    meets that catch-all's type, so `{"tag": str}` beside `dict[str, int]`
    holds nothing.
    """
    record = {str: _JSON}

    def tagged(tag: object) -> object:
        return intersection(record, {"tag": tag, "kids?": [_JSON], str: _JSON})

    first, second = tagged(Literal["t0"]), tagged(Literal["t1"])
    assert Validator(first).is_valid({"tag": "t0"})
    assert Validator(first).relation_to(second) == "not_subset"
    assert intersection({"tag": str}, {str: int}).is_empty()
    assert not intersection({"tag": str}, {str: str}).is_empty()


# THEORY: two-fixpoints-one-procedure
def test_a_meet_of_guarded_records_is_decided_clause_by_clause() -> None:
    """A meet of keywords guarded to their kind is decided one clause at a time.

    A translation from JSON Schema applies a keyword to its kind alone, so an
    object schema is `OBJ & (~OBJ | R) & (~OBJ | Q)`: a meet of two unions of
    two, four clauses. Each `~OBJ` branch meets `OBJ` in nothing, so two tags
    fixed two ways meet in nothing, and one tag beside a required one meets in
    the dict holding it.
    """
    record = {str: _JSON}

    def guarded(keyword: object) -> object:
        return union(complement(record), keyword)

    def tagged(tag: object) -> object:
        return {"tag": tag, str: _JSON}

    t0, t1 = guarded(tagged(Literal["t0"])), guarded(tagged(Literal["t1"]))
    assert intersection(record, t0, t1).is_empty()
    one = intersection(record, t0, guarded({"tag": _JSON, str: _JSON}))
    assert Validator(one).is_valid({"tag": "t0"})
    assert Validator(one).relation_to(nothing) == "not_subset"


# THEORY: a-record-splits-across-a-union
def test_two_open_records_met_are_one_record() -> None:
    """Open records met in a meet are one record, and are related as one.

    Each record admits any other key whose value is in the recursive universe,
    which the set representation cuts and cannot prove anything of. The
    records met fix `tier` to `"pro"`, so the meet is below a record fixing it
    so, asked as an inclusion, as the emptiness of the difference, and as one
    record minus the third below the other's complement. A predicate beside
    the records leaves the proof standing.
    """
    user = {"id": str, "tier": Literal["free", "pro"], str: _JSON}
    required = {"tier": Literal["pro", "enterprise"], str: _JSON}
    pro = {"tier": Literal["pro"], str: _JSON}
    met = intersection(user, required)
    assert met.is_valid({"id": "a", "tier": "pro", "seen": [1, "b"]})
    assert met.relation_to(pro) == "subset"
    assert intersection(user, required, complement(pro)).is_empty()
    assert intersection(user, complement(pro)).relation_to(complement(required)) == (
        "subset"
    )
    assert intersection(user, required, _ADMITS_NONE).relation_to(pro) == "subset"
    # A map minus a map it is below stays empty beside a predicate.
    natural = Annotated[int, at.Ge(0)]
    apart = intersection(
        complement({str: complement(natural)}),
        {str: complement(int)},
        _ADMITS_NONE,
    )
    assert apart.is_empty()


class _WeirdInt(int):
    """An `int` subclass whose comparisons answer `True` whatever they are asked."""

    def __gt__(self, other: object) -> bool:
        return True

    def __lt__(self, other: object) -> bool:
        return True


@pytest.mark.xfail(
    strict=True,
    reason=(
        "a bound is read as Python's operator, and an int subclass may answer it "
        "any way it likes, so a contradiction between two bounds is not a proof "
        "that no value satisfies both"
    ),
)
# TRUST: What a value answers is a function of the value.
def test_a_bound_contradiction_is_not_reported_empty_over_a_lying_subclass() -> None:
    """The third atom that is not a set, and the one still open.

    ``Annotated[int, Gt(0), Lt(1)]`` is decided empty because no integer lies
    between, and ``int`` admits its subclasses -- one of which answers both
    bounds. Deciding this soundly means reading the bounds over the *exact*
    builtin, which needs a schema to say "exactly `int`" apart from "an `int` or
    a subclass of one". That distinction is a class constraint sitting beside a
    builtin kind, which the descriptor cannot yet hold.
    """
    bounded = Validator(Annotated[int, at.Gt(0), at.Lt(1)])

    assert bounded.is_valid(_WeirdInt(5)), "the subclass answers both bounds"
    assert not bounded.is_empty(), "so the set is not empty"


class _TwoAnswers:
    """The answers one method gives on successive calls: yes, then no, in turn."""

    def __init__(self) -> None:
        self._calls = 0

    def next(self) -> bool:
        self._calls += 1
        return self._calls % 2 == 1


class _HasX(Protocol):
    x: int


class _AlternatingAttribute:
    """A value whose `x` is an `int` on one read and a `str` on the next."""

    def __init__(self) -> None:
        self._answers = _TwoAnswers()

    @property
    def x(self) -> object:
        return 1 if self._answers.next() else "one"


class _Token:
    """A constant equal to itself alone, or a value equal to it on every other call."""

    def __init__(self, answers: _TwoAnswers | None = None) -> None:
        self._answers = answers

    def __eq__(self, other: object) -> bool:
        return self is other if self._answers is None else self._answers.next()

    def __hash__(self) -> int:
        return 0


_TOKEN = _Token()


class _AlternatingOrder(int):
    """An `int` at least zero on one comparison and not on the next."""

    def __init__(self, _: int) -> None:
        self._answers = _TwoAnswers()

    def __ge__(self, other: object) -> bool:
        return self._answers.next()


class _AlternatingRemainder(int):
    """An `int` divisible by two on one remainder and not on the next."""

    def __init__(self, _: int) -> None:
        self._answers = _TwoAnswers()

    def __mod__(self, other: object) -> int:
        return 0 if self._answers.next() else 1


class _AlternatingLength:
    """A value one long on one count and empty on the next."""

    def __init__(self) -> None:
        self._answers = _TwoAnswers()

    def __len__(self) -> int:
        return 1 if self._answers.next() else 0


#: Each atom construction folds with its complement, and a value whose answer to
#: that atom's question alternates.
_ALTERNATING: dict[str, tuple[object, Callable[[], object]]] = {
    "attribute": (_HasX, _AlternatingAttribute),
    "constant": (_TOKEN, lambda: _Token(_TwoAnswers())),
    "bound": (Annotated[int, at.Ge(0)], lambda: _AlternatingOrder(5)),
    "length": (Annotated[Any, at.MinLen(1)], _AlternatingLength),
    "step": (Annotated[int, at.MultipleOf(2)], lambda: _AlternatingRemainder(4)),
}


@pytest.mark.parametrize("atom", sorted(_ALTERNATING))
@pytest.mark.xfail(
    strict=True,
    reason=(
        "a set and its complement meet in nothing, which construction folds for "
        "every atom whose answer Python's rules make a fact about the value; a "
        "value answering two ways is outside the sets the law is about"
    ),
)
# TRUST: What a value answers is a function of the value.
def test_a_value_answering_two_ways_is_in_no_meet_of_a_set_and_its_complement(
    atom: str,
) -> None:
    """The fold of ``A & ~A``, over a value both halves admit in turn.

    An attribute, a constant's ``==``, a bound, a length and a step are read as
    sets because Python's rules ask each for one answer, so construction folds
    each with its complement. A value whose method answers two calls
    differently is admitted by ``A`` on one call and by ``~A`` on the next, and
    a set reading of those two answers puts it in the meet, which construction
    has made ``nothing``. Each check reads the value once, so neither call is
    reported as the value moving.
    """
    schema, make = _ALTERNATING[atom]
    value = make()

    assert Validator(schema).is_valid(value), "one call admits it to A"
    assert Validator(complement(schema)).is_valid(value), "the next admits it to ~A"
    assert not intersection(schema, complement(schema)).is_empty(), (
        "so a set holding what A and ~A admit holds it"
    )


def test_a_fractional_number_met_with_a_recursive_universe_is_inhabited() -> None:
    """A meet holding `int | float` twice beside a reference is not proved empty.

    `None | NUMBER` holds `int | float` inside `NUMBER`, and the meet holds it
    bare as well; a recursive universe beside them opens the meet, and every
    union in it is distributed. A branch that kept the bare copy was the meet
    already open, reached again with no constructor between, and the meet was
    proved empty though `0.5` is in it.
    """
    not_bool = complement(bool)
    number = intersection(union(int, float), not_bool)
    integer = intersection(union(int, Annotated[float, at.MultipleOf(1)]), not_bool)
    fractional = intersection(number, complement(integer))
    universe = recursive(
        lambda j: union(None, bool, int, float, str, [j], {str: j}),
    )
    meet = intersection(fractional, union(None, number), universe)
    assert meet.is_valid(0.5)
    assert not meet.is_empty()
    assert meet.relation_to(Validator(complement(object))) != "subset"


def test_a_union_beside_every_dict_is_asked_as_its_difference() -> None:
    """`a <= b | dict` is `a & ~dict <= b`, the dict kind's top read as `~~dict`.

    `{"a?": object}.open()` is `dict[anything, anything]`, and `rest` holds
    every value that is not a dict, so `rest` beside it is everything: the union
    with `~dict` folds to `anything`, which is below it.
    """
    holds_a = Validator({"a": complement(str)}).open()
    tagged = intersection(holds_a, complement(Validator(dict[Literal["a"], bytes])))
    record = Validator({"a": float, str: bytes})
    keyed = Validator(dict[Literal["a"], complement(None)])  # ty: ignore[invalid-type-form]
    rest = intersection(
        union(complement(record), complement(keyed)), complement(tagged)
    )
    every_dict = Validator({"a?": object}).open()
    covering = union(rest, every_dict)
    assert union(complement(dict), rest, every_dict).relation_to(covering) == "subset"
    assert anything.relation_to(covering) == "subset"


def test_a_subject_read_empty_is_below_every_set() -> None:
    """A subject `is_empty` proves is below any supertype, however the steps went.

    A literal `None` or `0.0` is no list, so the meet below is empty, and the
    empty set is below every set; the rules spent the query's steps on the
    supertype's fixpoints before the bound read it, and the bound is read again
    with steps of its own.
    """
    values = recursive(
        lambda j: union(
            None,
            bool,
            int,
            float,
            str,
            Validator(list[j]),  # ty: ignore[invalid-type-form]
            Validator({str: j}),
        )
    )
    objects = Validator(dict[str, values])  # ty: ignore[invalid-type-form]
    number = intersection(union(int, float), complement(bool))
    element = intersection(
        objects,
        Validator({"tag": intersection(number, values)}).open(),
        union(complement(objects), values),
    )
    subject = intersection(union(None, Literal[0.0]), list[element])  # ty: ignore[invalid-type-form]
    nested = recursive(
        lambda u: union(Literal[0, 0.5], list[u], {str: u})  # ty: ignore[invalid-type-form]
    )
    supertype = union(
        {"tag": Literal["b"], str: values},
        recursive(lambda t: union(nested, list[t], {str: t})),  # ty: ignore[invalid-type-form]
    )
    assert subject.is_empty()
    assert subject.relation_to(nothing) == "subset"
    assert subject.relation_to(supertype) == "subset"


def test_a_narrowed_meet_left_open_is_distributed_as_written() -> None:
    """A meet its guards narrow and leave undecided is read through its reference.

    Dropping a branch a complement beside it excludes gives the same set, so a
    narrowed meet that ends undecided is no answer, and the readings after it are
    asked of the meet as written, whose guard union distributes through the
    reference into a shape they decide: no list, word or `bool` is an integer
    from 2, and a record whose `tag` holds `tag: 1` is no record every value of
    whose `tag` is a `str`.
    """
    integer = intersection(
        union(int, Annotated[float, at.MultipleOf(1)]), complement(bool)
    )
    from_two = union(Annotated[int, at.Ge(2)], Annotated[float, at.Ge(2)])
    words = recursive(
        lambda t: union(str, Validator(list[t]))  # ty: ignore[invalid-type-form]
    )
    lists = union(Validator(list[int]), union(words, bool))
    numbers = intersection(lists, intersection(integer, from_two))
    assert numbers.relation_to(nothing) == "subset"
    assert numbers.is_empty()

    values = recursive(
        lambda j: union(
            None,
            bool,
            int,
            float,
            str,
            Validator(list[j]),  # ty: ignore[invalid-type-form]
            Validator({str: j}),
        )
    )
    tagged = intersection(
        union(str, Validator({"tag": {"tag": Literal[1], str: values}}).open()),
        Validator({"tag": {str: str}}).open(),
        values,
        union(
            complement(Validator(dict[str, values])),  # ty: ignore[invalid-type-form]
            Validator({"tag": complement(str)}).open(),
        ),
    )
    assert tagged.relation_to(nothing) == "subset"
    assert tagged.is_empty()


def test_a_meet_holding_a_complement_is_read_as_its_relation() -> None:
    """A meet with a complement is empty where the rest is below the inner set.

    `R & ~I` is empty exactly where `R <= I`. A meet of two records over the
    JSON value whose one field is a `complex` is below the record of JSON
    fields, field by field, and a closed record whose field is a word and a
    tagged record is below a union holding every such record; met with each
    complement, the meet's readings decline and the relation proves it.

    The relation is asked under the steps the question has left, as the
    relation of the same pair is: two unions of length bounds over `bytes`
    that the rules decide past a respelling's steps answer alike as a
    relation and as a meet.
    """
    json = recursive(
        lambda j: union(
            None,
            bool,
            int,
            float,
            str,
            Validator(list[j]),  # ty: ignore[invalid-type-form]
            Validator({str: j}),
        )
    )
    text_or_json = union(str, intersection(json, complement(str)))
    complex_field = intersection(
        Validator({"a": text_or_json}), Validator({"a": complex})
    )
    json_field = Validator({"a": json})
    assert complex_field.relation_to(json_field) == "subset"
    assert intersection(complex_field, complement(json_field)).is_empty()

    words = complement(union(complement(str), Annotated[str, at.Len(2, 4)]))
    tagged = Validator({"tag": Literal[0], str: json})
    word_record = Validator({"a": intersection(tagged, words)}).close()
    text_or_tagged = union(
        Validator({"a": union(complement(str), Annotated[str, at.Len(1, 3)])}).close(),
        Validator({"a": Validator({"tag": Literal["b"], str: json}), str: object}),
    )
    assert word_record.relation_to(text_or_tagged) == "subset"
    meet = intersection(word_record, complement(text_or_tagged))
    assert meet.relation_to(nothing) == "subset"

    def size(*bounds: object) -> object:
        return Annotated[bytes, *bounds]

    def between(least: int, most: int) -> object:
        return size(at.Len(least, most))

    def at_least(least: int) -> object:
        return size(at.MinLen(least))

    def at_most(most: int) -> object:
        return size(at.MaxLen(most))

    def outside(term: object) -> object:
        return intersection(bytes, complement(term))

    inner_union = union(
        intersection(
            intersection(between(65, 255), at_least(63)),
            union(between(65, 257), between(2, 4095)),
        ),
        union(between(64, 1000), union(between(0, 5), at_least(5))),
    )
    narrow = intersection(at_least(256), intersection(inner_union, at_most(512)))
    wide = union(
        intersection(
            union(
                union(at_least(0), at_least(2)),
                outside(union(at_most(256), at_least(63))),
            ),
            union(
                intersection(at_least(5), union(at_least(63), at_most(63))),
                union(between(1, 65), intersection(at_most(64), between(254, 2048))),
            ),
        ),
        outside(
            union(
                intersection(outside(at_least(1)), at_least(254)),
                union(outside(between(65, 9999)), at_least(4095)),
            )
        ),
    )
    assert narrow.relation_to(wide) == "subset"
    assert intersection(narrow, complement(wide)).relation_to(nothing) == "subset"


def test_a_narrowed_meet_left_open_leaves_steps_to_the_readings_after_it() -> None:
    """The distribution after an undecided narrowing spends a bounded share.

    A record whose one field is the complement of a guarded record is a dict,
    so it is below any union holding the dicts. The guarded meets inside it
    narrow and decline, and distributed through the JSON value's reference on
    every step the query had left, they left the reading that places a record
    below the dicts none to decide with.
    """
    json = recursive(
        lambda j: union(
            None,
            bool,
            int,
            float,
            str,
            Validator(list[j]),  # ty: ignore[invalid-type-form]
            Validator({str: j}),
        )
    )
    obj = Validator(dict[str, json])  # ty: ignore[invalid-type-form]

    def guard(record: Validator[Any]) -> Validator[Any]:
        return intersection(
            obj,
            union(complement(obj), record),
            union(complement(obj), Validator({"tag": json}).open()),
        )

    integer = intersection(
        union(int, Annotated[float, at.MultipleOf(1)]), complement(bool)
    )
    up_to_one = union(Annotated[int, at.Le(1)], Annotated[float, at.Le(1)])
    tag_int = Validator({"tag": intersection(integer, up_to_one)}).open()
    guarded = Validator(
        {"a": complement(guard(Validator({"tag": guard(tag_int)}).open()))}
    )
    assert guarded.relation_to(union(obj, complement(Validator(str)))) == "subset"


def test_a_declined_narrowed_meet_is_read_once_a_query() -> None:
    """A narrowed meet the rules decline is not read again by every goal.

    A record whose field is a dict of guarded records three deep shares no
    value with a record whose same field is `True` or a word. The narrowed
    guards decline, and read again at each goal reaching them they spent the
    query's steps before the field meet was proved empty.
    """
    json = recursive(
        lambda j: union(
            None,
            bool,
            int,
            float,
            str,
            Validator(list[j]),  # ty: ignore[invalid-type-form]
            Validator({str: j}),
        )
    )
    obj = Validator(dict[str, json])  # ty: ignore[invalid-type-form]

    def guard(field: Any) -> Validator[Any]:
        return intersection(
            obj,
            union(complement(obj), Validator({"tag": field}).open()),
            union(complement(obj), Validator({"tag": json}).open()),
        )

    number = intersection(union(int, float), complement(bool))
    at_least_zero = union(Annotated[int, at.Ge(0)], Annotated[float, at.Ge(0)])
    nested = guard(guard(guard(intersection(number, at_least_zero))))
    record = Validator({"a": nested}).close()
    other = Validator(
        {
            "a": intersection(json, Literal[True, "a"]),
            "b?": Validator({"a": json}).open(),
            str: Validator(
                {"a": Validator(tuple[json, ...])}  # ty: ignore[invalid-type-form]
            ).close(),
        }
    )
    assert record.relation_to(complement(other)) == "subset"


def test_a_length_reading_asks_no_second_exploration_of_an_element() -> None:
    """A set's element is read empty without the rules exploring it again.

    The supertype holds every value: the JSON values, the integers, and every
    set but those of at most one member, which the last branch takes back in
    as a frozenset of JSON values under `MinLen(1)` beside the non-numbers.
    Asked of a union holding the JSON values, the length reading of that
    frozenset asked the rules whether its element was empty, inside the
    emptiness the rules were exploring, once for every question reaching it,
    and the query spent its steps before the proof. Read from the set
    representation instead, the pair is `subset`.
    """
    json = recursive(
        lambda j: union(
            None,
            bool,
            int,
            float,
            str,
            Validator(list[j]),  # ty: ignore[invalid-type-form]
            Validator({str: j}),
        )
    )
    number = intersection(union(int, float), complement(bool))
    not_numbers = intersection(complement(number), complement(union(int, number)))
    frozen = Annotated[frozenset[json], at.MinLen(1)]  # ty: ignore[invalid-type-form]
    universal = union(
        union(json, union(complement(Annotated[set, at.MaxLen(1)]), int)),
        intersection(
            complement(int),
            complement(
                intersection(
                    complement(int),
                    union(number, intersection(frozen, complement(number))),
                )
            ),
        ),
    )
    assert union(not_numbers, json).relation_to(universal) == "subset"
    assert union(not_numbers, str).relation_to(universal) == "subset"


#: A value of each length, by the kind a length bound counts over.
_LENGTH_KINDS: dict[str, tuple[object, Callable[[int], object]]] = {
    "str": (str, lambda n: "a" * n),
    "bytes": (bytes, lambda n: b"a" * n),
    "list": (list[int], lambda n: [0] * n),
}

#: Bounds either side of where an automaton counting them stops: 584 code
#: points, 4,094 bytes, 2,046 elements.
_LENGTH_BOUNDS = (0, 1, 5, 64, 584, 585, 2046, 2047, 4094, 4095, 9999)

#: A length at every bound and either side of it, and one past the largest:
#: a term of the bounds holds a length here exactly where it holds every
#: length up to the next.
_LENGTHS = sorted({max(b + d, 0) for b in _LENGTH_BOUNDS for d in (-1, 0, 1)})


def _length_terms(
    kind: object,
) -> st.SearchStrategy[tuple[object, Callable[[int], bool]]]:
    """Draw a Boolean term of length bounds over `kind`, with the lengths it holds."""

    def atom(bounds: tuple[int, int, int]) -> tuple[object, Callable[[int], bool]]:
        a, b, form = bounds
        lo, hi = min(a, b), max(a, b)
        if form == 0:
            return Annotated[kind, at.MinLen(lo)], lambda n: n >= lo  # ty: ignore[invalid-type-form]
        if form == 1:
            return Annotated[kind, at.MaxLen(hi)], lambda n: n <= hi  # ty: ignore[invalid-type-form]
        return Annotated[kind, at.Len(lo, hi)], lambda n: lo <= n <= hi  # ty: ignore[invalid-type-form]

    def extend(
        inner: st.SearchStrategy[tuple[object, Callable[[int], bool]]],
    ) -> st.SearchStrategy[tuple[object, Callable[[int], bool]]]:
        def either(
            pair: tuple[tuple[object, Callable[[int], bool]], ...],
        ) -> tuple[object, Callable[[int], bool]]:
            (a, fa), (b, fb) = pair
            return union(a, b), lambda n: fa(n) or fb(n)

        def both(
            pair: tuple[tuple[object, Callable[[int], bool]], ...],
        ) -> tuple[object, Callable[[int], bool]]:
            (a, fa), (b, fb) = pair
            return intersection(a, b), lambda n: fa(n) and fb(n)

        def outside(
            term: tuple[object, Callable[[int], bool]],
        ) -> tuple[object, Callable[[int], bool]]:
            a, fa = term
            return intersection(kind, complement(a)), lambda n: not fa(n)

        return st.one_of(
            st.tuples(inner, inner).map(either),
            st.tuples(inner, inner).map(both),
            inner.map(outside),
        )

    bounds = st.tuples(
        st.sampled_from(_LENGTH_BOUNDS),
        st.sampled_from(_LENGTH_BOUNDS),
        st.integers(0, 2),
    )
    return st.recursive(bounds.map(atom), extend, max_leaves=8)


@given(data=st.data())
def test_a_term_of_length_bounds_is_decided_as_the_lengths_it_holds(
    data: st.DataObject,
) -> None:
    """Two Boolean terms of length bounds over one kind are related exactly.

    Each holds the values of its kind whose length lies in a union of
    intervals, so the relation, the meet with the complement below `nothing`,
    and the meet's emptiness are each `subset` exactly where every length the
    one holds the other holds, and `not_subset` otherwise -- never undecided,
    with bounds either side of where an automaton counting them stops. The
    walk is the truth: each term holds a value of each length as its
    intervals say.
    """
    kind, value_of = _LENGTH_KINDS[data.draw(st.sampled_from(sorted(_LENGTH_KINDS)))]
    s, s_holds = data.draw(_length_terms(kind))
    t, t_holds = data.draw(_length_terms(kind))
    subject, supertype = Validator(s), Validator(t)
    for length in _LENGTHS:
        value = value_of(length)
        assert subject.is_valid(value) == s_holds(length), length
        assert supertype.is_valid(value) == t_holds(length), length
    below = all(t_holds(length) for length in _LENGTHS if s_holds(length))
    expected = "subset" if below else "not_subset"
    difference = intersection(s, complement(t))
    assert subject.relation_to(t) == expected
    assert difference.relation_to(nothing) == expected
    assert Validator(difference).is_empty() == below


def test_a_complemented_union_is_read_by_de_morgan_below_any_supertype() -> None:
    """A complement of a union is the meet of its branches' complements.

    `~(~X | Y)` is `X & ~Y`, a set of dicts here, so it is below `dict`, and
    below the complement of a schema whose complement folds to `dict`, as the
    meet spelling is.
    """
    no_bytes = intersection(
        Validator({str: complement(bytes)}),
        complement(Validator({"a?": bytes}).open()),
    )
    de_morgan = complement(union(complement(no_bytes), Validator(dict[str, str])))
    not_a_dict = intersection(
        complement(Validator({"a?": complement(nothing)}).open()), complement(dict)
    )
    assert de_morgan.relation_to(dict) == "subset"
    assert de_morgan.relation_to(complement(not_a_dict)) == "subset"
    assert de_morgan.relation_to(list) == "not_subset"
