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
    complement,
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
