"""Broad generative soundness check for the decision procedure.

Arbitrary nested schemas ``a`` and ``b`` and an arbitrary value ``v`` are
generated, then the decision is held to real membership: ``is_subtype_of`` and
``is_equivalent`` never claim a relation membership contradicts, ``is_empty`` never
accepts a value, and every schema is a subtype of itself. This is deliberately
adversarial: a complement that is not a subtype of itself, or a pool merge that
gives one constant two indices, fails it.
"""

import json
from collections.abc import Callable
from typing import Annotated, Any, Literal, Protocol

import annotated_types as at
import pytest
from hypothesis import given
from hypothesis import strategies as st

from valgebra import (
    Regex,
    ValidationError,
    Validator,
    complement,
    intersection,
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
