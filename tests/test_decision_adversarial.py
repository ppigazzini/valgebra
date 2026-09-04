"""Broad generative soundness check for the decision procedure.

Arbitrary nested schemas ``a`` and ``b`` and an arbitrary value ``v`` are
generated, then the decision is held to real membership: ``is_subtype_of`` and
``is_equivalent`` never claim a relation membership contradicts, ``is_empty`` never
accepts a value, and every schema is a subtype of itself. This is deliberately
adversarial — it found the complement-reflexivity and pool-merge bugs.
"""

import json
from typing import Annotated, Literal

import annotated_types as at
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from valgebra import (
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
    # test_metamorphic.py; recursion reflexivity holes are tracked in
    # test_completeness_ledger.py. This fuzzer holds the decision to membership.
    try:
        a, b = _build(sa), _build(sb)
        left, right = Validator(a), Validator(b)
    except (ValueError, TypeError, NotImplementedError, RecursionError):
        # An unbuildable combination is not under test; reject it through assume
        # so Hypothesis counts it toward the rejection rate rather than passing.
        assume(False)
        return
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
    # simplify preserves acceptance, the JSON path matches validating the parsed
    # value, and ensure returns the input unchanged exactly when it is a member.
    try:
        compiled = Validator(_build(sa))
    except (ValueError, TypeError, NotImplementedError, RecursionError):
        return
    member = compiled.is_valid(v)
    try:
        compiled.validate(v)
        explained = True
    except ValidationError:
        explained = False
    assert member == explained
    assert compiled.simplify().is_valid(v) == member
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

    assert meet.is_valid(7), "the alternating answers admit the value"
    assert not meet.is_empty(), "so the meet is not empty"


def test_a_hooked_class_is_not_reported_empty_against_itself() -> None:
    """The same law, and the same reason, one atom along.

    ``isinstance`` against a metaclass that overrides ``__instancecheck__`` runs
    user code, so the class is not a fixed set either. A class whose metaclass
    leaves the hooks alone *is* one, and the law still applies to it -- the
    restriction is to the atoms it does not hold for, not to the law.
    """
    hooked = Validator(intersection(_Coin, complement(_Coin)))
    assert hooked.is_valid(7), "the alternating answers admit the value"
    assert not hooked.is_empty(), "so the meet is not empty"

    pure = Validator(intersection(_Pure, complement(_Pure)))
    assert not pure.is_valid(_Pure()), "no value is in a class and outside it"
    assert pure.is_empty(), "and the law still decides that"


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
