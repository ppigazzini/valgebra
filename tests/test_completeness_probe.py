"""Search for relations the procedure answers `False` that no value refutes.

Every other instrument in this tree can only notice a completeness gap that
someone already thought of. The ledger in ``tests/test_completeness_ledger.py``
enumerates relations a human wrote down; the property suites assert soundness,
which inspects a ``True`` and has nothing to say about a ``False``; a mutation
sweep changes code that exists and cannot report a rule never written. So a
missing rule was invisible to all of them at once, which is how the lattice
bounds stayed keyed on the ``Nothing``/``Anything`` atoms rather than on
emptiness while every gate stayed green.

This probe is the missing direction. ``is_subtype_of`` is sound, so a ``False``
means "not proven" -- but when a wide value universe holds no witness at all (no
value in ``a`` and outside ``b``), that ``False`` is a *suspected gap*: the
relation looks true and the procedure did not see it. Suspected, not certain,
because the universe is finite; that is exactly why the result is a ledger to
read rather than a failure.

Held in both directions:

* a suspected gap with no ledger entry fails, so an incompleteness cannot arrive
  unnoticed -- the direction that was missing;
* a ledger entry the procedure now decides, or that a witness now refutes, fails,
  so an excuse cannot outlive the gap it excuses.

And one hard failure that is not a ledger matter at all: a relation decided
``True`` that a value refutes is **unsoundness**, the contract this library
actually promises.

LEDGER: every suspected completeness gap is accepted with a reason
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal, Protocol, TypedDict, runtime_checkable

import annotated_types as at
import pytest

from valgebra import Regex, Validator, complement, intersection, recursive, union


class _Rec(TypedDict):
    a: int


class _Rec2(TypedDict):
    a: int
    b: str


class _Plain:
    """A class with no hook, and the set `_Adopts` answers for."""


class _AdoptsMeta(type):
    """A metaclass that answers both class questions by running code.

    It answers for `_Plain`'s instances rather than for everything, so the class
    denotes a set a value can be outside of: a hook that admits every value
    makes every relation into it vacuously true, and a universe cannot tell such
    a class from a procedure that declines to read it.
    """

    def __instancecheck__(cls, instance: object) -> bool:
        return isinstance(instance, _Plain)

    def __subclasscheck__(cls, subclass: type) -> bool:
        return issubclass(subclass, _Plain)


class _Adopts(metaclass=_AdoptsMeta):
    """A class whose instance and subclass questions run code."""


@runtime_checkable
class _HasX(Protocol):
    """A runtime-checkable protocol with a data member, whose `issubclass` raises."""

    x: int


class _HasXValue:
    """A value the protocol above admits, so the protocol's set is not empty."""

    x = 1


class _HashableList(list):
    """A hashable `list`, which is a legal member of a `set` of lists."""

    __hash__ = object.__hash__  # type: ignore[assignment]


def _v(annotation: Any) -> Validator:
    return annotation if isinstance(annotation, Validator) else Validator(annotation)


_GE0 = Annotated[int, at.Ge(0)]
_GE1 = Annotated[int, at.Ge(1)]

# A fixpoint and its own unfolding: two spellings of one set.
_LINKED = recursive(lambda t: union(None, {"a": int, "next": t}))


# Small schemas, chosen to cross the kinds the decision procedure treats
# differently: scalars, the gradual atom, literals, containers, records, maps,
# refinements, and each algebra node. Names are the ledger's keys, so they are
# stable.
#
# The refinements carry both constraint families deliberately. An order bound
# entails a looser one by value, so those atoms report nothing; a regex is opaque
# to the entailment and reports. A universe holding only the decided family would
# look clean while saying nothing about the other, which is the shape of a probe
# that cannot fail.
SCHEMAS: list[tuple[str, Validator]] = [
    ("int", _v(int)),
    ("bool", _v(bool)),
    ("str", _v(str)),
    ("bytes", _v(bytes)),
    ("float", _v(float)),
    ("none", _v(None)),
    ("object", _v(object)),
    # `Any` is the top, spelled: `typing.Any` and `anything` build one node and
    # every relation reads it alike, so this row is the `anything` row under the
    # name a caller writes. It is kept for that -- the probe's universe is what a
    # reader checks a claim about `Any` against -- and not because the two can
    # answer differently.
    ("Any", _v(Any)),
    ("int|str", _v(int | str)),
    ("~int", complement(int)),
    ("~str", complement(str)),
    ("int&~int", intersection(int, complement(int))),
    ("int|~int", union(int, complement(int))),
    ("Lit['a']", _v(Literal["a"])),
    ("Lit['a','b']", _v(Literal["a", "b"])),
    ("str&~Lit['a']", intersection(str, complement(Literal["a"]))),
    ("list[int]", _v(list[int])),
    ("list[bool]", _v(list[bool])),
    ("list[object]", _v(list[object])),
    ("tuple[int,str]", _v(tuple[int, str])),
    ("set[int]", _v(set[int])),
    ("dict[str,int]", _v(dict[str, int])),
    ("dict[str,object]", _v(dict[str, object])),
    ("dict[Lit['a'],int]", _v(dict[Literal["a"], int])),
    ("dict[Lit['a','b'],int]", _v(dict[Literal["a", "b"], int])),
    ("dict[object,int]", _v(dict[object, int])),
    ("{a:int}", _v(_Rec)),
    ("{a:int,...}", _v(_Rec).open()),
    ("{a:int,b:str}", _v(_Rec2)),
    ("{a:int}|{a:int,b:str}", union(_Rec, _Rec2)),
    # A record whose fields each take two types, beside the union of the four
    # records that fix both -- one set through two shapes, and the pair that
    # separates a decider reading the set from one reading how it was written.
    # The union spelled as a whole and the corners spelled apart are the two
    # sides of De Morgan, and a search over values cannot tell them apart:
    # neither the proof nor the decline admits a value the other refuses.
    ("{a:int|str,b:int|str}", _v({"a": int | str, "b": int | str})),
    (
        "{a:int,b:int}|{a:int,b:str}|{a:str,b:int}|{a:str,b:str}",
        union(
            {"a": int, "b": int},
            {"a": int, "b": str},
            {"a": str, "b": int},
            {"a": str, "b": str},
        ),
    ),
    ("int&Ge(0)", _v(Annotated[int, at.Ge(0)])),
    ("int&Ge(1)", _v(Annotated[int, at.Ge(1)])),
    ("str&Regex['a']", _v(Annotated[str, Regex("a")])),
    ("str&Regex['ab?']", _v(Annotated[str, Regex("ab?")])),
    # A meet of two refinements and the single refinement carrying both bounds
    # denote one set through two shapes, which is the pair that separates a rule
    # reading the schema from one reading the constraints.
    ("int&Ge(0)&Ge(1)", intersection(_GE0, _GE1)),
    ("int&(Ge(0),Ge(1))", _v(Annotated[int, at.Ge(0), at.Ge(1)])),
    # A fixpoint, whose definitions table nothing else in this universe reaches.
    ("mu t.None|{a:int,next:t}", _LINKED),
    ("None|{a:int,next:mu t}", union(None, {"a": int, "next": _LINKED})),
    # The edges of each kind's universe, which a schema written for a reader
    # does not reach and a relation is decided at: the end of the integer
    # carrier, the first float no integer equals, a key kind that is a subset of
    # another, a set whose element kind is unhashable, and the two class
    # questions a metaclass answers by running code.
    ("Lit[-2**63+1]", _v(Literal[-(2**63) + 1])),  # ty: ignore[invalid-type-form]
    ("int&Ge(-2**63+1)", _v(Annotated[int, at.Ge(-(2**63) + 1)])),
    ("float&Lt(2**53+1)", _v(Annotated[float, at.Lt(2**53 + 1)])),
    ("dict[bool,int]", _v(dict[bool, int])),
    ("dict[Lit[True,False],int]", _v(dict[Literal[True, False], int])),
    ("dict[int,str]", _v(dict[int, str])),
    ("{a?:int,int:str}", _v({"a?": int, int: str})),
    ("set[list[int]]", _v(set[list[int]])),
    ("Adopts", _v(_Adopts)),
    ("Plain", _v(_Plain)),
    ("HasX", _v(_HasX)),
    # The pair a polarity cut widens: every list of `a+` strings is a member of
    # the fixpoint, through `list[a*] <= list[X] <= X`.
    ("list[str&Regex['a+']]", _v(list[Annotated[str, Regex("a+")]])),
    (
        "mu X.list[X]|str&Regex['a*']",
        recursive(
            lambda x: union(list[x], Annotated[str, Regex("a*")]),  # ty: ignore[invalid-type-form]
        ),
    ),
]


class _Obj:
    a = 1


# The seed of the universe. A thin universe turns a decided-false relation into
# a reported gap, so each addition here is a false report removed: the non-string
# key is what separates an open record (whose catch-all admits any key) from a
# `dict[str, ...]`, and without it the two look equal.
#
# The seed is where a reader adds a value by hand. `VALUES` below is derived
# from it and from `SCHEMAS`, because a corpus written by hand holds the values
# its author thought of and a schema decides at edges the author never saw.
_SEED: list[Any] = [
    # A link of the fixpoint above, and the value that separates it from `None`:
    # without one, every relation with the fixpoint on the left looks true.
    {"a": 1, "next": None},
    {"a": 1, "next": {"a": 2, "next": None}},
    0,
    1,
    2,
    -1,
    True,
    False,
    3.5,
    0.0,
    "",
    "a",
    "b",
    "ab",
    b"",
    b"a",
    None,
    [],
    [1],
    [True],
    ["a"],
    [1, 2],
    [[1]],
    (),
    (1,),
    (1, "a"),
    ("a", 1),
    set(),
    {1},
    {"a"},
    frozenset(),
    frozenset({1}),
    {},
    {"a": 1},
    {"a": "x"},
    {"b": 2},
    {"a": 1, "b": "s"},
    {"a": 1, "b": 2},
    {1: 1},
    {"": 1},
    {"a": 1, 1: 2},
    {"a": 1, (): 2},
    _Obj(),
    object(),
    Ellipsis,
    range(3),
    {"a": [1]},
    {"a": {"b": 1}},
    [{"a": 1}],
    ({"a": 1},),
]

#: The values at the edge of a kind's universe, which no schema has to name and
#: every relation over that kind is decided at: the ends of the integer carrier
#: the descriptor lifts a residue class across, the first float no integer
#: equals, the value outside every order, the newline a length bound counts, and
#: the members a kind is said not to have -- a hashable `list`, and an instance
#: of a class whose metaclass answers both questions by running code.
_KIND_EDGES: list[Any] = [
    -(2**63),
    -(2**63) + 1,
    2**63 - 1,
    2**63,
    2**53,
    2**53 + 1,
    float(2**53),
    float("nan"),
    float("inf"),
    -0.0,
    "\n",
    "a\nb",
    b"\n",
    _HashableList([1]),
    _Plain(),
    _Adopts(),
    _HasXValue(),
]

#: How a value is carried into a container, by the shape that carries it. A
#: shape is used only where some schema admits a value of it, so the universe
#: grows with the schemas rather than with this table.
_SHAPES: list[tuple[Any, Any]] = [
    ([1], lambda value: [value]),
    ((1,), lambda value: (value,)),
    ({1}, lambda value: {value}),
    (frozenset({1}), lambda value: frozenset({value})),
    ({"a": 1}, lambda value: {"a": value}),
    ({"z": 1}, lambda value: {"z": value}),
    ({1: 1}, lambda value: {value: 1}),
    ({1: "a"}, lambda value: {value: "a"}),
    # Two named keys at once, and the same dict carrying a third key no record
    # names. A shape that fills one slot builds neither: a two-field record
    # admits no dict with one key, and the value that separates an open record
    # from a closed one is a dict with a key besides the ones declared. Without
    # them a refutation about either is true and has no witness here, which
    # reads as a suspected unsoundness rather than as a thin universe.
    ({"a": 1, "b": "x"}, lambda value: {"a": value, "b": "x"}),
    ({"a": 1, "b": 1}, lambda value: {"a": value, "b": 1}),
    ({"a": 1, "b": "x", "z": 1}, lambda value: {"a": 1, "b": "x", "z": value}),
]


def _built(into: Any, value: Any) -> Any:
    """Carry the value into the shape, or report nothing where it cannot go."""
    try:
        return into(value)
    except TypeError:
        return None


def _carried(value: Any) -> bool:
    """Report whether a value is one a container is filled with here.

    A scalar, or anything a key and a set member can be: the element that
    separates a set of an unhashable kind from a set of another is a hashable
    subclass of that kind, so a container is not excluded for being one.
    """
    if not isinstance(value, (list, tuple, set, frozenset, dict)):
        return True
    try:
        hash(value)
    except TypeError:
        return False
    return True


def _universe(schemas: list[tuple[str, Validator]]) -> list[Any]:
    """Build the seed and the kind edges, and carry each into the schemas' shapes.

    A container schema and a container of its complement are told apart by one
    value: an element of the difference, inside the container. No fixed list
    holds that for a schema nobody has written yet, so the shapes come from the
    schemas themselves -- a shape is filled only where some schema admits a
    value of it -- and the elements come from the seed. Adding a schema to the
    table above therefore adds the values that decide it.
    """
    values = [*_SEED, *_KIND_EDGES]
    carried = [value for value in values if _carried(value)]
    for witness, into in _SHAPES:
        if not any(_admits(schema, witness) for _, schema in schemas):
            continue  # no schema reads this shape, so a value of it decides nothing
        # An unhashable value is no key and no set member, so the shape skips it.
        values.extend(
            built for value in carried if (built := _built(into, value)) is not None
        )
    return values


# Suspected gaps accepted for now, each with why it is not decided. An entry is
# an admission, not a design: a gap described as a decision is what keeps it
# alive. Every one of these has a known route to being decided.
#
# Two families sit here, and they are different in kind. The class rows are the
# open world: a value universe is a closed one, so a relation that holds of every
# value here is not a relation that holds of every value, and the procedure is
# right to decline. The fixpoint row is a real incompleteness with a named route.
#
# `{a:int}` here is a `TypedDict`, which the typing spec makes **open**, so the
# two entries about a literal-keyed catch-all covering its field are gone: an
# open record admits a dict carrying a key the catch-all does not name, and a
# value refutes each relation rather than the procedure failing to decide it.
_OPEN_WORLD = (
    "The open world, not a gap in the rules: a subclass of this class may also "
    "derive from {kind}, so no value of this universe refutes the relation and "
    "no argument proves it. Deciding it would mean reading the class hierarchy "
    "as closed, which is the one assumption `docs/15-decidability.md` names and "
    "declines to make."
)

ACCEPTED: dict[str, str] = {
    "Plain <= ~int": _OPEN_WORLD.format(kind="`int`"),
    "Plain <= ~str": _OPEN_WORLD.format(kind="`str`"),
    "Adopts <= ~int": _OPEN_WORLD.format(kind="`int`"),
    "Adopts <= ~str": _OPEN_WORLD.format(kind="`str`"),
    "HasX <= ~int": _OPEN_WORLD.format(kind="`int`"),
    "HasX <= ~str": _OPEN_WORLD.format(kind="`str`"),
    "Plain <= Adopts": (
        "`_Adopts`'s metaclass answers `issubclass` by running code, so the "
        "class denotes whatever that code says at the moment it is asked and "
        "the oracle declines to read it as a set. The relation holds of every "
        "value here because the hook admits `_Plain`'s instances today. The "
        "route is a decision about the hook, not about the rules: reading a "
        "hooked class would make a relation's answer depend on user code that "
        "may raise, may be slow, and may answer differently twice."
    ),
    "list[str&Regex['a+']] <= mu X.list[X]|str&Regex['a*']": (
        "Every list of `a+` strings belongs to the fixpoint, through "
        "`list[a*] <= list[X] <= X`. The descriptor reads the pair through a "
        "polarity cut: the supertype's reference becomes the bottom under one, "
        "so the difference the reading holds is wider than the real one and its "
        "inhabitance proves nothing. Emptiness of the widened difference still "
        "proves the inclusion, which is why the answer is `undecided` rather "
        "than a refutation. The route is an unfolding of the supertype past one "
        "level before the cut, bounded by the reach the descriptor records."
    ),
}


def _admits(schema: Validator, value: Any) -> bool:
    """Report whether the schema admits the value, counting a refusal as no.

    A value the walk cannot reach at all -- a recursive container, an object
    whose attribute access raises -- is not a member for this survey, and
    swallowing that is deliberate: the probe is searching for a missing *rule*,
    and a value it cannot classify must not be read as a witness either way.
    """
    try:
        return bool(schema.is_valid(value))
    except Exception:  # noqa: BLE001 - see the docstring
        return False


#: Every value the survey is judged over.
VALUES: list[Any] = _universe(SCHEMAS)


def _members(schema: Validator) -> frozenset[int]:
    """Return the indices of every value the schema admits."""
    return frozenset(i for i, value in enumerate(VALUES) if _admits(schema, value))


def _survey() -> tuple[dict[str, str], list[str], int, int]:
    """Every ordered pair: (suspected gaps, unsound decisions, trues, falses)."""
    memberships = {name: _members(schema) for name, schema in SCHEMAS}
    gaps: dict[str, str] = {}
    unsound: list[str] = []
    decided_true = decided_false = 0

    for name_a, a in SCHEMAS:
        for name_b, b in SCHEMAS:
            if name_a == name_b:
                continue
            relation = f"{name_a} <= {name_b}"
            witness = memberships[name_a] - memberships[name_b]
            if a.is_subtype_of(b):
                decided_true += 1
                if witness:
                    example = repr(VALUES[min(witness)])
                    unsound.append(f"{relation} decided True, refuted by {example}")
            else:
                decided_false += 1
                if not witness:
                    gaps[relation] = ""
    return gaps, unsound, decided_true, decided_false


@pytest.fixture(scope="module")
def survey() -> tuple[dict[str, str], list[str], int, int]:
    return _survey()


def test_the_probe_actually_compared_something(survey) -> None:
    # A probe that decided nothing would pass every check below having compared
    # nothing at all. Both directions must be exercised for the search to mean
    # anything: only-true says the universe is trivial, only-false says the
    # procedure is.
    _, _, trues, falses = survey
    assert trues > 50, f"only {trues} relations decided True; the survey is degenerate"
    assert falses > 50, f"only {falses} relations decided False; nothing to search"
    assert len(VALUES) > 40, "the value universe is too thin to refute anything"
    assert len(SCHEMAS) > 20, "the schema universe covers too few kinds"


def test_a_refutation_names_a_value_outside(survey) -> None:
    """A reported refutation stands on a value, or it is a false claim.

    `is_subtype_of` folds "a value of this schema is outside the other" and "no
    rule here answers" into one `False`, and a caller reading it can tell them
    apart only by `relation_to`. That answer promises more than the `False`
    does: `"not_subset"` asserts the witness exists. So the universe below is
    asked for it -- a refutation over a pair whose members it can enumerate must
    name a value in the subject and outside the supertype.

    The direction the sibling checks cannot reach: they search for a `False`
    that no value refutes, which is a *gap* and is sound. This searches for a
    refutation no value supports, which is not.
    """
    memberships = {name: _members(schema) for name, schema in SCHEMAS}
    unsupported = []
    for name_a, a in SCHEMAS:
        for name_b, b in SCHEMAS:
            if name_a == name_b:
                continue
            if a.relation_to(b) != "not_subset":
                continue
            if not memberships[name_a] - memberships[name_b]:
                unsupported.append(f"{name_a} <= {name_b}")
    assert not unsupported, (
        "refutations no value in the universe supports: "
        f"{sorted(unsupported)[:8]}. A `not_subset` claims a value of the "
        "subject that the supertype rejects; where none exists the answer "
        "belongs on the conservative side as `undecided`."
    )


# THEORY: the-decision-has-three-answers
def test_the_three_answers_agree_with_the_two(survey) -> None:
    """`relation_to` and `is_subtype_of` answer the same question."""
    for name_a, a in SCHEMAS:
        for name_b, b in SCHEMAS:
            answer = a.relation_to(b)
            assert answer in {"subset", "not_subset", "undecided"}, answer
            assert a.is_subtype_of(b) == (answer == "subset"), (
                f"{name_a} <= {name_b}: {answer} against {a.is_subtype_of(b)}"
            )


def test_no_decided_relation_is_refuted_by_a_value(survey) -> None:
    # Unsoundness. Not a ledger matter -- this is the contract itself.
    _, unsound, _, _ = survey
    assert not unsound, "UNSOUND: " + "; ".join(unsound)


def test_every_suspected_gap_is_on_the_ledger(survey) -> None:
    gaps, _, _, _ = survey
    unlisted = sorted(set(gaps) - set(ACCEPTED))
    assert not unlisted, (
        f"relations answered False that no value refutes: {unlisted}. Each is a "
        "suspected completeness gap. Decide it, or add it to ACCEPTED with why "
        "it is not decided and the route to deciding it."
    )


def test_no_ledger_entry_is_stale(survey) -> None:
    gaps, _, _, _ = survey
    closed = sorted(set(ACCEPTED) - set(gaps))
    assert not closed, (
        f"ledger entries that are no longer suspected gaps: {closed}. Either the "
        "procedure decides them now -- remove the entry -- or a value refutes "
        "them, and the entry was never a gap."
    )


def test_every_ledger_entry_carries_a_reason() -> None:
    for relation, why in ACCEPTED.items():
        assert len(why) > 40, f"{relation}: an accepted gap with no reason"


def test_the_ledger_is_serialisable_for_a_report() -> None:
    # The set is small enough to read in a review; this keeps it that way and
    # keeps the keys plain strings so a diff on it is legible.
    assert len(ACCEPTED) <= 12, (
        f"{len(ACCEPTED)} accepted gaps; a list that grows quietly is how a "
        "procedure stops being the thing its docs describe."
    )
    json.dumps(ACCEPTED)


def _disagreements() -> list[str]:
    """Pairs where `a <= b` and `a & ~b is empty` give different answers.

    `docs/dev/02-decision.md` defines `a <= b` as "`a & ~b` admitting no value",
    and `docs/15-decidability.md` decides by `a ∧ ¬b = ∅`. Both procedures are
    sound, so a disagreement is one of them being *less complete* than the
    definition the pages give -- not an unsoundness, and not something a caller
    reading the identity would expect.
    """
    found = []
    for name_a, a in SCHEMAS:
        for name_b, b in SCHEMAS:
            if name_a == name_b:
                continue
            by_rule = a.is_subtype_of(b)
            by_emptiness = intersection(a, complement(b)).is_empty()
            if by_rule != by_emptiness:
                found.append(
                    f"{name_a} <= {name_b}: is_subtype_of={by_rule} "
                    f"(a & ~b).is_empty()={by_emptiness}"
                )
    return found


@pytest.fixture(scope="module")
def disagreements() -> list[str]:
    return _disagreements()


# THEORY: semantic-subtyping, an-exhaustible-procedure-is-searched
# THEORY: structural-rather-than-reduction
def test_the_two_deciders_are_measured_against_each_other(
    disagreements: list[str],
) -> None:
    """The count is recorded, so it can only shrink.

    Neither decider is wrong where they differ: `is_subtype_of` carries a
    coinductive rule for recursion that the emptiness route reaches by a single
    unfolding, so the rule side proves more about a fixpoint. What the number
    holds is that the gap does not *grow* -- a change that made either less
    complete would show up here rather than in a caller's undecided relation.
    """
    assert len(disagreements) <= 3, (
        f"{len(disagreements)} pairs where the two deciders disagree, up from "
        "the recorded 3:\n" + "\n".join(disagreements[:10])
    )
    # Every one of them is about a fixpoint, which is the whole of the known
    # gap: a rule that assumes its goal decides more about a recursive schema
    # than one unfolding of it does. A disagreement over anything else is a new
    # fact and fails the line above by arriving.
    assert all("mu t" in row for row in disagreements), disagreements


def test_neither_decider_is_unsound_where_they_disagree(
    disagreements: list[str], survey: tuple[dict[str, str], list[str], int, int]
) -> None:
    """A disagreement must be incompleteness, never a wrong `True`.

    Both routes claiming `a <= b` are checked against the value universe by
    `test_no_decided_relation_is_refuted_by_a_value`; this asks the same of the
    emptiness route, which that survey does not walk.
    """
    memberships = {name: _members(schema) for name, schema in SCHEMAS}
    refuted = []
    for name_a, a in SCHEMAS:
        for name_b, b in SCHEMAS:
            if name_a == name_b:
                continue
            if not intersection(a, complement(b)).is_empty():
                continue
            witness = memberships[name_a] - memberships[name_b]
            if witness:
                example = repr(VALUES[min(witness)])
                refuted.append(
                    f"{name_a} & ~{name_b} decided empty, refuted by {example}"
                )
    assert not refuted, refuted
    # And the survey ran, so an empty `refuted` is not an empty universe.
    assert survey[2] > 50
