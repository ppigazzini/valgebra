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
from typing import Annotated, Any, Literal, TypedDict

import annotated_types as at
import pytest

from valgebra import Regex, Validator, complement, intersection, union


class _Rec(TypedDict):
    a: int


class _Rec2(TypedDict):
    a: int
    b: str


def _v(annotation: Any) -> Validator:
    return annotation if isinstance(annotation, Validator) else Validator(annotation)


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
    ("int&Ge(0)", _v(Annotated[int, at.Ge(0)])),
    ("int&Ge(1)", _v(Annotated[int, at.Ge(1)])),
    ("str&Regex['a']", _v(Annotated[str, Regex("a")])),
    ("str&Regex['ab?']", _v(Annotated[str, Regex("ab?")])),
]


class _Obj:
    a = 1


# The witnesses. A thin universe turns a decided-false relation into a reported
# gap, so each addition here is a false report removed: the non-string key is
# what separates an open record (whose catch-all admits any key) from a
# `dict[str, ...]`, and without it the two look equal.
VALUES: list[Any] = [
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

# Suspected gaps accepted for now, each with why it is not decided. An entry is
# an admission, not a design: a gap described as a decision is what keeps it
# alive. Every one of these has a known route to being decided.
ACCEPTED: dict[str, str] = {
    "{a:int} <= dict[Lit['a'],int]": (
        "Whether a supertype catch-all covers a field name is asked by matching "
        "the key against the `Str`/`Anything` atoms rather than by asking whether "
        "the name belongs to the key's set. A literal-keyed clause therefore never "
        "covers a field, though it names exactly that key. The field name is a "
        "bare `String` in the core, so deciding it needs an oracle method that "
        "compares a pooled constant to a name."
    ),
    "{a:int} <= dict[Lit['a','b'],int]": "As above, with the key a union of literals.",
    "str&Regex['a'] <= str&Regex['ab?']": (
        "A regex is opaque to `constraint_entailed`, which gives it no value "
        "entailment, so a refinement relates through one only when the supertype "
        "carries it verbatim. Inclusion of one regular language in another is "
        "decidable, so the route is a language comparison over the two pooled "
        "patterns rather than equality of them."
    ),
    "str&Regex['a'] <= Lit['a']": (
        "The same opacity read the other way: deciding it means computing that "
        "the pattern's language is the singleton the literal denotes, and the "
        "pattern is never turned into a language at all."
    ),
    "str&Regex['a'] <= Lit['a','b']": (
        "As above, with the supertype a union of literals."
    ),
    "bool <= int&Ge(0)": (
        "A subtype that is not itself a refinement reaches one only through the "
        "value oracle, which answers for a literal and not for a class. Deciding "
        "it means comparing the bound against the subtype's own value range -- "
        "`bool` denotes exactly `{False, True}` -- where the core reads a scalar "
        "region rather than an enumerated set."
    ),
}


def _admits(schema: Validator, value: Any) -> bool:
    """Report whether the schema admits the value, counting a refusal as no.

    A value the walk cannot reach at all -- a recursive container, an object
    whose attribute access raises -- is simply not a member for this survey, and
    swallowing that is deliberate: the probe is searching for a missing *rule*,
    and a value it cannot classify must not be read as a witness either way.
    """
    try:
        return bool(schema.is_valid(value))
    except Exception:  # noqa: BLE001 - see the docstring
        return False


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
            # The gradual atom is deliberately never decided as a supertype: it
            # admits everything at runtime and is a distinct atom to the algebra,
            # so `X <= Any` staying conservative is the documented policy rather
            # than a gap. It stays in the universe on the *subtype* side.
            if name_b == "Any":
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
