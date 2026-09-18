"""Every ordered pair of schema variants is decided, or declined with a reason.

The relation suites in this tree each pick the pairs they are about. That is one
direction, and the direction a hand-written table satisfies: a pair nobody
thought of is a pair nobody asked, and the procedure's answer for it is whatever
it happens to be. The product of the node set with itself is the universe those
tables are samples of, and it is read here out of `ir.rs` rather than listed.

A pair is one of three things, and each has to be shown rather than asserted:

* a **proof** -- `relation_to` says `"subset"` -- and no value of the corpus is
  in the subject and outside the supertype, because such a value would refute it;
* a **refutation** -- `"not_subset"` -- and the corpus carries that value, so the
  answer rests on something the *walk* decides rather than on the rules agreeing
  with themselves;
* a **decline** -- `"undecided"` -- accepted only against a reason, and the
  reason says which of the two kinds it is.

The two kinds of decline are what the product is worth reading for. A decline
**no value decides** is the open world: an instance of a plain class may also be
an `int`, because a class deriving from both can be written, and no snapshot of
the class order refutes it. A decline **a value does decide** is incompleteness
with a name: the corpus holds a member of the subject that the supertype does not
admit, so the relation is refutable and the procedure did not refute it. Both are
sound. Only the second is a gap, and separating them is the whole point --
folded together, a conservative answer and a necessary one read alike, and the
number that ought to fall reads as the number that cannot.

A second list is held the same way at the end of the file, because it is the
one decline that is a **fact** rather than an incompleteness. A class whose
metaclass answers `__instancecheck__` or `__subclasscheck__` decides membership
itself: `register` is a call a caller makes after the class is written, and a
metaclass may answer differently on the next question. No snapshot of the class
order predicts either, so the oracle declines every question about such a class
-- and the hooks it reads before answering are read out of the binding, so a
third one arrives here without a class that takes it over. What the decline
costs is inclusion, and what it does not cost is membership: the walk asks
`isinstance` and gets the class's own answer, which each row holds against
`isinstance` itself rather than against a verdict written down beside it.

LEDGER: every ordered pair of schema variants is decided or declined with a reason

PRODUCT: every ordered pair of schema variants
"""

from __future__ import annotations

import abc
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Protocol, runtime_checkable

import annotated_types as at
import pytest

from valgebra import (
    Validator,
    anything,
    complement,
    intersection,
    nothing,
    recursive,
    union,
)

ROOT = Path(__file__).resolve().parent.parent
IR = ROOT / "crates" / "valgebra-core" / "src" / "ir.rs"
ORACLE = ROOT / "crates" / "valgebra-py" / "src" / "oracle.rs"

#: The variants of `pub enum Schema`, read from the tree rather than restated.
_ENUM = re.compile(r"^pub enum Schema \{$(.*?)^\}$", re.DOTALL | re.MULTILINE)
_VARIANT = re.compile(r"^    ([A-Z][A-Za-z]*)[ ({,]", re.MULTILINE)

#: The hooks `denotes_a_set` reads before the oracle will answer about a
#: class. Read out of the binding rather than restated: a third hook added
#: there arrives here without a row.
_HOOK = re.compile(r'intern!\(self\.py, "(__\w+check__)"\)')


class Plain:
    """A class deriving from no builtin, so it narrows no kind."""


@dataclass
class Point:
    """A class with a declared attribute, which is an `Instance ∧ AttrRecord`."""

    x: int


@runtime_checkable
class HasX(Protocol):
    """A protocol with a data member, which is what an `AttrRecord` alone is."""

    x: int


#: One representative per variant. Each is the form `tests/test_node_matrix.py`
#: names for that node, so the two tables describe one node set: a pair here is
#: a pair of the schemas that file already drives values through.
REPRESENTATIVES: dict[str, Any] = {
    "Anything": anything,
    "Nothing": nothing,
    "NoneType": None,
    "Bool": bool,
    "Int": int,
    "Float": float,
    "Str": str,
    "Bytes": bytes,
    "Literal": 1,
    "Instance": Plain,
    "Seq": list[int],
    "Coll": set[int],
    "KeyedMap": {"x": int},
    "AttrRecord": HasX,
    "Refine": Annotated[int, at.Ge(0)],
    "Union": union(int, str),
    "Complement": complement(int),
    # `intersection(int, complement(str))` is the form the node table names
    # and it is equivalent to `Int`: no integer is a string. A column equal
    # to another is a pair that answers `"subset"` both ways and reads as
    # covered, so the meet here is one with a member the other column lacks.
    "Intersection": intersection(int, complement(bool)),
    # A fixpoint's back edge is what `recursive` compiles to.
    "Ref": recursive(lambda t: union(None, {"next": t})),
}

#: The variant no compiled schema holds: a build-time marker the frontend
#: resolves to a `Ref` before a validator is returned, so there is nothing for a
#: representative to be. Held in its own direction below, so an excuse for a
#: variant the IR has dropped fails too.
NOT_IN_A_COMPILED_SCHEMA = {"SelfRef"}

#: The values every pair is decided against.
#:
#: A corpus rather than a sample: one member and one non-member of each kind the
#: representatives reach, plus the values at the edges between them -- a `bool`,
#: which is an `int`; a float no integer equals; a hashable value of an
#: unhashable kind; an instance carrying a declared attribute, and one carrying
#: none. A pair whose answer no value here touches is a pair this ledger reports
#: as undecided by the corpus as well, which is the honest reading of it.
CORPUS: list[Any] = [
    None,
    True,
    False,
    0,
    1,
    -1,
    2**63,
    1.5,
    0.0,
    float("nan"),
    "",
    "a",
    b"",
    b"x",
    [],
    [1],
    [1, "a"],
    (),
    (1,),
    set(),
    {1},
    frozenset({1}),
    {},
    {"x": 1},
    {"x": "a"},
    {"next": None},
    Plain(),
    Point(1),
    object(),
]

#: A protocol with a data member asks whether a value carries an attribute, and
#: that is a question about the object rather than about its kind: any instance
#: of any class may have one set on it, and which classes exist is not something
#: the core can enumerate. So the rules decline and the descriptor, which holds
#: a kind as a set of values, has nothing finer to say.
AN_ATTRIBUTE_IS_A_PYTHON_QUESTION = (
    "an attribute record asks whether a value carries a name, which is a "
    "question about the object rather than about its kind; the rules decline "
    "it and the descriptor holds no finer set -- see the attribute-record entry "
    "under `docs/15-decidability.md`'s conservative list"
)

#: `a ≤ b` is `a ∧ ¬b = ∅`, so the supertype of an inclusion is where a schema
#: appears *under a complement* -- and a reference under one is lowered to the
#: bottom, which is what keeps the difference sound. The difference then
#: contains the refuting value and is not proved to, so the refutation the
#: corpus has is not one the procedure can reach.
A_FIXPOINT_IS_LOWERED_ONCE = (
    "the supertype of an inclusion appears under a complement, and a reference "
    "there is lowered to a bound rather than to a set, so the difference holds "
    "the refuting value without being proved inhabited -- the unfolding entry "
    "under `docs/15-decidability.md`'s conservative list"
)

#: Two classes neither of which derives from the other may still share an
#: instance, through a class deriving from both -- which is a class that may be
#: written after the question is asked. The kind stands as the second class
#: here: an instance of a plain class may be an `int`, so it is not below the
#: complement of one.
THE_CLASS_ORDER_IS_OPEN = (
    "a class deriving from no builtin narrows no kind, and a class deriving "
    "from both it and the kind may be written after the question is asked, so "
    "neither direction is refutable -- `docs/15-decidability.md`'s open-world "
    "assumption"
)

#: The declines the corpus *does* decide: a member of the subject the supertype
#: refuses, found by the walk, for a relation the procedure left open. Each is
#: an incompleteness rather than a necessity, and the list may only shrink --
#: a rule that begins refuting one of these moves it out of the table and fails
#: here until somebody does.
DECLINED_THOUGH_A_VALUE_DECIDES: dict[tuple[str, str], str] = {
    ("Anything", "AttrRecord"): AN_ATTRIBUTE_IS_A_PYTHON_QUESTION,
    ("Anything", "Ref"): A_FIXPOINT_IS_LOWERED_ONCE,
    ("NoneType", "AttrRecord"): AN_ATTRIBUTE_IS_A_PYTHON_QUESTION,
    ("Bool", "AttrRecord"): AN_ATTRIBUTE_IS_A_PYTHON_QUESTION,
    ("Int", "AttrRecord"): AN_ATTRIBUTE_IS_A_PYTHON_QUESTION,
    ("Float", "AttrRecord"): AN_ATTRIBUTE_IS_A_PYTHON_QUESTION,
    ("Str", "AttrRecord"): AN_ATTRIBUTE_IS_A_PYTHON_QUESTION,
    ("Bytes", "AttrRecord"): AN_ATTRIBUTE_IS_A_PYTHON_QUESTION,
    ("Instance", "AttrRecord"): AN_ATTRIBUTE_IS_A_PYTHON_QUESTION,
    ("Instance", "Ref"): A_FIXPOINT_IS_LOWERED_ONCE,
    ("Seq", "AttrRecord"): AN_ATTRIBUTE_IS_A_PYTHON_QUESTION,
    ("Coll", "AttrRecord"): AN_ATTRIBUTE_IS_A_PYTHON_QUESTION,
    ("KeyedMap", "AttrRecord"): AN_ATTRIBUTE_IS_A_PYTHON_QUESTION,
    ("AttrRecord", "Nothing"): AN_ATTRIBUTE_IS_A_PYTHON_QUESTION,
    ("AttrRecord", "NoneType"): AN_ATTRIBUTE_IS_A_PYTHON_QUESTION,
    ("AttrRecord", "Bool"): AN_ATTRIBUTE_IS_A_PYTHON_QUESTION,
    ("AttrRecord", "Int"): AN_ATTRIBUTE_IS_A_PYTHON_QUESTION,
    ("AttrRecord", "Float"): AN_ATTRIBUTE_IS_A_PYTHON_QUESTION,
    ("AttrRecord", "Str"): AN_ATTRIBUTE_IS_A_PYTHON_QUESTION,
    ("AttrRecord", "Bytes"): AN_ATTRIBUTE_IS_A_PYTHON_QUESTION,
    ("AttrRecord", "Literal"): AN_ATTRIBUTE_IS_A_PYTHON_QUESTION,
    ("AttrRecord", "Instance"): AN_ATTRIBUTE_IS_A_PYTHON_QUESTION,
    ("AttrRecord", "Seq"): AN_ATTRIBUTE_IS_A_PYTHON_QUESTION,
    ("AttrRecord", "Coll"): AN_ATTRIBUTE_IS_A_PYTHON_QUESTION,
    ("AttrRecord", "KeyedMap"): AN_ATTRIBUTE_IS_A_PYTHON_QUESTION,
    ("AttrRecord", "Refine"): AN_ATTRIBUTE_IS_A_PYTHON_QUESTION,
    ("AttrRecord", "Union"): AN_ATTRIBUTE_IS_A_PYTHON_QUESTION,
    ("AttrRecord", "Intersection"): AN_ATTRIBUTE_IS_A_PYTHON_QUESTION,
    ("AttrRecord", "Ref"): AN_ATTRIBUTE_IS_A_PYTHON_QUESTION,
    ("Refine", "AttrRecord"): AN_ATTRIBUTE_IS_A_PYTHON_QUESTION,
    ("Union", "AttrRecord"): AN_ATTRIBUTE_IS_A_PYTHON_QUESTION,
    ("Complement", "AttrRecord"): AN_ATTRIBUTE_IS_A_PYTHON_QUESTION,
    ("Complement", "Ref"): A_FIXPOINT_IS_LOWERED_ONCE,
    ("Intersection", "AttrRecord"): AN_ATTRIBUTE_IS_A_PYTHON_QUESTION,
    ("Intersection", "Ref"): A_FIXPOINT_IS_LOWERED_ONCE,
    ("Ref", "AttrRecord"): AN_ATTRIBUTE_IS_A_PYTHON_QUESTION,
}


#: The declines nothing decides: the open world, where the answer is not a gap
#: in the procedure but a fact about what a class order can be asked. A pair
#: here that a value later refutes is a corpus that grew, not a rule that did.
DECLINED_AND_NOTHING_DECIDES: dict[tuple[str, str], str] = {
    ("Instance", "Complement"): THE_CLASS_ORDER_IS_OPEN,
    ("AttrRecord", "Complement"): THE_CLASS_ORDER_IS_OPEN,
}


def _variants() -> set[str]:
    """Read the `Schema` variant names out of the IR rather than restating them."""
    body = _ENUM.search(IR.read_text(encoding="utf-8"))
    assert body, "ir.rs has no `pub enum Schema`"
    found = set(_VARIANT.findall(body.group(1)))
    # The scan is the detector: an empty variant set would pass both directions
    # having read nothing at all.
    assert len(found) >= 19, f"the variant scan found only {sorted(found)}"
    return found


@pytest.fixture(scope="module")
def built() -> dict[str, Validator]:
    """Each representative, compiled once for the whole product."""
    return {name: Validator(spec) for name, spec in REPRESENTATIVES.items()}


def _pairs() -> list[tuple[str, str]]:
    # Spelled as a comprehension rather than `itertools.product(x, repeat=2)`,
    # whose element type is a variadic tuple: the pair is what every row below
    # unpacks, and a checker that cannot see two of them cannot see a row that
    # unpacks three.
    return [(a, b) for a in REPRESENTATIVES for b in REPRESENTATIVES]


def _witnesses(built: dict[str, Validator], subject: str, supertype: str) -> list[Any]:
    """Give the corpus values in `subject` and outside `supertype`.

    A list rather than the first value found, because `None` is in the corpus on
    purpose: returning it as the witness and returning it as "nothing found"
    would be one answer, and every pair whose only witness is `None` would read
    as a pair with none.
    """
    return [
        value
        for value in CORPUS
        if built[subject].is_valid(value) and not built[supertype].is_valid(value)
    ]


def test_the_universe_is_the_ir_variants_in_both_directions() -> None:
    """A node added to the IR arrives here without a representative, and fails."""
    variants = _variants()
    missing = sorted(variants - set(REPRESENTATIVES) - NOT_IN_A_COMPILED_SCHEMA)
    assert not missing, (
        f"schema variants with no representative: {missing}. Give each one the "
        "form `tests/test_node_matrix.py` names for it, or excuse it with the "
        "reason no compiled schema holds one."
    )
    stale = sorted(set(REPRESENTATIVES) - variants)
    assert not stale, f"representatives naming no IR variant: {stale}"
    # The excuse expires in its own direction.
    assert variants >= NOT_IN_A_COMPILED_SCHEMA


def test_the_representatives_are_that_many_different_sets(
    built: dict[str, Validator],
) -> None:
    """The product is over distinct schemas, so a pair is a pair of two things.

    A table that named one set twice would answer `"subset"` for that pair and
    read as covered. Equivalence is the check rather than equality of the specs,
    because two spellings of one set are what the algebra is for.
    """
    same = [
        (a, b)
        for a, b in _pairs()
        if a < b and built[a].is_equivalent(REPRESENTATIVES[b])
    ]
    assert not same, f"representatives denoting one set: {same}"


def test_every_pair_is_proved_refuted_or_declined_with_a_reason(
    built: dict[str, Validator],
) -> None:
    """The product has no cell nobody asked, which is what a sample leaves."""
    unexplained = [
        (a, b)
        for a, b in _pairs()
        if built[a].relation_to(REPRESENTATIVES[b]) == "undecided"
        and (a, b) not in DECLINED_THOUGH_A_VALUE_DECIDES
        and (a, b) not in DECLINED_AND_NOTHING_DECIDES
    ]
    assert not unexplained, (
        "pairs the procedure declines with no reason recorded:\n"
        + "\n".join(f"  {a} <= {b}" for a, b in unexplained)
        + "\n\nRecord each as a decline a value decides, or as one nothing "
        "does, with the reason it is that kind."
    )


def test_a_refutation_carries_a_value_the_walk_checks(
    built: dict[str, Validator],
) -> None:
    """`not_subset` claims a value exists, so the corpus is made to produce it.

    The walk is a different procedure from the relations -- a different module,
    no shared code -- so it is an oracle they can be wrong against. A refutation
    no value supports is the relations agreeing with themselves.
    """
    empty = [
        (a, b)
        for a, b in _pairs()
        if built[a].relation_to(REPRESENTATIVES[b]) == "not_subset"
        and not _witnesses(built, a, b)
    ]
    assert not empty, (
        "pairs refuted with no value in the corpus to refute them:\n"
        + "\n".join(f"  {a} </= {b}" for a, b in empty)
    )


def test_a_proof_is_not_refuted_by_the_corpus(built: dict[str, Validator]) -> None:
    """A `subset` a value contradicts is a wrong answer, however it was reached."""
    refuted = [
        (a, b, found)
        for a, b in _pairs()
        if built[a].relation_to(REPRESENTATIVES[b]) == "subset"
        and (found := _witnesses(built, a, b))
    ]
    assert not refuted, "inclusions the walk refutes:\n" + "\n".join(
        f"  {a} <= {b}, but {found[0]!r} is in the first" for a, b, found in refuted
    )


def test_a_decline_is_recorded_as_the_kind_it_is(built: dict[str, Validator]) -> None:
    """The two tables are held apart, which is what makes the first countable.

    A decline listed as one nothing decides, against which the corpus produces a
    value, is an incompleteness filed as a necessity -- the reading that makes a
    gap unfixable by making it look like a fact.
    """
    misfiled = [
        (a, b) for a, b in DECLINED_AND_NOTHING_DECIDES if _witnesses(built, a, b)
    ]
    assert not misfiled, (
        f"declines filed as necessary that a corpus value refutes: {misfiled}"
    )
    undecided_by_the_corpus = [
        (a, b)
        for a, b in DECLINED_THOUGH_A_VALUE_DECIDES
        if not _witnesses(built, a, b)
    ]
    assert not undecided_by_the_corpus, (
        "declines filed as refutable that no corpus value refutes: "
        f"{undecided_by_the_corpus}"
    )


def test_no_reason_outlives_the_pair_it_excuses(
    built: dict[str, Validator],
) -> None:
    """A pair the procedure decides keeps no reason for declining it."""
    reasoned = {**DECLINED_THOUGH_A_VALUE_DECIDES, **DECLINED_AND_NOTHING_DECIDES}
    decided = sorted(
        pair
        for pair in reasoned
        if built[pair[0]].relation_to(REPRESENTATIVES[pair[1]]) != "undecided"
    )
    assert not decided, (
        f"reasons for pairs the procedure decides after all: {decided}. A rule "
        "that begins deciding one drops its row here, in the same commit."
    )
    unknown = sorted(pair for pair in reasoned if set(pair) - set(REPRESENTATIVES))
    assert not unknown, f"reasons for pairs that are not pairs of variants: {unknown}"
    for pair, reason in reasoned.items():
        assert len(reason) > 60, f"{pair}: {reason!r}"


def test_the_corpus_reaches_every_representative(
    built: dict[str, Validator],
) -> None:
    """The oracle is shown to say something about each node, in both directions.

    A corpus that admitted nothing of a kind would make every refutation about
    that kind unwitnessed and every proof unrefuted, which reads exactly like a
    node the procedure handles perfectly.
    """
    for name, compiled in built.items():
        members = [value for value in CORPUS if compiled.is_valid(value)]
        outside = [value for value in CORPUS if not compiled.is_valid(value)]
        if name != "Nothing":
            assert members, f"no corpus value is a member of {name}"
        if name != "Anything":
            assert outside, f"every corpus value is a member of {name}"


@dataclass(frozen=True)
class Hooked:
    """A class whose metaclass answers membership, and a value on each side.

    Both sides, for the reason every other row here carries both: a schema
    admitting everything agrees with `isinstance` on the member alone, and one
    admitting nothing agrees on the outsider alone.
    """

    spec: Any
    member: Any
    outsider: Any
    hook: str


class _Discriminating(type):
    """A metaclass that answers both hooks itself, for the integers."""

    def __instancecheck__(cls, other: object) -> bool:
        return isinstance(other, int)

    def __subclasscheck__(cls, other: type) -> bool:
        return issubclass(other, int)


class _Hooked(metaclass=_Discriminating):
    """A class whose membership its metaclass decides."""


class _Registered(abc.ABC):  # noqa: B024 - the point is the registration
    """An abstract base a kind is registered against after the fact."""


_Registered.register(int)


@runtime_checkable
class _Runs(Protocol):
    """A protocol with a method and no data member, which `isinstance` answers."""

    def run(self) -> None: ...


class _Runner:
    """A value the protocol above admits, by having the method."""

    def run(self) -> None:
        """Do nothing; the protocol asks only that the name is there."""


#: The ways a class takes over one of the hooks, one row each.
#:
#: The oracle declines every question about such a class, and that is not a gap
#: it could close: `register` is a call a caller makes after the class is
#: written, and a metaclass may answer differently on the next question it is
#: asked. A snapshot of the class order predicts neither. What the decline costs
#: is *inclusion*, and what it does not cost is membership -- the walk asks
#: `isinstance` and gets the class's own answer -- so each row holds both.
HOOKED: dict[str, Hooked] = {
    "a metaclass of its own": Hooked(_Hooked, 1, "a", "__instancecheck__"),
    "an abstract base with a registration": Hooked(
        _Registered, 1, "a", "__subclasscheck__"
    ),
    "a runtime-checkable protocol": Hooked(_Runs, _Runner(), 1, "__instancecheck__"),
}


def _hooks() -> set[str]:
    """Read the hooks the binding checks before it will answer about a class."""
    text = ORACLE.read_text(encoding="utf-8")
    start = text.index("fn denotes_a_set")
    found = set(_HOOK.findall(text[start : text.index("\n    }", start)]))
    # The parse is the detector, so it is shown to have read something.
    assert len(found) >= 2, sorted(found)
    return found


def test_every_hook_the_binding_reads_has_a_row() -> None:
    """A third hook added to the guard arrives here with no class that takes it."""
    hooks = _hooks()
    covered = {row.hook for row in HOOKED.values()}
    missing = sorted(hooks - covered)
    assert not missing, (
        f"hooks the binding reads that no row drives: {missing}. Add a class "
        "that takes the hook over, with the value each way."
    )
    stale = sorted(covered - hooks)
    assert not stale, f"rows naming a hook the binding does not read: {stale}"


@pytest.mark.parametrize(("shape", "row"), sorted(HOOKED.items()), ids=sorted(HOOKED))
def test_a_class_that_answers_membership_itself_is_declined(
    shape: str, row: Hooked
) -> None:
    """The relation declines both ways, rather than answering from the order.

    This is the one decline of the product that is a *fact* rather than an
    incompleteness, and it has to be driven rather than reasoned about: an
    answer here would be the procedure predicting a call the caller has not
    made yet.
    """
    compiled = Validator(row.spec)
    assert compiled.relation_to(int) == "undecided", shape
    assert Validator(int).relation_to(row.spec) == "undecided", shape


@pytest.mark.parametrize(("shape", "row"), sorted(HOOKED.items()), ids=sorted(HOOKED))
def test_the_walk_answers_where_the_relation_declines(shape: str, row: Hooked) -> None:
    """A declined inclusion costs nothing on the value question.

    The walk asks `isinstance`, so it gets the class's own answer and is exact
    where the relation has nothing to say. Held against `isinstance` itself
    rather than against a recorded verdict, because the class is the authority
    and a recorded answer would be a second one.
    """
    compiled = Validator(row.spec)
    for value in (row.member, row.outsider):
        assert compiled.is_valid(value) == isinstance(value, row.spec), (
            f"{shape}: the walk and the class disagree about {value!r}"
        )
    assert compiled.is_valid(row.member), shape
    assert not compiled.is_valid(row.outsider), shape
