"""Every entry of the published boundary is an entry a test drives.

`docs/15-decidability.md` is the page a caller reads to learn what the relations
decide. It is three lists -- decided exactly, sound but conservative,
undecidable at runtime -- and until this file nothing read it: the completeness
ledger enumerates relations without citing a bullet, the relation ledger and the
completeness probe quote the page in prose inside their reasons, and the page
itself was maintained by hand against a procedure that moves.

Both halves of that are failures, and they are different. A bullet on the
decided list that the procedure stopped deciding is a **published promise the
tree breaks**. A bullet on the conservative list that the procedure started
deciding is a page telling a caller to expect less than it gets, which is the
gentler one and is how the list grows stale until nobody trusts it. A relation
that is on neither list is the third: neither promised nor declined, so a
caller has no way to read an `undecided` as kept or broken -- and that is the
state a record against a union of records sat in.

So the universe is the **page**, read bullet by bullet, and each bullet carries
a query this file drives:

* a decided bullet answers `"subset"`, `"not_subset"` or `True` from
  `is_empty()` -- an answer the procedure commits to;
* a conservative bullet answers `"undecided"`, which is the decline the page
  promises, and the row says which value would decide it if one could;
* an undecidable bullet is refused when the validator is built, or is read as
  an opaque atom, which the row states.

Both directions: a bullet with no row fails, and a row naming no bullet fails.
A conservative row that begins deciding fails too, which is what makes closing
a gap a change to the page rather than a quiet improvement.

LEDGER: every entry of the published decidability boundary is driven by a test

PRODUCT: every entry of the published decidability boundary
"""

from __future__ import annotations

import enum
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Final,
    Literal,
    Protocol,
    TypeVar,
    runtime_checkable,
)

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
PAGE = ROOT / "docs" / "15-decidability.md"

#: The three lists, by the heading each sits under.
DECIDED = "Decided exactly"
CONSERVATIVE = "Sound but conservative"
UNDECIDABLE = "Undecidable at runtime"
_SECTIONS = (DECIDED, CONSERVATIVE, UNDECIDABLE)

#: A bullet's lead-in, which is how the page names an entry.
_BULLET = re.compile(r"^- \*\*(.+?)\*\*", re.MULTILINE)


@dataclass(frozen=True)
class Decides:
    """A pair the page promises an answer for, and the answer.

    `answer` is what `relation_to` reports, or `"empty"` for a bullet the page
    states as an emptiness rather than an inclusion.
    """

    left: Any
    right: Any
    answer: str


@dataclass(frozen=True)
class Declines:
    """A pair the page promises no answer for, with what would decide it.

    `would_decide` is the route the page names -- the representation that would
    have to reach it -- and it is prose, because a decline is a statement about
    what has not been built. What is driven is the `"undecided"`.

    `beside` is the other half of such an entry. A conservative entry usually
    says which neighbouring question *is* decided: the direction the carrier
    proves, the inclusion two steps settle between them, the length a word's
    automaton counts. That sentence is a promise of the same page, and driving
    only the decline leaves it stated and unheld -- so the entry carries the
    pair and the answer the page gives it, and both run.
    """

    left: Any
    right: Any
    would_decide: str
    beside: tuple[Any, Any, str] | None = None


@dataclass(frozen=True)
class Refuses:
    """A form the page says is rejected rather than decided."""

    build: Any
    says: str


@dataclass(frozen=True)
class Opaque:
    """A form the page says is read as an atom rather than taken apart.

    Driven by a value: the schema builds, and it admits what the atom admits
    rather than what the form it is written as would suggest.
    """

    spec: Any
    member: Any
    outsider: Any


Row = Decides | Declines | Refuses | Opaque

_T = TypeVar("_T")


@runtime_checkable
class _HasX(Protocol):
    """A protocol with a data member, which is an attribute record alone."""

    x: int


class _Colour(enum.Enum):
    """An enumeration whose members compare by identity, which is the default."""

    RED = 1
    GREEN = 2


def _nested(depth: int, inner: Any) -> Any:
    """Wrap `inner` in that many records, which is what nesting costs.

    The bounds on a build are three, and nesting is the one that grows the work
    exponentially -- so the shape the allowance is sized against is a record
    minus the union of its siblings, one level at a time.
    """
    for _ in range(depth):
        inner = {"a": inner}
    return inner


_JSON = recursive(lambda j: union(None, bool, int, float, str, [j], {str: j}))
_CHAIN = recursive(lambda t: union(None, {"next": t}))


#: Every bullet the page carries, with the query that drives it.
#:
#: The key is the bullet's lead-in, verbatim: a bullet reworded is a bullet
#: whose row has to be re-read, which is the point rather than a cost.
ROWS: dict[str, Row] = {
    # -- decided exactly -----------------------------------------------------
    "The scalar Boolean algebra.": Decides(
        intersection(int, complement(int)), nothing, "subset"
    ),
    "Complement and disjointness across kinds.": Decides(
        intersection(list[int], set[int]), nothing, "subset"
    ),
    "A bare container class and its parameterised form.": Decides(
        list, list[object], "subset"
    ),
    "Class and literal inclusion.": Decides(_Colour.RED, _Colour, "subset"),
    "Literals against other kinds.": Decides(Literal["a"], complement(int), "subset"),
    "An enumeration against the union of its members": Decides(
        _Colour, Literal[_Colour.RED, _Colour.GREEN], "subset"
    ),
    "Divisibility between two moduli": Decides(
        Annotated[int, at.MultipleOf(5000)],
        Annotated[int, at.MultipleOf(2500)],
        "subset",
    ),
    "Refinements.": Decides(
        Annotated[int, at.Ge(1)], Annotated[int, at.Ge(0)], "subset"
    ),
    "Sequences.": Decides(
        tuple[int | str, int], union(tuple[int, int], tuple[str, int]), "subset"
    ),
    "Sets and frozensets.": Decides(set[bool], set[int], "subset"),
    "Records and mappings.": Decides({"x": int}, {str: int}, "subset"),
    "A difference written as one complemented union.": Decides(
        intersection(
            {"a": union(int, str), "b": union(int, str)},
            complement(
                union(
                    {"a": int, "b": int},
                    {"a": int, "b": str},
                    {"a": str, "b": int},
                    {"a": str, "b": str},
                )
            ),
        ),
        nothing,
        "subset",
    ),
    "A length bound over a base that takes any length.": Decides(
        Annotated[str, at.MinLen(1)], complement(Literal[""]), "subset"
    ),
    "Two schemas that share no value.": Decides(
        list[int], tuple[int, int], "not_subset"
    ),
    "A subject outside a base.": Decides(Annotated[int, at.Ge(0)], str, "not_subset"),
    "Inclusion in a complement.": Decides(int, complement(str), "subset"),
    "Recursion.": Decides(_JSON, union(bytes, _JSON), "subset"),
    "The complement laws, where the constructors reach them.": Decides(
        union(int, complement(int)), anything, "subset"
    ),
    # -- sound but conservative ----------------------------------------------
    "A clause keyed by a complement.": Declines(
        {complement(str): int},
        dict[int, int],
        "a part of the key partition for a key-type spanning every kind but one",
    ),
    "Recursion, past one unfolding.": Declines(
        complement(int),
        _CHAIN,
        "an unfolding of the supertype past one level before the reference is cut",
    ),
    "An attribute record, on either side.": Declines(
        int,
        complement(_HasX),
        "an oracle that enumerates the values of a kind, which the open world denies",
    ),
    "A length bound over a set or a dict, in the sets.": Declines(
        Annotated[set[int], at.MinLen(1)],
        Annotated[set[int], at.MinLen(2)],
        "a length in the powerset lattice, which holds members rather than a count",
        # The same bound over a sequence is a shape the automaton holds, so a
        # length above the positions there are is decided empty.
        beside=(Annotated[tuple[int, int], at.MinLen(3)], nothing, "subset"),
    ),
    "An integer bound outside the 64-bit range.": Declines(
        Annotated[int, at.Ge(2**70)],
        Annotated[int, at.Ge(2**70 + 1)],
        "a carrier wider than the 64-bit intervals the integer component holds",
        # The direction the carrier proves: an inclusion between two bounds is
        # settled by comparing the bounds, so the size of either is beside the
        # point and only the refutation wants an interval.
        beside=(
            Annotated[int, at.Ge(2**70 + 1)],
            Annotated[int, at.Ge(2**70)],
            "subset",
        ),
    ),
    "A meet of two moduli the representation cannot hold.": Declines(
        intersection(
            Annotated[int, at.MultipleOf(64)], Annotated[int, at.MultipleOf(81)]
        ),
        Annotated[int, at.MultipleOf(5184)],
        "a residue representation whose period is the one two steps share, "
        "rather than one materialised up to a recorded bound",
        # The refutation the residues do reach, which is the one the entry
        # contrasts the declining pair with.
        beside=(
            Annotated[int, at.MultipleOf(2)],
            Annotated[int, at.MultipleOf(4)],
            "not_subset",
        ),
    ),
    "A schema too large to build.": Declines(
        _nested(3, {"x": union(int, str, bytes, float)}),
        union(*[_nested(3, {"x": kind}) for kind in (int, str, bytes, float)]),
        "a larger allowance, which bounds the cost of asking and not the schema",
    ),
    "A predicate.": Declines(
        int,
        Annotated[int, at.Predicate(lambda value: value > 0)],
        "deciding whether a Python callable holds of every value, which is undecidable",
    ),
    # -- undecidable at runtime ----------------------------------------------
    "Erased generics and type variables.": Refuses(lambda: Validator(_T), "TypeVar"),
    "Abstract-collection generics.": Refuses(
        lambda: Validator(Sequence[int]), "Sequence"
    ),
    "Callable signatures.": Opaque(Callable[[int], str], str, 1),
    "Predicates.": Opaque(Annotated[int, at.Predicate(lambda value: value > 0)], 1, 0),
    "Typing qualifiers.": Refuses(lambda: Validator(Final[int]), "Final"),
}


def _bullets() -> dict[str, str]:
    """Every bullet the page carries, with the list it is on."""
    text = PAGE.read_text(encoding="utf-8")
    found: dict[str, str] = {}
    for index, name in enumerate(_SECTIONS):
        start = text.index(f"## {name}")
        after = [
            text.index(f"## {other}")
            for other in (*_SECTIONS[index + 1 :], "The contract")
            if text.find(f"## {other}") > start
        ]
        body = text[start : min(after)] if after else text[start:]
        for title in _BULLET.findall(body):
            found[title] = name
    # The scan is the detector: a page this read nothing from would pass both
    # directions having checked nothing at all.
    assert len(found) >= 25, sorted(found)
    return found


def test_the_universe_is_the_page_in_both_directions() -> None:
    """A bullet added to the page arrives here without a row, and fails."""
    bullets = _bullets()
    missing = sorted(set(bullets) - set(ROWS))
    assert not missing, (
        f"entries of the published boundary with no row: {missing}. Give each "
        "one the query that drives it: an answer for a decided entry, an "
        "`undecided` for a conservative one, a refusal or an atom for an "
        "undecidable one."
    )
    stale = sorted(set(ROWS) - set(bullets))
    assert not stale, f"rows naming no entry of the page: {stale}"


def test_every_row_is_the_kind_its_list_calls_for() -> None:
    """A decided entry carries an answer and a conservative one a decline.

    The lists are the claim, so the *kind* of each row is held to the list its
    bullet sits on. A `Declines` under "decided exactly" would drive an
    `undecided` and report the page kept, which is the shape that lets a
    promise rot without failing.
    """
    expected: dict[str, tuple[type, ...]] = {
        DECIDED: (Decides,),
        CONSERVATIVE: (Declines,),
        UNDECIDABLE: (Refuses, Opaque),
    }
    wrong = [
        f"{title!r} is on {section!r} and carries {type(ROWS[title]).__name__}"
        for title, section in _bullets().items()
        if not isinstance(ROWS[title], expected[section])
    ]
    assert not wrong, "\n".join(wrong)


@pytest.mark.parametrize("title", sorted(ROWS))
def test_every_entry_of_the_boundary_answers_as_the_page_says(title: str) -> None:
    """The row, driven."""
    row = ROWS[title]
    if isinstance(row, Decides):
        left = Validator(row.left)
        if row.answer == "empty":
            assert left.is_empty(), f"{title}: the page promises an empty set"
            return
        assert left.relation_to(row.right) == row.answer, (
            f"{title}: the page puts this on the decided list"
        )
        return
    if isinstance(row, Declines):
        answer = Validator(row.left).relation_to(row.right)
        assert answer == "undecided", (
            f"{title}: the page puts this on the conservative list and the "
            f"procedure answers {answer!r}. Deciding it is a change to the "
            "page, not a quiet improvement."
        )
        assert len(row.would_decide) > 30, title
        if row.beside is not None:
            left, right, decided = row.beside
            beside = Validator(left).relation_to(right)
            assert beside == decided, (
                f"{title}: the page says the neighbouring relation is "
                f"{decided!r} and the procedure answers {beside!r}. A "
                "conservative entry naming a decided question is held to both."
            )
        return
    if isinstance(row, Refuses):
        with pytest.raises((NotImplementedError, TypeError, ValueError)) as caught:
            row.build()
        assert row.says in str(caught.value), (
            f"{title}: the refusal does not name what it is about"
        )
        return
    compiled = Validator(row.spec)
    assert compiled.is_valid(row.member), f"{title}: the atom refuses its own value"
    assert not compiled.is_valid(row.outsider), (
        f"{title}: the atom admits a value of another kind"
    )


# --- The trust base, held to the tests that show what breaking it costs ------
#
# `docs/14-soundness.md` ends with what the soundness argument *assumes*, and
# an assumption is the one kind of claim a passing suite says nothing about: it
# holds wherever the suite looks, because the suite looks at values that satisfy
# it. What a reader needs is the other half -- a value that violates it and what
# the library then answers.
#
# So each assumption names a test through a `# TRUST:` marker carrying the
# sentence verbatim. The strict expected failure is the model: it drives an
# `int` subclass whose comparisons lie, pins the wrong answer that follows, and
# fails the day the answer changes.
#
# The universe is the page's list, which means the *whole* of it. A scan keyed
# on the bolded lead-in reads a bullet written without one as no assumption at
# all, so the formatting decides what is held rather than the page -- and a
# sentence the argument rests on drops out of the ledger by being typed
# differently. Both halves are checked here: every top-level bullet leads with
# the sentence it assumes, and the count of lead-ins is the count of bullets.

SOUNDNESS = ROOT / "docs" / "14-soundness.md"

#: An assumption, which the page writes as a bolded lead-in under its last
#: heading. The sub-points inside one are qualifications of it rather than
#: assumptions of their own, so the scan takes the top level alone.
_ASSUMPTION = re.compile(r"^- \*\*(.+?)\*\*", re.MULTILINE)

#: A top-level bullet of the same list, whatever it leads with. Beside
#: `_ASSUMPTION`, which reads the ones written to be found: this reads the list
#: itself, so the two counts can be held equal.
_TOP_LEVEL = re.compile(r"^- (.+)$", re.MULTILINE)

#: The marker a test carries to name the assumption it shows the cost of.
_TRUST = re.compile(r"^#\s*TRUST:\s*(.+)$", re.MULTILINE)


def _trust_base() -> str:
    """Give the page's list of what the soundness argument takes on trust."""
    text = SOUNDNESS.read_text(encoding="utf-8")
    return text[text.index("## What this argument assumes") :]


def _trust_bullets() -> list[str]:
    """Every top-level bullet of the trust base, lead-in and all."""
    found = _TOP_LEVEL.findall(_trust_base())
    assert len(found) >= 6, found
    return found


def _assumed() -> set[str]:
    """Every assumption the soundness page's trust base states."""
    found = set(_ASSUMPTION.findall(_trust_base()))
    assert len(found) == len(_trust_bullets()), sorted(found)
    return found


def test_every_assumption_leads_with_the_sentence_it_assumes() -> None:
    """A bullet with no bolded lead-in is a claim outside this ledger.

    The scan above reads the trust base by its formatting, so a bullet typed
    without a lead-in is not a smaller assumption: it is one the marker check
    never asks about, and the count of assumptions becomes a count of how the
    page is written. The lead-in is also what a `# TRUST:` marker carries
    verbatim, so a bullet with none has nothing a test could name.
    """
    plain = [
        " ".join(bullet.split())[:90]
        for bullet in _trust_bullets()
        if not bullet.startswith("**")
    ]
    assert not plain, (
        "assumptions written without the bolded sentence they assume:\n"
        + "\n".join(f"  {bullet}" for bullet in plain)
        + "\n\nLead each bullet with `**<the sentence>**`, which is what a "
        "`# TRUST:` marker names."
    )


def _shown() -> set[str]:
    """Every assumption a test names."""
    found: set[str] = set()
    for path in sorted((ROOT / "tests").rglob("test_*.py")):
        found.update(
            line.strip() for line in _TRUST.findall(path.read_text(encoding="utf-8"))
        )
    return found


def test_every_assumption_names_a_test_that_shows_its_cost() -> None:
    """An assumption with no test is a sentence the suite cannot be wrong about."""
    unshown = sorted(_assumed() - _shown())
    assert not unshown, (
        f"assumptions of the soundness argument no test names: {unshown}. Put "
        "`# TRUST: <the sentence>` above the test that drives a value "
        "violating it, and shows what the library answers."
    )


def test_every_trust_marker_names_an_assumption_the_page_states() -> None:
    """The other direction: a marker left behind by a rewritten sentence."""
    unknown = sorted(_shown() - _assumed())
    assert not unknown, (
        f"markers naming no assumption of the trust base: {unknown}. The "
        "sentence is carried verbatim, so a rewrite is read again rather than "
        "passing on the old wording."
    )
