"""Every result, obligation and deviation names the test that holds it.

`docs/dev/10-theory.md` says what the design rests on: "A citation here is a
claim that a specific line exists because of it, not a reading list." A claim
tagged `[LOAD-BEARING]` says code in this tree would be wrong without the result;
one tagged `[OBLIGATION]` says the theory demands a shape of the implementation;
one tagged `[DEVIATION]` says the tree departs from its source on purpose and
names the cost. Each is a sentence a reader can act on, and a sentence nothing
checks is one the tree can stop satisfying without anyone noticing.

This is the ledger that ends that. Each tagged paragraph carries a `HELD-BY:`
line naming tests, or an `OWED:` line naming the tests it is owed and why they
do not exist yet, and this file holds the directions apart:

* a claim with neither line fails, so a result cannot be added to the page
  without saying what would catch it going wrong, or naming the test that would;
* an owed name that already resolves to a test fails, so a debt cannot outlive
  the test that pays it;
* a name that resolves to no test fails, so a test renamed or deleted takes the
  claim's evidence with it rather than leaving a page asserting something
  nothing checks;
* a name that resolves to *several* tests fails unless it says which, because a
  name two files define is one that survives either being deleted -- so the
  line carries the file as well, and each entry addresses one test;
* the count of `OWED:` lines is recorded and may only shrink, so the debt the
  page admits is a number the tree holds rather than a sentence a reader skims.

What it cannot do is read a test and judge whether it holds the claim. That is
the reviewer's, and the `HELD-BY:` line is where the reviewer's answer is
written down instead of being re-derived.

LEDGER: every theory result, obligation and deviation names a test, held or owed

PRODUCT: every result, obligation and deviation the design rests on
"""

from __future__ import annotations

import functools
import re
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import pytest

if TYPE_CHECKING:
    from collections.abc import Sequence

pytestmark = pytest.mark.repository

ROOT = Path(__file__).resolve().parent.parent
THEORY = ROOT / "docs" / "dev" / "10-theory.md"

#: The tags a *claim* carries, each with the id that names it. The page's own
#: paragraph explaining them writes the words without brackets, which is what
#: keeps it out of the universe.
CLAIMS = ("LOAD-BEARING", "OBLIGATION", "DEVIATION", "NOT-REACHED")

#: A `NOT-REACHED` tag is a claim like the other three, and the claim is a
#: negative one: this tree does not do the thing, and something fails on the day
#: it starts. That is a real test and a rare one -- the closure ledger is the
#: model, since a variant added to the algebra fails it -- so the tag is given
#: only where such a test exists. A result the tree merely does not implement,
#: with nothing to notice if it did, stays `GUIDING`: pretending otherwise would
#: put a holding line beside every citation and make the lines mean less.
#:
#: The tags that carry *context* rather than a claim: a result that shapes a
#: decision without being an algorithm here, and one on the path and unbuilt.
#:
#: These take an id and no holding line. A guiding result is not a sentence the
#: tree can be false against -- that is what separates it from a load-bearing
#: one -- so asking it to name a test would put a test beside every citation
#: and make the holding lines mean less, not more. What the id buys is that the
#: page is uniformly addressable: a test *may* name one, the reverse direction
#: reads it, and a tag cannot be added without one.
CONTEXT = ("GUIDING", "PLANNED")

#: A tag as the page writes it: the kind, and the id a test names it by.
_TAG = re.compile(
    r"\*\*\[(" + "|".join((*CLAIMS, *CONTEXT)) + r"): ([a-z][a-z0-9-]*)\]\*\*"
)

#: The marker a *test* carries to name the claim it holds, in either language.
#: Read within this many lines above the definition, so it may sit above a doc
#: comment, an attribute or a decorator rather than between them and the name.
_MARKER = re.compile(r"(?://|#)\s*THEORY:\s*([a-z0-9-]+(?:\s*,\s*[a-z0-9-]+)*)")
_BESIDE = 30
HELD_BY = "HELD-BY:"
OWED = "OWED:"

#: An `OWED:` line names the tests the claim is owed -- the names a landing
#: commit moves to a `HELD-BY:` line -- and says why they do not exist yet.
_OWED_FORM = re.compile(
    r"^OWED: [a-z][a-z0-9_]*(?:, [a-z][a-z0-9_]*)* -- \S.{24,}$", re.DOTALL
)

#: How many claims the page admits are held by nothing. Recorded so the number
#: can only fall: a claim added as owed rather than held is a decision this
#: file makes visible, and one that lands its test lowers the figure here in
#: the same commit.
RECORDED_OWED = 0

#: Where a named test may live. Rust tests are `fn name(`, Python `def name(`.
_SOURCES = (
    (ROOT / "crates", "*.rs", re.compile(r"\bfn\s+(\w+)\s*[(<]")),
    (ROOT / "tests", "*.py", re.compile(r"^def\s+(\w+)\s*\(", re.MULTILINE)),
)


#: A `HELD-BY:` entry: a test name, optionally led by where the test lives.
#: The qualifier is a suffix of the defining file's path
#: (`descr/sets/tests.rs::the_lattice_laws_hold_of_the_sets`), which is enough
#: to pick one definition out and short enough to read on a page.
_ENTRY = re.compile(r"^(?:(\S+)::)?(\w+)$")


def _split(entry: str) -> tuple[str | None, str]:
    """Give where a `HELD-BY:` entry says its test lives, and what it is called."""
    found = _ENTRY.match(entry)
    assert found, f"a HELD-BY entry that is no test name: {entry!r}"
    return found.group(1), found.group(2)


def _name_of(entry: str) -> str:
    """Give the test an entry names, with any qualifier dropped."""
    return _split(entry)[1]


class Claim(NamedTuple):
    """One tagged paragraph and what the page says holds it."""

    text: str
    tag: str
    identifier: str
    names: list[str]
    owed: str | None


@functools.cache
def _paragraphs() -> tuple[str, ...]:
    """Give the tracked page as the paragraphs the ledger reads it in."""
    return tuple(THEORY.read_text(encoding="utf-8").split("\n\n"))


@functools.cache
def _claims() -> tuple[Claim, ...]:
    """Each tagged claim, with the tests it says hold it, or the ones it is owed.

    A `HELD-BY:` or `OWED:` belongs to the claim it follows, with the qualifying
    prose a claim usually carries allowed to sit between them: a citation is a
    paragraph and its caveats, and forcing the line to butt against the tag
    would put it in the middle of a sentence.
    """
    found: list[Claim] = []
    for paragraph in _paragraphs():
        if paragraph.startswith(HELD_BY):
            names = [
                name.strip()
                for name in paragraph[len(HELD_BY) :].replace("\n", " ").split(",")
                if name.strip()
            ]
            assert found, f"a HELD-BY line before any claim: {paragraph!r}"
            last = found[-1]
            assert not last.names, (
                f"two holding lines for one claim: {_summarise(last.text)}"
            )
            assert last.owed is None, (
                f"two holding lines for one claim: {_summarise(last.text)}"
            )
            found[-1] = last._replace(names=names)
            continue
        if paragraph.startswith(OWED):
            assert found, f"an OWED line before any claim: {paragraph!r}"
            last = found[-1]
            assert not last.names, (
                f"two holding lines for one claim: {_summarise(last.text)}"
            )
            assert last.owed is None, (
                f"two holding lines for one claim: {_summarise(last.text)}"
            )
            found[-1] = last._replace(owed=" ".join(paragraph.split()))
            continue
        tags = _TAG.findall(paragraph)
        if tags:
            assert len(tags) == 1, (
                f"a paragraph carrying two tags: {_summarise(paragraph)}"
            )
            kind, identifier = tags[0]
            found.append(Claim(paragraph, kind, identifier, [], None))
            continue
        bare = [tag for tag in (*CLAIMS, *CONTEXT) if f"[{tag}]" in paragraph]
        assert not bare, (
            "a claim tagged without an id, which no test can name: "
            + _summarise(paragraph)
        )
    return tuple(found)


def _held() -> list[Claim]:
    return [claim for claim in _claims() if claim.names]


def _must_be_held(claims: Sequence[Claim]) -> list[Claim]:
    """Give the claims a holding line is asked of, which is not the context."""
    return [claim for claim in claims if claim.tag in CLAIMS]


@functools.cache
def _defined_names() -> set[str]:
    """Every function name the tree defines, in either language."""
    names: set[str] = set()
    for root, glob, pattern in _SOURCES:
        for path in root.rglob(glob):
            names.update(pattern.findall(path.read_text(encoding="utf-8")))
    return names


def _relative(path: Path) -> str:
    """Give a source file's path as this ledger spells one, everywhere.

    Posix, because these strings are *compared*: a `HELD-BY:` qualifier is
    written with `/` on the page, and `str(Path)` gives a backslash on Windows,
    so two readers spelling a path differently agree on no file there while
    agreeing on every file here. One helper rather than a convention, because a
    convention is what the two readers already had.
    """
    return path.relative_to(ROOT).as_posix()


@functools.cache
def _definitions() -> dict[str, set[str]]:
    """Give, for every function name the tree defines, the files defining it.

    Beside `_defined_names`, which answers whether a name exists at all, and
    `_homes_of`, which answers where a *set* of names lives between them. This
    answers the question one entry asks: which files define this one name, and
    therefore whether naming it addresses a test or a namesake.
    """
    found: dict[str, set[str]] = {}
    for root, glob, pattern in _SOURCES:
        for path in root.rglob(glob):
            where = _relative(path)
            for name in pattern.findall(path.read_text(encoding="utf-8")):
                found.setdefault(name, set()).add(where)
    return found


@functools.cache
def _resolve(entry: str) -> tuple[str, frozenset[str]]:
    """Give the name an entry addresses and the files its qualifier leaves.

    An unqualified entry leaves every file defining the name; a qualified one
    leaves those whose path ends with the qualifier. The tests below read the
    count: one file is an address, several are a namesake, none is a stale
    qualifier.
    """
    where, name = _split(entry)
    homes = frozenset(_definitions().get(name, set()))
    if where is None:
        return name, homes
    return name, frozenset(home for home in homes if home.endswith(where))


def _homes_of(names: set[str]) -> set[str]:
    """Give the files that define each of `names`, relative to the root.

    Beside `_defined_names`, which answers whether a name exists at all. This
    answers *where*, which is what says the ledger reaches the tree rather than
    one module of it.
    """
    homes: set[str] = set()
    for root, glob, pattern in _SOURCES:
        for path in root.rglob(glob):
            found = set(pattern.findall(path.read_text(encoding="utf-8")))
            if found & names:
                homes.add(_relative(path))
    return homes


def _summarise(paragraph: str) -> str:
    """Give the first sentence of a claim, for a failure a reader can place."""
    return " ".join(paragraph.split())[:110]


def test_the_page_carries_tagged_claims() -> None:
    """The parse is a detector, so it must be shown to have read something.

    A ledger over an empty universe passes having checked nothing, which is the
    failure mode every other ledger here guards the same way. Every tag the
    page writes is a claim the parse found, in each of the three kinds.
    """
    claims = _claims()
    text = THEORY.read_text(encoding="utf-8")
    assert len(_TAG.findall(text)) == len(claims)
    by_tag = {
        tag: sum(1 for claim in claims if claim.tag == tag)
        for tag in (*CLAIMS, *CONTEXT)
    }
    assert by_tag["LOAD-BEARING"] >= 8, by_tag
    assert by_tag["OBLIGATION"] >= 4, by_tag
    assert by_tag["DEVIATION"] >= 4, by_tag
    assert by_tag["GUIDING"] >= 4, by_tag
    assert by_tag["NOT-REACHED"] >= 1, by_tag

    # And each id is its own, since a test names a claim by it.
    identifiers = [claim.identifier for claim in claims]
    repeated = sorted({name for name in identifiers if identifiers.count(name) > 1})
    assert not repeated, f"claims sharing an id: {repeated}"


def test_every_claim_names_a_test_held_or_owed() -> None:
    """A result the design rests on says what would catch it going wrong.

    Or says, in a form this file can read, that nothing does yet and which
    test would. A claim with neither is a sentence.
    """
    unheld = [
        _summarise(claim.text)
        for claim in _must_be_held(_claims())
        if not claim.names and claim.owed is None
    ]
    assert not unheld, "claims with neither a HELD-BY nor an OWED line:\n" + "\n".join(
        unheld
    )


def test_an_owed_claim_names_the_test_and_the_reason() -> None:
    """An `OWED:` line is a debt with an address, not a shrug.

    The address is a test name, so the line reads like the `HELD-BY:` it becomes
    when the test lands, and a milestone code -- which the docs lint refuses on
    a tracked page, for dangling on every reader but its author -- is not one.
    """
    malformed = [
        f"{_summarise(claim.text)}\n  {claim.owed}"
        for claim in _claims()
        if claim.owed is not None and not _OWED_FORM.match(claim.owed)
    ]
    assert not malformed, (
        "OWED lines must read "
        "`OWED: <test_name>[, ...] -- <why it does not exist yet>`:\n"
        + "\n".join(malformed)
    )


def test_what_is_owed_only_shrinks() -> None:
    """The debt is a recorded number, so a claim cannot be added as owed quietly.

    A claim that lands its test lowers `RECORDED_OWED` in the same commit; a
    claim added with an `OWED:` line has to raise it, in a diff a reviewer sees.
    """
    owed = [_summarise(claim.text) for claim in _claims() if claim.owed is not None]
    assert len(owed) <= RECORDED_OWED, (
        f"{len(owed)} claims are owed, up from the recorded {RECORDED_OWED}:\n"
        + "\n".join(owed)
    )
    assert len(owed) == RECORDED_OWED, (
        f"{len(owed)} claims are owed; set RECORDED_OWED to {len(owed)}"
    )


def _owed_names() -> list[tuple[str, str]]:
    return [
        (claim.text, name.strip())
        for claim in _claims()
        if claim.owed is not None
        for name in claim.owed[len(OWED) :].split(" -- ")[0].split(",")
    ]


def test_an_owed_test_does_not_already_exist() -> None:
    """A debt paid is a `HELD-BY:` line, not an `OWED:` one left standing."""
    defined = _defined_names()
    paid = [
        f"{_summarise(text)}\n  {name}"
        for text, name in _owed_names()
        if name in defined
    ]
    assert not paid, (
        "owed tests that already exist; move each to a HELD-BY line:\n"
        + "\n".join(paid)
    )


@pytest.mark.parametrize(
    ("claim", "names"),
    [(c.text, c.names) for c in _held()],
    ids=[_summarise(c.text)[:40] for c in _held()],
)
def test_every_named_test_exists(claim: str, names: list[str]) -> None:
    """A name that resolves to no test is a claim with no evidence left."""
    defined = _defined_names()
    missing = [name for name in names if _name_of(name) not in defined]
    assert not missing, f"{_summarise(claim)}\nnames no test: {missing}"


def test_every_named_test_addresses_one_definition() -> None:
    """A name several files define is a name that outlives its own test.

    The guarantee above is that deleting a test takes the claim's evidence
    with it. A name two modules define does not carry it: one of them holds
    the claim, the other happens to be called the same thing, and the line
    resolves to whichever the scan reaches. Delete the holder and the page
    goes on naming a test that exists.

    So an ambiguous name says which file, and a qualified one is held to
    resolving: a qualifier matching no definition is as stale as a name
    matching none. The property names are the shape this catches -- one law,
    written over the schemas, over the descriptor and over a single component,
    is three tests with one name and three different subjects.
    """
    ambiguous: list[str] = []
    for claim in _held():
        for entry in claim.names:
            _, homes = _resolve(entry)
            if len(homes) == 1:
                continue
            where = ", ".join(sorted(homes)) if homes else "nothing"
            ambiguous.append(f"{_summarise(claim.text)[:60]}\n    {entry} -> {where}")
    assert not ambiguous, (
        "HELD-BY entries that address more than one test, or none:\n"
        + "\n".join(f"  {row}" for row in ambiguous)
        + "\n\nWrite the file before the name, as "
        "`descr/sets/tests.rs::the_lattice_laws_hold_of_the_sets`."
    )


def test_the_names_are_test_functions_rather_than_helpers() -> None:
    """A `HELD-BY:` naming a helper would resolve and hold nothing.

    Every name must be a Rust `#[test]`/`proptest!` body or a pytest function,
    which is what makes resolving it evidence rather than a spelling check.
    """
    python_tests: set[str] = set()
    for path in (ROOT / "tests").rglob("*.py"):
        python_tests.update(
            re.findall(
                r"^def\s+(test_\w+)\s*\(",
                path.read_text(encoding="utf-8"),
                re.MULTILINE,
            )
        )
    rust_tests: set[str] = set()
    for path in (ROOT / "crates").rglob("*.rs"):
        text = path.read_text(encoding="utf-8")
        # A `#[test]` attribute, or a body inside a `proptest!` block: both are
        # run by `cargo test` and neither is a helper the tests call.
        rust_tests.update(re.findall(r"#\[test\]\s*\n\s*fn\s+(\w+)", text))
        for block in re.findall(r"proptest!\s*\{(.*?)\n\}", text, re.DOTALL):
            rust_tests.update(re.findall(r"\bfn\s+(\w+)\s*\(", block))
    runnable = python_tests | rust_tests
    assert len(runnable) > 400, f"the test parse found only {len(runnable)}"
    for claim in _held():
        for entry in claim.names:
            name = _name_of(entry)
            assert name in runnable, f"{_summarise(claim.text)}\n{name!r} is not a test"


def test_a_claim_is_held_by_more_than_its_own_restatement() -> None:
    """A name must not simply repeat the claim's own words.

    The failure this guards is a `HELD-BY:` written to satisfy the ledger: a
    test whose name is the claim, asserting the claim. Nothing mechanical can
    tell that apart in general, and what it can tell is that the page's claims
    do not each rest on a single test of their own -- so a reviewer reading one
    finds a suite rather than a mirror.
    """
    counts = [len(claim.names) for claim in _held()]
    assert min(counts) >= 2, "every held claim names at least two tests"
    all_names = {_name_of(entry) for claim in _held() for entry in claim.names}
    assert len(all_names) >= 15, sorted(all_names)

    # And the names are spread across the tree rather than gathered in one
    # module: a ledger whose tests all sit in one file is a statement about
    # that file, and these claims are about the whole of what the tree decides.
    # Both languages, because the theory is implemented in both -- a page held
    # only by the Python side would say nothing about the core, where the
    # decision procedure and the set representation live.
    homes = _homes_of(all_names)
    assert len(homes) >= 15, sorted(homes)
    for language in (".rs", ".py"):
        written_in = sorted(home for home in homes if home.endswith(language))
        assert len(written_in) >= 5, (
            f"the held tests reach only {len(written_in)} {language} file(s): "
            f"{written_in}"
        )


@functools.cache
def _markers() -> dict[str, list[tuple[str, str]]]:
    """Every `THEORY:` marker in the tree: the id, the file, and the test below it.

    Read by walking down from the marker to the next definition, which is the
    direction the placement rule states: the marker sits above whatever preamble
    the item carries -- a doc comment, an attribute, a decorator -- so what it
    names is the first definition after it and never one before.
    """
    found: dict[str, list[tuple[str, str]]] = {}
    for root, glob, pattern in _SOURCES:
        for path in root.rglob(glob):
            text = path.read_text(encoding="utf-8")
            for marker in _MARKER.finditer(text):
                below = pattern.search(text, marker.end())
                name = below.group(1) if below else ""
                for identifier in marker.group(1).replace(" ", "").split(","):
                    found.setdefault(identifier, []).append((_relative(path), name))
    return found


def test_the_readers_spell_a_path_the_same_way() -> None:
    """A marker's file and a definition's file are compared, so they are one string.

    `test_every_held_test_carries_the_marker` asks whether the marker naming a
    claim sits in the file the entry resolves to, which is a comparison of two
    paths produced by two readers. Spelled `str(Path)` on one side and
    posix on the other, they agree on every file on a posix machine and on no
    file on Windows -- so the check passes locally and reports every held name
    as unmarked in the lane that runs there.

    Held as a subset rather than by looking for a separator, because the defect
    is not "a backslash appeared": it is two readers naming one file two ways,
    and that is what a Windows run turns into a failure.
    """
    marked = {path for sites in _markers().values() for path, _ in sites}
    known = {home for homes in _definitions().values() for home in homes}
    assert marked, "the marker scan read nothing"
    stray = sorted(marked - known)
    assert not stray, (
        f"marker files no definition reader names: {stray}. Both sides spell a "
        "path through `_relative`, so a comparison between them is a "
        "comparison of one string."
    )


def test_every_marker_names_a_claim_the_page_carries() -> None:
    """The reverse direction: a test that says what it holds is held to saying so.

    A `HELD-BY:` line goes stale when its test is renamed, and
    `test_every_named_test_exists` catches that. This catches the other end -- a
    marker naming a claim the page has dropped or renamed, which would otherwise
    point a reader at nothing and read as if it pointed somewhere.
    """
    markers = _markers()
    # The scan is a detector, so it is shown to have read the tree.
    assert len(markers) >= 20, sorted(markers)
    identifiers = {claim.identifier for claim in _claims()}
    unknown = sorted(set(markers) - identifiers)
    assert not unknown, "markers naming a claim the page does not carry:\n" + "\n".join(
        f"  {identifier} -- in {', '.join(path for path, _ in markers[identifier])}"
        for identifier in unknown
    )


def test_every_held_test_carries_the_marker() -> None:
    """And a test the page names says, at the test, which claim it is for.

    Without it a reader at the test has no way back to the sentence it holds,
    and a test edited until it no longer asserts the claim looks like any other
    edit. The marker does not make that impossible -- nothing mechanical can --
    but it puts the claim in front of whoever is editing.
    """
    markers = _markers()
    marked: set[tuple[str, str, str]] = {
        (identifier, path, name)
        for identifier, sites in markers.items()
        for path, name in sites
    }
    unmarked = sorted(
        f"{entry} ({claim.identifier})"
        for claim in _held()
        for entry in claim.names
        if not any(
            (claim.identifier, home, _name_of(entry)) in marked
            for home in _resolve(entry)[1]
        )
    )
    assert not unmarked, (
        "tests the page says hold a claim and that do not name it:\n"
        + "\n".join(f"  {row}" for row in unmarked)
        + "\n\nPut `# THEORY: <id>` (or `// THEORY: <id>`) above the test."
    )


def test_every_marked_test_is_one_its_claim_names() -> None:
    """A test that says which claim it holds is a test that claim names.

    The marker above is the page reaching the test. This is the test reaching
    the page, and it is the direction that was missing: a test written for a
    claim, carrying the claim's own id, and left off the holding line reads --
    from the page, which is where a reader looks -- as evidence nobody has.
    Sixteen tests were in that state at once, the seven strongest among them
    written the week the claims they hold were tightened, so the page pointed
    at the weaker tests beside them while the ledger stayed green.

    A marker naming a claim the page does not carry fails here too, which is
    the same line gone stale from the other end.
    """
    claims = {claim.identifier for claim in _claims()}
    named = {
        (claim.identifier, _name_of(entry))
        for claim in _held()
        for entry in claim.names
    }
    adrift = sorted(
        f"{name} ({path}) marks {identifier!r}, which "
        + (
            "no claim on this page carries"
            if identifier not in claims
            else "does not name it"
        )
        for identifier, sites in _markers().items()
        for path, name in sites
        if (identifier, name) not in named
    )
    assert not adrift, (
        "tests that name a claim and are not named by it:\n"
        + "\n".join(f"  {row}" for row in adrift)
        + "\n\nAdd the test to that claim's `HELD-BY:` line, or drop the "
        "marker if the test does not hold the claim."
    )


#: The argument the tracked page restates. Not redistributed, and both halves of
#: the path are spelled from their pieces: the internal area and the note's own
#: name each dangle for every reader but the author, and `docs_lint` refuses
#: either written out -- here as much as on a page.
ARGUMENT = ROOT / ("__" + "DEV") / ("5-" + "THEORY.md")

SOURCE = "SOURCE:"

#: One entry of a `SOURCE:` line: the section of the argument, and a fragment of
#: the sentence this page restates. Entries are separated by `;`, because a
#: fragment carries commas and a claim may restate two paragraphs -- Nakano's
#: modality is argued in two places and lands here as one sentence.
_SOURCE_ENTRY = re.compile(r'§(\d+(?:\.\d+[a-z]?)?) "([^"]{12,})"')

#: Where the argument's results are tagged, and where its obligations are whole
#: numbered sections rather than tagged paragraphs.
#:
#: Three universes, read at every heading level the argument uses. The
#: bibliography (§1--§12, `## N.`) tags a paragraph `[LOAD-BEARING]` where code
#: rests on it; the results ledger (§13.x) tags a paragraph `[USED]` or
#: `[DEVIATION]`; the obligations (§14.x) are whole sections. Each is a claim
#: the tree can be false against, so each must be restated here by a
#: paragraph that carries a holding line -- a `GUIDING` restatement of a
#: `[USED]` result is a citation, not a claim, and does not count.
_RESULT = re.compile(r"\*\*\[(?:USED|DEVIATION)")
_FOUNDATION = re.compile(r"`\[LOAD-BEARING\]`")
_SECTION = re.compile(r"^#{1,2} (\d+(?:\.\d+[a-z]?)?)\.? ", re.MULTILINE)
_RESULTS_FROM = "13."
_OBLIGATIONS_FROM = "14."
_FOUNDATIONS_BELOW = 13


def _argument() -> dict[str, str]:
    """Read the argument's numbered sections, or `{}` where it is absent."""
    if not ARGUMENT.exists():
        return {}
    text = ARGUMENT.read_text(encoding="utf-8")
    cuts = [(match.group(1), match.start()) for match in _SECTION.finditer(text)]
    return {
        number: text[start : cuts[index + 1][1] if index + 1 < len(cuts) else len(text)]
        for index, (number, start) in enumerate(cuts)
    }


def _sources() -> list[tuple[str, str, str]]:
    """Every `(claim id, section, fragment)` the page cites, read off the page."""
    cited: list[tuple[str, str, str]] = []
    claims = _claims()
    by_position = {claim.text: claim for claim in claims}
    last: Claim | None = None
    for paragraph in _paragraphs():
        if _TAG.search(paragraph):
            last = by_position.get(paragraph)
            continue
        if not paragraph.startswith(SOURCE):
            continue
        assert last is not None, f"a SOURCE line before any claim: {paragraph!r}"
        entries = _SOURCE_ENTRY.findall(paragraph.replace("\n", " "))
        assert entries, (
            f"a SOURCE line this ledger cannot read: {paragraph!r}. The form is "
            f'`SOURCE: §13.3 "a fragment of the sentence"`, entries separated by `;`.'
        )
        cited += [(last.identifier, section, fragment) for section, fragment in entries]
    return cited


def test_every_source_line_is_one_this_ledger_can_read() -> None:
    """The half of the citation a clone can check, which is where it runs.

    The argument is not redistributed, so the two checks below stand down
    everywhere but the author's box -- every runner included. A `SOURCE:` line
    nobody can read would then be a line nobody reads: malformed on the page,
    skipped in the lane, and caught only by whoever next opened the notes.

    So the form is held here, where the page is all it takes: the line belongs
    to a claim, and it names a section and quotes a sentence.
    """
    cited = _sources()
    assert cited, "no claim on the page says where it comes from"
    unknown = sorted(
        {identifier for identifier, _, _ in cited}
        - {claim.identifier for claim in _claims()}
    )
    assert not unknown, f"SOURCE lines attributed to no claim: {unknown}"


def test_every_cited_source_is_a_section_the_argument_has() -> None:
    """A claim that names where it comes from names somewhere that exists.

    The page restates an argument kept out of the distribution, so a reader
    holding both has no way to find the paragraph a sentence came from and a
    reader holding one has no way to know a citation went stale. The fragment
    is quoted rather than summarised for the same reason a citation carries a
    page number: a section is a screenful, and the sentence is the claim.
    """
    sections = _argument()
    if not sections:
        pytest.skip("the argument is not redistributed, so a clone has none to read")
    cited = _sources()
    missing = sorted(
        f"{identifier} cites §{section}, which the argument does not have"
        for identifier, section, _ in cited
        if section not in sections
    )
    stale = sorted(
        f"{identifier} quotes {fragment!r}, which is not in §{section}"
        for identifier, section, fragment in cited
        if section in sections and fragment not in sections[section]
    )
    assert not missing + stale, "\n".join(missing + stale)


#: A row of the argument's deviation table: its number, the cost it states,
#: and the id it is restated under -- or `--` where the row is closed.
_DEVIATION_ROW = re.compile(
    r"^\| (\d+) \| .+? \| (.+) \| [^|]+ \| `([a-z0-9-]+|--)` \|$", re.MULTILINE
)

#: What a row quotes from the paragraph restating it. A fragment rather than
#: the whole cost, for the reason a `SOURCE:` entry is one: the row is a
#: summary and the paragraph is the argument, so what the two can be held equal
#: on is a sentence they share. Twelve characters, because a shorter quote
#: matches somewhere by accident.
_RESTATED = re.compile(r'"([^"]{12,})"')


class Row(NamedTuple):
    """One row of the argument's deviation table."""

    number: str
    cost: str
    identifier: str


@functools.cache
def _deviation_rows() -> tuple[Row, ...]:
    """Give every row of the argument's deviation table, with the id it names.

    Empty only where the argument is absent, which is every clone but the
    author's. A table that is *there* and parses to nothing is the row pattern
    having gone stale, and that reads exactly like a clone unless it is caught
    here -- so the two are separated at the parse rather than at each caller.
    """
    if not ARGUMENT.exists():
        return ()
    text = ARGUMENT.read_text(encoding="utf-8")
    heading = text.find("# 15. The deviations")
    if heading == -1:
        return ()
    found = tuple(Row(*row) for row in _DEVIATION_ROW.findall(text[heading:]))
    assert found, "the deviation table is present and no row parses"
    return found


def _flatten(text: str) -> str:
    """Give a paragraph as one line, so a quote spanning a wrap still matches."""
    return " ".join(text.split())


def test_every_deviation_the_argument_tables_is_restated_here() -> None:
    """The departures are a numbered table, and the table has an address column.

    A deviation is the kind of claim that goes stale quietly: it is not a rule
    the code enforces, it is a place the code *departs* from its source, so
    nothing fails when one is closed and nothing fails when one is added. The
    table and this page were kept in step by hand, and the two differed by four
    rows when that was last read.

    So each live row names the tagged paragraph restating it. A row with no id
    fails, an id naming no tag fails, and a row whose id resolves to a tag of
    another kind fails. A closed row carries `--`. The other direction is
    deliberately loose: this page may carry deviations the table does not,
    because a departure found in the code is a departure whether or not the
    argument reached it first.
    """
    rows = _deviation_rows()
    if not rows:
        pytest.skip("the argument is not redistributed, so a clone has none to read")
    assert len(rows) >= 9, f"the deviation table reads as {rows}"
    deviations = {claim.identifier for claim in _claims() if claim.tag == "DEVIATION"}
    unrestated = sorted(
        f"row {row.number} names {row.identifier!r}, which is no deviation on this page"
        for row in rows
        if row.identifier != "--" and row.identifier not in deviations
    )
    assert not unrestated, "\n".join(unrestated)
    live = [row.number for row in rows if row.identifier != "--"]
    assert len(live) >= 8, f"only {len(live)} live rows, which reads as a table gone"


def test_every_deviation_row_quotes_the_paragraph_restating_it() -> None:
    """A row and its restatement cannot say different things about one limit.

    The row above ties a row to a *paragraph*. What it cannot tie is the row's
    **cost** to what that paragraph says the cost is, and those are two pieces
    of prose kept in step by hand -- the arrangement that lets a limit be
    written in one and corrected in the other. A table reading "a relation
    declines where the carrier cannot spell the bound" beside a page reading
    "decided in the direction the carrier proves" is two answers to one
    question, and a reader has no way to tell which is the tree's.

    Prose cannot be compared to prose, so the row quotes the paragraph instead:
    a fragment of the restatement, verbatim, the way a `SOURCE:` entry quotes
    the argument. Rewriting either side without the other fails. That is the
    whole of what a mechanical check reaches here, and it is the half that goes
    wrong -- a correction lands on one page and the other keeps the claim.
    """
    rows = _deviation_rows()
    if not rows:
        pytest.skip("the argument is not redistributed, so a clone has none to read")
    restatements = {
        claim.identifier: _flatten(claim.text)
        for claim in _claims()
        if claim.tag == "DEVIATION"
    }
    adrift: list[str] = []
    for row in rows:
        if row.identifier == "--":
            continue
        paragraph = restatements[row.identifier]
        quoted = _RESTATED.findall(row.cost)
        if not quoted:
            adrift.append(
                f"row {row.number} ({row.identifier}) quotes nothing of the "
                "paragraph restating it"
            )
            continue
        adrift += [
            f"row {row.number} ({row.identifier}) quotes {fragment!r}, which "
            "the paragraph restating it does not say"
            for fragment in quoted
            if _flatten(fragment) not in paragraph
        ]
    assert not adrift, (
        "deviation rows adrift from the page that restates them:\n"
        + "\n".join(f"  {row}" for row in adrift)
        + "\n\nQuote a fragment of the restating paragraph in the cost "
        "column, so a rewrite of either side has to reach both."
    )


def test_every_result_the_argument_carries_is_restated_here() -> None:
    """The direction that keeps the argument from being the ledger.

    A result added to the argument and not to this page is a claim the tree
    rests on with no tagged paragraph, no `HELD-BY:` line and no test -- which
    is the state every one of them started in, and the state this page exists
    to end. The argument keeps the reasoning; what a reader acts on is here.

    Its results are tagged paragraphs and its obligations are whole numbered
    sections, so the universe is read both ways rather than by one pattern.
    """
    sections = _argument()
    if not sections:
        pytest.skip("the argument is not redistributed, so a clone has none to read")
    tags = {claim.identifier: claim.tag for claim in _claims()}
    # What each citation is worth: a `SOURCE:` on a claim that carries a
    # holding line restates a result; one on a guiding or planned paragraph
    # cites it, which is a different thing and is not counted here.
    claimed = {
        (section, fragment)
        for identifier, section, fragment in _sources()
        if tags[identifier] in CLAIMS
    }

    def restated(number: str, paragraph: str) -> bool:
        return any(
            section == number and fragment in paragraph for section, fragment in claimed
        )

    def unrestated_in(number: str, body: str, tagged: re.Pattern[str]) -> list[str]:
        return [
            f"§{number}: {' '.join(paragraph.split())[:90]}"
            for paragraph in body.split("\n\n")
            if tagged.search(paragraph) and not restated(number, paragraph)
        ]

    unrestated: list[str] = []
    for number, body in sections.items():
        if number.startswith(_RESULTS_FROM):
            unrestated += unrestated_in(number, body, _RESULT)
        elif number.startswith(_OBLIGATIONS_FROM):
            if not any(section == number for section, _ in claimed):
                unrestated.append(f"§{number}: {body.splitlines()[0]}")
        elif number.isdigit() and int(number) < _FOUNDATIONS_BELOW:
            unrestated += unrestated_in(number, body, _FOUNDATION)

    assert not unrestated, (
        "results the argument carries and this page does not restate:\n"
        + "\n".join(f"  {row}" for row in unrestated)
        + "\n\nAdd the tagged paragraph here, with a SOURCE line naming it."
    )
