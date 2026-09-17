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
* the count of `OWED:` lines is recorded and may only shrink, so the debt the
  page admits is a number the tree holds rather than a sentence a reader skims.

What it cannot do is read a test and judge whether it holds the claim. That is
the reviewer's, and the `HELD-BY:` line is where the reviewer's answer is
written down instead of being re-derived.

LEDGER: every theory result, obligation and deviation names a test, held or owed
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

import pytest

pytestmark = pytest.mark.repository

ROOT = Path(__file__).resolve().parent.parent
THEORY = ROOT / "docs" / "dev" / "10-theory.md"

#: The tags a *claim* carries. The page's own paragraph explaining them writes
#: the words without brackets, which is what keeps it out of the universe.
CLAIMS = ("[LOAD-BEARING]", "[OBLIGATION]", "[DEVIATION]")
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
RECORDED_OWED = 7

#: Where a named test may live. Rust tests are `fn name(`, Python `def name(`.
_SOURCES = (
    (ROOT / "crates", "*.rs", re.compile(r"\bfn\s+(\w+)\s*[(<]")),
    (ROOT / "tests", "*.py", re.compile(r"^def\s+(\w+)\s*\(", re.MULTILINE)),
)


class Claim(NamedTuple):
    """One tagged paragraph and what the page says holds it."""

    text: str
    tag: str
    names: list[str]
    owed: str | None


def _paragraphs() -> list[str]:
    return THEORY.read_text(encoding="utf-8").split("\n\n")


def _claims() -> list[Claim]:
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
        tags = [tag for tag in CLAIMS if tag in paragraph]
        if tags:
            assert len(tags) == 1, (
                f"a paragraph carrying two tags: {_summarise(paragraph)}"
            )
            found.append(Claim(paragraph, tags[0], [], None))
    return found


def _held() -> list[Claim]:
    return [claim for claim in _claims() if claim.names]


def _defined_names() -> set[str]:
    """Every function name the tree defines, in either language."""
    names: set[str] = set()
    for root, glob, pattern in _SOURCES:
        for path in root.rglob(glob):
            names.update(pattern.findall(path.read_text(encoding="utf-8")))
    return names


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
    assert sum(text.count(tag) for tag in CLAIMS) == len(claims)
    by_tag = {tag: sum(1 for claim in claims if claim.tag == tag) for tag in CLAIMS}
    assert by_tag["[LOAD-BEARING]"] >= 8, by_tag
    assert by_tag["[OBLIGATION]"] >= 4, by_tag
    assert by_tag["[DEVIATION]"] >= 4, by_tag


def test_every_claim_names_a_test_held_or_owed() -> None:
    """A result the design rests on says what would catch it going wrong.

    Or says, in a form this file can read, that nothing does yet and which
    test would. A claim with neither is a sentence.
    """
    unheld = [
        _summarise(claim.text)
        for claim in _claims()
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
    missing = [name for name in names if name not in defined]
    assert not missing, f"{_summarise(claim)}\nnames no test: {missing}"


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
        for name in claim.names:
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
    # And the names are not all in one file, which would make the ledger a
    # statement about one module rather than about the tree.
    all_names = {name for claim in _held() for name in claim.names}
    assert len(all_names) >= 15, sorted(all_names)
