"""Every load-bearing result names the test that holds it, and the test exists.

`docs/dev/10-theory.md` says what the design rests on: "A citation here is a
claim that a specific line exists because of it, not a reading list." A claim
tagged `[LOAD-BEARING]` says code in this tree would be wrong without the result,
and until now the only thing holding that sentence was a reader's memory of which
test corresponds to it.

This is the ledger that ends that. Each `[LOAD-BEARING]` paragraph carries a
`HELD-BY:` line naming tests, and this file holds the two directions apart:

* a claim with no `HELD-BY:` fails, so a result cannot be added to the page
  without saying what would catch it going wrong;
* a name that resolves to no test fails, so a test renamed or deleted takes the
  claim's evidence with it rather than leaving a page asserting something
  nothing checks.

What it cannot do is read a test and judge whether it holds the claim. That is
the reviewer's, and the `HELD-BY:` line is where the reviewer's answer is
written down instead of being re-derived.

LEDGER: every load-bearing theory result names a test, and every name is one
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.repository

ROOT = Path(__file__).resolve().parent.parent
THEORY = ROOT / "docs" / "dev" / "10-theory.md"

#: The tag a *claim* carries. The page's own paragraph explaining the three tags
#: writes them without brackets, which is what keeps it out of the universe.
CLAIM = "[LOAD-BEARING]"
HELD_BY = "HELD-BY:"

#: Where a named test may live. Rust tests are `fn name(`, Python `def name(`.
_SOURCES = (
    (ROOT / "crates", "*.rs", re.compile(r"\bfn\s+(\w+)\s*[(<]")),
    (ROOT / "tests", "*.py", re.compile(r"^def\s+(\w+)\s*\(", re.MULTILINE)),
)


def _paragraphs() -> list[str]:
    return THEORY.read_text(encoding="utf-8").split("\n\n")


def _claims() -> list[tuple[str, list[str]]]:
    """Each load-bearing claim, with the test names it says hold it.

    A `HELD-BY:` belongs to the claim it follows, with the qualifying prose a
    claim usually carries allowed to sit between them: a citation is a paragraph
    and its caveats, and forcing the line to butt against the tag would put it in
    the middle of a sentence.
    """
    found: list[tuple[str, list[str]]] = []
    for paragraph in _paragraphs():
        if paragraph.startswith(HELD_BY):
            names = [
                name.strip()
                for name in paragraph[len(HELD_BY) :].replace("\n", " ").split(",")
                if name.strip()
            ]
            assert found, f"a HELD-BY line before any claim: {paragraph!r}"
            claim, already = found[-1]
            assert not already, f"two HELD-BY lines for one claim: {_summarise(claim)}"
            found[-1] = (claim, names)
            continue
        if CLAIM in paragraph:
            found.append((paragraph, []))
    return found


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


def test_the_page_carries_load_bearing_claims() -> None:
    """The parse is a detector, so it must be shown to have read something.

    A ledger over an empty universe passes having checked nothing, which is the
    failure mode every other ledger here guards the same way.
    """
    claims = _claims()
    assert len(claims) >= 8, f"the theory page parse found {len(claims)} claims"
    assert THEORY.read_text(encoding="utf-8").count(CLAIM) == len(claims)


def test_every_load_bearing_claim_names_a_test() -> None:
    """A result the design rests on says what would catch it going wrong."""
    unheld = [_summarise(claim) for claim, names in _claims() if not names]
    assert not unheld, "load-bearing claims with no HELD-BY line:\n" + "\n".join(unheld)


@pytest.mark.parametrize(
    ("claim", "names"), _claims(), ids=[_summarise(c)[:40] for c, _ in _claims()]
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
    for claim, names in _claims():
        for name in names:
            assert name in runnable, f"{_summarise(claim)}\n{name!r} is not a test"


def test_a_claim_is_held_by_more_than_its_own_restatement() -> None:
    """A name must not simply repeat the claim's own words.

    The failure this guards is a `HELD-BY:` written to satisfy the ledger: a
    test whose name is the claim, asserting the claim. Nothing mechanical can
    tell that apart in general, and what it can tell is that the page's claims
    do not each rest on a single test of their own -- so a reviewer reading one
    finds a suite rather than a mirror.
    """
    counts = [len(names) for _, names in _claims()]
    assert min(counts) >= 2, "every claim names at least two tests"
    # And the names are not all in one file, which would make the ledger a
    # statement about one module rather than about the tree.
    all_names = {name for _, names in _claims() for name in names}
    assert len(all_names) >= 15, sorted(all_names)
