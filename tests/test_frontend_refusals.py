"""Every refusal the frontend writes is one a test reads back.

A refusal is a sentence, and the sentence is the product. A caller who writes an
annotation this library will not compile meets the message and nothing else: the
exception's *type* is `NotImplementedError` or `ValueError` at thirty-odd sites
and says which one went off at none of them. So a test asserting the type holds
the refusal happening and not the refusal being right, and a message can be
reworded into nonsense, or into a sentence about a different schema, with every
such test still green.

This is the ledger that ends that. The universe is read out of the Rust: every
site in the frontend that constructs a refusal, with the literal text it writes.
A site is **held** when the product suite carries a pattern the site's message
satisfies -- the same question `pytest.raises` asks, put to the message the tree
has rather than to the one the test was written against. Reword the sentence and
the pattern stops matching, which is the failure this exists to produce.

What it cannot do is judge whether the sentence is *good*. That is the
reviewer's, and `docs/03-schema-language.md` and `docs/05-refinements.md` are
where the reviewer's answer is written down.

The other direction is the accepted list: a site the suite does not read is
written down with a reason, and a reason for a site that *is* read fails too, so
an excuse cannot outlive the gap it excuses.

LEDGER: every frontend refusal message is matched by a test, or accepted
"""

from __future__ import annotations

import ast
import functools
import re
import warnings
from pathlib import Path

import pytest

pytestmark = pytest.mark.repository

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "crates" / "valgebra-py" / "src"

#: The frontend's files: the dispatch, and the four surfaces beside it. A
#: refusal anywhere else is about a value rather than about an annotation, and
#: `tests/test_error_contract.py` is where those are held.
SOURCES = (
    "build.rs",
    "build/generics.rs",
    "build/refine.rs",
    "build/classes.rs",
    "build/dialect.rs",
)

#: Every way the frontend constructs a refusal.
_OPENS = re.compile(
    r"\b(?:PyValueError::new_err|PyTypeError::new_err"
    r"|PyNotImplementedError::new_err|not_implemented)\s*\("
)

#: A `{}` or `{name}` hole, which the message fills at the site. The ledger
#: reads the literal text around them, because that is the part a test can
#: match without knowing the value.
_HOLE = re.compile(r"\{[^{}]*\}")

#: The shortest string that counts as a pattern about a refusal.
#:
#: A refusal is a sentence and a pattern about one is a phrase from it, so this
#: is a floor on *what a pattern is*. It applies to a string spelled at `match=`
#: as much as to one read out of a table, because a short one holds a sentence
#: by accident either way: `match="set"` is satisfied by any refusal with the
#: word in it, and the planted defect -- a message reworded to "this class does
#: not name a set" -- went on reading as held while it was counted.
#:
#: Eight leaves every phrase the suite writes and drops every bare word it
#: writes: `MultipleOf` and `contractive` are kept, `set` and `tuple` are not.
MIN_PATTERN = 8

#: Sites the suite does not read, each with the reason it does not.
#:
#: A reason is a sentence about the site, not a note that nobody got to it.
ACCEPTED: dict[str, str] = {}


def _read_literal(text: str, start: int) -> tuple[int, str]:
    """Read one Rust string literal, returning where it ended and its text."""
    escapes = {"n": "\n", "t": "\t", "\\": "\\", '"': '"', "'": "'"}
    index, size, out = start + 1, len(text), []
    while index < size and text[index] != '"':
        if text[index] != "\\":
            out.append(text[index])
            index += 1
            continue
        following = text[index + 1]
        if following == "\n":
            # A continuation: the newline and the indent after it are not part
            # of the sentence.
            index += 2
            while index < size and text[index] in " \t":
                index += 1
            continue
        out.append(escapes.get(following, following))
        index += 2
    return index + 1, "".join(out)


def _literals(text: str, start: int) -> list[str]:
    """Give the string literals inside the call whose ``(`` is at `start`.

    A scan rather than a regex because a refusal is one sentence spelled across
    several source lines: Rust's line continuation joins them, and a pattern
    stopping at the first closing quote would read a tenth of the message.
    """
    depth, index, found, size = 0, start, [], len(text)
    while index < size:
        char = text[index]
        if char == '"':
            index, literal = _read_literal(text, index)
            found.append(literal)
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return found
        index += 1
    return found


@functools.cache
def _sites() -> dict[str, str]:
    """Give every refusal the frontend writes, keyed by where it is written."""
    found: dict[str, str] = {}
    for name in SOURCES:
        text = (FRONTEND / name).read_text(encoding="utf-8")
        for opener in _OPENS.finditer(text):
            line = text.count("\n", 0, opener.start()) + 1
            message = " ".join(_literals(text, opener.end() - 1)).strip()
            if message:
                found[f"{name}:{line}"] = message
    return found


@functools.cache
def _product_files() -> list[tuple[Path, ast.Module]]:
    """Give the product suite, parsed. A repository check is not part of it."""
    parsed = []
    for path in sorted((ROOT / "tests").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "pytestmark = pytest.mark.repository" in text:
            continue
        try:
            parsed.append((path, ast.parse(text)))
        except SyntaxError:  # pragma: no cover - the suite parses
            continue
    return parsed


def _match_arguments(tree: ast.Module) -> tuple[set[str], bool]:
    """Give one file's `match=` strings, and whether any is passed by name.

    The second half is what lets a table be read. A test that provokes one
    refusal passes the sentence where it stands; a table of rows passes a
    *variable*, and the sentences sit in the table. Reading only the first
    found nothing in the file written for this ledger -- thirteen rows, every
    one of them a `match=`, none spelled beside the keyword.
    """
    spelled: set[str] = set()
    by_name = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg != "match":
                continue
            if isinstance(keyword.value, ast.Constant) and isinstance(
                keyword.value.value, str
            ):
                spelled.add(keyword.value.value)
            else:
                by_name = True
    return spelled, by_name


def _table_strings(tree: ast.Module) -> set[str]:
    """Give the string elements of a file's tuples and lists.

    Elements, rather than every string in the file: a table row is a tuple, and
    a docstring is not. A rule taking every literal would let this file's own
    paragraphs hold a refusal by quoting it.
    """
    return {
        element.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Tuple | ast.List)
        for element in node.elts
        if isinstance(element, ast.Constant) and isinstance(element.value, str)
    }


@functools.cache
def _patterns() -> set[str]:
    """Give every pattern the product suite asks a refusal to satisfy.

    A file's table is read only where that file passes a variable to `match=`,
    so a table of regexes written as schemas under test -- there is one, in
    `tests/test_surrogate_relations.py` -- is not mistaken for evidence about a
    message.
    """
    found: set[str] = set()
    for _, tree in _product_files():
        spelled, by_name = _match_arguments(tree)
        found |= spelled
        if by_name:
            found |= _table_strings(tree)
    return {pattern for pattern in found if len(pattern.strip()) >= MIN_PATTERN}


def _matches(message: str, pattern: str) -> bool:
    """Whether one pattern matches one message, as `pytest.raises` would.

    The holes are cut rather than filled: what a site interpolates is a class
    name or a number the ledger has no value for, and a pattern depending on
    one would be a pattern about a row rather than about the sentence.
    """
    try:
        with warnings.catch_warnings():
            # A table element may be a character class the engine warns about,
            # because it was written for some other reader. That warning is
            # about the literal, not about this tree.
            warnings.simplefilter("ignore")
            return re.search(pattern, _HOLE.sub("", message)) is not None
    except (re.error, RecursionError):
        # A literal that is not a regex is a string the suite keeps for some
        # other purpose, and holds no refusal.
        return False
    finally:
        # Leave the module cache as it was found. `re` warns about a pattern
        # once per *compilation*, and a compiled pattern is cached, so reading
        # another file's literals here consumed the `FutureWarning` that
        # `tests/test_regex_dialect.py` asserts Python emits for
        # `[[:alpha:]]`. That test then failed in a full run and passed alone,
        # which is the shape of a coupling nobody would look for here.
        re.purge()


def _too_broad(patterns: set[str]) -> set[str]:
    """Give the patterns that match more than half the refusals.

    A pattern this wide says nothing about the site it is counted against, so
    it is evidence for none of them: `match="."` satisfies `pytest.raises` for
    any message at all, and a row carrying one reads here as evidence for every
    refusal in the tree. That row was in this file's own first draft, on a case
    that only needed the exception to be raised, and the plant is what found
    it.

    Half is the line because two sites do share a sentence -- a `...` out of
    place is the same mistake whether the list ends in one or not -- and a rule
    tighter than that would refuse the pattern that correctly holds both.
    """
    sites = _sites()
    limit = len(sites) // 2
    return {
        pattern
        for pattern in patterns
        if sum(1 for message in sites.values() if _matches(message, pattern)) > limit
    }


@functools.cache
def _evidence() -> set[str]:
    """Give the patterns that can hold a site: every one not over-broad."""
    patterns = _patterns()
    return patterns - _too_broad(patterns)


def _is_read(message: str) -> bool:
    """Whether any pattern that counts as evidence matches this message."""
    return any(_matches(message, pattern) for pattern in _evidence())


def test_the_universe_is_read_from_the_frontend() -> None:
    """The parse is a detector, so it must be shown to have read something."""
    sites = _sites()
    assert len(sites) >= 25, sorted(sites)
    # A sentence a reader can check by hand, so a parse that drifted is visible.
    assert len(max(sites.values(), key=len)) > 150
    assert any("is not a schema" in message for message in sites.values())


def test_the_product_suite_is_what_is_searched() -> None:
    """Search the product suite, not the checks that read the tree.

    A repository check's own string would hold a site with work about the tree
    rather than with a refusal a caller can provoke.
    """
    patterns = _patterns()
    assert len(patterns) >= 15, sorted(patterns)
    assert "every frontend refusal message" not in patterns
    read = {path.name for path, _ in _product_files()}
    assert "test_refusal_messages.py" in read
    assert "test_frontend_refusals.py" not in read


def test_a_pattern_that_holds_everything_is_not_evidence() -> None:
    """The breadth rule fires, shown against a pattern that holds every site."""
    assert _too_broad({"."}) == {"."}
    assert "." not in _evidence()
    # And a real sentence survives it, so the rule is a filter and not a ban.
    assert _too_broad({"must be @runtime_checkable"}) == set()


@pytest.mark.parametrize("where", sorted(_sites()))
def test_every_refusal_is_matched_by_a_test_or_accepted(where: str) -> None:
    """A sentence the frontend writes and no test reads fails here."""
    message = _sites()[where]
    read = _is_read(message)
    accepted = where in ACCEPTED
    assert read or accepted, (
        f"{where} writes a refusal no test matches, and has no accepted "
        f"reason:\n  {message}"
    )
    assert not (read and accepted), (
        f"{where} is matched by a test and still carries a reason for not being"
    )


def test_every_accepted_reason_is_a_sentence() -> None:
    """An excuse short enough to be a shrug is not one."""
    sites = _sites()
    for where, reason in ACCEPTED.items():
        assert len(reason) > 40, f"{where}: {reason!r}"
        assert where in sites, f"{where} is accepted and is not a refusal site"
