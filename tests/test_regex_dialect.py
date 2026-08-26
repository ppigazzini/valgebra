"""The `Regex` dialect is Rust's, and the pages have to say so.

`Regex` runs the pattern natively, which is what buys the linear-time guarantee.
It also means the dialect is the Rust engine's rather than `re`'s, and the two
disagree on patterns *both* accept — so compiling successfully is not a test of
which language a pattern denotes.

Each case below pins the behaviour and asserts `docs/05-refinements.md` records it.
A reader porting patterns from `re` has no other way to find out.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated

import pytest

from valgebra import Regex, Validator

_PAGE = (
    Path(__file__).resolve().parent.parent / "docs" / "05-refinements.md"
).read_text(encoding="utf-8")


def _matches(pattern: str, text: str) -> bool:
    """Whether valgebra's anchored native match admits ``text``."""
    return Validator(Annotated[str, Regex(pattern)]).is_valid(text)


def test_a_posix_bracket_expression_means_different_things() -> None:
    """Python has no POSIX classes, so it reads a class and a literal bracket."""
    assert _matches(r"[[:alpha:]]", "a")
    assert not _matches(r"[[:alpha:]]", "a]")

    with pytest.warns(FutureWarning):  # Python's own hint that it reads a set
        assert re.fullmatch(r"[[:alpha:]]", "a") is None
    assert re.fullmatch(r"[[:alpha:]]", "a]") is not None


def test_case_folding_stops_at_the_dotless_pair() -> None:
    """The engines fold the ASCII pair alike and the Turkish pair differently."""
    assert _matches("(?i)i", "I")
    # Written as escapes: a bare dotless i is indistinguishable from `i` in
    # source, and the point of the case is that they are different characters.
    dotless, dotted = "\u0131", "\u0130"
    assert not _matches("(?i)i", dotless)
    assert not _matches("(?i)i", dotted)

    assert re.fullmatch("(?i)i", dotless) is not None
    assert re.fullmatch("(?i)i", dotted) is not None


def test_a_property_escape_builds_here_and_not_in_python() -> None:
    """A pattern only one engine accepts is the loudest case, and the rarest."""
    assert _matches(r"\p{L}+", "ab")

    with pytest.raises(re.error):
        re.compile(r"\p{L}+")


@pytest.mark.parametrize(
    "topic",
    ["[[:alpha:]]", r"\p{L}", "case folding"],
    ids=["posix-classes", "property-escapes", "case-folding"],
)
def test_the_page_names_the_divergence(topic: str) -> None:
    """Each class a reader can hit is one the refinements page names.

    Matched case-insensitively: prose capitalises at the start of a sentence,
    and the assertion is about the topic being covered, not its typography.
    """
    assert topic.lower() in _PAGE.lower(), (
        f"docs/05-refinements.md does not mention {topic!r}; a pattern valid in "
        "both engines can denote different sets, and nothing else says so"
    )
