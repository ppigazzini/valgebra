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
import warnings
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
    ("pattern", "operator"),
    [
        (r"[\w--\d]", "set difference"),
        (r"[\w&&\d]", "set intersection"),
        (r"[\w~~\d]", "set symmetric difference"),
        (r"[a[bc]]", "nested set"),
    ],
    ids=["difference", "intersection", "symmetric-difference", "nested-set"],
)
def test_a_class_set_operator_is_refused_rather_than_read(
    pattern: str, operator: str
) -> None:
    """The quiet divergence is turned into a loud one.

    This engine reads each of these as an operator over classes. `re` reads
    none of them that way, so the same pattern denotes two sets and compiling it
    says nothing about which. The refusal names the operator this engine would
    have read, which is the fact a reader porting the pattern needs.
    """
    with pytest.raises(ValueError, match=operator):
        Validator(Annotated[str, Regex(pattern)])


def test_what_python_does_with_each_refused_form() -> None:
    """`re`'s three answers, which are why the refusal covers all three.

    The forms are not one case. `re` **raises** on a doubled hyphen, **warns**
    that it may one day read the two symbol operators, and says **nothing at
    all** about a nested set — so a reader who ported a pattern would find out
    at once, eventually, or never. The last is the one this refusal is most for,
    and it is the one no reservation in `re` would have caught.
    """
    with warnings.catch_warnings():
        # It warns *and* raises, and the raise is the part being pinned.
        warnings.simplefilter("ignore", FutureWarning)
        with pytest.raises(re.error):
            re.compile(r"[\w--\d]")

    for pattern in (r"[\w&&\d]", r"[\w~~\d]"):
        with pytest.warns(FutureWarning):
            re.compile(pattern)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # One of `a [ b c`, then a literal `]`. This engine would have read the
        # union of `a` and `[bc]`, and the two share not one string.
        assert re.compile(r"[a[bc]]").fullmatch("a]") is not None
        assert re.compile(r"[a[bc]]").fullmatch("b") is None


@pytest.mark.parametrize(
    "pattern",
    ["a--b", "[--/]", "[^--/]", r"[\w\-\-\d]", r"[[:alpha:]]"],
    ids=["outside-a-class", "range-from-hyphen", "negated-range", "escaped", "posix"],
)
def test_a_form_the_two_engines_agree_on_is_not_refused(pattern: str) -> None:
    """A refusal wider than the reservation it mirrors turns away a valid pattern.

    Each of these carries the characters and none of them carries the operator:
    outside a class they are literals to both engines, at the first position of
    a class the hyphen is literal to both, escaped they are literal by
    construction, and a POSIX class is the divergence the page documents above
    rather than one it hides.
    """
    assert Validator(Annotated[str, Regex(pattern)]) is not None


def test_a_verbose_pattern_may_end_in_a_comment() -> None:
    """The anchor a whole-string match needs does not eat the pattern's comment.

    In extended mode a `#` runs to the end of the line, so a pattern ending in a
    comment would swallow the closing half of the anchor. It is the form a
    reader writes a long pattern in, and `re` accepts it.
    """
    pattern = r"""(?x)
        \d{4}   # year
        -
        \d{2}   # month
    """
    assert _matches(pattern, "2026-09")
    assert not _matches(pattern, "2026-09-15")
    assert re.fullmatch(pattern, "2026-09", re.VERBOSE) is not None


@pytest.mark.parametrize(
    "topic",
    ["[[:alpha:]]", r"\p{L}", "case folding", "set symmetric difference"],
    ids=["posix-classes", "property-escapes", "case-folding", "class-set-operators"],
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
