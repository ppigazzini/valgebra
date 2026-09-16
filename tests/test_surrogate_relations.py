r"""A `str` the codecs cannot encode is a `str`, and no pattern matches it.

`Schema::Str` denotes every Python `str`, and a Python `str` is a sequence of
code points -- surrogates included. `"\\ud800"` is one character long, compares
and hashes like any other string, and is a member of `str`.

A pattern is a different matter. `docs/14-soundness.md` records it: a `Regex`
constraint matches the *text* of a string, and a string carrying a lone surrogate
has no text, so it matches no pattern at all. The walk says so, in both
directions.

Put together, the two make a relation the sets have to answer: `str` is **not**
below `Annotated[str, Regex("(?s).*")]`, and `"\\ud800"` is the value that says
so. Reading the kind's universe as the encodable strings alone loses that value,
and the difference between the two schemas comes out empty -- a proof against a
string a caller can write in one line.

The rows below pair each relation with the value that decides it, and the rows
beside them are the ones the change must not move: a pattern inside another
pattern, a literal inside its own language, and the `bytes` kind, whose universe
is every byte string and always was.
"""

from __future__ import annotations

from typing import Annotated, Literal

import annotated_types as at
import pytest

from valgebra import Regex, ValidationError, Validator, complement, intersection

# One lone surrogate, and a string carrying one among ordinary characters.
LONE = "\ud800"
MIXED = "a\udfffb"


def test_a_lone_surrogate_is_a_member_of_str() -> None:
    """The kind admits it, which is what makes the rest a question."""
    assert Validator(str).is_valid(LONE) is True
    assert Validator(str).is_valid(MIXED) is True
    assert len(LONE) == 1


def test_no_pattern_matches_a_lone_surrogate() -> None:
    """The walk's own answer, which the sets have to agree with."""
    for pattern in ("(?s).*", "[\\s\\S]*", ".*", "\\w*"):
        catch_all = Annotated[str, Regex(pattern)]
        assert Validator(catch_all).is_valid(LONE) is False, pattern
        assert Validator(catch_all).is_valid(MIXED) is False, pattern
        # And it does match an ordinary string, so the pattern is not empty.
        assert Validator(catch_all).is_valid("") is True, pattern


def test_the_kind_is_not_below_a_pattern_that_matches_every_text() -> None:
    """The relation the witness settles."""
    catch_all = Annotated[str, Regex("(?s).*")]
    assert Validator(str).relation_to(catch_all) == "not_subset"
    assert (
        Validator(str).relation_to(Annotated[str, Regex("[\\s\\S]*")]) == "not_subset"
    )
    # The difference holds the witness, so it is not empty.
    difference = intersection(str, complement(Validator(catch_all)))
    assert difference.is_valid(LONE) is True
    assert difference.is_empty() is False


def test_a_pattern_is_below_the_kind_it_constrains() -> None:
    """The other direction holds, and is proved: every match is a `str`."""
    assert Validator(Annotated[str, Regex("(?s).*")]).relation_to(str) == "subset"
    assert Validator(Annotated[str, Regex("a+")]).relation_to(str) == "subset"


def test_a_length_bound_counts_a_surrogate_as_a_character() -> None:
    """A bound is about how many characters a value has, not how it encodes."""
    assert Validator(Annotated[str, at.MinLen(1)]).is_valid(LONE) is True
    assert Validator(Annotated[str, at.MaxLen(1)]).is_valid(LONE) is True
    assert Validator(Annotated[str, at.MinLen(2)]).is_valid(LONE) is False
    assert Validator(Annotated[str, at.MinLen(3)]).is_valid(MIXED) is True
    # So the set a bound denotes holds it too, and `str` is below the bound of
    # zero for that reason.
    assert Validator(str).relation_to(Annotated[str, at.MinLen(0)]) == "subset"
    assert Validator(Annotated[str, at.MinLen(1)]).relation_to(str) == "subset"


def test_a_bound_that_a_surrogate_string_meets_is_not_below_a_pattern() -> None:
    """The two readings meet here: a bound holds it and a pattern does not."""
    bounded = Annotated[str, at.MinLen(1)]
    catch_all = Annotated[str, Regex("(?s).+")]
    assert Validator(bounded).is_valid(LONE) is True
    assert Validator(catch_all).is_valid(LONE) is False
    assert Validator(bounded).relation_to(catch_all) == "not_subset"


def test_the_patterns_still_decide_against_each_other() -> None:
    """Inclusion between two languages is unmoved: neither holds a surrogate."""
    assert (
        Validator(Annotated[str, Regex("a")]).relation_to(Annotated[str, Regex("ab?")])
        == "subset"
    )
    assert Validator(Annotated[str, Regex("a")]).relation_to(Literal["a"]) == "subset"
    assert Validator(Literal["a"]).relation_to(Annotated[str, Regex("a")]) == "subset"
    assert (
        Validator(Annotated[str, Regex("ab?")]).relation_to(Annotated[str, Regex("a")])
        == "not_subset"
    )


def test_the_bytes_kind_is_every_byte_string_as_before() -> None:
    """`bytes` has no encoding question, so its universe is unchanged.

    A pattern is matched against text, which the frontend refuses to ask of a
    `bytes` base, so the kind's rows here are its length bounds and the byte
    strings no `str` can carry.
    """
    assert Validator(bytes).relation_to(Annotated[bytes, at.MinLen(0)]) == "subset"
    assert Validator(bytes).relation_to(Annotated[bytes, at.MinLen(1)]) == "not_subset"
    assert Validator(Annotated[bytes, at.MinLen(1)]).relation_to(bytes) == "subset"
    assert Validator(bytes).is_valid(b"\xff\xfe") is True
    assert Validator(Annotated[bytes, at.MinLen(2)]).is_valid(b"\xff\xfe") is True


@pytest.mark.parametrize("value", [LONE, MIXED])
def test_a_surrogate_string_is_outside_every_pattern_in_both_modes(value: str) -> None:
    """`is_valid` and `validate` give one answer, as they do everywhere."""
    compiled = Validator(Annotated[str, Regex("(?s).*")])
    assert compiled.is_valid(value) is False
    with pytest.raises(ValidationError) as caught:
        compiled.validate(value)
    assert [item["code"] for item in caught.value.errors] == ["string_pattern_mismatch"]
