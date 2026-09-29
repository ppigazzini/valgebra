"""A ledger's written reason is a sentence, and not one written to be replaced.

`tests/_reason.py` is what the ledgers hold an excuse's reason to, in place of a
length floor filler passes. Driven here on both sides of each rule.
"""

from __future__ import annotations

import pytest

from _reason import is_a_reason

# The repository checks are not the product suite: this file holds a helper of
# the repository's own ledgers, which does not ship in a wheel.
pytestmark = pytest.mark.repository


@pytest.mark.parametrize(
    "text",
    [
        "a snippet is not a package",
        "the file is 56 regions, and 14 of them are the error arm of a `write!`",
    ],
)
def test_a_sentence_is_a_reason(text: str) -> None:
    assert is_a_reason(text)


@pytest.mark.parametrize(
    "text",
    [
        "TODO: write down why this pair declines once somebody has read it.",
        "FIXME later, when the lowering lands",
        "a placeholder until the review",
        "see above",
        "",
    ],
)
def test_filler_is_not_a_reason(text: str) -> None:
    assert not is_a_reason(text)
