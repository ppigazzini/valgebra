"""What a ledger's written reason is held to, and what it is not.

A length cannot tell a reason from filler: `len(reason) > 40` passes "TODO: say
why this entry is here once somebody has read it". So a reason is held to the
two things a reader can check without judging the argument -- it is a sentence
rather than a word, and it is not a placeholder -- and the argument itself is
left to review, which is the only reader that can weigh it.
"""

from __future__ import annotations

import re

#: The markers a reason written to be replaced carries.
_PLACEHOLDER = re.compile(r"\b(?:TODO|TBD|FIXME|XXX)\b|placeholder", re.IGNORECASE)

#: The fewest words a reason is written in: "a snippet is not a package" is six.
_WORDS = 4


def is_a_reason(text: str) -> bool:
    """Give whether `text` reads as a reason rather than a gap in one."""
    return len(text.split()) >= _WORDS and _PLACEHOLDER.search(text) is None
