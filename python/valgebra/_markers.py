"""Refinement markers valgebra defines because standard typing has none.

`Annotated` refinements normally use `annotated_types` markers (`Ge`, `Len`,
...), but that package — and the typing stdlib — define no marker for a string
pattern. `Regex` fills that one gap: it is `Annotated` metadata, not a
combinator, so the typing-first surface stays the single way to express a
constraint.
"""

from __future__ import annotations


class Regex:
    """`Annotated` metadata: a string fully matches this regular expression.

    Use as `Annotated[str, Regex(r"[0-9a-f]{24}")]`. The match is anchored — the
    whole string must match, as `re.fullmatch` does — and runs natively on the
    Rust path (a linear-time engine), so a pattern check stays on the validation
    fast path rather than crossing into Python like a predicate. A bare
    `re.Pattern` (from `re.compile`) is accepted as metadata too.
    """

    __slots__ = ("pattern",)

    def __init__(self, pattern: str) -> None:
        self.pattern = pattern

    def __repr__(self) -> str:
        return f"Regex({self.pattern!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Regex) and other.pattern == self.pattern

    def __hash__(self) -> int:
        return hash((Regex, self.pattern))
