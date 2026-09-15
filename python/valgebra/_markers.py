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

    Immutable, because it is hashable: a marker whose `pattern` can be rebound
    after it is written into an `Annotated` is one whose hash changes while a
    schema holds it. Written out rather than taken from `dataclasses`, and
    annotated without `typing`, because this module is on the import path of
    every program that imports the package and both cost it modules --
    `tests/test_version.py` holds the count.
    """

    __slots__ = ("pattern",)

    pattern: str

    def __init__(self, pattern: str) -> None:
        object.__setattr__(self, "pattern", pattern)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(self._immutable())

    def __delattr__(self, name: str) -> None:
        raise AttributeError(self._immutable())

    def _immutable(self) -> str:
        return f"{type(self).__name__} is immutable"

    def __repr__(self) -> str:
        return f"Regex({self.pattern!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Regex) and other.pattern == self.pattern

    def __hash__(self) -> int:
        return hash((Regex, self.pattern))
