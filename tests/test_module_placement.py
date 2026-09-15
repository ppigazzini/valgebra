"""A test module longer than a screen lives in a file of its own.

`docs/dev/08-testing.md` states the rule and the reason: a module holding more
assertions than code makes a reader looking for the subtype rules scroll past
them to find the rules. The shape that fixes it is `#[cfg(test)] mod tests;`
with the body in a sibling file -- a child module, reaching private items
through `use super::*`, compiled only under `cfg(test)`, and out of the way.

A page stating a rule the tree does not follow is the same defect as a page
stating a dependency order the module graph does not, which
`tests/test_module_direction.py` holds for that case: the page is the definition
of what the code is organised into, and a rule nothing holds decays into prose.

"A screen" needs a number to be checkable, and this file is where that number
lives. The bar is generous on purpose: a module short enough to read past is not
in anybody's way, and several sit under it and stay where they are. What the bar
refuses is the drift -- a module that grows, one case at a time, until the file
it tests is mostly not that file.

The count is of the module's own body, braces included, which is what a reader
scrolls past. A file may hold more than one such module (`check/ctx.rs` declares
two), so each is measured on its own.

LEDGER: no inline test module is longer than a screen
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# This reads the tree rather than the library: it is a claim about how the
# sources are laid out, and nothing it touches ships in a wheel.
pytestmark = pytest.mark.repository

ROOT = Path(__file__).resolve().parent.parent
CRATES = ROOT / "crates"

#: The longest inline test module the rule allows, in lines. `docs/dev/08-testing.md`
#: says "a screen"; this is that, rounded up past the three modules that stay.
BAR = 100

#: `#[cfg(test)] mod <name> {` -- the inline form, as against `mod <name>;`.
INLINE = re.compile(
    r"^#\[cfg\((?:test|all\(test[^\n]*)\)\]\n^mod (\w+) \{$", re.MULTILINE
)


def _sources() -> list[Path]:
    """Collect every Rust source in the workspace, build output excluded."""
    sources = [
        path for path in sorted(CRATES.rglob("*.rs")) if "target" not in path.parts
    ]
    # The glob is the detector: an empty universe would pass having read nothing.
    assert len(sources) >= 20, f"the source glob found only {len(sources)} files"
    return sources


def _inline_modules(path: Path) -> list[tuple[str, int]]:
    """Return each inline test module's name and its length in lines.

    The body runs from the `mod` line to the closing brace at column zero that
    follows it, which is how these modules are written and how `rustfmt` keeps
    them.
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    found = []
    for match in INLINE.finditer(text):
        start = text[: match.start()].count("\n") + 1  # the `#[cfg(test)]` line
        for offset in range(start + 1, len(lines)):
            if lines[offset] == "}":
                found.append((match.group(1), offset - start))
                break
    return found


def test_no_inline_test_module_is_longer_than_a_screen() -> None:
    oversized = sorted(
        (length, f"{path.relative_to(ROOT)}::{name}")
        for path in _sources()
        for name, length in _inline_modules(path)
        if length > BAR
    )
    assert not oversized, (
        "inline test modules past the bar: "
        + ", ".join(f"{where} ({length} lines)" for length, where in oversized)
        + f". The rule is {BAR} lines; docs/dev/08-testing.md says why. Declare "
        "`#[cfg(test)] mod tests;` and move the body to a sibling file."
    )


def test_the_scan_reads_the_modules_that_are_there() -> None:
    """The other direction: a scan finding nothing would pass having read nothing.

    Both forms must be present for this ledger to mean anything -- inline
    modules under the bar, which is what the rule permits, and sibling
    declarations, which is what it asks for.
    """
    inline = [
        (path, name, length)
        for path in _sources()
        for name, length in _inline_modules(path)
    ]
    assert inline, "no inline test module found at all; the pattern stopped matching"
    siblings = [
        path
        for path in _sources()
        if re.search(
            r"^#\[cfg\((?:test|all\(test[^\n]*)\)\]\n^mod (\w+);$",
            path.read_text(encoding="utf-8"),
            re.MULTILINE,
        )
    ]
    assert len(siblings) >= 8, (
        f"only {len(siblings)} sibling test module declarations; the tree has more"
    )
