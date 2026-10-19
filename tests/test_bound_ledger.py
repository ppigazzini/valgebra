"""Every bound the tree declares is driven by a test, or accepted with a reason.

A bound is a constant an untrusted input can reach: a depth the walk
descends, a count the descriptor builds, a size a table may grow to. Each one
is a promise on the limits page -- the operation stops here, cleanly -- and a
promise a test does not drive to its edge is one the tree can stop keeping
without a gate noticing, since a bound that moves changes no answer on any
input short of it.

So the universe is read out of the source, by the scan the architecture page's
bounds table is held to (`scripts/docs_lint.py`): every file-scope integer
constant a crate chooses, less the rows that table kinds "not a bound" -- a
capacity, such as how much a buffer reserves first -- and with the three the
stub exports. A name is no test of whether a constant bounds anything, and a
scan by name read 27 of the 35 the table declares. A bound is covered when a
test names it -- a Rust test by the identifier, since the crate's tests reach
its constants; a Python test by a `# BOUND: <name>` marker on the test that
drives it, since the binding's constants are not exported to Python. A bound no
test drives is accepted with the reason it is out of reach, and a reason for a
bound a test does drive fails, so an excuse cannot outlive the gap it excuses.

LEDGER: every declared bound is driven by a test, or accepted with a reason

PRODUCT: every declared bound
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

from _reason import is_a_reason
from _tree import ROOT, load_script

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.repository

CRATES = ROOT / "crates"
STUB = ROOT / "python" / "valgebra" / "_valgebra.pyi"
PAGE = ROOT / "docs" / "dev" / "00-architecture.md"

#: A row of the bounds table kinded as no bound: a capacity the scan reads
#: because it is a figure somebody chose, which bounds nothing a test could
#: drive to an edge.
_NOT_A_BOUND = re.compile(
    r"^\| `[^`]+` \| `([^`]+)` \| `[^`]+` \| not a bound \|", re.MULTILINE
)
_STUB_BOUND = re.compile(r"^(MAX_[A-Z_]+): int$", re.MULTILINE)
_MARKER = re.compile(r"#\s*BOUND:\s*([A-Z_]+)")

#: Bounds no test drives to their edge, each with the reason it is out of
#: reach. A reason is a sentence about the bound, not a note that nobody got
#: to it.
ACCEPTED: dict[str, str] = {
    "BUILD_SIZE_LIMIT": (
        "headroom for the automaton builder, above the state bound that "
        "decides: the smallest pattern whose table sits inside the band this "
        "leaves takes thirteen seconds to build, which is more than one line "
        "is worth on every run of the suite"
    ),
}


def _test_files() -> list[Path]:
    """Give every file a bound may be named in: the Rust and Python tests."""
    rust = [
        path
        for path in sorted(CRATES.rglob("*.rs"))
        if path.name.endswith("tests.rs")
        or path.name in {"laws.rs", "index_laws.rs"}
        or "interpreter" in path.name
    ]
    return [*rust, *sorted((ROOT / "tests").glob("*.py"))]


def _declared() -> dict[str, str]:
    """Give every bound the tree declares, with the file that owns it."""
    capacities = set(_NOT_A_BOUND.findall(PAGE.read_text(encoding="utf-8")))
    found = {
        name: where
        for where, name in load_script("docs_lint").chosen_constants()
        if name not in capacities
    }
    for match in _STUB_BOUND.finditer(STUB.read_text(encoding="utf-8")):
        found[match.group(1)] = STUB.relative_to(ROOT).as_posix()
    # The scan is the detector: an empty universe would pass having read nothing,
    # and the table's capacities are what keeps it from reading two of them.
    assert len(found) >= 30, f"the scan found only {sorted(found)}"
    assert capacities, "the bounds table kinds nothing as not a bound"
    return found


def _named() -> set[str]:
    """Give every bound a test names, by identifier or by marker.

    A Rust test names a core bound by its identifier. A Python test names one
    of the stub's bounds by its identifier and any other only by a marker: a
    core or binding constant is not a name Python can reach, so a bare word
    matching one in a Python file is prose about it -- or, in the plant
    harness, the text of the plant itself.
    """
    stub = set(_STUB_BOUND.findall(STUB.read_text(encoding="utf-8")))
    declared = re.compile(
        rf"\b({'|'.join(sorted(_declared(), key=len, reverse=True))})\b"
    )
    names: set[str] = set()
    for path in _test_files():
        text = path.read_text(encoding="utf-8")
        names.update(_MARKER.findall(text))
        if path.suffix == ".rs":
            names.update(declared.findall(text))
        else:
            names.update(name for name in stub if re.search(rf"\b{name}\b", text))
    return names


def test_every_declared_bound_is_driven_or_accepted() -> None:
    declared = _declared()
    named = _named()
    undriven = sorted(
        name for name in declared if name not in named and name not in ACCEPTED
    )
    assert not undriven, (
        f"bounds no test names: {undriven}. Drive each to its edge, naming it by "
        "identifier in a Rust test or with `# BOUND: <name>` on the Python test, "
        "or accept it here with the reason it is out of reach."
    )


def test_no_accepted_bound_is_driven() -> None:
    """The other direction: an excuse for a bound a test does drive is stale."""
    named = _named()
    stale = sorted(name for name in ACCEPTED if name in named)
    assert not stale, f"accepted bounds a test now drives: {stale}"


def test_every_accepted_bound_exists_and_has_a_reason() -> None:
    declared = _declared()
    unknown = sorted(name for name in ACCEPTED if name not in declared)
    assert not unknown, f"accepted bounds the tree does not declare: {unknown}"
    short = sorted(name for name, reason in ACCEPTED.items() if not is_a_reason(reason))
    assert not short, f"accepted bounds with no reason: {short}"


def test_every_marker_names_a_declared_bound() -> None:
    declared = _declared()
    stale = sorted(
        name
        for path in (ROOT / "tests").glob("*.py")
        for name in _MARKER.findall(path.read_text(encoding="utf-8"))
        if name not in declared
    )
    assert not stale, f"`# BOUND:` markers naming no bound the tree declares: {stale}"
