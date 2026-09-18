"""The crate roots carry the attributes the pages take on trust.

A page may rest on a property of the source rather than on a value the library
answers about. `docs/14-soundness.md` rests on one: the crates contain no
`unsafe`, so the soundness argument carries no memory-safety obligation beyond
the compiler's. Nothing about a schema shows that, and no amount of driving
values would -- an `unsafe` block added tomorrow answers every question here
exactly as the safe code did.

What shows it is the attribute. `#![forbid(unsafe_code)]` at a crate root makes
an `unsafe` block below it a compile error rather than a lint a build can allow
through, so the claim is the compiler's to keep. This file holds the other half:
that the attribute is *there*. Removing it compiles, and the page would go on
asserting a property nothing enforces.

Read from the tracked source rather than from a built artefact, because the
claim is about what the crates say and not about what one build of them did.

LEDGER: every crate root forbids unsafe code
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.repository

ROOT = Path(__file__).resolve().parent.parent
CRATES = ROOT / "crates"

#: The attribute that makes the claim the compiler's. Written as an inner
#: attribute at the crate root, which is the only place it reaches a whole
#: crate: on a module it would forbid `unsafe` in that module alone.
FORBIDDEN = "#![forbid(unsafe_code)]"


def _roots() -> list[Path]:
    """Every crate root of the workspace, read from the tree."""
    found = sorted(CRATES.glob("*/src/lib.rs"))
    assert len(found) >= 2, f"the workspace reads as {found}"
    return found


# TRUST: The crates contain no `unsafe`.
def test_every_crate_root_forbids_unsafe_code() -> None:
    """A crate root without the attribute is one `unsafe` compiles in."""
    without = [
        path.relative_to(ROOT).as_posix()
        for path in _roots()
        if FORBIDDEN not in path.read_text(encoding="utf-8")
    ]
    assert not without, (
        f"crate roots that do not forbid unsafe code: {without}. Put "
        f"`{FORBIDDEN}` at the root, so an `unsafe` block below it is a "
        "compile error rather than a reviewer's job."
    )


def test_no_source_file_writes_unsafe() -> None:
    """The other half: the attribute forbids `unsafe`, and none is written.

    The attribute alone would hold this, and reading the source says so without
    a build: a reader asking whether the trust base is met gets the answer from
    the tree rather than from a compiler they have to run. The fuzz crate is a
    detached workspace the stable gate never compiles
    (`tests/test_build_surfaces.py`), so it is read here rather than left to a
    lane that runs elsewhere.
    """
    written = [
        f"{path.relative_to(ROOT).as_posix()}:{number}"
        for path in sorted((ROOT / "crates").rglob("*.rs"))
        + sorted((ROOT / "fuzz").rglob("*.rs"))
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        )
        if line.lstrip().startswith(("unsafe ", "unsafe{", "unsafe fn"))
    ]
    assert not written, f"`unsafe` in the tracked source: {written}"
