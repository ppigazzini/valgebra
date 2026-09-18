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

import re
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


#: Comment text, blanked rather than cut so a line keeps its number. The pages
#: and the doc comments write the word constantly -- the claim held here *is*
#: "the crates contain no `unsafe`" -- so a scan that reads prose reports the
#: sentence stating the rule as a breach of it.
_COMMENT = re.compile(r"//.*$", re.MULTILINE)

#: The keyword, as a token rather than as a spelling. Anchored on both sides:
#: `unsafe_code` is one word to a boundary, so the attribute that forbids the
#: keyword does not read as a use of it, and a name ending in `unsafe` is not
#: one either.
_UNSAFE = re.compile(r"\bunsafe\b")


def _unsafe_sites(source: str) -> list[int]:
    """Give the 1-indexed lines on which `source` writes the `unsafe` keyword."""
    code = _COMMENT.sub(lambda comment: " " * len(comment.group()), source)
    return [
        number
        for number, line in enumerate(code.splitlines(), start=1)
        if _UNSAFE.search(line)
    ]


def test_the_scan_reads_the_keyword_rather_than_the_word() -> None:
    """The detector, against the shapes it exists to find and the ones to skip.

    A ledger is worth what its scan is worth, and this one is a scan for a
    keyword in a language that writes the same letters in three other places: a
    doc comment explaining the rule, the attribute enforcing it, and an
    identifier carrying it. A scan matching a line's opening reads `unsafe fn`
    and misses `pub unsafe fn`, `unsafe impl` behind a visibility, and the block
    form inside an expression -- which is the shape most `unsafe` takes.

    So the shapes are driven here rather than assumed, in both directions: what
    must be found, and what must be read past.
    """
    found = (
        "unsafe fn raw() {}\n"
        "pub unsafe fn raw() {}\n"
        "    pub(crate) unsafe fn raw() {}\n"
        "unsafe impl Send for Handle {}\n"
        "    let value = unsafe { *pointer };\n"
        "        unsafe { core::hint::unreachable_unchecked() }\n"
        "    let block = unsafe{1};"
    )
    assert _unsafe_sites(found) == [1, 2, 3, 4, 5, 6, 7]

    skipped = (
        "#![forbid(unsafe_code)]\n"
        '#[expect(unsafe_code, reason = "none")]\n'
        "/// The crates contain no `unsafe`, so the obligation is the\n"
        "// unsafe is what this forbids\n"
        "fn unsafely_named() {} // not the keyword\n"
        "let unsafely = 1;"
    )
    assert _unsafe_sites(skipped) == []


def test_no_source_file_writes_unsafe() -> None:
    """The other half: the attribute forbids `unsafe`, and none is written.

    The attribute alone would hold this for the two crates that carry it, and
    reading the source says so without a build: a reader asking whether the
    trust base is met gets the answer from the tree rather than from a compiler
    they have to run.

    The fuzz crate is why the scan is wider than the roots above. It is a
    detached workspace the stable gate never compiles
    (`tests/test_build_surfaces.py`) and it carries no crate-root attribute,
    because a fuzz target's harness macro is not this tree's code to constrain.
    So for that crate this row is the whole of the claim rather than a second
    reading of it.
    """
    written = [
        f"{path.relative_to(ROOT).as_posix()}:{number}"
        for path in sorted((ROOT / "crates").rglob("*.rs"))
        + sorted((ROOT / "fuzz").rglob("*.rs"))
        for number in _unsafe_sites(path.read_text(encoding="utf-8"))
    ]
    assert not written, f"`unsafe` in the tracked source: {written}"
