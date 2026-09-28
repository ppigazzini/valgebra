"""Every persisted proptest seed sits beside the property tests that replay it.

proptest keeps the seed of a case that failed in `proptest-regressions/`, at the
path of the source file whose `proptest!` block drew it, and replays every seed
there before it draws anything new. The path is the whole link: nothing else
says which tests a seed belongs to. A block that moves -- a test module put in a
sibling file, which `docs/dev/08-testing.md` asks of a long one -- leaves its
seeds at the old path, where no test opens them, and no run says so.

So each seed file is read against the tree: the source file at its path exists
and holds a `proptest!` block. The other direction is no defect, since a
property test that never failed has no seeds.

LEDGER: every persisted proptest seed sits beside the property tests that replay it
"""

from __future__ import annotations

from pathlib import Path

import pytest

# This reads the tree rather than the library: it is a claim about how the
# sources are laid out, and nothing it touches ships in a wheel.
pytestmark = pytest.mark.repository

ROOT = Path(__file__).resolve().parent.parent

#: The directory proptest persists seeds to, beside each crate's `src`.
SEEDS = "proptest-regressions"


def _seed_files() -> list[Path]:
    return sorted(ROOT.glob(f"crates/*/{SEEDS}/**/*.txt"))


def _reader(seeds: Path) -> Path:
    """Name the source file whose `proptest!` blocks replay `seeds`."""
    parts = seeds.parts
    at = parts.index(SEEDS)
    return Path(*parts[:at], "src", *parts[at + 1 :]).with_suffix(".rs")


def test_every_seed_file_sits_beside_a_property_test() -> None:
    orphaned = [
        f"{seeds.relative_to(ROOT)} (no `proptest!` in {reader.relative_to(ROOT)})"
        for seeds in _seed_files()
        if not (
            (reader := _reader(seeds)).is_file()
            and "proptest!" in reader.read_text(encoding="utf-8")
        )
    ]
    assert not orphaned, (
        "seed files no property test replays: "
        + ", ".join(orphaned)
        + ". Move each to the path of the file its tests moved to."
    )


def test_the_scan_reads_the_seed_files_that_are_there() -> None:
    """The other direction: a scan finding nothing would pass having read nothing."""
    files = _seed_files()
    assert len(files) >= 5, f"the scan found {len(files)} seed files"
    for seeds in files:
        text = seeds.read_text(encoding="utf-8")
        assert any(line.startswith("cc ") for line in text.splitlines()), seeds
