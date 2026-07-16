"""Mutation survivor ratchet.

Compares the survivors of a `cargo mutants` run against a committed baseline and
fails when a *new* survivor appears -- a mutation the tests no longer catch. It
never targets zero survivors: equivalent mutants exist and are undecidable in
general, so the honest gate is "no regression", not "no survivors".

The baseline is a set of survivor identities, each the survivor line with its
`:LINE:COL` position stripped, so a survivor is matched by file, function, and
mutation rather than by a line number that drifts when unrelated code moves.

Usage:
    python scripts/mutation_gate.py            # gate: exit 1 on a new survivor
    python scripts/mutation_gate.py --update   # re-record the baseline

Reads `mutants.out/missed.txt` (survivors) and `mutants.out/timeout.txt`
(unjudged, treated as survivors) produced by a prior `cargo mutants` run.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "scripts" / "mutation_baseline.json"
OUT = ROOT / "mutants.out"

# `path:LINE:COL: description` -> `path: description`
_POS = re.compile(r"^(?P<path>.+?):\d+:\d+:\s*(?P<desc>.*)$")


def _identity(line: str) -> str:
    line = line.strip()
    m = _POS.match(line)
    return f"{m['path']}: {m['desc']}" if m else line


def _read(name: str) -> list[str]:
    f = OUT / name
    if not f.exists():
        return []
    return [ln for ln in f.read_text().splitlines() if ln.strip()]


def _measured() -> set[str]:
    return {_identity(ln) for ln in _read("missed.txt") + _read("timeout.txt")}


def _require_run() -> None:
    # A missing output dir, or a run that generated nothing, is a broken
    # detector, not a clean tree: refuse to pass rather than report success.
    if not OUT.is_dir():
        sys.exit("mutation_gate: mutants.out/ is absent; run `cargo mutants` first")
    if not (OUT / "caught.txt").exists() and not (OUT / "missed.txt").exists():
        sys.exit("mutation_gate: mutants.out/ holds no results; the sweep did not run")


def main() -> int:
    update = "--update" in sys.argv[1:]
    _require_run()
    measured = _measured()

    if update:
        BASELINE.write_text(
            json.dumps(
                {
                    "_comment": (
                        "Mutation survivors accepted as a baseline for the core "
                        "crate. The nightly ratchet fails when a survivor appears "
                        "outside this set. Regenerate with "
                        "`python scripts/mutation_gate.py --update` after a "
                        "sweep, and only ever let this set shrink. Entries are "
                        "the survivor line without its line:col position, so they "
                        "survive unrelated code motion."
                    ),
                    "survivors": sorted(measured),
                },
                indent=2,
            )
            + "\n"
        )
        print(f"mutation_gate: baseline updated to {len(measured)} survivor(s)")
        return 0

    if not BASELINE.exists():
        sys.exit("mutation_gate: no baseline; create one with --update")
    baseline = set(json.loads(BASELINE.read_text())["survivors"])

    new = sorted(measured - baseline)
    killed = sorted(baseline - measured)

    for s in killed:
        print(f"killed (was in baseline): {s}")
    if killed:
        print(
            f"mutation_gate: {len(killed)} baseline survivor(s) now caught; "
            "run --update to ratchet the baseline down"
        )

    if new:
        print()
        for s in new:
            print(f"NEW SURVIVOR: {s}")
        print(
            f"\nmutation_gate: {len(new)} mutation(s) survive that the baseline "
            "does not accept. Add a test that kills each, or, if it is an "
            "equivalent mutant, justify it and re-baseline with --update."
        )
        return 1

    print(f"mutation_gate: no new survivors ({len(measured)} known, baseline OK)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
