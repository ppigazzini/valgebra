"""Mutation survivor ratchet.

Compares the survivors of a `cargo mutants` run against a committed baseline and
fails when a *new* survivor appears -- a mutation the tests no longer catch. It
never targets zero survivors: equivalent mutants exist and are undecidable in
general, so the honest gate is "no regression", not "no survivors".

The baseline is a set of survivor identities, each the survivor line with its
`:LINE:COL` position stripped, so a survivor is matched by file, function, and
mutation rather than by a line number that drifts when unrelated code moves.

One sweep, one baseline. The core crate and the membership walk are swept by
separate commands with separate test harnesses, so each carries its own accepted
set and neither can absorb the other's survivors.

Usage:
    python scripts/mutation_gate.py                     # gate the core sweep
    python scripts/mutation_gate.py --update            # re-record it
    python scripts/mutation_gate.py --baseline walk     # gate the walk sweep

Reads `mutants.out/missed.txt` (survivors) and `mutants.out/timeout.txt`
(unjudged, treated as survivors) produced by a prior `cargo mutants` run; a
different output directory is named with `--out`.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# One baseline per sweep, named by what the sweep covers.
BASELINES = {
    "core": ROOT / "scripts" / "mutation_baseline.json",
    "walk": ROOT / "scripts" / "mutation_baseline_walk.json",
}

# `path:LINE:COL: description` -> `path: description`
_POS = re.compile(r"^(?P<path>.+?):\d+:\d+:\s*(?P<desc>.*)$")


def _identity(line: str) -> str:
    line = line.strip()
    m = _POS.match(line)
    return f"{m['path']}: {m['desc']}" if m else line


def _read(out: Path, name: str) -> list[str]:
    f = out / name
    if not f.exists():
        return []
    return [ln for ln in f.read_text().splitlines() if ln.strip()]


def _measured(out: Path) -> set[str]:
    return {
        _identity(ln) for ln in _read(out, "missed.txt") + _read(out, "timeout.txt")
    }


def _require_run(out: Path) -> None:
    # A missing output dir, or a run that generated nothing, is a broken
    # detector, not a clean tree: refuse to pass rather than report success.
    if not out.is_dir():
        sys.exit(f"mutation_gate: {out.name}/ is absent; run `cargo mutants` first")
    if not (out / "caught.txt").exists() and not (out / "missed.txt").exists():
        sys.exit(f"mutation_gate: {out.name}/ holds no results; the sweep did not run")


def _option(args: list[str], name: str, default: str) -> str:
    if name in args:
        index = args.index(name)
        if index + 1 < len(args):
            return args[index + 1]
        sys.exit(f"mutation_gate: {name} needs a value")
    return default


def main() -> int:
    args = sys.argv[1:]
    update = "--update" in args
    which = _option(args, "--baseline", "core")
    if which not in BASELINES:
        sys.exit(
            f"mutation_gate: unknown baseline {which!r}; one of {sorted(BASELINES)}"
        )
    baseline_file = BASELINES[which]
    out = ROOT / _option(args, "--out", "mutants.out")
    _require_run(out)
    measured = _measured(out)

    if update:
        baseline_file.write_text(
            json.dumps(
                {
                    "_comment": (
                        f"Mutation survivors accepted as a baseline for the "
                        f"{which} sweep. The nightly ratchet fails when a "
                        "survivor appears outside this set, and when an entry "
                        "is no longer a survivor. Regenerate with "
                        f"`python scripts/mutation_gate.py --baseline {which} "
                        "--update` after a sweep, and only ever let this set "
                        "shrink. Entries are the survivor line without its "
                        "line:col position, so they survive unrelated code "
                        "motion."
                    ),
                    "survivors": sorted(measured),
                },
                indent=2,
            )
            + "\n"
        )
        print(f"mutation_gate: baseline updated to {len(measured)} survivor(s)")
        return 0

    if not baseline_file.exists():
        sys.exit(f"mutation_gate: no {which} baseline; create one with --update")
    baseline = set(json.loads(baseline_file.read_text())["survivors"])

    new = sorted(measured - baseline)
    killed = sorted(baseline - measured)

    if new:
        for s in new:
            print(f"NEW SURVIVOR: {s}")
        print(
            f"\nmutation_gate: {len(new)} mutation(s) survive that the baseline "
            "does not accept. Add a test that kills each, or, if it is an "
            "equivalent mutant, justify it and re-baseline with --update."
        )
        return 1

    # The baseline is an accepted hole, and an accepted hole expires in its own
    # direction: an entry the tests now catch, or one naming a mutation the sweep
    # no longer generates, is a claim about a gap the tree does not have. Left
    # standing it silently re-accepts a future survivor with the same identity,
    # so it fails here and the fix is one `--update`.
    if killed:
        for s in killed:
            print(f"STALE BASELINE ENTRY: {s}")
        print(
            f"\nmutation_gate: {len(killed)} baseline survivor(s) are no longer "
            "survivors -- caught by a test, or no longer generated. Ratchet the "
            "baseline down with --update; it must only ever shrink."
        )
        return 1

    print(f"mutation_gate: no new survivors ({len(measured)} known, baseline OK)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
