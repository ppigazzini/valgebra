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
    python scripts/mutation_gate.py --new-only          # gate a PARTIAL sweep

`--new-only` checks the new-survivor direction alone. It is for a sweep that
covers a subset of the mutants the baseline was recorded over -- an `--in-diff`
run on a pull request -- where every accepted survivor the diff did not touch is
absent by construction and would otherwise read as a whole baseline gone stale.
A full sweep must never use it: the expiry direction is what keeps the accepted
set honest.

Reads `mutants.out/missed.txt` (survivors) and `mutants.out/timeout.txt`
(unjudged, treated as survivors) produced by a prior `cargo mutants` run; a
different output directory is named with `--out`.

Three outcomes, three exit codes, so a caller can dispatch on them:

* **0** the ratchet ran and the baseline holds;
* **1** the ratchet ran and it does not -- a new survivor, or a stale entry;
* **2** the ratchet **could not run** -- no sweep output, no baseline. A gate
  that could not run has proven nothing, and must not read as one that passed.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Three outcomes, three exit codes; see the module docstring.
EXIT_OK = 0
EXIT_FAIL = 1
EXIT_CANNOT_RUN = 2

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


def _cannot_run(message: str) -> None:
    print(f"mutation_gate: {message}", file=sys.stderr)
    raise SystemExit(EXIT_CANNOT_RUN)


def _require_run(out: Path) -> None:
    # A missing output dir, or a run that generated nothing, is a broken
    # detector, not a clean tree: refuse to pass rather than report success, and
    # exit 2 so "could not measure" is distinguishable from "a survivor
    # appeared".
    if not out.is_dir():
        _cannot_run(f"{out.name}/ is absent; run `cargo mutants` first")
    if not (out / "caught.txt").exists() and not (out / "missed.txt").exists():
        _cannot_run(f"{out.name}/ holds no results; the sweep did not run")


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
    new_only = "--new-only" in args
    out = ROOT / _option(args, "--out", "mutants.out")
    _require_run(out)
    measured = _measured(out)

    if update:
        # Carry forward every hand-written note beside the set. The argument for
        # why a survivor is accepted lives in one of these, and a re-record that
        # dropped it would leave the accepted set with no reason behind it.
        notes = {}
        if baseline_file.exists():
            existing = json.loads(baseline_file.read_text())
            notes = {
                key: value
                for key, value in existing.items()
                if key.startswith("_") and key != "_comment"
            }
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
                    **notes,
                    "survivors": sorted(measured),
                },
                indent=2,
            )
            + "\n"
        )
        print(f"mutation_gate: baseline updated to {len(measured)} survivor(s)")
        return EXIT_OK

    if not baseline_file.exists():
        _cannot_run(f"no {which} baseline; create one with --update")
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
        return EXIT_FAIL

    # The baseline is an accepted hole, and an accepted hole expires in its own
    # direction: an entry the tests now catch, or one naming a mutation the sweep
    # no longer generates, is a claim about a gap the tree does not have. Left
    # standing it silently re-accepts a future survivor with the same identity,
    # so it fails here and the fix is one `--update`.
    #
    # A partial sweep cannot judge this direction -- it never generated most of
    # the baseline -- so it opts out by name rather than by the gate guessing.
    if killed and not new_only:
        for s in killed:
            print(f"STALE BASELINE ENTRY: {s}")
        print(
            f"\nmutation_gate: {len(killed)} baseline survivor(s) are no longer "
            "survivors -- caught by a test, or no longer generated. Ratchet the "
            "baseline down with --update; it must only ever shrink."
        )
        return EXIT_FAIL

    scope = "in this diff" if new_only else "known"
    print(f"mutation_gate: no new survivors ({len(measured)} {scope}, baseline OK)")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
