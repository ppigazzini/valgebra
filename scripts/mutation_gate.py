"""Mutation survivor ratchet.

Compares the survivors of a `cargo mutants` run against a committed baseline and
fails when a *new* survivor appears -- a mutation the tests no longer catch. It
never targets zero survivors: equivalent mutants exist and are undecidable in
general, so the honest gate is "no regression", not "no survivors".

The baseline is a set of survivor identities, each the survivor line with its
`:LINE:COL` position stripped, so a survivor is matched by file, function, and
mutation rather than by a line number that drifts when unrelated code moves.
Where one function holds several mutants of one description -- the two `||` of
one condition -- the description names none of them, and an entry accepting one
accepted all: so each carries its place among them as well, `#2` for the second
in the source. The places are read from every mutant the sweep generated, which
the four outcome files list between them; a sweep that holds only part of a
function's mutants -- one shard -- names the whole listing with `--all`.

One sweep, one baseline. The core crate and the membership walk are swept by
separate commands with separate test harnesses, so each carries its own accepted
set and neither can absorb the other's survivors.

Usage:
    python scripts/mutation_gate.py                     # gate the core sweep
    python scripts/mutation_gate.py --update            # re-record it
    python scripts/mutation_gate.py --baseline walk     # gate the walk sweep
    python scripts/mutation_gate.py --new-only          # gate a PARTIAL sweep
    python scripts/mutation_gate.py --new-only --all listed.txt   # ... a shard

`--new-only` checks the new-survivor direction alone. It is for a sweep that
covers a subset of the mutants the baseline was recorded over -- an `--in-diff`
run on a pull request -- where every accepted survivor the diff did not touch is
absent by construction and would otherwise read as a whole baseline gone stale.
A full sweep must never use it: the expiry direction is what keeps the accepted
set honest.

Reads `mutants.out/missed.txt` (survivors) produced by a prior `cargo mutants`
run; a different output directory is named with `--out`.
`mutants.out/timeout.txt` is read too and is *not* a survivor: a mutant whose
experiment never finished returned no verdict, which is a **rig fault** -- the
glossary's word -- and it stops the run at exit 2 rather than being counted as a
hole in the tests. The two are repaired in opposite directions, and folding them
together lets `--update` write an overloaded runner into the baseline as a
permanent excuse for a mutant nobody judged.

Three outcomes, three exit codes, so a caller can dispatch on them:

* **0** the ratchet ran and the baseline holds;
* **1** the ratchet ran and it does not -- a new survivor, or a stale entry;
* **2** the ratchet **could not run** -- no sweep output, no baseline. A gate
  that could not run has proven nothing, and must not read as one that passed.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

# Three outcomes, three exit codes; see the module docstring.
EXIT_OK = 0
EXIT_FAIL = 1
EXIT_CANNOT_RUN = 2

ROOT = Path(__file__).resolve().parent.parent
# One baseline per sweep, named by what the sweep covers.
BASELINES = {
    "core": ROOT / "scripts" / "mutation_baseline.json",
    "walk": ROOT / "scripts" / "mutation_baseline_walk.json",
    # The files the shipped extension is the only caller of, swept with a test
    # command that loads it. Its own set, for the reason the other two have
    # theirs: a different command with a different harness, and neither can
    # absorb the other's survivors.
    "pytest": ROOT / "scripts" / "mutation_baseline_pytest.json",
}

# `path:LINE:COL: description` -> `path: description`
_POS = re.compile(r"^(?P<path>.+?):(?P<line>\d+):(?P<col>\d+):\s*(?P<desc>.*)$")

# Every verdict a sweep writes; between them they list each mutant it generated.
_OUTCOMES = ("caught.txt", "missed.txt", "timeout.txt", "unviable.txt")


def identities(listed: Iterable[str]) -> dict[str, str]:
    """Name each listed mutant as a baseline records it.

    A mutant is its file and its description, without the position, and where
    the listing holds several of one file and description -- one function's two
    `||` -- each also takes its place among them by line and column, `#1` first.
    A line that carries no position is its own name.
    """
    siblings: dict[str, list[tuple[int, int, str]]] = {}
    names: dict[str, str] = {}
    for line in listed:
        stripped = line.strip()
        m = _POS.match(stripped)
        if not m:
            names[stripped] = stripped
            continue
        place = (int(m["line"]), int(m["col"]), stripped)
        siblings.setdefault(f"{m['path']}: {m['desc']}", []).append(place)
    for name, places in siblings.items():
        ordered = sorted(set(places))
        for rank, (_, _, stripped) in enumerate(ordered, start=1):
            names[stripped] = name if len(ordered) == 1 else f"{name} #{rank}"
    return names


def _read(out: Path, name: str) -> list[str]:
    f = out / name
    if not f.exists():
        return []
    return [ln.strip() for ln in f.read_text().splitlines() if ln.strip()]


def _named(lines: list[str], names: dict[str, str]) -> list[str]:
    """Name a sweep's lines by the listing, refusing one the listing lacks.

    A line missing from the listing is a listing of some other tree or some
    other sweep, and a place read from it would name a different mutant.
    """
    missing = [line for line in lines if line not in names]
    if missing:
        for line in missing:
            print(f"NOT IN THE LISTING: {line}", file=sys.stderr)
        _cannot_run(
            f"{len(missing)} mutant(s) are not in the listing their places are "
            "read from; name the listing of this sweep's own files with --all"
        )
    return [names[line] for line in lines]


def _listed(out: Path, listing: str | None) -> list[str]:
    """Give every mutant the sweep was cut from, which places are read among.

    The outcome files list each mutant the sweep generated once between them,
    so a whole sweep needs nothing else. A shard holds part of each function's
    mutants and names the whole listing.
    """
    if listing is None:
        return [line for name in _OUTCOMES for line in _read(out, name)]
    path = Path(listing)
    if not path.is_file():
        _cannot_run(f"{listing} is absent; list the sweep's mutants first")
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def _measured(out: Path, names: dict[str, str]) -> set[str]:
    """Give the mutants the sweep judged to survive."""
    return set(_named(_read(out, "missed.txt"), names))


def _unjudged(out: Path) -> list[str]:
    """Give the mutants whose experiment never returned a verdict.

    Read apart from the survivors, because they are a different claim. A
    survivor says the tests do not catch a change; a timeout says the *run* did
    not finish, which is about the runner and is repaired in the other
    direction -- a budget, a seed, a bound on a shrink. Folded together, an
    overloaded machine reads as a hole in the tests, and `--update` writes that
    reading into the baseline as a permanent excuse for a mutant nobody judged.
    """
    return _read(out, "timeout.txt")


def _cannot_run(message: str) -> None:
    print(f"mutation_gate: {message}", file=sys.stderr)
    raise SystemExit(EXIT_CANNOT_RUN)


def _refuse_a_rig_fault(out: Path) -> None:
    """Stop where a mutant returned no verdict, before any ratchet is read.

    Before, because every direction below is a comparison against the measured
    set, and a set missing the mutants nobody judged is not the sweep's answer.
    """
    unjudged = _unjudged(out)
    if not unjudged:
        return
    for mutant in unjudged:
        print(f"RIG FAULT: {mutant}", file=sys.stderr)
    _cannot_run(
        f"{len(unjudged)} mutant(s) returned no verdict: a rig fault rather than "
        "a result. Raise the timeout multiplier, bound what the tests shrink, or "
        "give the sweep a machine that finishes them -- and read the ratchet "
        "again once they are judged."
    )


def _require_run(out: Path) -> None:
    # A missing output dir, or a run that generated nothing, is a broken
    # detector, not a clean tree: refuse to pass rather than report success, and
    # exit 2 so "could not measure" is distinguishable from "a survivor
    # appeared".
    if not out.is_dir():
        _cannot_run(f"{out.name}/ is absent; run `cargo mutants` first")
    if not (out / "caught.txt").exists() and not (out / "missed.txt").exists():
        _cannot_run(f"{out.name}/ holds no results; the sweep did not run")


def _arguments(argv: Sequence[str]) -> argparse.Namespace:
    """Read the command line, refusing a flag it does not know.

    An unknown flag, a missing value and an unknown baseline are each a gate
    that could not run, so each exits 2 -- `argparse`'s own code for them.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--update", action="store_true", help="re-record the baseline")
    parser.add_argument("--baseline", choices=sorted(BASELINES), default="core")
    parser.add_argument(
        "--new-only", action="store_true", help="gate a partial sweep's new survivors"
    )
    parser.add_argument("--out", default="mutants.out", help="the sweep's output")
    parser.add_argument(
        "--all",
        default=None,
        help="every mutant the sweep was cut from, as `cargo mutants --list` "
        "prints them, for a sweep holding part of them",
    )
    return parser.parse_args(argv)


def _stale_entries(recorded: dict, baseline: set[str]) -> bool:
    """Report both ways an entry can outlive what it records.

    Not short-circuited: a move that also orphans a note should say both, since
    a reader fixing one would otherwise be told about the other on the next run.
    """
    absent = _keys_for_absent_files(baseline)
    orphans = _orphan_notes(recorded, baseline)
    return absent or orphans


def _keys_for_absent_files(baseline: set[str]) -> bool:
    """Report accepted survivors whose file is not in the tree, and refuse them.

    This baseline is keyed by path, so a file that moves takes every entry
    naming it out of the set. The sweep then reads each of that file's
    survivors as *new* and the lane fails -- with the mutants listed and no hint
    that what changed was the path. Eight entries moved with the frontend's
    surfaces and the lane is where that was found, nine minutes into a shard.

    A path is cheaper to check than a sweep: an entry naming a file that does
    not exist is stale by construction, whatever any sweep would say, and this
    runs before one. Read from `git ls-files` rather than the filesystem so a
    build directory or a stray copy cannot answer for a tracked file.

    Keying by something other than the path was considered and is not done: two
    functions of one name in two files would share a key, and an argument for
    one would silently excuse the other.
    """
    listed = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z"],
        capture_output=True,
        text=True,
        check=False,
    )
    tracked = {name for name in listed.stdout.split("\0") if name}
    if not tracked:
        return False  # not a checkout: the sweep's own reading still stands
    absent = sorted(
        {key.split(":", 1)[0] for key in baseline} - tracked,
    )
    for path in absent:
        print(f"ACCEPTED FOR A FILE THAT IS NOT HERE: {path}")
    if absent:
        print(
            f"\nmutation_gate: {len(absent)} accepted survivor(s) name a file "
            "this tree does not track. A file that moved takes its entries with "
            "it: re-key each to the path the code has now, keeping the argument "
            "it was accepted under."
        )
    return bool(absent)


def _orphan_notes(recorded: dict, baseline: set[str]) -> bool:
    """Report accepted notes naming no survivor, and whether any were found.

    An accepted survivor carries a hand-written argument beside it, and nothing
    read those arguments: a note could name a mutant that stopped surviving, or
    name one in a spelling no survivor takes, and the set passed either way.
    Both were true of this tree. An argument matching nothing is an excuse
    outliving the thing it excused, which is what the ratchet exists to refuse.
    """
    orphans = sorted(
        key for key in recorded.get("_accepted", {}) if key not in baseline
    )
    for key in orphans:
        print(f"ACCEPTED WITHOUT A SURVIVOR: {key}")
    if orphans:
        print(
            f"\nmutation_gate: {len(orphans)} accepted note(s) name no survivor "
            "in this baseline. Delete each whose mutant a test kills, and spell "
            "the rest as the survivor line spells them."
        )
    return bool(orphans)


def main(argv: Sequence[str] = ()) -> int:
    args = _arguments(argv)
    update = args.update
    which = args.baseline
    baseline_file = BASELINES[which]
    new_only = args.new_only
    out = ROOT / args.out
    _require_run(out)
    _refuse_a_rig_fault(out)
    measured = _measured(out, identities(_listed(out, args.all)))

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
                        "motion, and with its place among the mutants of one "
                        "description in one function, `#2`, where there are "
                        "several."
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
    recorded = json.loads(baseline_file.read_text())
    baseline = set(recorded["survivors"])

    if _stale_entries(recorded, baseline):
        return EXIT_FAIL

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
    raise SystemExit(main(sys.argv[1:]))
