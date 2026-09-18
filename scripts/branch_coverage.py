"""Hold the branch-coverage figure to a floor that only ever moves up.

A line is covered when any part of it ran and a region when its span did, so a
two-armed branch passes both having taken one arm wherever the arms are not
separate spans. The branch figure counts the arms, and on the core's shipped
scope it reads several points below either of the others -- which is the gap a
line or region floor cannot see.

`cargo llvm-cov` has no `--fail-under-branches`, so the number is recorded here
and ratcheted the way `scripts/mutation_gate.py` ratchets survivors: a lane
measures it, this compares the measurement against `scripts/branch_coverage.json`,
and the floor moves up with the measurement and never ahead of it.

**And one region floor per file, from the same report.** A total absorbs a hole
the size of a file: half of the decision procedure going unreached moves a
scope-wide figure by a few points, which is the width of the tolerance any
figure carries between machines. So each file carries its own floor, and a file
that loses its tests fails under its own name rather than under a fraction of a
point in a number about everything.

Usage:
    python scripts/branch_coverage.py <report.json>            # gate it
    python scripts/branch_coverage.py <report.json> --update   # re-record it
    python scripts/branch_coverage.py <report.json> --record <path>

Three outcomes, three exit codes: **0** the measurement is at or above the
floor, **1** it is below, **2** the measurement could not be read -- a missing
or unparseable report, which is "did not measure" rather than "did not
regress".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RECORD = ROOT / "scripts" / "branch_coverage.json"


def _record(args: list[str]) -> Path:
    """Give the file the floors are read from and written to.

    Overridable so a test can put the gate to a record of its own: the per-file
    floors are a scope, and a synthesised report of one file against the tree's
    thirty is a scope mismatch rather than a regression.
    """
    if "--record" in args:
        return Path(args[args.index("--record") + 1])
    return RECORD


EXIT_OK = 0
EXIT_FAIL = 1
EXIT_CANNOT_RUN = 2

#: How far under the measurement a re-recorded floor sits.
#:
#: Enough that an ordinary commit does not have to re-record, and small enough
#: that losing a file's worth of arms is caught. The same reasoning as the line
#: and region floors in `ci.yml`, and the same number.
SLACK = 1.0

#: How far under the measurement a re-recorded *file* floor sits.
#:
#: Wider than the scope-wide one, because a single file's figure swings on a
#: test moving between files and a scope-wide figure does not. What it has to
#: catch is a file losing its tests, which is tens of points rather than two.
FILE_SLACK = 8.0


def _per_file(report: Path) -> dict[str, float]:
    """Give each file's region percentage, keyed by its path below the crate.

    Regions rather than lines: a line counts as covered when any part of it
    ran, so a file whose two-armed branches all take one arm reads high in
    lines and low in regions, and the lower reading is the one a floor should
    hold. Branches would be lower still and are not per-file here, because a
    file with no branch at all has no figure and would need a rule of its own.
    """
    data = json.loads(report.read_text(encoding="utf-8"))
    found: dict[str, float] = {}
    for entry in data["data"][0]["files"]:
        regions = entry["summary"]["regions"]
        if regions["count"] == 0:
            continue
        name = entry["filename"]
        _, _, tail = name.partition("/src/")
        found[tail or name] = 100 * regions["covered"] / regions["count"]
    return found


def _measured(report: Path) -> tuple[float, int, int]:
    """Give the branch percentage, the count and the covered count.

    Read from the whole report rather than from its own `totals`, because the
    scope is set by the flags the lane passes and a file left out of the
    measurement is left out of this too.
    """
    data = json.loads(report.read_text(encoding="utf-8"))
    count = covered = 0
    for entry in data["data"][0]["files"]:
        branches = entry["summary"]["branches"]
        count += branches["count"]
        covered += branches["covered"]
    if count == 0:
        message = "the report holds no branches"
        raise ValueError(message)
    return 100 * covered / count, count, covered


def _report_paths(args: list[str]) -> list[str]:
    """Give the positional arguments, with `--record`'s value not among them."""
    skip = args.index("--record") + 1 if "--record" in args else -1
    return [
        arg
        for index, arg in enumerate(args)
        if not arg.startswith("-") and index != skip
    ]


def _reading(args: list[str]) -> tuple[float, int, int] | None:
    """Give the measurement, or say why it could not be read.

    Separate from the gate below because an unreadable measurement is not a
    verdict: it has its own exit code, and a caller must not be able to reach
    "did not regress" by way of "did not measure".
    """
    paths = _report_paths(args)
    if not paths:
        print("branch_coverage: name the JSON report to read", file=sys.stderr)
        return None
    report = Path(paths[0])
    if not report.is_file():
        print(f"branch_coverage: {report} is not a file", file=sys.stderr)
        return None
    try:
        return _measured(report)
    except (OSError, ValueError, KeyError, IndexError) as error:
        print(f"branch_coverage: cannot read {report}: {error}", file=sys.stderr)
        return None


def _files_verdict(measured: dict[str, float], floors: dict[str, float]) -> int:
    """Hold each file to its own floor, and say which ones fell.

    Both directions. A file below its floor fails, and a floor for a file the
    scope no longer has fails too -- a stale row is a floor nothing can breach,
    and a scope that quietly stops measuring a file would otherwise look like a
    file that never regressed.
    """
    fallen = sorted(
        f"{name}: {measured[name]:.2f}% is below its floor of {floors[name]:.2f}%"
        for name in measured
        if name in floors and measured[name] < floors[name]
    )
    unmeasured = sorted(set(floors) - set(measured))
    unrecorded = sorted(set(measured) - set(floors))
    if fallen:
        print("\n" + "\n".join(f"  {row}" for row in fallen))
        print(
            "\nbranch_coverage: a file lost regions its tests used to reach. A "
            "total absorbs a hole the size of a file, which is why each carries "
            "its own floor. Reach them, or re-record with --update in the same "
            "change."
        )
        return EXIT_FAIL
    if unmeasured:
        print(
            "\nbranch_coverage: floors for files the measurement does not "
            f"carry: {unmeasured}. Either the scope changed or the file is "
            "gone; re-record with --update in the same change."
        )
        return EXIT_FAIL
    if unrecorded:
        print(
            "\nbranch_coverage: files the measurement carries and no floor "
            f"does: {unrecorded}. Re-record with --update in the same change."
        )
        return EXIT_FAIL
    print(f"branch_coverage: {len(measured)} file(s) at or above their own floor")
    return EXIT_OK


def main() -> int:
    args = sys.argv[1:]
    record_path = _record(args)
    reading = _reading(args)
    if reading is None:
        return EXIT_CANNOT_RUN
    percent, count, covered = reading
    paths = _report_paths(args)
    measured = _per_file(Path(paths[0]))

    print(f"branches: {covered} of {count} covered ({percent:.2f}%)")

    if "--update" in args:
        floor = round(percent - SLACK, 2)
        record_path.write_text(
            json.dumps(
                {
                    "_comment": (
                        "The branch-coverage floor for the core's shipped "
                        "scope, and a region floor per file beside it. "
                        "Recorded by scripts/branch_coverage.py --update after "
                        "a measurement, held by that script without it, and "
                        "only ever moved up. A line is covered when any part "
                        "of it ran and a region when its span did; the scope "
                        "figure counts the arms. The per-file floors are "
                        "regions, and they are what a total cannot see: half "
                        "of one file unreached moves a scope-wide figure by "
                        "about the width of its tolerance."
                    ),
                    "floor": floor,
                    "files": {
                        name: round(percent - FILE_SLACK, 2)
                        for name, percent in sorted(measured.items())
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            f"branch_coverage: floor recorded at {floor:.2f}%, and "
            f"{len(measured)} file floor(s) beside it"
        )
        return EXIT_OK

    if not record_path.exists():
        print(f"branch_coverage: no {record_path.name}; create one with --update")
        return EXIT_CANNOT_RUN
    record = json.loads(record_path.read_text(encoding="utf-8"))
    floor = float(record["floor"])
    if percent < floor:
        print(
            f"\nbranch_coverage: {percent:.2f}% is below the recorded floor of "
            f"{floor:.2f}%. A branch arm that was reached is not any more. "
            "Reach it, or -- if the arms it counted are gone with the code -- "
            "re-record with --update in the same change."
        )
        return EXIT_FAIL
    print(f"branch_coverage: at or above the floor of {floor:.2f}%")
    return _files_verdict(measured, {k: float(v) for k, v in record["files"].items()})


if __name__ == "__main__":
    raise SystemExit(main())
