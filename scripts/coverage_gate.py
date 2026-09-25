"""Per-file coverage floor.

The merge path's floors are **totals**: `--fail-under-lines` and
`--fail-under-regions` are read against the crate, so a file may sit far under
either as long as the rest of the crate carries it. That is the right shape for
a ratchet and the wrong shape for a detector, and the difference is not
academic -- a module was added to the core reading 84% of lines and 86% of
regions, worst in the crate by twelve points, and every merge-path gate stayed
green because the total moved by a hundredth of a percent.

`scripts/branch_coverage.py` already holds a region floor **per file**, and it
is what reported that module. It runs on the nightly lane, on a pinned nightly
toolchain, so it answered hours after the change had merged. This is the same
question asked where the change is: on the merge path, on the toolchain the
lane already has. The two are not duplicates -- that one ratchets a recorded
figure and moves with the measurement, this one is a fixed floor a file either
clears or is named under -- and neither makes the other redundant.

So this reads the merge path's own report per file. It is deliberately not a second
ratchet: the floor is low, far below what the crate is held to, because what it
detects is a file nothing drives rather than a file that could be driven
harder. A file below it is either a gap to fill or an entry in
[`BELOW_THE_FLOOR`] with the reason it is there, and the entry expires on its
own -- a file that climbs back over the floor fails until its excuse is
deleted.

Each lane measures a scope of its own -- the core's, the binding's -- so the
files named under the floor are named per scope: an excuse belongs to the lane
that measures the file, and a lane that does not measure it must not be asked
to account for it.

Usage:
    python scripts/coverage_gate.py --json cov.json --scope core
    python scripts/coverage_gate.py --json cov.json --scope binding

The JSON is llvm-cov's export, which `cargo llvm-cov report --json` writes.

Three outcomes, three exit codes, the same vocabulary the other gates answer
in:

* **0** every file is over the floor, or excused with a reason that holds;
* **1** a file is under it, or an excuse has outlived what it excused;
* **2** the gate could not run -- no report, or one it cannot read.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import NoReturn

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_CANNOT_RUN = 2

ROOT = Path(__file__).resolve().parent.parent

#: The floor per scope, in percent. Low on purpose: see the module docstring.
#: A file under it is one the tests reach incidentally or not at all, which is
#: a different finding from a file that could be covered better.
#:
#: Per scope for the reason the crate floors already are -- the binding is held
#: to 96/93 where the core is held to 98/97 -- and then lower again by more
#: than a figure moves between machines. The same file read 87.65% of regions
#: here and 84.77% on a runner, a spread of nearly three points, so a floor set
#: within three points of a measurement reports the machine rather than the
#: tests. What it has to catch is a file nothing drives, which reads near zero.
#: Each entry is the pair `(lines, regions)` the scope is held to.
FLOORS: dict[str, tuple[float, float]] = {
    "core": (90.0, 85.0),
    "binding": (88.0, 78.0),
}

#: The fewest files a report has to name before its reading is believed. A
#: filter that matched nothing writes a report that passes having read none,
#: which is the one way this gate can be green and mean nothing.
LEAST_FILES = 5

#: Files under the floor, per scope, each with the reason it is there. The list
#: may only shrink: a file that climbs over the floor fails here until its entry
#: goes, so an excuse cannot outlive what it excused.
#:
#: Keyed by scope because the two lanes measure two sets of files. One table
#: over both would ask each lane to account for the other's excuses, and the
#: expiry check -- an entry naming a file the report does not carry -- would
#: fire on every entry belonging to the other lane.
#:
#: A path is relative to the repository root, as the report spells it.
BELOW_THE_FLOOR: dict[str, dict[str, str]] = {
    "core": {
        "crates/valgebra-core/src/violation.rs": (
            "the file is 56 regions, and 14 of them are the error arm of a "
            "`write!` into a `String`, which `fmt::Write` cannot fail -- so the "
            "count holds a branch no value reaches and each one costs 1.8 "
            "points. The lines the arms sit on are covered; what is not is a "
            "path that does not exist"
        ),
    },
    "binding": {},
}


def _cannot_run(message: str) -> NoReturn:
    print(f"coverage_gate: {message}", file=sys.stderr)
    raise SystemExit(EXIT_CANNOT_RUN)


def _option(args: list[str], name: str, default: str) -> str:
    """Read `--name value`, or give the default."""
    if name not in args:
        return default
    index = args.index(name)
    if index + 1 >= len(args):
        _cannot_run(f"{name} needs a value")
    return args[index + 1]


def _relative(filename: str) -> str:
    """Spell a report's absolute path the way an entry here does."""
    try:
        return str(Path(filename).resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return filename


def read_report(path: Path) -> dict[str, tuple[float, float]]:
    """Give each file's line and region percentage, by path.

    llvm-cov's export nests one export object per binary; a file measured by
    more than one is reported once per binary, and the highest reading is the
    one a merged profile supports -- taking the lowest would report a file as
    uncovered because some other binary never linked it.
    """
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        _cannot_run(f"cannot read {path}: {error}")
    except json.JSONDecodeError as error:
        _cannot_run(f"{path} is not the JSON llvm-cov writes: {error}")

    found: dict[str, tuple[float, float]] = {}
    for export in report.get("data", []):
        for entry in export.get("files", []):
            summary = entry.get("summary") or {}
            lines = summary.get("lines") or {}
            regions = summary.get("regions") or {}
            if "percent" not in lines or "percent" not in regions:
                continue
            name = _relative(entry["filename"])
            reading = (float(lines["percent"]), float(regions["percent"]))
            held = found.get(name)
            found[name] = reading if held is None else max(held, reading)
    if not found:
        _cannot_run(f"{path} names no file with a summary")
    return found


def failures(
    measured: dict[str, tuple[float, float]],
    *,
    floor_lines: float,
    floor_regions: float,
    excused: dict[str, str],
) -> list[str]:
    """Report every file under the floor and every excuse that has expired."""
    problems: list[str] = []
    for name, (lines, regions) in sorted(measured.items()):
        under = lines < floor_lines or regions < floor_regions
        is_excused = name in excused
        if under and not is_excused:
            problems.append(
                f"{name}: {lines:.2f}% of lines and {regions:.2f}% of regions, "
                f"under the floor of {floor_lines:.0f}/{floor_regions:.0f}. "
                "Drive it, or record it in BELOW_THE_FLOOR with the reason."
            )
        if is_excused and not under:
            problems.append(
                f"{name}: {lines:.2f}% of lines and {regions:.2f}% of regions, "
                "over the floor, and BELOW_THE_FLOOR excuses it. Delete the "
                "entry with the argument beside it."
            )
    problems.extend(
        f"{name}: excused and absent from the report. Delete the entry, "
        "or fix the path it names."
        for name in sorted(excused)
        if name not in measured
    )
    return problems


def main() -> int:
    args = sys.argv[1:]
    report = Path(_option(args, "--json", "coverage.json"))
    scope = _option(args, "--scope", "core")
    if scope not in FLOORS or scope not in BELOW_THE_FLOOR:
        _cannot_run(f"--scope {scope} names no lane; try {sorted(FLOORS)}")
    lines, regions = FLOORS[scope]
    floor_lines = float(_option(args, "--lines", str(lines)))
    floor_regions = float(_option(args, "--regions", str(regions)))
    excused = BELOW_THE_FLOOR[scope]

    measured = read_report(report)
    # The report is the detector: an empty one would pass having read nothing,
    # and a filter that matched no file writes exactly that.
    if len(measured) < LEAST_FILES:
        _cannot_run(f"{report} names only {sorted(measured)}")

    problems = failures(
        measured,
        floor_lines=floor_lines,
        floor_regions=floor_regions,
        excused=excused,
    )
    if problems:
        print(
            f"coverage_gate: {len(problems)} file(s) the per-file floor refuses:",
            file=sys.stderr,
        )
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return EXIT_FAIL

    print(
        f"coverage_gate: {scope}: {len(measured)} file(s) over "
        f"{floor_lines:.0f}% of lines and {floor_regions:.0f}% of regions"
        + (f", {len(excused)} excused by name" if excused else "")
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
