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

Usage:
    python scripts/branch_coverage.py <report.json>            # gate it
    python scripts/branch_coverage.py <report.json> --update   # re-record it

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

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_CANNOT_RUN = 2

#: How far under the measurement a re-recorded floor sits.
#:
#: Enough that an ordinary commit does not have to re-record, and small enough
#: that losing a file's worth of arms is caught. The same reasoning as the line
#: and region floors in `ci.yml`, and the same number.
SLACK = 5.0


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


def _reading(args: list[str]) -> tuple[float, int, int] | None:
    """Give the measurement, or say why it could not be read.

    Separate from the gate below because an unreadable measurement is not a
    verdict: it has its own exit code, and a caller must not be able to reach
    "did not regress" by way of "did not measure".
    """
    paths = [arg for arg in args if not arg.startswith("-")]
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


def main() -> int:
    args = sys.argv[1:]
    reading = _reading(args)
    if reading is None:
        return EXIT_CANNOT_RUN
    percent, count, covered = reading

    print(f"branches: {covered} of {count} covered ({percent:.2f}%)")

    if "--update" in args:
        floor = round(percent - SLACK, 2)
        RECORD.write_text(
            json.dumps(
                {
                    "_comment": (
                        "The branch-coverage floor for the core's shipped "
                        "scope. Recorded by scripts/branch_coverage.py "
                        "--update after a measurement, held by that script "
                        "without it, and only ever moved up. A line is "
                        "covered when any part of it ran and a region when its "
                        "span did; this is the figure that counts the arms."
                    ),
                    "floor": floor,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"branch_coverage: floor recorded at {floor:.2f}%")
        return EXIT_OK

    if not RECORD.exists():
        print(f"branch_coverage: no {RECORD.name}; create one with --update")
        return EXIT_CANNOT_RUN
    floor = float(json.loads(RECORD.read_text(encoding="utf-8"))["floor"])
    if percent < floor:
        print(
            f"\nbranch_coverage: {percent:.2f}% is below the recorded floor of "
            f"{floor:.2f}%. A branch arm that was reached is not any more. "
            "Reach it, or -- if the arms it counted are gone with the code -- "
            "re-record with --update in the same change."
        )
        return EXIT_FAIL
    print(f"branch_coverage: at or above the floor of {floor:.2f}%")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
