"""Deterministic instruction-count regression gate for the core operations.

Builds the fixed core workload, runs it under cachegrind, and compares the
executed-instruction count against the committed budget. The count is identical
across runs of a given build, so the gate catches an algorithmic regression
without depending on a noisy wall clock. Shared CI runners are too variable for
a wall-clock budget; instruction count is not.

Usage:
    python scripts/perf_gate.py            # check against the budget; exit 1 over
    python scripts/perf_gate.py --update   # re-record the budget from this run

Requires valgrind on PATH and a Rust toolchain.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUDGET_FILE = ROOT / "scripts" / "perf_budget.json"
WORKLOAD = ROOT / "target" / "release" / "examples" / "perf_workload"
IREFS = re.compile(r"I\s+refs:\s*([\d,]+)")


def build_workload() -> None:
    subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "--example",
            "perf_workload",
            "-p",
            "valgebra-core",
        ],
        cwd=ROOT,
        check=True,
    )


def measure_instructions() -> int:
    result = subprocess.run(
        [
            "valgrind",
            "--tool=cachegrind",
            "--cachegrind-out-file=/dev/null",
            str(WORKLOAD),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    match = IREFS.search(result.stderr)
    if match is None:
        print("could not find an instruction count in cachegrind output:")
        print(result.stderr)
        raise SystemExit(2)
    return int(match.group(1).replace(",", ""))


def main() -> int:
    update = "--update" in sys.argv[1:]
    build_workload()
    measured = measure_instructions()

    budget = json.loads(BUDGET_FILE.read_text(encoding="utf-8"))
    if update:
        budget["core_workload_irefs"] = measured
        BUDGET_FILE.write_text(json.dumps(budget, indent=2) + "\n", encoding="utf-8")
        print(f"recorded budget: {measured:,} instructions")
        return 0

    recorded = int(budget["core_workload_irefs"])
    tolerance = float(budget["tolerance"])
    ceiling = int(recorded * (1 + tolerance))
    delta = (measured - recorded) / recorded

    print(f"measured: {measured:,} instructions")
    print(f"budget:   {recorded:,} (+{tolerance:.0%} -> ceiling {ceiling:,})")
    print(f"delta:    {delta:+.2%}")

    if measured > ceiling:
        print("REGRESSION: instruction count exceeds the budget ceiling.")
        return 1
    print("OK: within budget.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
