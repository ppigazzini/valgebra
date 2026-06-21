"""Deterministic instruction-count regression gate.

Builds a fixed workload, runs it under cachegrind, and compares the
executed-instruction count against the committed budget. The count is identical
across runs of a given build, so the gate catches an algorithmic regression
without depending on a noisy wall clock. Shared CI runners are too variable for
a wall-clock budget; instruction count is not.

Two workloads:

* The default **core** workload (`perf_workload`, pure Rust) measures the schema
  operations and is fully deterministic, so its budget is tight.
* The **binding** workload (`--binding`) measures the membership walk over a live
  Python value -- the shipped hot path the core workload does not reach. It embeds
  CPython, whose startup is not a fixed instruction count, so the gate measures
  the *difference* between two iteration counts: startup cancels, leaving the
  deterministic per-iteration walk cost. Its budget carries a wider tolerance to
  absorb cross-interpreter FFI variance while still catching a per-node regression
  (the ``ctx.fatal.borrow`` tax), which is far larger.

Usage:
    python scripts/perf_gate.py                      # check the core budget
    python scripts/perf_gate.py --update             # re-record the core budget
    python scripts/perf_gate.py --binding            # check the binding budget
    python scripts/perf_gate.py --binding --update   # re-record it

Requires valgrind on PATH and a Rust toolchain. ``--binding`` also needs an
embedded interpreter: the build links libpython, so run it with the interpreter's
library directory on the loader path.
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
BINDING_WORKLOAD = ROOT / "target" / "release" / "examples" / "binding_workload"
# The two iteration counts whose cachegrind difference isolates the per-iteration
# binding walk cost from the fixed (cancelling) interpreter startup.
BINDING_ITERS_LOW = 50_000
BINDING_ITERS_HIGH = 150_000
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


def build_binding_workload() -> None:
    # Needs an embedded interpreter to acquire the GIL in a standalone binary.
    subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "--example",
            "binding_workload",
            "-p",
            "valgebra-py",
            "--features",
            "interpreter-tests",
        ],
        cwd=ROOT,
        check=True,
    )


def measure_instructions(binary: Path, *args: str) -> int:
    result = subprocess.run(
        [
            "valgrind",
            "--tool=cachegrind",
            "--cachegrind-out-file=/dev/null",
            str(binary),
            *args,
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


def measure_binding_difference() -> int:
    # The walk's per-iteration cost, isolated by subtracting two runs so the
    # embedded interpreter's (identical) startup cancels out.
    high = measure_instructions(BINDING_WORKLOAD, str(BINDING_ITERS_HIGH))
    low = measure_instructions(BINDING_WORKLOAD, str(BINDING_ITERS_LOW))
    return high - low


def check_against_budget(measured: int, recorded: int, tolerance: float) -> int:
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


def run_core(budget: dict, *, update: bool) -> int:
    build_workload()
    measured = measure_instructions(WORKLOAD)
    if update:
        budget["core_workload_irefs"] = measured
        BUDGET_FILE.write_text(json.dumps(budget, indent=2) + "\n", encoding="utf-8")
        print(f"recorded core budget: {measured:,} instructions")
        return 0
    return check_against_budget(
        measured, int(budget["core_workload_irefs"]), float(budget["tolerance"])
    )


def run_binding(budget: dict, *, update: bool) -> int:
    build_binding_workload()
    measured = measure_binding_difference()
    print(
        f"binding walk over {BINDING_ITERS_HIGH - BINDING_ITERS_LOW:,} iterations "
        f"(difference of {BINDING_ITERS_HIGH:,} and {BINDING_ITERS_LOW:,} runs)"
    )
    if update:
        budget["binding_workload_irefs"] = measured
        BUDGET_FILE.write_text(json.dumps(budget, indent=2) + "\n", encoding="utf-8")
        print(f"recorded binding budget: {measured:,} instructions")
        return 0
    return check_against_budget(
        measured,
        int(budget["binding_workload_irefs"]),
        float(budget["binding_tolerance"]),
    )


def main() -> int:
    args = sys.argv[1:]
    update = "--update" in args
    budget = json.loads(BUDGET_FILE.read_text(encoding="utf-8"))
    if "--binding" in args:
        return run_binding(budget, update=update)
    return run_core(budget, update=update)


if __name__ == "__main__":
    sys.exit(main())
