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

Three refusals, because a measurement that did not happen must not read as a
verdict:

* **An unreadable measurement is not a pass.** A cachegrind run whose instruction
  count or checksum cannot be parsed exits 2 -- "could not measure", never "did
  not regress".
* **The workload must prove it did the work.** Each workload prints a checksum
  folded through every result. The core workload's is a recorded constant; the
  binding workload's is its iteration count by construction, so it is asserted as
  an identity and needs no recording. A workload whose body collapsed prints a
  different checksum and reddens *before* any count is compared.
* **The budget is two-sided.** A count far *below* the budget is not a pass
  either: a workload that stopped doing the work measures low, and a one-sided
  ceiling publishes that as an improvement it never earned. An intentional
  optimization past the floor is re-recorded with ``--update``, which is the
  ledger discipline the budget exists for.

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

# Three outcomes, three exit codes: 0 within budget, 1 outside it or the wrong
# workload, 2 could not measure.
EXIT_OK = 0
EXIT_FAIL = 1
EXIT_CANNOT_RUN = 2

ROOT = Path(__file__).resolve().parent.parent
BUDGET_FILE = ROOT / "scripts" / "perf_budget.json"
WORKLOAD = ROOT / "target" / "release" / "examples" / "perf_workload"
BINDING_WORKLOAD = ROOT / "target" / "release" / "examples" / "binding_workload"
# The two iteration counts whose cachegrind difference isolates the per-iteration
# binding walk cost from the fixed (cancelling) interpreter startup.
BINDING_ITERS_LOW = 50_000
BINDING_ITERS_HIGH = 150_000
IREFS = re.compile(r"I\s+refs:\s*([\d,]+)")
# Both workloads print one trailing line holding the checksum, bare or prefixed.
CHECKSUM = re.compile(r"^(?:checksum=)?(\d+)$")


class Measurement:
    """One cachegrind run: what it executed, and what the workload computed."""

    def __init__(self, irefs: int, checksum: int) -> None:
        self.irefs = irefs
        self.checksum = checksum


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


def parse_measurement(stdout: str, stderr: str) -> Measurement:
    """Read the instruction count and the workload's checksum, or refuse.

    Both halves are required. A run whose count is unreadable measured nothing;
    a run whose checksum is unreadable measured something that cannot be shown to
    be the workload. Neither is a verdict, so both raise rather than return.
    """
    match = IREFS.search(stderr)
    if match is None:
        print("could not find an instruction count in cachegrind output:")
        print(stderr)
        raise SystemExit(EXIT_CANNOT_RUN)
    irefs = int(match.group(1).replace(",", ""))

    checksum = None
    for line in reversed(stdout.strip().splitlines()):
        found = CHECKSUM.match(line.strip())
        if found is not None:
            checksum = int(found.group(1))
            break
    if checksum is None:
        print("could not find a workload checksum in the output:")
        print(stdout)
        raise SystemExit(EXIT_CANNOT_RUN)
    return Measurement(irefs, checksum)


def measure(binary: Path, *args: str) -> Measurement:
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
    return parse_measurement(result.stdout, result.stderr)


def check_checksum(measured: int, expected: int, subject: str) -> int:
    """Refuse a measurement whose workload did not compute what it must.

    A workload whose body collapsed still runs, still executes instructions, and
    still reports a count the budget would accept from below. The checksum is the
    only thing that says the count belongs to the work it claims to measure, so
    it is compared first and a mismatch stops the run.
    """
    if measured != expected:
        print(f"RIG FAULT: {subject} checksum {measured}, expected {expected}.")
        print("The workload did not compute what it must; the count measures")
        print("something other than the work this budget is for.")
        return 1
    print(f"checksum: {measured} ({subject}, as expected)")
    return 0


def check_against_budget(measured: int, recorded: int, tolerance: float) -> int:
    """Hold a count inside a two-sided band around the recorded budget."""
    ceiling = int(recorded * (1 + tolerance))
    floor = int(recorded * (1 - tolerance))
    delta = (measured - recorded) / recorded
    print(f"measured: {measured:,} instructions")
    print(f"budget:   {recorded:,} (+/-{tolerance:.0%} -> {floor:,} .. {ceiling:,})")
    print(f"delta:    {delta:+.2%}")
    if measured > ceiling:
        print("REGRESSION: instruction count exceeds the budget ceiling.")
        return 1
    if measured < floor:
        print("UNDER-RUN: instruction count falls below the budget floor.")
        print("Either the workload stopped doing the work it measures, or this is")
        print("a real optimization -- re-record with --update and say which.")
        return 1
    print("OK: within budget.")
    return 0


def run_core(budget: dict, *, update: bool) -> int:
    build_workload()
    result = measure(WORKLOAD)
    if update:
        budget["core_workload_irefs"] = result.irefs
        budget["core_workload_checksum"] = result.checksum
        BUDGET_FILE.write_text(json.dumps(budget, indent=2) + "\n", encoding="utf-8")
        print(f"recorded core budget: {result.irefs:,} instructions")
        print(f"recorded core checksum: {result.checksum}")
        return 0
    failed = check_checksum(
        result.checksum, int(budget["core_workload_checksum"]), "core workload"
    )
    if failed:
        return failed
    return check_against_budget(
        result.irefs, int(budget["core_workload_irefs"]), float(budget["tolerance"])
    )


def run_binding(budget: dict, *, update: bool) -> int:
    build_binding_workload()
    # The walk's per-iteration cost, isolated by subtracting two runs so the
    # embedded interpreter's (identical) startup cancels out.
    high = measure(BINDING_WORKLOAD, str(BINDING_ITERS_HIGH))
    low = measure(BINDING_WORKLOAD, str(BINDING_ITERS_LOW))
    # The workload folds one unit per successful walk, so its checksum IS its
    # iteration count. That identity is what proves the argument was honoured:
    # a workload that ignored it would report a difference near zero, which a
    # ceiling-only budget accepts and this does not.
    for result, iters in ((high, BINDING_ITERS_HIGH), (low, BINDING_ITERS_LOW)):
        failed = check_checksum(result.checksum, iters, f"binding walk x{iters:,}")
        if failed:
            return failed
    measured = high.irefs - low.irefs
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
