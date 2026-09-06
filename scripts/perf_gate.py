"""Deterministic instruction-count regression gate.

Builds a fixed workload, runs it under cachegrind, and compares the
executed-instruction count against the committed budget. The count is identical
across runs of a given build, so the gate catches an algorithmic regression
without depending on a noisy wall clock. Shared CI runners are too variable for
a wall-clock budget; instruction count is not.

Two ways to compare, and the relative one is the merge gate:

* **Against the merge base** (``--against <rev>``). The same workload is built
  and measured twice in one job -- once at ``HEAD``, once at ``rev`` in a
  throwaway worktree -- with the same toolchain, the same flags and the same
  machine, and the gate holds the *difference*. Everything an absolute budget
  cannot control cancels, so the tolerance is tight (2%) and a regression this
  change introduced is what is left.
* **Against a recorded number** (the default). Kept for the nightly lane and for
  a local reading, and it is the weaker of the two: the count is deterministic
  for a build, but not across toolchains or target features. Measured, the
  commit that recorded the current core budget re-measures 5.35% away from it on
  another machine of the same rustc line -- half the band -- so an absolute gate
  wide enough not to flake is too wide to see a real 5% regression. That is what
  ``--against`` exists to fix, and why the merge gate uses it.

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
    python scripts/perf_gate.py --against origin/main    # the merge gate
    python scripts/perf_gate.py --against HEAD~1 --decision
    python scripts/perf_gate.py                      # check the core budget
    python scripts/perf_gate.py --decision           # gate the decision procedures
    python scripts/perf_gate.py --decision --update  # re-record that budget
    python scripts/perf_gate.py --update             # re-record the core budget
    python scripts/perf_gate.py --binding            # check the binding budget
    python scripts/perf_gate.py --binding --update   # re-record it

Requires valgrind on PATH and a Rust toolchain. ``--binding`` also needs an
embedded interpreter: the build links libpython, so run it with the interpreter's
library directory on the loader path.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

# Three outcomes, three exit codes: 0 within budget, 1 outside it or the wrong
# workload, 2 could not measure.
EXIT_OK = 0
EXIT_FAIL = 1
EXIT_CANNOT_RUN = 2

ROOT = Path(__file__).resolve().parent.parent
BUDGET_FILE = ROOT / "scripts" / "perf_budget.json"
WORKLOAD = ROOT / "target" / "release" / "examples" / "perf_workload"
DECISION_WORKLOAD = ROOT / "target" / "release" / "examples" / "decision_workload"
BINDING_WORKLOAD = ROOT / "target" / "release" / "examples" / "binding_workload"
# The two iteration counts whose cachegrind difference isolates the per-iteration
# binding walk cost from the fixed (cancelling) interpreter startup.
BINDING_ITERS_LOW = 50_000
BINDING_ITERS_HIGH = 150_000
# What a change may add against its own merge base, measured in one job with one
# toolchain. Two percent is far outside the run-to-run variation of a count that
# is deterministic per build (it is zero) and far inside the smallest regression
# worth a conversation, so the band is here to absorb the one thing that does
# move: which inlining decisions the linker makes as code is added elsewhere.
RELATIVE_TOLERANCE = 0.02
IREFS = re.compile(r"I\s+refs:\s*([\d,]+)")
# Both workloads print one trailing line holding the checksum, bare or prefixed.
CHECKSUM = re.compile(r"^(?:checksum=)?(\d+)$")


@dataclass(frozen=True, slots=True)
class Measurement:
    """One cachegrind run: what it executed, and what the workload computed."""

    irefs: int
    checksum: int


def _cargo(args: list[str], root: Path, target: Path | None) -> None:
    """Run cargo in `root`, writing to `target` where one is given.

    A comparison run builds a second checkout, and the two builds must not share
    an output directory: cargo would rebuild the whole workspace each way and the
    second measurement would be of a binary the first run had already replaced.
    """
    env = None
    if target is not None:
        env = {**os.environ, "CARGO_TARGET_DIR": str(target)}
    subprocess.run(["cargo", *args], cwd=root, check=True, env=env)


def build_workload(
    example: str = "perf_workload",
    root: Path = ROOT,
    target: Path | None = None,
) -> None:
    _cargo(
        ["build", "--release", "--example", example, "-p", "valgebra-core"],
        root,
        target,
    )


def build_binding_workload(root: Path = ROOT, target: Path | None = None) -> None:
    # Needs an embedded interpreter to acquire the GIL in a standalone binary.
    _cargo(
        [
            "build",
            "--release",
            "--example",
            "binding_workload",
            "-p",
            "valgebra-py",
            "--features",
            "interpreter-tests",
        ],
        root,
        target,
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


#: The three modes, each naming the example it builds and where it lands.
MODES = {
    "core": ("perf_workload", "core workload"),
    "decision": ("decision_workload", "decision workload"),
    "binding": ("binding_workload", "binding walk"),
}


def measure_mode(
    mode: str, root: Path = ROOT, target: Path | None = None
) -> Measurement:
    """Build and measure one workload in one checkout.

    The binding workload is the difference of two iteration counts, so what this
    returns for it is that difference beside the high run's iteration count --
    which the workload's own checksum must equal, and which is checked here
    rather than by the caller: a run that ignored its argument is a rig fault
    wherever it is measured.
    """
    example, subject = MODES[mode]
    if mode != "binding":
        build_workload(example, root, target)
        binary = (target or root / "target") / "release" / "examples" / example
        return measure(binary)

    build_binding_workload(root, target)
    binary = (target or root / "target") / "release" / "examples" / example
    high = measure(binary, str(BINDING_ITERS_HIGH))
    low = measure(binary, str(BINDING_ITERS_LOW))
    for result, iters in ((high, BINDING_ITERS_HIGH), (low, BINDING_ITERS_LOW)):
        if check_checksum(result.checksum, iters, f"{subject} x{iters:,}"):
            raise SystemExit(EXIT_FAIL)
    return Measurement(high.irefs - low.irefs, BINDING_ITERS_HIGH)


def resolve_rev(rev: str) -> str:
    """Resolve `rev` to a commit, or refuse: a base that is not there is not a pass."""
    result = subprocess.run(
        ["git", "rev-parse", "--verify", f"{rev}^{{commit}}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(f"could not resolve {rev!r} to a commit: {result.stderr.strip()}")
        raise SystemExit(EXIT_CANNOT_RUN)
    return result.stdout.strip()


def check_against_base(head: Measurement, base: Measurement, subject: str) -> int:
    """Hold this change's count to its own merge base's, measured beside it.

    One-sided, unlike the recorded-budget check. The floor there guards against a
    workload that stopped doing the work, and here the checksums do that better:
    two builds of a workload that computes the same thing agree exactly, so a
    count that fell because the body collapsed is caught by the comparison the
    caller makes before this one, and a count that fell because the code got
    faster is the outcome this gate exists to allow.
    """
    delta = (head.irefs - base.irefs) / base.irefs
    print(f"base:     {base.irefs:,} instructions ({subject})")
    print(f"head:     {head.irefs:,} instructions")
    print(f"delta:    {delta:+.2%} (ceiling +{RELATIVE_TOLERANCE:.0%})")
    if delta > RELATIVE_TOLERANCE:
        print("REGRESSION: this change costs more than its own merge base.")
        print("Both counts were measured in this job, with one toolchain, so the")
        print("difference is the change rather than the environment.")
        return 1
    print("OK: no regression against the base.")
    return 0


def run_relative(mode: str, rev: str) -> int:
    """Measure this checkout and `rev` side by side, and hold the difference.

    The base is built in a throwaway worktree with its own target directory, so
    the two builds neither share nor overwrite each other's output, and both use
    whatever toolchain is on PATH -- which is the point: an absolute budget
    compares against a number recorded on a machine that is not this one.
    """
    sha = resolve_rev(rev)
    _, subject = MODES[mode]
    head = measure_mode(mode)
    worktree = Path(tempfile.mkdtemp(prefix="valgebra-perf-base-"))
    checkout = worktree / "tree"
    try:
        subprocess.run(
            ["git", "worktree", "add", "--detach", "--quiet", str(checkout), sha],
            cwd=ROOT,
            check=True,
        )
        base = measure_mode(mode, checkout, worktree / "target")
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(checkout)],
            cwd=ROOT,
            check=False,
        )
        shutil.rmtree(worktree, ignore_errors=True)

    print(f"comparing {subject} against {rev} ({sha[:12]})")
    return judge_relative(head, base, subject)


def judge_relative(head: Measurement, base: Measurement, subject: str) -> int:
    """Compare two measurements of the same workload, or refuse to compare them.

    The checksum is asked first, as it is in the recorded-budget path and for a
    sharper reason: two runs of a workload that computes the same thing agree
    exactly, so a disagreement says the workload itself moved between the two
    commits and the counts are of different work. That is not a regression and
    not a pass -- it is a comparison that cannot be made, which is exit code 2.
    """
    if head.checksum != base.checksum:
        print(f"WORKLOAD CHANGED: base checksum {base.checksum}, head {head.checksum}.")
        print("The two runs measured different work, so their counts do not")
        print("compare. Re-record the recorded budget in the same commit that")
        print("changes the workload, and say what moved.")
        return EXIT_CANNOT_RUN
    print(f"checksum: {head.checksum} ({subject}, unchanged from the base)")
    return check_against_base(head, base, subject)


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


def run_decision(budget: dict, *, update: bool) -> int:
    """Gate the decision procedures: subtyping, emptiness, equivalence.

    A separate workload from the core one because it measures a separate
    surface. Without it a rule added to `decision.rs` is invisible to the gate in
    both directions -- neither the cost of a new one nor the saving from a
    cheaper one shows up.
    """
    build_workload("decision_workload")
    result = measure(DECISION_WORKLOAD)
    if update:
        budget["decision_workload_irefs"] = result.irefs
        budget["decision_workload_checksum"] = result.checksum
        BUDGET_FILE.write_text(json.dumps(budget, indent=2) + "\n", encoding="utf-8")
        print(f"recorded decision budget: {result.irefs:,} instructions")
        print(f"recorded decision checksum: {result.checksum}")
        return 0
    failed = check_checksum(
        result.checksum, int(budget["decision_workload_checksum"]), "decision workload"
    )
    if failed:
        return failed
    return check_against_budget(
        result.irefs,
        int(budget["decision_workload_irefs"]),
        float(budget["tolerance"]),
    )


def run_binding(budget: dict, *, update: bool) -> int:
    # The walk's per-iteration cost, isolated by subtracting two runs so the
    # embedded interpreter's (identical) startup cancels out. The workload folds
    # one unit per successful walk, so its checksum IS its iteration count, and
    # `measure_mode` holds it to that identity: a run that ignored its argument
    # reports a difference near zero, which a ceiling-only budget accepts.
    measured = measure_mode("binding").irefs
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
    mode = (
        "binding"
        if "--binding" in args
        else "decision"
        if "--decision" in args
        else "core"
    )
    if "--against" in args:
        at = args.index("--against")
        if at + 1 >= len(args):
            print("--against needs a revision: --against origin/main")
            return EXIT_CANNOT_RUN
        return run_relative(mode, args[at + 1])
    update = "--update" in args
    budget = json.loads(BUDGET_FILE.read_text(encoding="utf-8"))
    if mode == "binding":
        return run_binding(budget, update=update)
    if mode == "decision":
        return run_decision(budget, update=update)
    return run_core(budget, update=update)


if __name__ == "__main__":
    sys.exit(main())
