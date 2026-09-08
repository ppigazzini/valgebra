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

Three workloads, and seven shapes across them:

* The default **core** workload (`perf_workload`, pure Rust) measures the schema
  operations and is fully deterministic, so its budget is tight.
* The **decision** workload (`--decision`) measures the three relations, which
  the core one never calls.
* The **binding** workload measures live Python values through the shipped
  entry points -- the hot path the pure-Rust workloads do not reach -- in five
  shapes: the membership walk (`--binding`), the call boundary alone
  (`--binding-boundary`), the walk over a wide record (`--binding-record`),
  building one from its Python spelling (`--binding-build`), and explaining a
  failure in one (`--binding-explain`). Each is the deterministic twin of a
  shape the comparison gate times, so a wall-clock movement there can be
  confirmed or refuted here. They embed CPython, whose startup is not a fixed
  instruction count, so each is measured as the *difference* between two
  iteration counts: startup cancels, leaving the deterministic per-iteration
  cost. Their budgets carry a wider tolerance to absorb cross-interpreter FFI
  variance while still catching a per-node regression, which is far larger.

``--against`` takes every shape named on the command line and measures them all
against **one** build of the base: the build is minutes and a measurement is
seconds, so shapes are cheap and invocations are not.

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
    python scripts/perf_gate.py --binding-build      # one of the other shapes
    python scripts/perf_gate.py --against HEAD~1 --binding --binding-build

Requires valgrind on PATH and a Rust toolchain. The binding shapes also need an
embedded interpreter: the build links libpython, so run them with the
interpreter's library directory on the loader path.
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


#: Every mode, each naming the example it builds and what it measures.
#:
#: The four shapes below `binding` are the deterministic twins of the comparison
#: gate's wall-clock shapes. The gate compared seven shapes against another
#: library and budgeted one of them, and the gap is how a shape moves without
#: anything saying so: schema construction grew twelve percent over one release
#: cycle, and no instruction count was watching the thing that grew.
#:
#: All five binding modes are the *difference* of two iteration counts, so the
#: embedded interpreter's startup cancels; they share the pair of counts and the
#: checksum rig, and differ only in which workload the example runs.
MODES = {
    "core": ("perf_workload", "core workload"),
    "decision": ("decision_workload", "decision workload"),
    "binding": ("binding_workload", "binding walk"),
    "binding-boundary": ("binding_workload", "binding call boundary"),
    "binding-record": ("binding_workload", "binding record walk"),
    "binding-build": ("binding_workload", "binding record build"),
    "binding-explain": ("binding_workload", "binding record explain"),
}

#: The workload argument each binding mode passes, and the budget key it reads.
#: Iterations per shape, high and low, chosen so each measurement is a minute or
#: so under cachegrind rather than ten. The walk keeps the pair its budget was
#: recorded with; the shapes that do more work per iteration run fewer of them,
#: and the difference still cancels startup because both runs share it.
BINDING_ITERATIONS = {
    "binding": (150_000, 50_000),
    "binding-boundary": (150_000, 50_000),
    "binding-record": (20_000, 5_000),
    "binding-build": (4_000, 1_000),
    "binding-explain": (8_000, 2_000),
}

BINDING_SHAPES = {
    "binding": "walk",
    "binding-boundary": "boundary",
    "binding-record": "record",
    "binding-build": "build",
    "binding-explain": "explain",
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
    if mode not in BINDING_SHAPES:
        build_workload(example, root, target)
        binary = (target or root / "target") / "release" / "examples" / example
        return measure(binary)

    shape = BINDING_SHAPES[mode]
    hi, lo = BINDING_ITERATIONS[mode]
    build_binding_workload(root, target)
    binary = (target or root / "target") / "release" / "examples" / example
    high = measure(binary, str(hi), shape)
    low = measure(binary, str(lo), shape)
    # The build shape folds a node count per iteration rather than one, so its
    # checksum is a multiple of the count rather than the count. The rig check is
    # that both runs agree on that multiple, which a run ignoring its argument
    # cannot do: it would report the same checksum twice.
    scale = high.checksum // hi if hi else 0
    for result, iters in ((high, hi), (low, lo)):
        if check_checksum(result.checksum, iters * scale, f"{subject} x{iters:,}"):
            raise SystemExit(EXIT_FAIL)
    return Measurement(high.irefs - low.irefs, hi - lo)


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


def commits_since(sha: str, root: Path = ROOT) -> int | None:
    """How many commits this checkout carries that the base does not.

    `None` where the question cannot be answered: a shallow clone may not reach
    the base at all, and an unrelated history has no path between the two.
    """
    result = subprocess.run(
        ["git", "rev-list", "--count", f"{sha}..HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    try:
        return int(result.stdout.strip())
    except ValueError:
        return None


def describe_window(count: int | None) -> tuple[str, bool]:
    """Say how much history one comparison is covering, and whether it is a batch.

    A relative gate compares two counts and applies one ceiling to the
    difference. That ceiling is written for a *push*: one commit's worth of
    movement, the amount a reviewer can look at. What the base actually names is
    whatever the event carried, and a force-push, a re-pushed branch or a first
    push of a long-lived branch make that twenty-odd commits, whose movements
    sum -- so a regression in one is paid for by an improvement in another and
    the gate says nothing, or the sum trips the ceiling and names no commit.

    Neither reading is wrong, and they are not the same reading. This does not
    widen the ceiling or refuse the run: it says which one is being taken, so a
    figure read off a batch is never mistaken for a figure read off a change.
    """
    if count is None:
        unknown = (
            "window:   unknown -- the base is not reachable from this checkout, "
            "so what the difference covers cannot be said"
        )
        return (unknown, True)
    if count <= 1:
        return (f"window:   {count} commit", False)
    batch = (
        f"window:   {count} commits -- a batch, so the difference below is "
        "their sum and no single commit is named by it"
    )
    return (batch, True)


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


def run_relative(modes: list[str], rev: str) -> int:
    """Measure this checkout and `rev` side by side, and hold each difference.

    The base is built in a throwaway worktree with its own target directory, so
    the two builds neither share nor overwrite each other's output, and both use
    whatever toolchain is on PATH -- which is the point: an absolute budget
    compares against a number recorded on a machine that is not this one.

    Several shapes are measured against **one** base build. Building the base
    once per shape is what kept the gate to a single shape: the build is minutes
    and the measurement is seconds, so a second shape costs almost nothing while
    a second invocation costs another build.

    The verdicts are aggregated by severity, and every shape is measured before
    any is judged, so one regression does not hide the next one's number.
    """
    sha = resolve_rev(rev)
    head = {mode: measure_mode(mode) for mode in modes}
    worktree = Path(tempfile.mkdtemp(prefix="valgebra-perf-base-"))
    checkout = worktree / "tree"
    try:
        subprocess.run(
            ["git", "worktree", "add", "--detach", "--quiet", str(checkout), sha],
            cwd=ROOT,
            check=True,
        )
        base = {
            mode: measure_mode(mode, checkout, worktree / "target") for mode in modes
        }
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(checkout)],
            cwd=ROOT,
            check=False,
        )
        shutil.rmtree(worktree, ignore_errors=True)

    print(f"comparing against {rev} ({sha[:12]})")
    note, batch = describe_window(commits_since(sha))
    print(note)
    outcomes = []
    for mode in modes:
        subject = MODES[mode][1]
        print(f"\n--- {subject}")
        outcomes.append(judge_relative(head[mode], base[mode], subject))
    outcome = EXIT_OK
    if EXIT_CANNOT_RUN in outcomes:
        outcome = EXIT_CANNOT_RUN
    elif EXIT_FAIL in outcomes:
        outcome = EXIT_FAIL
    if batch and outcome == EXIT_FAIL:
        print("\nThe ceiling is written for one commit and this reading covers")
        print("more, so the first step is `git bisect` over the window rather")
        print("than a repair to the tip.")
    elif batch:
        print("\nA pass over a batch is a pass on the sum: a regression inside it")
        print("cancelled by an improvement beside it reads exactly like this.")
    return outcome


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


def run_binding(budget: dict, mode: str, *, update: bool) -> int:
    """Measure one binding shape against its own budget.

    Every shape here is the difference of two iteration counts, so the embedded
    interpreter's (identical) startup cancels out. `measure_mode` holds each run
    to its checksum first: a run that ignored its argument reports a difference
    near zero, which a ceiling-only budget would accept.
    """
    subject = MODES[mode][1]
    key = f"{mode.replace('-', '_')}_workload_irefs"
    hi, lo = BINDING_ITERATIONS[mode]
    measured = measure_mode(mode).irefs
    print(
        f"{subject} over {hi - lo:,} iterations (difference of {hi:,} and {lo:,} runs)"
    )
    if update:
        budget[key] = measured
        BUDGET_FILE.write_text(json.dumps(budget, indent=2) + "\n", encoding="utf-8")
        print(f"recorded {subject} budget: {measured:,} instructions")
        return 0
    if key not in budget:
        print(f"perf_gate: no budget recorded for {subject}; run with --update")
        return EXIT_CANNOT_RUN
    return check_against_budget(
        measured,
        int(budget[key]),
        float(budget["binding_tolerance"]),
    )


def main() -> int:
    args = sys.argv[1:]
    # `--binding` keeps its meaning -- the walk -- so an existing invocation and
    # the budget recorded under it still name the same measurement.
    flagged = [name for name in MODES if f"--{name}" in args] or ["core"]
    if "--against" in args:
        at = args.index("--against")
        if at + 1 >= len(args):
            print("--against needs a revision: --against origin/main")
            return EXIT_CANNOT_RUN
        # Every flagged shape against one base build, since the build is the
        # expensive half and the shapes share it.
        return run_relative(flagged, args[at + 1])
    update = "--update" in args
    budget = json.loads(BUDGET_FILE.read_text(encoding="utf-8"))
    outcomes = []
    for mode in flagged:
        if len(flagged) > 1:
            print(f"\n--- {MODES[mode][1]}")
        if mode in BINDING_SHAPES:
            outcomes.append(run_binding(budget, mode, update=update))
        elif mode == "decision":
            outcomes.append(run_decision(budget, update=update))
        else:
            outcomes.append(run_core(budget, update=update))
    if EXIT_CANNOT_RUN in outcomes:
        return EXIT_CANNOT_RUN
    return EXIT_FAIL if EXIT_FAIL in outcomes else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
