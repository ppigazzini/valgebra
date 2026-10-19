"""Deterministic instruction-count regression gate.

Builds a fixed workload, runs it under cachegrind, and compares the
executed-instruction count against the same workload built at a base. The count
is identical across runs of a given build, so the gate catches an algorithmic
regression without depending on a noisy wall clock. Shared CI runners are too
variable for a wall-clock budget; instruction count is not.

**The gate compares a change with its base** (``--against <rev>``). The same
workload is built and measured twice in one job -- once at ``HEAD``, once at
``rev`` in a throwaway worktree -- with the same toolchain, the same flags and
the same machine, and the gate holds the *difference*. Everything a recorded
number cannot control cancels, so the tolerance is tight (2%) and a regression
this change introduced is what is left. ``--against HEAD`` reads an
uncommitted change.

**The counts in ``scripts/perf_budget.json`` are a record, not a gate.** The
count is deterministic for a build, but not across toolchains or target
features. Measured, the core count recorded on one machine re-measures 5.35%
away from it on another machine of the same rustc line, so a band wide enough
not to flake is too wide to see a real 5% regression. The record is read where
there is no base count to compare with -- a shape the base does not carry, and
a re-recorded workload the base cannot build -- and the shape is held to its
recorded count inside that band instead. A re-recorded workload the base can
build is built there from this change's source and compared at 2%.
``--update`` writes it.

Three workloads, and the shapes each is read in:

* The default **core** workload (`perf_workload`, pure Rust) measures the schema
  operations and is fully deterministic, so its recorded count carries the
  narrower band.
* The **decision** workloads measure the three relations, which the core one
  never calls: relations that hold (`--decision`), relations that are refuted
  (`--decision-refute`), goals asked again (`--decision-repeat`), the pairs
  that reach the set representation (`--decision-matrix`), one corpus asked of
  the rules (`--decision-rules`) and of the sets (`--decision-sets`), whose two
  counts are what a rule's answer saves over building the sets for it, and
  length bounds over a list asked of the sets (`--decision-lengths`), which
  read what a bound costs an element.
* The **binding** workload measures live Python values through the shipped
  entry points -- the hot path the pure-Rust workloads do not reach -- in
  twenty-seven shapes: the membership walk (`--binding`), the call boundary
  alone (`--binding-boundary`), the walk over a wide record
  (`--binding-record`), the same walk over a value whose keys are interned
  (`--binding-keys`), the same fields declared by an open record
  (`--binding-open`), a mapping read through its clause (`--binding-mapping`),
  walking a `tuple` subclass (`--binding-subclass`), matching a string against
  a compiled pattern (`--binding-pattern`), walking a value against a
  recursive schema (`--binding-recursive`), walking a value nested twenty-five
  deep (`--binding-deep`), walking refined elements (`--binding-refined`),
  walking elements of a union of scalars (`--binding-nullable`), walking a set
  (`--binding-set`), parsing and walking a JSON document (`--binding-json`),
  the same document rejected halfway (`--binding-json-reject`), a document of
  two record kinds against their union (`--binding-json-union`), records read
  through a key-type clause (`--binding-json-open`), the same records under a
  length bound (`--binding-json-bounded`), a recursive schema over a
  nested document (`--binding-json-deep`), building a validator from its
  Python spelling (`--binding-build`), compiling one written as a `TypedDict`
  of refined integers (`--binding-annotated`), compiling a fifty-field
  dataclass (`--binding-object`), compiling a fifty-member protocol
  (`--binding-protocol`), relating two dataclasses (`--binding-relation`),
  explaining a failure in a record (`--binding-explain`) or accepting one in
  the same mode (`--binding-explain-accept`), and explaining a refused list
  (`--binding-explain-list`). Most are the deterministic twin
  of a shape the comparison gate times, so a wall-clock movement there can be
  confirmed or refuted here. They embed CPython, whose startup is not a fixed
  instruction count, so each is measured as the *difference* between two
  iteration counts: startup cancels, leaving the deterministic per-iteration
  cost. Their recorded counts carry a wider band to absorb cross-interpreter
  FFI variance while still catching a per-node regression, which is far larger.

**The profiled shapes** (`--profiled`) are the comparison gate's seven, counted
on the wheel a release ships: built with profile-guided optimisation, which the
three workloads above are not, so a change to the profile's training moves
nothing they read. The wheel is built at the change and at its base, each from
a snapshot of its tree unpacked into **one** directory, since the directory a
profiled build runs in moves the layout of a loop the training does not reach
by a tenth; each side's wheel is installed into one bare environment in turn,
and each shape is the difference of two call counts, as a binding shape is.
They are compared with the base and never recorded: their counts are of a
build no record could reproduce elsewhere. So a recorded step cannot excuse one
of them, since a step expires by its shape's recorded count; a trade on the
profiled wheel is red until one is argued here.

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
* **A new shape's band is two-sided.** A count far *below* the recorded one
  is not a pass either: a workload that stopped doing the work measures low,
  and a one-sided ceiling publishes that as an improvement it never earned.

Usage:
    python scripts/perf_gate.py --against origin/main    # the merge gate
    python scripts/perf_gate.py --against HEAD           # an uncommitted change
    python scripts/perf_gate.py --against HEAD~1 --decision
    python scripts/perf_gate.py --against HEAD~1 --binding --binding-build
    python scripts/perf_gate.py --against origin/main --profiled
    python scripts/perf_gate.py --update                 # record the core count
    python scripts/perf_gate.py --decision --update      # record that shape's
    python scripts/perf_gate.py --binding --update       # and the walk's

A command line naming neither ``--against`` nor ``--update`` is refused with
exit 2: there is nothing to compare with.

Requires valgrind on PATH and a Rust toolchain. The binding shapes also need an
embedded interpreter: the build links libpython, so run them with the
interpreter's library directory on the loader path. The profiled shapes need
maturin and the LLVM tools a profile is merged with, and build for the
interpreter running this script.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from _lib import EXIT_CANNOT_RUN, EXIT_FAIL, EXIT_OK, ROOT, arguments, git_output

if TYPE_CHECKING:
    import argparse
    from collections.abc import Mapping, Sequence


BUDGET_FILE = ROOT / "scripts" / "perf_budget.json"
# What a change may add against its own merge base, measured in one job with one
# toolchain. Two percent is far outside the run-to-run variation of a count that
# is deterministic per build (it is zero) and far inside the smallest regression
# worth a conversation, so the band is here to absorb the one thing that does
# move: which inlining decisions the linker makes as code is added elsewhere.
RELATIVE_TOLERANCE = 0.02
IREFS = re.compile(r"I\s+refs:\s*([\d,]+)")
MISPREDICTS = re.compile(r"Mispredicts:\s*([\d,]+)")

#: The shapes whose simulated branch mispredicts are read beside their count.
#: The decision path dispatches on a node's kind through jump tables taken
#: several times per goal, and removing them trades instructions for predicted
#: branches: +4% instructions and -6 to -8% wall clock on the matrix shape, on
#: one machine. An instruction-only gate reads that as a regression, so the
#: mispredicts are printed where such a change would be judged -- and not gated,
#: because cachegrind predicts an indirect branch by its last target, a model its
#: manual calls older than the processors the lanes run on.
BRANCH_MODES = frozenset(
    {
        "decision",
        "decision-refute",
        "decision-repeat",
        "decision-matrix",
        "decision-rules",
        "decision-sets",
        "decision-lengths",
    }
)
# Both workloads print one trailing line holding the checksum, bare or prefixed.
CHECKSUM = re.compile(r"^(?:checksum=)?(\d+)$")


@dataclass(frozen=True, slots=True)
class Measurement:
    """One cachegrind run: what it executed, and what the workload computed."""

    irefs: int
    checksum: int
    #: Simulated branch mispredicts, for a mode in `BRANCH_MODES`; else `None`.
    mispredicts: int | None = None


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
    branched = MISPREDICTS.search(stderr)
    mispredicts = int(branched.group(1).replace(",", "")) if branched else None
    return Measurement(irefs, checksum, mispredicts)


#: The variables a workload runs with, when the caller has them set: the loader
#: path an embedded interpreter's library is found on, the home a relocated
#: interpreter needs to find its standard library, and valgrind's own override.
#: Nothing else of the caller's environment reaches a measurement.
PASSED_THROUGH = ("LD_LIBRARY_PATH", "PYTHONHOME", "VALGRIND_LIB")


#: The glibc allocator setting every measurement runs under: no fastbins, so the
#: heap a loop starts from does not decide what its allocations cost.
MALLOC_TUNABLES = "glibc.malloc.mxfast=0"


def workload_environment(caller: Mapping[str, str]) -> dict[str, str]:
    """Return the environment a measurement runs in: the few names it needs.

    A count depends on the environment it is taken in, and not only through the
    variables a workload reads. The block is copied onto the stack at start-up,
    so its size moves everything laid out after it, and an embedded interpreter
    copies it onto the heap: one binary read the JSON shape 504,839,830 under
    `uv run` and 550,142,966 from a login shell, and a pure-Rust workload, which
    is not differenced, carries 2.97% between the two as a constant. A caller's
    `PATH`, virtual environment and tool variables are the caller's; the
    measurement runs without them, with the hash seed fixed, so the same binary
    reads the same count from any shell.

    **glibc's fastbins are turned off.** Whether a large request first
    consolidates the small chunks freed before it follows from where the setup's
    blocks landed, and that follows from the binary's own layout: one commit
    built in two directories read the JSON shape 458,071,250 and 527,858,866,
    all of the difference in `_int_malloc` and `malloc_consolidate`. With
    `mxfast=0` the two read 535,119,725 and 534,992,150. A count taken this
    way is a count of the tree's work rather than of where the linker put it.

    **No run writes bytecode.** A binding shape is the difference of two runs,
    which cancels the interpreter's start-up only while both runs start the same
    way. An interpreter that has never imported a module compiles it on first
    import and writes the result beside the source, so the first run pays a
    compile its partner reads back: `settle_the_heap` imports `ctypes`, which
    nothing else in the bench job imports, and on a fresh interpreter the first
    walk read 166,682,582 against 125,400,049 for the same binary run again.
    With nothing written, both runs compile or both read, and a cold
    interpreter and a warm one read the same count.
    """
    kept = {name: caller[name] for name in PASSED_THROUGH if name in caller}
    kept["PYTHONHASHSEED"] = "0"
    kept["GLIBC_TUNABLES"] = MALLOC_TUNABLES
    kept["PYTHONDONTWRITEBYTECODE"] = "1"
    return kept


def measuring_toolchain() -> dict[str, str]:
    """Name the valgrind and C library a count is taken with.

    A recorded budget is a reading of one environment, and two parts of it move
    a count without any change to the tree: the valgrind that simulates the
    machine, and the allocator the workloads call, whose free lists
    `settle_the_heap` normalises and which glibc 2.43 reorganised.
    """
    valgrind = subprocess.run(
        ["valgrind", "--version"], capture_output=True, text=True, check=False
    ).stdout.strip()
    try:
        libc = os.confstr("CS_GNU_LIBC_VERSION") or "unknown"
    except (ValueError, OSError):
        libc = "unknown"
    return {"valgrind": valgrind or "unknown", "libc": libc}


def record_budget(budget: dict) -> None:
    """Write the budget file, with the toolchain the counts in it were taken with."""
    budget["measured_with"] = measuring_toolchain()
    BUDGET_FILE.write_text(json.dumps(budget, indent=2) + "\n", encoding="utf-8")


def measure(binary: Path, *args: str, branches: bool = False) -> Measurement:
    """Count the instructions one run of a workload executes.

    The count of a given binary is deterministic, which is the whole premise of
    this gate -- but a workload that embeds CPython inherits the interpreter's
    own randomness, and one of those is enough to break it. A **string hash seed
    is drawn per process**, and the record shapes probe a dict of fifty string
    keys: a different seed lays that dict out differently, collides differently,
    and executes a different number of instructions. Three runs of one binary
    read 255,472,587, 255,818,996 and 256,656,284; three with the seed fixed
    read 258,773,618 three times.

    Half a percent on a raw count is several percent on the *difference* of two
    counts this gate reads, which is the two-percent ceiling it holds changes to.
    So the seed is fixed here, in the one place every measurement passes through,
    rather than left to the caller's environment -- and the rest of that
    environment is left behind with it (`workload_environment`).
    """
    valgrind = shutil.which("valgrind")
    if valgrind is None:
        print("perf_gate: valgrind is not on PATH")
        sys.exit(EXIT_CANNOT_RUN)
    result = subprocess.run(
        [
            valgrind,
            "--tool=cachegrind",
            "--cachegrind-out-file=/dev/null",
            *(["--branch-sim=yes"] if branches else []),
            str(binary),
            *args,
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=workload_environment(os.environ),
    )
    return parse_measurement(result.stdout, result.stderr)


def check_checksum(measured: int, expected: int, subject: str) -> int:
    """Refuse a measurement whose workload did not compute what it must.

    A workload whose body collapsed still runs, still executes instructions, and
    still reports a count a band would accept from below. The checksum is the
    only thing that says the count belongs to the work it claims to measure, so
    it is compared first and a mismatch stops the run.
    """
    if measured != expected:
        print(f"RIG FAULT: {subject} checksum {measured}, expected {expected}.")
        print("The workload did not compute what it must; the count measures")
        print("something other than the work this count is for.")
        return 1
    print(f"checksum: {measured} ({subject}, as expected)")
    return 0


def per_iteration(checksum: int, iterations: int, subject: str) -> int | None:
    """Read what a binding run computes per iteration, or refuse a run of none.

    The scale is read off the run itself, because the build shape folds a node
    count where the others fold one, and both runs are then held to it. A walk
    that answers every value wrong, or half of them, computes less than one per
    iteration: read as a scale of nought, it would hold both runs to a checksum
    of nought, which such a walk meets, and its count -- cheap, as wrong answers
    are -- would read as an improvement.
    """
    scale = checksum // iterations if iterations else 0
    if scale < 1:
        print(f"RIG FAULT: {subject} checksum {checksum} over {iterations:,}")
        print("iterations: the workload computed less than one a pass, so the")
        print("count is not of the work it is for.")
        return None
    return scale


def check_against_budget(measured: int, recorded: int, tolerance: float) -> int:
    """Hold a count inside a two-sided band around the recorded one."""
    ceiling = int(recorded * (1 + tolerance))
    floor = int(recorded * (1 - tolerance))
    delta = (measured - recorded) / recorded
    print(f"measured: {measured:,} instructions")
    print(f"recorded: {recorded:,} (+/-{tolerance:.0%} -> {floor:,} .. {ceiling:,})")
    print(f"delta:    {delta:+.2%}")
    if measured > ceiling:
        print("REGRESSION: instruction count exceeds the recorded band.")
        return 1
    if measured < floor:
        print("UNDER-RUN: instruction count falls below the recorded band.")
        print("Either the workload stopped doing the work it measures, or this is")
        print("a real optimization -- re-record with --update and say which.")
        return 1
    print("OK: within the recorded band.")
    return 0


#: Every mode, each naming the example it builds and what it measures.
#:
#: The four modes after `binding` -- `binding-boundary`, `binding-record`,
#: `binding-build` and `binding-explain` -- are the deterministic twins of the
#: comparison gate's wall-clock shapes. A shape that gate times against another
#: library and nothing counts is how a shape moves without anything saying so:
#: schema construction can grow over a release cycle with no instruction count
#: watching the thing that grows.
#:
#: Every binding mode is the *difference* of two iteration counts, so the
#: embedded interpreter's startup cancels; the modes share the pair of counts and
#: the checksum rig, and differ only in which workload the example runs.
MODES = {
    "core": ("perf_workload", "core workload"),
    "decision": ("decision_workload", "decision workload"),
    "decision-refute": ("decision_refute_workload", "refuting decision workload"),
    "decision-repeat": ("decision_repeat_workload", "repeating decision workload"),
    "decision-matrix": ("decision_matrix_workload", "matrix decision workload"),
    "decision-rules": ("decision_sets_workload", "relations asked of the rules"),
    "decision-sets": ("decision_sets_workload", "relations asked of the sets"),
    "decision-lengths": ("decision_sets_workload", "length bounds asked of the sets"),
    "binding": ("binding_workload", "binding walk"),
    "binding-boundary": ("binding_workload", "binding call boundary"),
    "binding-record": ("binding_workload", "binding record walk"),
    "binding-build": ("binding_workload", "binding record build"),
    "binding-explain": ("binding_workload", "binding record explain"),
    "binding-explain-accept": ("binding_workload", "binding record explain accept"),
    "binding-explain-list": ("binding_workload", "binding refused list explain"),
    "binding-open": ("binding_workload", "binding open record walk"),
    "binding-annotated": ("binding_workload", "binding annotated build"),
    "binding-object": ("binding_workload", "binding dataclass build"),
    "binding-protocol": ("binding_workload", "binding protocol build"),
    "binding-relation": ("binding_workload", "binding relation between dataclasses"),
    "binding-subclass": ("binding_workload", "binding tuple subclass walk"),
    "binding-keys": ("binding_workload", "binding record walk by interned keys"),
    "binding-json": ("binding_workload", "binding JSON document parse and walk"),
    "binding-pattern": ("binding_workload", "binding pattern refinement walk"),
    "binding-recursive": ("binding_workload", "binding recursive schema walk"),
    "binding-json-reject": ("binding_workload", "binding JSON document rejected"),
    "binding-json-union": ("binding_workload", "binding JSON union of records"),
    "binding-json-open": ("binding_workload", "binding JSON records read by clause"),
    "binding-json-bounded": (
        "binding_workload",
        "binding JSON records under a length bound",
    ),
    "binding-json-deep": ("binding_workload", "binding JSON recursive document"),
    "binding-deep": ("binding_workload", "binding deep nesting walk"),
    "binding-refined": ("binding_workload", "binding refined element walk"),
    "binding-mapping": ("binding_workload", "binding mapping walk"),
    "binding-nullable": ("binding_workload", "binding nullable element walk"),
    "binding-set": ("binding_workload", "binding set walk"),
}

#: The reading a core shape passes its example, for the three that share one.
#:
#: The rules and the sets are asked the same pairs, so the ratio of the two
#: counts is a ratio over one question rather than over two corpora. One
#: example holds the corpus, and the argument names which decider answers it.
#: The length bounds are a corpus of their own, asked of the sets through the
#: same reading.
CORE_READINGS = {
    "decision-rules": "rules",
    "decision-sets": "sets",
    "decision-lengths": "lengths",
}

#: The iteration pair each binding mode runs at.
#: Iterations per shape, high and low, chosen so each measurement is a minute or
#: so under cachegrind rather than ten. The walk runs the pair its recorded
#: count is taken at; the shapes that do more work per iteration run fewer of them,
#: and the difference still cancels startup because both runs share it.
BINDING_ITERATIONS = {
    "binding": (150_000, 50_000),
    "binding-boundary": (150_000, 50_000),
    "binding-record": (20_000, 5_000),
    "binding-build": (4_000, 1_000),
    "binding-explain": (8_000, 2_000),
    "binding-explain-accept": (20_000, 5_000),
    "binding-explain-list": (2_000, 500),
    "binding-open": (20_000, 5_000),
    "binding-annotated": (300, 100),
    "binding-object": (2_000, 500),
    "binding-protocol": (2_000, 500),
    "binding-relation": (20_000, 5_000),
    "binding-subclass": (150_000, 50_000),
    "binding-keys": (20_000, 5_000),
    "binding-json": (500, 150),
    "binding-pattern": (150_000, 50_000),
    "binding-recursive": (50_000, 15_000),
    "binding-json-reject": (500, 150),
    "binding-json-union": (500, 150),
    "binding-json-open": (500, 150),
    "binding-json-bounded": (500, 150),
    "binding-json-deep": (1_000, 300),
    "binding-deep": (50_000, 15_000),
    "binding-refined": (20_000, 5_000),
    "binding-mapping": (20_000, 5_000),
    "binding-nullable": (150_000, 50_000),
    "binding-set": (150_000, 50_000),
}

#: Where every binding shape's work is written: the example the gate builds
#: names a shape, and calls this module to run it.
BINDING_WORKLOAD = ROOT / "crates" / "valgebra-py" / "src" / "workload.rs"

BINDING_SHAPES = {
    "binding": "walk",
    "binding-boundary": "boundary",
    "binding-record": "record",
    "binding-build": "build",
    "binding-explain": "explain",
    "binding-explain-accept": "explain-accept",
    "binding-explain-list": "explain-list",
    "binding-open": "open",
    "binding-annotated": "annotated",
    "binding-object": "object",
    "binding-protocol": "protocol",
    "binding-relation": "relation",
    "binding-subclass": "subclass",
    "binding-keys": "keys",
    "binding-json": "json",
    "binding-pattern": "pattern",
    "binding-recursive": "recursive",
    "binding-json-reject": "json-reject",
    "binding-json-union": "json-union",
    "binding-json-open": "json-open",
    "binding-json-bounded": "json-bounded",
    "binding-json-deep": "json-deep",
    "binding-deep": "deep",
    "binding-refined": "refined",
    "binding-mapping": "mapping",
    "binding-nullable": "nullable",
    "binding-set": "set",
}


def warm_the_interpreter(binary: Path, shape: str) -> None:
    """Run one iteration of a shape, uncounted, so every module it imports is compiled.

    The counted runs write no bytecode (`workload_environment`), so they read
    whatever cache the interpreter has when they start, and the side measured
    first would read a colder one than the side after it: building the base runs
    the interpreter in between. A cold start compiles what a warm one reads, and
    leaves the loop a different heap. This run may write, so both sides of a
    comparison count from the same cache.
    """
    environment = workload_environment(os.environ)
    del environment["PYTHONDONTWRITEBYTECODE"]
    subprocess.run(
        [str(binary), "1", shape],
        cwd=ROOT,
        check=True,
        capture_output=True,
        env=environment,
    )


def measure_mode(
    mode: str, root: Path = ROOT, target: Path | None = None
) -> Measurement:
    """Build and measure one workload in one checkout.

    The binding workload is the difference of two iteration counts, so what this
    returns for it is that difference beside what one iteration folds into the
    checksum: its answer and the size of the work it reads. Both runs must
    agree on that figure, which is checked here rather than by the caller: a run
    that ignored its argument is a rig fault wherever it is measured. The figure
    is the checksum compared with the base's, so a value cut short reads as
    other work rather than as a saving.
    """
    example, subject = MODES[mode]
    if mode not in BINDING_SHAPES:
        build_workload(example, root, target)
        binary = (target or root / "target") / "release" / "examples" / example
        reading = (CORE_READINGS[mode],) if mode in CORE_READINGS else ()
        return measure(binary, *reading, branches=mode in BRANCH_MODES)

    shape = BINDING_SHAPES[mode]
    hi, lo = BINDING_ITERATIONS[mode]
    build_binding_workload(root, target)
    binary = (target or root / "target") / "release" / "examples" / example
    warm_the_interpreter(binary, shape)
    high = measure(binary, str(hi), shape)
    low = measure(binary, str(lo), shape)
    # Every iteration folds the same figure, so a checksum is a multiple of the
    # iteration count. The rig check is that both runs agree on that multiple,
    # which a run ignoring its argument cannot do: it would report the same
    # checksum twice.
    scale = per_iteration(high.checksum, hi, subject)
    if scale is None:
        raise SystemExit(EXIT_FAIL)
    for result, iters in ((high, hi), (low, lo)):
        if check_checksum(result.checksum, iters * scale, f"{subject} x{iters:,}"):
            raise SystemExit(EXIT_FAIL)
    return Measurement(high.irefs - low.irefs, scale)


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
    counted = git_output("rev-list", "--count", f"{sha}..HEAD", cwd=root)
    try:
        return None if counted is None else int(counted)
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
    if count == 0:
        return ("window:   no commit -- what the checkout changes on its base", False)
    if count == 1:
        return ("window:   1 commit", False)
    batch = (
        f"window:   {count} commits -- a batch, so the difference below is "
        "their sum and no single commit is named by it"
    )
    return (batch, True)


def recorded_step(
    mode: str, steps: Sequence[dict], carried: Sequence[dict]
) -> dict | None:
    """Return the argued step this comparison crosses, if one is recorded.

    A shape's instruction count is a proxy for its cost, and once in a while a
    change moves the two apart: reading a list's elements through a snapshot of
    it executes *more* instructions and spends less time, because the reference
    counts it pays land in two tight loops instead of one dependent chain. The
    ceiling below is written for the case where the two agree, and refusing a
    change the wall clock says is two and a half times faster would be the
    proxy governing the thing it stands for.

    So the commit that takes a step records it in the budget file (`steps`),
    and the step excuses a comparison **whose base does not carry it** -- the
    head's record holds it and the base's does not. That is the comparison
    crossing the commit, and no later one: once the base is at or past the
    step, its own record carries it and the ordinary ceiling applies again. A
    step is named by its shape and the base count it steps from, so rewording
    its argument does not make it a new one.

    A base count alone is not an expiry. A step is a difference, and a later
    change that gives part of it back brings the shape's count back near the
    base the step names, where a match on the count would excuse every change
    after it. Which commits a base carries is a fact of the tree, the same on
    every machine, so a step can be recorded from a local reading.
    """
    held = {(step.get("shape"), step.get("base_irefs")) for step in carried}
    for step in steps:
        if step.get("shape") == mode and (mode, step.get("base_irefs")) not in held:
            return step
    return None


def carried_steps(checkout: Path) -> list[dict]:
    """Read the steps a checkout's own record carries; none where it has none."""
    try:
        record = json.loads(
            (checkout / "scripts" / "perf_budget.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return []
    steps = record.get("steps", []) if isinstance(record, dict) else []
    return steps if isinstance(steps, list) else []


def stale_steps(budget: dict) -> list[str]:
    """Name every recorded step the file has outlived.

    A step describes one transition: from `base_irefs`, by at most `ceiling`.
    It is worth keeping while the shape's recorded count is still somewhere in
    that window -- the count a base measures, or the count a head measures once
    the step is taken. A count outside it says the shape has been re-recorded
    for some other reason since, so the comparison the record was written to
    excuse is one nobody can make any more.

    This is the changelog roll's rule, one file over: a record that outlives
    what it records is worse than no record, because it reads as an argument
    somebody still stands behind.
    """
    stale = []
    for step in budget.get("steps", []):
        shape = step.get("shape", "")
        recorded = budget.get(record_key(shape))
        if recorded is None:
            stale.append(f"{shape}: no budget is recorded for the shape")
            continue
        base = int(step["base_irefs"])
        low = base * (1 - RELATIVE_TOLERANCE)
        high = base * (1 + float(step["ceiling"])) * (1 + RELATIVE_TOLERANCE)
        if not low <= int(recorded) <= high:
            stale.append(
                f"{shape}: steps from {base:,} by at most "
                f"{float(step['ceiling']):.0%}, and the shape records "
                f"{int(recorded):,}, which is outside that window"
            )
    return stale


def check_against_base(
    head: Measurement, base: Measurement, subject: str, step: dict | None = None
) -> int:
    """Hold this change's count to its own merge base's, measured beside it.

    One-sided, unlike a new shape's band. The floor there guards against a
    workload that stopped doing the work, and here the checksums do that better:
    two builds of a workload that computes the same thing agree exactly, so a
    count that fell because the body collapsed is caught by the comparison the
    caller makes before this one, and a count that fell because the code got
    faster is the outcome this gate exists to allow.
    """
    delta = (head.irefs - base.irefs) / base.irefs
    ceiling = float(step["ceiling"]) if step else RELATIVE_TOLERANCE
    print(f"base:     {base.irefs:,} instructions ({subject})")
    print(f"head:     {head.irefs:,} instructions")
    print(f"delta:    {delta:+.2%} (ceiling +{ceiling:.0%})")
    if head.mispredicts is not None and base.mispredicts:
        moved = (head.mispredicts - base.mispredicts) / base.mispredicts
        print(
            f"branch mispredicts: {base.mispredicts:,} -> {head.mispredicts:,} "
            f"({moved:+.2%}; simulated, read and not gated)"
        )
    if step:
        print(f"recorded step from this base: {step['why']}")
    if delta > ceiling:
        print("REGRESSION: this change costs more than its own merge base.")
        print("Both counts were measured in this job, with one toolchain, so the")
        print("difference is the change rather than the environment.")
        return 1
    print("OK: no regression against the base.")
    return 0


def absent_at(checkout: Path, mode: str) -> bool:
    """Whether the workload this mode measures is missing from `checkout`.

    A shape added in the change being measured has no counterpart at the base,
    and building one there fails with "no example target". That is not a
    regression and not a pass: there is no comparison to make, and saying so is
    the honest reading, and `judge_new` holds such a shape to its recorded
    count instead, in the same run.

    Asked of the source rather than of cargo's message, so a genuine build
    failure at the base stays a failure rather than reading as a new shape.

    A binding shape is an argument of one example, and the base can carry the
    example without the shape: its parser has no arm for the name, and the
    binary refuses the argument. That is a shape the base does not carry, read
    where the base names its shapes rather than from the refusal. A core
    reading is the same: the example names it in an arm of its own, and a base
    whose example has no such arm carries the example and not the shape.
    """
    example = MODES[mode][0]
    sources = list(checkout.glob(f"crates/*/examples/{example}.rs"))
    if not sources:
        return True
    reading = CORE_READINGS.get(mode)
    if reading is not None:
        return f'Some("{reading}")' not in sources[0].read_text(encoding="utf-8")
    shape = BINDING_SHAPES.get(mode)
    if shape is None:
        return False
    crate = sources[0].parent.parent
    return not any(
        f'"{shape}" =>' in path.read_text(encoding="utf-8")
        for path in crate.glob("src/**/*.rs")
    )


def rerecorded_at(checkout: Path, record: dict, mode: str) -> bool:
    """Whether this change records another workload for `mode` than `checkout`.

    A workload that moved between the two commits is no comparison, and the
    commit that moves one re-records its count and checksum in the same change.
    A shape whose recorded checksum differs from the one the base records, or
    that the base records none for, is that: a workload replaced on the record,
    which the base is built with (`transplant_workloads`) and `judge_replaced`
    judges. A checksum that moved with nothing re-recorded reads as a workload
    that changed.
    """
    key = checksum_key(mode)
    try:
        carried = json.loads(
            (checkout / "scripts" / "perf_budget.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return False
    return key in record and carried.get(key) != record[key]


def transplant_workloads(checkout: Path, modes: Sequence[str]) -> None:
    """Put this change's workload source for each of `modes` into `checkout`.

    A shape whose workload this change re-records has no base count of the work
    it measures, and the base's own example measures other work. Built at the
    base from this change's source instead, the workload measures the same work
    against the code it runs on there, which is the comparison the gate exists
    for. Where the base cannot build it -- the source names something the base
    does not have -- the base reading fails, and `judge_replaced` falls back to
    the record.
    """
    sources = [
        source
        for example in sorted({MODES[mode][0] for mode in modes})
        for source in ROOT.glob(f"crates/*/examples/{example}.rs")
    ]
    if any(mode in BINDING_SHAPES for mode in modes):
        # A binding shape's example only calls the extension's own workload,
        # which is where what it measures is written.
        sources.append(BINDING_WORKLOAD)
    for source in sources:
        target = checkout / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def judge_replaced(
    head: Measurement,
    base: Measurement | None,
    budget: dict,
    mode: str,
    step: dict | None = None,
) -> int:
    """Judge a shape whose workload this change re-records.

    The base was built with this change's workload source
    ([`transplant_workloads`]). Where it built and computed what the change
    computes, the two counts are of one work and the ordinary ceiling holds;
    where it did not, there is no base count, and the shape is held to the
    count recorded with it, as a new shape is.
    """
    subject = MODES[mode][1]
    print("RE-RECORDED: this change records another workload for the shape than")
    print("the base does, so the base is built with this change's workload.")
    if base is not None and base.checksum == head.checksum:
        return judge_relative(head, base, subject, step)
    if base is None:
        print("The base cannot build it, so it is held to the count recorded")
        print("with it instead.")
    else:
        print(f"The base computes other work with it (checksum {base.checksum},")
        print(f"head {head.checksum}), so it is held to the count recorded with it.")
    return judge_recorded(head, budget, mode)


def measure_base(
    modes: list[str], checkout: Path, target: Path
) -> tuple[dict[str, Measurement], dict[str, BaseException]]:
    """Measure each mode at the base, and name the ones the base could not give.

    A base that cannot build or run a workload says nothing about the change
    being measured: the rig failed, not the tree. [`absent_at`](absent_at) reads
    the shape a base does not *carry*; this reads the shape it carries and
    cannot produce -- a toolchain its lockfile will not build, an example whose
    own dependency moved. Both are "no comparison to make". The first is a new
    shape, held to its recorded count; the second is the rig's failure rather
    than the change's, and the run exits 2.
    """
    readings = {mode: measured_or_blamed(mode, checkout, target) for mode in modes}
    measured = {
        mode: read for mode, read in readings.items() if isinstance(read, Measurement)
    }
    broken = {
        mode: read for mode, read in readings.items() if isinstance(read, BaseException)
    }
    return measured, broken


def measured_or_blamed(
    mode: str, checkout: Path, target: Path
) -> Measurement | BaseException:
    """One mode's reading at the base, or what stopped the base from giving it."""
    try:
        return measure_mode(mode, checkout, target)
    except (subprocess.CalledProcessError, SystemExit) as err:
        return err


def run_relative(modes: list[str], rev: str) -> int:
    """Measure this checkout and `rev` side by side, and hold each difference.

    The base is built in a throwaway worktree with its own target directory, so
    the two builds neither share nor overwrite each other's output, and both use
    whatever toolchain is on PATH -- which is the point: a recorded count is a
    number taken on a machine that is not this one.

    Several shapes are measured against **one** base build. Building the base
    once per shape is what kept the gate to a single shape: the build is minutes
    and the measurement is seconds, so a second shape costs almost nothing while
    a second invocation costs another build.

    The verdicts are aggregated by severity, and every shape is measured before
    any is judged, so one regression does not hide the next one's number.
    """
    sha = resolve_rev(rev)
    record = json.loads(BUDGET_FILE.read_text(encoding="utf-8"))
    head = {mode: measure_mode(mode) for mode in modes}
    worktree = Path(tempfile.mkdtemp(prefix="valgebra-perf-base-"))
    checkout = worktree / "tree"
    try:
        subprocess.run(
            ["git", "worktree", "add", "--detach", "--quiet", str(checkout), sha],
            cwd=ROOT,
            check=True,
        )
        fresh = [mode for mode in modes if absent_at(checkout, mode)]
        replaced = [
            mode
            for mode in modes
            if mode not in fresh and rerecorded_at(checkout, record, mode)
        ]
        carried = carried_steps(checkout)
        transplant_workloads(checkout, replaced)
        base, broken = measure_base(
            [mode for mode in modes if mode not in fresh],
            checkout,
            worktree / "target",
        )
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
        if mode in fresh:
            outcomes.append(judge_new(head[mode], record, mode))
            continue
        if mode in replaced:
            step = recorded_step(mode, record.get("steps", []), carried)
            outcomes.append(
                judge_replaced(head[mode], base.get(mode), record, mode, step)
            )
            continue
        if mode in broken:
            print(f"head:     {head[mode].irefs:,} instructions")
            print("NOT COMPARABLE: the base carries this workload and could not")
            print(f"produce it ({broken[mode]}). That is the rig rather than this")
            print("change, and no verdict on it.")
            outcomes.append(EXIT_CANNOT_RUN)
            continue
        step = recorded_step(mode, record.get("steps", []), carried)
        outcomes.append(judge_relative(head[mode], base[mode], subject, step))
    outcome = worst(outcomes)
    if batch and outcome == EXIT_FAIL:
        print("\nThe ceiling is written for one commit and this reading covers")
        print("more, so the first step is `git bisect` over the window rather")
        print("than a repair to the tip.")
    elif batch:
        print("\nA pass over a batch is a pass on the sum: a regression inside it")
        print("cancelled by an improvement beside it reads exactly like this.")
    return outcome


def worst(outcomes: Sequence[int]) -> int:
    """Rank verdicts by severity: could not run, then failed, then passed."""
    if EXIT_CANNOT_RUN in outcomes:
        return EXIT_CANNOT_RUN
    if EXIT_FAIL in outcomes:
        return EXIT_FAIL
    return EXIT_OK


def judge_relative(
    head: Measurement, base: Measurement, subject: str, step: dict | None = None
) -> int:
    """Compare two measurements of the same workload, or refuse to compare them.

    The checksum is asked first, as it is for a new shape and for a sharper
    reason: two runs of a workload that computes the same thing agree
    exactly, so a disagreement says the workload itself moved between the two
    commits and the counts are of different work. That is not a regression and
    not a pass -- it is a comparison that cannot be made, which is exit code 2.
    """
    if head.checksum != base.checksum:
        print(f"WORKLOAD CHANGED: base checksum {base.checksum}, head {head.checksum}.")
        print("The two runs measured different work, so their counts do not")
        print("compare. Re-record the count and checksum with --update in the")
        print("commit that changes the workload, which holds the shape to that")
        print("record, and say what moved.")
        return EXIT_CANNOT_RUN
    print(f"checksum: {head.checksum} ({subject}, unchanged from the base)")
    return check_against_base(head, base, subject, step)


def record_key(mode: str) -> str:
    """Name the key a shape's count is recorded under in the budget file."""
    return f"{mode.replace('-', '_')}_workload_irefs"


def checksum_key(mode: str) -> str:
    """Name the key a shape's checksum is recorded under.

    For a binding shape it is what one iteration folds, since the count of
    iterations is the gate's to choose and `measure_mode` asserts it.
    """
    return f"{mode.replace('-', '_')}_workload_checksum"


def record_mode(budget: dict, mode: str) -> None:
    """Measure one shape and write its count and its checksum."""
    measured = measure_mode(mode)
    subject = MODES[mode][1]
    if mode in BINDING_SHAPES:
        hi, lo = BINDING_ITERATIONS[mode]
        print(
            f"{subject} over {hi - lo:,} iterations (difference of {hi:,} and {lo:,})"
        )
    budget[checksum_key(mode)] = measured.checksum
    if measured.mispredicts is not None:
        print(
            f"branch mispredicts: {measured.mispredicts:,} "
            "(simulated, read and not gated)"
        )
    budget[record_key(mode)] = measured.irefs
    record_budget(budget)
    print(f"recorded {subject}: {measured.irefs:,} instructions")


def judge_new(head: Measurement, budget: dict, mode: str) -> int:
    """Hold a shape the base does not carry to the count recorded for it.

    There is no base count to compare with, so the record stands in: the count
    the commit adding the shape took, inside the band a count needs to survive
    a change of machine. That band cannot see a 5% regression and does not need
    to here, since the next change has this one as its base and is compared
    with it at 2%. What it sees is a shape measuring other work than the one
    recorded, and a workload that stopped doing the work.
    """
    print("NEW SHAPE: the base does not carry this workload, so there is no")
    print("comparison to make. It is held to its recorded count instead.")
    return judge_recorded(head, budget, mode)


def judge_recorded(head: Measurement, budget: dict, mode: str) -> int:
    """Hold a shape with no base count to the count and checksum recorded for it.

    A new shape and a workload this change re-recorded are both read here: the
    checksum first, since a count inside the band of other work is other work,
    and then the band the record carries for a change of machine.
    """
    subject = MODES[mode][1]
    recorded = budget.get(record_key(mode))
    expected = None if mode in BINDING_SHAPES else budget.get(checksum_key(mode))
    if recorded is None or (mode not in BINDING_SHAPES and expected is None):
        print(f"No count is recorded for the {subject}: the commit that adds a")
        print("shape records it with --update.")
        return EXIT_CANNOT_RUN
    if expected is not None and check_checksum(head.checksum, int(expected), subject):
        return EXIT_FAIL
    band = budget["binding_tolerance" if mode in BINDING_SHAPES else "tolerance"]
    return check_against_budget(head.irefs, int(recorded), float(band))


COMPARE_GATE = ROOT / "scripts" / "compare_gate.py"

#: The pair of call counts each of the comparison gate's shapes is counted at on
#: a profiled wheel, chosen so a difference is tens of millions of instructions
#: and a run a few seconds under cachegrind. A pair, not a count per call: a
#: shape's cost per call moves a little with how many calls a run makes -- the
#: list of ten thousand integers reads 210,083 a call over 300 and 100 calls
#: and 210,204 over 1,000 and 300 -- so a count is exact for its pair alone.
PROFILED_CALLS = {
    "scalar": (200_000, 50_000),
    "large_array": (300, 100),
    "wide_record": (20_000, 5_000),
    "deep_nesting": (50_000, 15_000),
    "json_document": (100, 30),
    "build": (2_000, 500),
    "error_report": (5_000, 1_500),
}


def snapshot(root: Path = ROOT) -> str:
    """Name the checkout as it stands -- changes, new files and all -- as a tree.

    The profiled wheels are built from trees unpacked into one directory, so
    the change has to be a tree as the base is. The files are written to an
    index of their own, so neither the caller's index nor its branch moves, and
    what `.gitignore` leaves out stays out, as it does of a commit.
    """
    with tempfile.TemporaryDirectory() as scratch:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(scratch) / "index")}
        for args in (["read-tree", "HEAD"], ["add", "--all"]):
            subprocess.run(["git", *args], cwd=root, check=True, env=env)
        return subprocess.run(
            ["git", "write-tree"],
            cwd=root,
            check=True,
            env=env,
            capture_output=True,
            text=True,
        ).stdout.strip()


def unpack(tree: str, into: Path, root: Path = ROOT) -> None:
    """Write the files of `tree` into the empty directory `into`."""
    archive = subprocess.Popen(
        ["git", "archive", "--format=tar", tree], cwd=root, stdout=subprocess.PIPE
    )
    subprocess.run(["tar", "-x", "-C", str(into)], stdin=archive.stdout, check=True)
    if archive.stdout is not None:
        archive.stdout.close()
    if archive.wait() != 0:
        raise subprocess.CalledProcessError(archive.returncode, "git archive")


def build_profiled(tree: str, scratch: Path, out: Path) -> Path:
    """Build the profile-guided wheel of `tree` in `scratch`, and return it.

    The tree is unpacked into the same directory and built into the same,
    emptied, target directory whichever side it is, so the two builds differ in
    their sources and nothing else. The target is emptied rather than reused:
    a profile is a file the compiler flags name by path, so a dependency
    optimised by the first side's profile would read as current to the second.
    The training runs with its hash seed fixed, so a profile is the tree's.
    """
    maturin = shutil.which("maturin")
    if maturin is None:
        print("perf_gate: maturin is not on PATH, so no profiled wheel can be built")
        raise SystemExit(EXIT_CANNOT_RUN)
    source, target = scratch / "tree", scratch / "target"
    for path in (source, target):
        shutil.rmtree(path, ignore_errors=True)
    source.mkdir()
    unpack(tree, source)
    env = {**os.environ, "CARGO_TARGET_DIR": str(target), "PYTHONHASHSEED": "0"}
    subprocess.run(
        [
            maturin,
            *("build", "--release", "--pgo", "--out", str(out)),
            *("--interpreter", sys.executable),
        ],
        cwd=source,
        check=True,
        env=env,
    )
    wheels = sorted(out.glob("*.whl"))
    if len(wheels) != 1:
        print(f"perf_gate: the profiled build left {len(wheels)} wheels in {out}")
        raise SystemExit(EXIT_CANNOT_RUN)
    return wheels[0]


def bare_environment(where: Path) -> Path:
    """Make an environment holding nothing but the standard library; return its Python.

    The shapes are counted with no dependency installed, so what a run imports
    is the interpreter's and the wheel's, and the same on both sides.
    """
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(where)], check=True
    )
    return where / "bin" / "python"


def install(wheel: Path, python: Path) -> None:
    """Put `wheel` in the environment of `python`, in place of any it holds.

    A wheel is unpacked rather than installed by a tool, which would be one
    more thing the job needs: the package and its metadata are all a run reads.
    """
    site = Path(
        subprocess.run(
            [
                str(python),
                "-c",
                "import sysconfig; print(sysconfig.get_path('platlib'))",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    for old in [*site.glob("valgebra"), *site.glob("valgebra-*.dist-info")]:
        shutil.rmtree(old)
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(site)


def measure_profiled(python: Path, shape: str) -> Measurement:
    """Count one shape's valgebra half on the installed wheel, as a difference.

    The interpreter's start-up and the shapes' construction cancel between the
    two counts; the checksum is the number of calls that did the work, which
    must be every call. One uncounted call first compiles what both counted
    runs import, as `warm_the_interpreter` does for a binding shape.
    """
    hi, lo = PROFILED_CALLS[shape]
    environment = workload_environment(os.environ)
    del environment["PYTHONDONTWRITEBYTECODE"]
    subprocess.run(
        [str(python), str(COMPARE_GATE), "--count", shape, "1"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        env=environment,
    )
    high = measure(python, str(COMPARE_GATE), "--count", shape, str(hi))
    low = measure(python, str(COMPARE_GATE), "--count", shape, str(lo))
    for result, calls in ((high, hi), (low, lo)):
        if check_checksum(result.checksum, calls, f"profiled {shape} x{calls:,}"):
            raise SystemExit(EXIT_FAIL)
    return Measurement(high.irefs - low.irefs, hi - lo)


def profiled_side(
    tree: str, scratch: Path, python: Path, side: str
) -> dict[str, Measurement] | BaseException:
    """Build one side's profiled wheel and count every shape on it."""
    try:
        install(build_profiled(tree, scratch, scratch / f"wheel-{side}"), python)
        return {shape: measure_profiled(python, shape) for shape in PROFILED_CALLS}
    except (subprocess.CalledProcessError, SystemExit) as err:
        return err


def run_profiled(rev: str) -> int:
    """Count the comparison gate's shapes on a profiled wheel here and at `rev`.

    The base is built second, in the directory the change was built in, so a
    base that cannot be built leaves the change's counts read and says why the
    comparison is the rig's failure. Every shape is counted before any is
    judged, as `run_relative` does.
    """
    sha = resolve_rev(rev)
    tree = snapshot()
    scratch = Path(tempfile.mkdtemp(prefix="valgebra-perf-profiled-"))
    try:
        python = bare_environment(scratch / "env")
        head = profiled_side(tree, scratch, python, "head")
        base = profiled_side(sha, scratch, python, "base")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    print(f"comparing profiled wheels against {rev} ({sha[:12]})")
    print(describe_window(commits_since(sha))[0])
    if isinstance(head, BaseException):
        print(
            f"NOT MEASURED: this checkout's profiled wheel could not be read ({head})."
        )
        return EXIT_CANNOT_RUN
    outcomes = []
    for shape, measured in head.items():
        subject = f"profiled {shape}"
        print(f"\n--- {subject}")
        if isinstance(base, BaseException):
            print(f"head:     {measured.irefs:,} instructions")
            print("NOT COMPARABLE: the base's profiled wheel could not be built or")
            print(f"read ({base}). That is the rig rather than this change.")
            outcomes.append(EXIT_CANNOT_RUN)
            continue
        outcomes.append(judge_relative(measured, base[shape], subject))
    return worst(outcomes)


def _arguments(argv: Sequence[str]) -> argparse.Namespace:
    """Read the command line, refusing a flag it does not know.

    A mistyped shape read as no shape would measure the core workload instead,
    and answer for a measurement nobody asked for. `argparse` refuses it with exit
    2, which is this gate's "could not run".
    """
    parser = arguments(__doc__)
    for name, (_, label) in MODES.items():
        parser.add_argument(f"--{name}", action="store_true", help=f"the {label}")
    parser.add_argument(
        "--profiled",
        action="store_true",
        help="the comparison gate's shapes, counted on a profile-guided wheel",
    )
    # One or the other: a command line naming neither has nothing to compare
    # with, and `argparse` refuses it with exit 2.
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--against", metavar="REV", help="compare with the same shapes built at REV"
    )
    action.add_argument(
        "--update", action="store_true", help="record each shape's count"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] = ()) -> int:
    """Measure the flagged shapes, or say why they could not be measured.

    A build or a run of this change's own workload that does not finish is no
    regression and no pass: nothing was counted, so it is exit code 2 with the
    command that stopped, rather than a traceback and 1. The base's are blamed
    where they are measured, beside the change's counts.
    """
    try:
        return measure_flagged(_arguments(argv))
    except subprocess.CalledProcessError as err:
        print(f"perf_gate: could not measure: {err}")
        return EXIT_CANNOT_RUN


def measure_flagged(args: argparse.Namespace) -> int:
    """Run what the command line asked for: a comparison, or a re-record."""
    # `--binding` keeps its meaning -- the walk -- so an existing invocation and
    # the count recorded under it still name the same measurement.
    chosen = vars(args)
    flagged = [name for name in MODES if chosen[name.replace("-", "_")]]
    if args.profiled:
        if args.against is None:
            print("perf_gate: the profiled shapes are compared with a base and never")
            print("recorded, since no record could rebuild the wheel they count.")
            return EXIT_CANNOT_RUN
        outcomes = [run_profiled(args.against)]
        if flagged:
            outcomes.append(run_relative(flagged, args.against))
        return worst(outcomes)
    flagged = flagged or ["core"]
    if args.against is not None:
        # Every flagged shape against one base build, since the build is the
        # expensive half and the shapes share it.
        return run_relative(flagged, args.against)
    budget = json.loads(BUDGET_FILE.read_text(encoding="utf-8"))
    for mode in flagged:
        if len(flagged) > 1:
            print(f"\n--- {MODES[mode][1]}")
        record_mode(budget, mode)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
