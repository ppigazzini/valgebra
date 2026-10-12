"""The instruction-count gate must fail on every way a measurement can be wrong.

A gate that cannot be shown to fail is not evidence. These drive the decision
logic of ``scripts/perf_gate.py`` directly -- no cachegrind, no build -- so each
refusal is exercised: a change dearer than its base, a workload that changed
between the two, a new shape over or under its recorded band (under it is the
shape a workload that stopped doing the work produces), a workload checksum that
does not match, and output the gate cannot read at all.

The under-floor case is the one a one-sided band misses: a hollow workload
measures low and reads as an improvement it never earned. And a command line
naming nothing to compare with is refused, since the recorded counts are a
record rather than a gate.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from _tree import ROOT, lay_script, load_script, read_workflow

# The repository checks are not the product suite: this file reads the tree,
# the configuration and the gate scripts, none of which ship in a wheel.
pytestmark = pytest.mark.repository

if TYPE_CHECKING:
    from types import ModuleType


def _load_gate() -> ModuleType:
    """Import the gate by path; ``scripts/`` is not an importable package."""
    return load_script("perf_gate")


gate = _load_gate()

BUDGET = 1_000_000
TOLERANCE = 0.10


def test_every_recorded_step_names_a_shape_and_says_why() -> None:
    budget = json.loads((ROOT / "scripts" / "perf_budget.json").read_text("utf-8"))
    for step in budget.get("steps", []):
        assert step["shape"] in gate.MODES, step["shape"]
        assert step["why"].strip(), step
        assert 0 < float(step["ceiling"]) < 5


def test_a_recorded_step_excuses_the_comparison_that_crosses_it() -> None:
    """A step excuses a comparison whose base does not carry it, and no other.

    The commit that takes a step records it, so a base before that commit has
    no such record and a base at or past it has one: that second comparison is
    held to the ordinary ceiling. A base count near the step's own is not a
    match, since a later change giving part of a step back measures there, and
    a step read off the count would excuse every change after it.
    """
    step = {
        "shape": "decision-refute",
        "base_irefs": 1_196_378,
        "ceiling": 0.04,
        "why": "the count and the cost moved apart",
    }
    assert gate.recorded_step("decision-refute", [step], []) == step
    assert gate.recorded_step("decision-refute", [step], [step]) is None
    # Named by its shape and its base, so a reworded argument is the same step.
    reworded = dict(step, why="the argument, said again")
    assert gate.recorded_step("decision-refute", [reworded], [step]) is None
    assert gate.recorded_step("decision", [step], []) is None

    # And the verdict it moves: 3% over the base passes under the step's
    # ceiling, and fails once the base carries the step.
    head = gate.Measurement(irefs=1_232_270, checksum=200)
    base = gate.Measurement(irefs=1_196_378, checksum=200)
    crossing = gate.recorded_step("decision-refute", [step], [])
    assert gate.judge_relative(head, base, "refute", crossing) == gate.EXIT_OK
    carried = gate.recorded_step("decision-refute", [step], [step])
    assert gate.judge_relative(head, base, "refute", carried) == gate.EXIT_FAIL


def test_a_base_reads_the_steps_its_own_record_carries(tmp_path: Path) -> None:
    # A base older than the record has none, and every step of the head's is
    # one the comparison crosses.
    assert gate.carried_steps(tmp_path) == []
    (tmp_path / "scripts").mkdir()
    record = tmp_path / "scripts" / "perf_budget.json"
    step = {"shape": "decision", "base_irefs": 1, "ceiling": 0.04, "why": "x"}
    record.write_text(json.dumps({"steps": [step]}), encoding="utf-8")
    assert gate.carried_steps(tmp_path) == [step]
    record.write_text(json.dumps({"tolerance": 0.1}), encoding="utf-8")
    assert gate.carried_steps(tmp_path) == []


def test_a_step_the_file_has_outlived_is_refused() -> None:
    """A record that outlives what it records is worse than no record.

    A step describes one transition -- from a base, by at most a ceiling -- and
    is worth keeping while the shape's budget is somewhere inside it. The
    `binding-build` record below is the one this rule was written for: its shape
    was re-recorded at less than half the base it steps from, for a reason that
    had nothing to do with the step, and the record sat in the file reading as
    an argument somebody still stood behind.
    """
    budget = json.loads((ROOT / "scripts" / "perf_budget.json").read_text("utf-8"))
    assert not gate.stale_steps(budget), (
        "these recorded steps name a transition the shape can no longer take; "
        "remove each with the reason, as the changelog roll is trimmed"
    )

    outlived = {
        "binding_build_workload_irefs": 222_939_220,
        "steps": [
            {
                "shape": "binding-build",
                "base_irefs": 464_986_514,
                "ceiling": 1.7,
                "why": "the instrument changed and the build did not",
            }
        ],
    }
    assert gate.stale_steps(outlived), "the rule cannot see the record it is for"
    # And the live shape of the same record: a budget inside the window stays.
    inside = dict(outlived, binding_build_workload_irefs=1_181_988_860)
    assert not gate.stale_steps(inside)


def test_a_command_line_names_a_base_or_a_recording() -> None:
    """The gate compares a change with its base, or records; it judges nothing else.

    A bare invocation would measure the core workload against its recorded
    count, a reading of another machine, and answer as if it were a gate.
    `argparse` refuses it with exit 2, the gate's "could not run", and refuses
    the two together, since a comparison records nothing.
    """
    for argv in ([], ["--binding"], ["--against", "HEAD", "--update"]):
        with pytest.raises(SystemExit) as excinfo:
            gate._arguments(argv)  # noqa: SLF001
        assert excinfo.value.code == gate.EXIT_CANNOT_RUN, argv
    assert gate._arguments(["--against", "HEAD"]).against == "HEAD"  # noqa: SLF001
    assert gate._arguments(["--update", "--decision"]).update  # noqa: SLF001


def _record() -> dict:
    return {
        "core_workload_irefs": BUDGET,
        "core_workload_checksum": 134000,
        "tolerance": TOLERANCE,
        "binding_set_workload_irefs": BUDGET,
        "binding_tolerance": 0.3,
    }


def test_a_new_shape_is_held_to_its_recorded_count() -> None:
    """A shape the base does not carry is judged against the record, both ways.

    There is no base count, so the count the commit adding the shape recorded
    stands in, inside the band a change of machine needs. Under it is the
    workload that stopped doing the work; over it, a shape measuring other work.
    """
    record = _record()
    core = gate.Measurement(irefs=1_050_000, checksum=134000)
    assert gate.judge_new(core, record, "core") == gate.EXIT_OK
    over = gate.Measurement(irefs=1_100_001, checksum=134000)
    assert gate.judge_new(over, record, "core") == gate.EXIT_FAIL
    hollow = gate.Measurement(irefs=899_999, checksum=134000)
    assert gate.judge_new(hollow, record, "core") == gate.EXIT_FAIL
    # The checksum first: a count inside the band of other work is other work.
    other = gate.Measurement(irefs=BUDGET, checksum=133999)
    assert gate.judge_new(other, record, "core") == gate.EXIT_FAIL
    # A binding shape carries the wider band, and its checksum is its own.
    wide = gate.Measurement(irefs=1_250_000, checksum=134000)
    assert gate.judge_new(wide, record, "binding-set") == gate.EXIT_OK
    assert gate.judge_new(wide, record, "core") == gate.EXIT_FAIL


def test_a_new_shape_with_no_recorded_count_cannot_be_judged() -> None:
    # Neither a pass nor a regression: the commit adding a shape records it, and
    # one that did not left the gate nothing to read.
    record = _record()
    read = gate.Measurement(irefs=BUDGET, checksum=134000)
    assert gate.judge_new(read, record, "binding-deep") == gate.EXIT_CANNOT_RUN
    del record["core_workload_checksum"]
    assert gate.judge_new(read, record, "core") == gate.EXIT_CANNOT_RUN


def test_a_count_inside_the_band_passes() -> None:
    assert gate.check_against_budget(BUDGET, BUDGET, TOLERANCE) == 0
    assert gate.check_against_budget(1_050_000, BUDGET, TOLERANCE) == 0
    assert gate.check_against_budget(950_000, BUDGET, TOLERANCE) == 0


def test_a_count_over_the_ceiling_fails() -> None:
    assert gate.check_against_budget(1_100_001, BUDGET, TOLERANCE) == 1


def test_a_count_under_the_floor_fails() -> None:
    # The shape a workload that stopped doing the work produces. A ceiling-only
    # budget calls this "within budget" and publishes an improvement it never
    # earned, which is the failure this direction exists to catch.
    assert gate.check_against_budget(899_999, BUDGET, TOLERANCE) == 1
    assert gate.check_against_budget(0, BUDGET, TOLERANCE) == 1


def test_a_mismatched_checksum_fails() -> None:
    assert gate.check_checksum(134_000, 134_000, "core workload") == 0
    assert gate.check_checksum(0, 134_000, "core workload") == 1
    assert gate.check_checksum(133_999, 134_000, "core workload") == 1


def test_an_unreadable_instruction_count_is_not_a_pass() -> None:
    with pytest.raises(SystemExit) as excinfo:
        gate.parse_measurement("checksum=1\n", "cachegrind died before its banner")
    assert excinfo.value.code == 2


def test_an_unreadable_checksum_is_not_a_pass() -> None:
    # The workload ran under cachegrind and printed nothing the gate can tie to
    # the work; a count with no checksum behind it is not a verdict.
    with pytest.raises(SystemExit) as excinfo:
        gate.parse_measurement("", "==1== I   refs:      1,234,567")
    assert excinfo.value.code == 2


def test_a_readable_measurement_carries_both_halves() -> None:
    measured = gate.parse_measurement(
        "checksum=134000\n", "==1== I   refs:      252,026,154"
    )
    assert measured.irefs == 252026154
    assert measured.checksum == 134000
    # The binding workload prints a bare number rather than a labelled one.
    bare = gate.parse_measurement("150000\n", "==1== I   refs:      1,000")
    assert bare.checksum == 150000


def test_a_workload_runs_without_the_callers_environment() -> None:
    """A measurement sees the loader path, a home, valgrind's override and a seed.

    The environment's size moves a count on its own -- one binary read 2.97% apart
    between a login shell and `uv run` -- so everything the caller carries that a
    workload does not need is left behind, and the seed is fixed whatever the
    caller set it to. No run writes bytecode: the first run on a fresh
    interpreter would otherwise compile a module its partner run reads back, and
    the difference of the two would count the compile.
    """
    caller = {
        "PATH": "/home/u/.venv/bin:/usr/bin",
        "VIRTUAL_ENV": "/home/u/.venv",
        "UV_RUN_RECURSION_DEPTH": "1",
        "HOME": "/home/u",
        "LD_LIBRARY_PATH": "/opt/python/lib",
        "PYTHONHOME": "/opt/python",
        "PYTHONHASHSEED": "random",
    }
    assert gate.workload_environment(caller) == {
        "LD_LIBRARY_PATH": "/opt/python/lib",
        "PYTHONHOME": "/opt/python",
        "PYTHONHASHSEED": "0",
        "GLIBC_TUNABLES": "glibc.malloc.mxfast=0",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    assert gate.workload_environment({}) == {
        "PYTHONHASHSEED": "0",
        "GLIBC_TUNABLES": "glibc.malloc.mxfast=0",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    # A caller's own allocator setting is the caller's, and is replaced.
    tuned = gate.workload_environment({"GLIBC_TUNABLES": "glibc.malloc.tcache_count=0"})
    assert tuned["GLIBC_TUNABLES"] == "glibc.malloc.mxfast=0"


def test_the_budget_file_names_the_toolchain_its_counts_were_taken_with() -> None:
    """A recorded count is a reading of one valgrind and one C library."""
    budget = json.loads((ROOT / "scripts" / "perf_budget.json").read_text())
    assert set(budget["measured_with"]) == {"valgrind", "libc"}


def test_a_branch_simulated_run_carries_its_mispredicts() -> None:
    """A run with branch simulation reads its mispredicts; one without has none."""
    stderr = (
        "==1== I   refs:      89,565,334\n"
        "==1== Branches:       10,196,616  (9,135,548 cond + 1,061,068 ind)\n"
        "==1== Mispredicts:     1,080,406  (  512,000 cond +   568,406 ind)\n"
    )
    measured = gate.parse_measurement("68000\n", stderr)
    assert measured.mispredicts == 1080406
    plain = gate.parse_measurement("68000\n", "==1== I   refs: 1,000")
    assert plain.mispredicts is None


def test_mispredicts_are_read_beside_the_count_and_never_gate_it() -> None:
    """A change that doubles the mispredicts and keeps its count passes.

    The column is there so a trade of instructions for predicted branches is
    seen where it is judged, and the predictor cachegrind simulates is older
    than the processors the lanes run on, so it informs and does not decide.
    """
    base = gate.Measurement(100_000_000, 134000, 1_000_000)
    head = gate.Measurement(100_000_000, 134000, 2_000_000)
    assert gate.check_against_base(head, base, "matrix decision workload") == 0
    assert set(gate.MODES) >= gate.BRANCH_MODES
    assert all(mode.startswith("decision") for mode in gate.BRANCH_MODES)


def test_the_committed_core_checksum_matches_the_workload() -> None:
    # The recorded checksum is a constant of the workload's fixed corpus and
    # iteration count, so it belongs in the tree beside the budget. If the
    # workload is edited without re-recording, this fails before CI spends a
    # cachegrind run finding out.
    import json  # noqa: PLC0415

    budget = json.loads(
        (ROOT / "scripts" / "perf_budget.json").read_text(encoding="utf-8")
    )
    assert budget["core_workload_checksum"] == 130000


BASE = gate.Measurement(irefs=100_000_000, checksum=134000)


def _relative(head_irefs: int, checksum: int = 134000) -> int:
    return gate.judge_relative(
        gate.Measurement(irefs=head_irefs, checksum=checksum), BASE, "core workload"
    )


def test_a_regression_against_the_base_fails() -> None:
    # The number the report asks this gate to catch and the recorded budget
    # cannot: three percent, well inside the +/-10% an absolute band must carry
    # to survive a change of machine, and well outside what one job's two builds
    # of one toolchain can produce by themselves.
    assert _relative(103_000_000) == 1


def test_a_change_inside_the_band_passes() -> None:
    assert _relative(101_000_000) == 0
    assert _relative(BASE.irefs) == 0


def test_an_improvement_against_the_base_passes() -> None:
    # One-sided on purpose. A count that fell because the work vanished is
    # caught by the checksum below; a count that fell because the code got
    # faster is what the gate is for.
    assert _relative(80_000_000) == 0


def test_a_workload_that_changed_is_not_a_comparison() -> None:
    # Neither a pass nor a regression: the two runs measured different work, so
    # exit 2 -- "could not measure" -- which is the code a lane must not read as
    # a verdict.
    assert _relative(100_000_000, checksum=134001) == gate.EXIT_CANNOT_RUN


def test_a_workload_the_change_re_records_is_held_to_its_record(
    tmp_path: Path,
) -> None:
    """A pure-Rust shape whose recorded checksum moved is a replaced workload.

    The commit that changes a workload records the new count and checksum, so
    the base records another checksum for the shape than the change does. That
    is no comparison and no failure: the shape is held to its record, as a new
    one is. A checksum the change did not re-record is a workload that changed,
    and a binding shape records no checksum to read.
    """
    record = _record()
    base = tmp_path / "base"
    (base / "scripts").mkdir(parents=True)
    carried = base / "scripts" / "perf_budget.json"
    assert not gate.rerecorded_at(base, record, "core"), "no record at the base"
    carried.write_text(
        json.dumps({**record, "core_workload_checksum": 133000}), encoding="utf-8"
    )
    assert gate.rerecorded_at(base, record, "core")
    assert not gate.rerecorded_at(base, record, "binding-set")
    carried.write_text(json.dumps(record), encoding="utf-8")
    assert not gate.rerecorded_at(base, record, "core")
    carried.write_text("{", encoding="utf-8")
    assert not gate.rerecorded_at(base, record, "core"), "an unreadable record"

    read = gate.Measurement(irefs=1_050_000, checksum=134000)
    assert gate.judge_recorded(read, record, "core") == gate.EXIT_OK
    other = gate.Measurement(irefs=BUDGET, checksum=133000)
    assert gate.judge_recorded(other, record, "core") == gate.EXIT_FAIL


def test_a_re_recorded_workload_is_compared_where_the_base_builds_it(
    tmp_path: Path,
) -> None:
    """A re-recorded shape is built at the base from this change's source.

    Where the base builds it and computes the same work, the two counts are of
    one work and the 2% ceiling holds, so re-recording a workload does not
    excuse a regression in the code it reads. Where the base cannot build it,
    or computes other work with it, the shape is held to its record.
    """
    record = _record()
    base = gate.Measurement(irefs=100_000_000, checksum=134000)
    slower = gate.Measurement(irefs=103_000_000, checksum=134000)
    assert gate.judge_replaced(slower, base, record, "core") == gate.EXIT_FAIL
    within = gate.Measurement(irefs=101_000_000, checksum=134000)
    assert gate.judge_replaced(within, base, record, "core") == gate.EXIT_OK

    recorded = gate.Measurement(irefs=1_050_000, checksum=134000)
    assert gate.judge_replaced(recorded, None, record, "core") == gate.EXIT_OK
    other = gate.Measurement(irefs=100_000_000, checksum=133000)
    assert gate.judge_replaced(recorded, other, record, "core") == gate.EXIT_OK
    off = gate.Measurement(irefs=2_000_000, checksum=134000)
    assert gate.judge_replaced(off, None, record, "core") == gate.EXIT_FAIL

    gate.transplant_workloads(tmp_path, ["core"])
    source = ROOT / "crates" / "valgebra-core" / "examples" / "perf_workload.rs"
    copied = tmp_path / source.relative_to(ROOT)
    assert copied.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")


def test_every_mode_names_an_example_the_tree_builds() -> None:
    # The relative gate builds by mode name, so a mode naming an example that
    # does not exist fails at build time in the lane rather than here. Held to
    # the tree instead: every example named is a file under examples/.
    for mode, (example, subject) in gate.MODES.items():
        # Which crate holds the example is the same question the gate asks when
        # it builds one: a binding shape needs the embedded interpreter and the
        # binding crate, everything else builds in the core.
        crate = "valgebra-py" if mode in gate.BINDING_SHAPES else "valgebra-core"
        path = ROOT / "crates" / crate / "examples" / f"{example}.rs"
        assert path.exists(), f"{mode} names {example}, which is not in the tree"
        assert subject


def test_the_usage_names_every_binding_shape_and_counts_them() -> None:
    """The module docstring is the gate's `--help`, and it drifts.

    A count written by hand beside a registry is a claim about the registry,
    and a shape added to the registry does not edit the sentence. So the count
    is read from the registry rather than written next to it.
    """
    usage = gate.__doc__ or ""
    # The backticked form, not the bare flag: `--binding` is a prefix of every
    # other shape's flag, so a bare substring test passes for it whatever the
    # text says, and `--binding-pattern` would be satisfied by a typo one
    # character longer. The usage writes each flag as `` `--flag` ``.
    named = {f"`--{shape}`" for shape in gate.BINDING_SHAPES}
    missing = sorted(flag for flag in named if flag not in usage)
    assert not missing, (
        f"binding shapes the usage does not name: {missing}. Every registered "
        "shape is a thing a caller can pass, so the help text names it."
    )
    words = {
        11: "eleven",
        12: "twelve",
        13: "thirteen",
        14: "fourteen",
        15: "fifteen",
        16: "sixteen",
        17: "seventeen",
        18: "eighteen",
        19: "nineteen",
        20: "twenty",
        21: "twenty-one",
        22: "twenty-two",
        23: "twenty-three",
        24: "twenty-four",
        25: "twenty-five",
        26: "twenty-six",
        27: "twenty-seven",
    }
    want = words.get(len(named))
    assert want, f"no spelling for {len(named)} shapes; add one above"
    assert f"{want} shapes" in usage, (
        f"the usage counts the binding shapes wrongly: there are {len(named)} "
        f"({want}). The sentence and the registry are one claim."
    )


def test_a_shape_the_base_does_not_name_is_absent_there(tmp_path: Path) -> None:
    """A base that carries the example but not the shape has no count to give.

    A binding shape is one argument of one example, parsed where the binding
    names its shapes. The base built for the comparison can be older than the
    shape: its example builds, and its binary refuses the name. That is a new
    shape and not a measurement, and the gate reads it from the base's source
    -- the arm that names the shape -- so it says "new shape" before it runs
    anything. The tree itself names every shape the gate can ask for.
    """
    mode = next(iter(gate.BINDING_SHAPES))
    example = gate.MODES[mode][0]
    base = tmp_path / "base"
    (base / "crates" / "valgebra-py" / "examples").mkdir(parents=True)
    (base / "crates" / "valgebra-py" / "src").mkdir(parents=True)
    assert gate.absent_at(base, mode), "no example, so no shape"
    (base / "crates" / "valgebra-py" / "examples" / f"{example}.rs").write_text("")
    assert gate.absent_at(base, mode), (
        "the example without the arm that names the shape"
    )
    (base / "crates" / "valgebra-py" / "src" / "lib.rs").write_text(
        f'"{gate.BINDING_SHAPES[mode]}" => BindingShape::Walk,'
    )
    assert not gate.absent_at(base, mode)
    for other in gate.BINDING_SHAPES:
        if other != mode:
            assert gate.absent_at(base, other), other
    for known in gate.BINDING_SHAPES:
        assert not gate.absent_at(ROOT, known), f"the tree does not name {known}"
    assert not gate.absent_at(ROOT, "core"), "a core workload is its example alone"


def test_a_reading_the_base_example_does_not_name_is_absent_there(
    tmp_path: Path,
) -> None:
    """A core shape that is a reading of a shared example is new where the arm is.

    The rules, the sets and the length bounds are three arguments of one
    example. A base older than one of them builds the example and refuses the
    argument, which is a shape the base does not carry rather than a rig that
    failed -- so the gate reads it from the arm that names the reading.
    """
    mode, reading = next(iter(gate.CORE_READINGS.items()))
    example = gate.MODES[mode][0]
    base = tmp_path / "base"
    examples = base / "crates" / "valgebra-core" / "examples"
    examples.mkdir(parents=True)
    assert gate.absent_at(base, mode), "no example, so no shape"
    (examples / f"{example}.rs").write_text('Some("other") => (),')
    assert gate.absent_at(base, mode), "the example without the reading's arm"
    (examples / f"{example}.rs").write_text(f'Some("{reading}") => (),')
    assert not gate.absent_at(base, mode)
    for known in gate.CORE_READINGS:
        assert not gate.absent_at(ROOT, known), f"the tree does not name {known}"


def test_every_binding_shape_has_iterations_and_a_budget() -> None:
    """A shape the gate can name is one it can measure and judge.

    Three tables have to agree: the mode, the iteration pair it runs at, and the
    key its count is recorded under. A shape in one and not the others is a mode
    that either cannot run or, new to a change, cannot be judged.
    """
    budget = json.loads(
        (ROOT / "scripts" / "perf_budget.json").read_text(encoding="utf-8")
    )
    for mode in gate.BINDING_SHAPES:
        assert mode in gate.MODES, f"{mode} has no example"
        assert mode in gate.BINDING_ITERATIONS, f"{mode} has no iteration pair"
        high, low = gate.BINDING_ITERATIONS[mode]
        assert high > low > 0, f"{mode} iterations do not difference"
    for mode in gate.MODES:
        key = gate.record_key(mode)
        assert budget.get(key, 0) > 0, f"{mode} has no recorded count under {key}"
        if mode not in gate.BINDING_SHAPES:
            checksum = gate.checksum_key(mode)
            assert budget.get(checksum, 0) > 0, f"{mode} records no {checksum}"


def _bench_steps(prefix: str = "--binding") -> list[str]:
    """Collect the bench job's steps that pass a flag starting with `prefix`."""
    workflow = read_workflow()
    steps = workflow["jobs"]["bench"]["steps"]
    return [step["run"] for step in steps if prefix in step.get("run", "")]


def test_every_binding_shape_is_measured_by_a_lane() -> None:
    """A shape no lane measures is a count nothing reads.

    A shape with a mode, an iteration pair and a recorded count still measures
    nothing until a step passes its flag, and the three tables above cannot see
    that. A mutation survivor accepted on the strength of one of these counts is
    accepted on a count no push takes.
    """
    passed = {
        flag[2:]
        for step in _bench_steps()
        for flag in re.findall(r"--binding[a-z-]*", step)
    }
    missing = sorted(set(gate.BINDING_SHAPES) - passed)
    assert not missing, (
        f"binding shapes with a recorded count that no bench step measures: "
        f"{missing}. Add the flag to the binding gate step of the bench job in "
        f".github/workflows/ci.yml, or drop the shape."
    )


def test_every_decision_shape_is_measured_by_a_lane() -> None:
    """A decision shape with a record and no flag in a step is a count nothing reads."""
    passed = {
        flag[2:]
        for step in _bench_steps(prefix="--decision")
        for flag in re.findall(r"--decision[a-z-]*", step)
    }
    shapes = {mode for mode in gate.MODES if mode.startswith("decision")}
    missing = sorted(shapes - passed)
    assert not missing, (
        f"decision shapes with a recorded count that no bench step measures: "
        f"{missing}. Add the flag to the decision gate step of the bench job."
    )


def test_the_profiled_shapes_are_the_comparison_gates() -> None:
    """The profiled shapes count what the comparison gate times, every one of it.

    A shape the comparison gate adds with no call pair is timed and never
    counted, and a pair for a shape it dropped asks `--count` for a shape it
    refuses. Both registries are the comparison gate's shapes, read from the
    file that carries its ceilings.
    """
    ceilings = json.loads(
        (ROOT / "scripts" / "perf_compare.json").read_text(encoding="utf-8")
    )["ceilings"]
    assert set(gate.PROFILED_CALLS) == set(ceilings), (
        f"profiled call pairs {sorted(gate.PROFILED_CALLS)} against the comparison "
        f"gate's shapes {sorted(ceilings)}: a pair per shape, and none else"
    )
    for shape, (hi, lo) in gate.PROFILED_CALLS.items():
        assert hi > lo > 0, f"{shape}: a difference needs two counts, high first"


def test_the_profiled_shapes_are_compared_and_never_recorded(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A recording of a profiled count is refused before anything is built.

    The wheel such a count reads is rebuilt by nobody: its layout is the
    directory and the toolchain it was built with. So the flag is a comparison
    with a base or nothing, and the refusal is exit 2, the gate's "could not
    run", after no build at all.
    """
    assert gate.main(["--profiled", "--update"]) == gate.EXIT_CANNOT_RUN
    assert "never" in capsys.readouterr().out


@pytest.mark.skipif(sys.platform == "win32", reason="the tree is unpacked by tar")
def test_a_snapshot_is_the_checkout_as_it_stands(tmp_path: Path) -> None:
    """The change's tree holds what is edited and what is new, not what is ignored.

    `--against HEAD` reads an uncommitted change, so the snapshot a profiled
    wheel is built from must hold it: a snapshot of the last commit would build
    the base twice and pass every change. What `.gitignore` names stays out of
    it, as it does of a commit, and taking it moves neither the index nor the
    branch.
    """
    tree = tmp_path / "tree"
    tree.mkdir()
    _git(tree, "init", "--quiet", "--initial-branch=main")
    _git(tree, "config", "user.email", "gate@example.invalid")
    _git(tree, "config", "user.name", "gate")
    (tree / ".gitignore").write_text("built/\n", encoding="utf-8")
    (tree / "kept.txt").write_text("committed\n", encoding="utf-8")
    _git(tree, "add", ".gitignore", "kept.txt")
    _git(tree, "commit", "--quiet", "-m", "base")
    (tree / "kept.txt").write_text("edited\n", encoding="utf-8")
    (tree / "new.txt").write_text("new\n", encoding="utf-8")
    (tree / "built").mkdir()
    (tree / "built" / "artifact").write_text("ignored\n", encoding="utf-8")
    status = _git(tree, "status", "--porcelain")

    unpacked = tmp_path / "unpacked"
    unpacked.mkdir()
    gate.unpack(gate.snapshot(tree), unpacked, tree)

    assert (unpacked / "kept.txt").read_text(encoding="utf-8") == "edited\n"
    assert (unpacked / "new.txt").read_text(encoding="utf-8") == "new\n"
    assert not (unpacked / "built").exists()
    assert _git(tree, "status", "--porcelain") == status
    head = _git(tree, "rev-parse", "HEAD:kept.txt")
    assert _git(tree, "cat-file", "-p", head) == "committed"


def test_a_wheel_takes_the_place_of_the_one_before(tmp_path: Path) -> None:
    """Installing the base's wheel leaves nothing of the change's behind.

    One environment holds each side's wheel in turn, so a module the change
    adds and the base lacks would otherwise be read by the base's count.
    """
    python = gate.bare_environment(tmp_path / "env")

    def wheel(name: str, modules: dict[str, str]) -> Path:
        path = tmp_path / f"{name}.whl"
        with zipfile.ZipFile(path, "w") as archive:
            for module, body in modules.items():
                archive.writestr(f"valgebra/{module}", body)
            archive.writestr(f"valgebra-{name}.dist-info/METADATA", f"Version: {name}")
        return path

    gate.install(wheel("1", {"__init__.py": "SIDE = 1\n", "added.py": ""}), python)
    gate.install(wheel("2", {"__init__.py": "SIDE = 2\n"}), python)
    read = subprocess.run(  # noqa: S603 - fixed argv, no shell, test-only
        [
            str(python),
            "-c",
            (
                "import importlib.util, valgebra; "
                "print(valgebra.SIDE, importlib.util.find_spec('valgebra.added'))"
            ),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert read.stdout.split() == ["2", "None"]


def test_the_profiled_shapes_are_measured_by_a_lane() -> None:
    """A gate no step runs reads nothing; the bench job passes the flag."""
    steps = _bench_steps(prefix="--profiled")
    assert steps, (
        "no bench step passes --profiled: the profile's training is read by no "
        "instruction count. Add the step to the bench job in "
        ".github/workflows/ci.yml."
    )
    assert all("--against" in step for step in steps)


#: The pages that give the gap between the two deciders as an order of
#: magnitude, each of which a re-recorded pair of counts can make wrong.
GAP_PAGES = (
    "docs/dev/00-architecture.md",
    "docs/dev/01-schema-ir.md",
    "docs/dev/02-decision.md",
    "docs/11-performance.md",
    "docs/13-foundations.md",
)


def test_the_pages_give_the_gap_between_the_deciders_the_record_holds() -> None:
    """The order of magnitude the pages give is the record's.

    `--decision-rules` and `--decision-sets` ask one corpus of the rules and of
    the sets, at the two iteration counts the example names, and the pages
    say how far apart the two are. That is a claim about two recorded counts,
    so it is read from them: a re-recorded count that moves the ratio to
    another order fails here until the pages say so. The two checksums are one
    fold at the two counts, and are held to it, since a pair one decider
    answers and the other does not is a corpus whose ratio says nothing.
    """
    budget = json.loads(
        (ROOT / "scripts" / "perf_budget.json").read_text(encoding="utf-8")
    )
    source = (
        ROOT / "crates" / "valgebra-core" / "examples" / "decision_sets_workload.rs"
    ).read_text(encoding="utf-8")

    def iterations(name: str) -> int:
        found = re.search(rf"const {name}_ITERATIONS: usize = ([\d_]+);", source)
        assert found, f"the workload names no {name}_ITERATIONS"
        return int(found.group(1).replace("_", ""))

    rules, sets = iterations("RULES"), iterations("SETS")
    assert (
        budget[gate.checksum_key("decision-rules")] * sets
        == budget[gate.checksum_key("decision-sets")] * rules
    ), "the two halves answered the corpus differently"
    ratio = (budget[gate.record_key("decision-sets")] / sets) / (
        budget[gate.record_key("decision-rules")] / rules
    )
    words = {2: "two", 3: "three", 4: "four"}
    order = round(math.log10(ratio))
    assert order in words, f"the sets cost {ratio:,.0f} times the rules"
    for page in GAP_PAGES:
        text = " ".join((ROOT / page).read_text(encoding="utf-8").split())
        assert f"{words[order]} orders of magnitude" in text, (
            f"{page} does not give the gap as {words[order]} orders of magnitude, "
            f"which the record reads ({ratio:,.0f} times)"
        )
        for other in set(words.values()) - {words[order]}:
            assert f"{other} orders of magnitude" not in text, (
                f"{page} gives the gap as {other} orders of magnitude, and the "
                f"record reads {words[order]} ({ratio:,.0f} times)"
            )


def _merge_base_step() -> str:
    """Read the shell of the workflow's "Name the merge base" step from it."""
    spec = read_workflow()
    for job in spec["jobs"].values():
        for step in job.get("steps", []):
            if step.get("name") == "Name the merge base":
                return str(step["run"])
    message = "the workflow has no 'Name the merge base' step"
    raise AssertionError(message)


def _run_step(tree: Path, base_sha: str, branch: str) -> str:
    """Run the step and read the `sha=` it recorded.

    A fresh output file per run: a step appends to `GITHUB_OUTPUT`, as the
    runner's does, so a shared one would hold every earlier answer too. The
    script the step calls is laid into the synthetic checkout, where the step
    finds it on a runner.
    """
    lay_script("change_base", tree)
    with tempfile.NamedTemporaryFile("w+", delete=False) as output:
        recorded = Path(output.name)
    try:
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell, test-only
            ["bash", "-euo", "pipefail", "-c", _merge_base_step()],  # noqa: S607
            cwd=tree,
            capture_output=True,
            text=True,
            check=False,
            env={
                **os.environ,
                "BASE_SHA": base_sha,
                "DEFAULT_BRANCH": branch,
                "GITHUB_OUTPUT": str(recorded),
            },
        )
        # Not `check=True`: that raises with the script in the message and the
        # shell's own words nowhere, which is a failure that cannot be read.
        assert result.returncode == 0, (
            f"the step exited {result.returncode}\n{result.stdout}{result.stderr}"
        )
        written = recorded.read_text(encoding="utf-8")
    finally:
        recorded.unlink(missing_ok=True)
    lines = [line for line in written.splitlines() if line.startswith("sha=")]
    assert len(lines) == 1, f"the step recorded {lines}"
    return lines[0].removeprefix("sha=").strip()


def _git(tree: Path, *args: str) -> str:
    return subprocess.run(  # noqa: S603 - fixed argv, no shell, test-only
        ["git", "-C", str(tree), *args],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


@pytest.fixture
def stage(tmp_path: Path) -> tuple[Path, Path]:
    """Build a checkout whose default branch, `HEAD` and `HEAD~1` differ.

    Built rather than cloned from this tree, for two reasons. A clone inherits
    whatever `origin/main` last pointed at, which on the lane that runs the
    suite with full history is `HEAD` itself -- so the fallback to the default
    branch and the guard against measuring `HEAD` against itself land on the
    same commit, and an assertion that separates them passes or fails by
    accident. And a synthetic history needs none of its own, so this runs in the
    shallow clones the rest of the lanes take rather than skipping there.

    `main` is left behind at the first commit, so the branch the step falls back
    to is neither the commit under test nor its parent. Returns the upstream and
    the checkout, because the case this test exists for is the one where the
    two agree.
    """
    upstream, tree = tmp_path / "upstream", tmp_path / "tree"
    upstream.mkdir()
    _git(upstream, "init", "--quiet", "--initial-branch=main")
    _git(upstream, "config", "user.email", "gate@example.invalid")
    _git(upstream, "config", "user.name", "gate")
    for step, branch in enumerate(["main", "work", None]):
        (upstream / "measured.txt").write_text(f"{step}\n", encoding="utf-8")
        if branch == "work":
            _git(upstream, "checkout", "--quiet", "-b", branch)
        _git(upstream, "add", "measured.txt")
        _git(upstream, "commit", "--quiet", "-m", f"commit {step}")
    _git(tmp_path, "clone", "--quiet", "--local", "--no-hardlinks", "upstream", "tree")
    _git(tree, "checkout", "--quiet", "--detach", "origin/work")
    return upstream, tree


#: `bench` is `runs-on: ubuntu-latest`, so this step's shell is that runner's.
#: Driving it through Git-Bash measures the shell rather than the step, and a
#: platform the step never reaches cannot say anything about it either way. Every
#: other lane runs this -- each Linux interpreter and macOS -- so the check is
#: narrowed to the one platform it says nothing on, not to one that it does.
@pytest.mark.skipif(
    sys.platform == "win32",
    reason="the step runs on ubuntu-latest; Git-Bash is a different shell",
)
def test_the_merge_base_is_never_the_commit_being_measured(
    stage: tuple[Path, Path],
) -> None:
    """A relative gate that measures a commit against itself passes anything.

    `github.event.before` is unreachable after a force-push, so the step falls
    back to the default branch -- and on a force-push *to* the default branch
    that is the very commit under test. The gate then compared a build with
    itself, reported 0.00%, and passed whatever the change did.

    Driven through the step's own shell, read out of the workflow, so the test
    cannot agree with a copy that has drifted.
    """
    upstream, tree = stage
    head, parent = _git(tree, "rev-parse", "HEAD"), _git(tree, "rev-parse", "HEAD~1")
    default = _git(tree, "rev-parse", "refs/remotes/origin/main")
    assert len({head, parent, default}) == 3, "the three answers must be tellable apart"

    # A pull request naming a real base: that base.
    assert _run_step(tree, parent, "main") == parent
    # A force-push whose `before` is gone: the default branch stands in, and it
    # is neither of the other two answers -- so this says the fallback ran.
    assert _run_step(tree, "0" * 40, "main") == default
    # The case this exists for -- a force-push *to* the default branch, where
    # the fallback fetches the very commit under test. The parent stands in
    # rather than the commit measuring itself.
    resolved = _run_step(tree, head, "main")
    assert resolved != head, "the gate would measure the commit against itself"
    assert resolved == parent
    # And the same thing reached the way CI reaches it: `before` is gone *and*
    # the default branch has been moved onto the commit under test, so the
    # fallback lands on `HEAD` and the guard is the only thing between the gate
    # and a comparison with itself.
    _git(upstream, "branch", "--force", "main", head)
    assert _run_step(tree, "0" * 40, "main") == parent


def test_a_base_more_than_one_commit_back_is_read_as_a_batch(tmp_path: Path) -> None:
    """The window a relative reading covers, said out loud.

    The ceiling this gate applies is written for one push. What names the base
    is the event, and a force-push or a first push of a long-lived branch makes
    that twenty-odd commits, whose movements sum: a regression in one paid for
    by an improvement in another reads as a pass, and a sum over the ceiling
    names no commit to repair. The gate still measures and still judges -- it
    says which of the two readings it is taking.
    """
    tree = tmp_path / "tree"
    tree.mkdir()
    _git(tree, "init", "--quiet", "--initial-branch=main")
    _git(tree, "config", "user.email", "gate@example.invalid")
    _git(tree, "config", "user.name", "gate")
    _git(tree, "commit", "--quiet", "--allow-empty", "-m", "root")
    base = _git(tree, "rev-parse", "HEAD")
    for step in range(26):
        _git(tree, "commit", "--quiet", "--allow-empty", "-m", f"step {step}")
    parent = _git(tree, "rev-parse", "HEAD~1")

    assert gate.commits_since(base, root=tree) == 26
    assert gate.commits_since(parent, root=tree) == 1
    # A base this checkout cannot reach: a shallow clone's, or an unrelated
    # history's. Unanswerable is not zero.
    assert gate.commits_since("0" * 40, root=tree) is None

    note, batch = gate.describe_window(26)
    assert batch
    assert "26 commits" in note
    note, batch = gate.describe_window(1)
    assert not batch
    assert "1 commit" in note
    # `--against HEAD` reads an uncommitted change, which is no commit at all.
    note, batch = gate.describe_window(0)
    assert not batch
    assert "no commit" in note
    note, batch = gate.describe_window(None)
    assert batch
    assert "unknown" in note
