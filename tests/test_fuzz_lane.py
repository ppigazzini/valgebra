"""The fuzz soak bounds one allocation, and bounds memory per batch.

A libFuzzer run has two memory ceilings and they answer different questions.
`-rss_limit_mb` watches the *process*, which under the sanitizer grows with the
number of executions whatever the target does, because freed memory is
quarantined rather than returned. `-malloc_limit_mb` watches one *allocation*,
which is the thing a bound in this tree actually promises -- the automaton
builder's eight megabytes, the product's edge count -- and the thing worth
reddening a lane.

Leaving both unnamed gives the second the value of the first, so the only check
that ran was the one that cannot name a defect: the soak crossed the unnamed
2,048 MB default partway through its own budget and reported an out-of-memory
against whichever input was in flight, which replayed in 25 ms.

Held in both directions: the step must carry a fork mode, so the process ceiling
bounds a batch rather than the whole run, and it must name the allocation
ceiling rather than inherit one. A step that drops either fails here.

And a ceiling has to be able to fail the lane. In fork mode libFuzzer counts an
out-of-memory or a timeout in a child and carries on, and the step piped the
soak through `tee`, which hid the exit status it ended with: with one 80 MB
allocation planted in a copy of the target, the step exited 0. So the step
turns both off, names the process ceiling that then becomes a verdict, and
fails with the pipe -- and the last is held by running the step's own script
against a stand-in soak, since a flag can be read where a shell option is
only shown by what it does.

LEDGER: the fuzz soak names its ceilings, forks its batches, and fails with them
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

# A repository check: it reads the workflow, which ships in no wheel.
pytestmark = pytest.mark.repository

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"

#: The step that runs the soak, by its `name:` in the workflow.
SOAK = "Fuzz the decision procedures"

#: The largest single allocation the soak may make, in megabytes. Above every
#: bound the code can ask for -- the regex builder's is eight -- and far below
#: the process ceiling, so it fires on a runaway allocation and on nothing else.
ALLOCATION_CEILING_MB = 64

#: The most one batch's process may hold, in megabytes: about three times the
#: 1,317 the first child peaked at in the five nights to 2026-10-05, since an
#: out-of-memory now fails the lane and the sanitizer's own growth must not.
PROCESS_CEILING_MB = 4096


def _step() -> dict[str, Any]:
    """Give the soak's step as the workflow declares it."""
    spec = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    for job in spec["jobs"].values():
        for step in job.get("steps", []):
            if step.get("name") == SOAK:
                return step
    message = f"the workflow has no {SOAK!r} step"
    raise AssertionError(message)


def _soak_step() -> str:
    """Give the soak's commands, with the shell's comments cut.

    A comment naming a flag is not the flag: the step explains `-fork=1` in a
    comment, and a search over the whole block reads it there after the
    command stops passing it.
    """
    return "\n".join(
        line
        for line in str(_step()["run"]).splitlines()
        if not line.lstrip().startswith("#")
    )


def test_the_soak_names_its_allocation_ceiling() -> None:
    """An unnamed `-malloc_limit_mb` is the process ceiling wearing its name."""
    step = _soak_step()
    found = re.search(r"-malloc_limit_mb=(\d+)", step)
    assert found, (
        "the soak does not name -malloc_limit_mb, so the only memory check it "
        "runs is the process one, which the sanitizer's own growth trips"
    )
    assert int(found.group(1)) == ALLOCATION_CEILING_MB, (
        f"the soak bounds one allocation at {found.group(1)} MB; this file "
        f"records {ALLOCATION_CEILING_MB}. Moving it is a decision with an "
        "argument, so move both."
    )


def test_the_soak_names_its_process_ceiling() -> None:
    """Once an out-of-memory fails the lane, the process ceiling is a verdict."""
    found = re.search(r"-rss_limit_mb=(\d+)", _soak_step())
    assert found, (
        "the soak leaves -rss_limit_mb at libFuzzer's 2,048 MB, which a growing "
        "corpus crosses with no defect, and an out-of-memory fails the lane"
    )
    assert int(found.group(1)) == PROCESS_CEILING_MB, (
        f"the soak bounds a batch at {found.group(1)} MB; this file records "
        f"{PROCESS_CEILING_MB}. Moving it is a decision with an argument, so "
        "move both."
    )


def test_the_soak_stops_on_an_out_of_memory_or_a_hang() -> None:
    """Fork mode counts both in a child and carries on unless told not to."""
    step = _soak_step()
    for flag in ("-ignore_ooms=0", "-ignore_timeouts=0"):
        assert flag in step, (
            f"the soak does not pass {flag}, so under -fork=1 a child's "
            "out-of-memory or timeout is counted and the soak goes on, and the "
            "ceiling it names fails nothing"
        )


#: What a soak prints last under `-fork=1`: the parent's job line, with the
#: rate the step's floor reads, and its closing note, with the seconds.
_CLOSING = (
    "#2845: cov: 7634 ft: 33350 corp: 385 exec/s: 66 oom/timeout/crash: {oom}/0/0 "
    "time: 372s job: 8 dft_time: 0\n"
    "INFO: fuzzed for 372 seconds, wrapping up soon\n"
)


# The step runs on an Ubuntu runner. On Windows, `bash` on the path is WSL's
# launcher, which with no distribution installed prints a usage message in
# UTF-16 and exits 1, and a shebang stand-in is no executable there anyway.
@pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None,
    reason="the step is a bash script an Ubuntu runner runs",
)
@pytest.mark.parametrize(
    ("printed", "status"),
    [
        (_CLOSING.format(oom=0), 0),
        (
            "==1== ERROR: libFuzzer: out-of-memory (malloc(83886080))\n"
            + _CLOSING.format(oom=1),
            71,
        ),
    ],
    ids=["a clean soak", "an out-of-memory"],
)
def test_the_step_ends_as_the_soak_does(
    tmp_path: Path, printed: str, status: int
) -> None:
    """The step's own script, run against a stand-in for `cargo fuzz run`.

    The stand-in prints what a soak prints and exits as libFuzzer does -- 71 for
    an out-of-memory, the planted allocation's ending -- and the script runs as
    GitHub runs a step with no shell named, `bash -e`. A clean soak passes the
    floor and ends 0; an out-of-memory ends the step with the soak's status
    rather than the `tee` behind it.
    """
    stand_in = tmp_path / "bin" / "cargo"
    stand_in.parent.mkdir()
    stand_in.write_text(
        f"#!/bin/sh\ncat <<'SOAK'\n{printed}SOAK\nexit {status}\n", encoding="utf-8"
    )
    stand_in.chmod(0o755)
    step = _step()
    environment = {
        **os.environ,
        **{name: str(value) for name, value in step["env"].items()},
        "PATH": f"{stand_in.parent}{os.pathsep}{os.environ['PATH']}",
    }
    ended = subprocess.run(  # noqa: S603 -- the workflow's own script, test-only
        ["bash", "-e", "-c", step["run"]],  # noqa: S607 -- the shell GitHub runs
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert ended.returncode == status, (
        f"the soak ended {status} and the step ended {ended.returncode}: "
        f"{ended.stderr.strip()}"
    )


def test_the_soak_bounds_memory_per_batch() -> None:
    """Without a fork the process ceiling bounds the whole run, which grows."""
    step = _soak_step()
    assert re.search(r"-fork=\d+", step), (
        "the soak does not fork, so its memory ceiling applies to a process "
        "whose resident size grows with every execution it has done"
    )


def test_the_soak_still_has_a_time_budget() -> None:
    """The flags above are additions, not a replacement for the budget."""
    assert re.search(r"-max_total_time=\d+", _soak_step())


def test_the_soak_has_a_floor_beneath_its_budget() -> None:
    """A budget says when to stop; it does not say the run did anything.

    `docs/dev/07-tooling-ci.md` already states the limit -- a run that finds
    nothing means "nothing failed inside that budget", never "there is nothing
    to find". What the budget alone cannot tell apart is a soak that explored
    for six minutes from one whose target died on its first input: both end
    quietly, both report no finding, and the lane is green either way.

    So the budget carries a floor, read from the two numbers the soak itself
    prints: the rate it managed and the seconds it lasted. Either alone has a
    silent shape -- a run with no summary executed nothing, and one at zero
    executions a second executed nothing however long it sat there -- and a run
    beneath the floor is a **rig fault**, the reading a mutation timeout gets,
    rather than a clean sheet.
    """
    step = _soak_step()
    # The comparisons, not the words: the step prints `MIN_SECONDS` and
    # `exec/s` in lines that decide nothing, and those outlive a deleted floor.
    assert re.search(r'"\$seconds" -lt "\$MIN_SECONDS"', step), (
        "the soak names a time budget and no floor, so a target that dies on "
        "its first input reads exactly like one that explored for the whole "
        "budget. Read what the soak printed and refuse a run beneath a floor."
    )
    assert re.search(r'"\$rate" -le 0', step), (
        "the floor reads no rate, so a stalled run passes it"
    )
    assert "RIG FAULT" in step, (
        "a run beneath the floor measured nothing, which is the glossary's rig "
        "fault rather than a failure of the tests"
    )


def test_the_generator_draws_every_node_the_ir_has() -> None:
    """A fuzzer explores the shapes it can build and no others.

    The generator is the target's universe, so a variant it never emits is one
    no soak has ever reached however long it runs -- and the laws it checks are
    laws about the fragment it draws rather than about the IR. Held to the
    `Schema` enum, which is where a node is added.

    Three variants are emitted through a name other than their own -- the top
    through its constant, and the two container kinds through the constructors
    that build them -- so each is listed with what reaches it. The top carries a
    *spelling* beside the set, and one arm covers both because a fuzz target
    reads verdicts rather than spellings. Anything else absent is a gap.
    """
    ir = (ROOT / "crates" / "valgebra-core" / "src" / "ir.rs").read_text(
        encoding="utf-8"
    )
    body = ir[ir.index("pub enum Schema") :]
    body = body[: body.index("\n}")]
    variants = set(re.findall(r"^    ([A-Z]\w+)", body, re.MULTILINE))
    assert {"Anything", "Nothing", "Union", "Complement"} <= variants, sorted(variants)

    generator = (ROOT / "fuzz" / "src" / "lib.rs").read_text(encoding="utf-8")
    # The constructors count as well as the bare variants: a sequence is built
    # through `Schema::set` and `Schema::frozen_set`, which name no variant.
    emitted = set(re.findall(r"Schema::(\w+)", generator))
    #: Variants the generator reaches through a constructor rather than by name,
    #: each with the constructor that builds it.
    through = {
        "Anything": ("ANYTHING",),
        "Coll": ("set", "frozen_set"),
        "Seq": ("Seq",),
    }
    for variant, names in through.items():
        if any(name in emitted for name in names):
            emitted.add(variant)

    missing = sorted(variants - emitted)
    assert not missing, (
        f"the fuzz generator emits no {missing}. A variant it cannot build is "
        "one no soak reaches, so every law the target checks is a law about "
        "the fragment it draws."
    )


def test_the_generator_draws_a_reference_wider_than_its_table() -> None:
    """Both a reference that resolves and one that does not are reached.

    A `Ref` is an index into the definitions table the target carries, and the
    two cases are decided differently: one unfolds, and one becomes the
    polarity's cut, which is what a decider meets after a pruned build. Drawing
    the index from the table's own width would reach only the first.
    """
    generator = (ROOT / "fuzz" / "src" / "lib.rs").read_text(encoding="utf-8")
    assert "Schema::Ref(" in generator
    table = re.search(r"let n = count\(u, (\d+)\)\?;\s*\n\s*let mut defs", generator)
    assert table is not None, "the target builds no definitions table"
    index = re.search(r"Schema::Ref\(DefIx::new\(usize::from\(.*?% (\d+)\)", generator)
    assert index is not None, "the reference index is not drawn from a bound"
    assert int(index.group(1)) > int(table.group(1)), (
        "the reference index is drawn no wider than the table is built, so "
        "every reference resolves and the cut is never reached"
    )
