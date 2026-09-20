"""The aggregate gate requires every job the workflow defines.

One job gates the merge by asserting each other job succeeded, and it names
them twice: once in `needs`, so their results are available, and once in the
condition that reads those results. A job missing from either list runs, goes
red, and blocks nothing -- which is the failure a gate over other gates exists
to rule out, and the reason adding a job to this workflow is riskier than it
looks.

Held in three directions: every job the workflow defines is needed, every need
is read by the condition, and nothing is named that the workflow does not
define. A job a push does not run -- a nightly, or one a dispatch input asks
for -- is read off its own condition, since the gate is about the pushes it
blocks.

LEDGER: the merge gate requires every job the workflow defines
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

# A repository check: it reads the workflow, which ships in no wheel.
pytestmark = pytest.mark.repository

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"

#: The job that gates the merge on the others.
GATE = "ci"

#: How a job says a push does not run it: it runs on a schedule, or it runs
#: because a dispatch asked for it. Read from the job's own `if` rather than
#: listed here, so a nightly or a dispatch-only job added to the workflow is one
#: this ledger already knows about.
NOT_ON_A_PUSH = re.compile(r"github\.event_name == 'schedule'|inputs\.[a-z_]+")

#: `needs.<job>.result` as the condition spells it, in both forms the expression
#: language offers: a name with a hyphen in it cannot be read with a dot, since
#: the hyphen parses as subtraction, so those are read by index.
READS = re.compile(r"needs(?:\.([a-z0-9-]+)|\['([a-z0-9-]+)'\])\.result")


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _gate_condition(gate: dict) -> str:
    conditions = [str(step.get("if", "")) for step in gate["steps"]]
    return " ".join(conditions)


def _jobs_the_condition_reads(gate: dict) -> set[str]:
    """Every job name the gate's condition reads a result for."""
    found = READS.findall(_gate_condition(gate))
    return {name for pair in found for name in pair if name}


def _not_on_a_push(jobs: dict) -> set[str]:
    """Read which jobs a push does not run, from their own conditions."""
    return {
        name
        for name, job in jobs.items()
        if NOT_ON_A_PUSH.search(str(job.get("if", ""))) is not None
    }


def test_every_job_is_required_by_the_gate() -> None:
    jobs = _workflow()["jobs"]
    gate = jobs[GATE]
    needed = set(gate["needs"])
    defined = set(jobs) - {GATE}
    missing = sorted(defined - needed)
    assert not missing, (
        f"jobs the merge gate does not wait on: {missing}. A job outside its "
        "`needs` can go red without blocking anything."
    )


def test_a_job_a_push_does_not_run_may_be_skipped_and_no_other_may() -> None:
    """The gate reads every job's result, and only those may be `skipped`.

    A push skips the scheduled jobs and the ones a dispatch input asks for, so
    the gate has to accept that answer from them -- and from nothing else, since
    `skipped` from a job a push does run is a job that did not run. What both
    readings refuse is everything else, which is where a **cancellation** lives:
    a job that reaches its timeout is reported cancelled rather than failed, and
    one that nothing waits on takes a whole scheduled run red without a red job
    to point at.
    """
    jobs = _workflow()["jobs"]
    gate = jobs[GATE]
    condition = " ".join(str(gate["steps"][0]["if"]).split())
    off_the_push = _not_on_a_push(jobs)
    assert off_the_push, "no job reads as off the push; the pattern has gone stale"
    for name in sorted(set(gate["needs"])):
        # Either spelling the expression language offers, since the condition
        # uses both and which one a name takes is not this ledger's business.
        spellings = (f"needs['{name}'].result", f"needs.{name}.result")
        reads = [reading for reading in spellings if reading in condition]
        assert reads, f"the gate does not read {name}'s result"
        allows_skipped = any(
            f"{reading} != 'skipped'" in condition for reading in reads
        )
        if name in off_the_push:
            assert allows_skipped, (
                f"{name} is not run by a push and the gate demands success from "
                "it, which fails every push"
            )
        else:
            assert not allows_skipped, (
                f"{name} runs on a push and the gate accepts `skipped` from it"
            )
        assert any(f"{reading} != 'success'" in condition for reading in reads), (
            f"the gate does not refuse a non-success from {name}"
        )


def test_every_need_is_read_by_the_condition() -> None:
    jobs = _workflow()["jobs"]
    gate = jobs[GATE]
    read = _jobs_the_condition_reads(gate)
    unread = sorted(set(gate["needs"]) - read)
    assert not unread, (
        f"jobs the gate waits on and does not check: {unread}. A need whose "
        "result the condition never reads is a job that gates nothing."
    )


def test_the_gate_names_no_job_the_workflow_lacks() -> None:
    jobs = _workflow()["jobs"]
    gate = jobs[GATE]
    read = _jobs_the_condition_reads(gate)
    named = set(gate["needs"]) | read
    unknown = sorted(named - set(jobs))
    assert not unknown, (
        f"the gate names jobs this workflow does not define: {unknown}. A need "
        "on a job that does not exist is a gate on nothing."
    )


def _merges(jobs: dict) -> dict[str, tuple[str, int]]:
    """Every job that merges a sharded sweep, with the sweep and its count.

    Derived from the step rather than listed, because a second sharded sweep is
    exactly the kind of thing a hand-written pair list does not grow to cover --
    and the pair that went unlisted would be the one whose guard was never
    checked. A merge is recognised by its step name, and the sweep it merges by
    the job it needs.
    """
    found = {}
    for name, job in jobs.items():
        merge = next(
            (
                step
                for step in job["steps"]
                if str(step.get("name", "")).startswith("Merge the shards")
            ),
            None,
        )
        if merge is None:
            continue
        needs = job["needs"]
        needed = [needs] if isinstance(needs, str) else list(needs)
        assert len(needed) == 1, f"{name} merges the shards of {needed}"
        found[name] = (needed[0], int(merge["env"]["SHARDS"]))
    return found


def test_every_merge_counts_the_shards_its_sweep_is_cut_into() -> None:
    """A merged sweep's shard count is the sweep's own matrix.

    The ratchet the merge feeds runs the expiry direction, which reads a
    mutant's absence as proof the tests killed it. A shard that dies before it
    uploads makes every mutant it held absent, so the merge counts before the
    ratchet reads -- and it can only count against a number. Two numbers that
    drift apart make the guard pass over a sweep with a shard missing, which is
    the state it exists to refuse.
    """
    jobs = _workflow()["jobs"]
    merges = _merges(jobs)
    assert merges, "no job merges a sharded sweep; the step name has moved"
    for merge, (sweep, counted) in sorted(merges.items()):
        cut = len(jobs[sweep]["strategy"]["matrix"]["shard"])
        assert counted == cut, (
            f"{merge} counts {counted} shards and {sweep} is cut into {cut}. "
            "A merge that counts fewer accepts a sweep with a shard missing."
        )


def test_both_binding_sweeps_read_the_same_files() -> None:
    """The nightly binding sweep covers what the push lane's ratchet judges.

    The push lane ratchets its sweep against a baseline, and that baseline is
    recorded by the nightly. A file the push lane sweeps and the nightly does
    not has every survivor in it read as *new* the first time a change touches
    the area -- which is a red lane about code nobody edited, arriving on
    whichever commit happened to reach the sweep.
    """
    jobs = _workflow()["jobs"]
    swept = {
        name: {
            line.strip().removeprefix("--file ").removesuffix("\\").strip()
            for step in job["steps"]
            for line in str(step.get("run", "")).splitlines()
            if line.strip().startswith("--file crates/valgebra-py/")
        }
        for name, job in jobs.items()
        if name in {"mutants-diff-walk", "nightly-mutants-walk"}
    }
    assert len(swept) == 2, f"expected both binding sweeps, found {sorted(swept)}"
    push, nightly = swept["mutants-diff-walk"], swept["nightly-mutants-walk"]
    assert push, "the push sweep names no files"
    assert push == nightly, (
        "the two binding sweeps read different files: "
        f"only the push lane sweeps {sorted(push - nightly)}, "
        f"only the nightly sweeps {sorted(nightly - push)}. The nightly records "
        "the baseline the push lane is judged against, so the two are one list."
    )


#: The line of the push lane's diff step that decides whether the binding
#: sweep runs at all: a regex over the changed paths, and the sweep is skipped
#: when nothing matches.
_WALK_TRIGGER = re.compile(r"walk=\$\(grep -E '([^']+)' changed\.txt")


def test_the_binding_sweep_triggers_on_every_file_it_sweeps() -> None:
    """A file the sweep lists is one the trigger matches.

    The push lane sweeps its files only when the change touches one, and
    "touches one" is a regex over the diff written apart from the `--file` list
    it guards. The two drifted: the list named the record, scalar and sequence
    walks, the index, the input decoders, the dialect and the codes, and the
    regex matched none of them, so a change to the membership procedure itself
    reached `main` with the sweep skipped. The list is what the sweep judges;
    the trigger is held to it here.
    """
    job = _workflow()["jobs"]["mutants-diff-walk"]
    runs = [str(step.get("run", "")) for step in job["steps"]]
    triggers = [found.group(1) for run in runs for found in _WALK_TRIGGER.finditer(run)]
    assert len(triggers) == 1, f"expected one trigger regex, found {triggers}"
    trigger = re.compile(triggers[0])
    swept = {
        line.strip().removeprefix("--file ").removesuffix("\\").strip()
        for run in runs
        for line in run.splitlines()
        if line.strip().startswith("--file crates/valgebra-py/")
    }
    assert swept, "the push sweep names no files"
    unguarded = sorted(path for path in swept if not trigger.fullmatch(path))
    assert not unguarded, (
        f"files the binding sweep lists and its trigger does not match: "
        f"{unguarded}. A change to one of them lands with the sweep skipped."
    )


def test_every_supported_interpreter_runs_on_every_event() -> None:
    """What a push runs across interpreters is the list the package claims.

    The release ships a wheel built per version against a version-specific ABI,
    so each interpreter is a separate artifact a caller installs. A lane that
    runs only at night is a wheel nothing exercised until somebody reported it,
    and a lane that runs on no event at all is one `requires-python` promises
    and nothing checks.

    Held in both directions: every version between the floor and the prerelease
    is here, and a version added to the package's own floor-to-ceiling range has
    to be added here too. The list is read from the matrix rather than from a
    schedule condition, because there is no longer one to read.
    """
    text = WORKFLOW.read_text(encoding="utf-8")
    matrix = re.search(r"python-version: (\[[^\]]*\])", text)
    assert matrix, "the python matrix is not a list"
    versions = json.loads(matrix.group(1))

    # The floor `requires-python` names, and every release up to the prerelease.
    assert versions == [
        "3.10",
        "3.11",
        "3.12",
        "3.13",
        "3.14",
        "3.14t",
        "3.15",
    ], f"the matrix is {versions}"

    # Read with a regex rather than parsed: `tomllib` is 3.11+ and this suite
    # runs from 3.10, which is the floor this assertion is about.
    manifest = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    requires = re.search(r'^requires-python\s*=\s*"([^"]+)"', manifest, re.MULTILINE)
    assert requires, "pyproject.toml declares no `requires-python`"
    claimed = requires.group(1).removeprefix(">=")
    assert versions[0] == claimed, (
        f"the matrix starts at {versions[0]} and the package claims {claimed}; "
        "the floor a caller installs on is the floor a lane runs"
    )
