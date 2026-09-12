"""The aggregate gate requires every job the workflow defines.

One job gates the merge by asserting each other job succeeded, and it names
them twice: once in `needs`, so their results are available, and once in the
condition that reads those results. A job missing from either list runs, goes
red, and blocks nothing -- which is the failure a gate over other gates exists
to rule out, and the reason adding a job to this workflow is riskier than it
looks.

Held in three directions: every job the workflow defines is needed, every need
is read by the condition, and nothing is named that the workflow does not
define. The nightly jobs are exempt by name, since a scheduled job does not run
on the pushes this gate is about.

LEDGER: the merge gate requires every job the workflow defines
"""

from __future__ import annotations

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

#: How a job says it runs on a schedule and not on a push. Read from the job's
#: own `if` rather than listed here, so a nightly job added to the workflow is
#: one this ledger already knows about.
SCHEDULED = re.compile(r"github\.event_name == 'schedule'")

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


def _scheduled_only(jobs: dict) -> set[str]:
    """Read which jobs a push does not run, from their own conditions."""
    return {
        name
        for name, job in jobs.items()
        if SCHEDULED.search(str(job.get("if", ""))) is not None
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


def test_a_scheduled_job_may_be_skipped_and_no_other_may() -> None:
    """The gate reads every job's result, and only a nightly may be `skipped`.

    A push skips the scheduled jobs, so the gate has to accept that answer from
    them -- and from nothing else, since `skipped` from a job a push does run is
    a job that did not run. What both readings refuse is everything else, which
    is where a **cancellation** lives: a job that reaches its timeout is
    reported cancelled rather than failed, and one that nothing waits on takes a
    whole scheduled run red without a red job to point at.
    """
    jobs = _workflow()["jobs"]
    gate = jobs[GATE]
    condition = " ".join(str(gate["steps"][0]["if"]).split())
    scheduled = _scheduled_only(jobs)
    assert scheduled, "no job reads as scheduled-only; the pattern has gone stale"
    for name in sorted(set(gate["needs"])):
        # Either spelling the expression language offers, since the condition
        # uses both and which one a name takes is not this ledger's business.
        spellings = (f"needs['{name}'].result", f"needs.{name}.result")
        reads = [reading for reading in spellings if reading in condition]
        assert reads, f"the gate does not read {name}'s result"
        allows_skipped = any(
            f"{reading} != 'skipped'" in condition for reading in reads
        )
        if name in scheduled:
            assert allows_skipped, (
                f"{name} runs only on a schedule and the gate demands success "
                "from it, which fails every push"
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


def test_the_merge_counts_the_shards_the_sweep_is_cut_into() -> None:
    """The merged sweep's shard count is the sweep's own matrix.

    The ratchet the merge feeds runs the expiry direction, which reads a
    mutant's absence as proof the tests killed it. A shard that dies before it
    uploads makes every mutant it held absent, so the merge counts before the
    ratchet reads -- and it can only count against a number. Two numbers that
    drift apart make the guard pass over a sweep with a shard missing, which is
    the state it exists to refuse.
    """
    jobs = _workflow()["jobs"]
    cut = len(jobs["nightly-mutants"]["strategy"]["matrix"]["shard"])
    merge = next(
        step
        for step in jobs["nightly-mutants-ratchet"]["steps"]
        if step.get("name", "").startswith("Merge the shards")
    )
    counted = int(merge["env"]["SHARDS"])
    assert counted == cut, (
        f"the merge counts {counted} shards and the sweep is cut into {cut}. "
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
