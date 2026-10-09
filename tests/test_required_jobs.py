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

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from _toml import load

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


#: The divisor a sharded sweep passes, as its step spells it.
DIVISOR = re.compile(r'--shard "\$\{\{ matrix\.shard \}\}/([0-9]+)"')


def test_a_sharded_sweep_covers_every_shard() -> None:
    """A shard index is zero-based, and an out-of-range one sweeps nothing.

    `cargo mutants --shard k/n` numbers the shards `0..n-1`, so `--shard n/n`
    selects no mutant at all -- and a sweep of no mutants is a job that passes
    having tested nothing. The matrix and the divisor are written in two places,
    so they are held to each other here: the shards a job runs are exactly the
    range the divisor names. Cutting a sweep wider edits both, and the one left
    behind either sweeps nothing past the old count or leaves the mutants past
    the old matrix to no shard at all.
    """
    sharded = False
    for name, job in sorted(_workflow()["jobs"].items()):
        matrix = job.get("strategy", {}).get("matrix", {})
        if "shard" not in matrix:
            continue
        sharded = True
        runs = "\n".join(str(step.get("run", "")) for step in job["steps"])
        divisors = {int(found) for found in DIVISOR.findall(runs)}
        assert len(divisors) == 1, (
            f"{name} shards its matrix and passes the divisors {sorted(divisors)}"
        )
        (count,) = divisors
        assert sorted(matrix["shard"]) == list(range(count)), (
            f"{name} runs shards {matrix['shard']}; a divisor of {count} covers "
            f"{list(range(count))}, and anything else leaves mutants unswept"
        )
    assert sharded, "no job shards its sweep; this ledger has no subject"


#: The workflow variable listing the binding files the two binding sweeps
#: examine, one per line.
SWEPT_LIST = "BINDING_SWEPT"

#: A binding path handed to `--file` as a literal: a list beside the one.
_LITERAL_FILE = re.compile(r"--file\s+crates/valgebra-py/")

#: The line of the push lane's diff step that picks the files its sweep takes.
_SELECTION = re.compile(r"^\s*(walk=\$\(.*\))\s*$", re.MULTILINE)

#: How a line of a sweep step that builds its `--file` arguments starts.
_BUILDS_FILES = ("files=", "for f in ")

#: What an invocation of the binding sweep hands `cargo mutants` first.
_PASSES = re.compile(r"cargo mutants --package valgebra-py (\S+)")


def _listing() -> str:
    """Read the workflow's list of the binding files the two sweeps examine."""
    listing = str((_workflow().get("env") or {}).get(SWEPT_LIST, ""))
    assert listing.split(), f"the workflow lists no binding files under {SWEPT_LIST}"
    return listing


def _sweep_step(job: dict) -> dict:
    """Give the one step of a job that runs the binding sweep."""
    steps = [
        step for step in job["steps"] if "cargo mutants" in str(step.get("run", ""))
    ]
    assert len(steps) == 1, f"expected one sweep step, found {len(steps)}"
    return steps[0]


def _bash(script: str, cwd: Path, **variables: str) -> str:
    """Run a fragment of a step as the runner runs a step, and give what it prints.

    Under the runner's shell and its flags, with the workflow's variables, which
    the runner hands every step, beside the step's own.
    """
    bash = shutil.which("bash")
    assert bash is not None, "bash runs every step of the workflow"
    workflow = {
        key: str(value) for key, value in (_workflow().get("env") or {}).items()
    }
    done = subprocess.run(  # noqa: S603  # a fragment of the tree's own workflow
        [bash, "--noprofile", "--norc", "-eo", "pipefail", "-c", script],
        cwd=cwd,
        env={"PATH": os.environ.get("PATH", ""), **workflow, **variables},
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return done.stdout


def _swept_by(step: dict, cwd: Path, **variables: str) -> list[str]:
    """Give the files a sweep step hands `cargo mutants`, by running what builds them.

    The lines that build the arguments run as written, and every invocation in
    the step is held to passing what they built, the listing a shard is named
    against as well as the sweep.
    """
    run = str(step["run"])
    built = "\n".join(
        line for line in run.splitlines() if line.strip().startswith(_BUILDS_FILES)
    )
    assert built, "the sweep step builds no file arguments"
    passed = set(_PASSES.findall(run))
    assert passed == {'"${files[@]}"'}, (
        f"an invocation passes {sorted(passed)} rather than the files it built"
    )
    printed = _bash(built + '\nprintf "%s\\n" "${files[@]}"', cwd, **variables).split()
    assert set(printed[0::2]) <= {"--file"}, printed
    return printed[1::2]


def test_both_binding_sweeps_read_one_list(tmp_path: Path) -> None:
    """The nightly binding sweep covers what the push lane's ratchet judges.

    The push lane ratchets its sweep against a baseline, and that baseline is
    recorded by the nightly. A file the push lane sweeps and the nightly does
    not has every survivor in it read as *new* the first time a change touches
    the area -- which is a red lane about code nobody edited, arriving on
    whichever commit happened to reach the sweep.

    So the list is written once, as a workflow variable, and neither job names
    a binding file beside it. The nightly is held to sweeping every file of it
    by running the lines that build its arguments.
    """
    jobs = _workflow()["jobs"]
    listing = _listing()
    for name in ("mutants-diff-walk", "nightly-mutants-walk"):
        runs = "\n".join(str(step.get("run", "")) for step in jobs[name]["steps"])
        assert not _LITERAL_FILE.search(runs), (
            f"{name} names a binding file of its own beside {SWEPT_LIST}. The "
            "nightly records the baseline the push lane is judged against, so "
            "the two read one list."
        )
    swept = _swept_by(_sweep_step(jobs["nightly-mutants-walk"]), tmp_path)
    assert swept == listing.split(), (
        f"the nightly sweeps {swept}, and the list is {listing.split()}"
    )


def test_the_push_sweep_takes_the_listed_files_the_change_touches(
    tmp_path: Path,
) -> None:
    """A push sweeps the listed files its change touches, and no others.

    The selection is the core lane's with the list in place of the crate: the
    changed files the list names. It runs here, and the sweep step's argument
    lines after it, on three changes drawn to tell the readings apart. Every
    listed file beside three unlisted ones must sweep the list and nothing
    else: a listed file the selection misses lands with its sweep skipped, and
    an unlisted one has every survivor in it read as new against a baseline
    that never swept it. One listed file must sweep that file alone, since the
    whole list for one file is the wait scoping removes. No listed file must
    skip the sweep. The unlisted paths are a binding file the list does not
    name, a core file, and a listed path with a suffix, which a match on a
    prefix would take.
    """
    job = _workflow()["jobs"]["mutants-diff-walk"]
    listing = _listing()
    listed = listing.split()
    diff = [step for step in job["steps"] if step.get("id") == "diff"]
    assert len(diff) == 1, "expected one step with the id `diff`"
    selections = _SELECTION.findall(str(diff[0]["run"]))
    assert len(selections) == 1, f"expected one selection line, found {selections}"
    sweep = _sweep_step(job)
    assert sweep.get("if") == "steps.diff.outputs.walk != ''", sweep.get("if")
    assert (sweep.get("env") or {}).get("FILES") == "${{ steps.diff.outputs.walk }}", (
        f"the sweep reads {sweep.get('env')} rather than the files the diff selected"
    )
    unlisted = [
        "crates/valgebra-py/src/planted.rs",
        "crates/valgebra-core/src/ir.rs",
        f"{listed[0]}.orig",
    ]
    assert not set(unlisted) & set(listed), "an unlisted path the list names"

    def swept(changed: list[str]) -> list[str]:
        (tmp_path / "changed.txt").write_text(
            "".join(f"{path}\n" for path in changed), encoding="utf-8"
        )
        selected = _bash(selections[0] + '\nprintf "%s" "$walk"', tmp_path)
        return _swept_by(sweep, tmp_path, FILES=selected) if selected else []

    assert swept([*unlisted, *listed]) == listed
    assert swept([unlisted[0], listed[0], unlisted[2]]) == [listed[0]]
    assert swept(unlisted) == []


#: The variable proptest reads its seed from.
SEED = "PROPTEST_RNG_SEED"

#: The seed a nightly draws: a new one each run.
RUN_ID = "${{ github.run_id }}"

#: The page whose local sweep recipes reproduce the push lanes.
TOOLING = ROOT / "docs" / "dev" / "07-tooling-ci.md"


def _seeds(jobs: dict) -> dict[str, str]:
    """Read the seed each job that sets one draws, by its job name."""
    seeds = {}
    for name, job in jobs.items():
        for step in job["steps"]:
            assert SEED not in (step.get("env") or {}), (
                f"a step of {name} sets its own {SEED}; the job's one seed is "
                "what makes the baseline and every mutant draw alike"
            )
        if SEED in (job.get("env") or {}):
            seeds[name] = str(job["env"][SEED])
    return seeds


def test_a_push_sweep_draws_one_literal_and_a_nightly_its_run_id() -> None:
    """A push sweep's verdict is the tree's, and a nightly's explores.

    A property law kills some mutants on some draws only, so a push sweep that
    draws from its run id judges an untouched file differently from one push
    to the next, and blocks a merge on a draw. Under one literal, a mutant the
    literal's draw does not kill survives every time, and a deterministic test
    kills it. The nightly keeps the run id, since finding those mutants is its
    work and its new survivor is a report rather than a block.
    """
    jobs = _workflow()["jobs"]
    seeds = _seeds(jobs)
    off_the_push = _not_on_a_push(jobs)
    pushed = {name: seed for name, seed in seeds.items() if name not in off_the_push}
    nightly = {name: seed for name, seed in seeds.items() if name in off_the_push}
    assert {"mutants-diff-core", "mutants-diff-walk"} <= set(pushed), sorted(seeds)
    assert {"nightly-mutants", "nightly-mutants-walk"} <= set(nightly), sorted(seeds)
    literals = set(pushed.values())
    assert len(literals) == 1, f"the push sweeps draw different seeds: {pushed}"
    (literal,) = literals
    assert literal.isdigit(), (
        f"a push sweep draws {literal!r}; a push's seed is a literal, so the "
        "same tree draws the same cases on every push"
    )
    explores = {name: seed for name, seed in nightly.items() if seed != RUN_ID}
    assert not explores, (
        f"nightly sweeps that do not draw from the run id: {explores}. A nightly "
        "that fixes its seed never finds a mutant only some draws kill."
    )


def test_the_local_sweep_draws_the_push_lanes_seed() -> None:
    """A local sweep of the files a change touches draws what the lane draws.

    The tooling page's recipes for the two push sweeps are how a change is
    checked before it is pushed, and a recipe drawing its own seed reproduces
    a different sweep: a mutant only some draws kill passes here and survives
    there. The recipes are the page's fenced blocks that judge a partial sweep
    with `--new-only`, which is what makes one a push sweep.
    """
    jobs = _workflow()["jobs"]
    pushed = {
        seed for name, seed in _seeds(jobs).items() if name not in _not_on_a_push(jobs)
    }
    page = TOOLING.read_text(encoding="utf-8")
    blocks = re.findall(r"```bash\n(.*?)```", page, re.DOTALL)
    recipes = [
        block for block in blocks if "cargo mutants" in block and "--new-only" in block
    ]
    assert len(recipes) >= 2, f"the page carries {len(recipes)} push sweep recipes"
    for recipe in recipes:
        drawn = re.findall(rf"^export {SEED}=(\S+)$", recipe, re.MULTILINE)
        assert drawn, f"a push sweep recipe draws no seed of its own:\n{recipe}"
        assert set(drawn) <= pushed, (
            f"a push sweep recipe draws {drawn} and the push lanes draw "
            f"{sorted(pushed)}:\n{recipe}"
        )


#: The job the scheduled run dispatches the nightly from.
DISPATCH = "nightly-dispatch"

#: The output it names `main`'s own nightly by, as a lane's condition reads it.
DIFFERS = f"needs['{DISPATCH}'].outputs.main_differs == 'true'"

#: The branch a schedule fires on.
DEFAULT_BRANCH = "main"


def test_the_nightly_runs_on_the_branch_development_is_on() -> None:
    """The scheduled run dispatches the nightly on the development branch.

    A schedule fires on the default branch alone, and development is on the
    other branch the push trigger names, which `main` follows at a release cut:
    a nightly that read `main` alone would read a release window's commits
    after they shipped. So the scheduled job dispatches the workflow on that
    branch, which runs the branch's own copy of the workflow, and every other
    scheduled lane runs on `main` only where `main`'s tip is not the branch's.
    A lane reads that from the dispatcher, or, a ratchet, from the sweep it
    ratchets, which reads it in turn.
    """
    workflow = _workflow()
    jobs = workflow["jobs"]
    dispatcher = jobs[DISPATCH]
    assert " ".join(str(dispatcher["if"]).split()) == (
        "github.event_name == 'schedule'"
    ), "the dispatcher runs on a schedule and on nothing else"
    assert dispatcher["permissions"].get("actions") == "write", (
        "dispatching a run needs `actions: write`, at the dispatcher's level"
    )
    dispatching = [
        step
        for step in dispatcher["steps"]
        if "gh workflow run ci.yml" in str(step.get("run", ""))
    ]
    assert len(dispatching) == 1, "the dispatcher dispatches the workflow once"
    (step,) = dispatching
    assert '--ref "$BRANCH"' in step["run"], step["run"]
    # PyYAML reads the bare key `on` as the boolean it spells in YAML 1.1.
    triggers = workflow.get("on", workflow.get(True))
    pushed = triggers["push"]["branches"]
    branch = step["env"]["BRANCH"]
    assert branch in pushed, f"{branch} is not a branch a push runs on: {pushed}"
    assert branch != DEFAULT_BRANCH, "the dispatch names the branch the schedule runs"

    scheduled = {
        name
        for name, job in jobs.items()
        if "github.event_name == 'schedule'" in str(job.get("if", ""))
    } - {DISPATCH}
    assert len(scheduled) >= 8, f"only {sorted(scheduled)} read as scheduled"
    unread = []
    for name in sorted(scheduled):
        job = jobs[name]
        condition = " ".join(str(job["if"]).split())
        needs = job.get("needs", [])
        needed = [needs] if isinstance(needs, str) else list(needs)
        if DISPATCH in needed:
            if DIFFERS not in condition:
                unread.append(f"{name} needs the dispatcher and reads no output of it")
            continue
        sweeps = [need for need in needed if DISPATCH in str(jobs[need].get("needs"))]
        if not sweeps or any(
            f"needs['{sweep}'].result != 'skipped'" not in condition for sweep in sweeps
        ):
            unread.append(f"{name} reads neither the dispatcher nor a sweep that does")
    assert not unread, (
        "scheduled jobs that run on `main` whatever its tip: "
        + "; ".join(unread)
        + ". Where the tips agree, the dispatched run reads the same tree."
    )


def test_every_supported_interpreter_runs_on_every_event() -> None:
    """The job that carries the supported interpreters runs on a push.

    The release ships a wheel built per version against a version-specific ABI,
    so each interpreter is a separate artifact a caller installs. A lane that
    runs only at night is a wheel nothing exercised until somebody reported it,
    and a lane that runs on no event at all is one `requires-python` promises
    and nothing checks.

    Which releases the `python` job carries is `tests/test_python_lifecycle.py`'s
    to hold, against the schedule each release follows; a list written here
    would be a second place to move when the floor moves. This holds the two
    things that ledger does not: the job runs on every event, and the
    free-threaded build the classifiers promise has a leg in it.
    """
    job = _workflow()["jobs"]["python"]
    condition = str(job.get("if", ""))
    assert not NOT_ON_A_PUSH.search(condition), (
        f"the python job runs only when {condition!r}, so a push reads no interpreter"
    )
    matrix = job["strategy"]["matrix"]
    versions = [str(version) for version in matrix["python-version"]]
    versions += [str(row["python-version"]) for row in matrix.get("include", [])]
    classifiers = load(ROOT / "pyproject.toml")["project"]["classifiers"]
    if any(
        row.startswith("Programming Language :: Python :: Free Threading")
        for row in classifiers
    ):
        assert any(version.endswith("t") for version in versions), (
            "the classifiers promise free threading and no leg of the python job "
            f"is a free-threaded build: {versions}"
        )
