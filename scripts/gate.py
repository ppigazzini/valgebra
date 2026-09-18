"""Run the merge gate's own commands, in the clone the merge gate gets.

`AGENTS.md` lists a build-health gate: a handful of commands a developer runs in
a full clone with a warm virtual environment. CI is forty-odd jobs in *shallow*
clones with pinned tool versions on three operating systems. A local clone
carries tags and a checkout does not, so a check that reads `git describe`
answers one way in each -- and a difference of that shape stays invisible until
a push finds it.

So this runs the lane's steps rather than a list that resembles them. The
commands come out of ``.github/workflows/ci.yml``; the working tree is a fresh
shallow clone of `HEAD` with no tags, which is what `actions/checkout` produces;
and every step of every merge-gate job is either executed here or named in
`NEEDS_A_RUNNER` with the reason it cannot be. `tests/test_local_gate.py` holds
that list to the workflow in both directions, so a step added to CI is a step
this either runs or refuses by name.

An excused step may still carry a `STANDINS` row: the part of it a developer's
machine can run, with what the substitute gives up written beside it. The
binding's interpreter-backed Rust tests are the case it was written for -- the
merge gate reached them only inside an instrumented coverage rebuild, so
excusing the rebuild left seventy-odd tests in no local step at all. They have a
lane of their own, so the list is empty; a stand-in for a step some lane runs
outright is that command twice, which `tests/test_local_gate.py` refuses.

What it is not is a CI replacement: the steps that need a PGO wheel, valgrind, a
mutation sweep or a second operating system are named and skipped. Those are the
lanes a push is for.

Usage:
    python scripts/gate.py                  # the runnable steps, in a shallow clone
    python scripts/gate.py --list           # what would run, and what would not
    python scripts/gate.py --job docs       # one job's steps
    python scripts/gate.py --here           # this working tree, no clone

Three outcomes, three exit codes: **0** every step passed, **1** a step failed,
**2** the gate could not run -- an unreadable workflow, a clone that failed. A
gate that could not run has proven nothing.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

#: A step to run: its job, its name, its command, and the job's own `env`.
Step = tuple[str, str, str, dict[str, str]]

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_CANNOT_RUN = 2

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"

#: Steps a developer's machine cannot reproduce, and why. Each key is a step's
#: `name:` in the workflow; a step with no name is matched by its command's
#: first line. The reason is the point: a skip nobody can read is a hole.
NEEDS_A_RUNNER = {
    "Install valgrind": "installs a system package with sudo",
    "Build and install the optimized wheel": "a PGO build, minutes per run",
    "Name the merge base": "reads the event payload the runner provides",
    "Instruction-count regression gate": "cachegrind, and a second build of the base",
    "Decision-path instruction-count regression gate": "cachegrind, as above",
    "Binding-path instruction-count regression gate": "cachegrind, as above",
    "Recorded instruction-count budgets (nightly)": "cachegrind, and nightly only",
    "Smoke-run the benchmarks": "needs the optimized wheel above",
    "Competitive comparison gate": "needs the optimized wheel above",
    "Record the competitive ratios": (
        "needs the optimized wheel above, and records a ratio that belongs to "
        "the lane it is measured in"
    ),
    "List the Rust files the diff touches": "reads the event payload",
    "Sweep the core files the change touches": "a mutation sweep, tens of minutes",
    "Sweep the binding's swept files when the change touches one": (
        "a mutation sweep, tens of minutes"
    ),
    "Measure binding coverage via the Python suite and Rust unit tests": (
        "instrumented rebuild of the whole workspace"
    ),
    "Install the differential oracles": (
        "resolves the bench group; the suite below needs it"
    ),
    (
        "cargo llvm-cov nextest --locked --workspace --summary-only "
        "--ignore-filename-regex 'valgebra-py|/(laws|index_laws)\\.rs$|tests\\.rs$' "
        "--fail-under-lines 97 --fail-under-regions 96"
    ): "an instrumented rebuild under nextest",
    "uv sync --locked --no-install-project --group bench": (
        "the bench group, for steps this list already skips"
    ),
    "Cross-check membership against pydantic-core and jsonschema": (
        "needs the bench group installed above"
    ),
    # These three were the candidates for running here rather than being named,
    # since a developer's machine has what they need. Lending the caller's
    # environment to the clone -- `UV_PROJECT_ENVIRONMENT` at the caller's
    # `.venv` -- was tried and reverted: `uv run` in the clone *wrote* that
    # environment, uninstalling the built extension and the bench group from the
    # tree the developer was working in. A gate that damages the environment it
    # is checking is worse than one that skips three steps, so the excuse is now
    # the cost rather than the assumption.
    "Build the extension into the venv": (
        "maturin develop into the clone is a full rebuild; into the caller's "
        "environment it overwrites what the caller built"
    ),
    "Install dev dependencies": (
        "uv sync into the clone resolves the whole tree; into the caller's "
        "environment it rewrites what the caller installed"
    ),
    "Stub matches the extension": (
        "needs the built extension, so it needs one of the two above"
    ),
    "A caller's strict types, on the floor and on the current": (
        "needs both interpreters installed"
    ),
    "Build the fuzz targets": "needs the pinned nightly toolchain and cargo-fuzz",
    "Test the fuzz harness": "needs the pinned nightly toolchain",
}

#: A step only a runner can run, and the part of it a developer can.
#:
#: An excuse is honest and is still a hole: the merge gate runs the binding's
#: interpreter-backed Rust tests only inside an instrumented coverage rebuild,
#: so naming that rebuild as a runner's step leaves seventy-odd tests in no
#: local step at all. The rebuild is the runner's; the tests are not. Each row
#: is a `NEEDS_A_RUNNER` step, the command that reaches the same failures here,
#: and what the substitute gives up -- because a stand-in nobody can tell from
#: the step is the next excuse.
#:
#: Held to `NEEDS_A_RUNNER` in both directions by `tests/test_local_gate.py`: a
#: stand-in for a step that is not excused is a step the gate should just run.
#: Empty today. The row it was written for stood in for the binding's
#: interpreter-backed tests, which the merge gate reached only inside the
#: coverage rebuild; they have a step of their own in the `python` lane, so this
#: gate runs them as itself rather than as a substitute. A row here would be
#: that command a second time.
STANDINS: dict[str, tuple[str, str]] = {}

#: A step that reaches the network, and the offline form this gate runs instead.
#:
#: The gate's three exit codes say what happened, and the middle one -- "could
#: not run" -- exists because a gate that could not run has proven nothing. A
#: step that fetches an advisory database proves nothing about this tree when
#: the fetch fails, and mapping that to the same red as a failing test is the
#: one confusion the exit codes were split to prevent. Two of four gate runs in
#: one afternoon went red on a fetch.
#:
#: Each row is the workflow's step name, the fetch that has to succeed first,
#: and the command to run once it has. The fetch is retried, and a fetch that
#: will not land ends the gate at `EXIT_CANNOT_RUN` rather than failing the
#: step. The runner keeps the online form: catching a newly published advisory
#: is exactly its job, and a runner that cannot reach the database should go
#: red.
#:
#: Held to the plan by `tests/test_local_gate.py`: a row for a step the gate
#: does not run is a row that stands for nothing.
NETWORK = {
    "Audit the workspace dependency tree": (
        "cargo deny fetch all",
        "cargo deny --offline check",
    ),
    "Audit the fuzz workspace dependency tree": (
        "cargo deny --manifest-path fuzz/Cargo.toml fetch all",
        "cargo deny --manifest-path fuzz/Cargo.toml --offline check",
    ),
}

#: How many times a fetch is tried before the gate gives up on it.
FETCH_TRIES = 3

#: How a job says it runs on a schedule and not on a push, read from the job's
#: own condition. The merge gate waits on the nightly jobs so that one reaching
#: its timeout takes the scheduled run red -- a job nothing waits on is
#: cancelled in silence -- and a push skips them. This gate stands in for what a
#: push runs, so it stands in for none of those.
SCHEDULED_ONLY = re.compile(r"github\.event_name == 'schedule'")

#: Jobs whose every step is a runner's, so naming each would say nothing more.
RUNNER_ONLY_JOBS = {
    "wheel",
    "wheel-macos",
    "wheel-windows",
    "sdist",
    "pages",
    "release",
}


def workflow() -> dict:
    try:
        return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as err:
        print(f"gate: cannot read {WORKFLOW}: {err}")
        raise SystemExit(EXIT_CANNOT_RUN) from err


def required_jobs(spec: dict) -> list[str]:
    """Read the jobs the `ci` aggregator waits for: the merge gate itself."""
    jobs = spec.get("jobs", {})
    needs = jobs.get("ci", {}).get("needs", [])
    if not needs:
        print("gate: the ci job lists no needs; there is no merge gate to run")
        raise SystemExit(EXIT_CANNOT_RUN)
    return [
        name
        for name in needs
        if name in jobs
        and name not in RUNNER_ONLY_JOBS
        and not SCHEDULED_ONLY.search(str(jobs[name].get("if", "")))
    ]


def steps(spec: dict, job: str) -> Iterator[tuple[str, str, dict[str, str]]]:
    """Yield the `(name, command, env)` of every `run:` step of `job`, in order.

    The step's own environment comes with it, and dropping it is not a detail.
    A step that denies warnings does so through `RUSTDOCFLAGS`, so a gate that
    ran the command without it ran a command that *cannot fail*: `cargo doc`
    reports a broken intra-doc link as a warning and exits zero. That is a step
    modelled here and green here while red on the runner, which is the one
    outcome this script exists to prevent.

    A value only a runner can answer is dropped, as the job's own environment
    already drops one. Those are the tokens and event fields a step reads when
    it has them -- the workflow audit takes a `GH_TOKEN` for its online checks
    and runs without one -- rather than the flags that decide whether a command
    can fail at all, which are written in the workflow as literals.
    """
    for step in spec["jobs"][job].get("steps", []):
        command = step.get("run")
        if command is None:
            continue  # a `uses:` step: an action, not a command
        env = {
            key: str(value)
            for key, value in (step.get("env") or {}).items()
            if "${{" not in str(value)
        }
        yield step.get("name", command.strip().splitlines()[0]), command, env


def runnable(name: str) -> bool:
    return name not in NEEDS_A_RUNNER


def shallow_clone(into: Path) -> Path:
    """Make a clone of `HEAD` with one commit and no tags, as a checkout does.

    The difference this exists for. A local clone carries every tag and the
    whole history, so a check that reads either passes here and fails there --
    which is exactly what happened, on eight jobs at once.
    """
    tree = into / "tree"
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--no-tags",
            "--quiet",
            f"file://{ROOT}",
            str(tree),
        ],
        check=True,
    )
    subprocess.run(["git", "-C", str(tree), "checkout", "--quiet", "HEAD"], check=False)
    return tree


#: The one job in the python matrix that checks out the whole history, and the
#: steps of it a shallow clone cannot run.
#:
#: `ci.yml` gives `fetch-depth: 0` to exactly one matrix entry -- ubuntu on the
#: floor interpreter -- so seven of the eight run against a checkout with no tags
#: and one runs against a checkout with all of them. The clone below models the
#: seven. The tests that read the history *skip* where there is no tag to measure
#: from, which is the right behaviour and also means the gate could not see them
#: fail: a commit message naming the internal working area passed here and failed
#: there, twice, before this was added.
#:
#: So these run in the caller's own tree, which has the tags a full checkout has.
#: They read git and touch nothing, which is what makes that safe.
DEEP_HISTORY_STEPS = (
    (
        "python (fetch-depth: 0)",
        "Checks that read the history, in a clone that has one and no identity",
        (
            f"{shlex.quote(sys.executable)} -m pytest -q -p no:cacheprovider "
            "tests/test_commit_messages.py tests/test_changelog_ledger.py "
            "tests/test_cited_commits.py"
        ),
    ),
)


#: The ledgers that read the maintainer's working notes, which the distribution
#: does not carry.
#:
#: Two directions of the theory ledger stand down where the notes are absent --
#: whether every result the argument tags is restated on the tracked page, and
#: whether every numbered citation names a work on the shelf -- and the notes
#: are absent on every runner. So those two checks run where they can run, which
#: is here, and a green matrix says nothing about them either way.
#:
#: Run in the caller's tree rather than in a clone: the notes are untracked, so
#: a checkout of `HEAD` does not have them and the checks would stand down there
#: exactly as they do on a runner.
NOTES_STEPS = (
    (
        "local only (the working notes)",
        "The ledgers that read the notes the distribution does not carry",
        (
            f"{shlex.quote(sys.executable)} -m pytest -q -p no:cacheprovider "
            "tests/test_theory_ledger.py tests/test_citation_ledger.py"
        ),
    ),
)


def floor_interpreter() -> str:
    """Read the oldest interpreter the python matrix runs.

    The floor is `ci.yml`'s to name, and it is read here rather than written
    down: a second copy of the number stops moving when the floor moves, and it
    stops in the direction that keeps passing -- the gate would go on building
    an interpreter nothing supports and reporting the suite green on it.
    `tests/test_local_gate.py` refuses the literal anywhere in this file.
    """
    matrix = workflow()["jobs"]["python"]["strategy"]["matrix"]
    versions = [str(version) for version in matrix["python-version"]]
    if not versions:
        message = "the python lane names no interpreter"
        raise RuntimeError(message)
    return versions[0]


def floor_environment() -> dict[str, str]:
    """Compose the environment the floor steps run in: a second one, throughout.

    Every name here points somewhere the caller's own interpreter does not.
    That is the whole content of this function, and it is what the reverted
    experiment got wrong in the other direction: a `uv` command handed the
    caller's `.venv` *wrote* it, uninstalling what the developer had built. A
    second release needs a second environment by definition, so the way to get
    that failure back is to leave one of these names off.

    The build directory is separate for the same reason one step down: the two
    interpreters link different libraries, so one `target/` shared between them
    rebuilds `pyo3` and everything under it on every alternation.
    """
    floor = floor_interpreter()
    venv = ROOT / "target" / f"gate-floor-{floor}"
    interpreter = venv / ("Scripts" if os.name == "nt" else "bin") / "python"
    return {
        **runner_environment(),
        "CI": "1",
        "UV_PROJECT_ENVIRONMENT": str(venv),
        "VIRTUAL_ENV": str(venv),
        # The extension is compiled against the floor's headers, not the
        # caller's. Left to `interpreter_env`, this would be the caller's
        # interpreter and the build would link the wrong ABI.
        "PYO3_PYTHON": str(interpreter),
        "CARGO_TARGET_DIR": str(ROOT / "target" / f"gate-cargo-{floor}"),
        # What the lane hands its own pytest step, so the suite draws the same
        # number of cases here as it does there.
        "HYPOTHESIS_PROFILE": "ci",
    }


def floor_steps() -> tuple[tuple[str, str, str], ...]:
    """Build the floor interpreter beside the caller's, and run the suite on it.

    The matrix runs seven releases; a developer runs one. Every difference
    between two of them is one this gate cannot see, and the floor is the end
    of the range where those land: the suite is written on the newest release
    the tree supports and read on the oldest, so a member that release does not
    have fails at *collection* and takes every job on that interpreter with it.
    Three did, in one push, from a green local run.

    The **product** suite alone. The repository checks read the tree, the
    workflow and the scripts; none of that answers differently by release, and
    running them twice would double the slowest half of the gate to re-derive
    the same verdict.

    Repeatable. The environment is cached under a path carrying the release, so
    a second run reuses it rather than refusing to touch it -- which is what
    `uv venv` does by default, and what made every run after the first red.
    """
    floor = floor_interpreter()
    venv = floor_environment()["UV_PROJECT_ENVIRONMENT"]
    return (
        (
            f"python {floor}",
            "The floor interpreter, and the extension built into it",
            (
                # `--allow-existing` because the environment is *cached*: the
                # path carries the release, so a floor that moves gets a
                # directory of its own, and one that has not moved is the one
                # built last time. Without it every run after the first failed
                # here, with an exit code and a hint that read as a broken
                # toolchain rather than as a directory already in place.
                f"uv venv --allow-existing --python {shlex.quote(floor)} "
                f"{shlex.quote(venv)} "
                "&& uv sync --locked --no-install-project "
                "&& uv run --no-sync maturin develop --uv"
            ),
        ),
        (
            f"python {floor}",
            "The product suite, on the floor interpreter",
            "uv run --no-sync pytest -q -p no:cacheprovider -m 'not repository'",
        ),
    )


def deep_clone(into: Path) -> Path:
    """Clone `HEAD` with its whole history and its tags, and with no committer.

    The other half of what `shallow_clone` models. One matrix leg takes
    `fetch-depth: 0` and the checks that read the history run there; the clone
    below is that leg, and what it adds beyond the history is an **absence**:
    a checkout configures no `user.name`, and a clone inherits none.

    That absence is the whole point. A ledger that *writes* a commit -- the
    planted orphan of `tests/test_cited_commits.py` -- refuses without a
    committer, and this repository has one in its own `.git/config`, so the
    check passed here and failed on the runner. Running it in the caller's tree
    could not have caught that at any price: the identity is in the tree. The
    global and system files are dropped with it, since a runner has neither.
    """
    tree = into / "deep"
    subprocess.run(
        ["git", "clone", "--quiet", "--no-hardlinks", f"file://{ROOT}", str(tree)],
        check=True,
    )
    return tree


#: What a checkout does not have and a developer's machine does.
#:
#: `actions/checkout` writes no `user.*`, so `git commit-tree` refuses on a
#: runner and answers here. Pointing both config files at an empty one is how
#: git is told there is no identity to find; the clone brings no local one.
NO_IDENTITY: dict[str, str] = {
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_SYSTEM": os.devnull,
}

#: `${{ env.NAME }}`, the one expression a step's command may carry that this
#: can resolve: the workflow's own `env:` block is in the file being read.
ENV_EXPRESSION = re.compile(r"\$\{\{\s*env\.(\w+)\s*\}\}")

#: Any expression at all. What is left after the `env.` ones are filled in is
#: the runner's to answer, and the gate must see that it is left rather than
#: hand the braces to bash.
ANY_EXPRESSION = re.compile(r"\$\{\{")


def resolved(command: str, environment: dict[str, str]) -> str | None:
    """Fill in `${{ env.X }}`, or answer `None` where anything is left.

    Every other expression -- `github.*`, `steps.*`, `matrix.*` -- is the
    runner's to answer, so a step carrying one is reported unresolved rather
    than guessed at. Checking only the `env.` ones would let `${{ github.sha }}`
    through untouched and hand the braces to bash, which is neither running the
    step nor refusing it.
    """
    missing = [
        name for name in ENV_EXPRESSION.findall(command) if name not in environment
    ]
    if missing:
        return None
    filled = ENV_EXPRESSION.sub(lambda m: environment[m.group(1)], command)
    return None if ANY_EXPRESSION.search(filled) else filled


def interpreter_env() -> dict[str, str]:
    """Where a developer's Python lives, which a runner does not have to say.

    The binding's tests link libpython, and a runner's system interpreter is on
    the default loader path while a virtual environment's is not. Passing it is
    not a deviation from the lane: it is what makes the same command mean the
    same thing here.
    """
    try:
        found = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys, sysconfig;"
                    "print(sys.executable);"
                    "print(sysconfig.get_config_var('LIBDIR') or '')"
                ),
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
    except (subprocess.CalledProcessError, OSError):
        return {}
    executable, libdir = [*found, "", ""][:2]
    return {
        "PYO3_PYTHON": executable,
        "LD_LIBRARY_PATH": f"{libdir}:{os.environ.get('LD_LIBRARY_PATH', '')}",
    }


def build_plan(spec: dict, jobs: list[str]) -> tuple[list[Step], list[str]]:
    """Collect the steps to run, and the ones only a runner can fill in."""
    workflow_env = {key: str(value) for key, value in (spec.get("env") or {}).items()}
    plan: list[Step] = []
    unresolved: list[str] = []
    for job in jobs:
        job_env = {
            key: str(value)
            for key, value in (spec["jobs"][job].get("env") or {}).items()
            if "${{" not in str(value)
        }
        for name, command, step_env in steps(spec, job):
            if not runnable(name):
                continue
            filled = resolved(command, workflow_env)
            if filled is None:
                unresolved.append(f"{job}: {name}")
                continue
            if name in NETWORK:
                filled = NETWORK[name][1]
            plan.append((job, name, filled, {**job_env, **step_env}))
    plan += [
        (job, f"{name} (the part that runs here)", command, {})
        for job in jobs
        for name, _, _ in steps(spec, job)
        if name in STANDINS
        for command, _ in [STANDINS[name]]
    ]
    return plan, unresolved


def fetched(plan: list[Step], cwd: Path) -> bool:
    """Run each planned network step's fetch, so the step itself can be offline.

    A fetch that will not land is not a verdict about this tree, so the caller
    turns a `False` here into `EXIT_CANNOT_RUN`. Retried, because the failure
    this guards against is a moment of the network rather than a state of it.
    """
    wanted = {name for _, name, _, _ in plan if name in NETWORK}
    for name in sorted(wanted):
        fetch = NETWORK[name][0]
        for attempt in range(1, FETCH_TRIES + 1):
            print(f"\n=== fetch for {name} (try {attempt} of {FETCH_TRIES})")
            result = subprocess.run(
                ["bash", "-euo", "pipefail", "-c", fetch],
                cwd=cwd,
                check=False,
                env={**runner_environment(), "CI": "1"},
            )
            if result.returncode == 0:
                break
        else:
            print(f"gate: cannot fetch what {name!r} reads: {fetch}")
            return False
    return True


#: Variables whose only purpose is to override a tool's "am I a terminal?"
#: check. A runner sets none of them.
FORCED_COLOUR = ("FORCE_COLOR", "CLICOLOR_FORCE")


def runner_environment() -> dict[str, str]:
    """Read the caller's environment, less what only a terminal would put in it.

    A step runs on the runner with no TTY and nothing forcing colour, so a tool
    there writes plain text -- including into any file it generates. A
    developer's terminal sets `FORCE_COLOR`, some harnesses set it for every
    child process, and a tool that honours it writes escape codes into that
    file too. `pip-audit` refuses a requirements file `uv export` wrote that
    way, and the gate reports a failed step: a verdict about the caller's
    terminal rather than about the tree, which is the one thing this gate must
    never give.

    Dropped rather than overridden with `NO_COLOR`, because the runner carries
    neither: what a step should see is the absence.
    """
    return {key: value for key, value in os.environ.items() if key not in FORCED_COLOUR}


def step_environment(environment: dict[str, str], outputs: Path) -> dict[str, str]:
    """Compose the environment a planned step runs in.

    A runner's environment, the interpreter the binding's tests link, and the
    step's own variables, in that order -- so a workflow's `env:` wins over a
    default and the runner's terminal wins over nothing.

    Named rather than built inside the run, because what a step is handed is
    half of what "the gate runs the lane's steps" means: a `cargo test` that
    cannot find libpython fails on a missing shared object, which reads as a
    broken tree rather than as a caller's loader path.
    """
    return {
        **runner_environment(),
        **interpreter_env(),
        **environment,
        "CI": "1",
        # One build directory across the jobs and across runs. The clone is what
        # this reproduces; a cold Rust build is not, and paying for one per job
        # would make the gate something nobody runs.
        "CARGO_TARGET_DIR": str(ROOT / "target" / "gate"),
        # A step that records an output writes to a file the runner names.
        # Nothing here reads it back -- the steps that would are the ones this
        # gate cannot run -- but the write must land somewhere rather than
        # failing.
        "GITHUB_OUTPUT": str(outputs / "github_output"),
        "GITHUB_STEP_SUMMARY": str(outputs / "step_summary"),
    }


def whole_environment(environment: dict[str, str], outputs: Path) -> dict[str, str]:
    """Take the environment as given, and only name the files a step may write.

    `step_environment` composes a *step's* `env:` over the runner's and then
    wins on four names of its own -- which is right for a step read out of the
    workflow and wrong for the floor, whose whole point is that two of those
    four names point somewhere else.
    """
    return {
        **environment,
        "GITHUB_OUTPUT": str(outputs / "github_output"),
        "GITHUB_STEP_SUMMARY": str(outputs / "step_summary"),
    }


def run_step(
    name: str,
    command: str,
    cwd: Path,
    environment: dict[str, str],
    compose: Callable[[dict[str, str], Path], dict[str, str]] = step_environment,
) -> bool:
    print(f"\n=== {name}")
    with tempfile.TemporaryDirectory() as outputs:
        result = subprocess.run(
            ["bash", "-euo", "pipefail", "-c", command],
            cwd=cwd,
            check=False,
            env=compose(environment, Path(outputs)),
        )
    if result.returncode != 0:
        print(f"FAILED: {name} (exit {result.returncode})")
        return False
    return True


def list_plan(
    spec: dict,
    jobs: list[str],
    plan: list[Step],
    unresolved: list[str],
    *,
    floor: bool,
) -> None:
    """Print every step a run would take, in the order the run takes them."""
    show_plan(spec, jobs, plan, unresolved)
    for job, name, _ in NOTES_STEPS:
        print(f"run   {job}: {name}")
    for job, name, _ in floor_steps() if floor else ():
        print(f"run   {job}: {name}")


def notes_failures() -> list[str]:
    """Run the ledgers that read the notes, and give back what failed.

    A function of its own so the one caller stays a list of lanes: the step
    runs in the caller's tree rather than in a clone, which is the whole of
    what makes it different from the lanes around it.
    """
    return [
        f"{job}: {name}"
        for job, name, command in NOTES_STEPS
        if not run_step(f"{job}: {name}", command, ROOT, runner_environment())
    ]


def show_plan(
    spec: dict, jobs: list[str], plan: list[Step], unresolved: list[str]
) -> None:
    """Print what would run and what would not, with the reason for each skip."""
    for job, name, _, _ in plan:
        print(f"run   {job}: {name}")
    for step in unresolved:
        print(f"skip  {step} -- carries an expression only a runner answers")
    for job in jobs:
        for name, _, _ in steps(spec, job):
            if not runnable(name):
                print(f"skip  {job}: {name} -- {NEEDS_A_RUNNER[name]}")
                if name in STANDINS:
                    print(f"      stands in: {STANDINS[name][0]} ({STANDINS[name][1]})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="show the plan and stop")
    parser.add_argument("--job", help="run one job's steps")
    parser.add_argument(
        "--here",
        action="store_true",
        help="use this working tree instead of a fresh shallow clone",
    )
    args = parser.parse_args()

    spec = workflow()
    jobs = [args.job] if args.job else required_jobs(spec)
    unknown = [job for job in jobs if job not in spec.get("jobs", {})]
    if unknown:
        print(f"gate: no such job: {', '.join(unknown)}")
        return EXIT_CANNOT_RUN

    plan, unresolved = build_plan(spec, jobs)
    # A second interpreter is `uv`'s to fetch, and one job asked for by name is
    # not the matrix. Decided here so `--list` says the same thing the run does.
    floor_runs = not args.job and shutil.which("uv") is not None
    if args.list:
        list_plan(spec, jobs, plan, unresolved, floor=floor_runs)
        return EXIT_OK

    if args.here:
        tree = ROOT
        holder = None
        print("gate: running in this working tree (--here)")
    else:
        holder = Path(tempfile.mkdtemp(prefix="valgebra-gate-"))
        tree = shallow_clone(holder)
        print(f"gate: shallow clone of HEAD, no tags, at {tree}")

    try:
        if not fetched(plan, tree):
            print()
            print("gate: a step's fetch did not land, so the gate could not run.")
            return EXIT_CANNOT_RUN
        failures = [
            f"{job}: {name}"
            for job, name, command, environment in plan
            if not run_step(f"{job}: {name}", command, tree, environment)
        ]
        # And the steps the shallow clone is the wrong shape for, in a clone
        # that is the right one. `holder` is the temporary directory both
        # clones live in, and it is `None` exactly when `--here` gave the
        # caller's tree instead -- so asking for it is the same condition as
        # asking for `--here`, and it is the one that also says where to clone.
        if holder is not None:
            deep = deep_clone(holder)
            failures += [
                f"{job}: {name}"
                for job, name, command in DEEP_HISTORY_STEPS
                if not run_step(f"{job}: {name}", command, deep, NO_IDENTITY)
            ]
        # The ledgers no runner can run, in the tree that has what they read.
        failures += notes_failures()
        # And the release the suite is read on rather than written on. This runs
        # in the same tree as the plan: what makes it a second lane is the
        # environment, not the checkout.
        if floor_runs:
            failures += [
                f"{job}: {name}"
                for job, name, command in floor_steps()
                if not run_step(
                    f"{job}: {name}",
                    command,
                    tree,
                    floor_environment(),
                    whole_environment,
                )
            ]
    finally:
        if holder is not None:
            shutil.rmtree(holder, ignore_errors=True)

    print()
    if failures:
        print(f"gate: {len(failures)} step(s) failed: {', '.join(failures)}")
        return EXIT_FAIL
    deep_steps = 0 if args.here else len(DEEP_HISTORY_STEPS)
    floor_count = len(floor_steps()) if floor_runs else 0
    total = len(plan) + deep_steps + floor_count + len(NOTES_STEPS)
    print(
        f"gate: {total} step(s) passed in a clone shaped like the runner's, in a "
        "clone that keeps the history and has no committer, on the floor "
        "interpreter, and over the notes no runner carries."
    )
    report_what_was_not_run(here=args.here, floor=floor_runs)
    return EXIT_OK


#: What no arrangement of this gate reaches, **by the job name it is**, so a
#: green run is read for what it is.
#:
#: `NEEDS_A_RUNNER` is per step and held to the workflow, and these are whole
#: jobs, so nothing carried them: three sentences of prose that a job added
#: tomorrow would not appear in, under a closing line that would still read
#: complete. Keyed by job instead, and `tests/test_local_gate.py` holds every
#: merge-gate job to being planned, excused step by step, or named here.
UNREACHED = {
    "python": (
        "the floor is built and the product suite runs on it; what is left "
        "unread is every release between the floor and the caller's, the "
        "prerelease, and the free-threaded build"
    ),
    "bench": "cachegrind and a base built beside the head",
    "bench-free-threaded": "the optimized wheel, and a second interpreter",
    "wheel": "the release build matrix",
    "binding-coverage": "an instrumented rebuild of the whole workspace",
    "mutants-diff-core": "a mutation sweep, tens of minutes",
    "mutants-diff-walk": (
        "a mutation sweep, and its verdict is the embedded interpreter's -- "
        "run it with PYO3_PYTHON at the 3.12 `ci.yml` names, or a mutant the "
        "lane kills reads here as a survivor"
    ),
}


def report_what_was_not_run(*, here: bool, floor: bool) -> None:
    """Say what a green gate did not check, because a green gate is read as more.

    The audit's standing rule is that a slice names the lanes this gate cannot
    run and what was done about each. The gate itself never said, so the reader
    doing the naming had to remember the list -- and three lanes went red in one
    week on things it does not reach. Printed on success only: a failure is
    already telling the reader to look somewhere.
    """
    print(
        f"gate: {len(NEEDS_A_RUNNER)} step(s) excused by name, and "
        f"{len(UNREACHED)} job(s) this gate does not reach at all:"
    )
    for job, why in UNREACHED.items():
        print(f"  - {job}: {why}")
    if here:
        print("  - python (fetch-depth: 0): --here skips the history checks")
    if not floor:
        print(
            f"  - python {floor_interpreter()}: `uv` is not on PATH, or one job "
            f"was asked for by name, so the floor interpreter was not built"
        )


if __name__ == "__main__":
    sys.exit(main())
