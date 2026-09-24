"""The local gate runs the merge gate's steps, or names the ones it cannot.

`AGENTS.md` lists commands a developer runs in a full clone with a warm virtual
environment; CI is forty-odd jobs in shallow clones with pinned tools on three
operating systems. A local clone carries tags and a checkout does not, so a
check reading `git describe` answers one way in each -- and a difference of that
shape is invisible until a push finds it.

`scripts/gate.py` closes that by running the workflow's own `run:` steps in a
clone shaped like the runner's. This holds the two halves of that claim:

* every `run:` step of every merge-gate job is either **in the plan the gate
  builds** or named in its `NEEDS_A_RUNNER` list with the reason, so a step
  added to CI is one the local gate runs or refuses by name;
* nothing in that list has outlived its step, so the excuses cannot accumulate.

The first of those is asked of the *plan*, not of `runnable`. Asked of
`runnable` it was `name in NEEDS_A_RUNNER and name not in NEEDS_A_RUNNER` -- a
contradiction, so the list it built was empty for every workflow and the
assertion could not fail. The defect it now catches is planted below.

And the property the whole thing exists for: the tree it runs in is a shallow
clone with no tags.

LEDGER: every merge-gate step is run by the local gate or excused by name
"""

from __future__ import annotations

import importlib.util
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

pytestmark = pytest.mark.repository

if TYPE_CHECKING:
    from types import ModuleType

ROOT = Path(__file__).resolve().parent.parent
GATE = ROOT / "scripts" / "gate.py"
SCRIPTS = ROOT / "scripts"


def _load_gate() -> ModuleType:
    """Import the gate by path; ``scripts/`` is not an importable package."""
    spec = importlib.util.spec_from_file_location("valgebra_gate", GATE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_gate()


def _merge_gate_steps() -> list[tuple[str, str]]:
    spec = gate.workflow()
    return [
        (job, name)
        for job in gate.required_jobs(spec)
        for name, _, _ in gate.steps(spec, job)
    ]


def _planned(spec: dict) -> set[str]:
    """Collect the steps `gate.py` would actually run, as `job: name`."""
    plan, _ = gate.build_plan(spec, gate.required_jobs(spec))
    return {f"{job}: {name}" for job, name, _, _ in plan}


def test_every_merge_gate_step_is_planned_or_excused_by_name() -> None:
    spec = gate.workflow()
    steps = _merge_gate_steps()
    # The scan is the detector: no steps at all would pass having read nothing.
    assert len(steps) >= 20, f"the workflow scan found only {steps}"

    planned = _planned(spec)
    unaccounted = sorted(
        f"{job}: {name}"
        for job, name in steps
        if f"{job}: {name}" not in planned and name not in gate.NEEDS_A_RUNNER
    )
    assert not unaccounted, (
        f"merge-gate steps the local gate neither runs nor excuses: {unaccounted}. "
        "A step carrying an expression only a runner answers is skipped silently "
        "unless it is named in NEEDS_A_RUNNER with the reason."
    )
    # Both columns must be non-empty: a gate that planned nothing and a list
    # that excused nothing would each pass the assertion above.
    assert planned, "the gate runs nothing"
    assert any(name in gate.NEEDS_A_RUNNER for _, name in steps), (
        "no step needs a runner, which means the list stopped being read"
    )


def test_a_step_only_a_runner_can_fill_in_is_unaccounted() -> None:
    """The defect the assertion above exists to catch, planted on a spec.

    `${{ github.sha }}` is the runner's to answer and `gate.py` cannot. Before
    this, `resolved` searched for `env.` expressions alone, so such a step came
    back "resolved" with its braces intact, was planned, and would have been
    handed to bash verbatim -- while the ledger passed, because its filter was a
    contradiction. Planting it on a synthetic workflow keeps the case in the
    suite without a step in `ci.yml` that exists only to be caught.
    """
    planted = "A step only a runner can fill in"
    spec = {
        "env": {},
        "jobs": {
            "ci": {"needs": ["planted"]},
            "planted": {
                "steps": [{"name": planted, "run": 'echo "${{ github.sha }}"'}]
            },
        },
    }
    plan, unresolved = gate.build_plan(spec, gate.required_jobs(spec))
    assert plan == [], "a step the gate cannot fill in must not be planned"
    assert unresolved == [f"planted: {planted}"]
    assert planted not in gate.NEEDS_A_RUNNER
    assert f"planted: {planted}" not in _planned(spec), (
        "the filter must flag a step that is neither planned nor excused"
    )


def test_an_env_expression_the_workflow_defines_is_filled_in() -> None:
    """And the other direction, so the refusal is not a blanket one."""
    assert gate.resolved("echo ${{ env.X }}", {"X": "1"}) == "echo 1"
    assert gate.resolved("echo ${{ env.X }}", {}) is None
    assert gate.resolved("echo ${{ github.sha }}", {"X": "1"}) is None
    assert gate.resolved("echo ${{ env.X }} ${{ matrix.os }}", {"X": "1"}) is None
    assert gate.resolved("echo plain", {}) == "echo plain"


def test_no_excuse_has_outlived_its_step() -> None:
    named = {name for _, name in _merge_gate_steps()}
    stale = sorted(set(gate.NEEDS_A_RUNNER) - named)
    assert not stale, (
        f"steps excused from the local gate that the workflow no longer has: "
        f"{stale}. Delete each with the reason beside it."
    )


def test_every_stand_in_is_for_a_step_the_gate_excuses() -> None:
    """A stand-in for a step the gate already runs is a step run twice."""
    loose = sorted(set(gate.STANDINS) - set(gate.NEEDS_A_RUNNER))
    assert not loose, (
        f"stand-ins for steps the gate is not excused from: {loose}. A step the "
        "gate can run is run, not stood in for."
    )
    named = {name for _, name in _merge_gate_steps()}
    stale = sorted(set(gate.STANDINS) - named)
    assert not stale, f"stand-ins for steps the workflow no longer has: {stale}"


def test_no_stand_in_repeats_a_step_the_gate_runs() -> None:
    """A stand-in is the part of an excused step a developer can reach.

    The other direction of the rule above, and the one a workflow change
    causes rather than a deletion: the excused step stays excused, and a *new*
    lane arrives that runs the very command the stand-in was written to
    substitute for. The gate then pays for it twice, and the second failure
    names a substitute rather than the step a reader would go looking for.

    That is what happened when the binding's corpora gained a lane of their
    own: they had been reachable here only through the coverage rebuild's
    stand-in, and afterwards through both.
    """
    spec = gate.workflow()
    plan, _ = gate.build_plan(spec, gate.required_jobs(spec))
    scripts = [script for _, _, script, _ in plan]
    repeated = sorted(
        step
        for step, (command, _) in gate.STANDINS.items()
        if any(command in script for script in scripts)
    )
    assert not repeated, (
        f"stand-ins whose command the gate also runs outright: {repeated}. "
        "Drop the row; the step it stood in for is reached by a step of its "
        "own, and running it twice buys the second nothing."
    )


def test_every_network_row_stands_for_a_step_the_gate_runs() -> None:
    """An offline substitute for a step the gate does not run stands for nothing.

    Held in both directions. A row whose step the workflow dropped is an
    offline command nothing runs, and a row for a step the gate excuses instead
    of running would fetch for a step that never happens.
    """
    spec = gate.workflow()
    named = {name for _, name in _merge_gate_steps()}
    stale = sorted(set(gate.NETWORK) - named)
    assert not stale, f"offline substitutes for steps the workflow lacks: {stale}"

    excused = sorted(set(gate.NETWORK) & set(gate.NEEDS_A_RUNNER))
    assert not excused, (
        f"offline substitutes for steps the gate does not run: {excused}. A step "
        "the gate excuses needs no fetch."
    )

    planned = {
        name for _, name, _, _ in gate.build_plan(spec, gate.required_jobs(spec))[0]
    }
    missing = sorted(set(gate.NETWORK) - planned)
    assert not missing, f"offline substitutes for steps outside the plan: {missing}"


def test_a_network_step_runs_its_offline_form() -> None:
    """The plan carries the offline command, not the workflow's online one.

    The substitution is the whole of the fix: the runner keeps the online form,
    because catching a newly published advisory is its job, and this gate reads
    the database the fetch already landed.
    """
    spec = gate.workflow()
    plan, _ = gate.build_plan(spec, gate.required_jobs(spec))
    for name, (_, offline) in gate.NETWORK.items():
        commands = [command for _, step, command, _ in plan if step == name]
        assert commands == [offline], (
            f"{name!r} is planned as {commands}, not as its offline form {offline!r}"
        )


def test_the_interpreter_backed_binding_tests_are_in_the_plan() -> None:
    """The hole the stand-ins exist for, asked of the plan the gate builds.

    Seventy-odd Rust tests link an embedded interpreter, and the merge gate
    reaches them only inside a coverage rebuild the local gate cannot run. Both
    halves are needed: the command has to carry the feature, and it has to be in
    the plan rather than in a table nothing reads.
    """
    spec = gate.workflow()
    plan, _ = gate.build_plan(spec, gate.required_jobs(spec))
    running = [command for _, _, command, _ in plan if "interpreter-tests" in command]
    assert running, (
        "no step in the plan runs the binding's interpreter-backed tests; "
        "without one they are in no local step at all"
    )


def test_the_workspace_test_step_is_handed_the_interpreter_on_the_loader_path(
    tmp_path: Path,
) -> None:
    """The other half: the environment the step is handed, not the plan it is in.

    The workspace test binary links libpython -- `crates/valgebra-py` sets no
    `extension-module` -- so `cargo test` starts only where a shared-libpython
    interpreter's library directory is on the loader path. A runner's system
    interpreter is there and a virtual environment's is not, which is what
    `interpreter_env` supplies and `step_environment` composes.

    Asked of the composition rather than of a spawned shell: what a step is
    handed is the same on every operating system, while the shell that would
    read it back is not.
    """
    spec = gate.workflow()
    plan, _ = gate.build_plan(spec, gate.required_jobs(spec))
    workspace = [
        command
        for _, _, command, _ in plan
        if "cargo test" in command and "--manifest-path" not in command
    ]
    assert workspace, "the workspace test step is in no plan the gate builds"

    supplied = gate.interpreter_env()
    assert set(supplied) == {"PYO3_PYTHON", "LD_LIBRARY_PATH"}
    composed = gate.step_environment({}, tmp_path)
    assert {name: composed[name] for name in supplied} == supplied
    # A step's own variables win over the defaults, and the runner's CI marker
    # is set whatever the caller's shell says.
    theirs = gate.step_environment({"PYO3_PYTHON": "theirs"}, tmp_path)
    assert theirs["PYO3_PYTHON"] == "theirs"
    assert composed["CI"] == "1"


def test_every_excuse_carries_a_reason() -> None:
    empty = sorted(name for name, why in gate.NEEDS_A_RUNNER.items() if not why.strip())
    assert not empty, f"steps excused with no reason: {empty}"


def test_the_python_suite_is_one_of_the_steps_it_runs() -> None:
    # The step that would have caught the failure this gate exists for: the
    # whole Python suite, run in a clone with no tags.
    runnable = {name for _, name in _merge_gate_steps() if gate.runnable(name)}
    assert "pytest" in runnable


def test_the_gate_runs_in_a_shallow_clone_with_no_tags(tmp_path: Path) -> None:
    """The property the gate exists for, checked on a real clone."""
    tree = gate.shallow_clone(tmp_path)
    depth = subprocess.run(  # noqa: S603 - fixed argv, no shell, test-only
        ["git", "-C", str(tree), "rev-list", "--count", "HEAD"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    )
    assert depth.stdout.strip() == "1", "the clone is not shallow"
    tags = subprocess.run(  # noqa: S603 - fixed argv, no shell, test-only
        ["git", "-C", str(tree), "tag", "--list"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    )
    assert tags.stdout.strip() == "", "the clone carries tags a checkout would not"


def test_a_forced_colour_variable_does_not_reach_a_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A step sees the environment a runner has, not the caller's terminal.

    The plant: `FORCE_COLOR` set in the caller's shell reached `uv export`,
    which wrote ANSI escapes into the requirements file it generates, and
    `pip-audit` refused the file. The gate reported a failed step for a
    property of the terminal it was run from.

    The second plant: `VIRTUAL_ENV`, which `uv run` sets for the gate itself,
    reached a clone step's `uv pip install`, which installed the wheel into the
    caller's venv, and the step after it ran under `uv run` in the clone's venv
    and could not import what was installed. The gate reported the profile
    comparison failed for a property of how it was launched.

    The third: that venv's `bin` on `PATH`, where `uv run --no-sync pytest` in a
    clone with no environment of its own found the caller's pytest, and the
    gate reported green on the caller's tree.
    """
    monkeypatch.setenv("FORCE_COLOR", "3")
    monkeypatch.setenv("CLICOLOR_FORCE", "1")
    venv = "/somewhere/else/.venv"
    monkeypatch.setenv("VIRTUAL_ENV", venv)
    monkeypatch.setenv("VALGEBRA_GATE_MARKER", "kept")
    toolchain = os.environ.get("PATH", "")
    # Spelled as the platform names a venv's scripts, which is `Scripts` under
    # a Windows separator and `bin` elsewhere: the entry `uv run` puts there.
    callers = str(Path(venv) / ("Scripts" if os.name == "nt" else "bin"))
    monkeypatch.setenv("PATH", os.pathsep.join([callers, toolchain]))
    environment = gate.runner_environment()
    assert "FORCE_COLOR" not in environment
    assert "CLICOLOR_FORCE" not in environment
    assert "VIRTUAL_ENV" not in environment
    # Everything else is carried: the gate runs the lane's steps in the
    # caller's toolchain, and dropping more than the terminal's own would make
    # it a different environment rather than a runner's. The one `PATH` entry
    # that goes is the caller's venv, which `uv run` put there with the
    # variable above: through it a clone step without an environment of its
    # own ran the caller's pytest against the caller's tree.
    assert environment["VALGEBRA_GATE_MARKER"] == "kept"
    assert environment.get("PATH") == toolchain


#: Every script that answers in the three-code vocabulary, by the path a caller
#: runs. Each is loaded by path rather than imported, because three of the four
#: are scripts rather than modules of a package.
GATE_SCRIPTS = [
    "docs_lint.py",
    "gate.py",
    "compare_gate.py",
    "mutation_gate.py",
    "perf_gate.py",
    "coverage_gate.py",
]


@pytest.mark.parametrize("script", GATE_SCRIPTS)
def test_a_gate_script_answers_in_the_three_code_vocabulary(script: str) -> None:
    """Every gate script says the same three things with the same three codes.

    `0` ran and passed, `1` ran and failed, `2` **could not run**. The third is
    the one that has to be its own code: a gate that could not run has proven
    nothing, and a caller that reads it as a failure reruns forever while one
    that reads it as a pass ships on no evidence.

    Held over every script at once rather than once per script, because the
    vocabulary is shared. Four copies of this assertion each said their own
    script uses 0, 1 and 2 and none of them said the four agree -- which is the
    half a caller dispatching on the code depends on.
    """
    name = f"gate_codes_{script}"
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / script)
    assert spec is not None, f"{script} has no spec"
    assert spec.loader is not None, f"{script} has no loader"
    module = importlib.util.module_from_spec(spec)
    # Registered before it runs: a script defining a dataclass has its class
    # read back out of `sys.modules` under its own `__module__`, and a module
    # executed without being registered is not there to read.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[name]

    codes = (module.EXIT_OK, module.EXIT_FAIL, module.EXIT_CANNOT_RUN)
    assert codes == (0, 1, 2), f"{script} answers with {codes}"


def test_every_merge_gate_job_is_planned_excused_or_named_unreached() -> None:
    """The third direction: a whole job, not a step of one.

    `NEEDS_A_RUNNER` is per step, so a job every one of whose steps is excused
    reads as accounted for while the gate touches none of it -- and the closing
    line, which is what a reader takes away, said "24 steps excused" and named
    the missing lanes in three sentences of prose. A job added to `ci.yml`
    tomorrow would appear in neither, and the line would still read complete.

    So every merge-gate job is in exactly one of three places: the gate plans a
    step of it, every step of it is excused by name, or the job itself is in
    `UNREACHED` with the reason it cannot run here.
    """
    spec = gate.workflow()
    jobs = gate.required_jobs(spec)
    assert len(jobs) >= 10, f"the workflow scan found only {jobs}"

    planned_jobs = {planned.split(":", 1)[0] for planned in _planned(spec)}
    unaccounted = []
    for job in jobs:
        if job in planned_jobs or job in gate.UNREACHED:
            continue
        names = [name for name, _, _ in gate.steps(spec, job)]
        if names and all(name in gate.NEEDS_A_RUNNER for name in names):
            continue
        unaccounted.append(job)
    assert not unaccounted, (
        f"merge-gate jobs the local gate neither reaches nor names: {unaccounted}. "
        "Add the job to `UNREACHED` with what stops it running here, or the "
        "gate's closing line reports a green run that did not touch it."
    )


def test_a_job_named_unreached_is_one_the_workflow_has() -> None:
    """The other way: a name that outlives its job excuses nothing.

    `UNREACHED` is read only to be printed, so a job renamed in `ci.yml` would
    leave a line naming nothing and drop the real job out of the count without
    failing anything.
    """
    spec = gate.workflow()
    jobs = set(spec.get("jobs", {}))
    absent = sorted(job for job in gate.UNREACHED if job not in jobs)
    assert not absent, (
        f"`UNREACHED` names jobs this workflow does not have: {absent}. "
        "A job that was renamed takes its reason with it."
    )
    assert all(reason.strip() for reason in gate.UNREACHED.values()), (
        "a job named with no reason is a job nobody can act on"
    )


def _matrix_interpreters() -> list[str]:
    """Read the interpreters the python lane runs, floor first."""
    matrix = gate.workflow()["jobs"]["python"]["strategy"]["matrix"]
    versions = [str(version) for version in matrix["python-version"]]
    assert versions, "the python lane names no interpreter"
    return versions


def test_the_gate_builds_the_floor_interpreter_the_matrix_names() -> None:
    """The floor the gate builds is the floor the workflow runs, or it is a guess.

    The matrix runs seven interpreters and the gate runs one: the caller's. A
    difference between two releases is therefore a difference this gate cannot
    see, and the floor is the end of that range where the difference actually
    lands -- `typing.Self`, `LiteralString` and `Unpack` are 3.11, and each of
    them reddened nine jobs from a green local run.

    So the gate builds the floor beside the caller's interpreter. Which release
    that is belongs to `ci.yml`, not here: a second copy of "3.10" is a number
    that goes stale the day the floor moves, silently, in the one direction
    where a stale floor tests *less* than it claims. Held by refusing the
    literal anywhere in the gate.
    """
    floor = _matrix_interpreters()[0]
    assert gate.floor_interpreter() == floor

    source = GATE.read_text(encoding="utf-8")
    assert floor not in source, (
        f"`scripts/gate.py` spells the floor interpreter {floor!r} itself. It is "
        f"in `ci.yml`, and a copy of it here is a copy that stops moving when "
        f"the floor does"
    )


def test_the_floor_interpreter_runs_the_product_suite() -> None:
    """Building the floor is not the claim; running the suite on it is.

    An extension that builds and imports on the floor is an extension the floor
    can load, which is the smaller half. The half that has gone red is the
    suite: a test module that names a typing member the floor does not have
    fails at collection, and every job on that interpreter goes with it.
    """
    commands = [command for _, _, command in gate.floor_steps()]
    assert commands, "the gate builds no floor interpreter"

    built = [command for command in commands if "uv venv" in command]
    assert built, f"no step builds the floor interpreter: {commands}"
    assert gate.floor_interpreter() in built[0], (
        f"the floor environment is built without naming the floor: {built[0]!r}"
    )

    ran = [command for command in commands if "pytest" in command]
    assert ran, f"the floor interpreter is built and nothing runs on it: {commands}"
    assert "not repository" in ran[0], (
        f"the floor runs {ran[0]!r}, which is not the product suite -- the "
        f"repository checks read the tree and answer the same on any release"
    )


def test_the_floor_build_does_not_write_the_callers_environment() -> None:
    """The floor is built beside the caller's, never into it.

    This is the reverted experiment, in the one place it would be tempting
    again: lending the caller's `.venv` to a step in the clone uninstalled the
    extension and the bench group from the tree the developer was working in.
    A second interpreter needs a second environment by definition, so the only
    way to get that failure back is to point this one at the first.
    """
    environment = gate.floor_environment()
    venv = Path(environment["UV_PROJECT_ENVIRONMENT"])
    assert venv != ROOT / ".venv", "the floor build would overwrite the caller's venv"
    assert not venv.is_relative_to(ROOT / ".venv")

    # And the Rust side of the same point: the floor links a different libpython,
    # so sharing one build directory with the caller's interpreter rebuilds
    # `pyo3` both ways on every run.
    shared = gate.step_environment({}, ROOT)["CARGO_TARGET_DIR"]
    assert environment["CARGO_TARGET_DIR"] != shared


#: The gate runs the merge gate's own steps, and those are `runs-on:
#: ubuntu-latest`; this drives one of them through a shell. A Windows checkout
#: resolves `bash` to Git-Bash or to WSL, neither of which is the shell the step
#: runs in, so a run there reports on the shell that answered rather than on the
#: step. Every other lane asks it -- each Linux interpreter and macOS -- so the
#: skip is the one platform the step never reaches.
@pytest.mark.skipif(
    sys.platform == "win32",
    reason="the gate's steps run on ubuntu-latest; Git-Bash is a different shell",
)
def test_the_floor_build_runs_a_second_time(tmp_path: Path) -> None:
    """A gate run is repeatable, and the floor step is the one that was not.

    The floor interpreter is built into a directory under `target/`, and the
    page says it is "a `uv venv` and a second build the first time, and cached
    after". Cached after is the claim; refusing to touch an existing
    environment is what `uv venv` does by default, so every run after the first
    failed at that step -- and it failed with an exit code and a hint, which a
    reader takes for a broken toolchain rather than for a directory that is
    already there.

    Driven rather than read off the command: the flag's name is uv's, the
    behaviour is what the step needs, and a test asserting the spelling would
    pass on a flag that means something else.
    """
    creation = gate.floor_steps()[0][2].split("&&")[0].strip()
    assert creation.startswith("uv venv"), creation

    # Point the same command at a directory of this test's own, twice.
    target = tmp_path / "floor"
    rerun = creation.rsplit(" ", 1)[0] + " " + shlex.quote(str(target))
    for attempt in (1, 2):
        done = subprocess.run(  # noqa: S603 - the gate's own command, test-only
            [  # noqa: S607 - bash is on the path of every machine this runs on
                "bash",
                "-euo",
                "pipefail",
                "-c",
                rerun,
            ],
            capture_output=True,
            text=True,
            check=False,
            cwd=ROOT,
        )
        assert done.returncode == 0, (
            f"the floor environment could not be built on attempt {attempt}: "
            f"{done.stdout}{done.stderr}"
        )
    assert (target / "pyvenv.cfg").is_file()


def test_the_instruction_gate_runs_here_or_says_why() -> None:
    """The bench lane is excused whole, and its comparison is not part of it.

    The lane wants cachegrind, a profiled wheel, a second interpreter and a
    system package installed with `sudo`, so the gate excuses it by name. The
    *comparison* wants none of those: two builds of one workload, measured in
    one job. Excusing it with the rest is how a change that read sound and
    cost seventy-one times the instructions passed this script with every
    step green.

    So the step runs, or it names the one thing it lacks. Both are answers;
    silence is not.
    """
    runs, excused = gate.perf_plan()
    assert runs or excused, "the instruction gate is neither run nor excused"
    assert not (runs and excused), "a step both run and excused"
    for row in excused:
        assert row.endswith(
            ("valgrind is not on PATH", "no commit to measure against")
        ), f"an excuse the plan does not name: {row}"


def test_the_instruction_gate_names_modes_the_measurer_has() -> None:
    """A mode the gate asks for and `perf_gate.py` does not define is a typo.

    The step is a command line rather than a call, so nothing but this reads
    the two together: the gate would run, the measurer would refuse the
    argument, and the step would fail for a reason that is not a regression.
    """
    measurer = (ROOT / "scripts" / "perf_gate.py").read_text(encoding="utf-8")
    for _, _, mode in gate.PERF_STEPS:
        assert f'"{mode.removeprefix("--")}"' in measurer, (
            f"the gate asks for {mode}, which `perf_gate.py` does not define"
        )


def test_the_instruction_gate_measures_against_an_ancestor(tmp_path: Path) -> None:
    """The base is the merge base, so it is always in this history.

    What the lane's event names for a pull request is `base.sha`, the commit
    the two histories share. A *tip* is that only where the branch is ahead of
    it: on a branch behind the default one the tip is not reachable from
    `HEAD`, and the step then compares two unrelated trees -- which reads as a
    regression or an improvement according to what else landed meanwhile,
    never as the change under test.

    The repository this runs in is ahead of its remote or equal to it, which
    is the one shape where a tip is right, so the rule is asked of a checkout
    put behind its default branch instead.

    Built rather than cloned from this tree, for the reason the perf gate's
    own stage is: a synthetic history carries its own commits, so this runs in
    the shallow checkouts the lanes take rather than skipping there. A clone of
    a depth-one checkout has one commit and nothing to go behind, and reading
    two back in it is an error rather than a branch behind its remote.
    """

    def git(*args: str, tree: Path) -> str:
        done = subprocess.run(  # noqa: S603 - fixed argv, no shell, test-only
            ["git", "-C", str(tree), *args],  # noqa: S607
            check=True,
            capture_output=True,
            text=True,
        )
        return done.stdout.strip()

    upstream = tmp_path / "upstream"
    upstream.mkdir()
    git("init", "--quiet", "--initial-branch=main", tree=upstream)
    git("config", "user.email", "gate@example.invalid", tree=upstream)
    git("config", "user.name", "gate", tree=upstream)
    for step in range(4):
        git("commit", "--quiet", "--allow-empty", "-m", f"commit {step}", tree=upstream)

    clone = tmp_path / "behind"
    subprocess.run(  # noqa: S603 - fixed argv, no shell, test-only
        ["git", "clone", "--no-hardlinks", "--quiet", str(upstream), str(clone)],  # noqa: S607
        check=True,
        capture_output=True,
    )
    behind = git("rev-parse", "HEAD~2", tree=clone)
    git("checkout", "--detach", "--quiet", behind, tree=clone)

    base = gate.perf_base(clone)
    assert base is not None, "a clone with history has something to measure against"
    reachable = subprocess.run(  # noqa: S603 - fixed argv, no shell, test-only
        ["git", "-C", str(clone), "merge-base", "--is-ancestor", base, "HEAD"],  # noqa: S607
        check=False,
        capture_output=True,
    )
    assert reachable.returncode == 0, (
        f"the gate would measure {behind[:12]} against {base[:12]}, which is not "
        f"in its history: a comparison of two unrelated trees"
    )
