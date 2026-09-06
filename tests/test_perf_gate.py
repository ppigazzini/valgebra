"""The instruction-count gate must fail on every way a measurement can be wrong.

A gate that cannot be shown to fail is not evidence. These drive the decision
logic of ``scripts/perf_gate.py`` directly -- no cachegrind, no build -- so each
refusal is exercised: a count over the ceiling, a count under the floor (the
shape a workload that stopped doing the work produces), a workload checksum that
does not match, and output the gate cannot read at all.

The under-floor case is the one a one-sided budget misses: a hollow workload
measures low and reads as an improvement it never earned.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import yaml

# The repository checks are not the product suite: this file reads the tree,
# the configuration and the gate scripts, none of which ship in a wheel.
pytestmark = pytest.mark.repository

if TYPE_CHECKING:
    from types import ModuleType

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
GATE = ROOT / "scripts" / "perf_gate.py"


def _load_gate() -> ModuleType:
    """Import the gate by path; ``scripts/`` is not an importable package."""
    spec = importlib.util.spec_from_file_location("perf_gate", GATE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_gate()

BUDGET = 1_000_000
TOLERANCE = 0.10


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


def test_the_committed_core_checksum_matches_the_workload() -> None:
    # The recorded checksum is a constant of the workload's fixed corpus and
    # iteration count, so it belongs in the tree beside the budget. If the
    # workload is edited without re-recording, this fails before CI spends a
    # cachegrind run finding out.
    import json  # noqa: PLC0415

    budget = json.loads(
        (ROOT / "scripts" / "perf_budget.json").read_text(encoding="utf-8")
    )
    assert budget["core_workload_checksum"] == 134000


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


def test_every_mode_names_an_example_the_tree_builds() -> None:
    # The relative gate builds by mode name, so a mode naming an example that
    # does not exist fails at build time in the lane rather than here. Held to
    # the tree instead: every example named is a file under examples/.
    for mode, (example, subject) in gate.MODES.items():
        crate = "valgebra-py" if mode == "binding" else "valgebra-core"
        path = ROOT / "crates" / crate / "examples" / f"{example}.rs"
        assert path.exists(), f"{mode} names {example}, which is not in the tree"
        assert subject


def _merge_base_step() -> str:
    """Read the shell of the workflow's "Name the merge base" step from it."""
    spec = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    for job in spec["jobs"].values():
        for step in job.get("steps", []):
            if step.get("name") == "Name the merge base":
                return str(step["run"])
    message = "the workflow has no 'Name the merge base' step"
    raise AssertionError(message)


def _run_step(tree: Path, base_sha: str, branch: str) -> str:
    """Run the step and read the `sha=` it recorded.

    A fresh output file per run: a step appends to `GITHUB_OUTPUT`, as the
    runner's does, so a shared one would hold every earlier answer too.
    """
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
