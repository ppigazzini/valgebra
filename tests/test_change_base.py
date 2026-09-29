"""A change is measured from where its commits start, a force-push included.

`scripts/change_base.py` names the base the bench gate builds and the
diff-scoped mutation sweeps diff against. The case it exists for is the
maintainer's own way of working: amend, force-push. The event's `before` is
then a tip no branch reaches, the fallback to the default branch is the commit
under test, and a sweep measuring an amended commit from itself reads `core
files: none` and passes.

Driven against synthetic histories rather than this tree, so the answer does
not depend on the clone the suite runs in: an upstream that is rewritten, and a
fresh checkout of it, as a runner takes one.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

# The repository checks are not the product suite: this file reads the workflow
# and drives a CI script, none of which ship in a wheel.
pytestmark = pytest.mark.repository

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "change_base.py"
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"

#: A clone inherits no committer, and a commit here must not depend on whether
#: the machine running the suite has one configured.
IDENTITY = {
    "GIT_AUTHOR_NAME": "base",
    "GIT_AUTHOR_EMAIL": "base@example.invalid",
    "GIT_COMMITTER_NAME": "base",
    "GIT_COMMITTER_EMAIL": "base@example.invalid",
}


def _git(tree: Path, *args: str) -> str:
    return subprocess.run(  # noqa: S603 - fixed argv, no shell, test-only
        ["git", "-C", str(tree), *args],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, **IDENTITY},
    ).stdout.strip()


def _commit(tree: Path, message: str) -> str:
    _git(tree, "commit", "--quiet", "--allow-empty", "-m", message)
    return _git(tree, "rev-parse", "HEAD")


def _base(checkout: Path, base_sha: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed argv, no shell, test-only
        [sys.executable, str(SCRIPT)],
        cwd=checkout,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "BASE_SHA": base_sha, "DEFAULT_BRANCH": "main"},
    )


def _checkout(tmp_path: Path, upstream: Path, name: str) -> Path:
    """Clone `upstream` the way a runner does after the push."""
    _git(tmp_path, "clone", "--quiet", "--no-local", str(upstream), name)
    return tmp_path / name


@pytest.fixture
def rewritten(tmp_path: Path) -> tuple[Path, str, str, str]:
    """Build an upstream whose last two commits were amended and force-pushed.

    `root - b - c` is rewritten to `root - b' - c'`. Returns the upstream, the
    fork point both histories share, the tip the push replaced (`before`) and
    the new tip.
    """
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    _git(upstream, "init", "--quiet", "--initial-branch=main")
    root = _commit(upstream, "root")
    _commit(upstream, "b")
    before = _commit(upstream, "c")
    _git(upstream, "reset", "--quiet", "--hard", root)
    _commit(upstream, "b, amended")
    after = _commit(upstream, "c, amended")
    return upstream, root, before, after


def test_a_rewrite_is_measured_from_its_fork_point(
    rewritten: tuple[Path, str, str, str], tmp_path: Path
) -> None:
    # A host that still serves the tip a push replaced: both rewritten commits
    # are measured, not the difference the amend made to them.
    upstream, root, before, _ = rewritten
    checkout = _checkout(tmp_path, upstream, "served")
    done = _base(checkout, before)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == root


def test_a_rewrite_the_host_forgot_is_measured_from_the_last_commit(
    rewritten: tuple[Path, str, str, str], tmp_path: Path
) -> None:
    # The default branch's tip is then the commit under test; its parent stands
    # in, so the last commit is measured rather than nothing. Collected here,
    # since a host that keeps a replaced tip serves it by id.
    upstream, _, before, after = rewritten
    _git(upstream, "reflog", "expire", "--expire=now", "--all")
    _git(upstream, "gc", "--quiet", "--prune=now")
    checkout = _checkout(tmp_path, upstream, "forgotten")
    done = _base(checkout, before)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == _git(checkout, "rev-parse", f"{after}~1")


def test_a_plain_push_is_measured_from_the_tip_it_extends(tmp_path: Path) -> None:
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    _git(upstream, "init", "--quiet", "--initial-branch=main")
    _commit(upstream, "root")
    before = _commit(upstream, "pushed earlier")
    _commit(upstream, "pushed now")
    _commit(upstream, "pushed now, too")
    checkout = _checkout(tmp_path, upstream, "pushed")
    assert _base(checkout, before).stdout.strip() == before


def test_a_branch_is_measured_from_where_it_left_the_default(tmp_path: Path) -> None:
    # A pull request's base is the default branch's tip, which may have moved on
    # since the branch left it; the branch's commits start at the fork point.
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    _git(upstream, "init", "--quiet", "--initial-branch=main")
    fork = _commit(upstream, "root")
    _git(upstream, "checkout", "--quiet", "-b", "work")
    _commit(upstream, "the change")
    _git(upstream, "checkout", "--quiet", "main")
    moved = _commit(upstream, "main moves on")
    checkout = _checkout(tmp_path, upstream, "branch")
    _git(checkout, "checkout", "--quiet", "--detach", "origin/work")
    assert _base(checkout, moved).stdout.strip() == fork
    # A first push of a branch names no base at all: the default branch stands in.
    assert _base(checkout, "").stdout.strip() == fork


def test_no_base_at_all_cannot_run(tmp_path: Path) -> None:
    tree = tmp_path / "alone"
    tree.mkdir()
    _git(tree, "init", "--quiet", "--initial-branch=main")
    _commit(tree, "root")
    done = _base(tree, "0" * 40)
    assert done.returncode == 2, done.stdout + done.stderr
    assert not done.stdout


def test_every_step_that_reads_the_event_base_names_it_through_the_script() -> None:
    """One reading of the base, so the gates cannot disagree about a push.

    The bench step guarded against measuring `HEAD` from itself and the sweeps
    did not, which is how an amended commit swept nothing on the lane that
    reads it most carefully. A step carrying `BASE_SHA` works the base out by
    calling the script, not by a shell of its own.
    """
    spec = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    readers = [
        (job, step.get("name", ""), step.get("run", ""))
        for job, body in spec["jobs"].items()
        for step in body.get("steps", [])
        if "BASE_SHA" in (step.get("env") or {})
    ]
    assert len(readers) >= 3, f"the workflow reads the event base only in {readers}"
    inline = [
        f"{job}: {name}"
        for job, name, run in readers
        if "scripts/change_base.py" not in run or "$BASE_SHA" in run
    ]
    assert not inline, f"steps working the base out for themselves: {inline}"
