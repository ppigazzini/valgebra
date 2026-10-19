"""A commit's hooks lint the way the lanes lint.

`.pre-commit-config.yaml` runs clippy on every commit that touches Rust, so a
lint the lane would refuse is caught before a push. That holds only while the
hook and the lane run one command. A clippy hook without `--locked`, which the
lane passes, lets cargo rewrite a lock that does not match the manifests where
the lane refuses it, so the hook and the lane read two locks.

So every clippy hook's command is one a lint step runs, flag for flag.
"""

from __future__ import annotations

import pytest

from _tree import ROOT, read_yaml, runs

# A repository check: it reads the hook configuration and the workflow, neither
# of which ships in a wheel.
pytestmark = pytest.mark.repository

HOOKS = ROOT / ".pre-commit-config.yaml"


def _words(command: str) -> str:
    """Read a command as its words, with the shell's line continuations joined."""
    return " ".join(command.replace("\\\n", " ").split())


def _clippy_hooks() -> list[str]:
    """Give the command of every hook that runs clippy."""
    return [
        _words(str(hook["entry"]))
        for repo in read_yaml(HOOKS)["repos"]
        for hook in repo.get("hooks", [])
        if str(hook.get("entry", "")).startswith("cargo clippy")
    ]


def _unmatched(hooks: list[str], steps: list[str]) -> list[str]:
    """Name the hooks no step runs, flag for flag."""
    read = [_words(step) for step in steps]
    return [hook for hook in hooks if not any(hook in step for step in read)]


def test_every_clippy_hook_is_a_lint_step() -> None:
    hooks = _clippy_hooks()
    # The scan is the detector: no hook at all would pass having read nothing.
    assert len(hooks) >= 2, f"the hook scan found {hooks}"
    unmatched = _unmatched(hooks, runs())
    assert not unmatched, (
        f"clippy hooks no lane runs as written: {unmatched}. Run the lint "
        "step's command, `--locked` and all."
    )


def test_a_hook_is_matched_flag_for_flag() -> None:
    # The readings the check turns on: a continued line is one command, and a
    # hook dropping a flag the step passes is not the step's command.
    step = "cargo clippy --locked --lib -- \\\n  -D clippy::unwrap_used\n"
    assert (
        _unmatched(["cargo clippy --locked --lib -- -D clippy::unwrap_used"], [step])
        == []
    )
    assert _unmatched(["cargo clippy --lib -- -D clippy::unwrap_used"], [step])
