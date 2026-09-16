"""A crate the local gate cannot compile is a crate that breaks in CI.

`cargo check --workspace` reaches the workspace members and nothing else. The
fuzz crate is deliberately a *detached* workspace -- it builds only under nightly
with libFuzzer -- so every stable gate a contributor runs locally skips it, and
the first thing that compiles it is a CI lane. That is exactly how a change to
the core's public types ships green and turns the fuzz lane red.

The universe is read from the files the repository tracks, because the direction
that matters is "a manifest arrived and nothing local builds it". Every tracked
`Cargo.toml` is therefore either a member of the root workspace, or carries a
**detached** entry naming the
command that builds it and the reason it is detached -- and that command must
appear in a workflow, or it is a build surface nothing drives.

Held in both directions: a manifest that is neither a member nor detached fails,
and a detached entry naming a manifest that is gone fails.

The second subject here is the other way a build surface goes wrong: not a crate
nothing compiles, but a build input nothing *notices*. uv keys a local package's
cached build on the patterns in `[tool.uv] cache-keys`, and an input outside them
is one a sync reinstalls the previous build over -- leaving a `.so` from one
build beside metadata from another. So every file the wheel is built from is
held to being covered by a pattern, and every pattern to matching something.

LEDGER: every manifest is a workspace member or a named detached surface
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

# The repository checks are not the product suite: this file reads the tree,
# the configuration and the gate scripts, none of which ship in a wheel.
pytestmark = pytest.mark.repository

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = ROOT / ".github" / "workflows"
CONTRIBUTING = ROOT / "CONTRIBUTING.md"
PYPROJECT = ROOT / "pyproject.toml"

# Manifests outside the root workspace, each with the command that builds it and
# why it is not a member. A detached surface is a hole in every local gate, so it
# carries an argument rather than a path alone.
DETACHED: dict[str, dict[str, str]] = {
    "fuzz/Cargo.toml": {
        "why": (
            "libFuzzer needs a nightly toolchain and the sanitizer flags; making "
            "it a workspace member would put nightly on the stable gates' path."
        ),
        # The command a contributor runs, and which a workflow must also run.
        "local": "cargo check --manifest-path fuzz/Cargo.toml",
        "lane": "cargo +${{ env.FUZZ_NIGHTLY }} fuzz build",
    },
}


def _manifests() -> set[str]:
    """Give every manifest in the tree that git does not ignore.

    A bare glob answers for anything sitting in the directory, and two things
    routinely do: a `git worktree` placed under `.claude/`, and a vendored
    checkout. Each carries a full copy of every manifest here, and the ledger
    then reports four that nothing builds -- true of the copies, and nothing
    about this tree.

    Ignored rather than tracked is the right line. Tracked would miss the
    direction that matters: a manifest arrives *untracked* first, and a ledger
    that waited for it to be committed would pass on the change that adds it.
    What an ignore says is that the path is not part of this repository at all.
    """
    found = {
        str(path.relative_to(ROOT)).replace("\\", "/")
        for path in ROOT.rglob("Cargo.toml")
        if "target" not in path.parts and ".venv" not in path.parts
    }
    if not found:
        return found
    ignored = subprocess.run(  # noqa: S603 - fixed argv, no shell, test-only
        ["git", "-C", str(ROOT), "check-ignore", "--stdin"],  # noqa: S607
        input="\n".join(sorted(found)),
        capture_output=True,
        text=True,
        check=False,
    )
    return found - {
        line.strip() for line in ignored.stdout.splitlines() if line.strip()
    }


def _workspace_members() -> set[str]:
    text = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    match = re.search(r"^members\s*=\s*\[(.*?)\]", text, re.DOTALL | re.MULTILINE)
    assert match is not None, "the root manifest declares no workspace members"
    return {f"{m}/Cargo.toml" for m in re.findall(r'"([^"]+)"', match.group(1))}


def test_every_manifest_is_a_member_or_detached_with_a_reason() -> None:
    manifests = _manifests()
    # The listing is the detector; an empty universe would pass having found
    # nothing to check.
    assert len(manifests) >= 3, f"the listing found only {sorted(manifests)}"

    accounted = _workspace_members() | set(DETACHED) | {"Cargo.toml"}
    orphans = sorted(manifests - accounted)
    assert not orphans, (
        f"manifests that are neither workspace members nor detached: {orphans}. "
        "Add each to the workspace, or record how it is built and why it is not."
    )


def test_no_detached_entry_is_stale() -> None:
    gone = sorted(set(DETACHED) - _manifests())
    assert not gone, f"detached entries naming no manifest: {gone}"


def test_every_detached_surface_carries_its_reason() -> None:
    for path, entry in DETACHED.items():
        assert len(entry["why"]) > 40, f"{path}: a detached entry with no reason"
        assert entry["local"].startswith("cargo "), path


def test_every_detached_surface_is_built_by_a_lane() -> None:
    # A detached crate the local gate skips and no workflow builds is a build
    # surface nothing drives at all -- worse than one that is merely local-only.
    workflows = "\n".join(
        p.read_text(encoding="utf-8") for p in WORKFLOWS.glob("*.yml")
    )
    for path, entry in DETACHED.items():
        assert entry["lane"] in workflows, f"{path}: no workflow runs {entry['lane']!r}"


def test_the_local_gate_names_every_detached_surface() -> None:
    # The point of the ledger: a contributor running the documented gate compiles
    # every crate the tree holds, so a public-API change cannot pass locally and
    # fail on a lane.
    gate = CONTRIBUTING.read_text(encoding="utf-8")
    for path, entry in DETACHED.items():
        assert entry["local"] in gate, (
            f"{path}: the contributor gate in CONTRIBUTING.md does not run "
            f"{entry['local']!r}, so a local run does not compile it"
        )


def _cache_key_patterns() -> list[str]:
    """Return the `file` globs `[tool.uv] cache-keys` declares.

    Read with a regex rather than parsed: `tomllib` is 3.11+ and this suite runs
    from 3.10, which is the floor the package claims.
    """
    text = PYPROJECT.read_text(encoding="utf-8")
    match = re.search(r"^cache-keys\s*=\s*\[(.*?)^\]", text, re.DOTALL | re.MULTILINE)
    assert match is not None, (
        "pyproject.toml declares no `[tool.uv] cache-keys`, so uv keys this "
        "project's cached build on pyproject.toml alone and notices no Rust change"
    )
    return re.findall(r'file\s*=\s*"([^"]+)"', match.group(1))


def _matches(pattern: str) -> set[str]:
    """Return the tree's files that `pattern` reaches, as repository paths.

    Expanded with `pathlib`, which is a *model* of uv's matcher and not the
    matcher itself. That is the right side to be wrong on: the two agree on the
    literal paths and the two glob shapes used here, and where they parted this
    would report a real input as uncovered rather than pass an uncovered one.
    """
    return {
        str(hit.relative_to(ROOT)).replace("\\", "/")
        for hit in ROOT.glob(pattern)
        if hit.is_file()
    }


def _build_inputs() -> set[str]:
    """Every file that decides what the built extension is.

    The version and the workspace shape (`Cargo.toml`), the resolved dependency
    set (`Cargo.lock`), each member's own manifest, and the sources themselves.
    The detached fuzz crate is deliberately absent: it is built by its own
    command and no wheel is built from it.
    """
    sources = {
        str(p.relative_to(ROOT)).replace("\\", "/")
        for p in (ROOT / "crates").rglob("*.rs")
        if "target" not in p.parts
    }
    manifests = {"Cargo.toml", "Cargo.lock", "pyproject.toml"} | _workspace_members()
    return manifests | sources


def test_every_build_input_is_covered_by_a_uv_cache_key() -> None:
    inputs = _build_inputs()
    # The glob is the detector; an empty universe would pass having found
    # nothing to check.
    assert len(inputs) >= 10, f"the build-input glob found only {sorted(inputs)}"

    covered = set().union(*(_matches(p) for p in _cache_key_patterns()))
    uncovered = sorted(inputs - covered)
    assert not uncovered, (
        f"build inputs no `[tool.uv] cache-keys` pattern reaches: {uncovered}. "
        "uv will reinstall the previous build over a newer one when these "
        "change, so the extension and its metadata come from different builds."
    )


def test_pyproject_is_named_among_the_cache_keys() -> None:
    # Naming any key replaces uv's default rather than adding to it, so the
    # default has to be written back out or a dependency-group edit stops
    # invalidating the build.
    assert "pyproject.toml" in _cache_key_patterns(), (
        "`pyproject.toml` is uv's default cache key and declaring any key "
        "replaces the default, so it has to be listed explicitly"
    )


def test_no_cache_key_pattern_is_dead() -> None:
    # A pattern matching nothing is a typo that reads as coverage.
    for pattern in _cache_key_patterns():
        assert _matches(pattern), (
            f"the cache-key pattern {pattern!r} matches no file in the tree"
        )
