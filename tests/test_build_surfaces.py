"""A crate the local gate cannot compile is a crate that breaks in CI.

`cargo check --workspace` reaches the workspace members and nothing else. The
fuzz crate is deliberately a *detached* workspace -- it builds only under nightly
with libFuzzer -- so every stable gate a contributor runs locally skips it, and
the first thing that compiles it is a CI lane. That is exactly how a change to
the core's public types ships green and turns the fuzz lane red.

The universe is globbed from the tree, because the direction that matters is "a
manifest arrived and nothing local builds it". Every `Cargo.toml` is therefore
either a member of the root workspace, or carries a **detached** entry naming the
command that builds it and the reason it is detached -- and that command must
appear in a workflow, or it is a build surface nothing drives.

Held in both directions: a manifest that is neither a member nor detached fails,
and a detached entry naming a manifest that is gone fails.

LEDGER: every manifest is a workspace member or a named detached surface
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = ROOT / ".github" / "workflows"
CONTRIBUTING = ROOT / "CONTRIBUTING.md"

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
    return {
        str(p.relative_to(ROOT)).replace("\\", "/")
        for p in ROOT.rglob("Cargo.toml")
        if "target" not in p.parts and ".venv" not in p.parts
    }


def _workspace_members() -> set[str]:
    text = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    match = re.search(r"^members\s*=\s*\[(.*?)\]", text, re.DOTALL | re.MULTILINE)
    assert match is not None, "the root manifest declares no workspace members"
    return {f"{m}/Cargo.toml" for m in re.findall(r'"([^"]+)"', match.group(1))}


def test_every_manifest_is_a_member_or_detached_with_a_reason() -> None:
    manifests = _manifests()
    # The glob is the detector; an empty universe would pass having found
    # nothing to check.
    assert len(manifests) >= 3, f"the manifest glob found only {sorted(manifests)}"

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
