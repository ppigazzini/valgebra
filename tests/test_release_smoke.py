"""Every wheel the release builds on a runner is smoked on that runner.

`release.yml` builds wheels in one matrix and smokes them in another, joined by
an artifact name each side spells for itself. A build row whose artifact no
smoke row downloads ships a wheel nothing ran; that is how the PyPy wheels
shipped for two releases as profile-guided builds that died at the walk's depth
bound while the push lane stayed green on a plain wheel it built itself. A
smoke row naming an artifact no build row uploads fails the release instead,
late and for the wrong reason.

Held here, both ways: every build row on a runner that can run its own output
has a smoke row for its artifact, and every smoke row names an artifact a build
row uploads. The musllinux wheels are the one accepted gap, with the reason the
workflow gives beside them. And the PyPy wheel is built in a row of its own,
without the profile: the rows that carry the profile name their interpreters,
and none of them names PyPy.

LEDGER: every wheel the release builds on a runner is smoked there; PyPy's is plain
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.repository

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = yaml.safe_load(
    (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
)
BUILDS: list[dict[str, object]] = WORKFLOW["jobs"]["wheels"]["strategy"]["matrix"][
    "platform"
]
SMOKES: list[dict[str, object]] = WORKFLOW["jobs"]["smoke"]["strategy"]["matrix"][
    "wheel"
]
#: The wheel sets the smoke does not download, with the workflow's own reason.
NOT_SMOKED = {
    "wheels-ubuntu-latest-x86_64-musllinux_1_2": "runs only in a musl container",
    "wheels-ubuntu-latest-aarch64-musllinux_1_2": "runs only in a musl container",
}


def _artifact(build: dict[str, object]) -> str:
    """Spell the artifact name the upload step composes for a build row."""
    flavour = build.get("variant") or build.get("manylinux") or "native"
    return f"wheels-{build['runner']}-{build['target']}-{flavour}"


def test_the_matrices_were_read() -> None:
    """The parse is a detector, so it must be shown to have read both matrices."""
    assert len(BUILDS) >= 8, BUILDS
    assert len(SMOKES) >= 6, SMOKES


def test_every_built_wheel_set_is_smoked_or_excused() -> None:
    smoked = {str(row["artifact"]) for row in SMOKES}
    built = {_artifact(row) for row in BUILDS}
    unsmoked = sorted(built - smoked - set(NOT_SMOKED))
    assert not unsmoked, (
        f"wheel sets the release builds and never runs: {unsmoked}. Add a smoke "
        "row for each, or name it in NOT_SMOKED with the reason."
    )
    stale = sorted(name for name in NOT_SMOKED if name not in built)
    assert not stale, f"excused wheel sets no build row uploads: {stale}"


def test_every_smoke_row_names_a_wheel_set_the_release_builds() -> None:
    built = {_artifact(row) for row in BUILDS}
    phantom = sorted(
        str(row["artifact"]) for row in SMOKES if row["artifact"] not in built
    )
    assert not phantom, f"smoke rows naming an artifact no build row uploads: {phantom}"


def test_pypy_is_built_plain_and_smoked_with_the_suite() -> None:
    """A profiled extension dies on PyPy at the depth bound; the plain one does not."""
    profiled = [row for row in BUILDS if row.get("pgo")]
    assert profiled, "no build row carries the profile"
    for row in profiled:
        interpreters = str(row.get("interpreter", row.get("pythons", "")))
        assert "pypy" not in interpreters.lower(), f"a profiled row builds PyPy: {row}"
    plain_pypy = [
        row
        for row in BUILDS
        if not row.get("pgo") and "pypy" in str(row.get("interpreter", ""))
    ]
    assert plain_pypy, "no plain build row names PyPy"
    for row in plain_pypy:
        smoke = [smoke for smoke in SMOKES if smoke["artifact"] == _artifact(row)]
        assert smoke, f"the PyPy wheel {_artifact(row)} has no smoke row"
        assert all(item.get("suite") for item in smoke), (
            f"the PyPy wheel {_artifact(row)} is not smoked with the product suite"
        )
