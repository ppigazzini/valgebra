"""Every wheel the release builds on a runner is smoked on that runner.

`release.yml` builds wheels in one matrix and smokes them in another, joined by
an artifact name each side spells for itself. A build row whose artifact no
smoke row downloads ships a wheel nothing ran, and a defect of that wheel alone
-- a profile-guided build dying at the walk's depth bound -- ships while the
push lane stays green on a plain wheel it builds itself. A smoke row naming an
artifact no build row uploads fails the release instead, late and for the wrong
reason.

Held here, both ways: every build row has a smoke row for its artifact, and
every smoke row names an artifact a build row uploads -- the musllinux sets in
Alpine containers, the only place a musl wheel loads. Within a set, every
release a build row names is loaded by its smoke row: the product suite on the
ends, an import, one check and the bounds file on the releases between, which
holds each profile-guided wheel's deepest walk to a 1 MiB thread. A set whose
build row names no interpreter, the musllinux image building every one it
carries, is read off its artifact by the smoke itself, which fails on a wheel no
container loaded unless `UNLOADABLE` names its tag. Each wheel is a build of
its own, profiled on its own interpreter, so a wheel no smoke row loads is a
wheel nothing ran. A smoke is the product suite rather than an import, for the
wheels and for the build from source, and the import it starts with fails on a
warning. And the PyPy wheel is built in a row of its own, without the profile:
the rows that carry the profile name their interpreters, and none of them names
PyPy.

LEDGER: every wheel the release builds is loaded, its ends tested; PyPy's is plain
"""

from __future__ import annotations

import os
import re
import subprocess
from typing import TYPE_CHECKING

import pytest

from _tree import WORKFLOWS, read_workflow

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.repository

WORKFLOW = read_workflow(WORKFLOWS / "release.yml")
BUILDS: list[dict[str, object]] = WORKFLOW["jobs"]["wheels"]["strategy"]["matrix"][
    "platform"
]
#: The jobs that load a wheel set, each a matrix of rows naming one.
SMOKE_JOBS = ("smoke", "smoke-musl")
SMOKES: list[dict[str, object]] = [
    row
    for job in SMOKE_JOBS
    for row in WORKFLOW["jobs"][job]["strategy"]["matrix"]["wheel"]
]
#: The wheel sets no smoke row downloads, with the reason. None: the musllinux
#: sets, the last ones here, are loaded in Alpine containers.
NOT_SMOKED: dict[str, str] = {}

#: The ABI tags a musllinux set holds and no container loads, with the reason.
#: The set's build row names no interpreter -- its image builds every one it
#: carries -- so the smoke reads the set off its artifact, and a wheel whose tag
#: is neither loaded nor here fails it.
UNLOADABLE = {
    "cp314t": "no Alpine image carries a free-threaded CPython",
    "pypy311_pp73": "no Alpine image carries a PyPy",
}


def _artifact(build: dict[str, object]) -> str:
    """Spell the artifact name the upload step composes for a build row."""
    flavor = build.get("variant") or build.get("manylinux") or "native"
    return f"wheels-{build['runner']}-{build['target']}-{flavor}"


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


def test_pypy_is_built_plain() -> None:
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
    unsmoked = [
        _artifact(row)
        for row in plain_pypy
        if not any(smoke["artifact"] == _artifact(row) for smoke in SMOKES)
    ]
    assert not unsmoked, f"PyPy wheel sets with no smoke row: {unsmoked}"


def _releases(spelled: str) -> set[str]:
    """Give the CPython releases a row spells, `3.14t` for a free-threaded one."""
    return set(re.findall(r"(?<![\d.@])3\.\d+t?\b", spelled))


def test_every_release_a_build_row_names_is_loaded() -> None:
    """A set's smoke row loads every release the set's build row names.

    Read for the CPython rows that name their interpreters. A PyPy set holds
    one wheel, which the set's own row loads; a row that builds what its image
    carries names none, and its smoke reads the set off the artifact instead,
    which the next two tests hold.
    """
    unloaded: dict[str, list[str]] = {}
    for build in BUILDS:
        if str(build.get("variant", "")).startswith("pypy"):
            continue
        named = _releases(f"{build.get('interpreter', '')} {build.get('pythons', '')}")
        if not named:
            continue
        loaded = set().union(
            *(
                _releases(f"{row.get('pythons', '')} {row.get('imports', '')}")
                for row in SMOKES
                if row["artifact"] == _artifact(build)
            )
        )
        if missing := sorted(named - loaded):
            unloaded[_artifact(build)] = missing
    assert not unloaded, (
        f"releases built and never loaded: {unloaded}. Name each in its smoke "
        "row's `imports`, or in `pythons` for the suite."
    )


def _steps(job: str) -> list[dict[str, str]]:
    return WORKFLOW["jobs"][job]["steps"]


@pytest.mark.parametrize("job", ["smoke", "smoke-musl", "sdist-smoke"])
def test_every_smoke_runs_the_product_suite(job: str) -> None:
    """An import is not a smoke: a crash at the depth bound shows only in the suite.

    A profiled build dying at the walk's depth bound passes an import, which
    reaches no depth. Every row of the job runs the suite, so the step carries no
    condition a row could leave unset.
    """
    runs = [
        step
        for step in _steps(job)
        if '-m "not repository"' in str(step.get("run", ""))
    ]
    assert runs, f"{job} runs no product suite"
    conditional = [str(step.get("name")) for step in runs if "if" in step]
    assert not conditional, (
        f"{job} runs the suite only where a row says so: {conditional}"
    )


@pytest.mark.parametrize("job", ["smoke", "smoke-musl", "sdist-smoke"])
def test_every_smoke_import_fails_on_a_warning(job: str) -> None:
    imports = [
        str(step["run"])
        for step in _steps(job)
        if "import" in str(step.get("run", "")) and "valgebra as v" in str(step["run"])
    ]
    assert imports, f"{job} imports nothing"
    silent = [run for run in imports if "-W error" not in run]
    assert not silent, f"{job} imports with warnings left as warnings"


#: The file that runs the deepest walks and relations on a 1 MiB thread.
_BOUNDS_FILE = "test_adversarial_bounds.py"

#: A loop over the releases between, up to the `done` that closes it.
_BETWEEN_LOOP = re.compile(r"for py in \$IMPORTS; do\n(.*?)\n\s*done", re.DOTALL)


@pytest.mark.parametrize("job", SMOKE_JOBS)
def test_every_release_between_runs_the_bounds_file(job: str) -> None:
    """A wheel the suite skips still runs the deepest walks on a 1 MiB thread.

    The limits page tells a caller 1 MiB is enough on every wheel the release
    loads, and that figure is a reading of each wheel: each is profiled and
    laid out on its own interpreter, so the 3.12 wheel's stack says nothing of
    the 3.13 one's. The releases between the ends are imported and nothing
    more unless the step that runs the suite runs this file on them too, over
    every release the row names and under no condition.
    """
    steps: list[dict] = [
        step for step in _steps(job) if _BOUNDS_FILE in str(step.get("run", ""))
    ]
    assert len(steps) == 1, f"{job} has {len(steps)} steps running {_BOUNDS_FILE}"
    step = steps[0]
    assert "if" not in step, f"{job} runs {_BOUNDS_FILE} only where a row says so"
    assert step.get("env", {}).get("IMPORTS") == "${{ matrix.wheel.imports }}", (
        f"{job} loops over something other than the row's releases between"
    )
    loops = _BETWEEN_LOOP.findall(str(step["run"]))
    assert len(loops) == 1, f"{job} has {len(loops)} loops over $IMPORTS"
    assert _BOUNDS_FILE in loops[0], (
        f"{job} loops over the releases between without running {_BOUNDS_FILE}"
    )


#: The step a set built from what its image carries is read off its artifact by.
_ARTIFACT_STEP = "Every wheel the set holds is loaded or named unloadable"


def _artifact_step(job: str) -> dict:
    found = [step for step in _steps(job) if step.get("name") == _ARTIFACT_STEP]
    assert len(found) == 1, f"{job} has {len(found)} steps named {_ARTIFACT_STEP!r}"
    return found[0]


def test_a_set_its_image_chooses_is_read_off_its_artifact() -> None:
    """Every smoke of a set whose build names no interpreter checks the artifact.

    The image builds every interpreter it carries, so the set is whatever the
    image carries on the day, and an image that gains a release ships a wheel
    no container loads unless the smoke reads the set off the artifact. The step
    names the tags it accepts unloaded, and they are the ones `UNLOADABLE`
    gives reasons for.
    """
    chosen = [
        _artifact(build)
        for build in BUILDS
        if not _releases(f"{build.get('interpreter', '')} {build.get('pythons', '')}")
        and not str(build.get("variant", "")).startswith("pypy")
    ]
    assert chosen, "no build row builds what its image carries"
    jobs = {
        job
        for job in SMOKE_JOBS
        for row in WORKFLOW["jobs"][job]["strategy"]["matrix"]["wheel"]
        if row["artifact"] in chosen
    }
    assert jobs == {"smoke-musl"}, jobs
    named = str(_artifact_step("smoke-musl")["env"]["UNLOADABLE"]).split()
    assert sorted(named) == sorted(UNLOADABLE), (
        f"the musl smoke accepts {sorted(named)} unloaded; the reasons here cover "
        f"{sorted(UNLOADABLE)}"
    )


def _wheel(tag: str) -> str:
    python = tag.removesuffix("t") if tag.startswith("cp") else "pp311"
    return f"valgebra-0.0.18-{python}-{tag}-musllinux_1_2_x86_64.whl"


@pytest.mark.parametrize(
    ("tags", "ends"),
    [
        (["cp310", "cp311", "cp312", "cp313", "cp314", "cp314t", "pypy311_pp73"], 0),
        (["cp310", "cp314", "cp315"], 1),
        ([], 1),
    ],
    ids=["the 0.0.18 set", "a release the image gained", "an empty artifact"],
)
def test_the_artifact_step_refuses_a_wheel_no_container_loads(
    tmp_path: Path, tags: list[str], ends: int
) -> None:
    """The step's own script, over a set of empty files named as wheels are."""
    (tmp_path / "dist").mkdir()
    for tag in tags:
        (tmp_path / "dist" / _wheel(tag)).touch()
    step = _artifact_step("smoke-musl")
    row = WORKFLOW["jobs"]["smoke-musl"]["strategy"]["matrix"]["wheel"][0]
    environment = {
        **os.environ,
        "LOADED": f"{row['pythons']} {row['imports']}",
        "UNLOADABLE": str(step["env"]["UNLOADABLE"]),
    }
    ended = subprocess.run(  # noqa: S603 -- the workflow's own script, test-only
        ["bash", "-e", "-c", step["run"]],  # noqa: S607 -- the shell GitHub runs
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert ended.returncode == ends, ended.stdout + ended.stderr
