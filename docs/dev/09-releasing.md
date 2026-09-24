# Cutting a release

How a version reaches an index: the surfaces a bump touches, the two dispatches
that publish, what to check between them, and the tag that records the result.
`.github/workflows/release.yml` owns the build and the upload; this page owns the
order and the checks that happen outside it.

## The version is declared once

`Cargo.toml`'s `[workspace.package] version` is the only declaration.
`pyproject.toml` is `dynamic = ["version"]`, so maturin reads the crate version
and the wheel cannot disagree with the workspace. `valgebra.__version__` is that
same crate version, compiled into the extension: reading it back out of the
installed metadata answered the same question and cost two thirds of
`import valgebra`, so the number travels in the `.so` instead.

That is what makes `tests/test_version.py` worth having. `__version__` and the
distribution metadata are two *different* readings of one manifest, so holding
them equal catches an install whose halves came from different builds --
which is a real state, because uv and maturin both write into the same venv.
`[tool.uv] cache-keys` in `pyproject.toml` is the other half of that: it names
the manifests and sources a rebuild depends on, so a sync after a bump rebuilds
rather than reinstalling the build before it.

Two lockfiles record the version and both must move with it: `Cargo.lock`, and
`fuzz/Cargo.lock` — the fuzz crate is a detached workspace, so a workspace-only
refresh leaves it naming the previous version. `uv.lock` records the project as
an editable source with no version of its own, so it does not.

## Publishing is a dispatch, and a tag publishes nothing

`release.yml` triggers on `workflow_dispatch` alone. Its `publish_target` input
selects the index (`none`, `testpypi`, `pypi`), and `none` builds the matrix and
the sdist without uploading, which is the dry run.

No workflow listens for a tag push. The tag records a release that already
happened; it is a marker, not a trigger.

Four conditions stand between a dispatch and an upload, and each is a step or a
job condition in `release.yml` rather than a convention:

- **The smoke must pass.** Each wheel set is imported on its own platform, on
  the floor, the newest release and the free-threaded builds where the set
  carries a wheel for them, with the free-threaded import required to leave the
  GIL off; the sdist is compiled from source and imported before the publish job
  runs. A version cannot be replaced on an index once uploaded, only yanked, so
  a broken wheel has to fail before the upload rather than after it.
- **`confirm_version` must equal the version in the built wheels**, and an empty
  input aborts. A dispatch cannot publish a version the run did not build.
- **The ref must be `main`.** A dispatch from a topic branch uploads nothing, so
  the bump commit has to land before step 2 — that is why step 1 says to land it.
- **The version must be absent from the target index.** The check fails closed: a
  200 is a stale re-publish and anything other than a definitive 404 leaves the
  question unanswered and also aborts.

The publish job also runs in a deployment environment named after the index, so a
release waits for whatever reviewers those environments require.

## The order

1. **Bump, in one commit.** The workspace version, both lockfiles, and the
   changelog: roll the `Unreleased` entries into a dated section for the version
   and add its compare and tag links. Land it on `main`, so the merge gates run
   against the tree that is about to be published.

   The changelog ledger (`tests/test_changelog_ledger.py`) reads the *page* for
   which version is released and measures the roll against that version's tag.
   Between this step and step 6 the section names a version no tag resolves yet,
   so the ledger skips: there is nothing left to account for, and the roll is
   empty because it was just emptied. It resumes the moment the tag lands — and
   nothing holds a `feat`/`fix` landed in the window to a roll line until it
   does, so the tag is the step to take promptly rather than last.

   ```bash
   cargo metadata --format-version 1 --offline >/dev/null    # refresh Cargo.lock
   cargo metadata --format-version 1 --offline --manifest-path fuzz/Cargo.toml >/dev/null
   ```

2. **Dispatch to TestPyPI** — `publish_target: testpypi`, `confirm_version` the
   new version.
3. **Check what the index serves** (below).
4. **Dispatch to PyPI** — `publish_target: pypi`, the same version.
5. **Check what the index serves** again, against PyPI.
6. **Tag the published commit**, annotated, subject `valgebra X.Y.Z`:

   ```bash
   git tag -a vX.Y.Z -m "valgebra X.Y.Z" <commit>
   git push origin vX.Y.Z
   ```

## Checking an index (steps 3 and 5)

The workflow's smoke jobs prove each **artifact** imports. They cannot prove the
**index** serves it: resolution, the wheel a real interpreter selects, and the
metadata a caller reads are all downstream of the upload. That is what these steps
check, and they are the only steps that do.

```bash
uv venv /tmp/vg
VIRTUAL_ENV=/tmp/vg uv pip install --index-url https://test.pypi.org/simple/ "valgebra==X.Y.Z"
/tmp/vg/bin/python -c "import valgebra as v; print(v.__version__); \
    assert v.Validator(int).is_valid(1) and not v.Validator(int).is_valid('x'); \
    assert v.Validator(list[int]).is_valid([1, 2]) and v.Validator(int).is_valid_json('1')"
```

Then run the suite against the installed wheel rather than a local build. Nothing
puts `python/` on the path, so the tests import whichever `valgebra` the
environment holds — install the dev group's test dependencies into the same
environment first, **from PyPI and in their own install**:

```bash
VIRTUAL_ENV=/tmp/vg uv pip install --group dev   # from the repository root
/tmp/vg/bin/python -m pytest -q
```

**The separate install is the point.** The test dependencies are not the package
under test, and resolving them against TestPyPI serves whatever anyone last
uploaded there — an ancient `syrupy` beside a `pytest` too old to start it. Only
valgebra comes from the index being checked; everything else comes from PyPI,
where it comes from in every other environment. The group is read from `pyproject.toml` rather than
listed here, so it cannot drift from the one the lanes install.

A test that needs a dependency the environment lacks skips rather than fails, so
read the skip list: a suite whose oracles are absent has checked less than the
same suite in a full development environment.

**Do not add PyPI as a second index while checking TestPyPI.** uv resolves a name
from the first index that carries it, so `--extra-index-url https://pypi.org/simple/`
makes it refuse the TestPyPI version in favour of the older released one — the
dependency-confusion guard, working as designed. valgebra has no runtime
dependencies, so the TestPyPI index alone resolves it; a package that needs PyPI
for its own dependencies passes `--index-strategy unsafe-best-match` instead.

**`--refresh-package valgebra` is not optional on a same-day second release.** uv
caches an index's responses, and a cached listing from before the upload resolves
`==X.Y.Z` to *no such version* — the same message an index that never received the
upload gives. Confirm against the index itself before believing it:

```bash
curl -sS https://test.pypi.org/simple/valgebra/ \
  -H "Accept: application/vnd.pypi.simple.v1+json" | python3 -c \
  "import json,sys; print(json.load(sys.stdin)['versions'])"
```

The per-version JSON endpoint (`/pypi/valgebra/X.Y.Z/json`) also answers with the
file list before the aggregate `/pypi/valgebra/json` stops naming the previous
release as the latest, so disagreement between those two is propagation and not a
failure.

**The simple index lags the JSON API, and the simple index is what a resolver
reads.** 0.0.10 was answered in full by `/pypi/valgebra/0.0.10/json` — fifty
files — while `uv pip install` still reported no such version, because the
simple listing had not caught up. That is the same message a failed upload
gives, so read the simple index itself (the `curl` above) before believing
either, and if it is the one that is behind, wait and retry rather than
re-dispatching the publish.

The interpreter is part of what is being checked, not a detail of the check. The
extension module is built per interpreter version rather than against the stable
ABI, so a release ships many wheels and one install exercises exactly one of them
— `release.yml` owns the matrix. On macOS and Windows the builds run on the
host, where `--find-interpreter` sees only the interpreters installed on the
image, so the workflow installs every supported one first; Windows builds its
free-threaded wheels in a job of their own, since a release and its
free-threaded build can fail to co-install in one step. The maturin a release builds with is the one `uv.lock`
resolves, pinned in the workflow rather than taken as the newest. A version
selector resolves to whichever build is on the machine: `uv venv --python 3.14`
can land on the free-threaded interpreter, so read
`sysconfig.get_config_var("Py_GIL_DISABLED")` in the venv to record which wheel
the check actually covered.

## What this does not cover

- **A platform outside the smoke matrix.** The musllinux wheels are built and not
  imported by CI — running them needs a musl interpreter, a lane that does not
  exist — so the first musl install is a user's.
- **A source install.** `uv pip install` takes the wheel; the sdist path is
  compiled once by the workflow, on Linux, and `--no-binary valgebra` locally is
  the only way to reach it on another platform.
- **A published version that is wrong.** It cannot be replaced, only yanked, and
  the workflow refuses a version the index already serves. The remedy is the next
  patch version, which is why step 3 exists before step 4.
