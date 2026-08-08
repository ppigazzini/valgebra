# Cutting a release

How a version reaches an index: the surfaces a bump touches, the two dispatches
that publish, what to check between them, and the tag that records the result.
`.github/workflows/release.yml` owns the build and the upload; this page owns the
order and the checks that happen outside it.

## The version is declared once

`Cargo.toml`'s `[workspace.package] version` is the only declaration.
`pyproject.toml` is `dynamic = ["version"]`, so maturin reads the crate version
and the wheel cannot disagree with the workspace. `valgebra.__version__` reads the
installed distribution's metadata (`python/valgebra/__init__.py`), so it reports
the wheel a caller actually has rather than a literal in the tree.

Two lockfiles record the version and both must move with it: `Cargo.lock`, and
`fuzz/Cargo.lock` — the fuzz crate is a detached workspace, so a workspace-only
refresh leaves it naming the previous version.

## Publishing is a dispatch, and a tag publishes nothing

`release.yml` triggers on `workflow_dispatch` alone. Its `publish_target` input
selects the index (`none`, `testpypi`, `pypi`), and `none` builds the matrix and
the sdist without uploading, which is the dry run.

No workflow listens for a tag push. The tag records a release that already
happened; it is a marker, not a trigger.

Four conditions stand between a dispatch and an upload, and each is a step or a
job condition in `release.yml` rather than a convention:

- **The smoke must pass.** Each wheel is imported on its own platform and the
  sdist is compiled from source and imported before the publish job runs. A
  version cannot be replaced on an index once uploaded, only yanked, so a broken
  wheel has to fail before the upload rather than after it.
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
environment first (`pyproject.toml` owns that list), then, from the repository
root:

```bash
/tmp/vg/bin/python -m pytest -q
```

A test that needs a dependency the environment lacks skips rather than fails, so
read the skip list: a suite whose oracles are absent has checked less than the
same suite in a full development environment.

**Do not add PyPI as a second index while checking TestPyPI.** uv resolves a name
from the first index that carries it, so `--extra-index-url https://pypi.org/simple/`
makes it refuse the TestPyPI version in favour of the older released one — the
dependency-confusion guard, working as designed. valgebra has no runtime
dependencies, so the TestPyPI index alone resolves it; a package that needs PyPI
for its own dependencies passes `--index-strategy unsafe-best-match` instead.

The interpreter is part of what is being checked, not a detail of the check. The
extension module is built per interpreter version rather than against the stable
ABI, so a release ships many wheels and one install exercises exactly one of them
— `release.yml` owns the matrix. A version selector resolves to whichever build
is on the machine: `uv venv --python 3.14` can land on the free-threaded
interpreter, so read `sysconfig.get_config_var("Py_GIL_DISABLED")` in the venv to
record which wheel the check actually covered.

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
