# Cutting a release

How a version reaches an index: what is read before the bump, the wheels a
release builds, the surfaces a bump touches, the two dispatches that publish,
what to check between them, and the tag that records the result.
`.github/workflows/release.yml` owns the build and the upload; this page owns the
order and the checks that happen outside it.

## The version is declared once

`Cargo.toml`'s `[workspace.package] version` is the only declaration.
`pyproject.toml` is `dynamic = ["version"]`, so maturin reads the crate version
and the wheel cannot disagree with the workspace. `valgebra.__version__` is that
same crate version, compiled into the extension: reading it from the installed
metadata answers the same question and costs two thirds of `import valgebra`,
so the number travels in the `.so` instead.

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
the sdist without uploading, which is the dry run. A third input,
`attach_github_release`, attaches the built wheels and sdist to a GitHub
pre-release tagged `build-<version>-<run>` for off-index installs, once the
smoke has passed; it uploads to no index, and neither the ref nor
`confirm_version` gates it.

No workflow listens for a tag push. The tag records a release that already
happened; it is a marker, not a trigger.

Five conditions stand between a dispatch and an upload, and each is a step or a
job condition in `release.yml` rather than a convention:

- **The smoke must pass.** Each wheel set is loaded on its own platform -- the
  musllinux sets in Alpine containers -- with every warning an error and the
  free-threaded import required to leave the GIL off. The product suite runs on
  the floor, the newest release and the free-threaded builds where the set
  carries a wheel for them, and every release between is installed, imported
  and checked, since each wheel is a build of its own; the sdist is compiled from source, imported and put through the
  same suite before the publish job runs. A version cannot be replaced on an
  index once uploaded, only yanked, so a broken wheel has to fail before the
  upload rather than after it.
- **`confirm_version` must equal the version in the built wheels**, and an empty
  input aborts. A dispatch cannot publish a version the run did not build.
- **The ref must be `main`.** The bump commit has to land before step 3 — that
  is why step 2 says to land it.
- **`ci.yml` must have passed on the commit.** The publish job's first step asks
  the API for a push run of `ci.yml` on the commit that concluded in success, and
  refuses on a failed, canceled or unfinished run, on none, and on no answer. A
  run on `github_ci` counts, since the commit, and the copy of `ci.yml` its run
  read, is the same. So step 3 waits for the push of step 2 to go green.
- **The version must be absent from the target index.** The check fails closed: a
  200 is a stale re-publish and anything other than a definitive 404 leaves the
  question unanswered and also aborts.

The condition on the ref is a line of `release.yml`, and a dispatch from a branch
runs that branch's copy of the file, which can drop it. The publish job runs in a
deployment environment named after the index, which the trusted publisher on the
index names, so each environment admits a deployment from `main` alone: a rule of
the repository's settings that no branch edits, which holds for every copy of the
workflow.

```bash
for env in testpypi pypi; do
  gh api -X PUT repos/ppigazzini/valgebra/environments/$env \
    -F 'deployment_branch_policy[protected_branches]=false' \
    -F 'deployment_branch_policy[custom_branch_policies]=true'
  gh api -X POST repos/ppigazzini/valgebra/environments/$env/deployment-branch-policies \
    -f name=main -f type=branch
done
```

## The order

1. **Read the tree against the last release**, on every CPython release a
   wheel is built for with a profile: an environment holding the version PyPI
   serves, built here from its tag, and one holding the tree, named at one
   length, and a `count` of the two, as [Reading one build against another](../11-performance.md#reading-one-build-against-another)
   reads them. Built here, the two differ in their sources alone. Run the
   three for each release, `LAST` the version PyPI serves:

   ```bash
   py="$(uv python find --managed-python 3.14)"
   uv run --no-sync python scripts/perf_inspect.py env last --python "$py" --build "v$LAST"
   uv run --no-sync python scripts/perf_inspect.py env head --python "$py" --build
   uv run --no-sync python scripts/perf_inspect.py count last head
   ```

   The merge gates hold each change to its base, on the interpreters the bench
   lanes run: a cost that grows by less than their band in each of several
   changes, or on another interpreter, passes every one of them. A shape that
   reads dearer is confirmed and placed as that section says, then fixed
   before the bump, or named in the version's section with what it buys.

2. **Bump, in one commit.** The workspace version, both lockfiles, and the
   changelog: roll the `Unreleased` entries into a section for the version,
   dated the day of this commit rather than the day the tag lands, and add its
   compare and tag links. Land it on `main`, so the merge gates run
   against the tree that is about to be published.

   The changelog ledger (`tests/test_changelog_ledger.py`) reads the *page* for
   which version is released and measures the roll against that version's tag.
   Between this step and step 7 the section names a version no tag resolves yet,
   so the ledger skips: there is nothing left to account for, and the roll is
   empty because it was just emptied. It resumes the moment the tag lands — and
   nothing holds a `feat`/`fix` landed in the window to a roll line until it
   does, so the tag is the step to take promptly rather than last.

   ```bash
   cargo metadata --format-version 1 --offline >/dev/null    # refresh Cargo.lock
   cargo metadata --format-version 1 --offline --manifest-path fuzz/Cargo.toml >/dev/null
   ```

3. **Dispatch to TestPyPI** — `publish_target: testpypi`, `confirm_version` the
   new version.
4. **Check what the index serves** (below).
5. **Dispatch to PyPI** — `publish_target: pypi`, the same version.
6. **Check what the index serves** again, against PyPI. A section with
   `Security` entries has their advisories published now, beside the release,
   as `SECURITY.md`'s process says: a fix and its advisory go out together.
7. **Tag the published commit**, annotated, subject `valgebra X.Y.Z`:

   ```bash
   git tag -a vX.Y.Z -m "valgebra X.Y.Z" <commit>
   git push origin vX.Y.Z
   ```

8. **Record the metamorphic reference from the published wheel**, in a commit of its
   own:

   ```bash
   <venv holding the published wheel>/bin/python scripts/metamorphic_gate.py --record
   ```

   Every later push is held to the answers that wheel gives, so a reference left on an
   older release holds a push to answers no user has, and a regression against the
   last one passes. The diff is read: every move is a widening or is a change this
   release's section says. `test_the_reference_is_the_last_release` holds the version.

9. **Read the published wheels' time against the last release's**, in the commit step 8
   makes, on every CPython release a wheel with the global lock is built for. A profiled
   build places its code anew each time it is built, so the wheel PyPI serves is a build
   no earlier step read, and step 1's count sees no placement at all. Four environments
   from PyPI, named at one length, two holding each release:

   ```bash
   py="$(uv python find --managed-python 3.15)"
   for name in old1 old2; do
     uv run --no-sync python scripts/perf_inspect.py env "$name" --python "$py" --release "$LAST"
   done
   for name in new1 new2; do
     uv run --no-sync python scripts/perf_inspect.py env "$name" --python "$py" --release "$NEW"
   done
   uv run --no-sync python scripts/perf_inspect.py count old1 new1
   uv run --no-sync python scripts/perf_inspect.py cycles old1 old2 new1 new2
   ```

   A shape both of this release's environments read moved alike, by 5% or more either
   way, where the last release's second one reads within the noise, is named in this
   release's section with the interpreter and the machine. A move one environment
   reads alone is that environment's: read beside one environment of the last release,
   3.14 moved eleven shapes 5% to 12% in both of 0.0.24's, and none of them repeated
   beside a second. A move is named and charged to no change, since the two wheels
   differ in the build that made them as well as in their source. The
   shapes hold a list of each scalar kind, each walked by a loop of its own that a build
   places apart from the others, and a list of ten thousand distinct integers beside the
   thousand. At one count, PyPI's CPython 3.15 wheel of 0.0.24 walks `list[None]` 62%
   slower than 0.0.23's, `list[bool]` 24% faster, and ten thousand integers a third
   slower where it walks a thousand in the same time.

   Three remedies are refused. A hash seed fixed in the release's training: two builds
   of one source by `scripts/page_figures.py`, which fixes it, placed the 3.15 list loop
   at offsets 48 and 0 in its line. A loop alignment flag:
   [measured and not taken](../11-performance.md#measured-and-not-taken). Publishing a
   dry run's wheels so a reading comes before the upload: each run already runs the
   suite on the wheels it uploads, and the section that names a move is part of the
   commit those wheels are built from.

## Checking an index (steps 4 and 6)

The workflow's smoke jobs prove each **artifact** imports and passes the product
suite. They cannot prove the
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
environment holds — install the test group into the same environment first,
at the lock's versions, **from PyPI and in their own install**, as the release
smokes do, and run the product suite, as they run it:

```bash
uv export --locked --only-group test --no-emit-project -o /tmp/vg-test.txt   # from the repository root
VIRTUAL_ENV=/tmp/vg uv pip install -r /tmp/vg-test.txt
/tmp/vg/bin/python -m pytest -q -m "not repository"
```

**The separate install is the point.** The test dependencies are not the package
under test, and resolving them against TestPyPI serves whatever anyone last
uploaded there — an ancient `syrupy` beside a `pytest` too old to start it. Only
valgebra comes from the index being checked; everything else comes from PyPI,
where it comes from in every other environment. The group and its versions are
read from the lock rather than listed here, so they cannot drift from what the
lanes install.

A test that needs a dependency the environment lacks skips rather than fails, so
read the skip list: a suite whose oracles are absent has checked less than the
same suite in a full development environment.

Run it on each PyPy series the release builds for as well (`uv venv --python
pypy@3.11`, `pypy@3.12`), because the PyPy wheels are different builds: plain
rather than profile-guided, since a profiled extension crashes there at the
walk's depth bound (`release.yml` says how), and the push lane's PyPy legs run the
suite on wheels they build themselves. The release smoke runs the
product suite on every wheel set that ships, PyPy's and the musllinux ones
included, and `tests/test_release_smoke.py` holds every wheel set the release
builds to a smoke row that runs it, and every release a build row names to a
row that loads it.

**Do not add PyPI as a second index while checking TestPyPI.** uv resolves a name
from the first index that carries it, so `--extra-index-url https://pypi.org/simple/`
makes it refuse the TestPyPI version in favor of the older released one — the
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
reads.** The per-version JSON endpoint can answer with every file of a release
while `uv pip install` still reports no such version, because the simple listing
has not caught up. That is the same message a failed upload gives, so read the
simple index itself (the `curl` above) before believing
either, and if it is the one that is behind, wait and retry rather than
re-dispatching the publish.

The interpreter is part of what is being checked, not a detail of the check. The
extension module is built per interpreter version rather than against the stable
ABI, so a release ships many wheels and one install exercises exactly one of them
— `release.yml` owns the matrix. On macOS and Windows the builds run on the
host, where `--find-interpreter` sees only the interpreters installed on the
image, so the workflow installs every supported one first; Windows builds its
free-threaded wheels in a job of their own, since a release and its
free-threaded build can fail to co-install in one step. On macOS a free-threaded
install also answers to the release's name, and `--find-interpreter` finds it
there in the release's place, so the macOS builds find each interpreter in the
tool cache `setup-python` records it in, under the runner's architecture, and
ask it which build it is -- a framework path holds only some releases. The
wheel job's ceiling (`timeout-minutes` in `release.yml`) is twice its slowest
row's reading, rounded up, which is the macOS x86_64 build: seven interpreters,
each built twice for its profile, on the slowest runner the matrix uses. A row
that reaches the ceiling is canceled, and the release waits on every row. The
Windows arm64 smoke runs
on the arm64 interpreters `setup-python` installs, because uv's own build of a
release candidate can be x64 alone. The maturin a release builds with is the one `uv.lock`
resolves, pinned in the workflow rather than taken as the newest. A version
selector resolves to whichever build is on the machine: `uv venv --python 3.14`
can land on the free-threaded interpreter, so read
`sysconfig.get_config_var("Py_GIL_DISABLED")` in the venv to record which wheel
the check actually covered.

## The wheels a release builds

**One wheel per interpreter version, not one per platform.** The extension is
built for each interpreter a platform row of `.github/workflows/release.yml`
names, not as a stable-ABI (`abi3`) wheel, so a minor `requires-python` admits
gets a wheel only where a row builds for it, and `tests/test_release_matrix.py`
holds each row's list to the classifiers. Why no stable-ABI wheel is on
[00-architecture.md](00-architecture.md#refused-and-what-each-one-stands-on). On manylinux and macOS the workflow
names each interpreter, on Windows it installs them before maturin's
`--find-interpreter` runs, and the musllinux rows build for what their image
carries. The rows name their free-threaded interpreters the same way: Linux,
macOS and Windows x64 by name, musllinux the free-threaded build its image
carries, and Windows arm64 none.

**Profile-guided where the host runs its own output.** A row marked `pgo: true`
builds an instrumented extension, trains it with the `pgo-command` in
`pyproject.toml` (`scripts/pgo_workload.py`), and rebuilds with the profile; the
musllinux cross-builds, the PyPy wheels and the Windows arm64 row ship plain
release builds with the same compatibility.

**No wheel links an interpreter.** maturin injects the pyo3 `extension-module`
behavior at build time, and `crates/valgebra-py/Cargo.toml` leaves the feature
unset, so `cargo test` and the workloads link an interpreter as any other
library does.

**Every musllinux wheel runs in a container.** The Docker library's
`python:<release>-alpine` images run the CPython wheels. It publishes no
free-threaded image, so a free-threaded wheel runs in plain Alpine on uv's musl build
of its release, mounted into the container. `tests/test_release_smoke.py` holds each
musl row to the free-threaded releases a glibc row builds.

**PyPy wheels carry one ABI, PyPy 8.0's.** Each series 8.0 implements, 3.11 and
3.12, gets a `pp80` wheel on manylinux x86_64 and aarch64, built from the
manylinux_2_28 image, since 8.0's binaries need glibc 2.27. That is a decision,
and what stands behind it is a rule: what ships has a lane. The `pypy` job runs
those two series on every push, on the PyPy 8.0 the pinned uv resolves, so a wheel
for PyPy 7.3 is one no push runs. An installer matches the ABI tag exactly, so a
7.3 interpreter installs from the source distribution, and nothing here tests that
install. The musllinux image builds PyPy wheels too, and the upload leaves them
out, because neither the Docker library's images nor uv's download list holds a
musl PyPy to load them.
`tests/test_release_matrix.py` holds the rows to that ABI and the series to the
job's legs, and `tests/test_release_smoke.py` holds the upload and each series'
smoke. A PyPy series or ABI moves in all three places together.

A dispatch whose publish target is left at none builds the whole matrix as a
dry run, which is how the matrix is checked before a release.

## Which interpreters a release supports

**The supported set is CPython's, on CPython's calendar.** A feature release is
supported from its first release candidate, when its ABI freezes, to the end of
the month its schedule PEP gives for its last security release. Each supported
release is a blocking leg of the `python` job in `ci.yml` and a classifier in
`pyproject.toml`, its free-threaded build is a blocking leg from 3.14, the first
PEP 779 supports, and the oldest is the floor that `requires-python`, ruff's
`target-version`, the ty and mypy floor steps and `tests/floor_names.json` name.
So the first release after a version's support ends drops it, and the first
release after the next one's first candidate supports it.

**The next release runs a year early, and forgiven.** From its first alpha it is
a leg of the `python` job with a `continue-on-error` that holds on that leg
alone, so a break shows while there is time to fix it and blocks no merge. Each
step the leg runs forgives it as well: the job's flag keeps the gate green but
not the job's own check, and a red check marks the commit with a red cross, so a
failure is forgiven at the step that fails and reads as that step's annotation
in the run. Each step it runs is bounded, and the bounds together fit inside the
job's limit, so a hang fails the step, which the leg forgives, rather than
running the job out: a job its own limit stops is canceled, and `ci` reads a
canceled job as red. The ledgers evaluate each flag on every leg, as the runner
does, rather than looking for the version in its text: `!= '3.16'` spells the
version and forgives every other leg. No other job forgives itself, and no other
step but the fuzz lane's corpus minimization, whose failure reports nothing
about the code. PyO3 builds for
the release after the newest it supports with a warning that the artifact must
not be distributed, which is what a forgiven leg is; it refuses the release
after that unless `UNSAFE_PYO3_SKIP_VERSION_CHECK=1` is set.

`tests/test_python_lifecycle.py` holds all of it against the day the suite runs,
from a table of the schedule PEPs' dates. It turns red on the calendar, with no
commit, and its message names the edit. It does not read the wheels
`release.yml` builds: `tests/test_release_matrix.py` holds each platform's list
to the classifiers the calendar moves, with the gaps the platforms set named
and a row that leaves its interpreters to the build image named too, and each
free-threaded build and each PyPy series a wheel is built for to a leg.

## A companion reads the release

`valgebra-vtjson` and `valgebra-jsonschema` are packages of their own over this one's public
surface, and each holds answers this tree's gates do not. What that asks of a release:

- **A `feat` to the deciders reads the companions' studies first**, against a wheel of the tree,
  before step 2. `valgebra-vtjson`'s differential suite runs; `valgebra-jsonschema`'s suite, corpus
  and containment studies run with `--check`, on a checkout whose `valgebra` is that wheel. A count
  may move: more proven, fewer undecided. A proof against a containment label, or a new
  disagreement with the `jsonschema` package, is a defect here, and it stops the bump.
- **A companion re-pins on a minor.** A `0.0.x` minor is a breaking release to them, as their
  `pyproject.toml` floors say, so each moves its floor with a release of its own, and holds its
  ceiling at the next minor: `valgebra>=0.0.25,<0.0.26`. A ceiling of `<0.1` admits the next
  minor before the companion's suite has run on it, so a resolver installs the two together the
  day this package releases.
- **The floor and the matrix are theirs too.** A companion's `requires-python` follows
  [Which interpreters a release supports](#which-interpreters-a-release-supports), and it builds no
  wheel, so it reads this package's on each.

Each bump costs a companion two commits and a session, so a release batches its `feat`s.

## What this does not cover

- **The wheels a local count cannot run.** Step 1 reads each CPython release's
  Linux x86_64 build, and step 9 the one PyPI serves. Every other wheel is a build
  of its own, for another platform, architecture or implementation, and no reading
  holds it to the release before.
- **A source install off Linux.** `uv pip install` takes the wheel; the sdist
  path is compiled and tested once by the workflow, on Linux, and
  `--no-binary valgebra` locally is the only way to reach it on another
  platform.
- **A published version that is wrong.** It cannot be replaced, only yanked, and
  the workflow refuses a version the index already serves. The remedy is the next
  patch version, which is why step 4 exists before step 5, and step 1 before
  the bump.
