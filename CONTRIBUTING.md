# Contributing to valgebra

Thanks for your interest. valgebra is a Rust-core Python validation library;
this guide covers local setup and the checks every change must pass. The
project's design rules and non-negotiable invariants live in
[AGENTS.md](AGENTS.md) — read that first.

## Orientation

valgebra is two Rust crates plus a Python package: the pure-Rust core
(`crates/valgebra-core/`) holds the schema IR, the denotation of every node, and
the two schema-to-schema deciders — the structural rules in `decision.rs` and
the set representation in `descr/`; the PyO3 bindings (`crates/valgebra-py/`)
hold the schema frontend, the single membership walk, and the oracle the core
asks about a Python class or constant; and `python/valgebra/` is the public
surface.
[ARCHITECTURE.md](ARCHITECTURE.md) maps the components and the path a value takes
from a typing annotation through the walk to a violation; read it before a
non-trivial change.

**The developer documentation set is [docs/dev/](docs/dev/README.md).** One page
per zone of the source, each the live claim about it: what a node denotes, what
is decidable, where soundness is decided, why each type has its shape, what every
gate can and cannot see, and the words this project uses without stopping to
define them. It is not published with the user guide — it describes what valgebra
is made of rather than how to use it. Change a zone, fix its page in the same
commit.

A change to the schema language flows the same way each time: extend the IR or
the frontend, **write the node's denotation** (the set of Python values it
accepts) in the same change, **cover its algebra laws** with property tests, then
run the gate. A combinator described only as "like some other tool" does not
land.

## Setup

Requirements: stable Rust (edition 2024, MSRV 1.88), Python >= 3.10, and
[`uv`](https://docs.astral.sh/uv/).

```bash
uv sync                                   # create .venv and install dev dependencies
uv run --no-sync maturin develop --uv     # build the Rust extension into the venv
uv run --no-sync pre-commit install       # enable the git hooks
```

Verify the build:

```bash
uv run --no-sync python -c "from valgebra import Validator; print(Validator(int).is_valid(7))"
```

Building the docs site locally needs the extension built first (`uv run
--no-sync maturin develop --uv`): the API reference introspects the compiled
module to render the public surface's docstrings, which live on the Rust objects
rather than being duplicated in the type stub.

## The gate

A change is not done until every command exits 0. CI runs the lanes these
preview on Linux, and `cargo test` and the Python suite on macOS and Windows as
well -- `.github/workflows/ci.yml` owns which lane runs where; local runs are
previews of that source of truth.

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
# The workspace links libpython: a virtual environment's interpreter is not on
# the default loader path, and a test binary that cannot find it does not start.
# `scripts/gate.py` sets the same two variables for the steps it runs.
export PYO3_PYTHON="$(uv run --no-sync python -c 'import sys; print(sys.executable)')"
export LD_LIBRARY_PATH="$(uv run --no-sync python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))'):${LD_LIBRARY_PATH:-}"
cargo test
cargo check --manifest-path fuzz/Cargo.toml --all-targets
uv run --no-sync python scripts/docs_lint.py
uv run --no-sync maturin develop --uv
uv run --no-sync ruff check . && uv run --no-sync ruff format --check .
uv run --no-sync ty check
uv run --no-sync pytest
```

Every `uv run` after the sync carries `--no-sync`: a bare one re-syncs the
environment first, which replaces the build `maturin develop` installed, so the
commands after it would test a module other than the one just built.

The binding's four corpora sit behind `--features interpreter-tests`, which
links an **embedded** interpreter -- one with no virtual environment around it,
reading its standard library from the prefix it was built for. Where the first
`python3` on your path is a different release from the one `PYO3_PYTHON` names,
that prefix is the wrong one and the test binary dies in `init_fs_encoding`
before a test runs, saying only `No module named 'encodings'`. Naming the
interpreter's own prefix for that one command is the fix, and it is scoped to
the command because exporting it would send every later `uv run` to the same
prefix and past the environment:

```bash
PYTHONHOME="$(uv run --no-sync python -c 'import sys; print(sys.base_prefix)')" \
  cargo test -p valgebra-py --features interpreter-tests
```

The fuzz crate is a **detached workspace** -- libFuzzer needs a nightly
toolchain, and making it a member would put nightly on every stable gate's path
-- so `cargo check --workspace` does not reach it and it needs its own line. A
change to the core's public types compiles cleanly without it and turns the fuzz
lane red. `tests/test_build_surfaces.py` holds every manifest in the tree to
being a workspace member or a detached surface named here with the command that
builds it.

`pre-commit run --all-files` runs the file-hygiene, ruff, `ty`, cargo and
docs-lint gates in one step.

## Contract inventory

The gate above is the whole set. This is the other question: **what does this
project promise, and how do I check just that one promise?** Every row names the
file that owns the contract and the single command that reproduces its verdict.

| Contract | Source of truth | First rerun command |
|---|---|---|
| the merge gate's steps a developer can run pass, in a clone shaped like a runner's | `.github/workflows/ci.yml` | `uv run --no-sync python scripts/gate.py` |
| Rust formatting | rustfmt defaults, unconfigured | `cargo fmt --check` |
| Rust lint policy | `Cargo.toml` `[workspace.lints]` | `cargo clippy --all-targets --all-features -- -D warnings && cargo clippy --all-targets -- -D warnings` |
| Rust behaviour | `crates/` | `cargo test` |
| Rust documentation links | the doc comments in `crates/` | `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps && RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --document-private-items` |
| the extension links on PyPy | `crates/valgebra-py/src/` | `uv venv --python pypy-3.11 .pypy && VIRTUAL_ENV=.pypy uv pip install maturin && VIRTUAL_ENV=.pypy .pypy/bin/maturin build --release --out dist-pypy -i "$(uv python find pypy-3.11)" && VIRTUAL_ENV=.pypy uv pip install --no-index dist-pypy/*.whl && .pypy/bin/python scripts/pypy_import_check.py` |
| the binding's four corpora | `crates/valgebra-py/src/check/walk/interpreter.rs`, `crates/valgebra-py/src/build/interpreter.rs`, `crates/valgebra-py/src/equality/interpreter.rs`, `crates/valgebra-py/src/oracle/interpreter.rs` | `cargo test -p valgebra-py --features interpreter-tests` |
| the detached fuzz surface | `fuzz/Cargo.toml` | `cargo check --manifest-path fuzz/Cargo.toml --all-targets` |
| fuzz harness laws | `fuzz/src/lib.rs` | `cargo +nightly test --manifest-path fuzz/Cargo.toml --lib` |
| Python behaviour | `tests/` | `uv run --no-sync pytest` |
| Python lint and format | `pyproject.toml` `[tool.ruff]` | `uv run --no-sync ruff check . && uv run --no-sync ruff format --check .` |
| Python types | `pyproject.toml` | `uv run --no-sync ty check` |
| documentation claims | every tracked `*.md` | `uv run --no-sync python scripts/docs_lint.py` |
| the API reference carries the compiled docstrings | `scripts/docs_stubs.py` | `uv run --no-sync --group docs mkdocs build --strict && uv run --no-sync python scripts/docs_stubs.py --check` |
| the use cases the tree has, and how many the suite names | the type stub and the codes the walk writes | `uv run --no-sync python scripts/use_case_ledger.py` |
| the branch arms the core's tests reach | an instrumented run on the pinned nightly | `uv run --no-sync python scripts/branch_coverage.py branches.json` |
| when a `typing` or `enum` name, or a stdlib module, arrived | `tests/floor_names.json` | `uv run --no-sync python scripts/floor_names.py --check` |
| doc examples run | `docs/` | `uv run --no-sync python scripts/run_doc_examples.py` |
| doc examples read clean under ty, mypy, pyright and ruff, or say why | `scripts/check_doc_examples.py` | `uv run --no-sync python scripts/check_doc_examples.py` |
| the rendered site builds | `mkdocs.yml` | `uv run --no-sync --group docs mkdocs build --strict` |
| core instruction budget | `scripts/perf_budget.json` | `uv run --no-sync python scripts/perf_gate.py` |
| binding instruction budget | `scripts/perf_budget.json` | `uv run --no-sync python scripts/perf_gate.py --binding` |
| competitive ratio | `scripts/perf_compare.json` | `uv run --no-sync --group bench python scripts/compare_gate.py` |
| what a profile buys, per shape | the two wheels a run builds | `uv run --no-sync --group bench python scripts/pgo_compare.py --record plain.json` |
| membership held and decisions only widened | `scripts/metamorphic_reference.json` | `uv run --no-sync python scripts/metamorphic_gate.py` |
| core mutation adequacy | `scripts/mutation_baseline.json` | `cargo mutants --package valgebra-core -- -- --skip deep_subtype_into_bottom_terminates --skip subtyping_terminates_on_a_distributed_tower` |
| walk mutation adequacy | `scripts/mutation_baseline_walk.json` | `cargo mutants --package valgebra-py --features interpreter-tests -- -- --skip recursion_deeper_than_the_bound_is_refused --skip the_two_readings_agree_at_the_walks_depth_bound` |
| suite mutation adequacy | `scripts/mutation_baseline_pytest.json` | `cargo mutants --config .cargo/mutants-pytest.toml --package valgebra-py --features pytest-sweep` |
| a mutation verdict | any baseline | `python3 scripts/mutation_gate.py --baseline core` |
| no file under the per-file coverage floor | `scripts/coverage_gate.py`'s floor and the files named under it, per lane scope | `uv run --no-sync python scripts/coverage_gate.py --json coverage-core.json --scope core` |
| where a change starts, for the bench gate and the diff sweeps, a force-push included | `scripts/change_base.py` | `uv run --no-sync pytest tests/test_change_base.py` |
| supply chain (Rust) | `deny.toml` | `cargo deny check` |
| supply chain (Python) | `uv.lock` | `uv export --locked --format requirements-txt --no-emit-project --all-groups > requirements-audit.txt && uvx pip-audit -r requirements-audit.txt` |
| workflow security | `.github/` | `uvx zizmor .github/` |
| the profile-guided build's training run | `scripts/pgo_workload.py`, `pyproject.toml` `pgo-command` | `uv run --no-sync --group bench maturin build --release --pgo --out dist` |

The first two mutation rows take a skip list; `docs/dev/07-tooling-ci.md` says
which and why. The third needs `VALGEBRA_SWEEP_VENV` naming a place outside the
tree for its environments -- one per sweep worker, built from the lock file --
and fails rather than skipping without one; that document carries the whole
invocation. The binding row sweeps the crate minus
the exclusions `.cargo/mutants.toml` names, which is a superset of what the
lane's own step runs; that step's `--file` list lives in `.github/workflows/ci.yml` and is not
copied here, because a list in two places drifts by one entry and reads exactly
like one that has not. Commands that need an embedded interpreter need its library directory
on the loader path — the binding-coverage job in `.github/workflows/ci.yml` shows
the two lines that set it. The lint row is the two workspace passes; the
`rust lint` job adds a third, three restriction lints over the library targets.
The `pip audit` and `zizmor` jobs pin the tool versions their `uvx` commands
run, and a local `uvx` without a pin takes the latest.

`tests/test_contract_inventory.py` holds this table to the tree in both
directions: a gate script with no row fails, and a row naming a script or a
source of truth that does not exist fails.

## Testing

Correctness is checked against the denotation, not against itself. The layers
below are the ones a contributor runs into first;
[docs/dev/08-testing.md](docs/dev/08-testing.md) owns the full table, with what
each layer is blind to beside it.

- **Denotation oracle.** Each node's denotation is written as a reference
  predicate over a value generator; the membership walk is property-tested to
  agree with it. This is the core correctness check.
- **Differential fuzzers.** The JSON path is fuzzed against the object path
  (a document is judged as `json.loads` of it would be), and the fast `bool`
  walk against the explaining walk, so the two never diverge.
- **External ground truth.** The same schemas and values run through valgebra and
  through pydantic-core (strict object path) and jsonschema (JSON path); every
  divergence is either a valgebra bug that fails the gate or one of a small,
  enumerated set of documented intentional differences, which the docstring of
  `tests/test_differential.py` lists with the case each is localized to.
- **Algebra laws as property tests.** Every claimed equivalence — associativity,
  De Morgan, the complement laws, the folds construction applies — is proved
  with proptest (Rust) and hypothesis (Python) against the membership relation,
  never asserted.
- **Snapshots.** Error messages and `repr` output are pinned with insta and
  syrupy so a wording change is a deliberate, reviewed diff.
- **Coverage-guided fuzzing.** The libFuzzer target in `fuzz/` drives the
  decision procedures with `arbitrary`-built schemas, asserting the sound
  invariants (no panic, the order laws). The same invariants run on the merge
  gate as structural property tests; the fuzz soak runs nightly, and its corpus
  is cached across runs and minimized after each, so the fuzzer accumulates the
  inputs it has learned reach new code instead of restarting cold from the
  committed seeds. Build and run the target with
  `cargo +nightly fuzz run decision fuzz/corpus/decision fuzz/seeds/decision`
  (needs `cargo-fuzz`); the corpus directory is named first, so what the run
  learns lands there rather than in the tracked seeds.

Run the Rust property suites with `cargo test`; raise the example count with
`PROPTEST_CASES=30000`. Run the Python suites with `uv run --no-sync pytest`;
the example count there is a **profile** rather than a number, selected by
`HYPOTHESIS_PROFILE` and registered in `tests/conftest.py`, which owns the
budgets:

```bash
HYPOTHESIS_PROFILE=nightly uv run --no-sync pytest tests/test_laws.py
```

`dev` is the default and the edit-test loop's, `ci` the wider budget that still
finishes a merge gate, and `nightly` the deep one that hunts the long tail.

## Continuous integration

The `ci.yml` workflow gates every push to `main` and to `github_ci`, and every
pull request against `main`; the aggregated `ci` check is green only when every
job is. `ci.yml` owns the job set, and
`tests/test_required_jobs.py` holds the aggregator to it in both directions, so
the list is not restated here: a second copy drifts by one entry and reads
exactly like one that has not. What is worth knowing about its shape is the
Python matrix, which runs **every supported interpreter on every push**, the
free-threaded build among them. `ci.yml` owns the matrix too:
`tests/test_python_lifecycle.py` holds the releases it carries to the schedule
each release follows, and which leg may fail without blocking to the release's
stage, and `tests/test_required_jobs.py` holds that the job runs on every event
and has a free-threaded leg. [docs/dev/07-tooling-ci.md](docs/dev/07-tooling-ci.md)
says why each interpreter is a lane of its own.

Scheduled lanes, on `github_ci` by a dispatch from the scheduled run on
`main`, run the deep property suites, a libFuzzer soak over the core, and three
mutation sweeps — the core crate, the membership walk under an
embedded interpreter, and the files the shipped extension is the only caller of,
swept with the Python suite as the test command — whose survivors are ratcheted
against their own committed baselines: a survivor the baseline does not accept fails the lane, and so does a
baseline entry whose mutant the tests kill. The target is never zero —
equivalent mutants exist and are undecidable — so an accepted survivor carries
the argument for why no test can kill it.
Every push also runs the first two **restricted to the whole files the diff
touches**, which is bounded by the change rather than by the tree and so blocks
merges, under one fixed proptest seed, so a push's verdict is the tree's and not
the draw's; the third costs a suite run per mutant and stays on the schedule. Each
checks the new-survivor direction alone, because a partial sweep never generates
most of the baseline and the expiry direction is not its to judge.
Performance is gated two ways: a **deterministic cachegrind instruction count**
of the core, decision and binding workloads, compared with the same workloads
built at the change's merge base (`scripts/perf_gate.py --against`), and a
**competitive ratio** of per-call time against pydantic-core across a shape
matrix. Both are independent of the runner's absolute speed — the instruction
count by construction, the ratio by cancellation — so they block merges where a
wall-clock budget could not.

## Vocabulary

These words carry weight in this repository and in its CI, and some of them
collide with an unrelated sense used nearby. Say which one you mean:
[docs/dev/13-glossary.md](docs/dev/13-glossary.md) defines them and owns the
list of the ones that collide, and a second copy here would drift from it.

Every gate script reports one of **three** outcomes, and the exit code says
which, so a caller can dispatch on it rather than parse the output:

| exit | meaning |
| --- | --- |
| **0** | the gate ran and the property holds |
| **1** | the gate ran and the property does not hold |
| **2** | the gate **could not run** — no measurement, no baseline, a missing dependency |

A gate that could not run has proven nothing, and must never read as one that
passed. `perf_gate.py` exits 2 on cachegrind output it cannot parse,
`mutation_gate.py` on a missing sweep result or baseline, and `compare_gate.py`
without its benchmark dependency.

## Working on changes

- Push to `github_ci`, force-pushing as often as the work needs, and move
  `main` only to a commit whose `ci` check is green there:
  `git push origin <commit>:main`, a fast-forward to the commit that passed. A
  red push lane is repaired on `github_ci`, before `main` has the commit. A
  pull request against `main` runs the same workflow, and the aggregated `ci`
  check must be green to merge it.
- Keep the Python/Rust boundary explicit: the validator tree runs in Rust;
  Python predicates are a documented slow path, never a silent fallback.
- No schema combinator or annotation form lands without its denotation written
  in the same change and its algebra laws covered by property tests.
- **A lane broken by a commit is repaired by amending that commit**, so the
  history carries no commit that was known to be red. The floor is the last
  release tag: a commit at or below it is repaired by a commit *above* it
  instead.

    Below the tag the cost is not the rewrite, it is what the rewrite leaves.
    A tag keeps resolving after the commit under it is replayed, `git show`
    keeps printing it, and no diff of the tree is different — the change is to
    reachability, which nothing reads. `tests/test_cited_commits.py` reads it:
    every `v*` tag is held to being an ancestor of the branch, beside the
    commits the tracked files cite.

See [AGENTS.md](AGENTS.md) for the full rules and the rationale behind them.

## Commit messages

Conventional commits, body wrapped at 80 columns, authoritative mood (describe
what the system does after the change, not the act of changing it).

```
feat: short imperative summary

Body wrapped at 80 columns describing the resulting behavior.
```
