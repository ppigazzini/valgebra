# Contributing to valgebra

Thanks for your interest. valgebra is a Rust-core Python validation library;
this guide covers local setup and the checks every change must pass. The
project's design rules and non-negotiable invariants live in
[AGENTS.md](AGENTS.md) — read that first.

## Orientation

valgebra is two Rust crates plus a Python package: the pure-Rust core
(`crates/valgebra-core/`) holds the schema IR and the denotation of every node;
the PyO3 bindings (`crates/valgebra-py/`) hold the schema frontend and the single
membership walk; and `python/valgebra/` is the public surface.
[ARCHITECTURE.md](ARCHITECTURE.md) maps the components and the path a value takes
from a typing annotation through the walk to a violation; read it before a
non-trivial change.

A change to the schema language flows the same way each time: extend the IR or
the frontend, **write the node's denotation** (the set of Python values it
accepts) in the same change, **cover its algebra laws** with property tests, then
run the gate. A combinator described only as "like some other tool" does not
land.

## Setup

Requirements: stable Rust (edition 2024, MSRV 1.88), Python >= 3.10, and
[`uv`](https://docs.astral.sh/uv/).

```bash
uv sync                     # create .venv and install dev dependencies
uv run maturin develop      # build the Rust extension into the venv
uv run pre-commit install   # enable the git hooks
```

Verify the build:

```bash
uv run python -c "from valgebra import Validator; print(Validator(int).is_valid(7))"
```

Building the docs site locally needs the extension built first
(`uv run maturin develop`): the API reference introspects the compiled module to
render the public surface's docstrings, which live on the Rust objects rather
than being duplicated in the type stub.

## The gate

A change is not done until every command exits 0. CI runs the same set on
Linux, macOS, and Windows; local runs are previews of that source of truth.

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo check --manifest-path fuzz/Cargo.toml
uv run maturin develop --uv
uv run ruff check . && uv run ruff format --check .
uv run ty check
uv run pytest
```

The fuzz crate is a **detached workspace** -- libFuzzer needs a nightly
toolchain, and making it a member would put nightly on every stable gate's path
-- so `cargo check --workspace` does not reach it and it needs its own line. A
change to the core's public types compiles cleanly without it and turns the fuzz
lane red. `tests/test_build_surfaces.py` holds every manifest in the tree to
being a workspace member or a detached surface named here with the command that
builds it.

`pre-commit run --all-files` runs the file-hygiene, ruff, and cargo gates in
one step.

## Testing

Correctness is checked against the denotation, not against itself. The harness
has six layers:

- **Denotation oracle.** Each node's denotation is written as a reference
  predicate over a value generator; the membership walk is property-tested to
  agree with it. This is the core correctness check.
- **Differential fuzzers.** The JSON path is fuzzed against the object path
  (a document is judged as `json.loads` of it would be), and the fast `bool`
  walk against the explaining walk, so the two never diverge.
- **External ground truth.** The same schemas and values run through valgebra and
  through pydantic-core (strict object path) and jsonschema (JSON path); every
  divergence is either a valgebra bug that fails the gate or one of a small,
  enumerated set of documented intentional differences (bool as a subtype of
  int, int and float as disjoint regions, exact-match `Literal` membership).
- **Algebra laws as property tests.** Every claimed equivalence — associativity,
  De Morgan, the complement laws, a simplifier rewrite — is proved with proptest
  (Rust) and hypothesis (Python) against the membership relation, never asserted.
- **Snapshots.** Error messages and `repr` output are pinned with insta and
  syrupy so a wording change is a deliberate, reviewed diff.
- **Coverage-guided fuzzing.** libFuzzer targets in `fuzz/` drive the simplifier
  and the decision procedures with `arbitrary`-built schemas, asserting the sound
  invariants (no panic, idempotent normalization, the order laws). The same
  invariants run on the merge gate as structural property tests; the fuzz soak
  runs nightly, and its corpus is cached across runs and minimized after each,
  so the fuzzer accumulates the inputs it has learned reach new code instead of
  restarting cold from the committed seeds. Build and run a target with
  `cargo +nightly fuzz run simplify fuzz/corpus/simplify fuzz/seeds/simplify`
  (needs `cargo-fuzz`); the corpus directory is named first, so what the run
  learns lands there rather than in the tracked seeds.

Run the Rust property suites with `cargo test`; raise the example count with
`PROPTEST_CASES=30000`. Run the Python suites with `uv run pytest`; raise it with
`HYPOTHESIS_MAX_EXAMPLES`.

## Continuous integration

The `ci.yml` workflow gates every push and pull request; the aggregated `ci`
check is green only when every job is. The jobs: Rust lint and test (Linux,
macOS, Windows), an MSRV build at the manifest's `rust-version`, two coverage
lanes (the core crate, and the bindings measured by instrumenting the extension
and driving it with the Python suite against a line floor), a Python matrix from
3.10 through 3.15 — the 3.15 prerelease lane runs without blocking, while the
free-threaded 3.14t lane blocks merges — a differential lane that cross-checks
membership against pydantic-core
and jsonschema, the doc-example runner, a strict docs build, and a Linux wheel
build. Scheduled lanes run the deep property suites, a libFuzzer soak over the
core, and two mutation sweeps — the core crate, and the membership walk under an
embedded interpreter — whose survivors are ratcheted against their own committed
baselines: a survivor the baseline does not accept fails the lane, and so does a
baseline entry that is no longer a survivor. The target is never zero —
equivalent mutants exist and are undecidable — so an accepted survivor carries
the argument for why no test can kill it.
Every push also runs the same sweeps **restricted to the lines the diff
touches**, which is bounded by the change rather than by the tree and so blocks
merges; it checks the new-survivor direction alone, because a partial sweep never
generates most of the baseline and the expiry direction is not its to judge.
Performance is gated two ways: a **deterministic cachegrind instruction count**
over the core engine compared to a committed budget, and a **competitive ratio**
of per-call time against pydantic-core across a shape matrix. Both are
independent of the runner's absolute speed — the instruction count by
construction, the ratio by cancellation — so they block merges where a
wall-clock budget could not.

## Vocabulary

These words carry weight in this repository and in its CI, and three of them
collide with an unrelated sense used nearby. Say which one you mean.

| term | here it means |
| --- | --- |
| **gate** | a step that **asserts** and exits non-zero when the assertion breaks. A step that only builds, measures, or records is not one |
| **lane** | one independently driven run: a CI job in `ci.yml`, or one target inside a step that drives several |
| **denotation** | the set of Python values a schema admits. Every node has one written down; membership is testing a value against it |
| **oracle** | a judge of a claim that does not go through the code under test |
| **survivor** | a mutation of the source that the tests did not notice. A signal about the tests, never about the mutation |
| **equivalent mutant** | a mutation that provably cannot change any result, so no test can kill it. Excluded from the sweep with its argument, never counted as a gap |
| **ratchet** | a committed floor that may only move one way. The mutation baseline is one; a budget is not |
| **budget** | a committed ceiling a measurement is held to |
| **ledger** | an enumerated list held to the tree in both directions, so an entry that stops being true fails and a subject with no entry fails too |
| **region** | one part of the value universe in the partition the decision procedure computes over |
| **pool** | the validator's table of Python objects: a literal's constant, a class, a comparison operand, a predicate |

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

Three collisions, and both senses are live:

| term | one sense | the other |
| --- | --- | --- |
| **gate** | a CI step that asserts | the local build-health command set above, which is a preview of the merge gate rather than a single check |
| **oracle** | an independent denotation predicate, or an external library, judging the walk | `LeafRelations`, the trait through which the decision procedure asks the bindings about a class or a value |
| **budget** | the committed instruction count a workload is held to | `DECISION_BUDGET`, the work ceiling one decision query may spend before returning the conservative answer |

## Working on changes

- Branch off `main`; open a pull request. Both push and PR trigger CI, and the
  aggregated `ci` check must be green to merge.
- Keep the Python/Rust boundary explicit: the validator tree runs in Rust;
  Python predicates are a documented slow path, never a silent fallback.
- No schema combinator or annotation form lands without its denotation written
  in the same change and its algebra laws covered by property tests.

See [AGENTS.md](AGENTS.md) for the full rules and the rationale behind them.

## Commit messages

Conventional commits, body wrapped at 80 columns, authoritative mood (describe
what the system does after the change, not the act of changing it).

```
feat: short imperative summary

Body wrapped at 80 columns describing the resulting behavior.
```
