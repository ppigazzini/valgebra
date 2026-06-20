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
uv run maturin develop --uv
uv run ruff check . && uv run ruff format --check .
uv run ty check
uv run pytest
```

`pre-commit run --all-files` runs the file-hygiene, ruff, and cargo gates in
one step.

## Testing

Correctness is checked against the denotation, not against itself. The harness
has four layers:

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

Run the Rust property suites with `cargo test`; raise the example count with
`PROPTEST_CASES=30000`. Run the Python suites with `uv run pytest`; raise it with
`HYPOTHESIS_MAX_EXAMPLES`.

## Continuous integration

The `ci.yml` workflow gates every push and pull request; the aggregated `ci`
check is green only when every job is. The jobs: Rust lint and test (Linux,
macOS, Windows), an MSRV build at the manifest's `rust-version`, two coverage
lanes (the core crate, and the bindings measured by instrumenting the extension
and driving it with the Python suite against a line floor), a Python matrix from
3.10 through 3.15 — the 3.15 prerelease and the free-threaded lane run but never
block — a differential lane that cross-checks membership against pydantic-core
and jsonschema, the doc-example runner, a strict docs build, and a Linux wheel
build.
Performance is gated by a **deterministic cachegrind instruction count** compared
to a committed budget: independent of runner noise, so it blocks merges where a
wall-clock budget could not.

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
