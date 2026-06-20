# Performance

valgebra compiles a schema once into a Rust validator tree and crosses into Rust
exactly once per validation call. This page records how that is measured, a
reproducible baseline against other validators, and the honest limits of the
numbers.

A speed claim is only as good as its methodology. Every number here states the
harness, the dataset, the library versions, and the machine class. Re-run the
harnesses on your own hardware before relying on a ratio: absolute times move
with the CPU, and the comparison points do different amounts of work.

## What is measured

Two harnesses, one per side of the boundary:

- **Core micro-benchmarks** (`crates/valgebra-core/benches/core.rs`, criterion)
  time the pure-Rust schema transformations — the simplifier, the index remap
  behind validator composition, and the recursive open/closed record transform.
  No Python is involved.
- **End-to-end benchmarks** (`benches/`, pytest-benchmark) time a single
  boundary-crossing validation call through the public API, over synthetic
  shapes that each stress one cost dimension.

Run them with:

```bash
# Core micro-benchmarks (Rust):
cargo bench --bench core

# End-to-end and comparison benchmarks (Python); install the bench group first:
uv sync --group bench
# To match the published figures, build the same PGO wheel the release ships
# (needs the llvm-tools rustup component) and install it; a plain build is slower:
uv run --group bench maturin build --release --pgo --out dist
uv pip install --reinstall --no-deps dist/*.whl
uv run --group bench pytest benches/bench_validate.py
uv run --group bench pytest benches/bench_compare.py --benchmark-group-by=group
```

## Comparison is not apples-to-apples

The comparison runs the same shapes through three checkers that do **different**
work. Read the ratios with that in mind:

- **valgebra** checks membership of the object already in hand: no copy, no
  coercion. `is_valid` returns a bool through the membership fast path.
- **jsonschema** (`Draft202012Validator.is_valid`) is also a pure check with no
  coercion — the closest semantic analogue — but it is pure Python.
- **pydantic** (`TypeAdapter.validate_python`, strict mode) validates *and
  constructs* a value. Strict mode disables coercion, but it still builds and
  returns output, so it does strictly more work than a membership check. It is
  the relevant point of comparison because it is the fast, Rust-cored validator
  most users reach for.

The record shape compares valgebra's closed record against a pydantic
`TypedDict` and a jsonschema object with `additionalProperties: false`, so all
three check the same set of named fields.

## Baseline matrix

Machine class: AMD Ryzen 7 PRO 7840U (Zen 4, 8c/16t, up to 5.1 GHz, a 2023-era
mobile part) under WSL2 on Linux 6.18. Toolchain: rustc 1.96.0, CPython 3.14.6,
pydantic 2.13.4, jsonschema 4.26.0, criterion 0.8.2, pytest-benchmark 5.2.3. The
extension is the **PGO** release build — the profile-guided, fat-LTO wheel the
release ships (`maturin build --release --pgo`); a plain `--release` build is a
few tens of percent slower on these shapes, and a debug build is not
representative. pydantic's PyPI wheels are likewise PGO-built, so this is a
release-to-release comparison. Figures are the per-call median; re-run on your
own hardware for absolute numbers. They are measured on the wheel carrying
valgebra's full feature set — the per-validator precompute (record-field
lookups, literal-union dispatch) and native string patterns — which leaves these
shapes unchanged: the features earn their keep elsewhere, not by regressing the
core.

End-to-end validation of a value that passes (median time per call, lower is
better):

| Shape | valgebra | pydantic (strict) | jsonschema |
| --- | --- | --- | --- |
| `list[int]`, 10,000 elements | 47 us | 88 us | 26,000 us |
| Closed record, 50 int fields | 0.97 us | 1.9 us | 134 us |
| Nested `list[...]`, depth 25 | 0.25 us | 1.9 us | 77 us |

valgebra relative to pydantic on this machine: ~7x faster on deep nesting, ~2x
faster on the wide record, and ~1.9x faster on the large flat array. It is
consistently far ahead of pure-Python jsonschema — by two to three orders of
magnitude on every shape. pydantic does strictly more work on the record (it
constructs output), so read that shape as a margin over a heavier operation, not
a like-for-like loss for pydantic.

Core micro-benchmarks (criterion, release+LTO, indicative single run):

| Operation | Corpus | Median |
| --- | --- | --- |
| `simplify` | redundant Boolean expression, depth 8 | ~1.3 us |
| `shifted` | 64-field pool-indexed record | ~2.0 us |
| `with_records_open` | record spine, depth 32 | ~4.8 us |

## Honest limits

- The numbers are a single machine class. They establish relative behavior, not
  a universal ranking. Shared CI runners are too noisy for a tight wall-clock
  budget, so the merge gate measures a deterministic instruction count instead.
- The margins against pydantic come with the caveat that the two tools do
  different work: pydantic constructs output, valgebra only checks membership.
  The ratios answer "how fast is each tool's validation step," not "how much
  faster is membership than construction." Deep nesting is the widest gap; the
  array and record margins are narrower but consistent.
- The comparison measures different operations (check vs check-and-construct vs
  pure-Python check). It answers "how fast is the validation step for each
  tool," not "are these tools interchangeable" — they are not. See the README
  for what valgebra is and is not for.
- These figures are for the object path — validating a value already in hand.
  The JSON input path is measured separately, on the same machine class, in the
  [JSON page](json.md).

## How the record fast path is tuned

The closed-record membership check visits each dict entry once and matches the
key against the declared fields, rather than looking up every declared field in
turn (which builds a temporary Python string per field) and then scanning the
dict a second time for undeclared keys. The key's UTF-8 is borrowed without
allocating, and the field-name index is computed once when the validator is
first used — with a fast non-cryptographic hasher, since the keys are the
schema's own declared names rather than attacker input — then reused across
calls, so a wide record no longer rebuilds and reallocates its name map on every
validation. On the 50-field record above this measures ~1.0 us per call (PGO
release build); the earlier per-field-lookup form was several times slower. Profiling
with cachegrind attributed the removed cost to temporary-string creation,
hashing, and allocation churn from the per-field lookups, and that attribution
is an instruction count, so it holds across machine classes. The bool fast path
and the aggregating explain walk stay membership-equivalent, locked by tests
that assert both reach the same verdict across record shapes.

## How large literal unions dispatch

A union whose members are all literals (a `Literal["a", "b", ...]` enum, or a
discriminator) is compiled once into value-keyed sets — one for the integer
literals, one for the string literals. An exact `int` or `str` value is then a
single set lookup rather than a scan of every branch, so membership cost stops
growing with the number of literals. The same-type literal rule is preserved:
the integer set is consulted only for an exact `int` (never a `bool`), the string
set only for an exact `str`, and any other value — a `bool`, `float`, `None`, a
subclass instance, a big integer, or a JSON value — falls back to the linear scan
that remains the single source of truth. On a 32-literal union this cuts the
per-call median several-fold; the decision is identical to the scan, locked by
tests over the cross-type cases.

## Regression gate

The wall-clock numbers above are for humans reading results; they are too noisy
on shared CI runners to gate a merge. The merge gate is instead a deterministic
instruction count: a fixed workload exercises the core schema operations
(`crates/valgebra-core/examples/perf_workload.rs`), runs under cachegrind, and
its executed-instruction count is compared against a committed budget
(`scripts/perf_budget.json`) by `scripts/perf_gate.py`. The count is identical
across runs of a given build, so a regression past the budget ceiling fails the
build without flaking. The tolerance absorbs cross-environment startup and
compiler-codegen drift while still catching algorithmic regressions, which are
far larger than the tolerance.

The gate covers the pure-Rust schema engine, which is portable enough for a
committed budget. The end-to-end wall-clock suites run on the same CI lane with
timing disabled, as a smoke test that they keep working.

The headline claim — that valgebra is pydantic-core-class — is gated too, by
`scripts/compare_gate.py`. For each shape in a matrix it measures the *ratio* of
per-call time (valgebra over pydantic-core), taking the minimum over many repeats,
and compares each ratio against a recorded baseline (`scripts/perf_compare.json`)
with a tolerance. A ratio cancels the runner's absolute speed: if the machine is
slow, both libraries are slow in proportion, so the comparison survives the
shared-runner noise an absolute budget cannot. A shape fails the merge gate when
valgebra's ratio rises materially past its baseline — a competitive regression,
whether from valgebra slowing down or ceding ground.

Re-record the budgets after an intentional change with:

```bash
python scripts/perf_gate.py --update            # core instruction budget
python scripts/compare_gate.py --update         # competitive ratios
```
