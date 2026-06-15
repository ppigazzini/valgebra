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
uv run --group bench maturin develop --uv
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
extension is the release build (`maturin develop --release`, fat LTO — the same
profile as the shipped wheel); a debug build is not representative. Figures are
the per-call median; re-run on your own hardware for absolute numbers.

End-to-end validation of a value that passes (median time per call, lower is
better):

| Shape | valgebra | pydantic (strict) | jsonschema |
| --- | --- | --- | --- |
| `list[int]`, 10,000 elements | 82 us | 88 us | 26,300 us |
| Closed record, 50 int fields | 2.2 us | 2.0 us | 135 us |
| Nested `list[...]`, depth 25 | 0.39 us | 1.95 us | 78.9 us |

valgebra relative to pydantic on this machine: ~5x faster on deep nesting,
roughly tied on the large flat array (~1.07x), and slightly behind on the wide
record (pydantic ~1.13x faster). It is consistently far ahead of pure-Python
jsonschema — by two orders of magnitude on every shape. The wide-record and
flat-array margins are within machine-to-machine swing, and pydantic does more
work there (it constructs output), so read those shapes as "comparable," not a
result either way.

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
- The wide-record and flat-array margins against pydantic are within
  machine-to-machine swing, and the two tools do different work (pydantic
  constructs output). Read those shapes as comparable, not a win or loss; a
  small swing could put either ahead. The deep-nesting margin is the one
  durable, large gap.
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
allocating. On the 50-field record above this single pass measures ~2.2 us per
call (release build); the earlier per-field-lookup form was roughly 2.7x slower.
Profiling with cachegrind attributed the removed cost to temporary-string
creation, hashing, and allocation churn from the per-field lookups, and that
attribution is an instruction count, so it holds across machine classes. The
bool fast path and the aggregating explain walk stay membership-equivalent,
locked by tests that assert both reach the same verdict across record shapes.

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
timing disabled, as a smoke test that they keep working; their numbers are
re-recorded by hand in the matrix above rather than gated.

Re-record the budget after an intentional workload change with:

```bash
python scripts/perf_gate.py --update
```
