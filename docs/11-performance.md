---
description: Measured benchmarks, datasets, and compared versions.
---

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
mobile part) under WSL2 on Linux 6.18. Toolchain: rustc 1.98.0 (the build these
numbers are measured on; the supported minimum is the lower `rust-version` in
the manifest), CPython 3.14.7 built from source with
`--enable-optimizations --with-lto`, `CC=clang` and
`CFLAGS=-march=native -mtune=native`, and **the GIL enabled**
(`sysconfig.get_config_var("Py_GIL_DISABLED")` is `0`; the free-threaded build
of the same version runs this work about twice as slow, so a figure measured on
one is not comparable with the other), criterion 0.8.2 and pytest-benchmark
5.2.3. The comparison packages are whichever versions the bench group resolves:
`uv.lock` records them and `scripts/compare_gate.py` prints them beside the
figures, so a version written here could only go stale.

`sysconfig.get_config_var("CONFIG_ARGS")` reports that build on the machine
these figures come from. The native tuning is the flag that matters when
reproducing them: a stock distribution interpreter is a different binary, so a
figure measured against one is not comparable with a figure here.

The extension is the **PGO** release build — the profile-guided, fat-LTO wheel
the release ships:

```bash
uv run maturin build --release --pgo -i .venv/bin/python
```

pydantic's PyPI wheels are likewise PGO-built, so this is a release-to-release
comparison.

**Build with PGO if you build your own wheel**, and read no figure taken from a
debug build as either. How much PGO adds over a plain `--release` build is not a
constant this page can state. It is whatever the profile can still arrange that
fat LTO did not, so it shrinks as the hot paths themselves get shorter: measured
on one machine it has ranged from 1.75x down to 1.01x, and the shapes where the
release build is already tightest are the ones it buys least on. If the number matters to you, measure it on your own build:
`scripts/compare_gate.py` against each wheel is the way.

The figures are measured on the wheel carrying valgebra's full feature set — the
per-validator precompute (record-field lookups, literal-union dispatch) and
native string patterns — which leaves these shapes unchanged: the features earn
their keep elsewhere, not by regressing the core.

### Method

Each cell is the **median of five independent runs** of the comparison
benchmark, each run reporting pytest-benchmark's own median over its rounds. The
`+/-` is the half-range across the five runs, not a standard deviation: it states
the observed spread rather than modelling one.

```bash
uv run --group bench pytest benches/bench_compare.py --benchmark-json=run.json
```

The **ratios** have their own gate, which measures only valgebra against
pydantic and takes the minimum over many repeats rather than a median:

```bash
uv run --group bench python scripts/compare_gate.py
```

That script owns the per-shape ratio **ceilings** (`scripts/perf_compare.json`)
-- what the project claims it stays under rather than what it once measured --
and the table below is the absolute record. The two estimators
do not agree to the last digit — a minimum sits below a median by however much
the run was disturbed — so read a cell here against the same cell, not against
the gate's output.

### The cheapest door, and where the floor is

`x in v` is `v.is_valid(x)` through the container protocol, and it is the
cheaper call: **30 ns against 40 ns** for a scalar on the machine below, because
the interpreter reaches a container slot directly and a method by its call
protocol. Neither number is the check. `Validator(anything).is_valid(1)` -- the
schema that answers `True` without looking -- costs 38 ns, so the *walk* for an
`int` is about 4 ns and everything else is the boundary a Python call crosses.
For reference on the same run, `isinstance(1, int)` is 21 ns and an empty Python
function call is 29 ns.

Read that as the floor it is: a per-call check cannot be much cheaper than a
Python call, and the way to spend less is to make fewer calls -- validate the
list, not each element -- rather than to look for a faster scalar.

### Results

End-to-end validation of a value that passes (lower is better):

| Shape | valgebra | pydantic (strict) | jsonschema |
| --- | --- | --- | --- |
| `list[int]`, 10,000 elements | 9.76 +/- 0.15 us | 78.4 +/- 0.75 us | 25,361 +/- 555 us |
| Closed record, 50 int fields | 0.704 +/- 0.010 us | 1.90 +/- 0.087 us | 130 +/- 4.3 us |
| Nested `list[...]`, depth 25 | 0.201 +/- 0.021 us | 1.97 +/- 0.031 us | 75.1 +/- 2.0 us |

valgebra relative to pydantic on this machine, under the CPython 3.14 the matrix
above names: **9.8x** faster on deep nesting, **8.0x** on the large flat array,
**2.7x** on the wide record. It is consistently far ahead of pure-Python
jsonschema — 2,600x on the array, 374x on the nesting and 185x on the record.
pydantic does strictly more work on the record (it constructs output), so read
that shape as a margin over a heavier operation, not a like-for-like loss for
pydantic.

### One of those margins moves with the interpreter

A ratio cancels the machine — a slower box slows both sides — and it does not
cancel the interpreter. Running the comparison gate on one box against three of
them, as the fraction of pydantic's time each shape takes:

| Shape | CPython 3.12 | CPython 3.14 | 3.14 free-threaded |
| --- | --- | --- | --- |
| `list[int]`, 10,000 elements | 0.191 | 0.151 | 0.159 |
| Closed record, 50 int fields | 0.341 | 0.349 | 0.341 |
| Nested `list[...]`, depth 25 | 0.158 | 0.138 | 0.332 |
| One `int` | 0.227 | 0.215 | 0.233 |

The **element** is what moves, not the check. A list hands out each of its items
as an owned reference — a count written on the object when the handle is made
and again when it drops — and the free-threaded build takes the list's lock for
each one besides. A schema nested twenty-five deep is twenty-five containers of
one element, so it is almost nothing but that cost, and it reads two and a half
times dearer there than under a global lock. A flat array of ten thousand is
read through a snapshot of the list instead, which pays the counts in two loops
inside the interpreter and none in the walk, and it carries across all three.

That is why the ceiling file holds a second set for the free-threaded build:
what the project claims of that build is what that build can hold.

### The two shapes this page did not show

The table above is the four shapes valgebra wins by a wide margin, and the
competitive gate measures seven. The two it leaves out are the two closest, and
leaving them out made the page a selection rather than a record. As the fraction
of pydantic-core's time each takes, on a PGO CPython 3.12 build:

| Shape | ratio | spread across runs | ceiling |
| --- | --- | --- | --- |
| JSON document, 200 records parsed and checked | 0.70 | 0.057 over twelve runs | 1.00 |
| Error report, 50-field record with one wrong field | 0.86 to 1.19 | 0.33 over five runs | 1.60 |

The JSON document is a single pass over bytes for both libraries, which is why
the margin is a third rather than a factor: neither is spending its time in the
check. A document's free-form sections are `dict[str, V]`, and covering their
keys is read two ways -- in place for a narrow object, through a table of last
values for a wide one ([dev/04-walk.md](dev/04-walk.md)).

The error report is the one shape where valgebra is sometimes *slower*, and the
one whose measurement is not trustworthy: a third of its own value in spread,
against 0.001 to 0.057 for every other shape. It is the only shape timing a path
that raises and formats a Python exception, so a Python exception's cost is
inside the number. Two things follow. It is excluded from the gate's drift
ratchet, which says so in `scripts/perf_compare.json` rather than by having no
entry. And a failing validation walks the value **twice** here -- once to decide,
once to say which field -- where pydantic-core walks it once and collects as it
goes. That is a deliberate trade for the passing path, which is the common one
and which walks once; it is not a margin anybody has shown how to recover, and
one attempt made it twenty times worse.

The scalar shape is absent from the table because it sits near timer resolution:
the competitive gate measures it at a 32.1 ns median with a spread reaching a
fifteenth of that, and the ratio it reports — around 5.8x — carries noise the
other three shapes do not. That gate measures it; this record does not.

Core micro-benchmarks (criterion, release+LTO, indicative single run):

| Operation | Corpus | Median |
| --- | --- | --- |
| `simplify` | redundant Boolean expression, depth 8 | ~1.1 us |
| `shifted` | 64-field pool-indexed record | ~2.0 us |
| `with_records_open` | record spine, depth 32 | ~4.6 us |

## Honest limits

- The numbers are a single machine class. They establish relative behavior, not
  a universal ranking. Shared CI runners are too noisy for a tight wall-clock
  budget, so the merge gate measures a deterministic instruction count instead.
- The margins against pydantic come with the caveat that the two tools do
  different work: pydantic constructs output, valgebra only checks membership.
  The ratios answer "how fast is each tool's validation step," not "how much
  faster is membership than construction." Deep nesting is the widest gap; the
  array and record margins are narrower but consistent.
- **The ratios are not the whole argument for using valgebra.** They are real,
  and they are also a regression gate. But the reasons to reach for a membership
  check over a parser are additionally semantic — an already-held object is
  re-examined rather than passed through, a value stays checkable after it is
  mutated, and schemas can be compared as sets — and every one of those would
  hold even at a ratio of 1.0. `tests/test_pydantic_boundary.py` pins them as
  verdicts, with no timer involved.
- The comparison measures different operations (check vs check-and-construct vs
  pure-Python check). It answers "how fast is the validation step for each
  tool," not "are these tools interchangeable" — they are not. See the README
  for what valgebra is and is not for.
- These figures are for the object path — validating a value already in hand.
  The JSON input path is measured separately, on the same machine class, in the
  [JSON page](07-json.md).

## How the record fast path is tuned

The closed-record membership check visits each dict entry once and matches the
key against the declared fields, rather than looking up every declared field in
turn (which builds a temporary Python string per field) and then scanning the
dict a second time for undeclared keys. The key's UTF-8 is borrowed without
allocating, and the field-name index is computed once when the validator is
first used — with a fast non-cryptographic hasher, since the keys are the
schema's own declared names rather than attacker input — then reused across
calls, so a wide record does not rebuild or reallocate its name map on every
validation. The `wide_record` row of the table above is what it costs. Profiling
with cachegrind attributed the removed cost to temporary-string creation,
hashing, and allocation churn from the per-field lookups, and that attribution
is an instruction count, so it holds across machine classes. The bool fast path
and the aggregating explain walk stay membership-equivalent, locked by tests
that assert both reach the same verdict across record shapes.

A **report** on that record -- one field wrong, explained, raised -- costs about
three accepting walks, and the count attributes the three. Two are the walks:
the fast pass, which stops at the field that fails, and the explaining pass,
which reads every field so the report names all of them, a hundred dict probes
between them. The third is the raise itself, which the interpreter charges for
building the exception and unwinding to the caller. Nothing in the three is a
walk over what the schema already knows, so a further cut would be a cheaper
report rather than a shorter walk -- and the shape the same count *did* find
was the open record, read a third dearer than the closed one until it was read
by its keys.

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

## What a relation between two validators costs

`is_subtype_of`, `relation_to` and `is_equivalent` are not the membership walk,
and they have a cost of their own with two clear levels. A pair a **rule**
decides costs about a microsecond: the rules recurse over the two schemas and
answer from their shapes. A pair the rules decline goes to the **set
representation**, which lowers both sides into automata and takes tens to
hundreds of microseconds. Two orders of magnitude is the gap, and it is the
reason a rule that stops declining is worth writing; `cargo bench --bench core`
measures both sides, the `subtype_*` rows against the `lower_*` rows.

Which pairs land on which side is the interesting part, and it moves. Most do
not reach the sets: a mismatch of kinds, a record missing a required key, a
sequence of the wrong arity, a subject outside a refinement's base. What still
reaches them is a schema whose atom only the bindings can read on one side and
a structure on the other -- a class deriving from no builtin, a complement as
the subject -- and a bound that has to be *compared* rather than matched, such
as a list against a length-bounded list of the same element.

There is no gate on this, and that is deliberate: the instruction budgets cover
the three decision workloads -- the relations that hold, the ones a rule
refutes, and the ones whose goals repeat, which `crates/valgebra-core/examples/`
holds and which are what a change to the rules moves. A relation's wall-clock cost is a property of the
pair, and pinning one would be pinning a number the next rule changes.

## Regression gate

The wall-clock numbers above are for humans reading results; they are too noisy
on shared CI runners to gate a merge. The merge gate is instead a deterministic
instruction count: fixed workloads run under cachegrind, and each
executed-instruction count is compared against a committed budget
(`scripts/perf_budget.json`) by `scripts/perf_gate.py`. The count is identical
across runs of a given build, so a regression past the budget ceiling fails the
build without flaking. The tolerance absorbs cross-environment startup and
compiler-codegen drift while still catching algorithmic regressions, which are
far larger than the tolerance.

The gate holds one workload per surface, because a gate only catches what it
exercises. `crates/valgebra-core/examples/` holds the pure-Rust ones: the schema
transformations (`perf_workload`), and three over the decision procedures -- the
relations that hold, the relations that are refuted, and the relations whose
goals repeat, since a proof, a refutation and a repeated goal walk three
different paths and a workload that asks only one of them measures only that
one. The binding's shapes are the membership walk over a live value, the call
boundary alone, a wide record closed and the same record open the way a
`TypedDict` is, building a validator, and explaining a failure
(`crates/valgebra-py/examples/binding_workload.rs`): the walk is the shipped hot
path neither pure-Rust workload reaches, schema construction grew twelve percent
over a release cycle while only the walk was counted, and an open record was
read a third dearer than a closed one while only the closed one was. Each binding shape
embeds CPython, whose startup is not a fixed count, so the gate measures the
difference between two iteration counts.

One thing an embedded interpreter brings with it is its **string hash seed**,
drawn per process; a shape that probes a dict of string keys executes a
different number of instructions under every seed, and on the difference of two
counts that is a few percent -- the size of the ceiling. The gate fixes the seed
where every measurement passes through, so a reading is of the code and not of
the interpreter's draw. A number in this page is a wall-clock figure for a
human; the counts belong to the budget file and are not repeated here.

The end-to-end wall-clock suites run on the same CI lane with timing disabled,
as a smoke test that they keep working.

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
python scripts/perf_gate.py --update            # the core budget; --decision, --binding-record, ... for the others
python scripts/compare_gate.py --update         # competitive ratios
```
