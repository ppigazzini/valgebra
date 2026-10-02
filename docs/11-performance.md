---
description: Measured benchmarks, datasets, and compared versions.
---

# Performance

valgebra compiles a schema once into a Rust validator tree and crosses into Rust
exactly once per validation call. This page records how that is measured, a
reproducible baseline against other validators, the honest limits of the
numbers, and the techniques behind them.

A speed claim is only as good as its methodology. Every number here states the
harness, the dataset, the library versions, and the machine class. Re-run the
harnesses on your own hardware before relying on a ratio: absolute times move
with the CPU, and the comparison points do different amounts of work.

## What is measured

Two harnesses, one per side of the boundary:

- **Core micro-benchmarks** (`crates/valgebra-core/benches/core.rs`, criterion)
  time the pure-Rust schema transformations — the simplifier, the index remap
  behind validator composition, and the recursive open/closed record transform —
  and the relations, a rule's answer (`subtype_*`, `is_empty_*`) beside the set
  representation's (`lower_*`). No Python is involved.
- **End-to-end benchmarks** (`benches/`, pytest-benchmark) time a single
  boundary-crossing validation call through the public API, over synthetic
  shapes that each stress one cost dimension.

Run them with:

```bash
# Core micro-benchmarks (Rust):
cargo bench --bench core

# End-to-end and comparison benchmarks (Python); install the bench group first,
# without the project, which the wheel below provides:
uv sync --locked --no-install-project --group bench
# To match the published figures, build the same PGO wheel the release ships
# (needs the llvm-tools rustup component) and install it; a plain build is slower.
# Every `uv run` from here on takes --no-sync: a sync reinstalls the project over
# the wheel, and the run then times a different build from the one installed.
uv run --no-sync --group bench maturin build --release --pgo --out dist
uv pip install --no-deps --reinstall dist/*.whl
uv run --no-sync --group bench pytest benches/bench_validate.py
uv run --no-sync --group bench pytest benches/bench_compare.py --benchmark-group-by=group
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
mobile part) under WSL2 on Linux 6.18. Toolchain: rustc 1.98.1 (the build these
numbers are measured on; the supported minimum is the lower `rust-version` in
the manifest), CPython 3.14.7 built from source with
`--enable-optimizations --with-lto`, `CC=clang` and
`CFLAGS=-march=native -mtune=native`, and **the GIL enabled**
(`sysconfig.get_config_var("Py_GIL_DISABLED")` is `0`; the free-threaded build
of the same version runs this work about twice as slow, so a figure measured on
one is not comparable with the other). Every package version -- the two
benchmark harnesses and the libraries compared against -- is whichever the bench
group resolves. `uv.lock` owns them and `scripts/compare_gate.py` prints them
beside the figures, so none is written here: a version copied into prose is
stale the next time the lock moves, and a reader cannot tell a stale one from a
current one.

`sysconfig.get_config_var("CONFIG_ARGS")` reports that build, and is how to
check you are on it. The native tuning is the flag that matters when reproducing
these numbers: a stock distribution interpreter is a different binary, and so is
the one `uv sync` provisions for this repository, which is a
python-build-standalone image rather than a source build. Point the bench
environment at the interpreter you mean -- the sync rebuilds `.venv` on it --

```bash
uv sync --locked --no-install-project --group bench --python /path/to/that/python
uv run --no-sync --group bench maturin build --release --pgo --out dist
uv pip install --no-deps --reinstall dist/*.whl
uv run --no-sync --group bench python scripts/compare_gate.py
```

-- because a figure measured against another binary is not comparable with a
figure here.

The extension is the **PGO** release build — the profile-guided, fat-LTO wheel
the release ships, which the `maturin build --release --pgo` step above builds.

pydantic's PyPI wheels are likewise PGO-built, so this is a release-to-release
comparison.

**Read no figure taken from a debug build as either** -- `maturin develop`
without `--release` installs one, it is indistinguishable from the release
extension at the Python prompt, and a timing of it reads an order of magnitude
slow. `scripts/compare_gate.py` refuses such a build outright, and so does
`benches/` from its own `conftest.py`; a figure timed by hand has only the habit
to protect it.

**Whether to add PGO is a question about your shapes**, not a setting to turn
on. Build both, install each in turn, and time the shapes you run on it -- the
sequence the `pgo compare` lane in `.github/workflows/ci.yml` runs:

```bash
uv sync --locked --no-install-project --group bench
uv run --no-sync --group bench maturin build --release --out plain
uv run --no-sync --group bench maturin build --release --pgo --out profiled
uv pip install --no-deps --reinstall plain/*.whl
uv run --no-sync --group bench python scripts/pgo_compare.py \
  --record plain.json --label plain
uv pip install --no-deps --reinstall profiled/*.whl
uv run --no-sync --group bench python scripts/pgo_compare.py \
  --record profiled.json --label pgo
uv run --no-sync --group bench python scripts/pgo_compare.py \
  --compare plain.json profiled.json
```

That script times every shape the competitive gate judges -- it reads them from
`scripts/compare_gate.py` -- so the two tables speak of the same workloads. It
refuses a reading from a debug build and refuses to compare two readings from
different interpreters, because a ratio between builds cancels the machine and
not the interpreter.

Measured that way on one box, best of nine and repeated three times, the
direction is not one way. A `list[int]` of ten thousand comes out ahead by about
a third and the JSON document by about a fifth; the accepting walk over a
fifty-field record comes out **behind**, by about a fifth. An instruction count
agrees with the wall clock, and re-weighting the training workload toward
accepting values recovers only a part of it.

The split is not container against scalar, and it is worth saying so because the
obvious reading is the wrong one: the ten-thousand element list runs a
container's whole element loop and is the shape PGO serves *best*. What the two
winners share is one hot loop over one element type, which is what a profile can
lay out straight. The record walk is the other shape: fifty key lookups, each
dispatching on the field's own schema, so the profile has many warm paths and no
hot one, and laying them out costs the branches it does not predict.

**The reading the release matrix is decided on is taken where the release
builds.** The `pgo compare` lane builds both wheels from one source, times every
shape on each, and uploads the two readings, on both interpreters the bench jobs
use -- a global lock changes what a per-element loop costs, and the shapes a
profile serves best are the per-element ones:

```bash
gh workflow run CI -f pgo_compare=true
```

**The training workload is the lever**, not the flag: the profile is taken over
`scripts/pgo_workload.py`, which `pyproject.toml`'s `pgo-command` names, and
what that workload spends its time on is what the layout is arranged for. A
change to it is measured on both sides of the comparison above, since a profile
re-weighted toward one shape is a profile taken away from another.

**A loop the profile holds no counts for is laid out by chance.** A homogeneous
list takes a loop of its own, so a workload reading only homogeneous lists holds
no counts for the general scan every list of records and every nested list is
read through, and whether that scan stays inside the recursive walk is the
inliner's guess. A change to the list arm's code can turn it: untrained on the
scan, the comparison gate's list nested twenty-five deep takes 64% longer in the
PGO wheel, 405 ns against 245 on one box, while the instruction gate, which
builds without a profile, reads 0.9%. So the workload reads a list of records
and a list nested twenty-five deep, and trained on them that shape reads 0.12 of
pydantic-core's time on CPython 3.14.

The same holds for a reader the workload never enters: the lists a reader of
their own settles -- a union of literals, one class, a union of scalars -- sit
beside the general scan, and untrained, the readers for literals and classes
move the PGO wheel's walk of a nested list by 6% in instructions. The workload
reads a list of each kind, which takes that to 3%.

**What the matrix does with the reading.** `--pgo` ships on five targets, and
that is a claim about those boxes rather than a default. Where the lane reads a
gain on the shapes the release serves, the matrix keeps `pgo: true` and this
page names the shapes it costs. Where it reads the record walk slower on both
interpreters -- the shape most callers spend their time in -- `pgo: true` leaves
the matrix and this page says so. Either way the decision is the reading's, and
the reading is the lane's.

That is one microarchitecture and one interpreter, and the release lane builds
on five targets none of which is this one, so it is a reason to measure your own
build rather than a ranking. `scripts/compare_gate.py` against each wheel is how.

**Every wall-clock figure on this page, and in the changelog, is a release
build on an idle machine, read over repeated runs -- a median of five for a
table, the comparison gate's minimum for a ratio -- and taken twice.** The
two guards above hold the first of those and neither holds the other two: a
timing taken while something else has the CPU reads slow, and one taken once
reads whatever that run did. Either is enough to put a published figure out by a
factor, so the discipline is written down rather than assumed.

How much PGO adds over a plain `--release` build is not a constant this page can
state, and on some shapes it is not a gain at all. It is whatever the profile can
still arrange that fat LTO did not, so it shrinks as the hot paths get shorter,
and where the layout it picks suits one shape it can cost another.

The figures are measured on the wheel carrying valgebra's full feature set — the
per-validator precompute (record-field lookups, literal-union dispatch) and
native string patterns — which leaves these shapes unchanged: the features earn
their keep elsewhere, not by regressing the core.

### Method

Each cell is the **median of five independent runs** of the comparison
benchmark, each run reporting pytest-benchmark's own median over its rounds. The
`+/-` is the half-range across the five runs, not a standard deviation: it states
the observed spread rather than modelling one. With the release wheel installed
as above:

```bash
uv run --no-sync --group bench pytest benches/bench_compare.py --benchmark-json=run.json
```

The **ratios** have their own gate, which measures only valgebra against
pydantic and takes the minimum over many repeats rather than a median:

```bash
uv run --no-sync --group bench python scripts/compare_gate.py
```

That script owns the per-shape ratio **ceilings** (`scripts/perf_compare.json`)
-- what the project claims it stays under rather than what it once measured --
and the table below is the absolute record. The two estimators can part by a
quarter of a multiplier — a minimum sits below a median by however much the run
was disturbed, and the 3.14 results below read both — so read a cell here
against the same cell, not against the gate's output.

### The cheapest door, and where the floor is

`x in v` is `v.is_valid(x)` through the container protocol, and it is the
cheaper call: **20 ns against 24 ns** for a scalar on the machine below, the
best of fifteen `timeit` repeats of a million calls, because the interpreter
reaches a container slot directly and a method by its call protocol. Neither
number is the check. `Validator(anything).is_valid(1)` -- the schema that
answers `True` without looking -- costs 23 ns, so the *walk* for an `int` is
under a nanosecond and everything else is the boundary a Python call crosses.
For reference on the same run, `isinstance(1, int)` is 12 ns and a call to an
empty one-argument Python function is 15 ns.

Read that as the floor it is: a per-call check cannot be much cheaper than a
Python call, and the way to spend less is to make fewer calls -- validate the
list, not each element -- rather than to look for a faster scalar.

### Results

End-to-end validation of a value that passes (lower is better):

| Shape | valgebra | pydantic (strict) | jsonschema |
| --- | --- | --- | --- |
| `list[int]`, 10,000 elements | 9.72 +/- 0.33 us | 77.2 +/- 1.9 us | 24,650 +/- 397 us |
| Closed record, 50 int fields | 0.518 +/- 0.015 us | 1.92 +/- 0.055 us | 128 +/- 2.7 us |
| Nested `list[...]`, depth 25 | 0.217 +/- 0.014 us | 1.95 +/- 0.049 us | 74.7 +/- 1.3 us |

valgebra relative to pydantic on this machine, under the CPython 3.14 the matrix
above names, as the medians above divide: **9.0x** faster on deep nesting,
**7.9x** on the large flat array, **3.7x** on the wide record. The comparison
gate's minimums read the same three at 8.2x, 8.0x and 3.9x (the 3.14 column
below). It is consistently far ahead of pure-Python jsonschema — 2,500x on the
array, 344x on the nesting and 248x on the record.
pydantic does strictly more work on the record (it constructs output), so read
that shape as a margin over a heavier operation, not a like-for-like loss for
pydantic.

### One of those margins moves with the interpreter

A ratio cancels the machine — a slower box slows both sides — and it does not
cancel the interpreter. Running the comparison gate on one box against three of
them -- CPython 3.12.14 and the free-threaded 3.14.6 as `uv` installs them,
beside the 3.14.7 build named above -- as the fraction of pydantic's time each
shape takes, the median over twelve runs on 3.12 and at least five on each 3.14
build:

| Shape | CPython 3.12 | CPython 3.14 | 3.14 free-threaded |
| --- | --- | --- | --- |
| `list[int]`, 10,000 elements | 0.177 | 0.125 | 0.137 |
| Closed record, 50 int fields | 0.239 | 0.255 | 0.282 |
| Nested `list[...]`, depth 25 | 0.132 | 0.122 | 0.274 |
| One `int` | 0.207 | 0.192 | 0.199 |

The **element** is what moves, not the check. A list hands out each of its items
as an owned reference — a count written on the object when the handle is made
and again when it drops — and the free-threaded build takes the list's lock for
each one besides. A schema nested twenty-five deep is twenty-five containers of
one element, so it is almost nothing but that cost, and it reads more than twice
as dear there as under a global lock. A flat array of ten thousand is read through
a snapshot of the list under 3.12 and the free-threaded build, which pays the
counts in two loops inside the interpreter and none in the walk;
CPython 3.14 with its global lock makes the counts cheap enough that the walk
reads the list in place (`snapshot_pays` in
`crates/valgebra-py/src/check/walk/sequence.rs`). The margin carries across all
three.

That is why the ceiling file holds a second set for the free-threaded build:
what the project claims of that build is what that build can hold.

### The closest races

The table above holds four of the shapes the competitive gate judges;
`scripts/perf_compare.json` names every one with the ceiling the project claims
for it, and `scripts/compare_gate.py` prints each beside its ceiling. A record of
only the wide margins would be a selection, so here are the two closest, as the
fraction of pydantic-core's time each takes, on a PGO CPython 3.12 build:

| Shape | ratio | spread across runs |
| --- | --- | --- |
| JSON document, 200 records parsed and checked | 0.61 | 0.035 over twelve runs |
| Error report, 50-field record with one wrong field | 0.46 | 0.042 over twelve runs |

The JSON document is a single pass over bytes for both libraries, which is why
valgebra takes three fifths of pydantic-core's time here rather than a fraction:
neither is spending its time in the check. A document's free-form sections are
`dict[str, V]`, and covering their keys is read two ways -- in place for a
narrow object, through a table of last values for a wide one
([dev/04-walk.md](dev/04-walk.md)).

The error report spreads 0.43 to 0.47 across the twelve runs, beside the JSON
document's 0.035 and the scalar's 0.061, which sits near timer resolution;
the array, the record, the nesting and the build spread between 0.002 and
0.025.
It is the only shape timing a path that raises and formats a Python exception,
so a Python exception's cost is inside the number. It is excluded from the
gate's drift ratchet, and `scripts/perf_compare.json` gives the reason rather
than leaving an absent entry to mean it: a spread read on another build. On
this one the shape spreads as the JSON document does, which the ratchet holds.

A failing validation walks the value **twice** here -- once to decide, once to
say which field -- where pydantic-core walks it once and collects as it goes.
That is a deliberate trade for the passing path, which is the common one and
which walks once. The second walk does not repeat the first: it resumes where
the deciding walk stopped, since a field that matched has no violation to report
([dev/04-walk.md](dev/04-walk.md)). Resuming rather than restarting is worth
36.8% of that walk on the instruction gate, which
`scripts/perf_gate.py --binding-explain` measures. What the report pays beyond
the deciding walk is one walk of the fields *after* the failure, and the
exception.

The scalar shape is absent from the table because it sits near timer resolution,
and most of it is the call rather than the check, as the floor above shows: on
CPython 3.14 the competitive gate measures it at a 36.7 ns median over seven
runs with a spread of 1.5 ns, around 5.2x. That gate measures it; this record
does not.

Core micro-benchmarks (criterion, release+LTO, indicative single run):

| Operation | Corpus | Median |
| --- | --- | --- |
| `simplify` | redundant Boolean expression, depth 8 | ~1.9 us |
| `shifted` | 64-field pool-indexed record | ~1.2 us |
| `with_records_open` | record spine, depth 32 | ~4.4 us |

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

## How the speed is made

Every technique below changes how an answer is reached and never the answer.
Where a cheap reading stands beside the general one, a test asks both and holds
them to one verdict, so a rule changed in one and not the other fails. Where
only the cost moves, no test can tell the readings apart, and a shape of the
instruction gate holds the cost instead: `scripts/perf_gate.py --against <rev>`
measures it against a base, and the flag a figure below names is the shape that
reproduces it. A percentage compares a technique with the general reading it
stands beside, which stays in the tree as the fallback.

Each entry names the code that owns it. The developer pages carry the arguments
in full: [the walk](dev/04-walk.md), [the frontend](dev/03-frontend.md), [the
relations](dev/02-decision.md), [the error model](dev/05-errors.md) and [the
schema representation](dev/01-schema-ir.md).

### Crossing into Rust

- **One crossing per call.** `is_valid`, `validate` and `is_valid_json` enter
  the extension once and walk the whole value in Rust. Python runs during a
  walk only where the schema or the value brings code of its own: a predicate,
  a class whose metaclass computes `isinstance`, a property read as a field, a
  value's own `__eq__`. What the crossing itself costs
  is the floor [the cheapest door](#the-cheapest-door-and-where-the-floor-is)
  measures.
- **No reference pool.** PyO3 keeps a pool of reference-count decrements
  deferred while a thread is detached, and once any of its lazy initialisers
  has detached and attached again -- the first interned name does -- every call
  into the extension locks that pool's mutex to ask it for them: forty
  instructions and eight branches on `Validator(int).is_valid(1)`. The binding
  never detaches, so the pool is always empty, and `.cargo/config.toml`
  compiles it out; `tests/test_build_flags.py` holds the flags there. The
  instruction gate's boundary shape calls the walk without crossing PyO3's call
  machinery, so the comparison gate's `scalar` is the instrument that reads it:
  36.7 ns with the pool gone against 39.9 with it, on CPython 3.14 and the
  machine named above.
- **A failure is raised in one step.** `into_pyerr` in
  `crates/valgebra-py/src/errors.rs` builds the `ValidationError` instance and
  hands it to PyO3 as the error's value, so the raise is one
  `PyErr_SetObject`. A lazily built error is normalised by releasing the
  interpreter lock and taking it back, which lets another thread run in the
  middle of a raise and costs a failing `validate` of a fifty-field record 5.4%
  more instructions. The exception being handled is chained as `__context__`
  either way, and a test holds that on every interpreter the suite runs.
- **Every fixed name is one the interpreter already holds.** An attribute, key
  or method name crosses as an interned string (`intern!`), which carries its
  hash; a name handed over as Rust text is decoded into a new string and hashed
  on every lookup. An attribute that may be absent is asked for with
  `getattr_opt`, which answers without building an exception wherever the
  interpreter offers a lookup that does (`PyObject_GetOptionalAttr`, from 3.13).
- **Internal maps hash with FxHash.** A validator's own maps are keyed by its
  field names and by object identities, never by a caller's data, so they use
  `rustc-hash` rather than the standard hasher and its per-map random seed.

### Building a validator

Compiling happens once per schema, so these are startup costs. They matter to a
program that builds validators per request, and to no validation call.

- **A type is dispatched before any `typing` introspection**, and an exact
  `bool`, `int`, `float`, `str` or `bytes` constant, or a `Validator`, is read
  before the frontend's dispatch (`builtin_constant` in
  `crates/valgebra-py/src/build.rs`). A `Literal` of two thousand string codes
  compiles in about three million instructions; read through the whole dispatch
  one constant at a time, it costs four times that. The pool indexes its
  constants by type and value or by address (`Pool::intern_keyed`), so interning
  one is a lookup rather than a scan.
- **The `typing` forms are resolved once per interpreter** (`forms` in
  `build.rs`), and `dataclasses` is imported on the first dataclass a program
  compiles (`IS_DATACLASS` in `build/classes.rs`). Held among the forms, its
  import would leave tracked objects behind that lengthen every later
  garbage-collection pass, and a build that compiles no dataclass would read
  6.45% dearer on `--binding-build`.
- **A class's annotations are read as written** wherever evaluating them would
  hand each one back unchanged (`annotations_as_written` in `build/classes.rs`).
  `typing.get_type_hints` exists to evaluate forward references, and on a class
  with none it still copies every base's namespace and walks every annotation
  in Python. The reading takes the call's steps and declines every case the
  call would rebuild -- a string, a `ForwardRef`, a nesting deeper than
  `MAX_ANNOTATION_DEPTH` -- so an answer or an error a caller sees is the
  call's. A Rust interpreter test compares the two over a corpus of classes on
  every supported interpreter.
- **A dataclass's fields are read as `dataclasses.fields` reads them**
  (`fields_as_declared`), and an order bound of exactly `int`, `float` or `bool`
  is placed without asking the `numbers.Number` ABC, whose `__instancecheck__`
  is a Python function (`is_a_number` in `build/refine.rs`). A subclass is still
  asked, because the ABC reads a value's `__class__`, which a subclass may
  answer with code of its own.
- **A refinement marker is read by its type, once.** A marker carries one or
  two of ten optional attributes and not the rest, and below 3.13 asking for one
  it does not have is answered by raising. So the frontend asks the marker's
  *type* which names it carries and remembers the answer, up to
  `MAX_MARKER_TYPES` types, which costs no exceptions on any interpreter: fifty
  `Annotated[int, Ge(0)]` fields compile in about 30 us on CPython 3.14 and 27
  on 3.12, on the machine named above.
- **A protocol reads its classes' namespaces once**, and a member annotated
  with a class `type` made is neither `ClassVar` nor `Final` without calling
  `typing.get_origin`, a Python function (`protocol_members` in
  `build/classes.rs`; `--binding-protocol`).
- **A validator is accepted on one walk of each tree.** `Schema::measure` in
  `crates/valgebra-core/src/ir.rs` reads the nesting depth, the node count and
  the self-reference marker in one level-by-level walk against two buffers, and
  descends no native stack.
- **What the walk needs is built once per validator.** `ValidatorIndex` in
  `crates/valgebra-py/src/check/index.rs` holds each record's name-to-position
  plan, its keys and attribute names interned, each literal union's tables and
  each compiled pattern. It is keyed by the address of the schema's own
  buffers, which are stable for the life of the immutable schema, so a copied
  validator never inherits another's index; and a node the build traversal does
  not reach falls back to reading its own text, so no answer depends on the
  index being complete.

### Walking a Python value

- **The leaf decision is a test where the answer is.** `admit` in
  `crates/valgebra-py/src/check/walk/scalar.rs` is inlined, and recording a
  violation is a `#[cold]` function of its own, so the accepting path is one
  comparison and the allocating path sits apart from it.
- **A scalar is its type test.** A union's scalar branch, a mapping clause's key
  and value, a record's scalar field and the elements of a sequence of a union
  of scalars ask the type test directly (`scalar_member`, and `field_holds` in
  `check/walk/record.rs`), reading the depth level and the fatal-signal flag as
  the general walk does, so each refuses exactly where the walk refuses. A
  thousand-element `list[int | None]` costs 87% fewer instructions that way. A
  Rust test holds the direct reading to the walk's verdict for every scalar
  schema against every kind of value.
- **An instance of the class itself is read off its type pointer.**
  `type(obj) is C` answers `isinstance(obj, C)` before `C.__instancecheck__` is
  asked, which is the test CPython's `PyObject_IsInstance` makes first
  (`is_exactly_a` in `check/walk.rs`). PyPy implements `isinstance` otherwise,
  and there the call answers.
- **A literal's own constant is answered by its address.** A literal written in
  a program's source is usually the very object it validates: an interned
  string, a cached small integer. An exact `str`, `int`, `bool`, `bytes` or
  `None` that *is* the pooled constant equals it, so `is_the_constant` answers
  without a comparison. A `float` is excluded, since a NaN is not equal to
  itself.
- **A refinement borrows its operand and renders nothing on a pass.** The bound
  is read out of the pool as a borrow for the whole check, so a passing check
  writes no reference count. A violation's message, which needs the bound's
  `repr`, is built only at the site that records one (`Expected` in
  `check/walk/scalar.rs`): `tests/test_refinements.py` counts the bound's
  `repr` calls, and a hundred passing checks make none.
- **The recursion trail is a stack, reserved on first use.** Entering a
  reference pushes a `(value, definition)` pair and leaving pops it (`Trail` in
  `check/ctx.rs`), so nothing is hashed. The first level reserves `FIRST_TRAIL`
  pairs in one small request, and a walk that enters no reference allocates
  nothing (`--binding-recursive`).

### Sequences and sets

- **A sequence of one scalar kind is a loop per kind.** `scalar_list_matches`
  in `crates/valgebra-py/src/check/walk/sequence.rs` reads the element kind once
  and runs a loop whose type test is a constant, compiled out of line from the
  rest of the sequence walk so that an edit elsewhere does not move its register
  allocation (`--binding`). A set, a frozenset and a parsed JSON array take the
  same reading.
- **A list is read through a snapshot where that pays.** Reading a list element
  hands out an owned reference, a count written on the element twice; a tuple
  copy pays those writes in two loops inside the interpreter and none in the
  walk. `snapshot_pays` decides it per interpreter and per length, as [the
  margins above](#one-of-those-margins-moves-with-the-interpreter) describe. The
  instruction count of a snapshot reading is *higher* than in place -- the copy
  is instructions, and the stall it removes is not -- which is why the wall
  clock decides this one.
- **A tuple's elements are borrowed**, because a tuple cannot change while the
  caller holds it; a list cannot be read that way. On CPython,
  `PyTuple_GET_SIZE` and `PyTuple_GET_ITEM` read a tuple's storage whatever its
  type overrides, so every tuple, subclass or not, is read where it lies and its
  type is asked nothing: a `NamedTuple` validates at what a tuple does
  (`--binding-subclass`). PyPy's `cpyext` answers those accessors through a
  subclass's own `__len__` and `__iter__`, so there the walk asks the type
  whether it inherits both and copies any subclass that does not. A length
  bound asks the type whether its `__len__` is the tuple's own on every
  interpreter, since it counts what the value holds rather than what an
  override answers.
- **A tuple of scalar positions is one type test a position**
  (`scalar_positions_tuple_matches`), and a list whose element is a union of
  literals, or one class, is read through that union's table or the
  type-pointer test (`element_list_matches`). An element the test does not
  settle is walked, which may run Python, so the list is read in place unless a
  snapshot settles it entirely. A thousand `datetime.date` values cost 76% fewer
  instructions than through the general loop.

### Records

- **A closed record probes the keys it declares.** Membership is settled by
  asking the dict for each declared key and counting: the value belongs exactly
  when each key it holds matches its field and it holds nothing else, which the
  dict's own length says, since a dict cannot repeat a key. A declared key is
  interned with the validator and carries its hash, so each probe is the probe
  alone, and a key is resolved the way Python resolves one -- a `str` subclass
  carrying a field's text is found exactly where indexing the dict finds it.
- **The probe is the floor for this shape.** Profiled under callgrind on CPython
  3.12, fifty probes of a fifty-field record are about 143 instructions each and
  64% of the accepting call. Iterating the dict once and resolving each key by
  name instead was measured and is 47.9% dearer: an iteration step increments
  two reference counts, casts and decodes the key and hashes it, where a probe
  on an interned key carries its hash already.
- **Interned keys meet on a pointer, and Python interns most of them.** A dict
  probe compares the key it is given with the key it holds by pointer before it
  compares hashes or bytes. A dict written as a literal, one built from
  `**kwargs` and an object's `__dict__` all carry interned keys, so a record
  walk over them settles each field in one comparison: on a fifty-field record,
  interned keys read about 28% cheaper than keys that are not
  (`--binding-keys` against `--binding-record`). Keys parsed from JSON or built
  with f-strings are not interned and take the hash-and-compare path, which is
  what the figures on this page are measured over; `sys.intern` on the keys of a
  dict you validate in a loop is worth trying if that loop is your bottleneck.
- **An open record is read by its keys too.** A `TypedDict` admits keys it does
  not declare through the clause `str: anything`, and what that clause says
  about an undeclared key -- it is admitted exactly when it is a `str` -- can be
  said without reading its value (`Undeclared::of` in
  `crates/valgebra-py/src/check/walk/record.rs`; `--binding-open`). A record
  whose clause reads a key together with its value scans its entries, because
  that clause has to see each one.
- **A dataclass's attribute names are interned once per validator**, so
  `getattr` is handed an object the interpreter already holds.

### Unions and literals

- **A union of literals is a table.** A union whose members are all literals --
  a `Literal["a", "b", ...]`, a discriminator -- is compiled once into
  value-keyed sets, one for its integer literals and one for its string
  literals (`UnionPlan` in `crates/valgebra-py/src/check/index.rs`). An exact
  `int` or `str` value is one set lookup, so the cost stops growing with the
  number of literals. The integer set is consulted only for an exact `int`
  (never a `bool`) and the string set only for an exact `str`; any other value
  -- a `bool`, a `float`, `None`, a subclass instance, a JSON value -- falls
  back to the scan, which remains the single source of truth, and tests hold
  the two to one verdict over the cross-type cases. The table also keeps its
  constants' addresses, so a value that is the constant itself is answered
  without reading its text: a thousand-element list of four string literals
  costs 37% fewer instructions than reading each value's text.
- **The closest branch is searched within a bound.** A failing union reports
  the branch whose first failure lies deepest, and the search re-walks at most
  `CLOSEST_BRANCH_PROBE_LIMIT` branches, so a very wide union bounds the cost of
  its own report.

### JSON

- **`is_valid_json` walks the parsed document in place.** The walk reads a
  value that is either a Python object or a parsed JSON value, so a document is
  checked without building Python objects for it; the two sources run one walk
  and stay equivalent by construction, which a property test holds over random
  schemas and documents ([the JSON page](07-json.md) has the figures).
- **A closed record reads a document's keys through its plan, once.** Each key
  of the parsed object is resolved through the record's name-to-position plan
  and every declared field's value gathered before any is checked, because a
  repeated key means the document's last entry, as `json.loads` has it. The
  gathered table sits on the stack for a record of up to `FOUND_ON_STACK`
  fields.
- **A parsed object's keys are strings.** A clause keyed by `str` therefore
  admits every key of a parsed object, and covering a free-form section is a
  check of its values alone. An object of up to `SMALL_OBJECT` entries is
  covered where it lies -- an entry is the one the document means exactly when
  no later entry repeats its key -- and a wider one through a table of last
  values.

### Reporting a failure

- **`validate` is one explaining walk**, and the cheap readings serve it. An
  explaining walk records nothing of what it admits, so an element that passes
  its type test is answered by the test (`admitted_quietly` in
  `check/walk/scalar.rs`), and a sequence of scalars is read as its tests, an
  element that fails being walked at its own position (`list_explained`,
  `tuple_explained`). `validate` on a thousand-element `list[int]` that belongs
  executes about a fifth of the instructions the general element loop takes.
- **A union explains no branch for a value it admits.** Only a refused
  value's report reads the branches that did not match, so a branch that fails
  at the union's own location -- a scalar, a literal, a container of another
  kind, a class -- is decided by its test (`decided_quietly` in
  `check/walk.rs`), and a record branch by its deciding pass, its explaining
  pass waiting until no branch has admitted the value (`explain_union`).
  `validate` on a hundred values of a tagged union of three `TypedDict`s costs
  a fifth of the instructions it costs with each refused branch explained in
  turn, close to what `is_valid` costs, and a union of dataclasses about as
  much, on CPython 3.12 and 3.14 alike: the report a refused branch builds
  summarizes the value, and a dataclass's summary is its `__repr__`, Python
  code. Tests count the reprs a member's report would run, and hold a refused
  union's report to the one its chosen branch gives alone.
- **The explaining walk resumes where the deciding walk stopped**, as
  [the closest races](#the-closest-races) describe: a field the first walk
  passed has no violation to report (`--binding-explain`).
- **A report builds each key and string once.** The keys of the error model are
  interned, a violation's message and path are built once and shared between
  the error item and the attribute that mirrors it, and a path segment shares
  the field name the schema already holds.
- **A value summary is bounded while it is built.** Every violation summarises
  the value it is about, and a container is rendered through `reprlib` under a
  depth and width bound (`BOUNDED_REPR` in `crates/valgebra-py/src/errors.rs`)
  rather than rendered whole and cut, so reporting on a value nested twenty
  thousand deep takes about a millisecond. A value small enough to print
  renders exactly as `repr` would.

### What a relation between two validators costs

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

- **The cheap questions are asked first.** A node is a subtype of itself, which
  a pointer comparison answers before anything else; a universe on the right is
  read off the region set the query already holds; a schema below a union is
  first looked for among the union's branches, which keeps widening a literal
  union linear in its members; and the readings that refute a pair no rule
  places are asked in the order they cost, the walk of both subtrees last
  (`crates/valgebra-core/src/decision.rs`).
- **The set representation is asked only where the rules are silent.** Both
  deciders answer in three values, so the descriptor is asked where the rules
  answer *unknown*, never where they proved the answer.
- **Records compare by a cursor.** Two name-sorted field lists are walked
  together, each lookup one ordering comparison, with no table built per call
  (`FieldCursor` in `decision/records.rs`); a record whose fields repeat one
  schema asks that goal once.
- **Every query spends a work budget** (`spend` in `decision.rs`), and returns
  the conservative answer once it is spent, so a deeply nested Boolean
  combination stops rather than running unbounded.
- **The set representation shares what it does not edit.** An automaton holds
  its edge guards and its word automata by handle, a complement of a minimal
  automaton keeps it minimal, and a question about which kinds a descriptor
  reaches is read off its components (`Descr::within`, `Descr::reaches` in
  `descr/mod.rs`) rather than answered by building a meet and testing it.

**A pair's constants are read once per validator, not once per question.**
Relating two validators pools one's constants into the other's, and the key a
constant is pooled by is read out of the interpreter. Each validator keeps the
keys of its own pool from the first relation it is a side of, so asking about a
pair again reads neither pool. Relating two tables of ten thousand codes costs
5.4 million instructions that way, 2.3 million of them the rules' own answer;
reading the keys afresh for each question costs 15.0 million. Two validators
built alike compare equal in one pass over their members, and a validator keeps
its hash once computed, as a `frozenset` does.

**A refutation about a class reads the class.** It stands on a direct instance
of the class, and what that instance can carry is read off the namespaces of
the class's `__mro__`: which names each defines, whether each is a slot or a
data descriptor, and whether instances carry a `__dict__`. The namespaces of
`object`, the builtins a kind is read from and `BaseException` are static types
no assignment reaches, so each is read once for the process; a class's own
namespace is read once per relation, because an assignment to the class between
two relations moves what the second one reads. The reading is most of what a
refutation between two dataclasses costs: twenty thousand of its thirty-two
thousand instructions on 3.14.

There is no wall-clock gate on this, and that is deliberate: a relation's
wall-clock cost is a property of the pair, and pinning one would be pinning a
number the next rule changes. The instruction budgets cover the decision
workloads `crates/valgebra-core/examples/` holds -- one for each path a relation
takes, which `MODES` in `scripts/perf_gate.py` names -- and those are what a
change to the rules moves. What the bindings read off a class is no rule's, and
no decision workload reaches it, because their classes are identities with
nothing behind them; `--binding-relation` relates two dataclasses through
`relation_to` and counts it.

### The schema representation

- **A node carries its children by a shared handle.** A child is a shared
  pointer and a node's lists are shared slices, so carrying a subtree across a
  rebuild is a reference count rather than a copy; a field's name is an
  `Arc<str>`, atomic because a validator is shared across threads on a
  free-threaded interpreter.
- **A transform hands back what it was given when nothing changed**, so opening
  a schema that holds no record, or shifting the half of a composition whose
  indices do not move, returns the caller's own handle.
- **A node built twice is one node.** A member list, field list, clause list or
  child built twice is one handle, held in a per-thread, direct-mapped table of
  weak references (`crates/valgebra-core/src/ir/intern.rs`). Sharing is
  decided by a test stricter than structural equality -- the same allocation,
  or the same value in every field -- so a caller cannot observe which handle
  it got; and the table is what makes two equal schemas cheap for a relation to
  compare.
- **Lists are assembled in per-thread buffers** and moved into the node, and
  the lists every record shares -- an open record's catch-all, a closed one's
  empty one -- are allocated once for the process.

### The build

The release profile in `Cargo.toml` links with fat LTO into one codegen unit and
wraps rather than checks integer overflow, which the dev and test profiles
trap; `[profile.profiling]` is the same build with its symbols kept, for a
profiler. The published wheels are profile-guided, trained by
`scripts/pgo_workload.py`, which `pyproject.toml` names as `pgo-command`; what
that buys, and on which shapes it costs, is [measured
above](#baseline-matrix). A path the training workload never enters is laid out
by chance, so the workload reaches every reading the walk has -- lists of
scalars, of literals and of one class, records open and closed, a deeply nested
list, the explaining walk -- and relations at both of their levels.

### Measured and not taken

Each of these reads as an obvious gain and measures as a loss on a shape the
gate holds, under `scripts/perf_gate.py --against` unless it says otherwise:

- **Resolving a record's keys by iterating the dict**, rather than probing each
  declared key: 47.9% dearer, as above.
- **Reserving a record's field vector from the dict's length** while building
  it: the build shape reads 9% dearer, because one large request leaves glibc's
  fastbins, where growing from empty is served from bins already warm.
- **An unstable sort for the canonical orders**: 7% dearer on the build shape,
  since its swaps over a record's fields cost more than the stable sort's
  scratch buffer.
- **A memo over every goal of a query**: up to 18% dearer on the queries whose
  goals do not repeat, which is most of them; the memo kept is one entry, in
  the one loop where a goal repeats.
- **Comparing an order bound as a native integer**: 28% dearer on
  `--binding-refined`, since reading a Python integer out costs more than the
  rich comparison it would replace.
- **A replacement global allocator**: up to 28% cheaper on the JSON shapes and
  12% dearer on the call boundary, past the gate's ceiling, for a C dependency
  on every wheel target.
- **A snapshot of a list on CPython 3.14**: 1.65 ns an element against 1.28 in
  place, timed, because that interpreter makes the reference counts cheap.

## Regression gate

The wall-clock numbers above are for humans reading results; they are too noisy
on shared CI runners to gate a merge. The merge gate is instead a deterministic
instruction count: `scripts/perf_gate.py --against` runs each fixed workload
under cachegrind at the change and at its merge base, in one job with one
toolchain, and fails on a rise past the band it allows a change. The count is
identical across runs of a given build, so the gate does not flake, and
measuring both sides in one job cancels what another machine or another
toolchain would add, while an algorithmic regression is far larger than the
band. The committed budgets in `scripts/perf_budget.json` are read on the
nightly, as a record of one environment rather than as a merge gate: the same
commit re-measures several percent away on another machine.

The gate holds one workload per surface, because a gate only catches what it
exercises. `crates/valgebra-core/examples/` holds the pure-Rust ones: the schema
transformations (`perf_workload`), and the decision procedures, one workload for
each path a relation takes -- a proof, a refutation, a repeated goal and a shape
only the structural readings decide each walk a different path, and a workload
that asks only one of them measures only that one. The binding's shapes are
`BindingShape` in `crates/valgebra-py/src/workload.rs`, run by
`crates/valgebra-py/examples/binding_workload.rs`, and `BINDING_SHAPES` in
`scripts/perf_gate.py` names every one the gate measures: the membership walk
over a live value, the call boundary, the record walks closed, open and over
interned keys, the builds, the JSON document and the explaining walk among
them. The walk is the shipped hot path neither pure-Rust workload reaches, and
construction and the open record are each a cost no other shape's count
carries. A surface with no shape of its own can move by percents a release at a
time with every gate green.

Each binding shape embeds CPython, whose startup is not a fixed count, so the
gate measures the difference between two iteration counts. **What a shape's loop
holds is part of what its count means**, and is read before the number is: a
build shape that assembles its fifty fields inside the loop spends most of its
count on the harness naming them and none of it on the annotation walk, which is
the half a build gate exists to measure.

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
and holds it under the ceiling `scripts/perf_compare.json` states for that shape
-- what the project claims, not what it measured -- and, where the file carries
a ratio recorded in the same environment, within that shape's tolerance of it. A
ratio cancels the runner's absolute speed: if the machine is slow, both
libraries are slow in proportion, so the comparison survives the shared-runner
noise an absolute budget cannot. A shape fails the merge gate when valgebra's
ratio crosses its ceiling, or drifts past its recorded ratio where one is armed
— a competitive regression, whether from valgebra slowing down or ceding ground.

Re-record the budgets after an intentional change with:

```bash
python scripts/perf_gate.py --update            # the core budget; --decision, --binding-record, ... for the others
python scripts/compare_gate.py --update         # competitive ratios
```
