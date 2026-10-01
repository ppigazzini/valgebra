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
cheaper call: **23 ns against 26 ns** for a scalar on the machine below, the
best of fifteen `timeit` repeats of a million calls, because the interpreter
reaches a container slot directly and a method by its call protocol. Neither
number is the check. `Validator(anything).is_valid(1)` -- the schema that
answers `True` without looking -- costs 25 ns, so the *walk* for an `int` is
about a nanosecond and everything else is the boundary a Python call crosses.
For reference on the same run, `isinstance(1, int)` is 12 ns and a call to an
empty one-argument Python function is 15 ns.

Read that as the floor it is: a per-call check cannot be much cheaper than a
Python call, and the way to spend less is to make fewer calls -- validate the
list, not each element -- rather than to look for a faster scalar.

**The crossing carries no reference pool.** PyO3 keeps a pool of
reference-count decrements deferred while a thread is detached, and once any of
its lazy initialisers has detached and attached again -- the first interned name
does -- every call into the extension locks that pool's mutex to ask it for
them: forty instructions and eight branches on `Validator(int).is_valid(1)`.
The binding never detaches, so the pool is always empty, and
`.cargo/config.toml` compiles it out; `tests/test_build_flags.py` holds the
flags there. The instruction gate's boundary shape calls the walk without
crossing PyO3's call machinery, so the comparison gate's `scalar` is the
instrument that reads it: 36.7 ns with the pool gone against 39.9 with it, on
CPython 3.14 and the machine below.

### Results

End-to-end validation of a value that passes (lower is better):

| Shape | valgebra | pydantic (strict) | jsonschema |
| --- | --- | --- | --- |
| `list[int]`, 10,000 elements | 9.95 +/- 0.14 us | 76.4 +/- 0.34 us | 24,601 +/- 253 us |
| Closed record, 50 int fields | 0.521 +/- 0.015 us | 1.86 +/- 0.030 us | 126 +/- 2.9 us |
| Nested `list[...]`, depth 25 | 0.223 +/- 0.031 us | 1.92 +/- 0.050 us | 73.8 +/- 1.7 us |

valgebra relative to pydantic on this machine, under the CPython 3.14 the matrix
above names, as the medians above divide: **8.6x** faster on deep nesting,
**7.7x** on the large flat array, **3.6x** on the wide record. The comparison
gate's minimums read the same three at 8.2x, 7.7x and 3.8x (the 3.14 column
below). It is consistently far ahead of pure-Python jsonschema — 2,500x on the
array, 331x on the nesting and 242x on the record.
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
| `list[int]`, 10,000 elements | 0.183 | 0.130 | 0.135 |
| Closed record, 50 int fields | 0.239 | 0.262 | 0.274 |
| Nested `list[...]`, depth 25 | 0.135 | 0.122 | 0.262 |
| One `int` | 0.221 | 0.208 | 0.216 |

The **element** is what moves, not the check. A list hands out each of its items
as an owned reference — a count written on the object when the handle is made
and again when it drops — and the free-threaded build takes the list's lock for
each one besides. A schema nested twenty-five deep is twenty-five containers of
one element, so it is almost nothing but that cost, and it reads about twice as
dear there as under a global lock. A flat array of ten thousand is read through
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
| JSON document, 200 records parsed and checked | 0.61 | 0.072 over twelve runs |
| Error report, 50-field record with one wrong field | 0.46 | 0.074 over twelve runs |

The JSON document is a single pass over bytes for both libraries, which is why
valgebra takes three fifths of pydantic-core's time here rather than a fraction:
neither is spending its time in the check. A document's free-form sections are
`dict[str, V]`, and covering their keys is read two ways -- in place for a
narrow object, through a table of last values for a wide one
([dev/04-walk.md](dev/04-walk.md)).

The error report spreads the widest, beside the JSON document: 0.41 to 0.49
across the twelve runs, where every other shape spreads between 0.001 and 0.021.
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

**What a closed record costs is the interpreter's own dict lookup.** Profiled
under callgrind on CPython 3.12, fifty probes of a fifty-field record are about
143 instructions each and 64% of the accepting call; on the failing call the
deciding and explaining walks together make fifty-one probes for fifty fields,
the one repeat being the field that failed. The obvious alternative — iterate
the dict once and resolve each key by name, rather than probe each declared key
— was measured and is **47.9% dearer**: an iterator step increments two
refcounts, casts and decodes the key, and hashes it, where a probe on an
interned key carries its hash already. So the probe is the floor for this shape
and the walk is at it; the experiment is recorded here rather than re-run.

The scalar shape is absent from the table because it sits near timer resolution,
and most of it is the call rather than the check, as the floor above shows: on
CPython 3.14 the competitive gate measures it at a 39.5 ns median over seven
runs with a spread of a twentieth of that, around 4.8x. That gate measures it;
this record does not.

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

## How the record fast path is tuned

A closed record is answered by **probing the dict for each declared key**, not
by scanning the value's entries: the probe carries the key's hash already, where
an iteration step increments two refcounts, casts and decodes the key, and hashes
it. The alternative was measured and is 47.9% dearer, which is the figure
recorded above; the probe is the floor for this shape and the walk is at it.

The keys a validator probes with are interned once when it is first used, so a
wide record rebuilds no name map per call, and a dict whose own keys are interned
settles each field on a pointer comparison. The keys settle a record whenever
what its clauses say about a key it does not declare can be said without that
key's value: a record with no clause refuses the key, the top clause admits it,
and the `str: anything` a `TypedDict` carries admits it exactly when it is a
`str` (`Undeclared::of` in `crates/valgebra-py/src/check/walk/record.rs`). Where
a clause reads an undeclared key together with its value, the entries are
scanned instead, because that clause has to see each one. The two readings
answer alike by construction: neither resolves a key by decoding its bytes, so a
`str` subclass carrying a field's text is found exactly where the dict finds it
([dev/04-walk.md](dev/04-walk.md)).

A **report** on that record -- one field wrong, explained, raised -- costs about
three accepting walks, and the count attributes the three. Two are the walks:
the fast pass, which stops at the field that fails, and the explaining pass,
which resumes there and reads every field after it so the report names all of
them, fifty-one dict probes between them. The third is the raise itself, which
the interpreter charges for building the exception and unwinding to the caller.
Nothing in the three is a walk over what the schema already knows, so a further
cut would be a cheaper report rather than a shorter walk. The same count reaches
a shape the wall clock does not: the same fifty fields under the clause a
`TypedDict` carries, which the walk reads by its keys as it reads the closed
record. A shape of its own is what would see that record fall back to the scan,
which is why both are budgeted in `scripts/perf_budget.json`.

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
tests over the cross-type cases. A literal a program spells in its own source is
usually the very object it validates -- an interned string, a cached small
integer -- so the table also keeps the addresses of its constants and answers
that object without reading its text: a thousand-element list of four string
literals costs 37% fewer instructions.

Compiling one is linear too. A constant is read as itself before the frontend's
dispatch, and the table's sets are sized once, so a `Literal` of two thousand
string codes compiles in about three million instructions; read through the
whole dispatch one constant at a time, it costs four times that.

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

**A pair's constants are read once per validator, not once per question.**
Relating two validators pools one's constants into the other's, and the key a
constant is pooled by is read out of the interpreter. Each validator keeps the
keys of its own pool from the first relation it is a side of, so asking about a
pair again reads neither pool. Relating two tables of ten thousand codes costs
5.4 million instructions that way, 2.3 million of them the rules' own answer;
reading the keys afresh for each question costs 15.0 million.

There is no gate on this, and that is deliberate: the instruction budgets cover
the decision workloads `crates/valgebra-core/examples/` holds -- one for each
path a relation takes, which `MODES` in `scripts/perf_gate.py` names -- and those
are what a change to the rules moves. A relation's wall-clock cost is a property
of the pair, and pinning one would be pinning a number the next rule changes.

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

**A refinement marker is read by its type, once.** A marker carries one or two
of ten optional attributes and not the rest, and below 3.13 asking for one it
does not have is answered by raising -- `PyObject_GetOptionalAttr` is the first
spelling that does not, and there is none before it. So the frontend asks the
marker's *type* and remembers what that type carries, which costs no exceptions
on any interpreter: fifty `Annotated[int, Ge(0)]` fields compile in about 30 us
on CPython 3.14 and 27 on 3.12, on the machine class above. Compilation happens
once per schema, so this is a startup figure rather than a per-call one -- it
matters to a program that builds validators per request, and to nothing else.

**A `NamedTuple` validates at what a tuple does on CPython.** CPython's
`PyTuple_GET_SIZE` and `PyTuple_GET_ITEM` read a tuple's storage whatever its
type overrides, so the walk reads every tuple, subclass or not, where it lies
and asks its type nothing. PyPy's `cpyext` answers those accessors through a
subclass's own `__len__` and `__iter__`, so a subclass that overrides either can
send the walk past the end of its storage; there the walk asks the type whether
it inherits both -- every `NamedTuple` does -- and copies any subclass that does
not ([dev/04-walk.md](dev/04-walk.md)). A length bound asks the type whether its
`__len__` is the tuple's own on every interpreter, since it counts what the
value holds rather than what an override answers.

**Interned keys are the fast path, and Python interns most of them for you.** A
validator holds an interned `str` for every declared field, and a dict probe
compares the key it is given with the key it holds by *pointer* before it
compares hashes or bytes. A dict written as a literal, one built from
`**kwargs`, and an object's `__dict__` all carry interned keys, so a record
walk over them settles each field in one comparison: measured on a fifty-field
record, both sides interned read **29% cheaper** than neither. Keys that are
not interned -- the usual case for a dict parsed from JSON or built with
f-strings -- take the hash-and-compare path, which is what the figures on this
page are measured over. `sys.intern` on the keys of a dict you validate in a
loop is worth trying if that loop is your bottleneck.

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
