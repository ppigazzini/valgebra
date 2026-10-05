# Tooling and CI

The gates, the lanes, what each instrument cannot see, and the exit code that
says which of the three things happened.

## Check the exit code, never a piped fragment

```sh
python scripts/perf_gate.py | tail -1     # WRONG -- reads 0 from tail while the gate is red
python scripts/perf_gate.py; echo $?      # right
```

## Three outcomes, three exit codes

| Outcome | Exit | Means |
|---|---|---|
| pass | 0 | the gate ran and the property holds |
| fail | 1 | the gate ran and the property does not hold |
| **could not run** | **2** | no measurement, no baseline, a missing dependency |

A gate that could not run has proven nothing and must never read as one that
passed. `scripts/perf_gate.py` exits 2 on cachegrind output it cannot parse,
`scripts/mutation_gate.py` on a missing sweep result or baseline,
`scripts/compare_gate.py` without its benchmark dependency,
`scripts/docs_lint.py` on a tree it cannot read.

**A command line it cannot read is one it could not run.** Every script reads
its flags through `argparse` before doing any work: `--help` prints the usage
and exits 0, and an unknown flag, a missing value or an unknown choice exits 2.
Read by substring, a mistyped flag is no flag at all, and a gate asked for one
measurement answers for another. `tests/test_script_arguments.py` holds every
file under `scripts/` to it.

## The local gate, and the contract inventory

**`scripts/gate.py` runs the lane's steps, not a list that resembles them.** It
reads every `run:` step of every job the `ci` aggregator waits for and a push
runs out of the workflow -- `SCHEDULED_ONLY` drops the nightly jobs and
`RUNNER_ONLY_JOBS` the ones left out whole -- and runs them in a fresh
**shallow clone of `HEAD` with no tags**, which is what `actions/checkout`
produces and what a developer's clone is not. A step that needs a runner
(valgrind, a base built beside the head, a mutation sweep, a second operating
system, the bench lane's optimized wheel) is named in `NEEDS_A_RUNNER` with the
reason instead, and `tests/test_local_gate.py` holds that list to the workflow
in both directions. `scripts/gate.py --list` prints the plan: which steps of
each job run and which are named. The `pgo-compare` job is in it whole, its
profiled build included, although a push skips that job.

**The instruction gate runs here, although the lane that owns it cannot.** The
`bench` lane wants cachegrind, a profiled wheel, a second interpreter and a
system package installed with `sudo`, so every step of it but the sync is
excused -- and the *comparison* it carries wants none of those. Two builds of
one workload, measured against the base by the lane's own rule (the merge base
with the remote-tracking branch, or `HEAD`'s parent where that merge base is
`HEAD` itself; `perf_base` in `scripts/gate.py`), is three minutes in the caller's
tree, and the tree is where it has to run: the shallow clone holds one commit
and a comparison needs the other. Excusing it with the rest of the lane is how a
change that read sound and cost **seventy-one times** the instructions -- 6.45
billion against 89.8 million on the relation matrix -- passed this script green.
One mode runs, `--decision-matrix`, because it is the workload whose shapes
reach the set representation and a union or complement change reads 0.00% on the
others; the `--binding-*` modes want the extension built into an interpreter and
stay with the lane. Missing valgrind is named as the reason rather than passed
over.

**The floor interpreter is built beside the caller's, and the suite runs on
it.** The matrix runs every supported release -- `ci.yml` owns the list -- and
a developer runs one, so every difference between two of them is a difference
the gate could not see -- and they are not rare: a typing member the floor does
not carry collects on the caller's release and fails collection on the floor,
taking every job there with it. The floor is the end of the range where that
lands, because the suite is written on the newest release the tree supports and
read on the oldest. So the gate makes a second environment on the floor
`ci.yml` names, builds the extension into it, and runs the **product** suite
there; the repository checks read the tree and answer the same on any release,
so they run once. Which release the floor is stays in `ci.yml`: the gate reads
it, and `tests/test_local_gate.py` refuses a copy of the number in
`scripts/gate.py`, because a stale floor is the one kind of stale that tests
less than it claims while staying green.

**What the clone models, the environment does not.** A step gets the caller's
environment plus the job's and the step's own `env:`, and the runner's own
absences are not modelled by that -- so a variable a developer's terminal sets
reaches a step that a lane runs with it unset. One of them gave a false red:
`FORCE_COLOR` overrides a tool's terminal check, `uv export` wrote escape codes
into the requirements file it generates, and `pip-audit` refused the file
against a dependency tree with no advisory in it. The two variables whose only
purpose is that override are dropped (`runner_environment`), and so is
`VIRTUAL_ENV`, which `uv run` sets for the gate it launches: in the clone it
sends a step's `uv pip install` to the caller's venv while `uv run` reads the
clone's, so the wheel the profile comparison times was installed where the
timing step could not import it, and the caller's venv kept a wheel built from
a clone of `HEAD`. Without it, `uv pip` finds the clone's `.venv` from the
working directory, as a runner's step does. The venv's `bin` leaves `PATH` with
it, and the clone runs the lane's own sync and build rather than excusing them,
because every later step runs under `uv run --no-sync`: with neither, a clone
with no environment of its own found the caller's pytest, which imported the
caller's package and reported green on the caller's tree.

**The whole list, since two of these cost a day each.** The table is what a
lane differs from a local run in, and the `modelled` column is what the gate
reproduces; the count is the table's rather than this sentence's:

| difference | modelled | what it cost when it was not |
| --- | --- | --- |
| the clone: one commit, no tags | yes, `shallow_clone` | a ledger read `git describe` and reddened eight jobs at once |
| the step's declared `env:` | yes, since the plan carries it | `RUSTDOCFLAGS=-D warnings` was dropped, so `cargo doc` could not fail locally |
| the terminal's own variables | yes, `runner_environment` | `FORCE_COLOR` turned `pip-audit` red against a clean dependency tree |
| the machine's git identity | yes, `deep_clone` and `NO_IDENTITY` | a checkout configures no `user.name`, this repository has one in its own `.git/config`, and a ledger that plants a commit with `git commit-tree` passed here and failed there |
| the interpreter the lane names | partly: `PYO3_PYTHON` follows the caller's, and the gate's closing line says so | the mutation lanes name CPython 3.12 in `ci.yml` and the cachegrind lane 3.14; a local sweep on 3.14 read two mutants as survivors that the lane kills, which is half an hour spent on a difference that was the interpreter |
| the *release* of the interpreter, not only the caller's | yes, at the floor: the gate builds it beside the caller's and runs the product suite on it | `typing.Self`, `LiteralString` and `Unpack` are 3.11 members; a test module naming them collected here on 3.14, failed to collect on the floor, and took nine jobs red with it |
| the operating system and architecture | no | a macOS or Windows leg fails where Linux does not, and nothing local sees it |
| the pinned tool versions | no, for the tools `taiki-e/install-action` installs | `ci.yml` installs `cargo-deny`, `cargo-llvm-cov` and `cargo-mutants` at versions it names, and the gate runs whichever is on the caller's `PATH`; a `uvx` step carries its pin in its own command, so the gate runs the lane's version of it |
| the build of the interpreter, not only its version | no: a rule answers it instead | `sys.stdlib_module_names` is the build's, not the release's -- this box's 3.12 lists the Windows-only `_wmi` and a runner's does not, so a table of every name reported a difference between two builds as a moved row. The floor table records the modules this tree imports, which are portable by construction |
| the machine's own speed | not modelled, and not a gate: every merge-blocking number is an instruction count | an absolute count reads 5--8% apart between two machines on one `rustc` line, which is what the recorded budgets carry a band for |
| secrets, tokens and the event payload | no, and the steps that need one are excused by name | the merge base comes from the event, so `--against` runs only in the lane |
| what a gate **counts**, against what it claims to | no: a scope is a claim in prose, and no check reads it | the binding coverage floor counted the instruction gate's own workloads, which no suite runs, and the lane went red at 94.50% over a change that added none of its own uncovered lines; the build shape counted the harness formatting fifty names, and three quarters of what it reported was that |

The first four are closed. Named rather than numbered from here, since the
table grows: *the interpreter the lane names* is a row rather than a fix
because the gate runs the caller's toolchain by design, and the gate's own
closing line names it; *the release of the interpreter* is closed at the
floor, which is the end of the range a difference lands at; *the build of the
interpreter* is answered by a rule rather than by the gate; the rest are
differences a local gate cannot remove, and naming them is what keeps a green
local run from being read as a promise it never made.

Naming them carries an obligation the other way, and the wheel lane is what
taught it: a job on that closing line went red within two minutes of a push, on
a `docker pull` that answered `504`, and the only thing that could have noticed
is a person reading the badge. A green gate followed by a red lane it named is
the expected shape, not an anomaly — so a lane the gate does not reach is read
after the push and its verdict reported, with a re-run, a fix, or a stated
reason it is not the tree's. The list exists to say what a green run did not
check; leaving the check undone afterwards spends the list on nothing.

**The fourth row is the one to read twice.** It is not a
setting a developer chose: it is an *absence* on the runner that is a presence
here. A check that writes anything -- a commit, a file, a config -- asks the
machine for something the runner does not have, and passes locally for that
reason alone. The gate models it by cloning: a clone inherits no `user.*`, and
pointing git's global and system config at an empty file removes the rest.

The last row is a different kind: not a difference between a lane and a local
run at all, but between what a gate measures and what its own text says it
measures. Both halves of a gate are a claim -- the number and the scope -- and
only the number is held to anything. **When a gate moves, read what it counts
against the sentence that says what it counts.**

**The clone builds its own environment.** The lane's dependency sync and
extension build run in the clone and write the clone's `.venv`, the one
environment a gate may write, and every later step reads that one. **Do not**
lend it the caller's virtual environment instead: `uv run` inside the clone
*writes* the environment it is pointed at, which uninstalls the built extension
and the bench group from the tree being worked in. A gate that damages the
environment it checks is worse than one that spends a build on its own.

**An excuse is a claim, and a false one is a hole.** A step is excused only
for what a developer's machine cannot do. A check that is static -- a type
checker given a target release, which reads the stubs for that release and
needs no interpreter of it -- is not one, and runs. Jobs left out whole
(`RUNNER_ONLY_JOBS`) are held to `ci.yml` and to a line in `UNREACHED` that the
closing report prints, so a green run says what it did not reach.

A step is **accounted for** when it is in the plan the gate builds or named in
`NEEDS_A_RUNNER`, and the ledger asks it of the **plan**. Asking instead whether
a name is excused is a question no workflow can fail. Every expression the gate
cannot fill in makes the step unresolved, which is a skip with a reason rather
than a command: a step carrying `${{ github.sha }}` is unresolved exactly as one
carrying `${{ env.X }}` is, so neither reaches bash with its braces intact.

The reason this gate exists is the clone shape rather than a principle. A check
reading `git describe` answers one way in a local clone, which carries tags, and
another in a checkout, which does not. Everything else on this page is about what the
gates check; this is about *where* they check it.

A change is not done until every command exits 0. `CONTRIBUTING.md` holds the
list; read it there rather than here, because a second copy drifts by one entry
and reads exactly like one that has not.

Beside it, the same file carries a **contract inventory**: one row per thing this
project promises, naming the file that owns it and the single command that
reproduces that one verdict. The gate answers "is my change ready"; the inventory
answers "what does this project promise, and how do I check just that one?" —
which is the question a reader has when a lane goes red and they want to
reproduce one result rather than all of them.

`tests/test_contract_inventory.py` holds the table to the tree in both
directions: a gate script with no row fails, and a row naming a script or a
source of truth that does not exist fails. A script can be a row's *subject*
rather than its command — the profile-guided training workload is driven through
the packaging config rather than typed — so the check reads the whole row, for
the script's path rather than its name, since one script's name can be part of
another's (`gate.py`, `perf_gate.py`).

One line in it is easy to miss and is the reason a lane went red: **the fuzz
crate is a detached workspace**. libFuzzer needs a nightly toolchain, and making
it a workspace member would put nightly on every stable gate's path, so
`cargo check --workspace` does not reach it. A change to the core's public types
compiles cleanly without `cargo check --manifest-path fuzz/Cargo.toml` and turns
the fuzz lane red. `tests/test_build_surfaces.py` holds every manifest in the
tree to being a workspace member or a detached surface named with the command
that builds it.

## What the linters excuse, and where

**An ignore in `pyproject.toml` covers the files its reason is about, and no
others.** ruff selects every rule. The global list holds only what is true of
every file -- the two formatter conflicts, the contradicting docstring pairs,
the licence header. The annotation and docstring rules apply to the shipped
package, which satisfies them, and are excused per directory in `tests/`,
`scripts/` and `benches/`, by the codes each one breaks and with the reason
beside them. An ignore written wider than its reason lifts a rule from the
package that nobody decided to lift, and nothing reports it: ruff flags an
unused `noqa` and says nothing about a per-file ignore that matches no finding.

**ty runs every stable rule it leaves off by default that the tree passes,
raised to an error**, and `[tool.ty.rules]` names the ones left at the default,
each with its reason. `[tool.ty.analysis]` turns strict equality on and says
why strict generic narrowing stays off. A warning fails the run as an error
does, so the level written is the intent, not the exit code.

**pyright runs in the profile `[tool.pyright]` names**, so a release that moves
pyright's default moves nothing here. The type lane runs it and `mypy --strict`
over the package as well as over the typed consumer, because a checker reports
nothing inside an installed library's stub: a stub one of them refuses is
refused nowhere a caller looks. An ignore in the stub is coded for the one
checker that reports, with the reason beside the line it covers.

**Each surface is read by the checkers whose reading is its contract.**
`.github/workflows/ci.yml` owns the commands, and ruff reads every surface:

| Surface | ty | mypy | pyright |
|---|---|---|---|
| the package: the stub and its two modules | the newest release and the floor | `--strict`, floor and newest | floor and newest |
| `tests/typing/consumer.py` | the newest release, with the tree | `--strict`, floor and newest | floor and newest |
| `tests/typing/readings/` | the floor | the floor | the floor |
| the published examples | yes | yes | yes |
| the rest of `tests/`, and `scripts/`, less what `[tool.ty.src] exclude` names | yes | no | no |

stubtest reads the stub against the built extension besides. The readings are
held where the checkers disagree, by `tests/test_checker_readings.py`, and the
examples by `scripts/check_doc_examples.py`. mypy and pyright read no other test
or script, because a suite checked under three dialects carries three sets of
ignore comments for code no caller imports.

**The limit.** ty's rule list moves with its releases, and a rule a release adds
off by default stays off here until someone raises it; `ty explain rule` gives
every rule's default level, which is the list to read against
`[tool.ty.rules]` when the lock moves ty.

## The numeric gates

### The instruction budget

Wall-clock benchmarks on shared runners are too noisy to gate, so
`scripts/perf_gate.py` gates **instruction counts under cachegrind**, which are
identical across runs of a build. Three workloads, one per surface, because a
gate only catches what it exercises:

- the **core** transformations — the constructors' normal form, the composition
  remap, the record transform;
- the **decision** procedures (`--decision`, `--decision-refute`,
  `--decision-repeat`, `--decision-matrix`) — subtyping, emptiness, equivalence;
  `MODES` in `scripts/perf_gate.py` owns the list, and the first three are the
  workloads one whose relations hold, one whose relations are refuted, and one
  whose goals repeat. The core workload never calls a decision, so without
  these the whole decision surface is unmeasured in both directions: neither
  what a new rule costs nor what a cheaper one saves. Three because a proof, a
  refutation and a repeated goal walk different paths, and a workload that
  asks only for proofs holds a refuting rule to nothing;
- the **binding** shapes (`--binding`, `--binding-boundary`,
  `--binding-record`, `--binding-keys`, `--binding-open`, `--binding-mapping`,
  `--binding-subclass`, `--binding-recursive`, `--binding-deep`,
  `--binding-refined`, `--binding-nullable`, `--binding-set`,
  `--binding-json`, `--binding-json-reject`, `--binding-json-union`,
  `--binding-json-open`, `--binding-json-deep`, `--binding-pattern`,
  `--binding-build`, `--binding-annotated`, `--binding-object`,
  `--binding-protocol`, `--binding-relation`,
  `--binding-explain`, `--binding-explain-accept`) —
  membership over a live Python value, the call boundary alone, a wide record
  closed, the same record walked over a value whose keys are interned, the
  record open the way a `TypedDict` is, a `dict[str, int]` read through its
  clause, walking a `NamedTuple`, a recursive schema's descent, a value nested
  twenty-five deep, elements that each read a refinement's bound, a list of
  `int | None`, a `set[str]`, parsing and walking a JSON document -- accepted,
  rejected halfway, as a union of two record kinds, read through a key-type
  clause, and against a recursive schema -- matching a string against a compiled
  pattern, building a validator from its Python spelling, compiling one written
  as a `TypedDict` of refined integers, compiling a fifty-field dataclass or a
  fifty-member protocol, relating two dataclasses, and explaining a failure in a
  record or accepting one in the same mode. The walk is the shipped
  hot path neither pure-Rust workload reaches; schema construction grew twelve
  percent over a release cycle while only the walk was counted, and an open
  record was read a third dearer than a closed one while only the closed one
  was. Each shape builds what it reads outside its loop: the build shape once
  formatted fifty names and filled a dict per iteration, and three quarters of
  its count was that.

  **One accepting JSON shape measured every JSON change**, and the paths a
  change could lose on had no count: the reject a fast walk stops at, the union
  that reads a parsed object once per branch, the clause that reads a key rather
  than a field, the reference on the parsed path. So did one walk over plain
  kinds measure the walk: a refined element reads its bound on every check,
  sixteen times the instructions of an element that only has a kind.

  The later shapes are there because the earlier ones could not see repairs
  worth a third, three quarters, and six percent of what they touched. **A dict of bare type
  objects is not what the frontend costs**: an annotation with any depth is
  read through `get_type_hints`, asked per field whether a qualifier states its
  required-ness, and asked per marker for four bound names it probably does not
  carry -- and none of that had a count. **A record walk over a dict the
  caller wrote as a literal is not the walk over one built with f-strings**
  either: the probe compares by pointer where both sides are interned, which is
  28% of the call, and the shape that walks non-interned keys moves by a
  percent when the validator's own side stops interning -- inside the relative
  gate's ceiling, and therefore invisible. **And a class is not a dict**: a
  dataclass is the one class form whose compile asks the standard library a
  question, and putting the `dataclasses` import back at the top of the
  frontend cost a build that compiles none of them 6.45% -- found with a
  profiler, because no shape here would move. **And a subclass is not its
  base**: the walk cannot trust the C length accessor for a `tuple` subclass,
  because `cpyext` answers it through the object's own `__len__` -- but it can
  trust it for one that *inherits* the base's slot, which is every
  `NamedTuple`. Telling those apart by "is this exactly a tuple" copied every
  `NamedTuple` and cost 100 ns against a plain tuple's 57 on CPython 3.14,
  under every green lane. The repair's own mutant decides the same things for
  every ordinary type -- the copy holds the same elements -- and only a
  metaclass that raises when asked for `__len__` tells the two apart, which is
  the test that kills it. What the line buys is the cost, and this count holds
  that: reverting the line reads +171.65%.

  **And a shape that wins by a wide margin measures nothing.** The JSON
  document is the comparison gate's closest race, and its ratio moves by a tenth
  -- 0.78 to 0.87 -- with no gate red, because a wall clock under a ceiling it
  clears by a third says nothing until somebody looks — which the section on the
  competitive ratio, below, records as the reason the recorded block exists at
  all. `--binding-json` is that shape's deterministic twin, over the same two
  hundred records read against the same `list[JsonRecord]`. Most of what it
  counts is not this crate's: profiled at `9700181`, about 55% of a call is
  `jiter` building the value tree and 14% is dropping it, with the walk in the
  remaining third — and `pydantic-core` parses to the same `jiter` tree before
  it validates, so the two thirds is the floor both libraries stand on and the
  third is what the ratio is about. A walk regression therefore shows here at
  roughly a third of its size, which is the price of measuring the shape a
  caller actually runs rather than a walk with the parse taken out.

  **And a relation is not a decision workload.** The four decision shapes
  relate classes that are identities with nothing behind them, so what the
  bindings read off a real class -- the namespaces of its `__mro__`, which a
  refutation about the class stands on -- had no count. Read whole for every
  class of every query, with `object`'s namespace among them and a `hasattr`
  per value, it made one relation between two dataclasses six times dearer on
  3.14 and twenty-two times on 3.12, with every gate green. `--binding-relation`
  relates two of them through `relation_to`.

  **And a compiled pattern is not a comparison.** Every other refinement costs
  an operator; a pattern costs a compiled object, built once when the
  validator's index is built and found again by the pattern's address. Losing
  that precompute is not a wrong answer — `check/walk/scalar.rs` compiles the
  pattern on the spot and decides the same thing — so no test holds it, and the
  mutation baseline accepts the index's survivor for that arm on exactly this
  argument. What makes accepting it honest is `--binding-pattern`: delete the
  arm and one validation costs 462,318 instructions against 783, which is the
  whole suite passing while every pattern check compiles a regex.

The binding workload embeds CPython, whose startup is not a fixed instruction
count, so the gate measures the **difference** between two iteration counts:
startup cancels and the per-iteration walk cost remains.

**Profiling the shipped extension, rather than the workload binary.** A
`--binding-*` count says a shape moved; attributing the movement to a symbol
needs a build that has some. `[profile.profiling]` in the workspace manifest is
that build — `release` plus debug info, minus the strip — and it is reached
through cargo rather than through maturin, which builds the extension module
the release lane ships:

```bash
export PYO3_PYTHON="$(uv python find 3.14)"          # the interpreter the bench lane pins
cargo build --profile profiling -p valgebra-py --features pyo3/extension-module
pkg="$(python -c 'import valgebra, pathlib; print(pathlib.Path(valgebra.__file__).parent)')"
cp target/profiling/lib_valgebra.so "$pkg/_valgebra.cpython-314-x86_64-linux-gnu.so"
PYTHONHASHSEED=0 valgrind --tool=callgrind --callgrind-out-file=out.callgrind python probe.py
callgrind_annotate --inclusive=yes out.callgrind
```

Two things make this work and are easy to get wrong. The interpreter must be
one valgrind can run: uv's builds are, and a source build compiled with
`-march=native` is not, since valgrind aborts on an instruction it does not
model. And
`pyproject.toml` must not set `strip`: a `strip` key under `[tool.maturin]`
overrides both profiles, so `maturin build --profile profiling` would produce a
binary with no symbols and a profile that attributes nothing, which reads as a
workload with nothing to profile rather than as a build setting. Stripping is
the profile's decision, and `[profile.release]` makes it — the shipped wheel is
stripped, and a wheel built from `profiling` is about twelve times its size
because it is not.

The rules that keep a measurement that did not happen from reading as a
verdict:

- **The band is two-sided.** A workload that stopped doing the work executes
  *fewer* instructions, and a ceiling alone publishes that as an improvement it
  never earned. A deliberate optimization past the floor is re-recorded rather
  than absorbed.
- **The workload proves it ran.** Each prints a checksum folded through every
  result, compared **before** any count. The core and decision workloads' are
  recorded constants; the binding workload's *is* its iteration count by
  construction, so a workload that ignored its argument — reporting a difference
  near zero — fails rather than passing under the ceiling. The decision
  workload's checksum counts verdicts, so a change that decides *differently*
  fails on the checksum before its instruction count is read.
- **An unreadable measurement exits 2.**
- **The interpreter's hash seed is fixed.** A binding shape embeds CPython,
  which draws a string hash seed per process, and a shape that probes a dict of
  string keys executes a different number of instructions under every seed. On
  the difference of two counts that is a few percent, which is the ceiling.
  `perf_gate.py` sets the seed in the one place every measurement passes
  through, so a reading taken by hand is the reading the gate takes — and a
  reading on a shape whose profile names no function of the changed file is
  read as the instrument's before it is read as the change's.
- **The environment is the gate's, not the caller's.** A process's environment
  block is copied onto its stack at start-up and, by an embedded interpreter,
  onto its heap, so its size moves a count on its own: a pure-Rust shape, which
  is not differenced, read 2.97% apart between a login shell and `uv run`.
  `workload_environment` in `perf_gate.py` passes a workload the loader path,
  `PYTHONHOME` and `VALGRIND_LIB` where the caller has them, and the fixed seed,
  and nothing else; valgrind is resolved to an absolute path first. The same
  binary then reads the same count from any shell. `--update` writes the
  valgrind and C library a budget was recorded with under `measured_with` in
  `scripts/perf_budget.json`, since both move a count with the tree unchanged.
- **No run writes bytecode.** A binding shape is the difference of two runs,
  which cancels the interpreter's start-up only while both runs start alike. An
  interpreter compiles a module the first time it imports it and writes the
  bytecode beside the source, so on a fresh interpreter the first run pays a
  compile its partner reads back. `settle_the_heap` imports `ctypes`, nothing
  else in the bench job does, and the job installs its interpreter fresh: the
  first walk the gate measured read a third dearer than the same binary run
  again, on whichever side came first. `workload_environment` sets
  `PYTHONDONTWRITEBYTECODE=1`, so both runs compile or both read. Before the
  two, `warm_the_interpreter` runs the shape once uncounted, and that run may
  write: without it the side measured first reads a colder cache than the side
  after it, since building the base runs the interpreter in between, and a
  cold start leaves the loop a different heap.
- **Branch mispredicts are read beside the decision shapes, and not gated.**
  The decision path dispatches on a node's kind through jump tables taken
  several times per goal, so a change can trade instructions for predicted
  branches, which an instruction count reads as a regression. The four
  decision modes run under `--branch-sim=yes`, which leaves the instruction
  count as it is, and print the simulated mispredicts and their movement
  against the base (`BRANCH_MODES` in `perf_gate.py`). No verdict reads them:
  cachegrind predicts an indirect branch by its last target, a model its
  manual dates to processors older than the lanes', so the column says where
  to look with a hardware counter rather than what a change costs.
- **The heap is settled before a shape counts.** What a loop's allocations
  cost depends on the heap it starts from: whether glibc serves a large request
  from the top chunk or first consolidates every small chunk the previous
  iteration freed follows from where the setup's long-lived blocks landed. The
  size of the process environment moves that, and so does any change to what
  runs before the loop -- a change to the *build* path moved the JSON shape by
  nine percent while its loop executed the same instructions. `settle_the_heap`
  in `crates/valgebra-py/src/workload.rs` calls `malloc_trim(0)` before every
  shape's loop, which took the JSON and recursive shapes from 8.2% and 7.2%
  apart between two environments to 0.06% and 0.36%. Trimming does not reach
  what the *binary's* layout decides: one commit built in two directories read
  the JSON shape 458M and 528M, all of it in `malloc_consolidate`, and the
  merge gate builds its base in a directory of its own. So every measurement
  runs with `GLIBC_TUNABLES=glibc.malloc.mxfast=0` (`MALLOC_TUNABLES` in
  `perf_gate.py`), which turns off the fastbins that consolidation empties,
  and the same two builds read 0.02% apart. A shape whose reading moves with
  the environment has lost one of the two:

  ```bash
  b=target/release/examples/binding_workload   # built by perf_gate.py
  for env in "uv run --no-sync" "env -i PATH=/usr/bin:/bin LD_LIBRARY_PATH=$LD_LIBRARY_PATH PYTHONHOME=$PYTHONHOME"; do
    $env valgrind --tool=cachegrind --cachegrind-out-file=/dev/null "$b" 500 json 2>&1 | grep 'I *refs'
  done
  ```

### What the merge gate compares against, and why not a number

**The merge gate is relative.** `--against <rev>` builds and measures the same
workload twice in one job -- once at `HEAD`, once at `rev` in a throwaway
worktree with its own target directory -- and holds the difference to 2%. The
toolchain, the flags, the machine and the cachegrind version are one and the
same across the pair, so what is left is the change.

The recorded numbers in `scripts/perf_budget.json` are a **record of one
environment**, read on the nightly and never on the merge path. The reason is
measured: the commit that recorded the current core budget re-measures 5.35%
away from it on another machine of the same rustc line. An absolute band wide
enough not to flake on that is ±10%, which cannot see a real 5% regression --
so the absolute check is either noisy or blind, and choosing between those is
not a gate. Re-record with `--update` when an intentional change moves the
number, and say what moved.

One refusal is the relative gate's own: **a workload that changed is not a
comparison.** Two builds of a workload computing the same thing agree on its
checksum exactly, so a disagreement says the workload moved between the two
commits and the counts are of different work. That exits 2 -- neither a pass nor
a regression.

**A shape the base does not carry is a new shape**, not a comparison and not a
pass: it is held to its recorded budget in the same job. The gate reads this
from the base's *source* before it builds anything -- the example the mode
names, and for a binding shape the arm that names the shape (`absent_at` in
`scripts/perf_gate.py`) -- because a base older than a shape builds the example
and its binary refuses the name, and a refusal read from the binary would look
like a broken build.

Do not copy a budget into prose: `scripts/docs_lint.py` fails on it, because a
figure that moves when the budget is re-recorded is stale the next time it
moves.

### The competitive ratio

`scripts/compare_gate.py` compares per-call time against pydantic-core across a
shape matrix -- the accept walk, a large array, a wide record, deep nesting, a
JSON document, compilation, and the report a failure builds -- as a **ratio**.
A ratio cancels the runner's absolute speed, which is what lets a wall-clock
measurement gate at all.

Each shape's ceiling is a **claim, not a recorded measurement**: the ratio the
project says it stays under, with headroom. A recorded ratio travels badly,
because the two libraries respond differently to a PGO build and to an
interpreter. The interpreter is the one that moves a shape far: on a single box
a schema nested twenty-five deep reads 0.13 to 0.14 under CPython 3.12 and 3.14
and 0.26 under the free-threaded build, where every read of an element out of a
mutable container takes that container's lock. This gate is the coarse tripwire
for ceding ground, with `perf_gate.py --against` doing the fine-grained work at
2%. Changing a ceiling is an edit with an argument in its commit message.

It is also the one gate that times the wheel the release ships. `perf_gate.py`
builds without a profile, so a change the profile lays out differently is
invisible there: a list arm that grew by one branch read 0.9% in its
instruction count and 64% in this gate's `deep_nesting`, because the training
workload held no counts for the loop the branch moved out of the walk.

**A ceiling a shape passes by a wide margin stops measuring it**, which is why a
claim is not the whole of the file. The JSON document's ratio moves between
0.62 and 0.78 under a ceiling of 1.00 with no gate red -- thirty runs on CPython
3.12 and 3.14 on one box -- and nothing reports the move until somebody re-runs
this gate and reads the number. So beside each
ceiling the file has a `recorded` block for the ratio the shape last measured
and the spread it was measured across -- the ratchet the mutation sweep and the
instruction gate already have. Once armed, a shape measuring worse than
`recorded + tolerance` is red while still under its ceiling.

**The ratchet is not armed.** The spreads are recorded; the ratios and the
environment in `scripts/perf_compare.json` are empty until the first `--update`
on the bench lane (`_how_to_arm` there says why), and until then the gate
prints `ratchet not armed` and judges the ceilings alone.

The travel problem is answered rather than avoided. The recorded block names the
environment it was taken in -- the interpreter, whether it has a global lock,
and the pydantic-core version it was compared against, since a faster
pydantic-core raises every ratio with nothing here changing -- and the gate
reports rather than judges anywhere else. `--update` re-records, and belongs on
the lane that builds the wheel the same way every time: a recording made
elsewhere would match the lane's fingerprint while carrying another machine's
noise.

A tolerance is the shape's own spread across runs of one build, measured per
shape rather than assumed, so `--update` leaves it alone -- one run cannot see a
spread, and a recording that narrowed it silently would fire on noise. A shape
whose spread is too wide to ratchet at all carries a written reason instead of
an absent number: `error_report` spreads a third of its own value, being the one
shape timing a path that raises and formats a Python exception, and a row holds
every shape to a tolerance or an argument in both directions.

**The ratchet beside the ceiling is armed from the lane.** The comparison gate
holds each shape to a ceiling, and beside it holds the shape to the ratio it
*last measured* -- which is the ratchet, and which needs a recording to arm.
A ratio belongs to the environment it was taken in (the interpreter's minor
version, whether it has a global lock, and the pydantic-core on the other
side), and `scripts/compare_gate.py` refuses to judge across those rather than
compare a number about somewhere else. So the recording is made where it is
read: run the CI workflow from the Actions tab with **`record_compare`**
checked, take the `perf-compare-recorded` artifact, and commit
`scripts/perf_compare.json` from it. The lane cannot commit, which is the point
-- a floor is a thing somebody chose. A run that reads the profiled build
against the plain one is the run to tick it on: both readings then come off one
box, one image and one toolchain.

**What a profile buys is read from the same lane, and it is per shape.**
Profile-guided optimisation arranges what fat LTO left to arrange, so a shape
whose cost is one hot loop over one element type is laid out straight and a
shape whose cost is fifty key lookups -- each dispatching on its field's own
schema -- can be laid out worse. A single figure for it would therefore be true
of one shape and false of the next. Run the CI workflow with **`pgo_compare`**
checked: the lane builds both wheels from one source, times every shape on each
with `scripts/pgo_compare.py`, prints the table into the run summary, and
uploads the two readings. It runs on both interpreters the bench jobs use,
because the shapes a profile serves best are the per-element ones and a global
lock is what those pay. The release matrix's `pgo: true` is decided on that
reading, and `docs/11-performance.md` carries the decision with its reason.

**The lane names the interpreter these are read on**, which is CPython 3.14,
and it is written in `ci.yml` rather than left to the runner image: a ratio
belongs to the pair of libraries *and* the interpreter running them, and a lane
that inherits one from an image makes claims nobody chose. The same holds for
the instruction budgets' bands, which cover the distance between two releases,
and for both mutation sweeps, where a mutant on a version-gated branch is
killable on the interpreter that takes the branch and unviable on the one that
compiles it out. `tests/test_lane_interpreters.py` holds every lane to naming
one.

**The bench job's build cache is keyed on that interpreter too**, and a change
that moves the lane moves the key with it. The binding workload links whichever
interpreter `PYO3_PYTHON` names, and pyo3 resolves it again only when that
variable's value changes; the lane's value is `.venv/bin/python` whatever
release is behind it. A cache saved on another release therefore links HEAD's
workload against that release, while the base, built in a fresh target
directory, links the lane's, and every binding shape reads the distance between
two interpreters as the change's own. The `python` matrix keys its cache on the
interpreter for the same reason.

The **free-threaded** build is held to its own set, in the same file, for the
shapes where it is a different environment rather than the same one on a slower
clock: reading an element out of a mutable container takes that container's lock
there, so a shape whose cost is per-element pays what no interpreter with a
global lock pays. A schema nested twenty-five deep is twenty-five
single-element lists, and it reads 0.26 to 0.27 against the 0.13 to 0.15 of the
builds with a lock. A shape absent from that set is held to the shared ceiling,
and the gate selects between them by asking the interpreter whether its global
lock is enabled.

It asserts each payload is **accepted** before timing it. A correctness
regression that made valgebra reject the data would take the fast reject path and
read as a speed-up; that check is the difference between measuring the accept
path and measuring nothing.

### The mutation ratchets

Three sweeps, each with its own committed baseline: the core crate, the
binding's soundness surfaces — the files the `--file` list of
`mutants-diff-walk` in `ci.yml` names: the membership walk with the context it
carries and the precompute it reads, the `Value` both input paths run over, the
frontend and the four surfaces beside it, equality, the oracle and the failure
codes — and the files the shipped extension is the only caller of.
`scripts/mutation_gate.py` fails in **four** directions — a survivor the
baseline does not accept, an accepted entry that no mutant answers to, an
accepted entry naming a file the tree does not track, and an accepted note
naming no survivor in the baseline. The second keeps the accepted set honest: an
accepted hole the tree does not have silently re-accepts a future survivor with
the same identity. The third is about the key rather than the entry: a baseline
keyed by path goes stale when the path moves, and splitting one module into
several moves every entry that file holds at once. The sweep reads each of its
survivors as new, minutes into a shard, with the mutants listed and no hint that
what changed was the path. Checking the path costs nothing and runs before the
sweep. The fourth holds the arguments beside the set: a note whose mutant a
test now kills, or one spelled as no survivor line spells it, is an excuse that
outlived what it excused.

**Run it on the interpreter the lane names.** The verdict is the embedded
interpreter's: a mutant this box's 3.14 reports as a survivor is one CPython
3.12 kills, so a disagreement with the lane is the interpreter before it is the
tree. The lane pins 3.12, so a local sweep does too:

```bash
export PYO3_PYTHON="$(uv python find 3.12)"
export LD_LIBRARY_PATH="$("$PYO3_PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))'):$LD_LIBRARY_PATH"
# Bound what a failing test shrinks, and draw a seed the run can report. Without
# the first, a mutant the tests caught spends the whole budget shrinking a
# counterexample nobody reads and returns a timeout instead of a verdict.
export PROPTEST_MAX_SHRINK_TIME=1000
export PROPTEST_RNG_SEED="${PROPTEST_RNG_SEED:-$RANDOM}"
cargo mutants --package valgebra-py --file <the files the change touches> \
  --features interpreter-tests -j 4 --timeout-multiplier 20 \
  --output sweep -- -- --skip recursion_deeper_than_the_bound_is_refused \
  --skip the_two_readings_agree_at_the_walks_depth_bound
python scripts/mutation_gate.py --baseline walk --new-only --out sweep/mutants.out
```

The core sweep is the same shape without an interpreter to point at, and its
skips are the two termination proofs:

```bash
export PROPTEST_MAX_SHRINK_TIME=1000
export PROPTEST_RNG_SEED="${PROPTEST_RNG_SEED:-$RANDOM}"
cargo mutants --package valgebra-core --file <the files the change touches> \
  -j 4 --timeout-multiplier 20 --output sweep \
  -- -- --skip deep_subtype_into_bottom_terminates \
  --skip subtyping_terminates_on_a_distributed_tower
python scripts/mutation_gate.py --baseline core --new-only --out sweep/mutants.out
```

The target is never zero. Equivalent mutants exist and are undecidable in
general, so an accepted survivor carries the argument for why no test can kill
it, in the baseline beside it.

**Read a mutation score with its skip list.** A test that exists to prove a
bound runs past any timeout the sweep sets under a mutation that removes the
bound, so the whole run returns no verdict; so does one that walks a union of
two walked branches to the depth bound, under a mutation that makes the union's
deciding walk explain. Each such test leaves the *sweep*
and stays in the test lane, marked `SWEEP-SKIP` in its own source with the
reason; `tests/test_sweep_skips.py` owns the list and holds the marks and the
workflow's skip list to each other in both directions. A mutant whose
experiment cannot finish is a rig fault, not a detection -- and a mutant the
skipped tests would hang on is still caught by the rest of the suite, which is
what the bound's own tests are for.

`scripts/mutation_gate.py` reads that distinction rather than restating it. A
line in `timeout.txt` stops the gate at exit 2, the code for a run that could
not measure, with the mutant named; only `missed.txt` reaches the ratchet. The
two are repaired in opposite directions -- a survivor wants a test, a timeout
wants a budget, a seed or a bound on what the tests shrink -- and folding them
together let `--update` write an overloaded runner into a baseline as a
permanent excuse for a mutant nobody judged.

**The third sweep runs the Python suite per mutant.** Seven files of the
binding are reached only through the shipped extension, and `cargo test` never
loads it, so an ordinary sweep reads every mutant of them as a survivor while
measuring nothing. `.cargo/mutants-pytest.toml` examines exactly those files
with a test command that does load it: behind the `pytest-sweep` feature,
`crates/valgebra-py/tests/pytest_sweep.rs` rebuilds the extension from the
mutated copy and runs the suite against it.

The environments live **outside** the tree, because a sweep runs from a copy: a
path inside one names a different directory per mutant, and the copy's own build
would write into it. `VALGEBRA_SWEEP_VENV` says where, and the wrapper fails
rather than skipping where the variable is unset -- a sweep that lost it would
report every mutant caught while running no suite, which is the reading this
whole configuration exists to end.

**One environment per worker.** `-j 2` runs two workers in two copies of the
tree, and two workers building the extension into one environment write the same
files at the same moment: `File exists`, and the run ends a third of the way
through with no verdict. The wrapper keys the environment on the checkout's own
directory name, which a sweep makes unique per worker, and builds it from the
lock file the first time that worker asks. Resolving the lock once up front is
what makes each of those cheap.

```bash
export VALGEBRA_SWEEP_VENV="$PWD/../sweep-venv"
UV_PROJECT_ENVIRONMENT="$VALGEBRA_SWEEP_VENV" uv sync --locked --no-install-project
export PYO3_PYTHON="$VALGEBRA_SWEEP_VENV/bin/python"
export LD_LIBRARY_PATH="$("$PYO3_PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))'):$LD_LIBRARY_PATH"
export PROPTEST_MAX_SHRINK_TIME=1000
export PROPTEST_RNG_SEED="${PROPTEST_RNG_SEED:-$RANDOM}"
cargo mutants --config .cargo/mutants-pytest.toml --package valgebra-py \
  --features pytest-sweep -j 2 --timeout-multiplier 20 --output pytest-sweep
python scripts/mutation_gate.py --baseline pytest --out pytest-sweep/mutants.out
```

Budget an hour a shard: a mutant costs a rebuild and a suite run, about a
minute each, against the seconds an ordinary mutant takes. The lane shards it
six ways and is scheduled only.

**The walk sweep links one interpreter.** `cargo test` for the binding embeds
the interpreter `ci.yml` names for that lane (CPython 3.12), and a survivor's
note in `scripts/mutation_baseline_walk.json` is an argument about *that*
interpreter: a mutant two spellings of `typing.Union` cannot tell apart on 3.14
is caught where they are two objects, and the ratchet reads it as caught. The
same holds for what the lane's environment installs. The embedded interpreter
imports from the lane's venv, where the lock puts `typing_extensions`, and on
3.12 that module's `Required`, `NotRequired` and `Unpack` are `typing`'s own
objects. So the frontend's arms for the module's own objects are killable only
where it defines them -- 3.10, or the corpus row's stand-in on an interpreter
that cannot import it -- and three notes in the baseline say so. A reading only
PyPy takes is compiled for PyPy alone, `#[cfg(PyPy)]`, so the sweep's build does
not compile it: every mutant of it builds and passes and says nothing, and
`exclude_re` names them in `.cargo/mutants.toml` beside the PyPy lane's rows
that hold them. Compiled everywhere, the same code was regions the binding's
coverage floor counted and no CPython test could reach.

**A reading compiled per interpreter names what it answers, and is measured on
one side.** `crates/valgebra-py/build.rs` re-emits the interpreter's `Py_3_x`
and `Py_GIL_DISABLED` flags, so a `cfg!` on one compiles a different reading per
interpreter. Such a site says in its doc comment which behaviour of which
release it answers, with the measurement for each reading; `snapshot_pays` in
the sequence walk is the one site. Each lane that measures reads one side: the
binding sweep and the coverage lane pin 3.12, where the snapshot reading
compiles in, and the bench lane pins 3.14, where the reading in place does, so
the instruction gate holds the cost of that reading and of no other. The
product suite holds the answers on every interpreter, since the readings answer
alike. What no lane holds is the cost of the snapshot reading and of the
free-threaded one, which is the doc comment's measurement and nothing more.

## A gate that compared nothing must not pass

"No mismatches" is true of an empty corpus. Every gate here refuses that shape
rather than publishing a comparison it never made: the perf gate on an
unreadable count, the compare gate on a rejected payload, the ratchet on an empty
output directory, the doc lint and every ledger on an empty universe.

**Every numeric gate has been seen to fail**, in a committed test. That is the
standard: a detector that cannot be shown to fail is not evidence.

## The lanes

`.github/workflows/ci.yml` runs them; read the job set there. Five properties of
the arrangement are worth stating because they are decisions rather than
mechanism:

**`main` receives a commit the push lane passed.** A push to `github_ci` runs
the workflow, and `main` moves to a commit only once its `ci` check is green
there, by a fast-forward to that same commit, so the hash the check was read on
is the one `main` carries. A red push lane is repaired on `github_ci`. A push
to `main` runs the workflow too, and the nightly runs on the default branch,
over what has landed.

**A skipped job cannot pass.** The `ci` aggregator lists every required job in
`needs:` *and* fails unless each result is `success` rather than merely
not-failure. The duplicated list is a deliberate second copy.

**A cancelled job is red.** A job that reaches its timeout is reported
`cancelled` rather than `failure`, and a job nothing waits on is cancelled in
silence -- so a ratchet behind a timeout runs nowhere while every lane reads
green. The aggregate therefore waits on the scheduled
jobs too, allowing one answer more from them than from the others — `skipped`,
which is what a push gives a job it does not run — and refusing everything
else. `tests/test_required_jobs.py` reads which jobs those are from their own
conditions and holds each reading to the kind of job it is.

**Every supported interpreter runs on every event.** The release ships a wheel
built per version against a version-specific ABI, so each interpreter is a
separate artifact a caller installs, and a leg that runs only at night is a
wheel nothing exercised until somebody reported it. `ci.yml` owns the matrix.
`tests/test_python_lifecycle.py` holds the releases it carries to the schedule
each follows, and the floor `requires-python` claims to the oldest of them;
`test_every_supported_interpreter_runs_on_every_event` in
`tests/test_required_jobs.py` holds that the job runs on every event and that
the free-threading classifier has a free-threaded leg behind it.

**One leg carries the history, and its name says so.** `actions/checkout`
takes one commit and no tags, and the checks that read `git log` back to the
last release tag -- the changelog roll, the commit messages since that tag, and
the reachability of every commit a page cites -- stand down where they cannot
read. Every leg of the matrix but one is that clone. The floor leg takes
`fetch-depth: 0` so those run somewhere, and it is *named* for it, because a
column of results with the same shape of name is a checks list nobody can read
the difference off: the one that matters is the one that is not skipping.
`tests/test_changelog_ledger.py` holds the depth and the name to the **same**
condition, since a name that advertises a history the leg no longer takes
answers the reader's question wrongly, which is worse than not answering it.
The same leg installs `cargo-mutants` at the version the sweeps pin, so the two
checks in `tests/test_mutation_scope.py` that list the mutants a sweep is
offered, and skip without it, run there too.

**The manylinux wheel is installed on every push.** The `wheel (linux)` job
builds the manylinux wheels from the image a release builds them in, then
installs each CPython wheel on the interpreter its ABI tag names and imports it
with every warning an error; the PyPy wheel the image also builds is the
`pypy 3.11` lane's, which builds and runs one of its own. The lanes test the extension `maturin develop` builds, so a
wheel that builds and does not load would otherwise wait for the release smoke.

**PyPy builds, links, and runs the suite.** The release matrix publishes four
PyPy 3.11 wheels, and a push that does not link against PyPy cannot see what
breaks there: `cpyext` carries the limited API and not every static type object
CPython exports, so an extension naming one links on CPython and fails at
`import` on PyPy. The release smoke runs the suite on every glibc, macOS and
Windows wheel it ships, the PyPy ones included -- the musllinux wheels are the
set it does not reach ([09-releasing.md](09-releasing.md)) -- but a release is
where a break is dearest to find, so the push runs one first. The `pypy 3.11`
job builds a release wheel against PyPy and runs `scripts/pypy_import_check.py`
first, which imports it and builds the annotation forms whose compilation
reaches a type object: that is the *link*, and it fails with one line naming the
form rather than in a stack of test output.

Then it runs the suite, with the packages it reads installed by a list in the
step rather than by the dev group, which carries the checkers and the build
tools too. `tests/test_suite_installs.py` holds that list, and the release
smoke's, to the dev group less the tools it excuses by name: a package left
off would not redden the lane, because a row reading an optional
implementation skips where it is absent.

The link is not the only property that differs there: `cpyext` implements
`PyTuple_Size` through the object's own `__len__`, so a walk that trusts that
accessor for a `tuple` subclass overriding it reads past the end of the storage
and takes the process down — an answer, not a symbol, and an import cannot see
it.
`tests/test_lane_interpreters.py` holds the rule both ways: an implementation
the packaging classifiers state has a lane running the suite, and a lane running
the suite is on an implementation somebody stated. The wheel is a **plain
release** build, because the frame size decides whether the deep-nesting cases
overflow the native stack budget `cpyext` sizes from the recursion limit: a
debug build's frames are large enough, and so are a profile-guided build's,
which is why the PyPy wheels the release ships are plain builds too and the
release smoke runs the suite on them ([09-releasing.md](09-releasing.md)). That
is a property of the build profile and not of the code.

Three cases the suite carries cannot be decided there and say so rather than
failing. `cpyext` builds a `PyTypeObject` proxy for every class an extension is
shown and never frees it, so a class that crosses the boundary is immortal
whatever a validator holds; the collector traces every object rather than
flagging some, so there is no per-object tracking to assert; and the interpreter
hands out one `nan` object where CPython hands out two. Each is read by *trying*
it rather than by naming an interpreter, so a release that changes one puts the
case back in the run without anyone editing a list.

**The full sweeps are scheduled, and a diff-scoped one is not.** A full sweep is
minutes of rebuilds and does not belong on a push, so a regression it catches is
visible the night after. Each full sweep runs sharded -- the core, the walk and
the pytest sweep alike, as the diff sweeps do (the shard counts are `ci.yml`'s)
-- each shard reporting its own slice; a job after them merges the slices and
ratchets once, since a survivor is a survivor
of the *sweep* and an entry that survives nothing is known to only when every
shard has reported. A shard's ceiling (`timeout-minutes` in `ci.yml`) is twice
the slowest shard's reading, because a job that reaches its ceiling is
cancelled and a cancelled job is red. Every sweep shards round-robin: cut in
consecutive slices, a file's mutants fall to one shard, and one whose mutants
build and run costs several times one whose mutants mostly do not compile --
the binding's push sweep read 15, 19 and past 30 minutes that way. Every push runs the core and walk sweeps
restricted to the **whole files the change touches** — bounded by the change
rather than by the tree — and blocks the merge; the pytest sweep costs a
rebuild and a suite run per mutant and stays on the schedule. A push sweep
checks the new-survivor direction alone, because a partial sweep never
generates most of the baseline.

Whole files, not the diff's lines: the miss that matters is an edit that stops an
**existing** test from killing a mutant elsewhere in the same file, and that
mutant is not in the diff.

**A change starts where its commits leave the history they replace.** The
bench gate's `rev` and the sweeps' file list come from one base, which
`scripts/change_base.py` names: the event's base (a pull request's, or the
`before` of a push) taken to its merge base with `HEAD`. A force-push names a
`before` no branch reaches; the script fetches it by id where the host still
serves it, so a rewritten history is measured from its fork point, and falls
back to the default branch's tip -- which, pushed to, is `HEAD` itself -- and
from there to `HEAD`'s parent. Measured from itself, an amended commit sweeps
`core files: none` and passes. `tests/test_change_base.py` drives each case
over a synthetic history, and holds every step reading the event's base to the
script.

**A slow network is not a red lane.** `astral-sh/setup-uv` reads its version
manifest from `raw.githubusercontent.com` under a hard five-second timeout and
never retries, so a slow response there does not make a job slow — it fails it,
and it fails that one job while every other lane in the same run installs uv
fine. Every lane installs through
`.github/actions/setup-uv` instead: it pins the uv release, which is one fewer
thing resolved over the network, and makes a second attempt when the first one
fails. The release workflow's two smoke jobs stay on the upstream action, because
they install uv before the repository is on disk -- their checkout comes later,
into `tree/`, for `tests/` alone -- and a local action needs its own files on
disk.

**A toolchain comes from rustup, not from an action.** A third-party action is
pinned by hash with a comment naming its tag, and `zizmor` checks that the tag
still names the hash. A toolchain action that publishes one moving tag moves it
with every upstream change, so the pin drifts from its comment and the audit
fails every lane behind it until somebody re-pins. Every lane installs through
`.github/actions/setup-rust` instead, which runs `rustup toolchain install` and
`rustup default` itself -- the runner carries rustup -- with the one retry the
release server needs while it publishes, and leaves no tag to drift. The
release workflow's sdist smoke job runs the same two commands as a plain step,
because nothing is checked out when it installs. `tests/test_workflow_actions.py`
fails on a toolchain installed by any third-party action.

**A lane syncs once and runs without re-resolving.** A bare `uv run` resolves
the environment again first, which uninstalls the editable build `maturin
develop` put there and installs the wheel from the lock, so every command after
it tests a module the lane did not build. Every lane installs what it needs with
one `uv sync --locked`, the dependency groups it names included, and runs every
later command under `uv run --no-sync`; `tests/test_workflow_syncs.py` holds both
halves over every workflow, the Pages deploy included. Dependabot proposes a release seven days
after it is published, so a version pulled in its first week never reaches a
pull request, and `zizmor` audits the whole of `.github/`, where that setting
lives.

**Every gate script runs in a lane.** `tests/test_lane_coverage.py` holds each
executable under `scripts/` to being driven by a workflow, by the packaging
config, or by the suite — or excused by name with a reason, on a list that fails
if the excuse goes stale in either direction. A script in no lane is not a gate.

## What each instrument cannot see

- **A line coverage floor cannot see a wrong answer.** It says a line ran, not
  that anything checked what it did. That is what the mutation sweeps are for.
- **A crate-wide floor cannot see one file.** `--fail-under-lines` and
  `--fail-under-regions` are read against the total, so a file sits under the
  floor for as long as the rest of the crate carries it. A module reading 84%
  of its lines and 86% of its regions, in a crate held to the floors the
  `rust coverage` job in `ci.yml` names, moves the total by a hundredth of a
  percent, so both floors, the mutation ratchet and the whole product suite
  stay green on the merge path while a scoped sweep of that module reports
  every mutant surviving.

    `scripts/branch_coverage.py` holds a region floor per file and refuses a
    file the measurement carries and no floor does. It runs on the **nightly**
    lane, on a pinned nightly toolchain, so it answers after a change has
    merged rather than before.

    So the merge path asks the same question of its own profile.
    `scripts/coverage_gate.py` carries a floor far below the crate's, because
    what it detects is a file nothing drives rather than a file that could be
    driven harder, and the files under it are named with the reason each is
    there. The list may only shrink: a file that climbs over the floor fails
    the gate until its entry goes, which is the rule every excuse in this tree
    is held to. The two per-file floors are not one check twice: the nightly
    one ratchets a recorded figure and moves up with the measurement, this one
    is a fixed floor a file clears or is named under.

    The floor is **per scope**, for the reason the crate floors already are,
    and then lower again by more than a figure moves between machines. One
    binding file read 87.65% of regions on a developer's machine and 84.77% on
    a runner, so a floor set within three points of a measurement reports which
    machine ran it. A file nothing drives reads near zero, which is what leaves
    the room to be that far under.
- **A coverage figure read from a shared target directory is not a figure.**
  `cargo llvm-cov` merges the profile against every instrumented object the
  directory holds, and an object built from an earlier tree contributes a
  mapping with zero counts for every function whose body has changed since. The
  result reads as unexecuted code and is not: on one machine `decision.rs`
  reported half its regions covered, with seven hundred of them landing on doc
  comments, blank lines and `use` items, while a fresh `CARGO_TARGET_DIR` on the
  same toolchain read the same file at 99.45% and the lane total at 97.34%.
  The tell is the report's function list, which named four copies of the crate
  in the shared run and two in the clean one.

    So each lane cleans before it measures, and a figure taken by hand is taken
    the same way. The binding lane has the same trap one layer down: `maturin
    develop` into a directory that already holds the plain extension reuses it,
    the suite then runs against a module with no coverage map, and the lane
    reads twenty points low.

- **A mutation score is a statement about one test command.** The core and walk
  sweeps run `cargo test`, so a survivor in either is a gap in the *Rust-side*
  corpus; the pytest sweep runs the Python suite, so a survivor there is a gap
  in that suite.
- **The walk sweep needs an embedded interpreter.** Without
  `--features interpreter-tests` every mutant reads as unviable, which is a sweep
  that measured nothing rather than a clean one.
- **A mutation sweep cannot see a rule nobody wrote.** A mutant is a change to
  code that *exists*; a missing match arm is not a mutation of anything. An
  adequacy figure of any size says nothing about a rule that was never there,
  which is the one defect the sweeps are most often assumed to cover.
- **`scripts/docs_lint.py` cannot tell you a sentence is false.**
  [12-writing.md](12-writing.md) names the classes it cannot reach.
- **A fuzz run that finds nothing means "nothing failed inside that budget"**,
  never "there is nothing to find". That is why it is not a merge gate. The
  budget carries a floor for the half of that a lane *can* tell apart: a soak
  that ran is distinguishable from one that only ended, so the rate and the
  duration the soak prints are read back and a run beneath the floor is a rig
  fault rather than a clean sheet.

    An out-of-memory artifact from this target is read the same way before it
    is believed. libFuzzer's unnamed process ceiling fires against whichever
    input happened to be running when the *process* crossed it, which is not a
    claim about that input: the two artifacts from 2026-09-08 replay in 0 ms and
    2 ms under the allocation bound, and are in `fuzz/seeds/decision/` as inputs
    rather than in `fuzz/artifacts/` as findings. `-malloc_limit_mb` is the flag
    that names a defect, because it fires on an allocation.
- **The binding gate cannot see what the extension pays for a thread-local.**
  Its workload is an executable, which reads a thread-local with one load off a
  segment register; the extension is a shared object, which reaches one through
  `__tls_get_addr`, a call into the dynamic linker of a dozen instructions. So a
  cost that lives in that call reads 0% on `scripts/perf_gate.py` and several
  percent in the extension: a drop of a PyO3 error inlined into a loop has its
  thread-local address taken ahead of the loop, on every value. Count it
  against a `--profile profiling` build of the extension itself, as
  [the performance page](../11-performance.md) does, and read the function's
  disassembly for a call to `__tls_get_addr` on its accepting path.
- **A soundness property has nothing to say about a `False`.** A law shaped
  `if a.is_subtype_of(&b) { ..check.. }` never examines the answers that are
  wrong in the conservative direction, and those are the majority of them.
  `tests/test_completeness_probe.py` is the instrument that faces that way, and
  its own blind spot is a gap no value in its universe can expose.
