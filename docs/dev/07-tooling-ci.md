# Tooling and CI

The gates, the lanes, what each instrument cannot see, and the exit code that
says which of the three things happened.

## Check the exit code, never a piped fragment

```sh
python scripts/perf_gate.py --against HEAD | tail -1   # WRONG -- reads 0 from tail while the gate is red
python scripts/perf_gate.py --against HEAD; echo $?    # right
```

## Three outcomes, three exit codes

| Outcome | Exit | Means |
|---|---|---|
| pass | 0 | the gate ran and the property holds |
| fail | 1 | the gate ran and the property does not hold |
| **could not run** | **2** | no measurement, no baseline, a missing dependency |

A gate that could not run has proven nothing and must never read as one that
passed. `scripts/perf_gate.py` exits 2 on cachegrind output it cannot parse and
on a build of the change that does not finish, `scripts/mutation_gate.py` on a
missing sweep result or baseline, `scripts/use_case_ledger.py` on a missing
record, `scripts/run_doc_examples.py` on a tree holding no example to run,
`scripts/gate.py` on a clone that does not land, `scripts/compare_gate.py`
without its benchmark dependency, `scripts/docs_lint.py` on a tree it cannot
read. A traceback exits 1, the code a failed property reads as, so each of
these is caught where it happens and named.

**A command line it cannot read is one it could not run.** Every script reads
its flags through `argparse` before doing any work: `--help` prints the usage
and exits 0, and an unknown flag, a missing value or an unknown choice exits 2.
Read by substring, a mistyped flag is no flag at all, and a gate asked for one
measurement answers for another. `tests/test_script_arguments.py` holds every
file under `scripts/` to it.

**The codes, the root and the parser are said once**, in `scripts/_lib.py`,
with `git_output`, which asks git a question of a directory the caller names,
and `is_test_module`, which reads whether a Rust file is test code from the
`mod` that declares it.
A script runs with its own directory first on the import path, so it finds the
module by name. A test or the benchmarks' conftest loading a script by path
puts the directory there first, and a test copying a script into a tree of its
own copies the module too, since a script's root is where `_lib.py` sits.
mkdocs loads `docs_stubs.py` by path as a griffe extension, with no loader of
ours between, so that script puts its own directory on the path. The leading underscore says the file is
no script: no lane runs it, it has no contract row and it takes no flags.
`tests/test_script_arguments.py` refuses a script that defines a code, `ROOT`
or a parser of its own.

What a script runs a step or a measurement in is not shared, because the two
are different rules. `gate.py` gives a step the caller's environment, less the
color a terminal forces and the venv `uv run` names, and over it what a runner
hands the step: the workflow's `env`, its job's and its own. `perf_gate.py` gives a
workload the few names it needs, with the hash seed and the allocator fixed,
since the environment's size moves a count. Either rule applied to the other's case fails a test:
`test_a_forced_color_variable_does_not_reach_a_step` needs a caller's variable
kept, and `test_a_workload_runs_without_the_callers_environment` needs it
dropped.

## The local gate, and the contract inventory

**`scripts/gate.py` runs the lane's steps, not a list that resembles them.** It
reads every `run:` step of every job the `ci` aggregator waits for and a push
runs out of the workflow -- `SCHEDULED_ONLY` drops the nightly jobs and
`RUNNER_ONLY_JOBS` the ones left out whole -- and runs them in a fresh
**shallow clone of `HEAD` with no tags**, which is what `actions/checkout`
produces and what a developer's clone is not. A step that needs a runner
(valgrind, a base built beside the head, a mutation sweep, the bench lane's
optimized wheel) is named in `NEEDS_A_RUNNER` with the
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
`HEAD` itself; `perf_base` in `scripts/gate.py`), and it runs in the clone that
keeps the history. `perf_gate.py` measures the checkout it runs in as the
change, so the caller's tree would add what is not committed, and the shallow
clone does not hold the base; under `--here` the caller's tree is the tree
every step reads. The base is named in the caller's tree, where `origin` is the
remote. Excused with the rest of the lane, a change
that reads sound and costs **seventy-one times** the instructions on the
relation matrix would pass this script green. One mode runs, `--decision-matrix`
(`PERF_STEPS` in `scripts/gate.py`), the relation matrix whose shapes reach the
set representation; `--decision-sets` and `--decision-lengths` reach it too and
are read in the lane alone, and the `--binding-*` modes want the extension
built into an interpreter and stay with the lane. Missing valgrind is named as the reason rather than passed
over.

**The floor interpreter is built beside the caller's, and the suite runs on
it.** The matrix runs every supported release -- `ci.yml` owns the list -- and
a developer runs one, so every difference between two of them is a difference
the gate could not see -- and they are not rare: a typing member the floor does
not carry collects on the caller's release and fails collection on the floor,
taking every job there with it. The floor is the end of the range where that
lands, because the suite is written on the newest release the tree supports and
read on the oldest. So the gate makes a second environment on the floor
`ci.yml` names, builds the extension into it, and runs there what a CI leg
other than the floor reads: the **product** suite, and the repository checks
marked `interpreter`. The rest of the audit reads the tree and answers the same
on any release, so it runs once. Which release the floor is stays in `ci.yml`,
as its `FLOOR`: the gate reads it, and `tests/test_local_gate.py` refuses a
copy of the number in
`scripts/gate.py`, because a stale floor is the one kind of stale that tests
less than it claims while staying green.

**What the clone models, the environment does not.** A step gets the caller's
environment plus the job's and the step's own `env:`, and the runner's own
absences are not modeled by that -- so a variable a developer's terminal sets
reaches a step that a lane runs with it unset. One of them gives a false red:
`FORCE_COLOR` overrides a tool's terminal check, so `uv export` writes escape
codes into the requirements file it generates, and `pip-audit` refuses the file
against a dependency tree with no advisory in it. The two variables whose only
purpose is that override are dropped (`runner_environment`), and so is
`VIRTUAL_ENV`, which `uv run` sets for the gate it launches: in the clone it
sends a step's `uv pip install` to the caller's venv while `uv run` reads the
clone's, so the wheel the profile comparison times is installed where the
timing step cannot import it, and the caller's venv keeps a wheel built from a
clone of `HEAD`. Without it, `uv pip` finds the clone's `.venv` from the
working directory, as a runner's step does. The venv's `bin` leaves `PATH` with
it, and the clone runs the lane's own sync and build rather than excusing them,
because every later step runs under `uv run --no-sync`: with neither, a clone
with no environment of its own would find the caller's pytest, which imports
the caller's package and reports green on the caller's tree.

**The whole list, since any one of these can cost a day.** The table is what a
lane differs from a local run in, and the `modeled` column is what the gate
reproduces; the count is the table's rather than this sentence's:

| difference | modeled | what it costs when it is not |
| --- | --- | --- |
| the clone: one commit, no tags | yes, `shallow_clone` | a ledger reading `git describe` passes here and reddens every job that runs it there |
| the step's declared `env:` | yes, since the plan carries it | without `RUSTDOCFLAGS=-D warnings`, `cargo doc` cannot fail locally |
| the terminal's own variables | yes, `runner_environment` | `FORCE_COLOR` turns `pip-audit` red against a clean dependency tree |
| the machine's git identity | yes, `deep_clone` and `NO_IDENTITY` | a checkout configures no `user.name`, this repository has one in its own `.git/config`, and a ledger that plants a commit with `git commit-tree` passes here and fails there |
| the interpreter the lane names | partly: `PYO3_PYTHON` and every CPython job's `uv` follow the caller's release, since the jobs share one clone, and a job on another implementation takes the interpreter its uv setup names; the gate's closing line says so | the binding's mutation lanes name CPython 3.13 in `ci.yml` and the cachegrind lane 3.15; a local sweep on 3.14 can read as survivors mutants the lane kills, a difference that is the interpreter's |
| the *release* of the interpreter, not only the caller's | yes, at the floor: the gate builds it beside the caller's and runs the product suite and the checks marked `interpreter` on it | `typing.Self`, `LiteralString` and `Unpack` are 3.11 members; a test module naming them collects here on 3.14, fails to collect on the floor, and takes every job on it red |
| the operating system and architecture | no | a macOS or Windows leg fails where Linux does not, and nothing local sees it |
| the pinned tool versions | no, for the tools `taiki-e/install-action` installs | `ci.yml` installs `cargo-deny`, `cargo-llvm-cov` and `cargo-mutants` at versions it names, and the gate runs whichever is on the caller's `PATH`; a `uvx` step carries its pin in its own command, so the gate runs the lane's version of it |
| the build of the interpreter, not only its version | no: a rule answers it instead | `sys.stdlib_module_names` is the build's, not the release's -- one 3.12 build lists the Windows-only `_wmi` and a runner's does not, so a table of every name would report a difference between two builds as a moved row. The floor table records the modules this tree imports, which are portable by construction |
| the machine's own speed | not modeled, and not a gate: every merge-blocking number is an instruction count | an absolute count reads 5--8% apart between two machines on one `rustc` line, which is why a recorded count is read only where there is no base count, inside a band that wide |
| secrets, tokens and the event payload | no, and the steps that need one are excused by name | the merge base comes from the event, so `--against` runs only in the lane |
| what a gate **counts**, against what it claims to | no: a scope is a claim in prose, and no check reads it | a coverage floor that counts the instruction gate's own workloads, which no suite runs, reddens over a change that adds no uncovered line of its own; a shape that counts its harness reports the harness |

The first four are closed. Named rather than numbered from here, since the
table grows: *the interpreter the lane names* is a row rather than a fix
because the gate runs the caller's toolchain by design, and the gate's own
closing line names it; *the release of the interpreter* is closed at the
floor, which is the end of the range a difference lands at; *the build of the
interpreter* is answered by a rule rather than by the gate; the rest are
differences a local gate cannot remove, and naming them is what keeps a green
local run from being read as a promise it never made.

Naming them carries an obligation the other way: a job on that closing line
can go red within two minutes of a push, on a `docker pull` that answers `504`,
and the only thing that notices is a person reading the badge. A green gate
followed by a red lane it named is the expected shape, not an anomaly — so a
lane the gate does not reach is read after the push and its verdict reported,
with a re-run, a fix, or a stated reason it is not the tree's. The list exists
to say what a green run did not check; leaving the check undone afterwards
spends the list on nothing.

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

One line in it is easy to miss, and missing it turns a lane red: **the fuzz
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
the license header. The annotation and docstring rules apply to the shipped
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
pyright's default moves nothing here. The `python` job's floor leg runs it and
`mypy --strict`
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

### The instruction gate

Wall-clock benchmarks on shared runners are too noisy to gate, so
`scripts/perf_gate.py` gates **instruction counts under cachegrind**, which are
identical across runs of a build. Three workloads, one per surface, because a
gate only catches what it exercises, and the wheel a release ships:

- the **core** transformations — the constructors' normal form, the composition
  remap, the record transform;
- the **decision** procedures (`--decision`, `--decision-refute`,
  `--decision-repeat`, `--decision-matrix`, `--decision-rules`,
  `--decision-sets`, `--decision-lengths`) — subtyping, emptiness,
  equivalence; `MODES` in
  `scripts/perf_gate.py` owns the list, and the first three are the
  workloads one whose relations hold, one whose relations are refuted, and one
  whose goals repeat. The core workload never calls a decision, so without
  these the whole decision surface is unmeasured in both directions: neither
  what a new rule costs nor what a cheaper one saves. Three because a proof, a
  refutation and a repeated goal walk different paths, and a workload that
  asks only for proofs holds a refuting rule to nothing. Two are one example
  asked of the rules and of the sets, so a lowering made dearer or cheaper
  reads on its own shape rather than as a share of the matrix's, and the last
  asks the sets length bounds over a list, whose automaton holds a state per
  element, so a cost an element pays reads as one;
- the **binding** shapes, which `MODES` in `scripts/perf_gate.py` owns (`--binding`, `--binding-boundary`,
  `--binding-record`, `--binding-keys`, `--binding-open`,
  `--binding-record-list`, `--binding-mapping`, `--binding-mapping-lists`,
  `--binding-subclass`, `--binding-recursive`, `--binding-deep`,
  `--binding-refined`, `--binding-nullable`, `--binding-set`,
  `--binding-json`, `--binding-json-reject`, `--binding-json-union`,
  `--binding-json-open`, `--binding-json-bounded`, `--binding-json-deep`,
  `--binding-pattern`,
  `--binding-build`, `--binding-annotated`, `--binding-object`,
  `--binding-object-walk`, `--binding-protocol`, `--binding-relation`,
  `--binding-explain`, `--binding-explain-accept`, `--binding-explain-list`) —
  membership over a live Python value, the call boundary alone, a wide record
  closed, the same record walked over a value whose keys are interned, the
  record open the way a `TypedDict` is, a list of records, a `dict[str, int]`
  read through its clause, a `dict[str, list[int]]` whose values are walked,
  walking a `NamedTuple`, a recursive schema's descent, a value nested
  twenty-five deep, elements that each read a refinement's bound, a list of
  `int | None`, a `set[str]`, parsing and walking a JSON document -- accepted,
  rejected halfway, as a union of two record kinds, read through a key-type
  clause, the same under a length bound, and against a recursive schema --
  matching a string against a compiled
  pattern, building a validator from its Python spelling, compiling one written
  as a `TypedDict` of refined integers, compiling a fifty-field dataclass and
  walking an instance of it, compiling a fifty-member protocol, relating two
  dataclasses, explaining a failure in a
  record or accepting one in the same mode, and explaining a refused list. The
  walk is the shipped hot path neither pure-Rust workload reaches; a count of
  the walk does not see schema construction, and a count of a closed record
  does not see an open one, which can read a third dearer. Nor does a record
  reached alone see the copy of its check a list of records inlines, nor a
  mapping of two scalar kinds the scan any other mapping takes. Each shape builds what it reads outside its loop: formatting its names
  and filling a dict per iteration would be most of a build shape's count.

  **One accepting JSON shape cannot measure every JSON change**: the paths a
  change can lose on need counts of their own -- the reject a fast walk stops
  at, the union that reads a parsed object once per branch, the clause that
  reads a key rather than a field, the reference on the parsed path, the length
  bound that counts a parsed object's keys. Nor does one walk over plain kinds
  measure the walk: a refined element reads its bound on every check, sixteen
  times the instructions of an element that only has a kind.

  Each shape below counts a cost the others cannot see. **A dict of bare type
  objects is not what the frontend costs**: an annotation with any depth is
  read through `get_type_hints`, asked per field whether a qualifier states its
  required-ness, and asked per marker for four bound names it probably does not
  carry -- and no other shape counts any of that. **A record walk over a dict
  the caller wrote as a literal is not the walk over one built with f-strings**
  either: the probe compares by pointer where both sides are interned, which is
  28% of the call, and the shape that walks non-interned keys moves by a
  percent when the validator's own side stops interning -- inside the relative
  gate's ceiling, and therefore invisible. **And a class is not a dict**: a
  dataclass is the one class form whose compile asks the standard library a
  question, and importing `dataclasses` at the top of the frontend would cost
  a build that compiles none of them 6.45%, which no other shape here moves.
  **And a subclass is not its base**: the walk cannot trust the C length
  accessor for a `tuple` subclass, because `cpyext` answers it through the
  object's own `__len__` -- but it can trust it for one that *inherits* the
  base's slot, which is every `NamedTuple`. Telling those apart by "is this
  exactly a tuple" would copy every `NamedTuple`, at 100 ns against a plain
  tuple's 57 on CPython 3.14, under every green lane. That exact-tuple check is
  the reading's own mutant: it decides the same things for every ordinary type
  -- the copy holds the same elements -- and only a metaclass that raises when
  asked for `__len__` tells the two apart, which is the test that kills it.
  What the reading buys is the cost, and this count holds that: the exact-tuple
  check reads +171.65%.

  **And a shape that wins by a wide margin measures nothing.** The JSON
  document is the comparison gate's closest race, and its ratio moves by a tenth
  with no gate red, because a wall clock under a ceiling it
  clears by a third says nothing until somebody looks — which the section on the
  competitive ratio, below, records as the reason the recorded block exists at
  all. `--binding-json` is that shape's deterministic twin, over the same two
  hundred records read against the same `list[JsonRecord]`. Most of what it
  counts is not this crate's: under callgrind, about 55% of a call is
  `jiter` building the value tree and 14% is dropping it, with the walk in the
  remaining third — and `pydantic-core` parses to the same `jiter` tree before
  it validates, so the two thirds is the floor both libraries stand on and the
  third is what the ratio is about. A walk regression therefore shows here at
  roughly a third of its size, which is the price of measuring the shape a
  caller actually runs rather than a walk with the parse taken out.

  **And a relation is not a decision workload.** The decision shapes relate
  classes that are identities with nothing behind them, so what the
  bindings read off a real class -- the namespaces of its `__mro__`, which a
  refutation about the class stands on -- has no count there. Read whole for
  every class of every query, with `object`'s namespace among them and a
  `hasattr` per value, it would make one relation between two dataclasses six
  times dearer on 3.14 and twenty-two times on 3.12, with every other gate
  green. `--binding-relation` relates two of them through `relation_to`.

  **And a compiled pattern is not a comparison.** Every other refinement costs
  an operator; a pattern costs a compiled object, built once when the
  validator's index is built and found again by the pattern's address. Losing
  that precompute is not a wrong answer — `check/walk/scalar.rs` compiles the
  pattern on the spot and decides the same thing — so no test holds it, and the
  mutation baseline accepts the index's survivor for that arm on exactly this
  argument. What makes accepting it honest is `--binding-pattern`: delete the
  arm and one validation costs 462,318 instructions against 783, which is the
  whole suite passing while every pattern check compiles a regex.

- the **profiled** shapes (`--profiled`) — the comparison gate's seven, counted
  on the profile-guided wheel rather than a plain build. The three workloads
  above build without a profile, so a change the profile lays out differently,
  or a change to what `scripts/pgo_workload.py` trains, moves nothing they
  read. `run_profiled` builds the wheel at the change and at the base, each
  from a snapshot of its tree unpacked into one directory and built into one
  emptied target, because the directory a profiled build runs in moves the
  layout of a loop the training does not reach: a source path one character
  longer moves the untrained JSON document by a tenth. Each shape is
  `compare_gate.py --count` calling its valgebra half, in an environment that
  holds the wheel and nothing else, as the difference of two call counts. Two
  builds of one tree read every shape to the same instruction; with the nested
  list's training taken out of the workload, `deep_nesting` reads 50.89% more
  and the other six move by under half a percent. They are compared and never
  recorded, since a count belongs to a wheel no record could rebuild -- which
  leaves a trade a change makes on purpose no step to excuse it, because a step
  expires by the count its shape records.

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
# The venv's interpreter, on the 3.15 the bench lane pins: the one valgebra is installed in.
export PYO3_PYTHON="$(uv run --no-sync python -c 'import sys; print(sys.executable)')"
cargo build --profile profiling -p valgebra-py --features pyo3/extension-module
pkg="$("$PYO3_PYTHON" -c 'import valgebra, pathlib; print(pathlib.Path(valgebra.__file__).parent)')"
suffix="$("$PYO3_PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
cp target/profiling/lib_valgebra.so "$pkg/_valgebra$suffix"
PYTHONHASHSEED=0 valgrind --tool=callgrind --callgrind-out-file=out.callgrind "$PYO3_PYTHON" probe.py
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

- **A new shape's band is two-sided.** A shape the base does not carry is
  held to its recorded count, and a workload that stopped doing the work
  executes *fewer* instructions, which a ceiling alone publishes as an
  improvement it never earned.
- **The workload proves it ran.** Each prints a checksum folded through every
  result, compared **before** any count: with the base's, and for a new shape
  with the recorded one. The core and decision workloads' are constants of
  their corpus. A binding shape folds the same figure every iteration -- its
  answer beside the size of what it reads, the schema's node count and the
  length of the value's `repr` or of the document -- so its checksum is a
  multiple of the iteration count, and a workload that ignored its argument,
  reporting a difference near zero, fails rather than passing under the
  ceiling. That figure is what is compared with the base's, so a value cut
  short reads as other work rather than as a saving. The decision workload's
  checksum counts verdicts, so a change that decides *differently* fails on the
  checksum before its instruction count is read.
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
  is not differenced, reads 2.97% apart between a login shell and `uv run`.
  `workload_environment` in `perf_gate.py` passes a workload the loader path,
  `PYTHONHOME` and `VALGRIND_LIB` where the caller has them, and sets the fixed
  seed, the allocator tunable and `PYTHONDONTWRITEBYTECODE`, and nothing else; valgrind is resolved to an absolute path first. The same
  binary then reads the same count from any shell. `--update` writes the
  valgrind and C library a count was recorded with under `measured_with` in
  `scripts/perf_budget.json`, since both move a count with the tree unchanged.
- **No run writes bytecode.** A binding shape is the difference of two runs,
  which cancels the interpreter's start-up only while both runs start alike. An
  interpreter compiles a module the first time it imports it and writes the
  bytecode beside the source, so on a fresh interpreter the first run pays a
  compile its partner reads back. `settle_the_heap` imports `ctypes`, nothing
  else in the bench job does, and the job installs its interpreter fresh: the
  first walk the gate measures reads a third dearer than the same binary run
  again, on whichever side comes first. `workload_environment` sets
  `PYTHONDONTWRITEBYTECODE=1`, so both runs compile or both read. Before the
  two, `warm_the_interpreter` runs the shape once uncounted, and that run may
  write: without it the side measured first reads a colder cache than the side
  after it, since building the base runs the interpreter in between, and a
  cold start leaves the loop a different heap.
- **Branch mispredicts are read beside the decision shapes, and not gated.**
  The decision path dispatches on a node's kind through jump tables taken
  several times per goal, so a change can trade instructions for predicted
  branches, which an instruction count reads as a regression. Every decision
  mode runs under `--branch-sim=yes`, which leaves the instruction
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
  runs before the loop -- a change to the *build* path can move the JSON shape
  by nine percent while its loop executes the same instructions.
  `settle_the_heap` in `crates/valgebra-py/src/workload.rs` calls
  `malloc_trim(0)` before every shape's loop: without it the JSON and recursive
  shapes read 8.2% and 7.2% apart between two environments, and with it 0.06%
  and 0.36%. Trimming does not reach what the *binary's* layout decides: one
  commit built in two directories reads the JSON shape 458M and 528M, all of it
  in `malloc_consolidate`, and the
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
worktree with its own target directory -- and holds the difference to the
gate's ceiling, `RELATIVE_TOLERANCE` in `scripts/perf_gate.py` (2%), whose
comment says why that width. The toolchain, the flags, the machine and the
cachegrind version are one and the same across the pair, so what is left is the
change.

The recorded numbers in `scripts/perf_budget.json` are a **record of one
environment**, and no lane holds a change to them. The reason is measured: the
commit that recorded the current core count re-measures 5.35% away from it on
another machine of the same rustc line. An absolute band wide enough not to
flake on that is ±10%, which cannot see a real 5% regression -- so an absolute
check is either noisy or blind, and choosing between those is not a gate. Read
on a schedule under `continue-on-error`, such a band would be a gate that gates
nothing, so no lane reads one. Re-record with `--update` when an intentional
change moves the number, and say what moved. `perf_gate.py` refuses a command
line that names neither `--against` nor `--update`, with exit 2, because there
is nothing to compare with.

One refusal is the relative gate's own: **a workload that changed is not a
comparison.** Two builds of a workload computing the same thing agree on its
checksum exactly, so a disagreement says the workload moved between the two
commits and the counts are of different work. That exits 2 -- neither a pass nor
a regression. The commit that moves a workload re-records its count and checksum
with `--update`, and a shape whose recorded checksum differs from the base's,
or that the base records none for, is a workload replaced on the record
(`rerecorded_at` in `scripts/perf_gate.py`). The base is built with the
change's workload source -- the example, and for a binding shape the
extension's workload module it calls -- so the two counts are of one work and
compared at the gate's ceiling (`judge_replaced`); where the base cannot build
that source, or computes other work with it, the shape is held to its record,
as a new shape is below.

**A step the budget argues is crossed once.** Where a change trades
instructions for time -- reading a list through a snapshot executes more and
spends less -- the commit records the step in `scripts/perf_budget.json`
(`steps`), named by its shape and the base count it steps from, and the
comparison that crosses that commit is excused for it; the next change is held
to the gate's ceiling again (`recorded_step` in `scripts/perf_gate.py`). The
profiled shapes record no steps.

**A shape the base does not carry is a new shape**, not a comparison and not a
pass: it is held to its recorded count in the same run, inside the band the
record carries for a change of machine (`judge_new` in `scripts/perf_gate.py`).
A new shape and a re-recorded workload the base cannot build are where the
record is read, and the band is wide enough for both, since the next change
has this one as its base and is compared with it at the gate's ceiling.
The gate reads this
from the base's *source* before it builds anything -- the example the mode
names, and for a binding shape the arm that names the shape (`absent_at` in
`scripts/perf_gate.py`) -- because a base older than a shape builds the example
and its binary refuses the name, and a refusal read from the binary would look
like a broken build.

Do not copy a recorded count into prose: `scripts/docs_lint.py` fails on it,
because a figure that moves when the count is re-recorded is stale the next
time it moves.

### The competitive ratio

`scripts/compare_gate.py` compares per-call time against pydantic's strict
`TypeAdapter.validate_python`, the entry point a caller holds, across a
shape matrix -- the accept walk, a large array, a wide record, deep nesting, a
JSON document, compilation, and the report a failure builds, the shapes
`ceilings` in `scripts/perf_compare.json` names -- as a **ratio**.
A ratio cancels the runner's absolute speed, which is what lets a wall-clock
measurement gate at all.

Each shape's ceiling is a **claim, not a recorded measurement**: the ratio the
project says it stays under, with headroom. A recorded ratio travels badly,
because the two libraries respond differently to a PGO build and to an
interpreter. The interpreter is the one that moves a shape far: on a single box
a schema nested twenty-five deep reads 0.13 to 0.14 under CPython 3.12 and 3.14
and 0.26 under the free-threaded build, where every read of an element out of a
mutable container takes that container's lock. This gate is the coarse tripwire
for ceding ground, with `perf_gate.py --against` doing the fine-grained work.
Changing a ceiling is an edit with an argument in its commit message.

It is also the one gate that times the wheel the release ships, and
`perf_gate.py --profiled` counts its shapes there. The rest of `perf_gate.py`
builds without a profile, so a change the profile lays out differently is
invisible to it: a list arm one branch longer can read 0.9% in its instruction
count and 64% in this gate's `deep_nesting`, where the training workload holds
no counts for the loop the branch moves out of the walk.

**A ceiling a shape passes by a wide margin stops measuring it**, which is why a
claim is not the whole of the file. The JSON document's ratio moves by a tenth
under its ceiling with no gate red -- the `window` in `scripts/perf_compare.json`
holds the readings -- and nothing reports the move until somebody re-runs this
gate and reads the number. So beside each
ceiling the file has a `recorded` block for the ratio the shape last measured
and the spread it was measured across -- the ratchet the mutation sweep and the
instruction gate already have. Once armed, a shape measuring worse than
`recorded + tolerance` is red while still under its ceiling.

**The ratchet is armed from a window of the lane's readings, because one run
cannot arm it.** The bench lane's runners come in classes of machine -- a
pydantic one-integer check takes about 160, 240 or 310 ns on them -- and one
tree reads differently on each: three runs of one commit read the one-integer
check 0.156, 0.197 and 0.159, and one commit's push read the nested schema 0.165
where its dispatch read 0.124 and 0.126. A recording taken from one run, which
is what `--update` writes, lands on one class and fires on the next run that
lands on another, with nothing changed. So `scripts/compare_gate.py
--record-window` arms it from a window instead: each shape's ratio is the
median of the window's readings, and its tolerance is the larger of the one it
carries and half again the widest reading above that median, rounded up to a
two-hundredth, so `--record-window` never narrows one. The window is kept in
`scripts/perf_compare.json` beside the ratios it gave, and a row holds the two
to each other and holds every reading of the window under its own floor.

Which readings the window holds, the runs they come from and why it starts
where it does are the `recorded` block's own `_how_recorded` note and `window`
list in `scripts/perf_compare.json`, beside the ratios they gave; a copy of
those figures here would be stale at the next recording. The tolerances also
reach half again the widest of the lane's earlier readings above the recorded
median, so a short window does not set a floor the runs it missed would cross.
A change that moves a shape on the lane, or a profile that lays one out
differently, is recorded again from readings taken after it.

The travel problem is answered rather than avoided. The recorded block names the
environment it was taken in -- the interpreter, whether it has a global lock,
and the pydantic-core version it was compared against, since a faster
pydantic-core raises every ratio with nothing here changing -- and the gate
reports rather than judges anywhere else. `--update` re-records, and belongs on
the lane that builds the wheel the same way every time: a recording made
elsewhere would match the lane's fingerprint while carrying another machine's
noise.

A tolerance is the shape's own spread across the lane's runs, measured per
shape rather than assumed, so `--update` leaves it alone -- one run cannot see a
spread, and a recording that narrowed it silently would fire on noise. A shape
whose spread is too wide to ratchet at all carries a written reason instead of
an absent number, under `unratcheted` in `scripts/perf_compare.json` with the
readings it stands on: `error_report` is the one shape timing a path that raises
and formats a Python exception, and its spread belongs to the build as much as
to the shape. A row holds every shape to a tolerance or an argument in both
directions.

**A recording is taken on the lane, and read before anything is committed.**
A ratio belongs to the environment it was taken in (the interpreter's minor
version, whether it has a global lock, and the pydantic-core on the other
side), and `scripts/compare_gate.py` refuses to judge across those rather than
compare a number about somewhere else. Run the CI workflow from the Actions tab
with **`record_compare`** checked, and the `perf-compare-recorded` artifact
holds that run's ratios. One run is one class of machine, so the artifact is a
reading to set beside the lane's others, not a block to commit as it stands:
the lane cannot commit, which is the point -- a floor is a thing somebody
chose, and choosing one wants the lane's spread in hand. A run that reads the
profiled build against the plain one is the run to tick it on: both readings
then come from one run, one image and one toolchain, though from two runners
that may be of different classes.

**What a profile buys is read from the same lane, and it is per shape.**
Profile-guided optimization arranges what fat LTO left to arrange, so a shape
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

**The lane names the interpreter these are read on**, which is CPython 3.15,
and it is written in `ci.yml` rather than left to the runner image: a ratio
belongs to the pair of libraries *and* the interpreter running them, and a lane
that inherits one from an image makes claims nobody chose. The same holds for
the recorded counts' bands, which cover the distance between two releases,
and for both mutation sweeps, where a mutant on a version-gated branch is
killable on the interpreter that takes the branch and unviable on the one that
compiles it out. `tests/test_lane_interpreters.py` holds every lane to naming
one. A name is a release, and the build behind it is uv's: the wrapper action
sets `UV_PYTHON_PREFERENCE=only-managed`, since uv otherwise takes a runner
image's own interpreter where the image has the release, which for a "3.12"
lane is Ubuntu's 3.12.3, patch releases behind the one uv installs. A lane
that builds against its interpreter keys its Rust cache on that choice, as
`bench` argues: `PYO3_PYTHON` is `.venv/bin/python` whichever build stands
behind it. And uv's Windows `python.exe` reserves 2,000,000 bytes for the main
thread's stack where python.org's reserves 3,000,000, which an unoptimized walk
to the depth bound outgrows, so the Windows rows build the dev profile at
`opt-level=1`.

**The bench job's build cache is keyed on that interpreter too**, and a change
that moves the lane moves the key with it. The binding workload links whichever
interpreter `PYO3_PYTHON` names, and pyo3 resolves it again only when that
variable's value changes; the lane's value is `.venv/bin/python` whatever
release is behind it. A cache saved on another release therefore links HEAD's
workload against that release, while the base, built in a fresh target
directory, links the lane's, and every binding shape reads the distance between
two interpreters as the change's own. Every lane that installs an interpreter
keys its cache on that interpreter for the same reason, the `python` matrix
included, and `tests/test_lane_interpreters.py` holds each key to naming every
interpreter its job installs.

**The uv release is named to pyo3 beside the key.** A key names the release a
lane asks for, and the build behind it is the newest patch the pinned uv's own
download list carries, so a uv release can move every lane's interpreter while
every key stands still: a restored `target/` then links a libpython at a path
the runner no longer has, and the workload fails to link. pyo3 also resolves its
interpreter again when `PYO3_ENVIRONMENT_SIGNATURE` changes value, and
`.github/actions/setup-uv` exports it naming the uv release, so a uv release
costs a rebuild of pyo3 and what links it rather than a link against a build
that is gone. `tests/test_lane_interpreters.py` holds the export to the
wrapper's input.

**Every build cache is keyed on the runner image as well.** A lane that names no
interpreter -- `rust test`, `rust coverage` -- builds against the image's own,
and pyo3 asks for it again only when `PYO3_PYTHON` changes. The key
Swatinem/rust-cache computes names the OS family, the architecture, the
toolchain and the variables whose names start with `CARGO`, `CC`, `CFLAGS`,
`CXX`, `CMAKE` or `RUST`, and not the image, so a `target/` saved on Ubuntu
24.04 restores on 26.04 and links libpython3.12, which 26.04 does not carry.
`.github/actions/setup-rust` exports the name every hosted image gives itself in
`ImageOS` as `RUST_CACHE_IMAGE`, and every lane restores its cache after that
action, so an image move costs a cold build rather than a link against an
interpreter the image does not have. `tests/test_lane_interpreters.py` holds
both halves.

The **free-threaded** build is held to its own set, in the same file, for the
shapes where it is a different environment rather than the same one on a slower
clock: reading an element out of a mutable container takes that container's lock
there, so a shape whose cost is per-element pays what no interpreter with a
global lock pays. A schema nested twenty-five deep is twenty-five
single-element lists, and it reads about twice what the builds with a lock
read; `_free_threaded_comment` in `scripts/perf_compare.json` holds the lane's
readings and the argument for each ceiling. A shape absent from that set is held to the shared ceiling,
and the gate selects between them by asking the interpreter whether its global
lock is enabled.

It asserts each payload is **accepted** before timing it. A correctness
regression that made valgebra reject the data would take the fast reject path and
read as a speed-up; that check is the difference between measuring the accept
path and measuring nothing.

### The mutation ratchets

Three sweeps, each with its own committed baseline: the core crate, the
binding's soundness surfaces — the files `BINDING_SWEPT` in `ci.yml` lists,
which both binding sweeps read: the membership walk with the context it
carries and the precompute it reads, the `Value` both input paths run over, the
frontend and the four surfaces beside it, equality, the oracle and the failure
codes — and the files the shipped extension is the only caller of.
`scripts/mutation_gate.py` fails in **five** directions — a survivor the
baseline does not accept, an accepted entry that no mutant answers to, an
accepted entry naming a file the tree does not track, an accepted note naming no
survivor in the baseline, and an accepted survivor carrying no note. The second keeps the accepted set honest: an
accepted hole the tree does not have silently re-accepts a future survivor with
the same identity. The third is about the key rather than the entry: a baseline
keyed by path goes stale when the path moves, and splitting one module into
several moves every entry that file holds at once. The sweep reads each of its
survivors as new, minutes into a shard, with the mutants listed and no hint that
what changed was the path. Checking the path costs nothing and runs before the
sweep. The fourth holds the arguments beside the set: a note whose mutant a
test now kills, or one spelled as no survivor line spells it, is an excuse that
outlived what it excused.

An entry is the survivor's file and description without its position, so it
outlives code moving around it, and where one function holds several mutants
of one description -- the two `||` of one condition -- its place among them
too, `#2` for the second in the source. Without the place, one entry would
accept every sibling of its description, including one its own note says a test
kills. The places are read from
every mutant the sweep generated, which its four outcome files list between
them; a shard holds part of each function's mutants, so the push sweeps pass
the gate the whole listing, `cargo mutants --list` over the same files, with
`--all`.

**Run it on the interpreter the lane names, with what it installs.** The
verdict is the embedded interpreter's: a mutant this box's 3.14 reports as a
survivor is one CPython 3.12 kills, so a disagreement with the lane is the
interpreter before it is the tree. The lane pins 3.12, and its embedded
interpreter imports `typing_extensions`. That interpreter takes its prefix
from the first `python3` on `PATH`, not from `PYO3_PYTHON`, which only
configures the build: with a bare 3.12, or a venv named by `PYO3_PYTHON` alone,
it starts on the base prefix, where `typing_extensions` is not installed, so
the corpus installs its stand-in and kills the mutants of the arms the lane
cannot reach -- a sweep that passes here and a ratchet that fails there. Both
walk lanes put their venv's `bin` first on `PATH` for that reason, since uv's
build of 3.12 has no `typing_extensions` of its own, and so does `binding
coverage`, whose figure would otherwise count the stand-in's arms.
`tests/test_baseline_interpreters.py` holds every sweep that embeds an
interpreter to doing so, and every other step that embeds one to naming its
prefix: the venv first, or the base installation as `PYTHONHOME`, which is how
the `python` job's corpora read each release. A local sweep runs in a venv the
lock fills, its `bin` first on `PATH`:

```bash
export VALGEBRA_WALK_VENV="$PWD/../walk-venv"
UV_PROJECT_ENVIRONMENT="$VALGEBRA_WALK_VENV" uv sync --locked --no-install-project --python 3.12
export PATH="$VALGEBRA_WALK_VENV/bin:$PATH"
export PYO3_PYTHON="$VALGEBRA_WALK_VENV/bin/python"
export LD_LIBRARY_PATH="$("$PYO3_PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))'):$LD_LIBRARY_PATH"
# Bound what a failing test shrinks, and draw the push lanes' seed. Without the
# first, a mutant the tests caught spends the whole budget shrinking a
# counterexample nobody reads and returns a timeout instead of a verdict;
# without the second, the sweep draws what the lane does not.
export PROPTEST_MAX_SHRINK_TIME=1000
export PROPTEST_RNG_SEED=0
cargo mutants --package valgebra-py --file <each file of BINDING_SWEPT the change touches> \
  --features interpreter-tests -j 2 --timeout-multiplier 20 \
  --output sweep -- -- --skip recursion_deeper_than_the_bound_is_refused \
  --skip the_two_readings_agree_at_the_walks_depth_bound
python scripts/mutation_gate.py --baseline walk --new-only --out sweep/mutants.out
```

The core sweep is the same shape without an interpreter to point at, and its
skips are the two termination proofs and the long chain's minimization:

```bash
export PROPTEST_MAX_SHRINK_TIME=1000
export PROPTEST_RNG_SEED=0
cargo mutants --package valgebra-core --file <each core file the change touches> \
  -j 4 --timeout-multiplier 20 --output sweep \
  -- -- --skip deep_subtype_into_bottom_terminates \
  --skip subtyping_terminates_on_a_distributed_tower \
  --skip a_chain_of_a_hundred_thousand_states_minimizes
python scripts/mutation_gate.py --baseline core --new-only --out sweep/mutants.out
```

The target is never zero. Equivalent mutants exist and are undecidable in
general, so an accepted survivor carries the argument for why no test can kill
it, in the baseline beside it. The gate holds both directions: a survivor with
no argument fails it, and so does an argument naming no survivor. `--update`
and `--new-only` refuse together, since a partial sweep's survivors are not the
set.

**A push sweep draws one seed, and a nightly a new one each night.** A
property law kills some mutants on some draws and not on others, so a sweep
that draws afresh judges an untouched file differently from one push to the
next: a mutant caught on one run is a new survivor on the next, and the merge
blocks on a draw. Both push sweeps read one literal, so a mutant its draw does
not kill survives every time, and the remedy is a test that fails on it
whatever the draw, which is the better test anyway. The nightly sweeps draw
from their run id, which is how such a mutant is found, as a report rather
than a block. The literal is arbitrary, and moving it re-draws every push
verdict, so a new one lands with a sweep under it of the mutants only a draw
kills. A nightly's logs name those: a caught mutant whose log shows nothing
failing but a property is one. `tests/test_required_jobs.py` holds both push
sweeps to one literal, both nightlies to the run id, and the recipes above to
the lanes' literal.

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
together would let `--update` write an overloaded runner into a baseline as a
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
(`nightly-mutants-pytest` in `ci.yml`) and is scheduled only.

**The walk sweep links one interpreter.** `cargo test` for the binding embeds
the interpreter `ci.yml` names for that lane (CPython 3.12), and a survivor's
note in `scripts/mutation_baseline_walk.json` is an argument about *that*
interpreter: a mutant two spellings of `typing.Union` cannot tell apart on 3.14
is caught where they are two objects, and the ratchet reads it as caught. The
same holds for what the lane's environment installs. The embedded interpreter
imports from the lane's venv, where the lock puts `typing_extensions`, and on
3.12 that module's `Required`, `NotRequired`, `Unpack`, `Any` and `Never` are
`typing`'s own objects. So the frontend's arms for the module's own objects are
killable only where it defines them -- 3.10, or the corpus row's stand-in on an
interpreter that cannot import it -- and the notes in
`scripts/mutation_baseline_walk.json` that cite the module say so. A reading only
PyPy takes is compiled for PyPy alone, `#[cfg(PyPy)]`, so the sweep's build does
not compile it: every mutant of it builds and passes and says nothing, and
`exclude_re` names them in `.cargo/mutants.toml` beside the PyPy lane's rows
that hold them. Compiled everywhere, the same code would be regions the
binding's coverage floor counts and no CPython test can reach.

**A reading compiled per interpreter names what it answers, and is measured on
one side.** `crates/valgebra-py/build.rs` re-emits the interpreter's `Py_3_x`
and `Py_GIL_DISABLED` flags, so a `cfg!` on one compiles a different reading per
interpreter. Such a site says in its doc comment which behavior of which
release it answers, with the measurement for each reading; `snapshot_pays` in
the sequence walk is the one site. Each lane that measures reads one side: the
binding sweep and the coverage lane pin 3.13, where the snapshot reading
compiles in, and the bench lane pins 3.15, where the reading in place does, so
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

`.github/workflows/ci.yml` runs them; read the job set there. The properties of
the arrangement stated here are the ones that are decisions rather than
mechanism:

**`main` receives a commit the push lane passed.** A push to `github_ci` runs
the workflow, and `main` moves to a commit only once its `ci` check is green
there, by a fast-forward to that same commit, so the hash the check was read on
is the one `main` carries. A red push lane is repaired on `github_ci`. A push
to `main` runs the workflow too.

**The nightly runs where development is.** A schedule fires on the default
branch alone, and a nightly that read `main` alone would read a release
window's commits after they shipped. So the scheduled run's first job,
`nightly-dispatch`, dispatches the workflow on `github_ci` with its `nightly`
input set, and a dispatch runs that branch's own copy of `ci.yml`: a scheduled
job checking `github_ci` out would run `main`'s copy against the other tree,
and a change to the workflow on `github_ci` would read as a false red or a
skip. The nightly lanes run on
`main` only where nothing was dispatched, a repository without the branch: two
full nightlies side by side compete for the runners and double the night, and
`main`'s tip is a tree the next fast-forward replaces with the one the
dispatched run read. Each lane reads the dispatcher's `dispatched` output, or
the result of the sweep it ratchets. A dispatch changes nothing, so its push
sweeps list no file: measured from `main`'s tip they would sweep the whole
release window at the push seed, which the push that moves `main` sweeps once
and the night's full sweeps read whole.
`tests/test_required_jobs.py` holds the dispatch to the push trigger's other
branch and every scheduled lane to that reading.

**A dispatch for a reading runs no night.** The nightly lanes read the
`nightly` input. A dispatch started by hand for a reading -- `record_compare`,
`pgo_compare` -- clears it and runs the push jobs and the job it asked for, not
tens of runner-hours of sweeps, fuzzers and deep laws. The input is on unless
cleared, because the schedule runs `main`'s copy of `ci.yml`, which may predate
the input on the branch it dispatches, and a dispatch that sets no input is the
night. The night is a concurrency group of its own, so a dispatch for a reading
on the same branch neither cancels it nor is canceled by it.
`tests/test_required_jobs.py` runs both dispatches.

**The push jobs run on `main`'s night only where nothing was dispatched.** The
dispatched run reads every push job on the development tree, and on `main`'s
tip they read a tree the next fast-forward replaces, and redden on what the
branch has already repaired. So each push job reads the dispatcher's `dispatched` output
and runs on a schedule only where it is `false`, a repository without the
branch, where the scheduled run is the only one.

**A skipped job cannot pass.** The `ci` aggregator lists every required job in
`needs:` *and* fails unless each result is `success` rather than merely
not-failure. The duplicated list is a deliberate second copy.

**A canceled job is red.** A job that reaches its timeout is reported
`cancelled` rather than `failure`, and a job nothing waits on is canceled in
silence -- so a ratchet behind a timeout runs nowhere while every lane reads
green. The aggregate therefore waits on the scheduled
jobs too, allowing one answer more from them than from the others — `skipped`,
which is what a push gives a job it does not run — and refusing everything
else. On a night handed to the development branch the push jobs answer
`skipped` and nothing else, since one that ran read `main`'s tip.
`tests/test_required_jobs.py` evaluates the gate's condition on every run its
`RUNS` table names -- a push, a dispatch for a reading, a nightly dispatch, a
night handed on with `main` behind the branch and with `main` on it, and a night
with no branch -- each job's result planted in turn,
and runs the dispatcher's step with `git` and `gh` stood in.

**Every supported interpreter runs on every event.** The release ships a wheel
built per version against a version-specific ABI, so each interpreter is a
separate artifact a caller installs, and a leg that runs only at night is a
wheel nothing exercised until somebody reported it. `ci.yml` owns the matrix.
The release in development is a leg beside them, forgiven by name from its
first alpha to its first candidate, as
[09-releasing.md](09-releasing.md#which-interpreters-a-release-supports) states.
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

**And that leg alone reads the repository audit.** The audit reads the tree,
the workflow and the scripts, which answer the same on every leg, and it is
most of what the suite costs: on one interpreter, 313 of 385 seconds. So the
floor leg reads it in a step of its own, and every leg reads the product suite
and the checks marked `interpreter`, the few whose answer is the running
interpreter's (`docs/dev/08-testing.md` names them). `pypy` reads what a leg
reads. `binding coverage` reads the product suite alone: the binding's
unreached regions are the same with the audit and without it.
`test_the_floor_leg_reads_the_audit_and_every_leg_the_rest` in
`tests/test_required_jobs.py` evaluates each step's selection and condition on
each leg, and holds the audit to the floor leg and every other kind to once a
leg. Each pytest step lists its thirty slowest tests, so what the suite costs
is read off a run's log.

**The floor is one variable.** `ci.yml`'s `FLOOR` names it, and the steps that
answer once for every release -- ruff, both `ty` runs, stubtest, a caller's
strict types, the audit, the checkout of the whole history -- run on the leg
whose interpreter equals it. The job's name cannot read the variable, so it
spells the release, and `tests/test_changelog_ledger.py` holds it to the
checkout's condition. `tests/test_version_gates.py` holds every release a step
compares a leg with to one the job's matrix runs: a floor that moved past the
release one step names would leave that step skipped on every leg, and a
skipped step is a green one.

**The manylinux wheel is installed on every push.** The `wheel (linux)` job
builds the manylinux wheels from the image a release builds them in, then
installs each CPython wheel on the interpreter its ABI tag names and imports it
with every warning an error; the PyPy wheel the image also builds is the
`pypy 3.11` lane's, which builds and runs one of its own. The lanes test the extension `maturin develop` builds, so a
wheel that builds and does not load would otherwise wait for the release smoke.

**PyPy builds, links, and runs the suite.** The release matrix publishes PyPy
3.11 wheels for each ABI on manylinux and for the one the musllinux image
carries (`release.yml` owns the rows), and a push that does not link against
PyPy cannot see what
breaks there: `cpyext` carries the limited API and not every static type object
CPython exports, so an extension naming one links on CPython and fails at
`import` on PyPy. The release smoke runs the product suite on every wheel set
it ships, the manylinux PyPy ones included and the musllinux sets in Alpine
containers, except the musllinux PyPy and free-threaded wheels, which no Alpine
image can load ([09-releasing.md](09-releasing.md)) -- but a release is where a
break is dearest to find, so the push runs one first. The `pypy 3.11`
job builds a release wheel against PyPy and runs `scripts/pypy_import_check.py`
first, which imports it and builds the annotation forms whose compilation
reaches a type object: that is the *link*, and it fails with one line naming the
form rather than in a stack of test output.

Then it runs the suite, with the packages it reads installed from the `test`
dependency group, which the dev group includes beside the checkers and the
build tools, at the versions `uv.lock` holds (`uv export --locked`). A list
written into the step would resolve against the index on every run, and a
release of any package on it would redden a push with nothing in the tree
changed. `tests/test_suite_installs.py` holds this install and the release
smokes' to the export, and the dev group to the test group and the tools it
excuses by name: a package the suite reads, left out of the group, would not
redden a lane, because a row reading an optional implementation skips where it
is absent.

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
visible the night after the push that carried it. Each full sweep runs sharded
-- the core, the walk and the pytest sweep alike, as the diff sweeps do (the
shard counts are `ci.yml`'s) -- each shard reporting its own slice; a job
after them merges the slices and ratchets once, since a survivor is a survivor
of the *sweep* and an entry that survives nothing is known to only when every
shard has reported. A shard's ceiling (`timeout-minutes` in `ci.yml`) is twice
the slowest shard's reading, because a job that reaches its ceiling is
canceled and a canceled job is red. The slowest reading is taken over every
event that runs the sweep, and the events differ: a push to `main` sweeps every
file a release window touched where a push to `github_ci` sweeps one change's,
so a push sweep's shard reads slowest on the push that moves `main`. And the runners differ: one class builds and tests a mutant at half
the speed of another, so a shard read on the faster class costs twice that on
the slower, and the factor of two is that margin. Every sweep shards
round-robin: cut in consecutive slices, a file's mutants fall to one shard,
and one whose mutants build and run costs several times one whose mutants
mostly do not compile -- the binding's push sweep read 15, 19 and past 30
minutes that way. Every push runs the core and walk sweeps
restricted to the **whole files the change touches** of the ones each sweep
examines, the core crate's and the binding files `BINDING_SWEPT` lists —
bounded by the change rather than by the tree — and blocks the merge; the
pytest sweep costs a rebuild and a suite run per mutant and stays on the
schedule. A push sweep
checks the new-survivor direction alone, because a partial sweep never
generates most of the baseline.

Whole files, not the diff's lines: the miss that matters is an edit that stops an
**existing** test from killing a mutant elsewhere in the same file, and that
mutant is not in the diff. **A file whose diff is comments and blank lines
alone is not swept:** it compiles to what it compiled to before, so each of its
mutants answers as it did, and a doc edit to a large file would otherwise put
the whole file's mutants on the push. `scripts/code_changes.py` names the files
whose code changed, for both sweeps. **A test module gives way to the code it
holds:** cargo-mutants mutates no `#[cfg(test)]` file, so a change to a long
`tests.rs` or an `interpreter.rs` corpus -- one that stops it killing a mutant
-- sweeps the file declaring it and the files beneath that module instead. The
crate-wide suites, `laws.rs`, `index_laws.rs` and `schema_tests.rs`, hold the
whole crate, which no push sweeps: the nightly sweep alone judges a change to
one of them. The diff is read in one shape whatever git is configured to print,
a rename as an added file among them.

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
fails. The release workflow's three smoke jobs stay on the upstream action:
each checks the repository out into `tree/`, for `tests/` alone, so the local
action's files are not where it looks for them, and two of them install uv
before that checkout.

**A toolchain comes from rustup, not from an action.** A third-party action is
pinned by hash with a comment naming its tag, and `zizmor` checks that the tag
still names the hash. A toolchain action that publishes one moving tag moves it
with every upstream change, so the pin drifts from its comment and the audit
fails every lane behind it until somebody re-pins. Every lane installs through
`.github/actions/setup-rust` instead, which runs `rustup toolchain install` and
`rustup default` itself -- the runner carries rustup -- retrying, a few times
and on the release server's checksum failure alone, while that server
publishes, and leaves no tag to drift. The
release workflow's sdist smoke job runs the same two commands as a plain step,
because nothing is checked out when it installs. `tests/test_workflow_actions.py`
fails on a toolchain installed by any third-party action.

**A lane syncs once and runs without re-resolving.** A bare `uv run` resolves
the environment again first, which uninstalls the editable build `maturin
develop` put there and installs the wheel from the lock, so every command after
it tests a module the lane did not build. Every lane installs what it needs with
one `uv sync --locked`, the dependency groups it names included, and runs every
later command under `uv run --no-sync`; `tests/test_workflow_syncs.py` holds both
halves over every workflow, the Pages deploy included. A build of the extension
that does not run through the lock -- the build action, which resolves the
build requirement against a manifest it fetches on the day, and the PyPy lane's
own venv -- names the maturin the lock resolves, and the same file holds each to
it. Every cargo build a lane runs passes `--locked`, so the lane, and `cargo
deny` with it, reads the `Cargo.lock` that was committed rather than one a
stale manifest rewrote; `cargo fuzz` takes no `--locked`, so the fuzz lane runs
its locked harness test before it builds the targets. Dependabot proposes a release seven days
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
  result reads as unexecuted code and is not: a shared directory can report
  `decision.rs` half covered, with hundreds of its regions landing on doc
  comments, blank lines and `use` items, where a fresh `CARGO_TARGET_DIR` on the
  same toolchain reads the file nearly whole. The tell is the report's function
  list, which names more copies of the crate in a shared run than in a clean
  one.

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
    claim about that input: an artifact that replays in milliseconds under the
    allocation bound is kept in `fuzz/seeds/decision/` as an input rather than in
    `fuzz/artifacts/` as a finding. `-malloc_limit_mb` is the flag
    that names a defect, because it fires on an allocation.

    And a ceiling fails the lane only if the soak is let stop on it. Under
    `-fork=1` libFuzzer counts a child's out-of-memory or timeout and goes on
    (`-ignore_ooms` and `-ignore_timeouts` default to 1 in fork mode), and a
    soak piped through `tee` ends with `tee`'s status: with one 80 MB
    allocation planted in a copy of the target the step exited 0. The step
    passes both flags as 0, sets `pipefail`, and names the process ceiling
    that then decides too, at about three times the largest child the soak
    has read. `tests/test_fuzz_lane.py` runs the step's script against a
    stand-in soak to hold the last of these.
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
