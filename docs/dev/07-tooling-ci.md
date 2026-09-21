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

## The local gate, and the contract inventory

**`scripts/gate.py` runs the lane's steps, not a list that resembles them.** It
reads every `run:` step of every job the `ci` aggregator waits for out of the
workflow, and runs them in a fresh **shallow clone of `HEAD` with no tags** --
which is what `actions/checkout` produces and what a developer's clone is not.
A step that needs a runner (a PGO wheel, valgrind, a mutation sweep, a second
operating system) is named with the reason instead, and `tests/test_local_gate.py`
holds that list to the workflow in both directions.

**The instruction gate runs here, although the lane that owns it cannot.**
The `bench` lane wants cachegrind, a profiled wheel, a second interpreter and
a system package installed with `sudo`, so it is excused whole -- and the
*comparison* it carries wants none of those. Two builds of one workload,
measured against the base by the lane's own rule (the remote-tracking branch,
or its parent where that resolves to `HEAD`), is three minutes in the caller's
tree, and the tree is where it has to run: the shallow clone holds one commit
and a comparison needs the other. Excusing it with the rest of the lane is how
a change that read sound and cost **seventy-one times** the instructions --
6.45 billion against 89.8 million on the relation matrix -- passed this script
with forty-five steps green. One mode runs, `--decision-matrix`, because it is
the workload whose shapes reach the set representation and a union or
complement change reads 0.00% on the others; the `--binding-*` modes want the
extension built into an interpreter and stay with the lane. Missing valgrind
is named as the reason rather than passed over.

**The floor interpreter is built beside the caller's, and the suite runs on
it.** The matrix runs seven releases and a developer runs one, so every
difference between two of them is a difference the gate could not see -- and
they are not rare: three typing members the floor does not carry passed a green
local run and reddened nine jobs. The floor is the end of the range where that
lands, because the suite is written on the newest release the tree supports and
read on the oldest. So the gate makes a second environment on the floor
`ci.yml` names, builds the extension into it, and runs the **product** suite
there; the repository checks read the tree and answer the same on any release,
so they run once. Which release the floor is stays in `ci.yml`: the gate reads
it, and `tests/test_local_gate.py` refuses a second copy of the number here,
because a stale floor is the one kind of stale that tests less than it claims
while staying green.

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
working directory, as a runner's step does.

**The whole list, since two of these cost a day each.** The table is what a
lane differs from a local run in, and the `modelled` column is what the gate
reproduces; the count is the table's rather than this sentence's:

| difference | modelled | what it cost when it was not |
| --- | --- | --- |
| the clone: one commit, no tags | yes, `shallow_clone` | a ledger read `git describe` and reddened eight jobs at once |
| the step's declared `env:` | yes, since the plan carries it | `RUSTDOCFLAGS=-D warnings` was dropped, so `cargo doc` could not fail locally |
| the terminal's own variables | yes, `runner_environment` | `FORCE_COLOR` turned `pip-audit` red against a clean dependency tree |
| the machine's git identity | yes, `deep_clone` and `NO_IDENTITY` | a checkout configures no `user.name`, this repository has one in its own `.git/config`, and a ledger that plants a commit with `git commit-tree` passed here and failed there |
| the interpreter the lane names | partly: `PYO3_PYTHON` follows the caller's, and the gate's closing line says so | the mutation and cachegrind lanes name CPython 3.12 in `ci.yml`; a local sweep on 3.14 read two mutants as survivors that the lane kills, which is half an hour spent on a difference that was the interpreter |
| the *release* of the interpreter, not only the caller's | yes, at the floor: the gate builds it beside the caller's and runs the product suite on it | `typing.Self`, `LiteralString` and `Unpack` are 3.11 members; a test module naming them collected here on 3.14, failed to collect on the floor, and took nine jobs red with it |
| the operating system and architecture | no | a macOS or Windows leg fails where Linux does not, and nothing local sees it |
| the pinned tool versions | no | `uvx pip-audit==2.10.1` and `uvx zizmor==1.30.1` are the lane's; a local `uvx` takes the latest |
| the build of the interpreter, not only its version | no: a rule answers it instead | `sys.stdlib_module_names` is the build's, not the release's -- this box's 3.12 lists the Windows-only `_wmi` and a runner's does not, so a table of every name reported a difference between two builds as a moved row. The floor table records the modules this tree imports, which are portable by construction |
| the machine's own speed | not modelled, and not a gate: every merge-blocking number is an instruction count | an absolute count reads 5--8% apart between two machines on one `rustc` line, which is what the recorded budgets carry a band for |
| secrets, tokens and the event payload | no, and the steps that need one are excused by name | the merge base comes from the event, so `--against` runs only in the lane |
| what a gate **counts**, against what it claims to | no: a scope is a claim in prose, and no check reads it | the binding coverage floor counted the instruction gate's own workloads, which no suite runs, and the lane went red at 94.50% over a change that added none of its own uncovered lines; the build shape counted the harness formatting fifty names, and three quarters of what it reported was that |

The first four are closed. Named rather than numbered from here, since the
table grows: *the interpreter the lane names* is a row rather than a fix
because the gate runs the caller's toolchain by design, and the gate's own
closing line now names it; *the release of the interpreter* is closed at the
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

**The fourth row is the newest and it is the one to read twice.** It is not a
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

Three of the excused steps need only what a developer's machine already has --
the dependency sync, the extension build, and the stub check that needs both --
and it is tempting to lend the caller's virtual environment to the clone so they
can run. **Do not**: `uv run` inside the clone *writes* the environment it is
pointed at, which uninstalls the built extension and the bench group from the
tree being worked in. A gate that damages the environment it checks is worse
than one that names three steps, so each carries
that cost as its reason rather than "the caller has already run it".

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
the packaging config rather than typed — so the check reads the whole row.

One line in it is easy to miss and is the reason a lane went red: **the fuzz
crate is a detached workspace**. libFuzzer needs a nightly toolchain, and making
it a workspace member would put nightly on every stable gate's path, so
`cargo check --workspace` does not reach it. A change to the core's public types
compiles cleanly without `cargo check --manifest-path fuzz/Cargo.toml` and turns
the fuzz lane red. `tests/test_build_surfaces.py` holds every manifest in the
tree to being a workspace member or a detached surface named with the command
that builds it.

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
  `--binding-record`, `--binding-keys`, `--binding-open`, `--binding-subclass`,
  `--binding-json`, `--binding-pattern`, `--binding-build`,
  `--binding-annotated`, `--binding-object`, `--binding-explain`,
  `--binding-explain-accept`) —
  membership over a live Python value, the call boundary
  alone, a wide record closed, the same record walked over a value whose keys
  are interned, the record open the way a `TypedDict` is, walking a
  `NamedTuple`, parsing and walking a JSON document, matching a string against
  a compiled pattern, building a validator
  from its Python spelling, compiling one written as a `TypedDict` of refined
  integers, compiling a fifty-field dataclass, and explaining a failure in a
  record or accepting one in the same mode. The walk is the shipped
  hot path neither pure-Rust workload reaches; schema construction grew twelve
  percent over a release cycle while only the walk was counted, and an open
  record was read a third dearer than a closed one while only the closed one
  was. Each shape builds what it reads outside its loop: the build shape once
  formatted fifty names and filled a dict per iteration, and three quarters of
  its count was that.

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
  under every green lane. The repair's own mutant is answer-equivalent -- the
  copy holds the same elements, so it decides the same things -- so no test can
  hold it and this count is what does: reverting the line reads +171.65%.

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
export PYO3_PYTHON="$(uv python find 3.12)"          # the interpreter the lanes pin
cargo build --profile profiling -p valgebra-py --features pyo3/extension-module
cp target/profiling/lib_valgebra.so \
  "$VIRTUAL_ENV/lib/python3.12/site-packages/valgebra/_valgebra.cpython-312-x86_64-linux-gnu.so"
PYTHONHASHSEED=0 valgrind --tool=callgrind --callgrind-out-file=out.callgrind python probe.py
callgrind_annotate --inclusive=yes out.callgrind
```

Two things make this work and are easy to get wrong. The interpreter must be
3.12 or valgrind aborts on an instruction it does not model in 3.14. And
`pyproject.toml` must not set `strip`. It did, overriding both profiles, so
`maturin build --profile profiling` produced a binary with no symbols and a
profile that could attribute nothing — two audits worked around it by
profiling the workload binary instead, without noticing why they had to. Stripping is the profile's
decision, and `[profile.release]` still makes it — the shipped wheel is
stripped, and a wheel built from `profiling` is about twelve times its size
because it is not.

Three refusals, because a measurement that did not happen must not read as a
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
a schema nested twenty-five deep reads 0.14 to 0.16 under CPython 3.12 and 3.14
and 0.33 under the free-threaded build, where every read of an element out of a
mutable container takes that container's lock. This gate is the coarse tripwire
for ceding ground, with `perf_gate.py --against` doing the fine-grained work at
2%. Changing a ceiling is an edit with an argument in its commit message.

**A ceiling a shape passes by a wide margin stops measuring it**, which is why a
claim is not the whole of the file. The JSON document sat at 0.87 under a
ceiling of 1.00 while a commit message published 0.78 for it, and no gate was
red for as long as it took somebody to re-run this one for an unrelated reason.
So beside each ceiling the file carries the ratio the shape last measured and
the spread it was measured across -- the ratchet the mutation sweep and the
instruction gate already have, in the one place that had only a claim. A shape
measuring worse than `recorded + tolerance` is red while still under its
ceiling.

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

**The lane names the interpreter these are read on**, which is CPython 3.12,
and it is written in `ci.yml` rather than left to the runner image: a ratio
belongs to the pair of libraries *and* the interpreter running them, and a lane
that inherits one from an image makes claims nobody chose. The same holds for
the instruction budgets' bands, which cover the distance between two releases,
and for both mutation sweeps, where a mutant on a version-gated branch is
killable on the interpreter that takes the branch and unviable on the one that
compiles it out. `tests/test_lane_interpreters.py` holds every lane to naming
one.

The **free-threaded** build is held to its own set, in the same file, for the
shapes where it is a different environment rather than the same one on a slower
clock: reading an element out of a mutable container takes that container's lock
there, so a shape whose cost is per-element pays what no interpreter with a
global lock pays. A schema nested twenty-five deep is twenty-five
single-element lists, and it reads 0.31 to 0.34 against the 0.14 to 0.16 of the
builds with a lock. A shape absent from that set is held to the shared ceiling,
and the gate selects between them by asking the interpreter whether its global
lock is enabled.

It asserts each payload is **accepted** before timing it. A correctness
regression that made valgebra reject the data would take the fast reject path and
read as a speed-up; that check is the difference between measuring the accept
path and measuring nothing.

### The mutation ratchets

Three sweeps, each with its own committed baseline: the core crate, the
binding's soundness surfaces — the membership walk with the context it carries
and the precompute it reads, the `Value` both input paths run over, the
frontend's four files, equality, and the oracle, and the files the shipped
extension is the only caller of. `scripts/mutation_gate.py` fails in
**three** directions — a survivor the baseline does not accept, an accepted
entry that no mutant answers to, and an accepted entry naming a file the tree
does not track. The second keeps the accepted set honest: an accepted hole the
tree does not have silently re-accepts a future survivor with the same identity.
The third is about the key rather than the entry: a baseline keyed by path goes
stale when the path moves, and splitting one module into several moves every
entry that file holds at once. The sweep reads each of its survivors as new,
nine minutes into a shard, with the mutants listed and no hint that what changed
was the path. Checking the path costs nothing and runs before the sweep.

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
  --output sweep -- -- --skip recursion_deeper_than_the_bound_is_refused
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
bound runs without end under a mutation that removes the bound, so the whole
run returns no verdict. Each such test leaves the *sweep* and stays in the test
lane, marked `SWEEP-SKIP` in its own source with the reason;
`tests/test_sweep_skips.py` owns the list and holds the marks and the
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
is caught where they are two objects, and the ratchet reads it as caught.

## A gate that compared nothing must not pass

"No mismatches" is true of an empty corpus. Every gate here refuses that shape
rather than publishing a comparison it never made: the perf gate on an
unreadable count, the compare gate on a rejected payload, the ratchet on an empty
output directory, the doc lint and every ledger on an empty universe.

**Every numeric gate has been seen to fail**, in a committed test. That is the
standard: a detector that cannot be shown to fail is not evidence.

## The lanes

`.github/workflows/ci.yml` runs them; read the job set there. Four properties of
the arrangement are worth stating because they are decisions rather than
mechanism:

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

**Every supported interpreter runs on every event.** The floor (3.10) through
the prerelease (3.15), the free-threaded build (3.14t) among them, plus one
macOS and one Windows leg. The release ships a wheel built per version against a
version-specific ABI, so each interpreter is a separate artifact a caller
installs, and a leg that runs only at night is a wheel nothing exercised until
somebody reported it. `test_every_supported_interpreter_runs_on_every_event` in
`tests/test_required_jobs.py` holds the list in both directions, and holds its
first entry to the floor `requires-python` claims.

**One leg carries the history, and its name says so.** `actions/checkout`
takes one commit and no tags, and the two checks that read `git log` back to
the last release tag -- the changelog roll, and the reachability of every
commit a page cites -- stand down where they cannot read. Eight of the nine
legs are that clone. The floor leg takes `fetch-depth: 0` so those run
somewhere, and it is *named* for it, because nine results with the same shape
of name is a checks list nobody can read the difference off: the one that
matters is the one that is not skipping. `tests/test_changelog_ledger.py`
holds the depth and the name to the **same** condition, since a name that
advertises a history the leg no longer takes answers the reader's question
wrongly, which is worse than not answering it.

**PyPy builds, links, and runs the suite.** The release matrix publishes four
PyPy 3.11 wheels, and a push that does not link against PyPy cannot see what
breaks there: `cpyext` carries the limited API and not every static type object
CPython exports, so an extension naming one links on CPython and fails at
`import` on PyPy. The smoke jobs run on CPython, so without a PyPy lane the
first run to find it is a user's. The `pypy 3.11`
job builds a release wheel against PyPy and runs `scripts/pypy_import_check.py`
first, which imports it and builds the annotation forms whose compilation
reaches a type object: that is the *link*, and it fails with one line naming the
form rather than in a stack of test output.

Then it runs the suite. The link is not the only property that differs there,
which the lane learned the first time it ran one: `cpyext` implements
`PyTuple_Size` through the object's own `__len__`, so a `tuple` subclass that
overrode it walked past the end of its storage and killed the process — an
answer, not a symbol, and an import cannot see it.
`tests/test_lane_interpreters.py` holds the rule both ways: an implementation
the packaging classifiers state has a lane running the suite, and a lane running
the suite is on an implementation somebody stated. The wheel is a **release**
build, because a debug one carries frames large enough that the deep-nesting
cases overflow PyPy's C stack, which is a property of the profile and not of the
code.

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
visible the night after. The core's full sweep runs sharded, as the diff sweep
does (the shard count is `ci.yml`'s), each shard reporting its own slice; a job
after them merges the slices and ratchets once, since a survivor is a survivor
of the *sweep* and an entry that survives nothing is known to only when every
shard has reported. A shard's ceiling (`timeout-minutes` in `ci.yml`) is twice
the slowest shard's reading, because a job that reaches its ceiling is
cancelled and a cancelled job is red. Every push runs the same sweeps
restricted to the
**whole files the change touches** — bounded by the change rather than by the
tree — and blocks the merge. It checks the new-survivor direction alone, because
a partial sweep never generates most of the baseline.

Whole files, not the diff's lines: the miss that matters is an edit that stops an
**existing** test from killing a mutant elsewhere in the same file, and that
mutant is not in the diff.

**A slow network is not a red lane.** `astral-sh/setup-uv` reads its version
manifest from `raw.githubusercontent.com` under a hard five-second timeout and
never retries, so a slow response there does not make a job slow — it fails it,
and it fails that one job while every other lane in the same run installs uv
fine. Every lane installs through
`.github/actions/setup-uv` instead: it pins the uv release, which is one fewer
thing resolved over the network, and makes a second attempt when the first one
fails. The release workflow's two smoke jobs stay on the upstream action, because
they deliberately never check the repository out and a local action needs its own
files on disk.

**Every gate script runs in a lane.** `tests/test_lane_coverage.py` holds each
executable under `scripts/` to being driven by a workflow, by the packaging
config, or by the suite — or excused by name with a reason, on a list that fails
if the excuse goes stale in either direction. A script in no lane is not a gate.

## What each instrument cannot see

- **A line coverage floor cannot see a wrong answer.** It says a line ran, not
  that anything checked what it did. That is what the mutation sweeps are for.
- **A crate-wide floor cannot see one file.** `--fail-under-lines` and
  `--fail-under-regions` are read against the total, so a file sits under the
  floor for as long as the rest of the crate carries it. A module entered the
  core reading 84.48% of lines and 85.90% of regions -- worst in a crate held
  to 98 and 97 -- and moved the total by a hundredth of a percent, so both
  floors, the mutation ratchet and the whole product suite stayed green on the
  merge path while a scoped sweep of it reported 22 of 22 mutants surviving.

    What did report it is `scripts/branch_coverage.py`, which holds a region
    floor per file and refuses a file the measurement carries and no floor
    does. It runs on the **nightly** lane, on a pinned nightly toolchain, so it
    answered after the change had merged rather than before.

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

- **A mutation score is a statement about one test command.** Neither sweep runs
  pytest, so a survivor in either is a gap in the *Rust-side* corpus.
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
- **A soundness property has nothing to say about a `False`.** A law shaped
  `if a.is_subtype_of(&b) { ..check.. }` never examines the answers that are
  wrong in the conservative direction, and those are the majority of them.
  `tests/test_completeness_probe.py` is the instrument that faces that way, and
  its own blind spot is a gap no value in its universe can expose.
