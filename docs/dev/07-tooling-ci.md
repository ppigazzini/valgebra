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
interpreter) is named with the reason instead, and `tests/test_local_gate.py`
holds that list to the workflow in both directions.

**What the clone models, the environment does not.** A step gets the caller's
environment plus the job's and the step's own `env:`, and the runner's own
absences are not modelled by that -- so a variable a developer's terminal sets
reaches a step that a lane runs with it unset. One of them gave a false red:
`FORCE_COLOR` overrides a tool's terminal check, `uv export` wrote escape codes
into the requirements file it generates, and `pip-audit` refused the file
against a dependency tree with no advisory in it. The two variables whose only
purpose is that override are dropped (`runner_environment`).

**The whole list, since two of these cost a day each.** A lane differs from a
local run in ten ways, and the gate models the first four:

| difference | modelled | what it cost when it was not |
| --- | --- | --- |
| the clone: one commit, no tags | yes, `shallow_clone` | a ledger read `git describe` and reddened eight jobs at once |
| the step's declared `env:` | yes, since the plan carries it | `RUSTDOCFLAGS=-D warnings` was dropped, so `cargo doc` could not fail locally |
| the terminal's own variables | yes, `runner_environment` | `FORCE_COLOR` turned `pip-audit` red against a clean dependency tree |
| the machine's git identity | yes, `deep_clone` and `NO_IDENTITY` | a checkout configures no `user.name`, this repository has one in its own `.git/config`, and a ledger that plants a commit with `git commit-tree` passed here and failed there |
| the interpreter the lane names | partly: `PYO3_PYTHON` follows the caller's, and the gate's closing line says so | the mutation and cachegrind lanes name CPython 3.12 in `ci.yml`; a local sweep on 3.14 read two mutants as survivors that the lane kills, which is half an hour spent on a difference that was the interpreter |
| the operating system and architecture | no | a macOS or Windows leg fails where Linux does not, and nothing local sees it |
| the pinned tool versions | no | `uvx pip-audit==2.10.1` and `uvx zizmor==1.30.1` are the lane's; a local `uvx` takes the latest |
| the build of the interpreter, not only its version | no: a rule answers it instead | `sys.stdlib_module_names` is the build's, not the release's -- this box's 3.12 lists the Windows-only `_wmi` and a runner's does not, so a table of every name reported a difference between two builds as a moved row. The floor table records the modules this tree imports, which are portable by construction |
| the machine's own speed | not modelled, and not a gate: every merge-blocking number is an instruction count | an absolute count reads 5--8% apart between two machines on one `rustc` line, which is what the recorded budgets carry a band for |
| secrets, tokens and the event payload | no, and the steps that need one are excused by name | the merge base comes from the event, so `--against` runs only in the lane |
| what a gate **counts**, against what it claims to | no: a scope is a claim in prose, and no check reads it | the binding coverage floor counted the instruction gate's own workloads, which no suite runs, and the lane went red at 94.50% over a change that added none of its own uncovered lines; the build shape counted the harness formatting fifty names, and three quarters of what it reported was that |

The first four are closed. The fifth is a row rather than a fix because the
gate runs the caller's toolchain by design, and the gate's own closing line now
names it; the sixth is answered by a rule rather than by the gate; the rest are
differences a local gate cannot remove, and naming them is what keeps a green
local run from being read as a promise it never made.

**The fourth row is the newest and it is the one to read twice.** It is not a
setting a developer chose: it is an *absence* on the runner that is a presence
here. A check that writes anything -- a commit, a file, a config -- asks the
machine for something the runner does not have, and passes locally for that
reason alone. The gate models it by cloning: a clone inherits no `user.*`, and
pointing git's global and system config at an empty file removes the rest.

The last is a different kind and is the newest: it is not a difference between
a lane and a local run at all, but between what a gate measures and what its
own text says it measures. Both halves of a gate are a claim -- the number and
the scope -- and only the number is held to anything. Three red lanes in one
week were that same finding, so it is a row here rather than a rule nobody
wrote: when a gate moves, read what it now counts against the sentence that
says what it counts.

Three of the excused steps need only what a developer's machine already has --
the dependency sync, the extension build, and the stub check that needs both --
and lending the caller's virtual environment to the clone was tried so they
could run. It is **reverted and recorded**: `uv run` inside the clone *writes*
the environment it is pointed at, and doing so uninstalled the built extension
and the bench group from the tree being worked in. A gate that damages the
environment it checks is worse than one that names three steps, so each carries
that cost as its reason rather than "the caller has already run it".

A step is **accounted for** when it is in the plan the gate builds or named in
`NEEDS_A_RUNNER`, and the ledger asks it of the plan. Asking it of "is this name
excused" instead was a contradiction that no workflow could fail, and underneath
it `resolved` searched for `${{ env.X }}` alone -- so a step carrying
`${{ github.sha }}` came back resolved with its braces intact and would have
reached bash that way. Every expression the gate cannot fill in makes the step
unresolved, which is a skip with a reason rather than a command.

The reason it exists is a measurement rather than a principle: a ledger reading
`git describe` passed locally for a week and turned eight jobs red on one push,
because the clone shape differed. Everything else on this page is about what the
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
  `--decision-repeat`) — subtyping, emptiness, equivalence, in three
  workloads: one whose relations hold, one whose relations are refuted, and one
  whose goals repeat. The core workload never calls a decision, so without
  these the whole decision surface is unmeasured in both directions: neither
  what a new rule costs nor what a cheaper one saves. Three because a proof, a
  refutation and a repeated goal walk different paths, and a workload that
  asks only for proofs holds a refuting rule to nothing;
- the **binding** shapes (`--binding`, `--binding-boundary`,
  `--binding-record`, `--binding-keys`, `--binding-open`, `--binding-build`,
  `--binding-annotated`, `--binding-object`, `--binding-subclass`,
  `--binding-explain`) — membership over a live Python value, the call boundary
  alone, a wide record closed, the same record walked over a value whose keys
  are interned, the record open the way a `TypedDict` is, building a validator
  from its Python spelling, compiling one written as a `TypedDict` of refined
  integers, compiling a fifty-field dataclass, walking a `NamedTuple`,
  explaining a failure. The walk is the shipped
  hot path neither pure-Rust workload reaches; schema construction grew twelve
  percent over a release cycle while only the walk was counted, and an open
  record was read a third dearer than a closed one while only the closed one
  was. Each shape builds what it reads outside its loop: the build shape once
  formatted fifty names and filled a dict per iteration, and three quarters of
  its count was that.

  The last three are there because the first six could not see three repairs
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

The binding workload embeds CPython, whose startup is not a fixed instruction
count, so the gate measures the **difference** between two iteration counts:
startup cancels and the per-iteration walk cost remains.

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
-- a floor is a thing somebody chose.

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

Two sweeps, each with its own committed baseline: the core crate, and the
membership walk with the context it carries. `scripts/mutation_gate.py` fails in
**three** directions — a survivor the baseline does not accept, an accepted
entry that no mutant answers to, and an accepted entry naming a file the tree
does not track. The second keeps the accepted set honest: an accepted hole the
tree does not have silently re-accepts a future survivor with the same identity.
The third is about the key rather than the entry: a baseline keyed by path goes
stale when the path moves, and eight entries did when the frontend's surfaces
became their own modules. The sweep reads each of that file's survivors as new,
nine minutes into a shard, with the mutants listed and no hint that what changed
was the path. Checking the path costs nothing and runs before the sweep.

**Run it on the interpreter the lane names.** The verdict is the embedded
interpreter's: a mutant this box's 3.14 reports as a survivor is one CPython
3.12 kills, and half an hour went into a difference that was the interpreter
rather than the tree. The lane pins 3.12, so a local sweep does too:

```bash
export PYO3_PYTHON="$(uv python find 3.12)"
export LD_LIBRARY_PATH="$("$PYO3_PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))'):$LD_LIBRARY_PATH"
cargo mutants --package valgebra-py --file <the files the change touches> \
  --features interpreter-tests -j 4 --timeout-multiplier 20 \
  --output sweep -- -- --skip recursion_deeper_than_the_bound_is_refused
python scripts/mutation_gate.py --baseline walk --new-only --out sweep/mutants.out
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
silence: the strict mutation ratchet ran nowhere for a week behind a two-hour
cancellation the aggregate never read. So the aggregate waits on the scheduled
jobs too, allowing one answer more from them than from the others — `skipped`,
which is what a push gives a job it does not run — and refusing everything
else. `tests/test_required_jobs.py` reads which jobs those are from their own
conditions and holds each reading to the kind of job it is.

**A push runs the ends of the interpreter range, not the middle.** The floor
(3.10), the current release (3.14), the free-threaded build (3.14t) and the
prerelease (3.15), plus one macOS and one Windows leg; 3.11, 3.12 and 3.13 run
nightly. The extension is compiled against a version-specific ABI, so what
differs between two adjacent interpreters differs at an end first, and seven
legs on every push bought minutes rather than information.
`tests/test_required_jobs.py` holds the split, because a matrix grows by one
line and nobody re-measures.

**PyPy builds, links, and runs the suite.** The release matrix publishes four
PyPy 3.11 wheels and nothing on a push linked against PyPy until 0.0.10 broke
there: `cpyext` carries the limited API and not every static type object CPython
exports, so an extension naming one links on CPython and fails at `import` on
PyPy — after the release, since the smoke jobs run on CPython. The `pypy 3.11`
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
never retries, so a slow response there does not make a job slow — it fails it.
That took the pip-audit lane down on 2026-08-10 while the twelve other lanes in
the same run installed uv fine. Every lane installs through
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
  never "there is nothing to find". That is why it is not a merge gate.
- **A soundness property has nothing to say about a `False`.** A law shaped
  `if a.is_subtype_of(&b) { ..check.. }` never examines the answers that are
  wrong in the conservative direction, and those are the majority of them.
  `tests/test_completeness_probe.py` is the instrument that faces that way, and
  its own blind spot is a gap no value in its universe can expose.
