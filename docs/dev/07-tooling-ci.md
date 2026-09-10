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

Three of the excused steps need only what a developer's machine already has --
the dependency sync, the extension build, and the stub check that needs both --
and lending the caller's virtual environment to the clone was tried so they
could run. It is **reverted and recorded**: `uv run` inside the clone *writes*
the environment it is pointed at, and doing so uninstalled the built extension
and the bench group from the tree being worked in. A gate that damages the
environment it checks is worse than one that names three steps, so each now
carries that cost as its reason rather than "the caller has already run it".

A step is **accounted for** when it is in the plan the gate builds or named in
`NEEDS_A_RUNNER`, and the ledger asks it of the plan. Asking it of "is this name
excused" instead was a contradiction that no workflow could fail, and underneath
it `resolved` searched for `${{ env.X }}` alone -- so a step carrying
`${{ github.sha }}` came back resolved with its braces intact and would have
reached bash that way. Every expression the gate cannot fill in now makes the
step unresolved, which is a skip with a reason rather than a command.

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
  `--binding-record`, `--binding-build`, `--binding-explain`) — membership
  over a live Python value, the call boundary alone, a wide record, building a
  validator, explaining a failure. The walk is the shipped hot path neither
  pure-Rust workload reaches, and schema construction grew twelve percent over
  a release cycle while only the walk was counted.

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
project says it stays under, with headroom. A recorded ratio would be another
number that travels badly, because the two libraries respond differently to a
PGO build and to an interpreter. The interpreter is the one that moves a shape
far: on a single box a schema nested twenty-five deep reads 0.14 to 0.16 under
CPython 3.12 and 3.14 and 0.33 under the free-threaded build, where every read
of an element out of a mutable container takes that container's lock. This gate
is the coarse tripwire for ceding ground, with `perf_gate.py --against` doing
the fine-grained work at 2%. Changing a ceiling is an edit with an argument in
its commit message.

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
**both** directions — a survivor the baseline does not accept, and an entry that
is no longer a survivor. The second keeps the accepted set honest: an accepted
hole the tree does not have silently re-accepts a future survivor with the same
identity.

The target is never zero. Equivalent mutants exist and are undecidable in
general, so an accepted survivor carries the argument for why no test can kill
it, in the baseline beside it.

**Read a mutation score with its skip list.** Three tests exist to prove a bound;
a mutation that removes the bound makes each run without end, so the whole run
returns no verdict. Each leaves the *sweep* and stays in the test lane, marked
`SWEEP-SKIP` in its own source with the reason, and `tests/test_sweep_skips.py`
holds the marks and the workflow's skip list to each other in both directions. A
mutant whose experiment cannot finish is a rig fault, not a detection.

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

**The full sweeps are scheduled, and a diff-scoped one is not.** A full sweep is
minutes of rebuilds and does not belong on a push, so a regression it catches is
visible the night after. The core's full sweep runs sharded, as the diff sweep
does (the shard count is `ci.yml`'s), each shard reporting its own slice; a job
after them merges the slices and ratchets once, since a survivor is a survivor
of the *sweep* and an entry that survives nothing is known to only when every
shard has reported. One unsharded job over the whole core reached its timeout
every night for a week. Every push runs the same sweeps restricted to the
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
