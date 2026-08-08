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
identical across runs of a build. Two workloads: the pure-Rust core, and the
membership walk over a live Python value, which is the shipped hot path the core
workload does not reach.

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
  result, compared **before** any count. The core workload's is a recorded
  constant; the binding workload's *is* its iteration count by construction, so a
  workload that ignored its argument — reporting a difference near zero — fails
  rather than passing under the ceiling.
- **An unreadable measurement exits 2.**

The budgets live in `scripts/perf_budget.json`. Do not copy one into prose:
`scripts/docs_lint.py` fails on it, because a figure that moves when the budget
is re-recorded is stale the next time it moves.

### The competitive ratio

`scripts/compare_gate.py` compares per-call time against pydantic-core across a
shape matrix, as a **ratio** against a recorded baseline. A ratio cancels the
runner's absolute speed, which is what lets a wall-clock measurement gate at all.

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

`.github/workflows/ci.yml` runs them; read the job set there. Three properties of
the arrangement are worth stating because they are decisions rather than
mechanism:

**A skipped job cannot pass.** The `ci` aggregator lists every required job in
`needs:` *and* fails unless each result is `success` rather than merely
not-failure. The duplicated list is a deliberate second copy.

**The full sweeps are scheduled, and a diff-scoped one is not.** A full sweep is
minutes of rebuilds and does not belong on a push, so a regression it catches is
visible the night after. Every push runs the same sweeps restricted to the
**whole files the change touches** — bounded by the change rather than by the
tree — and blocks the merge. It checks the new-survivor direction alone, because
a partial sweep never generates most of the baseline.

Whole files, not the diff's lines: the miss that matters is an edit that stops an
**existing** test from killing a mutant elsewhere in the same file, and that
mutant is not in the diff.

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
