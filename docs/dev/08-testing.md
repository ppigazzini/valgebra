# Testing

Correctness here is checked against the **denotation**, not against itself. A
test that asserts the validator agrees with the validator proves nothing about
what a schema means.

## Two suites in one directory

`tests/` holds two kinds of file, and only one of them says anything about the
library. Most exercise valgebra. The rest read *this repository* -- its
configuration, its gate scripts, its documentation, its CI workflows -- and
answer for the project rather than for the product. They carry the `repository`
marker:

```bash
pytest -m "not repository"   # the product suite
pytest -m repository         # the project's own audit
pytest                       # both, which is what CI runs
```

The split is worth drawing for two reasons. A repository check counted as a
product test makes the library look tested by work that never touched it. And a
reader who has the library without the tree -- from a wheel, an sdist, a
distribution package -- cannot run those checks at all, so a suite that mixes
them fails for a reason that is not about valgebra.

The marker is not maintained by hand. A test file that never imports `valgebra`
exercises nothing and must carry it; a marked file that *does* import valgebra
is a product test the marker would silently drop. `test_suite_partition.py`
reads both facts from the syntax and fails either way round, so a new file lands
on one side the day it arrives.

## The layers, and what each can see

| Layer | Judges | Blind to |
|---|---|---|
| Denotation oracle | the walk, against an independent Python predicate over generated values | what the generator does not draw |
| Model oracle (Rust) | the IR, the constructors' normal form and the decision procedure, against a value model | the walk — the core cannot see Python |
| Differential | the walk, against pydantic-core and jsonschema | the fragment where the semantics deliberately differ |
| Boundary | the claims about *why* a check-only tool differs from a parser, as verdicts against pydantic | anything the two libraries answer the same way |
| Metamorphic | the JSON path against the object path; fast mode against explain mode | a defect both sides share |
| Algebra laws | every claimed equivalence, against membership | a law nobody thought to claim |
| Completeness ledger | that the procedure *decides* an enumerated relation | a relation nobody enumerated |
| Completeness probe | that every `False` has a value witnessing it | a gap no value in its universe can expose |
| Snapshots | error messages and `repr` | whether the message is *right* |
| Fuzzing | panic-freedom, idempotence, the order laws | anything outside the budget |

The denotation oracle is the load-bearing one: it shares none of the validator's
frontend, so an agreement is evidence rather than a tautology.

## Enumerating and searching are not the same instrument

Read the "blind to" column of the ledger row again: *a relation nobody
enumerated*. That is not a small gap. Every layer above it shares a version of
it, and they shared it at once:

- the algebra laws assert soundness, which inspects a `True` and has nothing to
  say about a `False`, so a relation wrongly answered `False` is examined by
  nothing;
- the ledger holds what a human wrote down, so a rule nobody thought of has no
  entry to fail;
- a mutation sweep changes code that **exists**. A missing arm is not a mutation
  of anything, so an adequacy figure of any size says nothing about it;
- a fuzz law that hardcodes an atom where a property is meant confirms the rule
  using the rule.

So a missing rule was invisible to all of them together, and stayed invisible for
months while every gate was green. `tests/test_completeness_probe.py` is the
direction none of them face. It takes a wide value universe, and for each pair
the procedure answers `False` it looks for a witness — a value in the subtype and
outside the supertype. **No witness means the relation looks true and the
procedure did not see it.** That is a suspected gap, and suspected is the honest
word: the universe is finite, so the result is a ledger to read rather than a
verdict to trust.

The lesson generalises past this procedure. An enumerated list can only confirm
the rules it was built from. When the failure you fear is *a rule nobody wrote*,
no list will find it — only a search will.

## Two directions, always

Every list in this repository that could rot is held to the tree in **both**
directions, because a hand-written list satisfies the direction it was written
for and misses the other. Fifteen of them:

| Ledger | Holds |
|---|---|
| `tests/test_lane_coverage.py` | every gate script runs in a lane; no excuse is stale |
| `tests/test_mutation_scope.py` | every binding file is swept or excluded by name |
| `tests/test_sweep_skips.py` | every `SWEEP-SKIP` mark is skipped; every skip is marked |
| `tests/test_build_surfaces.py` | every manifest is a workspace member or a named detached surface |
| `tests/test_harness_conditionals.py` | every `cfg(feature = ..)` site is test-only, or named |
| `tests/test_contract_inventory.py` | every gate script has a contract row; every row names a real source |
| `tests/test_completeness_probe.py` | every suspected completeness gap is accepted with a reason |
| `tests/test_completeness_ledger.py` | every relation the procedure must decide is decided; every relation it declines is still declined |
| `tests/test_suite_partition.py` | every test file is a product test or a marked repository check |
| `tests/test_metamorphic_gate.py` | every relation the metamorphic gate holds can be driven to fail |
| `tests/test_required_jobs.py` | every pull-request job is required by the merge gate |
| `tests/test_changelog_ledger.py` | every `feat`/`fix` commit since the last release is on the changelog roll |
| `tests/test_closure_ledger.py` | every schema variant is a generator, a representative, or a marker |
| `tests/test_local_gate.py` | every merge-gate step is planned by the local gate or excused by name |
| `tests/test_commit_messages.py` | no commit message names the internal working area |
| `tests/test_fuzz_lane.py` | the fuzz soak names its allocation ceiling and forks its batches |
| `tests/test_ledger_plants.py` | every ledger fails on the defect it exists to catch |

Each declares itself with a `LEDGER:` marker, and `scripts/docs_lint.py` holds
this table to those markers both ways, so a ledger added without a row fails
rather than passing quietly. The count is spelled here and in the glossary
because a table nothing counts is the one that drifts: there are fifteen.

The last is a ledger over the rest, and it exists because reading a
ledger cannot tell you whether it can fail. `test_local_gate.py` filtered its
steps with `not runnable(name) and name not in NEEDS_A_RUNNER`, which is `X and
not X` -- so its list was empty for every possible workflow and the assertion
passed on a tree that had already broken the claim. `tests/test_ledger_plants.py`
plants, for each ledger, the defect that ledger exists to catch: a schema variant
in no column, a gate script in no lane, a job the merge gate does not require, a
`feat` commit off the roll. Each is planted in a throwaway clone, that ledger is
run there, and it must **fail**. A ledger with no plant fails the list, so the
next one arrives with the evidence that it works.

`tests/test_node_matrix.py` is the same shape one level in: it reads the `Schema`
variants out of the IR and fails when one carries no row, so the universe is
derived from the tree rather than restated.

Each ledger asserts it read something. A check over an empty universe passes
having checked nothing, which is worse than a bare failure.

## Where a test module lives

A test module that is more than a screen long sits in a **sibling file**, not in
the file it tests: `decision.rs` declares `#[cfg(test)] mod tests;` and the body
is `decision/tests.rs`. It is still a child module -- it reaches private items
through `use super::*`, which an integration test in `tests/` cannot -- and it is
still compiled only under `cfg(test)`, so nothing about the shipped build
changes.

What changes is reading. `decision.rs` was 3,570 lines of which 1,500 were
tests, and `lib.rs` was 4,211 of which nearly all were: a reader looking for the
subtype rules scrolled past a thousand lines of assertions to find them, and a
bound had to be told apart from a test fixture by its indentation.

The two ledgers that read the tree know the shape: `tests/test_mutation_scope.py`
does not ask a test module to be swept, and `tests/test_harness_conditionals.py`
does not read a feature gate inside one as reaching production. Both find them
the same way -- by the `#[cfg(test)] mod NAME;` that declares them -- rather than
by a naming convention.

## The binding's own corpora

`cargo test` cannot reach the binding without an interpreter, so the two files
where a mistake changes what a schema means carry their own corpus under the
`interpreter-tests` feature, which links an embedded Python.

The **walk** carries a value corpus in `crates/valgebra-py/src/check/walk.rs`.
Every case runs in **both** the fast and the explaining mode with the two
required to agree, and with the violation count asserted where it distinguishes
the modes. A corpus driven only fast leaves half of every composite unobserved.

The **frontend** carries an annotation corpus in
`crates/valgebra-py/src/build.rs`: a table of annotations as a caller writes
them beside the schema each must build, spelled as that schema's render, plus
the refusals and the message each carries. It exists so the frontend can be
swept: pytest exercises that file thoroughly and a mutation harness cannot
observe pytest, so before the corpus every mutant of it read as a survivor and
the file sat outside the sweep by name. It reads a marker by *attribute* rather
than importing `annotated_types`, because an embedded interpreter starts on the
base prefix and sees no virtual environment -- which would make the corpus depend
on how the harness was launched.

That feature enables an embedded interpreter for the test binary and nothing
else: all its sites are inside test modules and the shipped wheel is built
without it. **An option that exists only because the project has a harness is a
defect** — it makes the tested build differ from the shipped one, and every gate
passes on the difference because the tests run one build and the wheel is
another.

`tests/test_harness_conditionals.py` holds it: a `cfg(feature = ...)` on a
production path fails, and a ledger entry naming a feature the manifest no
longer declares fails. The ledger is empty, which is the claim. An option that
is neither allowed nor banned is a decision nobody has made.

## Property depth

The Rust suites run at the library default locally and deeper on the merge gate;
the Python suites take a per-profile example count. Both knobs are set in the
workflow and `tests/conftest.py` — read them there.

`tests/conftest.py` sets no hypothesis deadline, and says why: the job timeout is
the bound. Every job carries one.

## What is not tested here, deliberately

**Denotation preservation is not re-checked in the fuzz targets.** They assert
procedure-agnostic laws — panic-freedom, reflexivity,
the top and bottom bounds, equivalence as mutual inclusion — over the **full**
IR, including the opaque fragment a value oracle cannot model. Membership
preservation is oracle-tested over the decidable fragment in the core law suite.
Duplicating it in the fuzzer would only cover the sub-fragment its wide generator
is built to exceed.

**Completeness cannot be fuzzed.** Over a finite universe only the sound
direction is assertable, so enumeration is the instrument: the ledger lists
relations true by construction and asserts the procedure decides each.

## The limit

**Adequacy is measured on the Rust side only.** Both mutation sweeps run
`cargo test`; the Python suite never executes under either, so a survivor is a
gap in the Rust corpus and not necessarily in the tests as a whole. That is the
reason a swept binding file carries a corpus of its own, and the reason the
files that carry none are excluded by name rather than swept and baselined: a
sweep whose survivors all say "pytest covers this" measures the harness.

**A coverage floor is read with its scope or it misleads.** The Python package
floor covers the re-export package, which is a hundred-odd lines; the extension
the Python suite exercises is Rust and is measured by the other two lanes.
