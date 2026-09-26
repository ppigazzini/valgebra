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
for and misses the other. Forty-three of them:

| Ledger | Holds |
|---|---|
| `tests/test_lane_coverage.py` | every gate script runs in a lane; no excuse is stale |
| `tests/test_mutation_scope.py` | every binding file is swept or excluded by name |
| `tests/test_sweep_skips.py` | every `SWEEP-SKIP` mark is skipped; every skip is marked |
| `tests/test_build_surfaces.py` | every manifest is a workspace member or a named detached surface |
| `tests/test_crate_attributes.py` | every crate root forbids unsafe code |
| `tests/test_harness_conditionals.py` | every `cfg(feature = ..)` site is test-only, or named |
| `tests/test_contract_inventory.py` | every gate script has a contract row; every row names a real source |
| `tests/test_completeness_probe.py` | every suspected completeness gap is accepted with a reason |
| `tests/test_completeness_ledger.py` | every relation the procedure must decide is decided; every relation it declines is still declined |
| `tests/test_suite_partition.py` | every test file is a product test or a marked repository check |
| `tests/test_metamorphic_gate.py` | every relation the metamorphic gate holds can be driven to fail |
| `tests/test_required_jobs.py` | the merge gate requires every job the workflow defines, and every supported interpreter runs on every event |
| `tests/test_changelog_ledger.py` | every `feat`/`fix` commit since the last release is on the changelog roll |
| `tests/test_closure_ledger.py` | every schema variant is a generator, a representative, or a marker |
| `tests/test_local_gate.py` | every merge-gate step is planned by the local gate or excused by name |
| `tests/test_commit_messages.py` | no commit message names the internal working area |
| `tests/test_clock_ledger.py` | no test measures time except the three that argue for it |
| `tests/test_lane_interpreters.py` | no lane installs an interpreter without naming it |
| `tests/test_feature_lanes.py` | every crate feature runs in a test lane or is excused by name |
| `tests/test_version_gates.py` | every release a version gate names has an enforced lane on each side |
| `tests/test_doc_examples.py` | every python example in a tracked page is run or marked with a reason |
| `tests/test_doc_example_checkers.py` | every checker diagnostic on a published example is expected with a reason |
| `tests/test_code_table.py` | every failure code is a name the table declares, and every name is used |
| `tests/test_cited_commits.py` | every commit a tracked file cites is one a clone can reach |
| `tests/test_fuzz_lane.py` | the fuzz soak names its allocation ceiling and forks its batches |
| `tests/test_floor_names.py` | every typing and enum name read at import time, and every stdlib module imported, exists on the floor |
| `tests/test_module_placement.py` | no inline test module is longer than a screen |
| `tests/test_theory_ledger.py` | every load-bearing theory result names a test, and every name is one; the rows that read the argument itself skip where it is absent, which is every lane |
| `tests/test_use_case_ledger.py` | every public name and every error code is named by the suite, or accepted with a reason |
| `tests/test_bound_ledger.py` | every declared bound is driven by a test, or accepted with a reason |
| `tests/test_coverage_scope.py` | every coverage lane names its scope, and the scope is the tree's |
| `tests/test_typed_consumer.py` | every public name is put through `assert_type` by the typed consumer |
| `tests/test_checker_readings.py` | every reading the checkers do not share is held per checker by a fixture |
| `tests/test_frontend_refusals.py` | every frontend refusal message is matched by a test, or accepted with a reason |
| `tests/test_pytest_sweep_scope.py` | every file the sweep excuses to pytest is examined under pytest |
| `tests/test_relation_ledger.py` | every ordered pair of schema variants is decided or declined with a reason |
| `tests/test_form_ledger.py` | every form the schema-language pages tabulate is driven by a test |
| `tests/test_surface_outcomes.py` | every outcome the binding's docstrings name is asserted by a test |
| `tests/test_citation_ledger.py` | every numbered result the theory page cites names a work on the shelf |
| `tests/test_boundary_ledger.py` | every entry of the published decidability boundary is driven by a test |
| `tests/test_constraint_matrix.py` | every constraint is driven against every kind, narrowing it or refused |
| `tests/test_python_lifecycle.py` | every supported interpreter is one CPython's calendar supports today |
| `tests/test_ledger_plants.py` | every ledger fails on the defect it exists to catch |

Each declares itself with a `LEDGER:` marker, and `scripts/docs_lint.py` holds
this table to those markers both ways, so a ledger added without a row fails
rather than passing quietly. The count is spelled here and in the glossary
because a table nothing counts is the one that drifts: there are forty-three.

**Which interpreter reads them.** A ledger is a repository check: it reads the
tree, the workflow and the scripts, none of which answers differently by
release. So every leg of the python matrix runs the whole list, and the reading
is the same nine times -- with two exceptions that can be read only once.
`test_changelog_ledger.py` and `test_cited_commits.py` measure from the last
release tag, and `actions/checkout` takes one commit and no tags; the **floor**
leg takes the whole history so those run somewhere, and its name says so, since
otherwise the one result that did not skip is indistinguishable from the eight
that did. `scripts/gate.py` builds that same floor beside the caller's
interpreter before a push -- and runs the *product* suite on it, not this list,
for the reason the first sentence gives.

**And two directions no runner can read at all.** The theory ledger holds the
tracked page to the maintainer's working notes, and the citation ledger holds a
numbered result to the paper on the shelf; neither the notes nor the shelf is
in the distribution, so both directions stand down in every clone but the
author's. A check that stands down everywhere it runs is a check nobody runs,
so `scripts/gate.py` runs those two in the working tree before a push, where
what they read is present. What that buys is bounded and worth saying plainly:
the page and the notes are held together on one machine, and a green matrix
says nothing about them.

The last is a ledger over the rest, and it exists because reading a
ledger cannot tell you whether it can fail. `test_local_gate.py` filtered its
steps with `not runnable(name) and name not in NEEDS_A_RUNNER`, which is `X and
not X` -- a list empty for every possible workflow, so the assertion passes on
any tree, including one that breaks the claim, and reads exactly like one that
checked something. `tests/test_ledger_plants.py`
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

What changes is reading. A module holding more assertions than code makes a
reader looking for the subtype rules scroll past them to find the rules, and
makes a bound something to tell apart from a test fixture by its indentation.

**A screen is a hundred lines**, and `tests/test_module_placement.py` holds the
rule to that number. It needs one: a bar stated in prose and held by nothing
drifts a module at a time, and the file that owns the count is the test rather
than this page. The bar is generous, and three modules sit under it and stay where they
are: a module short enough to read past is in nobody's way. What the bar catches
is the drift, one case at a time, until a file is mostly not the thing it is
named for.

The two ledgers that read the tree know the shape: `tests/test_mutation_scope.py`
does not ask a test module to be swept, and `tests/test_harness_conditionals.py`
does not read a feature gate inside one as reaching production. Both find them
the same way -- by the `#[cfg(test)] mod NAME;` that declares them -- rather than
by a naming convention.

## The binding's own corpora

`cargo test` cannot reach the binding without an interpreter, so the files
where a mistake changes what a schema means carry their own corpus under the
`interpreter-tests` feature, which links an embedded Python.

**Where they run.** Every `ubuntu-latest` leg of the `python` matrix runs them,
against the interpreter that leg installs, and `tests/test_feature_lanes.py`
holds the rule that puts them there: a feature a crate declares is passed to a
`cargo test` by some lane that is not itself a measurement, or it is excused
there by name. For one push none was. The only lane naming `interpreter-tests`
was `binding coverage`, so a corpus row made stale by a change to the very check
it describes reddened the lane whose job is to print a percentage -- while
`cargo test --workspace` stayed green on three operating systems and said
nothing.

Running them against **every** supported interpreter rather than one is what a
corpus of live objects is for: a `typing` member arrives in a release, and the
floor is where the tree finds out. Four rows named `Never`, `Required`,
`NotRequired` and the star-unpack inside a subscript, each of which reaches the
language in 3.11, and on 3.10 the interpreter answers `AttributeError` or a
`SyntaxError` the row does not expect. So a row states the release it needs --
`Since(11)`, which is the one spelling the corpora use for one -- and stands
down below it, which is the `skipif` the Python suite writes one layer up.

A release written onto a row is a claim about the lanes, and
`tests/test_version_gates.py` holds it to them: each release a gate names has a
lane below it, where the guard is taken, and an **enforced** lane at or above
it, where the guarded code runs. A gate above every lane is code that runs
nowhere and passes; one below the floor is a guard always taken; and a gate
whose only interpreter above it is a prerelease leg is a gate nothing enforces,
since such a leg runs under `continue-on-error`. The same ledger refuses
a corpus that compares `version_info` for itself, because a release spelled any
other way is one it cannot read -- which would reopen the hole a file down.

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

The **oracle** carries a question corpus in
`crates/valgebra-py/src/oracle/interpreter.rs`: one row per question
`LeafRelations` asks, each handing the oracle two pool slots or a class and a
kind and reading the `Option` it answers. It exists for the reason the
frontend's does, and its number is the sharpest of the three: without a corpus
the only Rust rows reaching that file build the pool and read it back, so a
sweep reports most of `oracle.rs` as surviving. `scripts/mutation_baseline.json`
records what survives with the corpus in place.

What it deliberately does not do is compile a schema and ask `is_subtype_of`.
That is what the decision suite in `tests/` does, and a Rust row shaped the same
way would prove the decision rather than the answer the decision was made from,
leaving the same mutants alive. The distinction matters most for the third
answer: `None` is "this oracle cannot read that", which the core folds
conservatively, and `false` is a refutation it may act on — so every row that
expects a decline says so rather than reading it as a negative.

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

**Each profile names its own randomness.** On a runner Hypothesis makes its own
`ci` settings -- derandomised, no database -- the default a registered profile
inherits, so a deep profile that names neither replays the same examples every
night. `ci` is derandomised on purpose, so a red merge gate is the same red on a
re-run; `nightly` draws at random and keeps what fails in `.hypothesis/examples`,
which the nightly lane uploads on failure. `tests/test_property_profiles.py`
reads each profile in a child interpreter with `CI` set, where the inheritance
happens.

## What a use case is, and how many there are

"Every use case is covered" is a claim only once a use case is a thing something
can count, and neither of the obvious candidates is one. A line count says a line
ran, not that anything checked what it did. A list written beside the code grows
a row when a reader remembers to.

So the universe is **derived from the tree**, in products, and each product has a
ledger that holds it in both directions:

| Product | Derived from | Covered means | Held by |
|---|---|---|---|
| every schema node, in every walk mode | the `Schema` enum in `ir.rs` | the node is driven through each entry point and the answer asserted | `tests/test_node_matrix.py` |
| every public name a caller reaches | the type stub the package ships | the product suite *names* the cell, read from the syntax tree with the prose cut | `tests/test_use_case_ledger.py` |
| every error code a report can carry | the walk that writes them | the same: the suite names the code | `tests/test_use_case_ledger.py` |
| every result, obligation and deviation the design rests on | `10-theory.md`'s tags, with the debt it admits recorded | a `HELD-BY:` names a test that fails when the sentence is false | `tests/test_theory_ledger.py` |
| every `SOURCE:` line, against the argument it quotes | the argument, which the distribution does not carry | the quotation is found where the line says it is -- **on a clone that has the argument only**, so these rows skip in every lane and run on the maintainer's gate | `tests/test_theory_ledger.py` |
| every refusal the frontend writes | the error constructors in `build*.rs` | a test matches the message the constructor writes | `tests/test_frontend_refusals.py` |
| every code a report can carry, in both modes and on both paths | the same walk | the code is driven at that mode and on that path | `tests/test_error_matrix.py`, held by `tests/test_use_case_ledger.py` |
| every ordered pair of schema variants | the `Schema` enum in `ir.rs`, one representative each | the pair is proved, refuted with a value the walk checks, or declined with a reason | `tests/test_relation_ledger.py` |
| every form the schema-language pages tabulate | the tables in `03-schema-language.md` and `05-refinements.md` | the form is accepted with its `repr` and a member, or refused with the refusals ledger's pattern | `tests/test_form_ledger.py` |
| every entry of the published decidability boundary | the three lists in `15-decidability.md` | a decided entry is driven to its answer, a conservative one to `undecided`, an undecidable one to its refusal or its atom | `tests/test_boundary_ledger.py` |
| every constraint, against every kind a base's values have | `Constraint` in `ir.rs` and `Kind` in `kind.rs` | the constraint narrows the base, with a value it holds of and one it does not, or the build is refused with the words naming what the base cannot answer | `tests/test_constraint_matrix.py` |
| every outcome a method's docstring names | the binding's `Raises:` blocks | the call sits inside a `pytest.raises` for it, read from the syntax tree | `tests/test_surface_outcomes.py` |
| every code under every location shape | the error-model page's location grammar, over the code table | the code is driven at that shape in both modes with the path segment asserted | `tests/test_error_matrix.py` |
| every declared bound | every integer `const` in the crates whose name says it limits something, and the three the stub exports | a test names it by identifier or by a `# BOUND:` marker, or it is accepted with a reason | `tests/test_bound_ledger.py` |

The number is computed rather than written down, and where this page repeats
one, a test holds the copy to the computation. A cell with no test fails; a
cell a test cannot reach is accepted with a reason, and a reason for a cell that
*is* reached fails too, so an excuse cannot outlive the gap it excuses. The
"covered" column is the third direction, and the reason each row needs its own
words: reaching a node is not asserting its answer, naming a method is not
driving its documented failure, and a pair of variants is covered by a
*witness* or by a reason, never by silence. A table that said only "a test
touches it" would report nine products as one claim and be wrong about eight of
them.

A cell is covered when a test *does* something with it, not when a paragraph
mentions it: the search runs over the code with the comments and docstrings cut,
and reading it with them in must find no cell the tighter reading misses. A test
that cannot spell a cell's name claims it with a `# USE-CASE:` marker instead,
and a marker naming no cell fails -- rare by design, because a marker is
bookkeeping a reader keeps true and a name in the code is not.

The largest is the public surface and the error codes together, and the lane
prints them **apart**, because one figure over two products hides which of the
two is growing. The public surface counts the operator surface too -- the
dunders the stub declares behind `in`, `|`, `==`, `hash`, `copy` and `pickle`
-- each reached only where the suite writes the operator:

| Product | Cells | Named | Asserted | Empty, with a reason |
|---|---|---|---|---|
| every public name a caller reaches | 38 | 38 | 38 | 0 |
| every error code a report can carry | 41 | 35 | 35 | 6 |

**Named and asserted are two different claims**, and the ratio the lane prints
is the second. A cell is *named* when the suite spells it, which is what a
search over the source can see and is satisfied by a call whose result nothing
reads. It is *asserted* when it appears inside an `assert` or a
`pytest.raises`, which is the suite saying what the cell answers. Reporting
only the first counts a name the suite never asks a question about, which is
the half a mention count cannot see; the two figures are equal today, and they
are printed apart so the day they part is a day the page shows it.

**And a cell is read by its receiver, not by its word.** A method is a call on
a receiver, an attribute is a read off one, and a read off the name a
`pytest.raises` block binds is pytest's exception rather than the cell: the
word `value` is on every line that catches an error and names
`ValidationError.value` on almost none of them, and a search over words
reported the attribute as asserted by the whole suite. The reading is the
syntax tree's, so `caught.value.code` counts `code` and not `value`.

Every empty cell is a **code**, and splitting the figure is what makes that
readable: five are arms no schema a caller can build reaches, and the sixth is
`validation_error`, the exception's type name carried where a report is built
from no violation at all -- a code the walk can write, however much the name
reads like a class. The public surface has none: every name the stub ships is
named by the suite, so a single total would have reported a gap against the
product that does not have one. `scripts/use_case_ledger.py` prints those figures
and `tests/test_use_case_ledger.py` holds this table to them, so the page is
what fails on the commit that adds a name -- where the older wording,
"seventy-odd cells", could not fail at all, being an approximation of two
numbers added together. What the *ledger* holds is still a floor rather than a
count: the universe grows with the tree, and a test pinning today's total would
fail on the commit that adds a name rather than on the one that leaves it
unreached.

**The outcomes a method documents are a product of their own.**
`tests/test_surface_outcomes.py` reads the binding's `Raises:` blocks — every
`(method, exception)` pair the doc comments promise — and holds each to a test
that puts the call inside a `pytest.raises` for it, read from the syntax tree
rather than from the text. That is the half the name ledger says it cannot see,
for the one outcome a derivation can name: a documented failure nobody drives is
how a page promises a `TypeError` while the suite calls the method only on
values that succeed. `BaseException` is read as the top of the exception
lattice, because a docstring saying it means whatever signal reaches the caller
and the suite drives the concrete ones.

What a ledger cannot see is the half that matters most: naming a method is not
asserting its documented outcome, and reaching a node is not checking every
answer it gives. That is what the matrices beside them are for — the node matrix
runs each node through all six entry points, and `test_error_contract.py` holds
each promise the error model makes at each site that makes it. The ledger's job
is narrower and nothing else does it: a name the tree grows and the suite never
mentions, which is the state every one of them starts in.

**The denotation oracle reads every case at the boundaries.** It pairs a drawn
schema with an independent predicate and checks a drawn value, and a boundary
reached only by the draw is one the suite reaches on some runs and not others:
measured over six hundred draws, three of five boundary kinds never appeared. So
each case is checked against a fixed spread as well -- `nan`, the infinities,
`-0.0`, the ends of the integer carriers, a newline in text and in bytes -- and
a disagreement about one of those is a failure rather than a flake.

**The lattice laws draw their schemas rather than sampling a list.** A law held
over a fixed spread of atoms and containers is a law about that spread. The
strategy builds around a drawn element, so a refinement can sit inside a list,
and it reaches three shapes the list never had: a refinement, whose meet
compares constraints rather than kinds; a class, whose membership is
`isinstance`; and a fixpoint, which the descriptor cannot hold, so every law
over one is decided by the rules. The witness spread carries the boundaries a
drawn value reaches only by luck — `nan`, the infinities, `-0.0`, the ends of
the integer carriers, and a newline.

**And each case runs under the allowance a relation builds its difference
under.** A law asks the descriptor operations directly, where a relation asks
them through `decision`, and the operations hold no ceiling of their own: each
lattice's width bound refuses a result too wide to represent and says nothing
about how long reaching a narrow one may take. So a law without an allowance
costs whatever its draw asks, and a suite whose cost is a property of its draw
is one the sweep cannot be sized against -- a mutant's timeout is a multiple of
the baseline's test time, so a baseline free to move between runs makes every
verdict in the shard a statement about the seed rather than about the mutation.
Every law that reads a refusal as a skip arms one per case through
`budget::law`, over whole descriptors in `descr/tests.rs` and over each lattice
in its own `tests.rs`; the laws over `descr()` assert that their operations
*succeed*, which is a claim about a fragment rather than about a build, and
they do not.

**A bounded shrink is the other half and does not stand in for it.** Each law
block bounds `max_shrink_time` so a broken invariant cannot outlast a sweep,
which is what a *failure* costs; the allowance is what a *passing case* costs.
The two come apart under a mutation that deletes a pruning shortcut: every case
grows dear and none of them breaks, so there is no failure to shrink, and a
mutant the sweep cannot finish returns no verdict at all --
`scripts/mutation_gate.py` reads that as a rig fault rather than as a survivor,
which is the one outcome a sweep cannot ratchet. To see what an unarmed law
costs under such a mutation, run one:

```bash
cargo mutants -p valgebra-core -j 2 --timeout-multiplier 20 \
  -F 'MapLattice<G>::complement'
```

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

**Adequacy is measured per harness, and every file has one.** Two of the three
sweeps run `cargo test`, so a survivor of either is a gap in the *Rust* corpus:
that is the reason a swept binding file carries a corpus of its own. The seven
files the extension is the only caller of carry none, and a `cargo test` sweep
over them measures the harness rather than the tests -- every mutant survives,
whatever the Python suite does.

The third sweep is those seven files, with a test command that loads the
extension. `crates/valgebra-py/tests/pytest_sweep.rs` rebuilds it from the
mutated copy and runs the Python suite against it, so a mutant one of them
carries is caught by that suite or by nothing, and the survivors are ratcheted
against a baseline of their own. The excuse those files were excluded under is
a number now: `tests/test_pytest_sweep_scope.py` holds the two configurations to
a partition of the binding, so a file excused to the suite and examined nowhere
fails.

**What a coverage figure leaves over, and why.** Reading the annotated report
for the core's shipped scope leaves **under a hundred** statement lines no
test executes, and they are four kinds rather than a backlog. The figure is
rounded on purpose: an exact one is right for a day and no gate reads it, and
this page has been wrong before by carrying a number nothing held. Take the
reading in a fresh `CARGO_TARGET_DIR` -- a profile merged against objects
from an earlier tree reports lines as unreached that are not -- and the
per-file floors in `scripts/branch_coverage.json` are the figures a gate does
hold.

- a `debug_assert!(false, ...)` and the `return` beside it. Each states an
  invariant a *constructed* value cannot break -- the guards leaving a state
  cover the letters, two components of one kind meet -- so the arm is reachable
  only from a table built by hand, and a test that built one would be asserting
  about a value the tree cannot produce;
- the arms past a width bound. A union too wide to rebuild is a shape the
  representation refuses, and driving one means building it, which the same
  bound refuses first. `budget::under` reaches some of them by shrinking the
  allowance instead, which is how the map lattice's declining verdict is
  driven, and the rest are the same arm one representation over;
- the deprecated simplifier's folds, which go with it in the next minor;
- an arm only the binding reaches. A label a report carries, a path segment a
  key renders into: the core writes them and the Python suite reads them, so
  the *line* runs on the other side of the boundary from the crate whose figure
  this is.

**A fifth kind was there and is not a kind: a branch the caller already
answered.** Three of them, and no input reached any: a reflexivity check whose
caller returns on schema equality one frame up, a region comparison its caller
makes four lines above, and the arm of a search for a class the guard beside it
had matched. They read like the first kind and are not -- an invariant a
*value* cannot break is a line worth keeping, while a question answered one
frame up is a line to delete. What separates them is whether the answer comes
from the shape of the data or from the shape of the call, and the second is
closed by deletion rather than by a test. Two files read every line and every
region once those went and the walk through a resolved reference gained the
test it never had.

None of the four is closed by a test worth writing, and saying so is what
separates them from the lines that were: a table over the node set closed a
file's worth of them in one commit, and it closed them because they were
reachable and nobody had asked.

**Some arms only a free-threaded interpreter reaches, and no lane measures
one.**
The walk snapshots a container and compares the snapshot against what it read,
because the container can move underneath the reading -- and under a global
interpreter lock another thread cannot be the one that moves it. Three tests
skip for that reason -- five items, since the one about a container written
underneath a walk is parametrised over the shapes that can move -- the arms they
would drive sit in the sequence and record walks,
and the binding's coverage lane runs on a single-threaded release. So those
arms are executed on the free-threaded leg of the python matrix and *measured*
nowhere: the figure the lane prints is a figure about the interpreter it ran on.
Naming that here is the honest reading, because the alternative -- a second
instrumented build per push -- buys a number for arms the matrix already drives.

**A coverage floor is read with its scope or it misleads.** The Python package
floor covers the re-export package, which is a hundred-odd lines; the extension
the Python suite exercises is Rust and is measured by the other two lanes.

Those two measure the code that **ships**, and did not always. The core's
property suites and the binding's four interpreter corpora are compiled into
their crates because they reach private items, so a coverage report counts
them — and a corpus is a table and a loop, so it runs by construction, arrives
at full coverage, and lifts the figure for the code around it. Nearly half the
lines the binding's figure was computed over were corpus. Both lanes exclude
them now, and `tests/test_coverage_scope.py` holds that scope to the corpus
files the tree has, in both directions.

**A coverage figure read on a developer's box is not the lane's, and the
difference can be one file.** `cargo llvm-cov` attributes a region to a source
span, and which span it picks for a `const` initialiser or for an item inside a
`#[cfg(test)]` module is the toolchain's business, not the tree's. On one
machine `decision.rs` reads about half its regions covered, with seven hundred
of them landing on doc-comment lines, blank lines and `use` items -- lines no
program executes -- while the lane reads the same file at 99% and the same
commit green.

That six-point difference in a **total** is indistinguishable from a real hole
of the same size, which is the reading the audit could not settle from either
figure alone. What settles it is the per-file column: a file whose zero-count
lines are comments has a mapping artefact, and a file whose zero-count lines are
statements has untested code. So the rule is to compare *a file's* figure
between the two runs and read the annotated report for that file, never to
compare totals -- a total absorbs the artefact and the hole equally well, and
says the same number for both.

**And one floor per file beside the scope-wide one.** A total absorbs a hole
the size of a file: half of the decision procedure going unreached moves a
scope-wide figure by about the width of the tolerance any figure carries
between machines, so the two readings are indistinguishable in a total and only
one of them is a defect. `scripts/branch_coverage.json` therefore records a
region floor per file beside the branch figure, ratcheted the same way, and a
file that loses its tests fails under its own name. Both directions: a floor
for a file the measurement no longer carries fails too, since a scope that
quietly stops measuring a file would otherwise read as a file that never
regressed.

And a nightly lane counts the **arms**. A region is a span the compiler emits,
and a two-armed branch inside one span contributes one region — so the core reads
98% of lines, 97% of regions and 90% of branches on the same files. Eight points
of arms sit under a floor both other figures pass. `cargo llvm-cov` has no
`--fail-under-branches`, so the number is recorded in
`scripts/branch_coverage.json` and ratcheted the way the mutation baseline
ratchets survivors: measured, compared, and moved up with the measurement and
never ahead of it.

Each lane enforces a **region** floor beside its line floor. A line counts as
covered when any part of it ran, so a branch with two arms on one line passes
having taken one; a region does not, and on the shipped scope the binding reads
several points lower in regions than in lines. The floor that notices an
unreached arm is the region one.
