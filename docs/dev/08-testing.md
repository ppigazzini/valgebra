# Testing

Correctness here is checked against the **denotation**, not against itself. A
test that asserts the validator agrees with the validator proves nothing about
what a schema means.

## The layers, and what each can see

| Layer | Judges | Blind to |
|---|---|---|
| Denotation oracle | the walk, against an independent Python predicate over generated values | what the generator does not draw |
| Model oracle (Rust) | the IR, `simplify` and the decision procedure, against a value model | the walk — the core cannot see Python |
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
for and misses the other. Eight of them:

| Ledger | Holds |
|---|---|
| `tests/test_lane_coverage.py` | every gate script runs in a lane; no excuse is stale |
| `tests/test_mutation_scope.py` | every binding file is swept or excluded by name |
| `tests/test_sweep_skips.py` | every `SWEEP-SKIP` mark is skipped; every skip is marked |
| `tests/test_build_surfaces.py` | every manifest is a workspace member or a named detached surface |
| `tests/test_harness_conditionals.py` | every `cfg(feature = ..)` site is test-only, or named |
| `tests/test_contract_inventory.py` | every gate script has a contract row; every row names a real source |
| `tests/test_completeness_probe.py` | every suspected completeness gap is accepted with a reason |
| `tests/test_completeness_ledger.py` | every relation the procedure must decide is decided; every recorded miss is still a miss |
| `tests/test_laws.py` | the complement laws decide against a respelling the procedure proves equal |

Each declares itself with a `LEDGER:` marker, and `scripts/docs_lint.py` holds
this table to those markers both ways, so a ledger added without a row fails
rather than passing quietly. The count is spelled here and in the glossary
because a table nothing counts is the one that drifts: there are eight.

`tests/test_node_matrix.py` is the same shape one level in: it reads the `Schema`
variants out of the IR and fails when one carries no row, so the universe is
derived from the tree rather than restated.

Each ledger asserts it read something. A check over an empty universe passes
having checked nothing, which is worse than a bare failure.

## The walk's own corpus

`cargo test` cannot reach the membership walk without an interpreter, so the walk
carries its own value corpus in `crates/valgebra-py/src/check/walk.rs` under the
`interpreter-tests` feature, which links an embedded Python.

Every case runs in **both** the fast and the explaining mode with the two
required to agree, and with the violation count asserted where it distinguishes
the modes. A corpus driven only fast leaves half of every composite unobserved.

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
procedure-agnostic laws — panic-freedom, `simplify` idempotence, reflexivity,
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
gap in the Rust corpus and not necessarily in the tests as a whole.

**A coverage floor is read with its scope or it misleads.** The Python package
floor covers the re-export package, which is a hundred-odd lines; the extension
the Python suite exercises is Rust and is measured by the other two lanes.
