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
| Metamorphic | the JSON path against the object path; fast mode against explain mode | a defect both sides share |
| Algebra laws | every claimed equivalence, against membership | a law nobody thought to claim |
| Completeness ledger | that the procedure *decides* an enumerated relation | a relation nobody enumerated |
| Snapshots | error messages and `repr` | whether the message is *right* |
| Fuzzing | panic-freedom, idempotence, the order laws | anything outside the budget |

The denotation oracle is the load-bearing one: it shares none of the validator's
frontend, so an agreement is evidence rather than a tautology.

## Two directions, always

Every list in this repository that could rot is held to the tree in **both**
directions, because a hand-written list satisfies the direction it was written
for and misses the other. Four of them:

| Ledger | Holds |
|---|---|
| `tests/test_lane_coverage.py` | every gate script runs in a lane; no excuse is stale |
| `tests/test_mutation_scope.py` | every binding file is swept or excluded by name |
| `tests/test_sweep_skips.py` | every `SWEEP-SKIP` mark is skipped; every skip is marked |
| `tests/test_build_surfaces.py` | every manifest is a workspace member or a named detached surface |

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
defect** — it makes the tested build differ from the shipped one — so a future
`cfg(feature = ...)` on a production path is a change to argue for, not a
convenience.

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
