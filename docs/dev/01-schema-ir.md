# The schema IR

`crates/valgebra-core/src/ir.rs` owns the node set and the pure structural
operations over it: construction, index shifting, self-reference resolution, and
the guardedness check.

## A node denotes a set of Python values

That is the whole frame, and every variant's doc comment states its set.
`Schema::Int` denotes the `int` instances; `Schema::Union` denotes the union of
its members' sets; validation is membership in the set the root node denotes.

Two consequences a reader needs before touching this file:

**Subtyping is subset inclusion, so `bool` is a subtype of `int`.** Python makes
`bool` a subclass of `int`, so `True` is an integer and `Schema::Int` admits it.
No value is carved out. `Schema::Float` is disjoint from `Schema::Int` for the
mirror reason: `int` does not subclass `float`.

**A `Literal` is a *typed* singleton.** Python's `==` conflates across types
(`1 == True == 1.0`), so equality alone would make `Literal[1]` admit `True` and
`1.0`. Membership requires `type(x) is type(c)` as well, which is what keeps the
typing spec's distinction between `Literal[1]`, `Literal[True]` and
`Literal[1.0]`. The same-type test runs **before** `==`, so a value of another
type never reaches the comparison — which is why a raising `__eq__` is only
observable from an object of the pooled constant's own type.

## Adding a variant

The compiler forces the exhaustive matches; the doc comment on `Schema` carries
the checklist of what it forces. Read it there rather than here: a second list
drifts by one entry and reads exactly like one that has not.

Two things the compiler does not force, and both are held by a test:

- **A representative in the node matrix.** `tests/test_node_matrix.py` reads the
  variants out of this file and fails when one carries no row, so a node cannot
  arrive without a case that exercises it.
- **A case in the denotation oracle.** `tests/test_denotation.py` pairs each
  generated schema with an independent Python predicate; a node it never draws is
  a node the oracle does not check.

## What the payloads address

Five payloads are integers addressing something the validator holds, and four of
them address the **same** constants pool: a literal's constant, a class, a
comparison operand, a user predicate. Each has its own type, and
[06-type-design.md](06-type-design.md) owns why and what a crossing would cost.

`Constraint::MinLen` and `Constraint::MaxLen` sit in the same enum and are **not**
pool indices at all — they carry the length inline. That is what the type says: a
length has no `shifted(PoolShift)`, so the arms that must not take a pool shift
cannot.

## Composition, and the two shifts

Two validators combine by concatenating their constants pools and their
definitions tables. The second schema's indices move past the first's lengths,
which `Schema::shifted` does — one shift per index space, and they are distinct
types so a caller cannot transpose them.

`Schema::reindexed` is the same operation where the second pool is *interned*
into the first rather than appended, so identity-shared constants collapse to one
slot. It is the one the binding actually calls; `shifted` is reached from the
tests and the fuzz targets.

## Recursion, and why the guardedness check answers what it does

`recursive` builds a fixpoint. The body is compiled with a `SelfRef` marker,
which `resolve_self` rewrites to a `Ref` into the definitions table before the
validator is returned — so no compiled schema holds a `SelfRef`, and the walk
treats one as a non-member if it ever sees one.

A definition is admitted only when it is **contractive**: every occurrence of the
self-reference sits under a structural constructor. `Schema::occurs_unguarded`
decides that, and its shape is worth stating because it is not obvious from
reading it.

**`Guarded::Yes` is absorbing.** The only arm that can answer true demands
`Guarded::No`, and the algebraic combinators pass the guard through unchanged, so
nothing below a structural constructor is ever reported unguarded however deeply
it nests. Each structural arm therefore answers false for every input — the same
answer the match's default gives. The arms are written out because they state
*which* constructors guard; they compute nothing.

Two properties in `crates/valgebra-core/src/lib.rs` pin that rather than leaving
it as a comment: the guard absorbs under every structural constructor, and read
from the top the check agrees exactly with "the reference is reachable through
algebraic combinators alone".

## The limit

The IR is a tree with back edges, not a graph with sharing. Two structurally
equal subtrees are two allocations, and nothing interns them. That is why the
decision procedure carries a work budget instead of a memo table
([02-decision.md](02-decision.md)), and it is the single largest thing standing
between the current procedure and a complete one.
