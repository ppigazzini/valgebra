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

## What fixes a node's carrier, and why it differs per node

A structural node denotes values of a *shape* held by some *carrier* — a Python
class. The three structural nodes fix their carrier three different ways, and
the difference decides how large a change is that widens one:

| node | how the carrier is fixed | widening it means |
|---|---|---|
| `Seq { container: SeqKind, shape }` | a **parameter** — `SeqKind` is `List \| Tuple` | another variant in an existing enum |
| `KeyedMap { fields, defaults }` | **the denotation** — the doc comment reads "Denotes dicts…", and there is no carrier field | giving the node a carrier it does not have |
| `Attrs { class_index, fields }` | a **required field** | making a required field optional, so the node denotes a set it currently cannot |

Read that table before answering "can valgebra express `Mapping[K, V]`" or "can
it express *any object* whose `.a` is an `int`". The answer is no in both cases,
and the three nodes are not one mechanism seen three times.

## Whether to add a variant

Before the mechanics below, the admission test. **valgebra is the smallest set
of schema nodes whose Boolean closure is consistent and complete for its
domain** — that is the definition, not a preference, and it is what makes
"the algebra" a claim rather than a collection.

So a proposed node is one of exactly two things:

1. **Its set is already in the closure of the existing atoms.** Then it is
   redundant: write it as a combination and add nothing.
2. **Its set is not.** Then it is an extension, and the case for it has to be
   that the algebra is *incomplete for its domain* without it — not that it
   would be convenient, and not that the code already computes something like it
   internally.

There is no third case. "It would be convenient", "a downstream user wants it",
and "the machinery is nearly there" are all case 2 arguing under another name,
and the only honest way to make them is to say which part of the domain is
unreachable without the node.

**The domain is the second half of the test**, and it is a real question rather
than a formality: valgebra exposes its algebra through standard Python typing
syntax, so a set that syntax can express and valgebra cannot is a completeness
gap, while a set only a bespoke combinator could name is not. Whoever owns the
algebra decides where that line sits; a page here records the decision, and
[10-theory.md](10-theory.md) already shows the form such a decision takes — the
map clauses are unordered where the paper's are ordered, written down as "a
deliberate narrowing".

### What does not count as evidence

Three arguments look like they settle case 1 and do not:

- **"The behaviour already looks right."** A `Validator` probe shows what the
  walk does. The walk can agree with a denotation by accident, and a node's
  denotation is what the doc comment in this file says.
- **"The implementation already computes it."** `build_object` computes
  per-attribute checks on its way to an `Attrs`. That does not put "any object
  with these attributes" in the closure, because the node it builds denotes
  instances of a class.
- **"It only changes which values a constructor sees, not the algebra."**
  Subtyping is defined from the denotation (`[[s ∧ ¬t]] = ∅`), so changing which
  values inhabit a constructor changes the subtyping relation. There is no lever
  that separates the two.

## Refused, and the test each one fails

These do not go in. Each carries the test it fails, so a proposal starts here
rather than at the beginning.

**A carrier for `KeyedMap`, so `Mapping[K, V]` and `Sequence[T]` build.**
Refused. It is a constructor extension, not an encoding change: `KeyedMap`
denotes dicts and has no carrier field, so supplying one changes the
constructor. Beyond the admission test, two costs specific to this one:

- The model `KeyedMap` is built from is recorded in
  [10-theory.md](10-theory.md) as *records and maps as quasi-constant
  functions* — named fields plus a key-typed default. No carrier appears in that
  description. Whether the source paper treats a nominal carrier is **not
  established here**, and its title names structs, so do not assume it does not.
  What follows is only that this project has no recorded reading of a
  carrier-indexed map: an absence of guidance, not a permission and not a
  prohibition.
- The decision procedures this project sources decide maps **without** a
  carrier: Elixir's `Module.Types.Descr`, and the negated-map-atom decomposition
  for deciding a keyed map under negation. A carrier-indexed map is outside the
  fragment with a published decision, which is the part the completeness claim
  rests on.

**A carrier-free attribute form, so "any object whose `.a` is an `int`" builds.**
Refused. `Attrs` requires its `class_index`; the set is denoted by no type, and
making the field optional makes the node denote a set it currently cannot.

**Records keyed by a non-string.** Refused. A record's fields are named by
strings, and arbitrary keys are a different labelling than the record model
assumes.

### Arguments for the carrier change, and why each fails

| argument | why it fails |
|---|---|
| "the behaviour already looks right" | a probe shows the walk, not the denotation |
| "the implementation already computes it" | `build_object` computes attribute checks *inside* a node whose denotation includes the class |
| "it only changes which values a constructor sees" | subtyping is defined from the denotation, so that *is* the algebra |
| "the sources name no carrier, so adding one is free" | an absence of guidance is not permission. It also is not a refusal — the refusal rests on the admission test, not on this row |
| "a class carrier is unsound because `register()` mutates it" | false — class relations are re-decided on every call, never cached |
| "a downstream package needs it, and pays to route around it" | a consumer's requirement is a fact about the consumer. It is the loudest argument for growth and the weakest: every package built on valgebra will want the node that would make its own job easier |

Two rows need separating from the rest.

The `register()` row is an argument *against* the change, and it is also wrong,
so it is not a reason to refuse. The refusal rests on the admission test and the
two costs above, and on nothing else.

The downstream row is the one to watch, because it arrives with evidence
attached — a measured cost, a real user, a working reproduction — and none of
that bears on whether the set belongs in the algebra. **valgebra is not extended
to make a consumer's job easier.** A package that cannot express something with
the algebra as it stands has found a fact about itself; if the same limitation
also fails the admission test on its own terms, the test is what carries it, and
the consumer is at most the reason someone looked.

The asymmetry is the point: a consumer can always route around a missing node —
with a predicate, a conversion, its own translation — at a cost it measures and
accepts. The algebra cannot route around a node that should not have been added.

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

Being the same operation, it is one walk: `Remap` names the difference — append
by a distance, or intern through a table — and the walk asks each index space how
it moves. A definitions index moves the same way under both, which is what makes
one walk enough. The two were separate walks, identical but for the leaf action,
and a payload site reached by one and missed by the other was a wrong index that
neither the types nor an exhaustive `match` could see: the compiler forces an arm
per variant and cannot check that the arm moved anything. `Schema::map_children`
holds the other half of that argument — it is the single place each variant's
child schemas are written down, so a walk that only descends inherits the child
set instead of restating it. `Schema::remapped_by` takes no wildcard on purpose:
a future variant carrying a pooled index must be a compile error there rather
than a node that silently keeps an index into the wrong pool.

The laws in `crates/valgebra-core/src/lib.rs` hold both entry points to moving
every payload by its own space's distance, counted through an enumeration written
in the test module rather than reached for in the IR — a check that judges the
walk against something other than itself.

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
