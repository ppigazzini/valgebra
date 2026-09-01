# The theory

What the design rests on, and which code each result touches. A citation here is
a claim that a specific line exists because of it, not a reading list.

Each entry is tagged **LOAD-BEARING** (code here rests on it), **GUIDING**
(shapes a decision without being an algorithm), or **PLANNED** (on the path, not
built). A planned reference is an intention; it never implies the thing is built.

Some of this is decades old and stays, because a theorem does not expire.
Stone's representation theorem and Nakano's guarded fixpoint are exactly the
results the design leans on, not dated approximations of them.

## The frame: a schema denotes a set

**Scott and Strachey, denotational semantics.** A schema's meaning is the set of
Python values it admits; validation is membership in that set; subtyping is
subset inclusion. **[LOAD-BEARING]** — every variant's doc comment in
`crates/valgebra-core/src/ir.rs` states a set, and
[01-schema-ir.md](01-schema-ir.md) is that frame written out.

The consequence worth naming: because subtyping is inclusion and Python makes
`bool` a subclass of `int`, `bool` is a **subtype** of `int` rather than disjoint
from it. That is not a valgebra choice; it follows from the frame plus a fact
about Python.

## The algebra: a Boolean lattice of value sets

**Birkhoff, _Lattice Theory_.** Union, intersection and complement over value
sets form a Boolean algebra; the simplifier's rewrites are its laws.
**[LOAD-BEARING]** — `crates/valgebra-core/src/simplify.rs`, and the property
suites that check each claimed equivalence against membership rather than
asserting it.

**Stone's representation theorem (1936).** Every Boolean algebra is isomorphic to
an algebra of sets. **[GUIDING]** — the licence for treating the scalar fragment
as a bitset over disjoint regions, which is what makes emptiness and subtyping
exact there ([02-decision.md](02-decision.md)).

## Semantic subtyping

**Frisch, Castagna & Benzaken, "Semantic Subtyping" (JACM 2008).** The model
valgebra *is*: value sets with union, intersection and negation, where subtyping
is `[[s ∧ ¬t]] = ∅`. **[LOAD-BEARING]** — but read the qualifier, because the
short version of this sentence was wrong for as long as it was written.
`is_subtype_of` is **not** that reduction applied uniformly. It is a structural
procedure whose arms decompose each pair by shape, and it calls `is_empty` at the
three places where no shape is available to recurse into: the two lattice bounds,
and a complement on the right. Everywhere else the arms decide directly.

The distinction is not pedantry. A rule stated as the reduction and implemented
structurally has a hole wherever an arm is missing, and the reduction's name over
the top is what stops anyone looking: a complement on the right had no arm at all
for months, and this page said the reduction covered it.

**It does not carry a theorem that a conservative simplifier cannot be complete
once negation exists**, and this page said it did. There is no result of that
form in the paper. What it has is an argument for the semantic approach, made of
a syntactic rule set rather than of a simplifier:

> when considering arrow, intersection and union types, one must take into
> account — that is, introduce rules for — many distributivity relations such as,
> for instance, `(t1 ∨ t2) → s ≃ (t1 → s) ∧ (t2 → s)`. Forgetting any of these
> rules yields a type system that, although sound, does not match (i.e., it is
> not complete with respect to) the intuitive semantics of types.

A rewriter missing rules is sound and incomplete. That shape is why
[02-decision.md](02-decision.md) states soundness as the contract and treats
completeness as a measured, growing property rather than a promise — but the
step from an observation about arrow distributivity to an expectation about this
simplifier is **valgebra's reasoning, not a result the paper proves**.

**Castagna, "Programming with Union, Intersection, and Negation Types" (2023).**
The modern synthesis. **[GUIDING]**

**Castagna, Duboc & Valim, "The Design Principles of the Elixir Type System"
(2023).** The same algebra in a production language, with the engineering
compromises stated. **[GUIDING]** — the closest thing to a peer implementation.

## Sequences as a regular language

**Hosoya, Vouillon & Pierce, "Regular Expression Types for XML" (TOPLAS 2005).**
Regular languages are closed under union, intersection and complement, so a
sequence type is a first-class member of the algebra rather than an ad-hoc node.
**[LOAD-BEARING]** — `SeqRegex` is one node subsuming the homogeneous, fixed and
prefix-plus-tail forms for exactly this reason
([01-schema-ir.md](01-schema-ir.md)).

## Records and maps

**Castagna, "Typing Records, Maps, and Structs" (ICFP 2023).** One node with
named fields plus default clauses subsumes the record, the homogeneous mapping,
the heterogeneous mapping and their combination. **[LOAD-BEARING]** —
`Schema::KeyedMap`, where a closed record is no default clause and `dict[K, V]`
is a single clause with no fields.

The paper's clauses are *ordered*, with the first matching clause governing a
key. valgebra's are not: a key belongs when **some** clause admits it and its
value, in the walk and in subtyping alike. That is a deliberate narrowing, and
the node's doc comment said "ordered" for a while, which is a semantics no code
here implements.

## Recursion

**Nakano, guarded recursion (2000).** A contractive map has a **unique** fixpoint.
**[LOAD-BEARING]** — that, not Knaster–Tarski, is what `recursive` denotes, and
this page said otherwise.

The distinction is forced by `complement`, which is **antitone**. A guarded body
may therefore be non-monotone, and one is reachable:
`recursive(lambda x: [complement(x)])` builds and validates, with the
self-reference under `list` so the contractiveness check accepts it. It has no
monotone `F`, so it has no Knaster–Tarski least fixpoint. What makes it well
defined is the guard: each unfolding consumes one constructor of a finite value,
so the recursion is productive and the fixpoint is unique whether or not the body
is monotone.

**Tarski's fixpoint theorem (1955).** A monotone map on a complete lattice has a
least fixpoint. **[GUIDING]** — it applies to the complement-free fragment, where
the unique guarded fixpoint coincides with the least one. It is not the general
justification.

The guard is therefore load-bearing twice over: it makes the fixpoint unique, and
it is what the contractiveness check in `crates/valgebra-core/src/ir.rs` enforces
by requiring every self-reference to sit under a constructor.
[01-schema-ir.md](01-schema-ir.md) records why its structural arms compute
nothing.

**Amadio & Cardelli, "Subtyping Recursive Types" (1993).** Subtyping between
recursive types is decided coinductively: assume the goal, unfold, and a cycle
back to an assumed goal is a proof. **[LOAD-BEARING]** — the assumption stack in
`crates/valgebra-core/src/decision.rs`.

Coinduction also governs the *value* side: a value that contains itself is
refused by identity rather than followed ([04-walk.md](04-walk.md)).

## Gradual typing

**Siek & Taha, "Gradual Typing for Functional Languages" (2006).** The dynamic
type is an atom with its own rules, not the top of the lattice.
**[LOAD-BEARING]** — `Schema::Dynamic` is distinct from `Schema::Anything`
precisely so the simplifier does not rewrite it by the lattice laws. At runtime
both admit everything; they are not interchangeable to the algebra, and
`docs/15-decidability.md` states where the gradual atom is deliberately never
decided.

## Refinement types

**Jhala & Vazou, "Refinement Types: A Tutorial" (2021).** A refinement is a base
set narrowed by predicates. **[GUIDING]** — `Schema::Refine` is the shape without
the SMT machinery: bounds are compared through the oracle, and a user predicate
is opaque.

## Decision procedures, for widening the decided fragment

**Gesbert, Genevès & Layaïda, "A Logical Approach to Deciding Semantic
Subtyping".** **[PLANNED]** — the interning and automata engine that would decide
the cases [02-decision.md](02-decision.md) records as conservative. Not built;
saying so is the point of the tag.

## Property-based testing

**Claessen & Hughes, QuickCheck (2000)**, and the modern shrinking work behind
hypothesis and proptest. **[LOAD-BEARING]** — every algebra law is proved against
membership by a property suite rather than asserted
([08-testing.md](08-testing.md)).

**Chen, Cheung & Yiu, "Metamorphic Testing: A New Approach for Generating Next
Test Cases" (1998)**, and **Chen et al., "Metamorphic Testing: A Review of
Challenges and Opportunities" (2018).** Derive a test case from one that passed
and check a relation between the two outputs; neither run needs an oracle.
**[LOAD-BEARING]** — the JSON path against the object path, and fast mode against
explain mode. Both relate a source run to a follow-up run, which is what makes
them metamorphic relations rather than invariants that happen to hold.

The two citations are not interchangeable, and this page carried only the first.
The 1998 report states the approach. The **criterion** for saying a property
qualifies is the 2018 review's: an MR relates multiple inputs and their outputs,
so a necessary property of a single input is not one — the review's example is
that `-1 ≤ sin(x) ≤ 1` is necessary and is not an MR. Neither "metamorphic
relation" nor "necessary property" occurs in the 1998 report; both are later
refinements, so a claim that rests on the criterion owes the later citation.

The review also states the limit that bounds what `docs/14-soundness.md` may rest
on these suites: MRs are *necessary* properties, so even a complete set of them
is not a test oracle. That page's trust base records it.

## The limit

**None of these is implemented as its paper describes it.** The decision
procedure is a sound, budget-bounded structural one that is exact on a published
fragment; it is not the interning automata engine. Where a page here says
otherwise, the page is wrong.

Tooling and toolchain facts are [11-references.md](11-references.md), not this
page.
