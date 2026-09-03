# The theory

What the design rests on, and which code each result touches. A citation here is
a claim that a specific line exists because of it, not a reading list.

Each entry is tagged **LOAD-BEARING** (code here rests on it), **GUIDING**
(shapes a decision without being an algorithm), or **PLANNED** (on the path, not
built). A planned reference is an intention; it never implies the thing is built.

Some of this is decades old and stays, because a theorem does not expire.
Stone's representation theorem and Nakano's guardedness modality are exactly the
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
is `[[s ∧ ¬t]] = ∅`. **[LOAD-BEARING]**, with a qualifier: `is_subtype_of` is
**not** that reduction applied uniformly. It is a structural procedure whose arms
decompose each pair by shape, and it calls `is_empty` at the three places where no
shape is available to recurse into: the two lattice bounds, and a complement on
the right. Everywhere else the arms decide directly.

The distinction is not pedantry. A rule stated as the reduction and implemented
structurally has a hole wherever an arm is missing, and the reduction's name over
the top is what stops anyone looking for one.

On the expectation to hold of a simplifier, the paper offers an observation in
§2.2, not a theorem, and it is about a syntactic rule set:

> Forgetting any of these rules yields a type system that, although sound, does
> not match (i.e., it is not complete with respect to) the intuitive semantics of
> types.

A rewriter missing rules is sound and incomplete. Reading that shape onto this
simplifier is **valgebra's reasoning, not a result of the paper**, and it is why
[02-decision.md](02-decision.md) states soundness as the contract and treats
completeness as a measured, growing property rather than a promise.

**Castagna, "Programming with Union, Intersection, and Negation Types"**
(arXiv:2111.03354, revised 2024). The modern synthesis. **[GUIDING]**

**Castagna, Duboc & Valim, "The Design Principles of the Elixir Type System"
(2023).** The same algebra in a production language, with the engineering
compromises stated. **[GUIDING]** — the closest thing to a peer implementation.

## Sequences as a regular language

**Hosoya, Vouillon & Pierce, "Regular Expression Types for XML" (TOPLAS 2005).**
Regular languages are closed under union, intersection and complement, so a
sequence type is a first-class member of the algebra rather than an ad-hoc node.
**[GUIDING]** — `SeqShape` is one node subsuming the homogeneous, fixed and
prefix-plus-tail forms for exactly this reason
([01-schema-ir.md](01-schema-ir.md)).

It is guiding rather than load-bearing, and the gap is the point: the general
form is a regular expression over element schemas, and deciding inclusion
between two of those wants the automaton construction
[15-decidability.md](../15-decidability.md) records as unbuilt. What the IR
carries is the *linear* fragment — a fixed prefix and an optional repeated tail
— which is every shape the schema language can spell. The closure the paper
gives is at the schema level here, through union, intersection and complement
over the sequence node, rather than inside the sequence body.

**The rule that splits a fixed-length sequence across a union** is the product
decomposition — Frisch, Castagna & Benzaken Lemma 6.5 for pairs, in the
backtrack-free form Castagna gives as `Φ`, and generalised to a fixed component
count the way Castagna & Duboc state the tuple rule for larger arities.
**[LOAD-BEARING]** — `product_subtype` in `decision.rs`. It applies in subtyping
and nowhere else: emptiness does not decompose a product, so the same relation
asked as a meet with a complement is not decided.

## Records and maps

**Castagna, "Typing Records, Maps, and Structs" (ICFP 2023).** One node with
named fields plus default clauses subsumes the record, the homogeneous mapping,
the heterogeneous mapping and their combination. **[LOAD-BEARING]** —
`Schema::KeyedMap`, where a closed record is no default clause and `dict[K, V]`
is a single clause with no fields.

The paper's model is a *quasi-constant function*: named labels over a finite
domain, with the rest given by a default keyed by a partition of the key space.
valgebra's clauses are neither ordered nor a partition — a key belongs when
**some** clause admits it and its value, in the walk and in subtyping alike, and
two clauses may claim the same key. That is valgebra's own model rather than the
paper's, and the paper says why it is not theirs: it forbids overlapping domains
in one map, and rejects the leftmost-match reading precisely because a semantic
subtyping relation disregards the order of the fields. The set each node denotes
is well defined; what it does not have is the paper's canonical form, which is
why the domain is the field list as written.

## Recursion

**What makes `recursive` well defined is not a fixpoint theorem.** The values are
finite, so the immediate-substructure relation on them is well founded, and for a
guarded body `F` the statement `v ∈ X ⟺ v ∈ F(X)` is a *definition by
well-founded recursion on `v`*: each unfolding consumes one constructor of the
value, so the question about `v` is answered from strictly smaller questions.
Existence and uniqueness among sets of finite values follow from that induction —
no metric, no lattice, and no monotonicity. **This is the argument the walk
implements**, and it is why `recursive(lambda x: [complement(x)])` is well
defined although `complement` is antitone and the body has no monotone `F`.

Two theorems sit nearby and neither is the justification. **Banach's fixed-point
theorem** — "a contractile map over a complete metric space has a unique
fixpoint", quoted at Amadio & Cardelli §3.3.2 — is the metric account of
recursive types, and its unique fixpoint lives among *infinite* trees, which the
walk never admits. **Tarski's fixpoint theorem (1955)** gives a monotone map on a
complete lattice a least fixpoint, and applies to the complement-free fragment,
where the inductive set is that least fixpoint. Both are **[GUIDING]**.

**Nakano, "A Modality for Recursion" (2000).** The guardedness modality: a
recursion variable under a guard is productive. **[LOAD-BEARING]** for the
*discipline* — `occurs_unguarded` is that condition, and it is what makes the
induction above well founded. The paper proves soundness of a modal type system
by a step-indexed realizability argument; it states no theorem about contractive
maps, and citing one to it is the error this page previously made.

[01-schema-ir.md](01-schema-ir.md) records why the check's structural arms
compute nothing.

**Amadio & Cardelli, "Subtyping Recursive Types" (1993).** Subtyping between
recursive types is decided coinductively over a **trail** of address pairs:
assume the goal, unfold, and a pair already on the trail is a local success
(§1.5, with the algorithm at §4.4). **[LOAD-BEARING]** — the assumption stack in
`crates/valgebra-core/src/decision.rs` is that idea, with terms where the paper
has addresses. The arms are not the paper's: it decides an ordering over
`⊥/⊤/→/µ` and valgebra decides a lattice with no arrow.

**Frisch, Castagna & Benzaken, Definition 6.9.** Emptiness is proved
coinductively too: a *simulation* is "a self-justifying set, that is a
co-inductive proof of the fact that all its elements are equal to `0`".
**[GUIDING]** — valgebra's emptiness recurses on the structure and reads a cycle
back to a visiting reference as uninhabited, which is the inductive reading of
the same fact over finite values.

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

**The descriptor**, the representation the two Castagna papers above give and
Elixir's `Module.Types.Descr` implements, is **[IN PROGRESS]** in
`crates/valgebra-core/src/descr/`. Where the procedure in `decision.rs` reads a
schema's syntax and applies inclusion rules, the descriptor gives the *set* a
representation closed under union, intersection and complement, so a relation is
decided by emptiness of one combination rather than by whether a rule matched the
shape a caller wrote.

It is built beside the structural procedure and decides nothing a caller can
reach. Each kind's component starts *coarse* — every value of the kind, or none —
and each step replaces one with a representation that separates its values; the
type says which is which, so what the descriptor can and cannot see is read off
it. Two properties hold of it already that the structural IR does not have: the
form is canonical, so admitting the same values *is* being equal, and emptiness
is a decision rather than a conservative answer over the fragment it covers.

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

The two are cited for different things. The 1998 report states the approach;
neither "metamorphic relation" nor "necessary property" occurs in it. The
**criterion** is the 2018 review's: an MR relates multiple inputs and their
outputs, so a necessary property of a single input is not one — the review's
example is that `-1 ≤ sin(x) ≤ 1` is necessary and is not an MR. Cite the review
for the criterion.

The review also bounds what `docs/14-soundness.md` may rest on these suites: MRs
are *necessary* properties, so even a complete set of them is not a test oracle.
That page's trust base records it.

## The limit

**None of these is implemented as its paper describes it.** The decision
procedure is a sound, budget-bounded structural one that is exact on a published
fragment; it is not the interning automata engine. Where a page here says
otherwise, the page is wrong.

Tooling and toolchain facts are [11-references.md](11-references.md), not this
page.
