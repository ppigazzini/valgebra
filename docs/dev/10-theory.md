# The theory

What the design rests on, and which code each result touches. A citation here is
a claim that a specific line exists because of it, not a reading list.

Each entry is tagged **LOAD-BEARING** (code here rests on it), **GUIDING**
(shapes a decision without being an algorithm), or **PLANNED** (on the path, not
built). A planned reference is an intention; it never implies the thing is built.
Two more tags carry what the results *demand* rather than what they say:
**OBLIGATION**, a shape the implementation must have for a result to do any
work, and **DEVIATION**, a point where the tree departs from its source on
purpose and names the cost.

Every tagged paragraph is followed by a `HELD-BY:` line naming the tests that
fail when the sentence is false, or an `OWED:` line naming the tests it is owed
and why they do not exist yet. `tests/test_theory_ledger.py` holds both
directions: a tagged sentence with neither line fails, a name that resolves to
no test fails, an owed name that already resolves to one fails, and the count of
`OWED:` lines is recorded there and may only fall.

A paragraph restating the project's own argument -- the working notes the
distribution does not carry -- also gives a `SOURCE:` line: the numbered section
it comes from, and the sentence quoted. That line is what keeps the argument
from being a second ledger. It is held in both directions too: a citation into a
section that is gone, or a quotation the section no longer contains, fails; and
a result the argument tags that no paragraph here restates fails, so a claim
cannot be argued there and left with no tag, no `HELD-BY:` line and no test.
Both stand down where the notes are not in the clone, which is every clone but
the author's, so the reading is done where it can be done.

Some of this is decades old and stays, because a theorem does not expire.
Stone's representation theorem and Nakano's guardedness modality are exactly the
results the design leans on, not dated approximations of them.

## The frame: a schema denotes a set

**Scott and Strachey, denotational semantics.** A schema's meaning is the set of
Python values it admits; validation is membership in that set; subtyping is
subset inclusion. **[LOAD-BEARING: denotational-semantics]** — every variant's doc comment in
`crates/valgebra-core/src/ir.rs` states a set, and
[01-schema-ir.md](01-schema-ir.md) is that frame written out.

HELD-BY: test_walk_matches_denotation, test_node_admits_its_denotation

The consequence worth naming: because subtyping is inclusion and Python makes
`bool` a subclass of `int`, `bool` is a **subtype** of `int` rather than disjoint
from it. That is not a valgebra choice; it follows from the frame plus a fact
about Python.

## The algebra: a Boolean lattice of value sets

**Birkhoff, _Lattice Theory_.** Union, intersection and complement over value
sets form a Boolean algebra; the folds the constructors apply are its laws.
**[LOAD-BEARING: lattice-theory]** — `crates/valgebra-core/src/ir.rs`, where a schema is built
in the lattice normal form, and the property suites that check each claimed
equivalence against membership rather than asserting it.

SOURCE: §13.2 "**The laws hold by construction**"

HELD-BY: the_lattice_laws_hold_of_the_sets, test_union_commutativity, test_absorption

**Stone's representation theorem (1936).** Every Boolean algebra is isomorphic to
an algebra of sets. **[GUIDING]** — the licence for treating the scalar fragment
as a bitset over disjoint regions, which is what makes emptiness and subtyping
exact there ([02-decision.md](02-decision.md)).

## Semantic subtyping

**Frisch, Castagna & Benzaken, "Semantic Subtyping" (JACM 2008).** The model
valgebra *is*: value sets with union, intersection and negation, where subtyping
is `[[s ∧ ¬t]] = ∅`. **[LOAD-BEARING: semantic-subtyping]**, with a qualifier: `is_subtype_of` is
**not** that reduction applied uniformly. It is a structural procedure whose arms
decompose each pair by shape, and it calls `is_empty` at the three places where no
shape is available to recurse into: the two lattice bounds, and a complement on
the right. Everywhere else the arms decide directly.

SOURCE: §13.0 "**Subtyping is semantic, not syntactic.**"

HELD-BY: the_two_deciders_agree_under_an_oracle, test_the_two_deciders_are_measured_against_each_other

The distinction is not pedantry. A rule stated as the reduction and implemented
structurally has a hole wherever an arm is missing, and the reduction's name over
the top is what stops anyone looking for one.

On the expectation to hold of a rewriter, the paper offers an observation in
§2.2, not a theorem, and it is about a syntactic rule set:

> Forgetting any of these rules yields a type system that, although sound, does
> not match (i.e., it is not complete with respect to) the intuitive semantics of
> types.

A rewriter missing rules is sound and incomplete. Reading that shape onto this
tree's normal form is **valgebra's reasoning, not a result of the paper**, and it
is why [02-decision.md](02-decision.md) states soundness as the contract and
treats completeness as a measured, growing property rather than a promise.

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

It is guiding rather than load-bearing for *sequences*, and the gap is the
point: the general form is a regular expression over element schemas, and
deciding inclusion between two of those wants an automaton construction over
schemas that [15-decidability.md](../15-decidability.md) records as unbuilt.
What the IR carries is the *linear* fragment — a fixed prefix and an optional
repeated tail — which is every shape the schema language can spell. The closure
the paper gives is at the schema level there, through union, intersection and
complement over the sequence node, rather than inside the sequence body.

The same closure argument *is* load-bearing one kind over, for strings and
bytes. A `str` refinement's language is regular — a literal is one word, a
length bound is a symbol count, a pattern is itself — and the descriptor's word
component is a minimal deterministic automaton, canonically numbered. Union and
intersection are product constructions, complement is a flip of the accepting
states, and emptiness is a reachability walk; two spellings of one language have
one table, so `Regex("a")` is decided inside `Regex("ab?")` and equal to
`Literal["a"]`. The automaton for one pattern comes from `regex-automata`, which
the binding already depends on; the three set operations and the emptiness
decision are valgebra's, because that crate builds automata for searching and
offers no complement.

**The rule that splits a fixed-length sequence across a union** is the product
decomposition — Frisch, Castagna & Benzaken Lemma 6.5 for pairs, in the
backtrack-free form Castagna gives as `Φ`, and generalised to a fixed component
count the way Castagna & Duboc state the tuple rule for larger arities.
**[LOAD-BEARING: a-sequence-splits-across-a-union]** — `product_subtype` in `decision.rs`. It applies in subtyping
and nowhere else: emptiness does not decompose a product, so the same relation
asked as a meet with a complement is not decided.

HELD-BY: a_fixed_sequence_splits_across_the_branches_that_share_its_shape, test_a_product_splits_across_union_branches

## Records and maps

**Castagna, "Typing Records, Maps, and Structs" (ICFP 2023).** One node with
named fields plus default clauses subsumes the record, the homogeneous mapping,
the heterogeneous mapping and their combination. **[LOAD-BEARING: records-maps-and-structs]** —
`Schema::KeyedMap`, where a closed record is no default clause and `dict[K, V]`
is a single clause with no fields.

HELD-BY: the_lattice_laws_hold_of_the_dicts, a_meet_of_maps_holds_only_the_dicts_of_both, a_map_constrains_one_part_of_the_key_partition

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
recursion variable under a guard is productive. **[LOAD-BEARING: guarded-recursion]** for the
*discipline* — `occurs_unguarded` is that condition, and it is what makes the
induction above well founded.

SOURCE: §13.0 "Contractivity buys **well-definedness of the fixpoint**; regularity buys"; §13.4 "**Guarded recursion.** Nakano's modality"

The paper proves soundness of a modal type system
by a step-indexed realizability argument; it states no theorem about contractive
maps, and citing one to it is an error this page is written to avoid.

HELD-BY: contractivity_requires_a_structural_guard, test_non_contractive_body_is_rejected

[01-schema-ir.md](01-schema-ir.md) records why the check's structural arms
compute nothing.

**Amadio & Cardelli, "Subtyping Recursive Types" (1993).** Subtyping between
recursive types is decided coinductively over a **trail** of address pairs:
assume the goal, unfold, and a pair already on the trail is a local success
(§1.5, with the algorithm at §4.4). **[LOAD-BEARING: recursive-subtyping]** — the assumption stack in
`crates/valgebra-core/src/decision.rs` is that idea, with terms where the paper
has addresses.

SOURCE: §13.4 "for the trail idea; CORRECTED 2026-09-02"

The arms are not the paper's: it decides an ordering over
`⊥/⊤/→/µ` and valgebra decides a lattice with no arrow.

HELD-BY: decides_recursive_subtyping_coinductively,
test_recursive_subtyping_is_coinductive,
a_decision_leaves_the_trail_it_was_given,
a_decision_leaves_an_assumption_it_did_not_make,
an_assumption_is_read_as_the_pair_it_is,
a_proof_over_a_fixpoint_has_no_witness_against_it

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
type is an atom with its own rules, not the top of the lattice, because a static
checker asks a second question of it — *consistency* at every site where a value
crosses between typed and untyped code. **[GUIDING, and declined]** — a runtime
validator asks one question, membership, and to it the dynamic type is the top.
There is no `Dynamic` node: `typing.Any` builds `Schema::Anything` carrying a
[`Spelling`](../../crates/valgebra-core/src/ir.rs) that `render` reads and
nothing else does, so every law and every relation sees the top
([01-schema-ir.md](01-schema-ir.md#what-any-is)). The paper is kept here because
the reason it separates the two is the reason this project does not: it names
the question a validator has no site for.

## Refinement types

**Jhala & Vazou, "Refinement Types: A Tutorial" (2021).** A refinement is a base
set narrowed by predicates. **[GUIDING]** — `Schema::Refine` is the shape without
the SMT machinery: bounds are compared through the oracle, and a user predicate
is opaque.

## Decision procedures, for widening the decided fragment

**Gesbert, Genevès & Layaïda, "A Logical Approach to Deciding Semantic
Subtyping".** **[PLANNED]** — and planned means unstarted. The paper translates
the relation into a tree logic and decides satisfiability there, which is a
*replacement* for `decision.rs` rather than a widening of it, so nothing in the
tree is a partial version of it. What the citation buys today is the knowledge
that the relation is decidable in EXPTIME, and that is a statement about the
relation rather than a budget an implementation can be held to.

**A goal memo is not the missing piece, on every shape but one.**
Interning is in the tree — `crates/valgebra-core/src/ir/intern.rs` shares the
nodes of two schemas built alike — and it is the identity a table over goals
would need for its key, which is why it read as half of one. The goals a query
*repeats* are **counted** rather than argued: `decision::goal_tests` records
each goal where the recursion is asked it, which is past the caches that answer
a repeat without asking, and reports how many asks were of a pair already asked.

Over the older workloads' shapes, and over a record of thirty-two fields sharing
one inner schema, the count is **zero**: the trail absorbs recursion and the
per-rule caches absorb the shape where one goal is asked once per field
([02-decision.md](02-decision.md#the-budget-and-what-exhausting-it-means)).

Over the relation matrix it is **not**. A meet against a union repeats **four**
goals per query — twice in that corpus, eight over it — because the union
distribution asks the meet of each branch and the meet rule asks the class atom
against the same thing once per member, with no cache between the two rules.
A shape where one goal is reached by two rules with no cache between them is
what reopens this question, and the matrix carries two of them.
So the interning earns its place for what it already does — a cheap key for the
mutation sweep's diff, and a pointer-identity short-circuit on the trail's
comparisons — and a table over goals would buy a saving on that one shape,
which is a measurement away rather than a guess.

**The descriptor**, the representation the two Castagna papers above give and
Elixir's `Module.Types.Descr` implements, is **[LOAD-BEARING: the-descriptor]** in
`crates/valgebra-core/src/descr/`.

Where the procedure in `decision.rs` reads a
schema's syntax and applies inclusion rules, the descriptor gives the *set* a
representation closed under union, intersection and complement, so a relation is
decided by emptiness of one combination rather than by whether a rule matched the
shape a caller wrote.

That last clause is the claim, and it is a claim about the **verdict** rather
than about the values: two spellings of one set admit the same values whatever
either decides, so a law read against membership holds of a pair where one
spelling answers and the other declines. `a ∧ ¬(b ∨ c)` against `a ∧ ¬b ∧ ¬c`
is the pair, De Morgan is why they are one set, and the tests below read what a
caller is told rather than what a value is.

HELD-BY: the_lattice_laws_hold_of_the_descriptors, the_complement_laws_hold_of_the_descriptors, emptiness_agrees_with_the_values, the_verdict_is_stable_under_de_morgan, a_meet_with_a_negated_union_answers_as_the_spelled_out_meet, test_the_verdict_is_stable_under_de_morgan

It is built beside the structural procedure and is the second decider a caller
reaches: the rules answer first and this answers where they decline
([02-decision.md](02-decision.md)). Every kind now carries a representation that
separates its values — `Component::top` in
[descr/mod.rs](../../crates/valgebra-core/src/descr/mod.rs) is the one place that
says which, and it is the sentence to read rather than this one. `Coarse`, the
all-or-nothing component, survives only for `NoneType`, where a kind with one
value makes it exact. What the descriptor cannot hold is a *cycle*, which is why
a recursive schema is lowered by unfolding once and belongs to the rules; and
anything past the three bounds a build is held to. Two properties hold of it
that the structural IR does not have. Emptiness
over the fragment it covers is a decision rather than a conservative answer,
with a third verdict, `Unknown`, where an atom is not a set or where a build ran
out of its allowance before the question could be answered. **[LOAD-BEARING: the-second-decider]** —
the third verdict is what lets a bounded procedure stay *sound*: a negated form
has to be expanded before its emptiness can be read, and past the allowance
there is no union left to read. Either decision there would stand on nothing,
and the one a caller may act on is the proof of emptiness, so answering that one
wrongly is the expensive direction.

HELD-BY: a_negated_set_the_allowance_cannot_expand_declines,
a_negated_union_of_lines_the_allowance_cannot_expand_declines,
a_covering_question_the_allowance_cannot_settle_answers_neither_way

And equality is semantic: the scalar components and the word automata compare as
sets, so admitting the same values *is* being equal there — the integer set by
lifting two tables to the period they share, since its spelling is not canonical
and its *order* reads the spelling (`descr/integers.rs`). The set, record and
sequence components are not canonical either way, and equality on them is
emptiness of both differences — which is the same question, asked twice.

## Three representations with no paper here

The page's contract is that a citation names a line that exists because of it.
Three components of the descriptor ship and have **no row above**, and saying so
is what keeps the rest of the page readable as a complete account rather than a
selective one. None is an unbuilt idea -- each is code, with a test suite, in the
file named -- and each carries the construction the literature would attribute it
to, written here as a description rather than as a citation because the source
was not read first-hand:

| where | the construction | the work it belongs to |
|---|---|---|
| `crates/valgebra-core/src/descr/regular.rs` | a minimal deterministic automaton per word kind: refine a partition of the states until the signatures stop changing, coarsen the alphabet, renumber canonically | **Moore's** algorithm, named at the function. Not Hopcroft's `n log n`, which this does not implement |
| `crates/valgebra-core/src/descr/symbolic.rs` | the same refinement one alphabet up, where a letter is a descriptor, an edge carries a guard and the total transition is an *else* edge rather than a guard naming the universe | the minimisation of **symbolic automata** over a Boolean algebra of guards |
| `crates/valgebra-core/src/descr/integers.rs` | eventually periodic sets as an interval set per residue class, two periods meeting at their least common multiple | the ultimately periodic sets, which are the **one-variable fragment of Presburger arithmetic**. The representation is semilinear rather than automaton-based |

**Why they are described and not cited.** A citation here is held to the rule the
rest of the page is: the theorem was read, and matched to what the code does. For
these three it was matched to the code and not to the paper, so a row above would
be an attribution from memory -- which is the failure mode this page exists to
avoid, in a politer form than quoting a theorem number a secondary source
reported. The reading that can be done here has been: each description above is
against the source, and `regular.rs` names Moore at the line that refines the
partition.

The consequence worth stating: **the descriptor's components are the part of
this design least anchored in a read source.** They are held by their own test
modules and by the completeness ledger rather than by a theorem, which is a
weaker guarantee than the rest of the page offers and is the honest place to
start looking if one of them is wrong.

## Property-based testing

**Claessen & Hughes, QuickCheck (2000)**, and the modern shrinking work behind
hypothesis and proptest. **[LOAD-BEARING: property-testing]** — every algebra law is proved against
membership by a property suite rather than asserted
([08-testing.md](08-testing.md)).

SOURCE: §13.8 "**Property-based testing.**"

HELD-BY: test_de_morgan, the_lattice_laws_hold_of_the_sets, test_simplify_preserves_acceptance

**Chen, Cheung & Yiu, "Metamorphic Testing: A New Approach for Generating Next
Test Cases" (1998)**, and **Chen et al., "Metamorphic Testing: A Review of
Challenges and Opportunities" (2018).** Derive a test case from one that passed
and check a relation between the two outputs; neither run needs an oracle.
**[LOAD-BEARING: metamorphic-testing]** — the JSON path against the object path, and fast mode against
explain mode. Both relate a source run to a follow-up run, which is what makes
them metamorphic relations rather than invariants that happen to hold.

SOURCE: §13.8 "**Metamorphic testing.**"

HELD-BY: test_json_walk_matches_the_object_walk, test_is_valid_agrees_with_validate

The two are cited for different things. The 1998 report states the approach;
neither "metamorphic relation" nor "necessary property" occurs in it. The
**criterion** is the 2018 review's: an MR relates multiple inputs and their
outputs, so a necessary property of a single input is not one — the review's
example is that `-1 ≤ sin(x) ≤ 1` is necessary and is not an MR. Cite the review
for the criterion.

The review also bounds what `docs/14-soundness.md` may rest on these suites: MRs
are *necessary* properties, so even a complete set of them is not a test oracle.
That page's trust base records it.

## The results, row by row, and what holds each

The rows above cite a paper at the line it explains. The rows here are the
*results* the design rests on, stated as a sentence a test can be false against,
one per result -- the ledger the page above is an index to. A result that is
also a citation above is not repeated; what is here is every result, obligation
and deviation that has a test of its own or is owed one.

### Results

**Subtyping is inclusion, in both directions.** `t1 <= t2` holds exactly when
every value of `t1` is a value of `t2`, and `is_equivalent` is both directions.
A proof from either decider admits no counterexample the walk can find, and a
refutation from either stands on a value the walk refuses, over the structural
fragment and over a recursive pair alike. **[LOAD-BEARING: subtyping-is-inclusion]**

SOURCE: §13.0 "**Subtyping, defined.** UIN Definition 4"

HELD-BY: subtyping_is_sound_over_the_structural_fragment, a_refutation_is_a_value, a_proof_over_a_fixpoint_has_no_witness_against_it, test_no_proof_is_refuted_by_a_value, test_a_claimed_refutation_stands_on_a_value

**A cut reference proves and never refutes.** The descriptor decides `a <= b`
by lowering the difference, and a recursive reference is cut to a bound: the
top where the schema is used positively, the bottom under a complement. The
widened difference contains the real one, so its emptiness proves the inclusion
and its inhabitance proves nothing; a refutation comes only from a pair the
lowering did not widen. **[LOAD-BEARING: a-cut-reference-proves]**

HELD-BY: an_inhabited_difference_over_a_cut_reference_refutes_nothing, the_descriptor_and_the_procedure_never_contradict_each_other

**Kinds decompose emptiness.** Positives of mixed kind make a clause empty
outright, negatives of another kind are dropped, and each kind is then an
independent question -- which is what lets every kind carry its own
representation and the whole be a product over them. `Kind` is the one
partition both deciders read; `Region` is derived from it, never maintained
beside it. **[LOAD-BEARING: kinds-decompose-emptiness]**

HELD-BY: the_same_shape_under_two_kinds_does_not_meet, every_kind_has_exactly_one_component, every_kind_lands_in_the_partition_and_the_scalars_land_apart

**Two fixpoints, one procedure.** Emptiness reads a cycle back to a visiting
reference as uninhabited -- the least fixpoint, no finite value reaches it --
and subtyping reads a cycle back to an assumed goal as proved -- the greatest.
The two are one procedure asked in two directions, and the unfolding is sound
in both. **[LOAD-BEARING: two-fixpoints-one-procedure]**

SOURCE: §13.4 "**Mixed induction and coinduction.**"

HELD-BY: detects_uninhabited_recursive_schemas, test_the_unfolding_is_sound_in_both_directions, decides_recursive_subtyping_coinductively

**A clause is a region with its own default.** A keyed map is a quasi-K-step
function: named labels, and a default per key-type region rather than one
default for everything unnamed. A key falls in exactly one region, and a
mapping constrains one region and leaves the others as the openness says.
**[LOAD-BEARING: a-clause-is-a-region]**

SOURCE: §13.3 "**That is `Schema::KeyedMap` exactly**"

HELD-BY: every_key_falls_in_exactly_one_part, a_map_constrains_one_part_of_the_key_partition, test_heterogeneous_mapping_by_key_schema

**A sequence is a regular language, and the automaton over it is well formed.**
The symbolic automaton that decides sequence inclusion has, at every state,
edges that are pairwise disjoint and cover the letters; a complement flips the
accepting states and no value flips twice; the lattice laws hold of the
languages it denotes. **[LOAD-BEARING: a-sequence-is-a-language]**

SOURCE: §13.3 "**Sequences as regular expression types.**"

HELD-BY: the_lattice_laws_hold_of_the_sequences, the_complement_laws_hold_of_the_sequences, the_edges_leaving_a_state_cover_the_letters, an_accepting_state_behind_an_undecided_letter_is_unproved

**`Any` is the top, spelled.** `typing.Any` builds the top with a spelling that
`repr` reads and nothing else does, so every law and every relation sees one
set and the complement laws hold of it however it is spelled.
**[LOAD-BEARING: any-is-the-top-spelled]**

HELD-BY: the_complement_laws_hold_of_the_top_however_it_is_spelled, no_relation_can_tell_the_two_spellings_apart

**Each kind's representation is closed under the three operations, and its top
denotes what the table says.** Union, intersection and complement stay inside
each kind's representation -- interval sets with a residue class per step for
integers, intervals with the three special points held apart for floats, a
minimal automaton per word kind, a symbolic automaton for sequences, a union of
lines for sets, labelled fields with a default per key kind for dicts, a class
lattice for instances -- and the lattice laws hold of each against membership.
Each top is checked at the value that decides it: the ends of the integer
carrier, `nan`, the newline, the empty container. **[LOAD-BEARING: each-kind-is-closed]**

SOURCE: §14.5 "Each kind's representation should be closed under complement"

HELD-BY: the_lattice_laws_hold_of_the_integers, the_lattice_laws_hold_of_the_floats, the_lattice_laws_hold_of_the_languages, the_lattice_laws_hold_of_the_sequences, the_lattice_laws_hold_of_the_sets, the_lattice_laws_hold_of_the_dicts, the_lattice_laws_hold_of_the_objects, the_lattice_laws_hold_of_the_values, the_complement_laws_hold_of_the_values, an_emptiness_is_refused_by_every_boundary_value, the_universe_separates_a_pair_inside_a_shape

**A node built alike is one handle.** Interning shares the nodes of two schemas
built the same way, so the trail's comparisons short-circuit on pointer
identity and the per-rule caches see one handle per DAG member. What it buys is
that sharing and nothing about a relation. **[LOAD-BEARING: a-node-built-alike-is-one-handle]**

SOURCE: §13.6 "**Hash consing.** Hosoya-Vouillon-Pierce"

HELD-BY: a_node_built_twice_is_one_node_in_every_family, sharing_a_child_shares_the_parent_over_it, a_list_built_twice_is_one_list_holding_what_was_built

### Obligations

**The decision has three answers.** `Relation` on the subtyping side and
`Verdict` on the emptiness side each carry proof, refutation and neither, and
the two-valued surface is a projection of them: a proof is a proof, a refutation
stands on a value, and a decline is reported as itself rather than as either.
**[OBLIGATION: the-decision-has-three-answers]**

SOURCE: §14.1 "The decision's return type destroys the contract's only observable"

HELD-BY: test_the_three_answers_agree_with_the_two, an_inhabited_difference_over_a_cut_reference_refutes_nothing, a_negated_set_the_allowance_cannot_expand_declines

**The budget declines; it never refutes.** Exhausting the work budget answers
neither, on every path a subtyping query can take: through a product, through
the disjointness reading, and through the shared cell an equivalence query
carries across its two directions. **[OBLIGATION: the-budget-declines]**

HELD-BY: the_budget_declines_on_every_subtyping_path, an_exhausted_budget_refuses_to_spend, a_budgeted_equivalence_query_decides_the_same_or_declines

**The goals a query repeats are counted.** The decision not to memoise goals
rests on a number: zero repeats over the workload shapes and the thirty-two
field DAG, four per query where a meet meets a union, eight over the relation
matrix. The number is held by a test-side counter compiled for the core's own
tests, and a memo that changed it would fail the rows. **[OBLIGATION: repeated-goals-are-counted]**

SOURCE: §14.2 "Regularity is held and unspent"

HELD-BY: the_matrix_repeats_a_goal_only_where_a_meet_meets_a_union, a_record_of_thirty_two_fields_sharing_one_schema_repeats_no_goal, the_counter_sees_the_goals_a_query_asks

**A cache under coinduction is revertible or absent.** A memo added to the
coinductive procedure must be persistent, so a failed disjunct can roll it
back, or hold only results that rested on no open hypothesis; a memo that is
neither turns a backtracked assumption into a cached falsehood.
**[OBLIGATION: a-memo-is-revertible-or-absent]**

SOURCE: §14.3 "A cache under coinduction must be revertible"

HELD-BY: test_a_memo_without_a_revert_condition_fails_placement, test_the_detector_finds_a_memo_and_reads_the_sentence

**An exhaustible procedure is searched, not enumerated.** Where the decision is
structural rather than the emptiness reduction, a missing arm is a silent
conservative answer, so the conservative set is *searched* against the second
decider over a drawn universe and the recorded holes are held to a ledger.
**[OBLIGATION: an-exhaustible-procedure-is-searched]**

SOURCE: §13.1 "enumeration is the source of the exponent"; §14.4 "An exhaustible procedure needs the reduction"

HELD-BY: test_the_two_deciders_are_measured_against_each_other, test_decision_decides_true_relations

**`open` and `close` read the region no clause claims, and nothing else.**
Openness is the default of the key-type region the clauses leave over: `open`
frees that region and `close` refuses it, and neither touches a region a clause
claims. So a mapping opened keeps what a `str` key maps to and admits an `int`
key with any value, and one clause is read the same whether or not a field is
declared beside it. `close` is therefore a function of the set. `open` is not,
in one place and by construction: it descends into a union, and a branch
declaring no field frees every key on its own, so `{"a?": int}` and
`{} | {"a": int}` -- one set, two spellings -- open into two.
**[OBLIGATION: open-and-close-read-the-region]**

SOURCE: §14.6 "Operators belong on the representation built for them"

HELD-BY: opening_a_mapping_frees_the_region_no_clause_claims, with_records_open_keeps_the_region_a_mapping_claims, opening_a_record_that_claims_a_region_leaves_one_clause, test_a_clause_is_read_the_same_with_or_without_a_field_beside_it, test_closing_is_a_function_of_the_set, test_closing_is_a_function_of_the_set_however_it_is_spelled, test_opening_is_not_a_function_of_the_set, closing_is_a_function_of_the_set_however_it_is_spelled, opening_widens_closing_narrows_and_the_round_trip_is_at_most_closing, opening_under_a_complement_narrows_and_closing_widens, closing_an_opened_record_is_not_closing_the_record, test_closing_an_opened_schema_is_not_closing_it

**The IR is exactly as expressive as its producers.** Every variant the enum
has is one the frontend or the fuzzer builds, and a variant no producer reaches
is a surface the decision must be sound over for nothing.
**[OBLIGATION: the-ir-matches-its-producers]**

SOURCE: §14.7 "The IR should be exactly as expressive as its producers"

HELD-BY: test_every_variant_is_a_generator_a_representative_or_a_marker, test_no_column_names_a_variant_that_is_gone

**The definition imports nothing from the optimisation.** The partition and the
answer types sit below both deciders; no descriptor module names the structural
procedure, and the procedure still imports the lowering it asks when its rules
decline, so the one permitted edge is seen to be used. **[OBLIGATION: the-definition-imports-nothing]**

SOURCE: §14.8 "The definition should not import the optimisation"

HELD-BY: test_the_definition_imports_nothing_from_the_optimisation, test_the_term_imports_nothing_from_the_optimisation, test_the_optimisation_may_still_depend_on_the_definition

### Where this tree departs from its sources

Each is a decision, and none is unsoundness: a `True` from any relation remains
a proof. They are collected because a departure nobody writes down is one the
next reader rediscovers as a bug.

**The decision is structural, not the emptiness reduction applied uniformly.**
The cost is a hole wherever an arm is missing, and the boundary is
[15-decidability.md](../15-decidability.md); the holes are ledgered and the
deciders are measured against each other. **[DEVIATION: structural-rather-than-reduction]**

SOURCE: §13.1 "`is_subtype_of` is not that reduction applied uniformly"

HELD-BY: test_decision_decides_true_relations, test_the_two_deciders_are_measured_against_each_other

**Simplification stops at negation normal form, which is not canonical.**
`simplify(a) == simplify(b)` is not an equivalence test, `==` is the form and
not the set, and two spellings of one set can order differently -- which is why
the laws are asserted over Boolean combinations and the deciders' agreement over
a corpus that adds shapes. **[DEVIATION: the-normal-form-is-not-canonical]**

SOURCE: §13.1 "`simplify` produces negation normal form and no further"

HELD-BY: test_eq_is_the_normal_form_and_not_the_set, test_simplify_preserves_acceptance

**The IR's keyed map carries no negative component and is not canonicalised.**
The descriptor has one -- `MapAtom::wanted` -- so a negated record has a
representation there; what remains is the IR node, whose `dom` is the field
list as written rather than the semantic domain the paper's operators read.
**[DEVIATION: no-negative-clause-component]**

SOURCE: §13.3 "**the `S` component is absent**"

HELD-BY: two_spellings_of_one_keyed_map_are_one_term, opening_drops_a_field_the_record_already_said

**Clauses are unordered where the source orders them.** A key belongs when
*some* clause admits it and its value, in the walk and in subtyping alike, and
two clauses may claim one key. **[DEVIATION: clauses-are-unordered]**

HELD-BY: test_heterogeneous_mapping_by_key_schema, test_a_parsed_object_is_covered_by_whichever_clause_can_read_its_keys, test_named_field_takes_precedence_over_the_catch_all

**Clauses are quasi-K-step rather than quasi-constant.** A default per
key-type region rather than one default for the rest; nothing against the
literature, which introduces exactly this generalisation. **[DEVIATION: clauses-are-quasi-k-step]**

HELD-BY: every_key_falls_in_exactly_one_part, a_map_constrains_one_part_of_the_key_partition

**The trail holds terms, not addresses.** Assumption pairs are compared by
structural equality in a linear scan, short-circuiting on pointer identity for
interned subtrees; the longest trail any recursive shape in hand builds is two
pairs. A decision leaves the trail as it was given. **[DEVIATION: the-trail-holds-terms]**

HELD-BY: a_decision_leaves_the_trail_it_was_given, an_assumption_is_read_as_the_pair_it_is

**The assumption set is popped, not threaded.** Every relation proved on the
way is discarded, so a goal reached twice by different paths is decided twice,
and a decision leaves behind no assumption it did not make. **[DEVIATION: the-assumption-set-is-popped]**

HELD-BY: a_decision_leaves_an_assumption_it_did_not_make, a_decision_leaves_the_trail_it_was_given

**The dict key partition is by kind.** `1` and `True` are one key to a dict and
two labels to the partition, so an atom requiring both under distinct values
holds no dict, and the descriptor reads it that way. **[DEVIATION: the-key-partition-is-by-kind]**

HELD-BY: an_atom_requiring_a_key_and_its_boolean_holds_no_dict, a_labelled_key_witnesses_a_wanted_key

**The integer and float carriers are `i64` and `f64`.** Membership is exact --
the walk reads the Python object -- and a relation *declines* where the carrier
cannot spell the bound: a step past the period bound, a bound past the
carrier's end. A rounded carrier would be wrong in one direction, so it refuses.
**[DEVIATION: the-carriers-are-i64-and-f64]**

HELD-BY: a_step_past_the_period_bound_is_refused, test_every_edge_of_the_integer_carrier_has_a_row, test_the_relation_is_the_one_recorded, test_a_multiple_is_a_remainder_of_zero

**A class is described by what the frontend can read of it.** A class laying
down a layout the frontend cannot read is its `isinstance` test and its kind and
nothing else, so every relation about the structure its instances have is a
question no rule can answer. The record half carries no class for the same
reason the halves are split -- each is then a set the rules already know -- and
it is decided against another record, by width and by depth, and against the
lattice bounds. Against every other node it **declines** in both directions: a
subclass of any kind may carry an attribute, so without an oracle that
enumerates a kind's values there is no witness either way, and a refutation
would report a value nobody has. **[DEVIATION: a-class-is-what-can-be-read]**

HELD-BY: an_attribute_record_relates_to_every_other_node, a_class_met_with_its_attributes_has_a_value_when_its_fields_do

Two departures are closed and kept here so they are not rediscovered: the
gradual atom, which is the top spelled rather than a node beside `Int`
([above](#gradual-typing)); and the necessary-property suite, which is named for
what all six of its properties are and names the two that are also metamorphic
relations.

**The descriptor's lattice laws are checked one kind at a time.** The word
automata, the integer sets, the floats, the sequences, the sets, the dicts, the
value sets and the descriptors as a whole each have a property suite checked
against membership. The `Lines` component -- a union of lines with a negation
flag, which is how a kind holds its structure beside the objects it admits --
takes a `Whole` naming the kind at every operation, so a law over it is asked
within one kind rather than across the eleven. `Int` is the kind it is asked in,
because its structure is an exact set of values and membership there is decided;
a law over a kind whose component is coarse would hold on a set with two
elements. **[DEVIATION: the-lines-are-checked-per-kind]**

HELD-BY: the_lattice_laws_hold_of_the_lines, the_complement_laws_hold_of_the_lines, a_class_the_order_cannot_close_leaves_the_kind_unknown

## The limit

**No result here is implemented whole.** What is implemented is named per
entry, and the shape of every gap is the same: a construction the paper gives
over its own type language, carried out here over the fragment this IR can
spell. The word component *is* Hosoya-Vouillon-Pierce's automaton construction,
on the regular languages a `str` refinement denotes; the tree-automata inclusion
their paper decides is not built, and neither is the tree-logic engine above. The
structural procedure is sound and budget-bounded and exact on a published
fragment, not a decision procedure for the whole relation. And the interning that
shares a name with hash consing is a table of nodes, not a decision procedure.
Where a page here says otherwise, the page is wrong.

**And three components are attributed to no source at all**, which is a
different and larger gap than any of the above: see [Three representations with
no paper here](#three-representations-with-no-paper-here).

The departures are tagged rows in [the ledger above](#where-this-tree-departs-from-its-sources),
each with the test that holds its cost or the milestone owing one.

Tooling and toolchain facts are [11-references.md](11-references.md), not this
page.
