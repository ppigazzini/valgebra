# The theory

What the design rests on, and which code each result touches. A citation here is
a claim that a specific line exists because of it, not a reading list.

Each entry is tagged **LOAD-BEARING** (code here rests on it), **GUIDING**
(shapes a decision without being an algorithm), or **PLANNED** (on the path, not
built). A planned reference is an intention; it never implies the thing is built.
Three more tags carry what the results *demand* rather than what they say:
**OBLIGATION**, a shape the implementation must have for a result to do any
work; **DEVIATION**, a point where the tree departs from its source on purpose
and names the cost; and **NOT-REACHED**, a thing this tree does not do, given
only where a test fails on the day it starts doing it.

Every paragraph tagged LOAD-BEARING, OBLIGATION, DEVIATION or NOT-REACHED is
followed by a `HELD-BY:` line naming the tests that fail when the sentence is
false, or an `OWED:` line naming the tests it is owed and why they do not exist
yet; a GUIDING or PLANNED tag carries an id and no holding line. A claim the code
implements also gives an `IMPLEMENTED-BY:` line naming the items that implement
it, as [12-writing.md](12-writing.md#citing-a-paper) says.
`tests/test_theory_ledger.py` holds both directions: a claim with neither line
fails, a name that resolves to
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

## Works cited by label

A citation names a work by its label and the place in it: `[Cas23, (22)]` is
the equation tagged (22) in the work labeled `Cas23`, and `[FCB08, Lem. 6.5]`
is the lemma numbered 6.5 in `FCB08`. A label is the one the `alpha`
bibliography style builds from the authors and the year, which is how these
works label one another, and this table is the one place a work's identifier
is written. A number is the one the copy in the third column prints; a
preprint and the published version can number a result apart.
[12-writing.md](12-writing.md#citing-a-paper) says when to cite and where.

| Label | Work | Owns here | Read in | Identifier |
|---|---|---|---|---|
| `AC93` | Amadio & Cardelli, "Subtyping Recursive Types", *TOPLAS* 15(4), 1993 | the trail of recursive subtyping, the unfolding of a reference | the published article | [doi:10.1145/155183.155231](https://doi.org/10.1145/155183.155231) |
| `BBC+26` | Baazizi, Ben Ali, Colazzo, Ghelli, Klessinger, Sartiani & Scherzinger, "JSON Schema Inclusion through Refutational Normalization" | the rules of a structural inclusion: the uninhabited subject, the unions on either side, the arrays | the preprint, v4 of 10 August 2026 | [arXiv:2603.25306v4](https://arxiv.org/abs/2603.25306v4) |
| `Cas23` | Castagna, "Typing Records, Maps, and Structs", *PACMPL* 7(ICFP), Article 196, 2023 | records and maps: the atoms, their operations, the backtrack-free rules | the published article, with its appendices | [doi:10.1145/3607838](https://doi.org/10.1145/3607838) |
| `Cas24` | Castagna, "Programming with Union, Intersection, and Negation Types", *The French School of Programming*, Springer, 2024, pp. 309-378 | subtyping as inclusion, the membership predicate | the preprint, v4 of 27 March 2024 (arXiv:2111.03354v4) | [doi:10.1007/978-3-031-34518-0_12](https://doi.org/10.1007/978-3-031-34518-0_12) |
| `CD24` | Castagna & Duboc, "Guard Analysis and Safe Erasure Gradual Typing: a Type System for Elixir" | a descriptor per kind of atom, tuples as unions of atoms, the gradual pair | the preprint, v4 of 2 June 2026 | [arXiv:2408.14345v4](https://arxiv.org/abs/2408.14345v4) |
| `DV14` | D'Antoni & Veanes, "Minimization of Symbolic Automata", POPL 2014 | the word and sequence automata: guards, minimality, partition refinement | the published paper | [doi:10.1145/2535838.2535849](https://doi.org/10.1145/2535838.2535849) |
| `FCB08` | Frisch, Castagna & Benzaken, "Semantic Subtyping: Dealing Set-Theoretically with Function, Union, Intersection, and Negation Types", *JACM* 55(4), 2008 | the model: interpretation, normal forms, kinds, products, simulations | the published article | [doi:10.1145/1391289.1391293](https://doi.org/10.1145/1391289.1391293) |
| `HSHP20` | Habib, Shinnar, Hirzel & Pradel, "Type Safety with JSON Subschema" | a step entailed by a step it divides | the preprint, v2, of the ISSTA 2021 paper | [arXiv:1911.12651v2](https://arxiv.org/abs/1911.12651v2) |
| `HVP05` | Hosoya, Vouillon & Pierce, "Regular Expression Types for XML", *TOPLAS* 27(1), 2005 | sequences as regular expression types, the classical algorithm, hash consing | the published article | [doi:10.1145/1053468.1053470](https://doi.org/10.1145/1053468.1053470) |
| `JV20` | Jhala & Vazou, "Refinement Types: A Tutorial" | refined base types and their subtyping | the preprint, v1 | [arXiv:2010.07763v1](https://arxiv.org/abs/2010.07763v1) |
| `LN26` | Laurent & Nguyễn, "Implementing Set-Theoretic Types", *PACMPL* 10(OOPSLA2), Article 334, 2026 | a node met during its own traversal read as empty | the HAL copy, hal-05369012 | [doi:10.1145/3839466](https://doi.org/10.1145/3839466) |
| `Nak00` | Nakano, "A Modality for Recursion", LICS 2000 | guardedness: the variable of a recursive type under a guard | the published paper | [doi:10.1109/LICS.2000.855774](https://doi.org/10.1109/LICS.2000.855774) |

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

**Union, intersection and complement over value sets form a Boolean algebra**,
and the folds the constructors apply are its laws. A set-theoretic
interpretation takes a union to the union of the sets, a negation to the
complement and the empty type to the empty set [FCB08, Def. 4.1], so every law
of the algebra of sets -- Birkhoff's _Lattice Theory_ -- is a law of the types,
the complement of a union the meet of the complements [FCB08, Lem. 6.2].
**[LOAD-BEARING: lattice-theory]** — `crates/valgebra-core/src/ir.rs`, where a
schema is built in the lattice normal form, and the property suites that check
each claimed equivalence against membership rather than asserting it.

SOURCE: §13.2 "**The laws hold by construction**"; §2 "every combinator builds the normal"

IMPLEMENTED-BY: decision.rs::Schema::union_complement_is_empty, decision/readings.rs::Schema::region_set, decision/emptiness.rs::intersection_verdict, decision/emptiness.rs::meet_through_a_reference, descr/polar.rs::PolarUnion::intersect, descr/polar.rs::complement_all, descr/polar.rs::product, descr/lower.rs::descend, ir.rs::Schema::union_within, ir.rs::Schema::meet_within, ir.rs::Schema::complement, kind.rs::Region::union, kind.rs::Region::intersect, kind.rs::Region::complement, oracle.rs::has_complementary_pair_within, valgebra-py/src/check/walk.rs::check_intersection

HELD-BY: laws.rs::the_lattice_laws_hold_of_the_sets, test_union_commutativity, test_absorption, test_the_complement_laws_hold_of_every_drawn_schema, a_reference_the_definitions_resolve_is_read_at_its_fixpoint, a_callback_inside_a_resolved_body_refuses_the_reference

**Stone's representation theorem (1936).** Every Boolean algebra is isomorphic to
an algebra of sets. **[GUIDING: stone-duality]** — the license for treating the scalar fragment
as a bitset over disjoint regions, which is what makes emptiness and subtyping
exact there ([02-decision.md](02-decision.md)).

## Semantic subtyping

**Frisch, Castagna & Benzaken, "Semantic Subtyping" (JACM 2008).** The model
valgebra *is*: value sets with union, intersection and negation, where subtyping
is `[[s ∧ ¬t]] = ∅` [FCB08, §4.1]. **[LOAD-BEARING: semantic-subtyping]**, with a qualifier: `is_subtype_of` is
**not** that reduction applied uniformly. It is a structural procedure whose arms
decompose each pair by shape, and it calls emptiness where no shape is available
to recurse into -- the two lattice bounds, and a complement on the right -- and on
the subject of every refutation, which the witness guard (`witnessed` in
`decision.rs`) reads before it believes one. Everywhere else the arms decide
directly, and the set representation answers where they decline. The arms are
rules of the kind a rule-based inclusion of JSON Schema states: an uninhabited
subject is below everything [BBC+26, (uninhabited)], a union is below a schema
where each member is [BBC+26, (anyOf-l)], a schema is below a union where it is
below one member [BBC+26, (anyOf-r)], and a sequence is below a sequence
position by position [BBC+26, (array)].

SOURCE: §13.0 "**Subtyping is semantic, not syntactic.**"; §3 "Status: the set-theoretic model is"

IMPLEMENTED-BY: decision.rs::Schema::witnessed, decision.rs::Schema::bounds_the_pair, decision.rs::Schema::shares_no_value_with, decision.rs::Schema::below_a_union, decision.rs::Schema::subtype_by_shape, decision/products.rs::linear_subtype, kind.rs::Region::subset_of, verdict.rs::Relation::of_difference, decision.rs::Schema::meet_below

HELD-BY: the_two_deciders_agree_under_an_oracle, test_the_two_deciders_are_measured_against_each_other, subtyping_is_sound_over_the_structural_fragment

The distinction is not pedantry. A rule stated as the reduction and implemented
structurally has a hole wherever an arm is missing, and the reduction's name over
the top is what stops anyone looking for one.

On the expectation to hold of a rewriter, the paper offers an observation
[FCB08, §2.2], not a theorem, and it is about a syntactic rule set:

> Forgetting any of these rules yields a type system that, although sound, does
> not match (i.e., it is not complete with respect to) the intuitive semantics of
> types.

A rewriter missing rules is sound and incomplete. Reading that shape onto this
tree's normal form is **valgebra's reasoning, not a result of the paper**, and it
is why [02-decision.md](02-decision.md) states soundness as the contract and
treats completeness as a measured, growing property rather than a promise.

**Castagna, "Programming with Union, Intersection, and Negation Types"**
(arXiv:2111.03354, revised 2024). The modern synthesis. **[GUIDING: the-modern-synthesis]**

**Castagna, Duboc & Valim, "The Design Principles of the Elixir Type System"
(2023).** The same algebra in a production language, with the engineering
compromises stated. **[GUIDING: a-peer-implementation]** — the closest thing to a peer
implementation.

**Castagna & Duboc, "Guard Analysis and Safe Erasure Gradual Typing: a Type
System for Elixir" (2024).** The same system formalized, with its
implementation: a descriptor per kind of atom, a type of tuples kept as a union
of tuple atoms by distributing a negated tuple over its components "and likewise
for larger arities", and a gradual type as two static descriptors [CD24, §7.1].

## Sequences as a regular language

**Hosoya, Vouillon & Pierce, "Regular Expression Types for XML" (TOPLAS 2005).**
Regular expression types correspond exactly to tree automata, whose languages
are closed under union, intersection and complement, so a sequence type is a
first-class member of the algebra rather than an ad-hoc node.
**[GUIDING: sequences-as-a-regular-language]** — `Schema::Seq`, carrying a
`SeqShape`, is one node subsuming the homogeneous, fixed and prefix-plus-tail
forms for exactly this reason ([01-schema-ir.md](01-schema-ir.md)).

It is guiding rather than load-bearing for the *IR's* sequences, and the gap is
the point: the general form is a regular expression over element schemas, and
the IR carries only the *linear* fragment — a fixed prefix and an optional
repeated tail — which is every shape the schema language can spell; a repeated
heterogeneous group has no syntax, which
[15-decidability.md](../15-decidability.md) records. The closure the paper
gives is at the schema level there, through union, intersection and complement
over the sequence node, rather than inside the sequence body. The automaton
construction over element sets is built one representation over, in
`crates/valgebra-core/src/descr/symbolic.rs`, and decides what the fragment
spells, a sequence below a union of sequence shapes included (the row *A
sequence is a regular language* below).

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
offers no complement. It is asked for every match rather than for its default,
leftmost-first, which keeps only the match a search prefers: `a|ab` would lose
`ab`, and `a*?` every word but the empty one. A law in
`crates/valgebra-core/src/descr/regular/tests.rs` holds each pattern's table
to the `regex` matcher the walk runs.

**The rule that splits a fixed-length sequence across a union** is the product
decomposition — [FCB08, Lem. 6.5] for pairs, in the backtrack-free form Castagna
gives as `Φ` [Cas23, §2.1.4], and generalized to a fixed component count as the
proof of [Cas23, Lem. 4.7] generalizes the pair to every arity, over the
mappings of the branches to positions [Cas23, App. C.1], which the paper
explains for the arity-`n` case [Cas23, App. D].
**[LOAD-BEARING: a-sequence-splits-across-a-union]** — `product_subtype` in
`decision/products.rs`. It applies in subtyping and nowhere else: emptiness's
rules do not decompose a product, so the same relation asked as a meet with a
complement is left to the set representation, which decides it where the
difference fits the build bounds -- `tuple[int | str, int]` against its two
corners -- and declines past them. `Φ` characterizes the inclusion exactly, so
the rule refutes as well as proves -- a product with a value and no branch left
is outside the union -- where every member the rule sets aside holds none of the
subject's values; and a branch that shares no value with the narrowed product at
some position is dropped, which the same characterization makes exact, as a
branch of another container or arity drops out as a negative of another kind
does [FCB08, (5)]. The
emptiness of every component is asked at every level, where the paper asks the
product's once beside `Φ`.

SOURCE: §13.1 "**The product decomposition.**"

IMPLEMENTED-BY: decision/products.rs::fixed_components, decision/products.rs::product_subtype, decision/products.rs::seq_splits_across_union, decision/products.rs

HELD-BY: a_fixed_sequence_splits_across_the_branches_that_share_its_shape, the_product_rule_is_lemma_6_5, test_a_product_splits_across_union_branches, the_product_rule_is_lemma_6_5_at_three_positions, the_product_rule_decides_every_covering_split, the_product_rule_decides_scalar_pairs_as_the_values_do, a_product_is_below_its_corners_and_refuted_by_a_missing_one, a_member_the_product_rule_sets_aside_withholds_its_refutation

## Records and maps

**Castagna, "Typing Records, Maps, and Structs" (ICFP 2023).** One node with
named fields plus default clauses subsumes the record, the homogeneous mapping,
the heterogeneous mapping and their combination.
**[LOAD-BEARING: records-maps-and-structs]** — `Schema::KeyedMap`, where a
closed record is no default clause and `dict[K, V]` is a single clause with no
fields. A field is read in `T⊥`, an optional one holding `⊥` [Cas23, §3.2], so
an absent key is outside a required field [Cas23, §3.2.1]; two map atoms meet
pointwise [Cas23, (12)] and are empty where a required label is empty [Cas23, (11)],
two mappings always sharing the empty dict [Cas23, fn. 11]; and one map is below
another one key type at a time [Cas23, Lem. 4.7]. A dict is a member where every
label holds its value or its absence and every other key what its part's default
allows [Cas23, (21)]. A set of dicts is a union of the atoms of
[Cas23, Def. 4.1], the complement of one a union of atoms again by
[Cas23, (13)], read in `T⊥` [Cas23, §4.1], where a label given its default is
one the notation may name or leave [Cas23, Def. 2.2]. An attribute record needs
none of the `S` a map atom carries: the paper's record types need no more of it
than whether it is empty [Cas23, Rem. 4.3].

SOURCE: §3 "The model behind `Schema::KeyedMap`"; §13.3 "**The representation Theorem 4.2 produces.**"

IMPLEMENTED-BY: decision.rs::Schema::holds_a_value_shallowly, decision/emptiness.rs::intersection_verdict, decision/records.rs::keyed_map_meet_verdict, decision/records.rs::keyed_map_subtype, decision/records.rs::entered, decision/records.rs::Labels::of, descr/maps.rs, descr/maps.rs::Wanted, descr/maps.rs::Search::assign, descr/maps.rs::MapAtom::absorbed, descr/maps.rs::MapAtom::emptiness, descr/maps.rs::MapAtom::meet, descr/maps.rs::MapAtom::complement, descr/maps.rs::MapLattice::record, descr/records.rs, descr/records.rs::Atom::meet, descr/records.rs::Atom::emptiness, descr/records.rs::Atom::complement, descr/records.rs::Atom::tidy, descr/records.rs::RecordLattice::attribute, descr/values.rs::Field::top, descr/values.rs::Field::meet, descr/values.rs::Field::join, descr/values.rs::Field::complement, descr/values.rs::Field::emptiness, descr/lower.rs::map, ir.rs::Schema::record, ir.rs::canonical_fields, ir.rs::already_said, valgebra-py/src/check/walk/record.rs::keyed_map_scan, valgebra-py/src/check/walk/record.rs::keyed_map_asks_for_its_keys, valgebra-py/src/check/walk/record.rs::keyed_map_matches_json, descr/values.rs::Field

HELD-BY: the_lattice_laws_hold_of_the_dicts, the_lattice_laws_hold_of_the_dicts_over_drawn_dicts, a_meet_of_maps_holds_only_the_dicts_of_both, a_meet_holds_the_dicts_of_both_over_drawn_dicts, an_emptiness_holds_no_drawn_dict, a_map_constrains_one_part_of_the_key_partition, a_key_of_whole_kinds_opens_the_parts_it_holds

**A gradual type is two static types** in Castagna & Duboc's implementation
[CD24, §7.1]. Every gradual `t` is `t_down ∨ (? ∧ t_up)` and is represented by
**two** descriptors, with the operations run on each separately. valgebra
carries one, because it has one type to carry: `typing.Any` builds the top with
a spelling beside it, a spelling is what `repr` reads to print and the sharing
table (`ir/intern.rs`) reads to keep the two apart, no relation or law reads it,
and the two spellings lower to the same descriptor. The pair exists so that
every lattice rule can exclude a gradual case; having no such case does the same
work by having nothing to exclude. **[NOT-REACHED: one-descriptor-one-type]**

SOURCE: §13.5 "There is one descriptor because there is one type"

IMPLEMENTED-BY: ir.rs::Spelling

HELD-BY: the_two_spellings_of_the_top_lower_to_one_descriptor, the_two_spellings_of_the_top_are_not_one_node

**The backtrack-free record rule, [Cas23, (17)] and [Cas23, (22)].** A record
below a union of records is decided by `Φ`, which narrows the record by one
member a level and a label at a time, building the difference at that one label
alone, and never backtracks [Cas23, §2.1.4].
**[LOAD-BEARING: a-record-splits-across-a-union]** —
`keyed_map_splits_across_union` in `decision/records.rs`: the product rule's
shape one component over, the defaults compared part by part as [Cas23, (22)]
compares them, and the clauses placed in the key partition by the set
representation's own `key_cover`, so the two deciders agree on which key falls
where. Its emptiness reading, `keyed_map_minus_keyed_maps`, answers
`R ∧ ¬R₁ ∧ … ∧ ¬Rₙ` the same way, so the two routes to one inclusion do not
part. Two points are the tree's own. A member set aside on a default that
escapes it refutes only through an infinite part of the partition:
[Cas23, Thm. 4.5] proves (22) under the hypothesis [Cas23, §4.4] makes of every
key type, that it is infinite, and the paper says the hypothesis "is actually
used" -- the proof of [Cas23, Lem. 4.7] takes a key of the part no label names
[Cas23, App. C.1]. `None` has one key and `bool` two, where the paper would
write their keys as labels. And a label narrowed to a table of constants minus
constants is read as the set it leaves (`FiniteMeet` in `decision/literals.rs`),
without which a tag of literals would decline at its first narrowing. `Φ` is
read in the if-then form [Cas23, (19)] gives it, in `T⊥` [Cas23, §4.1], a label
narrowed as [Cas23, (13)] narrows it and an absence outside a required label
[Cas23, §3.2.1], with the paper's base case `false` [Cas23, (18)] and
[Cas23, (17)]'s `R ≤ 0` read as [Cas23, (11)] reads emptiness; its emptiness
reading is the meet of records with negated records the paper decides `Φ` on
[Cas23, §4.3], the positive records moved into one record R◦ by their pointwise
meet [Cas23, (12)] (`record_of_maps`), a label to the meet of what each record
reads under it and absent only where each allows it [Cas23, §4.1]; and a meet of
records is asked below a record or a union of them as R◦ (`keyed_meet_below`).
A member of another shape beside the records is left out, the meet being below
the records, so both readings then prove and do not refute. A meet of records
alone, one of them open, is R◦ for its emptiness too, an atom built from
schemas wanting no value: empty where a label it requires has none and
inhabited where each has one [Cas23, (11)], the dict of those labels being in
every record.

SOURCE: §13.3 "`keyed_map_splits_across_union` runs `Phi` beside the one-map rule"

IMPLEMENTED-BY: decision/records.rs::keyed_map_splits_across_union, decision/records.rs::keyed_map_minus_keyed_maps, decision/records.rs::split_across, decision/records.rs::finite, decision/records.rs::Defaults, decision/records.rs::defaults_against, decision/records.rs::Slot, decision/records.rs::Slot::below, decision/records.rs::Slot::without, decision/records.rs::Labels::names_with, decision/records.rs::Labels::base_case, decision/records.rs::phi, decision/records.rs::record_of_maps, decision/records.rs::keyed_meet_below, decision/records.rs::record_of_maps_verdict

HELD-BY: a_record_below_a_union_of_records_is_the_rules_answer, a_record_split_stops_where_the_decision_budget_does, a_record_splits_across_the_records_that_fix_its_tag, a_record_splits_on_whether_a_key_is_present, a_member_apart_at_an_optional_label_is_dropped, a_member_set_aside_on_a_finite_part_keeps_only_the_proof, a_member_whose_default_the_rules_cannot_relate_is_set_aside, a_clause_the_partition_does_not_hold_declines_the_split, a_record_minus_the_records_that_cover_it_is_empty, a_record_split_agrees_with_the_set_reading, keyed_maps_met_are_one_record, a_map_minus_a_map_above_it_is_empty_beside_a_predicate, a_meet_of_records_beside_another_member_proves_or_declines, a_meet_of_records_splits_as_one_record, test_two_open_records_met_are_one_record, a_meet_of_maps_is_inhabited_where_the_record_they_meet_in_is, a_meet_of_records_holds_the_dict_of_its_required_names, test_a_meet_of_open_records_holds_the_dict_of_its_required_fields

The paper's model is a *quasi K-step function* [Cas23, Def. 4.6], named labels
and every other key given the default of the key type it falls in, the key types
a partition of the keys. valgebra's clauses need not be a partition — a key
belongs when **some** clause admits it and its value, in the walk and in
subtyping alike, and two clauses may claim the same key. That is valgebra's own
model rather than the paper's, and the paper says why it is not theirs: it
forbids overlapping domains in one map [Cas23, §4.4], and an overlap read at
face value makes a key in two domains satisfy both, a conjunction where valgebra
reads a disjunction. Neither reads the clauses in order: the paper rejects the
leftmost-match reading because a semantic subtyping relation disregards the
order of the fields. The set each node denotes is well defined; what it does not
have is the paper's canonical form, which is why the domain is the field list as
written.

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

**An attribute is computed, not a constructor of the value.** An attribute
record reads its fields with `getattr`, and that order is not the substructure
one: `(5).real is 5`, so an immutable integer is its own attribute, and a
property can return a new object at every read. A value whose attributes lead
back to it, or on without end, is outside the finite values the induction
ranges over, as a list that contains itself is. The walk answers neither way
for such a value: a protocol fixpoint whose one member `real` is the fixpoint
reports `recursion_loop` for `5`, and `recursion_limit` for an object whose
`real` is always new, under the fixpoint and under its complement alike. The
decision proves that fixpoint empty, which is its answer on the finite values.

Two theorems sit nearby and neither is the justification. **Banach's fixed-point
theorem** — "a contractile map over a complete metric space has a unique
fixpoint", quoted at [AC93, §3.3.2] — is the metric account of
recursive types, and its unique fixpoint lives among *infinite* trees, which the
walk never admits. **Tarski's fixpoint theorem (1955)** gives a monotone map on a
complete lattice a least fixpoint, and applies to the complement-free fragment,
where the inductive set is that least fixpoint. Both are
**[GUIDING: the-fixpoint-theorems-nearby]**.

**Nakano, "A Modality for Recursion" (2000).** The guardedness modality: a
recursion variable under a guard is productive, which the paper states as a type
expression proper in its variable [Nak00, Def. 8], [Nak00, Def. 9]. Castagna
asks the same of a recursive type as contractivity, every infinite branch
passing through infinitely many constructors [Cas23, Def. 2.1].
**[LOAD-BEARING: guarded-recursion]** for the *discipline* —
`occurs_unguarded_under` is that condition, with the structural constructors for
the guard, and it is what makes the induction above well founded.

SOURCE: §13.0 "Contractivity buys **well-definedness of the fixpoint**; regularity buys"; §13.4 "**Guarded recursion.** Nakano's modality"; §4 "for the *discipline* (`occurs_unguarded`)"; §4 "recursion is sound when the variable sits under a guard"; §4 "`Schema::occurs_unguarded` (the contractivity check)"

The paper proves soundness of a modal type system
by a step-indexed realizability argument; it states no theorem about contractive
maps, and citing one to it is an error this page is written to avoid.

IMPLEMENTED-BY: ir.rs::Guarded::join, ir.rs::Schema::guards_children, ir.rs::Schema::occurs_unguarded_under, ir.rs::Schema::union_within, oracle.rs::has_complementary_pair_within, valgebra-py/src/build.rs::tie_alias, ir.rs::Schema::meet_within, oracle.rs::denotes_a_set_within

HELD-BY: contractivity_requires_a_structural_guard, test_non_contractive_body_is_rejected, structural_constructors_absorb_the_guard, a_reference_under_only_combinators_is_unguarded, a_reference_the_definitions_resolve_is_read_at_its_fixpoint

[01-schema-ir.md](01-schema-ir.md) records why the check's structural arms
compute nothing.

**A reference denotes the definition it names.** `Ref(i)` and `defs[i]` are one
set, so a fixpoint and its own unfolding are equivalent in both directions and a
caller who writes the body out by hand has written the same schema. That is the
**equirecursive** reading, which makes unfolding transparent, a type and its
unfolding each below the other [AC93, §3.2], and it is a choice: the
isorecursive one makes the two distinct types with a coercion between them, and
every relation is then about which side of that coercion a value sits on.
Pierce, *TAPL* ch. 20--21 is where the two are set beside each other; the
definitions table is the same shape pydantic-core's `definition-ref` has.
**[LOAD-BEARING: a-reference-denotes-its-definition]**

SOURCE: §4 "an **equirecursive** reading"

IMPLEMENTED-BY: ir/transform.rs::Schema::unfolded, valgebra-py/src/check/walk.rs::check_ref

HELD-BY: test_a_reference_denotes_the_definition_it_names, test_the_unfolding_is_sound_in_both_directions, decides_recursive_subtyping_coinductively, a_reference_and_its_definition_are_one_set_over_drawn_definitions, test_a_fixpoint_and_its_unfolding_are_one_set

**A descent past the bound is an error, never a crash.** The walk unfolds to
`MAX_RECURSION_DEPTH` and reports `recursion_limit` past it, which converts any
unbounded descent -- a contractivity case the build check might miss, an exotic
mutual cycle -- into an answer rather than a native stack overflow. The JSON
path has a third bound, the parser's own nesting limit, which sits between the
unfolding bound and the descent bound: a document past the unfolding bound is
refused by the walk, and one past the parser's is refused as unreadable before
the walk sees it, so the descent bound is unreachable through a document;
[10-limits.md](../10-limits.md) names the three regions. The conventional default, 128,
follows `serde_json`. **[LOAD-BEARING: the-depth-bound-reports-itself]**

SOURCE: §5 "converts any unbounded descent"

HELD-BY: test_every_entry_point_reports_the_bound_it_reaches, test_a_value_past_the_depth_bound_reports_the_bound, test_the_parser_and_the_walk_bound_a_document_in_that_order

**A value that contains itself is caught by identity.** The guard on the
descent path is a stack of `(id(value), ref)` pairs, entered and left level by
level, so a value reachable from
itself is refused with `recursion_loop` and a value that merely *repeats* is
not: a list holding two equal sublists is an ordinary finite value, and a guard
reading equality would refuse it. That is the coinductive reading observing a
regular tree, and the inductive one -- the finite values membership accepts --
is its dual. **[LOAD-BEARING: a-cycle-is-caught-by-identity]**

HELD-BY: test_a_cycle_is_caught_by_identity_rather_than_by_equality, test_a_value_containing_itself_reports_the_cycle

**Amadio & Cardelli, "Subtyping Recursive Types" (1993).** Subtyping between
recursive types is decided coinductively over a **trail** of address pairs:
assume the goal, unfold, and a pair already on the trail is a local success
([AC93, §1.5], with the algorithm at [AC93, §4.4]).
**[LOAD-BEARING: recursive-subtyping]** — the assumption stack in
`crates/valgebra-core/src/decision.rs` is that idea, with terms where the paper
has addresses.

SOURCE: §13.4 "for the trail idea, not arm for arm"

The arms are not the paper's: it decides an ordering over
`⊥/⊤/→/µ` and valgebra decides a lattice with no arrow.

IMPLEMENTED-BY: decision.rs::Schema::subtype_by_rules, decision.rs::Schema::left_reduces_below, decision.rs::Schema::below_a_reference, decision.rs::Cx::assuming

HELD-BY: decides_recursive_subtyping_coinductively,
test_recursive_subtyping_is_coinductive,
a_decision_leaves_the_trail_it_was_given,
a_decision_leaves_an_assumption_it_did_not_make,
an_assumption_is_read_as_the_pair_it_is,
a_proof_over_a_fixpoint_has_no_witness_against_it

**There is no arrow type, and the bootstrapping apparatus is not needed.**
Semantic subtyping's central difficulty is circular: types cannot be
interpreted as sets of values directly, because knowing which values inhabit a
type needs the type system, which needs subtyping, which needs the
interpretation -- and *arrows* are why. A `Callable[...]` here is an
`isinstance` check whose argument and return types are not enforced, and the
membership walk is structural and never consults `is_subtype_of`, so the naive
interpretation is available and the whole apparatus -- the universal model, the
extensional interpretation, the theorems that close the circle -- is machinery
this project does not need and does not have. The consequence is a license
rather than a lack. **[NOT-REACHED: no-arrow-type]**

SOURCE: §13.0 "**The circularity, and why valgebra escapes it.**"

HELD-BY: test_the_membership_walk_never_consults_subtyping, test_a_callable_annotation_erases_the_arrow_it_writes, test_every_variant_is_a_generator_a_representative_or_a_marker, test_no_column_names_a_variant_that_is_gone

**Emptiness is proved coinductively too** in Frisch, Castagna & Benzaken's
model: a *simulation* [FCB08, Def. 6.9] is "a self-justifying set, that is a
co-inductive proof of the fact that all its elements are equal to `0`".
**[GUIDING: emptiness-is-a-simulation]** — valgebra's emptiness recurses on the
structure and reads a cycle back to a visiting reference, or to a meet held open
on its path (`OPEN_MEETS` in `decision/emptiness.rs`), as uninhabited, which is
the inductive reading of the same fact over finite values.

Coinduction also governs the *value* side: a value that contains itself is
refused by identity rather than followed ([04-walk.md](04-walk.md)).

## Gradual typing

**Siek & Taha, "Gradual Typing for Functional Languages" (2006).** The dynamic
type is an atom with its own rules, not the top of the lattice, because a static
checker asks a second question of it — *consistency* at every site where a value
crosses between typed and untyped code. **[GUIDING: the-dynamic-type-declined]**
— a runtime
validator asks one question, membership, and to it the dynamic type is the top.
There is no `Dynamic` node: `typing.Any` builds `Schema::Anything` carrying a
[`Spelling`](../../crates/valgebra-core/src/ir.rs) that `render` reads to print
and the sharing table reads to keep the two apart, and no law or relation
reads, so every law and every relation sees the top
([01-schema-ir.md](01-schema-ir.md#what-any-is)). The paper is kept here because
the reason it separates the two is the reason this project does not: it names
the question a validator has no site for.

## Refinement types

**Jhala & Vazou, "Refinement Types: A Tutorial" (2021).** A refinement is a base
set narrowed by predicates. **[GUIDING: refinement-types]** — `Schema::Refine` is the shape without
the SMT machinery: bounds are compared through the oracle, and a user predicate
is opaque.

**A refinement is its base narrowed, and a predicate is opaque.**
`Refine { base, constraints }` denotes `{x ∈ ⟦base⟧ | every constraint holds}`,
a refined base type [JV20, §3.2], so it is below its base whatever the
constraints say -- a subset does not depend on *which* subset -- and the base is
not below it, because that would need the predicate to hold of every value. A
predicate is compared by the object carrying it: two refinements over one
predicate are one set, and two over separately written predicates that agree
everywhere are undecided, since deciding them means deciding whether two Python
callables agree. Two refinements compare as refined base types do under
Sub-Base, by an entailment of their predicates [JV20, Fig. 3.4], here without a
solver: a bound or a step is entailed through the oracle one constraint at a
time, and a predicate only by itself. A refinement whose predicate is not well
sorted over its base is not a type at all [JV20, Fig. 3.3], and the frontend
refuses a constraint no value of the base can answer.
**[LOAD-BEARING: a-refinement-narrows-its-base]**

SOURCE: §10 "the runtime check; the static/SMT side is"

IMPLEMENTED-BY: decision.rs::refinement_subtype, decision/constraints.rs::constraint_entailed, ir.rs::Schema::refine, valgebra-py/src/check/walk/scalar.rs::check_refine, valgebra-py/src/build/refine.rs::check_constraint_fits, descr/lower.rs::refine

HELD-BY: test_a_refinement_is_its_base_narrowed_and_a_predicate_is_opaque, test_predicate_marker, test_every_cell_narrows_its_base_or_is_refused

## Decision procedures, for widening the decided fragment

**Gesbert, Genevès & Layaïda, "A Logical Approach to Deciding Semantic
Subtyping".** **[PLANNED: the-exact-decision]** — and planned means unstarted. The paper translates
the relation into a tree logic and decides satisfiability there, which is a
*replacement* for `decision.rs` rather than a widening of it, so nothing in the
tree is a partial version of it. What the citation buys is the knowledge
that the relation is decidable in EXPTIME, and that is a statement about the
relation rather than a budget an implementation can be held to.

**A goal memo is not the missing piece, on every shape but one.**
Interning is in the tree — `crates/valgebra-core/src/ir/intern.rs` shares the
nodes of two schemas built alike — and it is the identity a table over goals
would need for its key, which is why it reads as half of one. The goals a query
*repeats* are **counted** rather than argued: `decision::goal_tests` records
each goal where the recursion is asked it, which is past the caches that answer
a repeat without asking, and reports how many asks were of a pair already asked.

Over the shapes of the `--decision`, `--decision-refute` and `--decision-repeat`
workloads of `scripts/perf_gate.py`, and over a record of thirty-two fields
sharing one inner schema, the count is **zero**: the trail absorbs recursion and
the per-rule caches absorb the shape where one goal is asked once per field
([02-decision.md](02-decision.md#the-budget-and-what-exhausting-it-means)).

Over the relation matrix it is **not**. A meet against a union repeats **four**
goals per query — twice in that corpus, eight over it — because the union
distribution asks the meet of each branch and the meet rule asks the class atom
against the same thing once per member, with no cache between the two rules.
A shape where one goal is reached by two rules with no cache between them is
what reopens this question, and the matrix carries two of them.
So the interning earns its place for what it already does — one handle per
structurally equal subtree, so a lowering and the per-rule caches do not repeat
themselves over a copy, and a pointer-identity short-circuit on the trail's
comparisons — and a table over goals would buy a saving on that one shape,
which is a measurement away rather than a guess.

**The descriptor**, the representation the two Castagna papers above give and
Elixir's `Module.Types.Descr` implements -- a record mapping each kind of atom
to its representation, the operations run kind by kind and a negation saturating
the other kinds [CD24, §7.1], so that a relation is the emptiness of one
combination [FCB08, §4.1] -- is **[LOAD-BEARING: the-descriptor]** in
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

IMPLEMENTED-BY: descr/mod.rs, descr/mod.rs::Component::top, kind.rs, descr/mod.rs::Descr::complement, descr/mod.rs::Descr::zip, descr/values.rs::Values::covers

HELD-BY: the_lattice_laws_hold_of_the_descriptors, the_complement_laws_hold_of_the_descriptors, emptiness_agrees_with_the_values, laws.rs::the_verdict_is_stable_under_de_morgan, descr/maps/tests.rs::the_verdict_is_stable_under_de_morgan, the_descriptor_and_the_procedure_never_contradict_each_other, a_meet_with_a_negated_union_answers_as_the_spelled_out_meet, test_the_verdict_is_stable_under_de_morgan, descr/maps/tests.rs::a_complement_is_expanded_only_where_it_is_one_product, descr/records/tests.rs::a_complement_is_expanded_only_where_it_is_one_product, descr/sets/tests.rs::a_complement_is_expanded_only_where_it_is_one_product, a_meet_against_a_negated_side_removes_one_line_at_a_time

It is built beside the structural procedure and is the second decider a caller
reaches: the rules answer first and this answers where they decline
([02-decision.md](02-decision.md)). Every kind carries a representation that
separates its values — `Component::top` in
[descr/mod.rs](../../crates/valgebra-core/src/descr/mod.rs) is the one place
that says which, and it is the sentence to read rather than this one. `Coarse`,
the all-or-nothing component, stands in two places: `NoneType`, where a kind
with one value makes it exact, and the kindless slot (`KINDLESS`), whose values
carry no structure it separates, so the classes and attributes do the work. What
the descriptor cannot hold is a *cycle*, which is why a recursive schema is
lowered by unfolding once and belongs to the rules; and anything past the three
bounds a build is held to. Two properties hold of it that the structural IR does
not have. Emptiness over the fragment it covers is a decision rather than a
conservative answer, with a third verdict, `Unknown`, where an atom is not a set
or where a build ran out of its allowance before the question could be answered.
**[LOAD-BEARING: the-second-decider]** — the third verdict is what lets a
bounded procedure stay *sound*: a negated form has to be expanded before its
emptiness can be read, and past the allowance there is no union left to read.
Either decision there would stand on nothing, and the one a caller may act on is
the proof of emptiness, so answering that one wrongly is the expensive
direction.

HELD-BY: a_starved_verdict_is_the_third_answer_or_the_decided_one, a_negated_set_the_allowance_cannot_expand_declines,
a_negated_union_of_lines_the_allowance_cannot_expand_declines,
a_covering_question_the_allowance_cannot_settle_answers_neither_way

And equality is semantic: the scalar components and the word automata compare as
sets, so admitting the same values *is* being equal there — the integer set by
lifting two tables to the period they share, since its spelling is not canonical
and its *order* reads the spelling (`descr/integers.rs`). The set, record and
sequence components are not canonical either way, and equality on them is
emptiness of both differences — which is the same question, asked twice.

## Automata over an algebra of letters

**D'Antoni & Veanes, "Minimization of Symbolic Automata" (POPL 2014).** A
symbolic automaton reads its letters through the predicates of an effective
Boolean algebra rather than one at a time from a finite alphabet [DV14, §2]; a
minimal one is unique up to renaming its states and the equivalence of its
predicates [DV14, Thm. 1], and Hopcroft's partition refinement lifts to it
[DV14, Fig. 3]. Both automata of the descriptor are this construction. The word
automaton's letters are bytes in classes no state separates, which stand in for
the minterms the figure computes [DV14, Prop. 1]. The sequence automaton's
letters are descriptors: an edge carries a guard, and what the guards leave over
is an *else* edge, since a descriptor naming its own universe would have to
build itself. **[LOAD-BEARING: one-language-one-table]** — the operations are
the classical construction [HVP05, §4.1], where a complement flips the accepting
states of a complete table, an intersection is a product, and emptiness is a
walk to an accepting state. Minimizing merges the states of the equivalence of
[DV14, Def. 3], which gives a minimal table [DV14, Thm. 2], and a cut reads the
transitions backwards by class, as the figure reads them by minterm [DV14, §4].
`equivalent_states` in `descr/regular.rs` is the figure at its lines 5 to 15 --
from the initial partition [DV14, Fig. 3, l. 2], the predecessors of a splitter
marked [DV14, Fig. 3, l. 8] and each block they cut split [DV14, Fig. 3, l. 9-10],
a queued block replaced by both parts [DV14, Fig. 3, l. 14] and an unqueued one
by its smaller part [DV14, Fig. 3, l. 15], the tie-break included -- with four
differences that move no partition: line 3 queues both initial blocks, which the
paper asks only of a partial automaton [DV14, §8] and which costs a complete one
a splitter; line 4 reads the byte classes for the minterms; the loop is bounded by
a pop per state; and the start block is renumbered first. The canonical
numbering then removes the renaming, so two spellings of one word language have
one table.

IMPLEMENTED-BY: descr/automaton.rs::reaches, descr/automaton.rs::canonical_order, descr/regular.rs, descr/regular.rs::Dfa::is_empty, descr/regular.rs::Dfa::word, descr/regular.rs::Dfa::complement, descr/regular.rs::Dfa::product, descr/regular.rs::Dfa::minimal, descr/regular.rs::Dfa::merge_equivalent, descr/regular.rs::Dfa::equivalent_states, descr/regular.rs::Dfa::coarsen, descr/regular.rs::Dfa::renumber, descr/regular.rs::refine_alphabets, descr/regular.rs::Predecessors::of, descr/regular.rs::Predecessors::leading_to, descr/regular.rs::Partition::by_acceptance, descr/regular.rs::Partition::mark, descr/regular.rs::Partition::cut_marked, descr/regular.rs::RegularSet::compile, descr/regular.rs::RegularSet::intersect, descr/symbolic.rs, descr/symbolic.rs::Guard, descr/symbolic.rs::SymbolicDfa, descr/symbolic.rs::SymbolicDfa::chain, descr/symbolic.rs::SymbolicDfa::emptiness, descr/symbolic.rs::SymbolicDfa::accepts_under, descr/symbolic.rs::SymbolicDfa::complement, descr/symbolic.rs::SymbolicDfa::intersect, descr/symbolic.rs::SymbolicDfa::product, descr/symbolic.rs::SymbolicDfa::merge_targets, descr/symbolic.rs::SymbolicDfa::sorted, descr/symbolic.rs::partition, descr/symbolic.rs::rest_of, descr/regular.rs::Dfa::wtf8, descr/regular.rs::RegularSet

HELD-BY: the_halves_cut_where_the_rounds_do, a_cut_moves_the_smaller_part, a_cut_of_a_queued_block_queues_both_parts, two_spellings_of_one_language_are_one_set, being_equal_is_holding_the_same_language, a_complement_is_already_minimal

**The sequence table names a language up to its guards.** Moore's algorithm
over a symbolic alphabet is a fixpoint over pairs of states, which tells two
apart where moves into a pair already told apart carry guards whose meet is
satisfiable [DV14, §3], and reaches the minimal table [DV14, Thm. 2], one per
language up to renaming and the equivalence of predicates [DV14, Thm. 1]. The
sequence automaton refines in Moore's rounds instead, toward the equivalence of
[DV14, Def. 3], telling a round's states apart by a signature of the blocks
their joined guards lead to -- a device of this tree's, not the paper's -- and
compares the joined guards by equality, which is finer than their sets, so two
states no sequence tells apart may stay apart. Its table names a language only
up to that, and the cost is that sequence equality is the emptiness of both
differences rather than a comparison of two tables. It does not read its
transitions backwards, which over guards needs the minterms, the minimization
of [DV14, Fig. 4], or the pair fixpoint above, which asks only that a meet of two
guards be satisfiable. **[DEVIATION: the-sequence-table-names-a-language-up-to-its-guards]**

IMPLEMENTED-BY: descr/automaton.rs, descr/automaton.rs::refine, descr/symbolic.rs::SymbolicDfa::minimal, descr/symbolic.rs::SymbolicDfa::merge_equivalent, descr/symbolic.rs::SymbolicDfa::signature, descr/symbolic.rs::SymbolicDfa::renumber

HELD-BY: a_complement_of_a_sequence_language_is_already_minimal, the_rounds_from_the_distances_end_where_the_rounds_from_one_block_do

## Two representations with no paper here

The page's contract is that a citation names a line that exists because of it.
Two components of the descriptor ship and have **no row above**, and saying so is
what keeps the rest of the page readable as a complete account rather than a
selective one. Neither is an unbuilt idea -- each is code, with a test suite, in
the file named -- and each carries the construction the literature would
attribute it to, written here as a description rather than as a citation because
no work on the shelf states it:

| where | the construction | the work it belongs to |
|---|---|---|
| `crates/valgebra-core/src/descr/integers.rs` | eventually periodic sets as an interval set per residue class, two periods meeting at their least common multiple | the ultimately periodic sets, which are the **one-variable fragment of Presburger arithmetic**. The representation is semilinear rather than automaton-based |
| `crates/valgebra-core/src/descr/floats.rs` | closed intervals of floats with `nan` held apart in a bit of its own, an open end written as the neighboring float and `-0.0` read as `0.0` | the order of IEEE 754 doubles, which are discrete: a union of closed intervals is the integers' representation without the residues |

**Why they are described and not cited.** A citation here is held to the rule the
rest of the page is: the theorem was read, and matched to what the code does. For
these two no work on the shelf states the construction, so a row above would be
an attribution from memory -- which is the failure mode this page exists to
avoid, in a politer form than quoting a theorem number a secondary source
reported. The reading that can be done here has been: each description above is
against the code.

The consequence worth stating: **the integer and float components are the part
of this design least anchored in a read source.** They are held by their own
test modules and by the completeness ledger rather than by a theorem, which is a
weaker guarantee than the rest of the page offers and is the honest place to
start looking if one of them is wrong.

## Property-based testing

**Claessen & Hughes, QuickCheck (2000)**, and the modern shrinking work behind
hypothesis and proptest. **[LOAD-BEARING: property-testing]** — every algebra law is proved against
membership by a property suite rather than asserted
([08-testing.md](08-testing.md)).

SOURCE: §13.8 "**Property-based testing.**"; §9 "Hypothesis adds integrated test-case shrinking"

HELD-BY: test_de_morgan, laws.rs::the_lattice_laws_hold_of_the_sets, laws.rs::construction_preserves_membership_over_values,
a_set_line_is_read_as_python_holds_its_members

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
are *necessary* properties, so even a complete set of them "might still not be
equivalent to a test oracle". That page's trust base records it.

## The results, row by row, and what holds each

The rows above cite a paper at the line it explains. The rows here are the
*results* the design rests on, stated as a sentence a test can be false against,
one per result -- the ledger the page above is an index to. A result that is
also a citation above is not repeated; what is here is every result, obligation
and deviation that has a test of its own or is owed one.

### Results

**Subtyping is inclusion, in both directions.** `t1 <= t2` holds exactly when
every value of `t1` is a value of `t2`, and `is_equivalent` is both directions
[Cas24, Def. 4].
A proof from either decider admits no counterexample the walk can find, and a
refutation from either stands on a value the walk refuses, over the structural
fragment, over maps keyed by every hashable kind, and over a recursive pair alike. **[LOAD-BEARING: subtyping-is-inclusion]**

SOURCE: §13.0 "**Subtyping, defined.** UIN Definition 4"

IMPLEMENTED-BY: decision.rs::Schema::equivalence_relation

HELD-BY: subtyping_is_sound_over_the_structural_fragment, a_refutation_is_a_value, a_map_relation_stands_on_a_dict, a_proof_over_a_fixpoint_has_no_witness_against_it, test_no_proof_is_refuted_by_a_value, test_a_claimed_refutation_stands_on_a_value

**A cut reference proves and never refutes.** The descriptor decides `a <= b` by
lowering the difference, `a ∧ ¬b = ∅` [FCB08, §4.1], and a recursive reference
is cut to a bound: the top
where the schema is used positively, the bottom under a complement, which is the
finite approximation of a recursive type at the depth of the cut [AC93, §3.4].
The widened difference contains the real one, so its emptiness proves the
inclusion and its inhabitance proves nothing; a refutation comes only from a
pair the lowering did not widen. A conjunct at the top that the lowering
refuses -- a meet's member, a refinement's constraint -- is left out the same
way, a meet being below each of its members [FCB08, Def. 4.1], so emptiness is
monotone in a meet's members. **[LOAD-BEARING: a-cut-reference-proves]**

SOURCE: §13.1 "**The polarity cut is one-directional.**"

IMPLEMENTED-BY: decision.rs::Schema::descriptor_contained_in, ir/transform.rs::Polarity::flipped, ir/transform.rs::Schema::unfolded, descr/lower.rs::lower_widened

HELD-BY: an_inhabited_difference_over_a_cut_reference_refutes_nothing, a_cut_reference_widens_the_subject_and_narrows_the_other, a_meet_with_a_recursive_schema_is_decided_by_one_unfolding, a_meet_is_proved_empty_beside_a_conjunct_the_lowering_refuses, a_reference_above_is_unfolded_for_the_sets, a_meet_stays_empty_beside_a_member_no_lowering_reads, test_a_meet_stays_empty_beside_a_member_no_lowering_reads, test_a_meet_of_patterns_is_empty_beside_a_predicate, a_member_past_the_depth_is_left_out_whatever_it_holds, a_member_whose_key_is_its_deepest_node_is_kept, a_thinned_reading_that_can_only_be_inhabited_is_built_where_it_answers

**Kinds decompose emptiness.** Positives of two kinds that share no value make a
clause empty outright [FCB08, (5)] -- every pair but `bool` and `int`, a
deviation below -- negatives of another kind are dropped, and each kind is then
an independent question [FCB08, (4)] -- which is what lets every kind carry its
own representation and the whole be a product over them. `Kind` is the one
partition both deciders read; `Region` is derived from it, never maintained
beside it, and a scalar atom's region is the set of constants its basic type
denotes [FCB08, §4.2]. A meet shares no value with what one of its members
shares none with, being below each [FCB08, Def. 4.1], so a meet a union holds
as a branch is read by its kinds too. **[LOAD-BEARING: kinds-decompose-emptiness]**

SOURCE: §13.1 "positives of mixed kind make a clause empty outright"

IMPLEMENTED-BY: decision.rs::has_disjoint_pair, decision/readings.rs::Schema::disjoint_with, decision/readings.rs::Schema::meet_disjoint_with, decision/readings.rs::Schema::atom_region, decision/emptiness.rs::intersection_verdict, decision/products.rs::seq_splits_across_union, descr/mod.rs::Descr::within, descr/mod.rs::Descr::emptiness

HELD-BY: the_same_shape_under_two_kinds_does_not_meet, every_kind_has_exactly_one_component, every_kind_lands_in_the_partition_and_the_scalars_land_apart, a_kind_is_an_independent_question, a_meet_is_disjoint_from_what_one_of_its_members_is, a_pair_read_disjoint_shares_no_value, test_a_meet_with_a_union_of_guarded_patterns_is_empty

**A set type is a powerset, and a line of them is empty where one powerset
covers it.** `set[T]` and `frozenset[T]` denote the finite sets whose members
all lie in `T`. A meet of powersets is the powerset of the meet, and a meet of
powersets is inside a union of powersets exactly where the meet of their
elements is inside one of them, stated for the finite subsets a Python set is
[FCB08, Lem. 6.30], which with one of each reads `set[A] <= set[B]` exactly
where `A <= B`.
The set representation reads the lemma a line at a time: a line
`P(T) ∧ ⋀ ¬P(Sⱼ)` is empty where some `Sⱼ` covers `T`. The converse is where
Python departs: `==` joins values the element keeps apart, `1` and `True` among
them, so the members a witness collects may be one, and a line is read
inhabited only where one member escapes every `Sⱼ` or the letter keeps one
member per `Sⱼ` apart (`Members::one_member_each` in `descr/sets.rs`).
**[LOAD-BEARING: a-set-is-a-powerset]**

IMPLEMENTED-BY: decision.rs::Schema::subtype_by_shape, descr/sets.rs, descr/sets.rs::Line::emptiness, descr/sets.rs::Line::one_member_each, descr/sets.rs::Line::tidy, descr/sets.rs::<Line as Summand>::meet

HELD-BY: the_structural_inclusion_rules_decide_without_the_descriptor, a_meet_of_powersets_is_the_powerset_of_the_meet, a_line_is_empty_when_a_subtraction_covers_it, the_powerset_of_nothing_holds_the_empty_set, emptiness_agrees_with_the_sets, a_set_line_is_read_as_python_holds_its_members

**Two fixpoints, one procedure.** Emptiness reads a cycle back to a visiting
reference, or to a meet held open on its path, as uninhabited, as a node met
during its own traversal is [LN26, §5.1] -- the least fixpoint, no finite value
reaches it, the coinductive proof of emptiness [FCB08, Def. 6.9] read over
finite values --
and subtyping reads a cycle back to an assumed goal as proved -- the greatest.
The two are one procedure asked in two directions, and the unfolding is sound
in both. A goal the emptiness recursion asks while it resolves a reference reads
that reference at positive positions only, so its answer is about the schema
with those occurrences emptied, whose emptiness is the definition's: a value of
least height needs no value of the definition at a positive position below it. A
meet met again on its path is read as empty only where a constructor stands
between, so a union a meet distributes leaves no copy of itself in a branch.
**[LOAD-BEARING: two-fixpoints-one-procedure]**

SOURCE: §13.4 "**Mixed induction and coinduction.**"

IMPLEMENTED-BY: decision/emptiness.rs, decision/emptiness.rs::Schema::empty_and_region_at, decision/emptiness.rs::meet_through_a_reference, decision/emptiness.rs::OpenMeet::open

HELD-BY: detects_uninhabited_recursive_schemas, test_the_unfolding_is_sound_in_both_directions, decides_recursive_subtyping_coinductively, the_fixpoint_laws_hold_over_drawn_definitions, a_reference_being_resolved_refutes_nothing_that_holds, a_record_minus_records_inside_a_definition_asks_its_goals_there, a_proof_over_a_fixpoint_has_no_witness_against_it, an_emptiness_is_refused_by_every_boundary_value, a_meet_with_a_reference_is_its_definition_met_with_the_rest, a_meet_with_a_reference_answers_as_its_values_do, a_meet_holding_a_union_twice_is_proved_empty_only_where_it_is, a_union_met_twice_is_distributed_once

**A clause is a region with its own default.** A keyed map is a quasi-K-step
function [Cas23, Def. 4.6], named labels and a default per key-type region
rather than one default for everything unnamed. A key falls in exactly one
region, the key types being parts the paper asks to be disjoint [Cas23, §4.4]
and a union of key types filling each part it covers [Cas23, fn. 15], and a
mapping constrains one region and leaves the others as the openness says. One
map is below another one region at a time [Cas23, Lem. 4.7]. A key schema
naming part of a kind is refused, overlapping domains being a theory of their
own [Cas23, §4.5], and the two finite parts, `None` and `bool`, are the ones the
proof of that lemma has no key to spare in [Cas23, App. C.1].
**[LOAD-BEARING: a-clause-is-a-region]**

SOURCE: §13.3 "**That is `Schema::KeyedMap` exactly**"

IMPLEMENTED-BY: decision/records.rs, decision/records.rs::clause_by_key, decision/records.rs::by_constants, decision/records.rs::parts_of, decision/records.rs::Labels::slot, descr/maps.rs, descr/maps.rs::key_slot, descr/maps.rs::finite_keys, descr/maps.rs::MapAtom::field, descr/maps.rs::MapAtom::emptiness, descr/maps.rs::MapAtom::meet, descr/lower.rs::map, descr/lower.rs::key_cover, valgebra-py/src/check/walk/record.rs::keyed_map_scan, valgebra-py/src/build.rs::checked_key, decision/records.rs::spelled_by, descr/maps.rs::MapAtom::absorbed

HELD-BY: every_key_falls_in_exactly_one_part, a_map_constrains_one_part_of_the_key_partition, test_heterogeneous_mapping_by_key_schema, a_clause_constrains_its_region_alone_over_drawn_dicts

**A sequence is a regular language, and the automaton over it is well formed.**
Regular expression types correspond to tree automata [HVP05, §1], and the
symbolic automaton that decides sequence inclusion [DV14, §2] has, at every
state, edges that are pairwise disjoint and cover the letters; a product meets
the guards of two edges and drops a meet that is empty [DV14, §1]; a complement
flips the accepting states and no value flips twice; the lattice laws hold of
the languages it denotes. **[LOAD-BEARING: a-sequence-is-a-language]**

SOURCE: §13.3 "**Sequences as regular expression types.**"

IMPLEMENTED-BY: descr/symbolic.rs, descr/symbolic.rs::SymbolicDfa::product

HELD-BY: the_lattice_laws_hold_of_the_sequences, the_complement_laws_hold_of_the_sequences, the_edges_leaving_a_state_cover_the_letters, an_accepting_state_behind_an_undecided_letter_is_unproved

**`Any` is the top, spelled.** `typing.Any` builds the top with a spelling that
`repr` reads to print and the sharing table reads to keep the two apart, and no
law or relation reads, so every law and every relation sees one set and the complement laws hold of it however it is spelled.
**[LOAD-BEARING: any-is-the-top-spelled]**

HELD-BY: the_complement_laws_hold_of_the_top_however_it_is_spelled, no_relation_can_tell_the_two_spellings_apart

**Each kind's representation is closed under the three operations, and its top
denotes what the table says.** Union, intersection and complement stay inside
each kind's representation -- interval sets with a residue class per step for
integers, closed intervals of floats with `nan` held apart in a bit of its own, a
minimal automaton per word kind, a symbolic automaton for sequences, a union of
lines for sets, labeled fields with a default per key kind for dicts, a class
lattice for instances -- and the lattice laws hold of each against membership.
Each top is checked at the value that decides it: the ends of the integer
carrier, `nan`, the newline, the empty container. **[LOAD-BEARING: each-kind-is-closed]**

SOURCE: §14.5 "Each kind's representation should be closed under complement"

HELD-BY: descr/integers/tests.rs::the_lattice_laws_hold_of_the_integers, the_lattice_laws_hold_of_the_floats, the_lattice_laws_hold_of_the_languages, the_lattice_laws_hold_of_the_sequences, descr/sets/tests.rs::the_lattice_laws_hold_of_the_sets, the_lattice_laws_hold_of_the_dicts, the_lattice_laws_hold_of_the_objects, the_lattice_laws_hold_of_the_values, the_complement_laws_hold_of_the_values, an_emptiness_is_refused_by_every_boundary_value, the_universe_separates_a_pair_inside_a_shape, the_universe_separates_a_pair_inside_a_position, the_universe_separates_a_pair_at_the_tail_of_a_position, the_universe_separates_a_pair_spelled_through_a_wrapper, test_each_kinds_top_is_decided_at_the_value_that_decides_it, test_a_hashable_subclass_of_an_unhashable_kind_is_a_set_member, test_a_class_whose_metaclass_runs_isinstance_declines_every_relation

**A node built alike is one handle.** Interning shares the nodes of two schemas
built the same way, as Hosoya, Vouillon and Pierce share theirs so that a
reflexivity check is a comparison of handles [HVP05, §5.1], so the trail's
comparisons short-circuit on pointer identity and the per-rule caches see one
handle per DAG member. What it buys is that sharing and nothing about a
relation. The table is lossy where theirs keeps every node: a collision evicts
and an entry is weak, so two schemas built alike are one handle while the first
lives and holds its slot. **[LOAD-BEARING: a-node-built-alike-is-one-handle]**

SOURCE: §13.6 "**Hash consing.** Hosoya-Vouillon-Pierce"

IMPLEMENTED-BY: ir/intern.rs::node, ir/intern.rs::members, ir.rs::Schema::union_within

HELD-BY: a_node_built_twice_is_one_node_in_every_family, sharing_a_child_shares_the_parent_over_it, a_list_built_twice_is_one_list_holding_what_was_built

### Obligations

**The decision has three answers.** `Relation` on the subtyping side and
`Verdict` on the emptiness side each carry proof, refutation and neither, and
the two-valued surface is a projection of them: a proof is a proof, a refutation
stands on a value, and a decline is reported as itself rather than as either.
**[OBLIGATION: the-decision-has-three-answers]**

SOURCE: §14.1 "The decision's return type destroys the contract's only observable"

HELD-BY: test_the_three_answers_agree_with_the_two, the_relation_folds_are_kleenes_strong_connectives, the_verdict_folds_are_the_relation_folds_read_as_inhabitation, an_inhabited_difference_over_a_cut_reference_refutes_nothing, a_negated_set_the_allowance_cannot_expand_declines, a_union_narrowed_to_one_branch_refutes_where_that_branch_does, test_every_predicate_is_the_proof_answer_of_a_relation

**The budget declines; it never refutes.** Exhausting the work budget answers
neither, on every path a subtyping query can take: through a product, through
the disjointness reading, and through the shared cell an equivalence query
carries across its two directions. Reaching the depth bound, which holds the
recursion inside the stack, answers neither the same way, in the goals and in
emptiness. **[OBLIGATION: the-budget-declines]**

HELD-BY: the_budget_declines_on_every_subtyping_path, an_exhausted_budget_refuses_to_spend, a_budgeted_equivalence_query_decides_the_same_or_declines, the_budget_declines_on_every_drawn_pair, the_depth_bound_declines_and_gives_its_levels_back

**The goals a query repeats are counted.** The decision not to memoize goals
rests on a number: zero repeats over the workload shapes and the thirty-two
field DAG, four per query where a meet meets a union, eight over the relation
matrix. The number is held by a test-side counter compiled for the core's own
tests, and a memo that changed it would fail the rows. **[OBLIGATION: repeated-goals-are-counted]**

SOURCE: §14.2 "Regularity is held and unspent"

HELD-BY: the_matrix_repeats_a_goal_only_where_a_meet_meets_a_union, a_record_of_thirty_two_fields_sharing_one_schema_repeats_no_goal, the_counter_sees_the_goals_a_query_asks

**Regularity bounds the goals a query asks.** A schema is a finite tree with
back edges into a finite table of definitions, so the subterms reachable by
unfolding it are finitely many, which is the regularity a type is asked
[Cas23, Def. 2.1]. A rule asks a pair of them or a pair built out
of them -- a meet of subterms and their complements from the product rule, a
union of some of a union's members from the union rule, a bound from a fold --
so every goal is drawn from a finite closure, which stands where the normal
forms over the atoms a type mentions stand [FCB08, §6.9] and is finite by the
argument of [LN26, §5.1] -- a narrowing builds a record atom no subterm is, and
finitely many of them -- and the distinct goals a
query
reaches are at most its square. With the trail cutting a pair that comes back,
that is the termination argument the saturating algorithm makes, and the work
budget bounds cost rather than standing in for it. The closure is held on the
workload shapes, on the recursive pairs the trail is for, on the rules that
build, and on drawn queries; the subterm pairs, a tighter count, are held on the
same shapes as a measurement. Emptiness is held to the same finiteness: a
recursive meet of maps is read under its cycle rather than unfolded again.
**[OBLIGATION: regularity-bounds-the-goals]**

SOURCE: §14.2 "Regularity bounds the number of distinct"

IMPLEMENTED-BY: decision/emptiness.rs::meet_through_a_reference, valgebra-py/src/build.rs::refuse_a_growing_argument

HELD-BY: every_goal_a_query_asks_is_drawn_from_the_closure_of_its_subterms, the_goals_a_query_asks_are_pairs_of_the_subterms, the_counter_sees_the_goals_a_query_asks, the_goals_a_drawn_query_asks_are_pairs_of_the_subterms, a_recursive_meet_of_maps_is_decided_within_its_unfoldings

**A cache under coinduction is revertible or absent.** A memo added to the
coinductive procedure must be persistent, so a failed disjunct can roll it
back, or hold only results that rested on no open hypothesis; a memo that is
neither turns a backtracked assumption into a cached falsehood.

What holds it is a **placement** test, and what that test holds is a sentence
rather than a property: a map keyed by a goal pair inside the procedure fails
until the condition is written beside it, and nothing mechanical reads whether
the sentence is true. The obligation is vacuous while there is no memo, which
is exactly when a reader is least likely to remember it -- so the guard's job
is to make writing the sentence a deliberate act a reviewer sees rather than
an omission nobody does. **[OBLIGATION: a-memo-is-revertible-or-absent]**

SOURCE: §14.3 "A cache under coinduction must be revertible"

HELD-BY: test_a_memo_without_a_revert_condition_fails_placement, test_the_detector_finds_a_memo_and_reads_the_sentence

**An exhaustible procedure is searched, not enumerated.** Where the decision is
structural rather than the emptiness reduction, a missing arm is a silent
conservative answer, so the conservative set is *searched* against the second
decider over a drawn universe and the recorded holes are held to a ledger.
**[OBLIGATION: an-exhaustible-procedure-is-searched]**

SOURCE: §13.1 "enumeration is the source of the exponent"; §14.4 "An exhaustible procedure needs the reduction"

HELD-BY: test_the_two_deciders_are_measured_against_each_other, test_decision_decides_true_relations, the_two_deciders_agree_under_an_oracle

**`open` and `close` read the region no clause claims, and nothing else.**
Openness is the default of the key-type region the clauses leave over: `open`
frees that region and `close` refuses it, and neither touches a region a clause
claims, open and closed being the two kinds of record type [Cas23, §3.2] and
each region carrying a default of its own [Cas23, Def. 4.6]. A clause short of
every value claims its keys; the clauses admitting every value are one clause
over the union of their keys, the wildcard of [Cas23, §4.4] written as the union
[Cas23, fn. 15] reads one as, sugar for the key types no clause names
[Cas23, fn. 13], of which the part over keys another clause claims stays and the
rest is the region. So a mapping opened keeps what a `str` key maps to and
admits an `int` key with any value, one clause is read the same whether or not a
field is declared beside it, and the top split over `str` and `complement(str)`
closes as the top does. Every record is rewritten, the one a clause maps a key
to included. What the laws hold of the sets is narrower than "a function of the
set": closing commutes with absorption, applying either operator twice gives the
set applying it once gave, though not always the same term, and a name saying
what its record says of every key it does not name moves no set either operator
gives. **[OBLIGATION: open-and-close-read-the-region]**

SOURCE: §14.6 "Operators belong on the representation built for them"

IMPLEMENTED-BY: ir/transform.rs::Schema::with_records_open, ir.rs::unclaimed_region, ir.rs::claimed_beside_the_region, ir.rs::clauses_over, ir.rs::tops_merged, ir.rs::merge_by_value

HELD-BY: test_opening_a_mapping_frees_the_keys_no_clause_claims, test_a_clause_is_read_the_same_with_or_without_a_field_beside_it, test_one_term_closes_one_way_whichever_reading_wrote_it, test_closing_commutes_with_absorption, closing_commutes_with_absorption, opening_or_closing_twice_gives_the_set_once_did, with_records_open_reaches_the_record_a_clause_maps_to, test_opening_reaches_the_record_a_mapping_maps_to, opening_widens_closing_narrows_and_the_round_trip_is_at_most_closing, closing_an_opened_mapping_takes_out_the_region_opening_freed, test_closing_an_opened_mapping_takes_out_only_the_region_opening_freed, opening_under_a_complement_narrows_and_closing_widens, closing_an_opened_record_is_not_closing_the_record, test_closing_an_opened_schema_is_not_closing_it, a_name_the_record_already_says_moves_no_set_either_operator_gives

**`open` and `close` are term rewrites, and the cost is a spelling.** They read
the clauses a term *writes* rather than the set it denotes, so two terms the
algebra calls equal need not open alike, or close alike. `{"a?": int}` and
`{} | {"a": int}` admit the same dicts, and the branch declaring no field frees
every key on its own when opened. A meet of two open records admits what the
open record of both fields admits, and closing the meet closes each record on
its own, to two records no dict belongs to both of. A `close` on sets would
need the record normal form with its positive atoms merged, which is a change
to the constructors for an operator no failing use case asks for, and
closedness is per declaration wherever else it is spelled. The argument licenses
exactly this -- a spelling-sensitive operator is permitted provided it is
declared -- and what it forbids is one clause read two ways according to an
unrelated field, which is incoherence rather than syntax-sensitivity.

The rows below are about the **term**: what the rewrite writes, which is the
thing a reader of `ir/transform.rs` checks against. The set the operators give
is the obligation above, and its rows ask values.
**[DEVIATION: open-and-close-are-term-rewrites]**

SOURCE: §14.6 "Operators belong on the representation built for them"; §13.4b "So a spelling-sensitive `open`/`close` is *permitted*"

HELD-BY: opening_a_mapping_frees_the_region_no_clause_claims, with_records_open_keeps_the_region_a_mapping_claims, opening_a_record_that_claims_a_region_leaves_one_clause, the_two_readings_of_one_term_close_to_one_set, test_opening_is_not_a_function_of_the_set, test_a_meet_of_open_records_closes_apart_from_its_record

**The IR is exactly as expressive as its producers.** Every variant the enum
has is one the frontend or the fuzzer builds, and a variant no producer reaches
is a surface the decision must be sound over for nothing.
**[OBLIGATION: the-ir-matches-its-producers]**

SOURCE: §14.7 "The IR should be exactly as expressive as its producers"

HELD-BY: every_variant_is_built_by_a_producer_and_the_placeholder_by_none, test_every_variant_is_a_generator_a_representative_or_a_marker, test_no_column_names_a_variant_that_is_gone, test_the_node_table_covers_every_ir_variant

**The definition imports nothing from the optimization.** The partition and the
answer types sit below both deciders; no descriptor module names the structural
procedure, and the procedure still imports the lowering it asks when its rules
decline, so the one permitted edge is seen to be used. **[OBLIGATION: the-definition-imports-nothing]**

SOURCE: §14.8 "The definition should not import"

HELD-BY: test_the_definition_imports_nothing_from_the_optimization, test_the_term_imports_nothing_from_the_optimization, test_the_optimization_may_still_depend_on_the_definition

### Where this tree departs from its sources

Each is a decision, and none is unsoundness: a `True` from any relation remains
a proof. They are collected because a departure nobody writes down is one the
next reader rediscovers as a bug.

**The decision is structural, not the emptiness reduction applied uniformly.**
The cost is a hole wherever an arm is missing, and the boundary is
[15-decidability.md](../15-decidability.md); what a missing arm costs is a
decline the descriptor answers, so every relation the completeness ledger
records as decided by neither is one past a build or decision bound, each a
strict row, and the two deciders are measured against each other over a
corpus. The rules distribute a union a meet holds only where it holds a
reference or a meet is already being unfolded (`meet_through_a_reference`),
where every source decides on the normal form, which distributes every one:
a meet the rules decline beside a union of guards is answered by the
descriptor within its depth, and declines past it.
**[DEVIATION: structural-rather-than-reduction]**

SOURCE: §13.1 "`is_subtype_of` is not that reduction applied uniformly"; §13.1 "That blow-up is why a DNF-based decision needs"

HELD-BY: test_decision_decides_true_relations, test_the_two_deciders_are_measured_against_each_other, the_structural_inclusion_rules_decide_without_the_descriptor

**The term is built in a normal form short of negation normal form, and it is
not canonical.** Construction applies associativity, commutativity,
idempotence and the identity laws, not absorption, and pushes no complement
inward; no pass reduces a term further. So `==` is the form and not the
set, and two spellings of one set can build to two terms -- `A | (A & B)` and
`A` -- which is why the laws are asserted over the values and the deciders'
agreement over a corpus that adds shapes.
**[DEVIATION: the-normal-form-is-not-canonical]**

SOURCE: §13.1 "construction builds the lattice normal form and no pass reduces it further"; §3 "A rewriter missing rules is sound and incomplete"

HELD-BY: test_eq_is_the_normal_form_and_not_the_set, laws.rs::construction_preserves_membership_over_values

**The IR's keyed map carries no negative component and is not canonicalized.**
The descriptor has one -- `MapAtom::wanted` -- so a negated record has a
representation there; what remains is the IR node, whose `dom` is the field list
as written rather than the semantic domain the paper's operators read. And each
wanted constraint names the labels it excludes where it is made, which is not
the paper's shape: [Cas23, Thm. 4.2] fixes one label set for a whole normal form
before its atoms combine, where an atom here carries its own labels and a meet
unions them. **[DEVIATION: no-negative-clause-component]**

SOURCE: §13.3 "**the `S` component is absent**"

IMPLEMENTED-BY: descr/maps.rs::Wanted

HELD-BY: two_spellings_of_one_keyed_map_are_one_term, opening_drops_a_field_the_record_already_said, a_label_carrying_its_regions_default_is_absorbed

**Clauses may overlap where the source forbids it.** A key belongs when
*some* clause admits it and its value, in the walk and in subtyping alike, and
two clauses may claim one key: a disjunction, where [Cas23, §4.4] forbids two
domains of one map to overlap and an overlap read at face value is a
conjunction. Neither reads the clauses in order.
**[DEVIATION: clauses-may-overlap]**

IMPLEMENTED-BY: descr/maps.rs::MapLattice::record, ir.rs::canonical_clauses, valgebra-py/src/check/walk/record.rs::covered

HELD-BY: test_heterogeneous_mapping_by_key_schema, test_two_clauses_claiming_one_key_are_a_disjunction, test_a_key_belongs_when_some_clause_admits_it_over_drawn_clauses, a_literal_keyed_clause_is_read_beside_the_clauses_that_cover_its_key, a_field_is_read_instead_of_a_clause_that_names_its_key, test_a_parsed_object_is_covered_by_whichever_clause_can_read_its_keys, test_named_field_takes_precedence_over_the_catch_all

**Clauses are quasi-K-step rather than quasi-constant.** A default per
key-type region rather than one default for the rest; nothing against the
literature, which introduces exactly this generalization. **[DEVIATION: clauses-are-quasi-k-step]**

HELD-BY: every_key_falls_in_exactly_one_part, a_map_constrains_one_part_of_the_key_partition

**The trail holds terms, not addresses.** Assumption pairs are compared by
structural equality in a linear scan, short-circuiting on pointer identity for
interned subtrees; the longest trail any recursive shape in hand builds is
three pairs, counted by a recorder at the one push site, `Cx::assuming`, which
both unfoldings of a reference go through, over the shapes the
goal counter carries -- a list over a reference against a union holding that
reference is the one that reaches three. A decision leaves the trail as it was
given. **[DEVIATION: the-trail-holds-terms]**

HELD-BY: a_decision_leaves_the_trail_it_was_given, an_assumption_is_read_as_the_pair_it_is, the_longest_trail_any_recursive_shape_builds_is_three_pairs

**The assumption set is popped, not threaded.** Every relation proved on the
way is discarded, so a goal reached twice by different paths is decided twice,
and a decision leaves behind no assumption it did not make. The departure is
from [HVP05, §4.2], whose assumption set serves as a cache of every subtype
relation verified; [AC93, §4.4] passes its trail down and never returns it,
the assumption of [AC93, §1.5] held for the path that makes it, which is what
this does. The cost is the one
an algorithm that keeps nothing across its calls pays, described here rather
than cited -- the analysis is Gapeyev, Levin & Pierce's (JFP 2002), which the
shelf does not hold: a family whose every level reaches the next twice doubles
its work per level, and the work budget is what caps it, declining the family
eighteen levels deep. **[DEVIATION: the-assumption-set-is-popped]**

IMPLEMENTED-BY: decision.rs::Cx::assuming

HELD-BY: a_decision_leaves_an_assumption_it_did_not_make, a_decision_leaves_the_trail_it_was_given, a_query_gives_its_stacks_back_as_it_was_given_them, a_goal_reached_by_two_paths_is_derived_twice_until_the_budget_declines

**The dict key partition is by kind.** `1` and `True` are one key to a dict and
two labels to the partition, so an atom requiring both under distinct values
holds no dict, and the descriptor reads it that way. **[DEVIATION: the-key-partition-is-by-kind]**

HELD-BY: an_atom_requiring_a_key_and_its_boolean_holds_no_dict, a_want_with_one_viable_witness_requires_that_key, a_labeled_key_witnesses_a_wanted_key

**The kind partition overlaps at `bool`.** The source's kinds are its three
constructors [FCB08, §6.1], and no value has two of them; its basic types may
overlap, two of them having a nonempty intersection [FCB08, §4.2]. The departure
is making the basic types kinds: `bool`, `int`, `float`, `str`, `bytes` and
`None` come out of the basic kind, and the containers beside them are
constructors of their own. `bool` subclasses `int`, so every boolean is an
integer: a meet of the
two is the booleans rather than empty, and `bool` met with `~int` is empty. The
rules read the pair through `Kind::shares_values_with`, the one exception it
names, and the set representation through the region an integer admits, which
holds the booleans (`Kind::admitted_regions`). A rule reading two kinds as two
disjoint sets would prove `bool & int` empty, which `True` refutes.
**[DEVIATION: the-kind-partition-overlaps-at-bool]**

IMPLEMENTED-BY: decision/readings.rs::Schema::type_tag_with, kind.rs, kind.rs::Kind::shares_values_with, kind.rs::Kind::admitted_regions

HELD-BY: a_boolean_is_the_one_kind_another_kind_holds, test_a_boolean_is_an_integer_and_no_other_kind_is

**A constraint no fresh key meets is read beside the others.** Property
[Cas23, (11)] reads each constraint of a map atom's `S` on its own, against the
default, under the hypothesis [Cas23, §4.4] states before [Cas23, Lem. 4.7],
that every key type is infinite, so each constraint takes a key no label names
and no other constraint wants. Two things break that here. The parts of `bool`
and `None` hold two keys and one, and a constraint the default cannot meet is
left to the keys the labels name. Those constraints are given keys together: a
key holds one value, which every constraint given it asks for, and the keys a
dict then carries must be keys one dict can hold -- so the one-key record
`{"a": int | str}` is below `{str: ~int} | {str: ~str}`. The search is bounded
by `MAX_ASSIGNMENTS` and declines past it.
**[DEVIATION: pinned-wants-are-read-together]**

IMPLEMENTED-BY: descr/maps.rs::Search::assign

HELD-BY: two_constraints_pinned_to_one_key_ask_it_for_both, a_boolean_the_integer_label_occupies_witnesses_nothing, the_one_none_key_cannot_witness_against_its_own_label, the_search_for_keys_past_its_bound_is_unknown, test_one_key_cannot_witness_two_clauses_of_a_union

**A constant is the values equal to it, so `nan` is the empty set.** The model
takes a singleton type for every constant, `[[b_c]] = {c}` [FCB08, §4.2]. A
literal here denotes `{x | type(x) is type(c) and x == c}`, the constant's exact
type refined by Python's equality: a refined base type [JV20, §3.2] whose
predicate is `==`. Where `==` joins no other value to the constant -- an `int`,
a `bool`, a `str`, `bytes`, `None`, a float or complex number with no zero part,
an instance of a type comparing by identity, which covers the spec's `Literal`
types and an `Enum`'s members -- that is the model's singleton. A constant that
does not equal itself, `float("nan")`, denotes the empty set: a `nan` literal
admits nothing, and a union carrying one beside members a supertype holds is
below that supertype. Where `==` joins more, the literal is wider than its
constant: `Literal[0.0]` holds `-0.0`, `Literal[Decimal("1.00")]` holds
`Decimal("1.0")`, and a type whose `__eq__` is its own may hold values that
differ. So the constant speaks for the literal only where nothing tells the two
apart.
Two equal values answer every atom alike under the trust base -- a type test,
`==`, a bound's comparison, a step's `%` -- but two: a callback, which may answer
any function of the value, and an attribute record, which reads what two equal
instances may hold apart. A proof that walks the constant through a supertype is
the literal's where the constant is the only value or the supertype reaches
neither, and a refutation by a constant equal to itself stands always, since the
constant is a value of its literal. A meet holding a literal is the union of
each constant's literal met with the rest [FCB08, Lem. 6.2], and takes the same
two answers the other way round: a constant every other member admits inhabits
it, and one a member refuses takes its literal out only where it speaks for the
literal, the refusal read through the member's complement, since the walk
refuses a value it stopped at from a set and its complement alike.
**[DEVIATION: a-constant-is-the-values-equal-to-it]**

IMPLEMENTED-BY: decision/literals.rs::finite_set_below, decision/literals.rs::the_constant_speaks_for_the_literal, valgebra-py/src/check/walk/scalar.rs::literal_matches, valgebra-py/src/oracle.rs::<PoolRelations as LeafRelations>::literals_disjoint, valgebra-py/src/oracle.rs::<PoolRelations as LeafRelations>::literal_sets_disjoint, valgebra-py/src/oracle.rs::<PoolRelations as LeafRelations>::constant_is_alone, decision/emptiness.rs::value_count_bounds, decision/literals.rs::FiniteMeet, decision/literals.rs::a_meet_read_by_its_constants, descr/lower.rs::singleton, descr/floats.rs::FloatSet::just, descr/mod.rs::Descr::float

HELD-BY: a_constant_that_does_not_equal_itself_denotes_no_value, a_literal_nan_is_the_empty_set, test_nan_never_matches_a_nan_literal, a_constant_speaks_for_its_literal_where_nothing_tells_equal_values_apart, a_constant_is_alone_where_equality_joins_nothing_to_it, test_a_float_zero_literal_is_not_proved_below_a_sign_predicate, test_a_decimal_literal_is_not_proved_below_a_predicate_on_its_exponent, test_an_attribute_record_tells_two_equal_constants_apart, test_a_literal_that_is_its_whole_class_is_proved_below_a_predicate, test_a_literal_is_not_always_a_singleton, a_meet_holding_a_table_is_read_by_its_constants, a_meet_holding_a_table_is_decided_as_its_constants_are_walked, test_a_meet_holding_a_literal_is_decided_as_its_values_are_walked, test_a_literal_met_with_a_deep_complement_is_decided_by_its_constant

**Membership has a third answer.** The membership predicate is two-valued over
the values it is defined on, and a negation is its `not` [Cas24, Def. 3]. The
walk answers neither for a value past what it can read -- one past the depth
bound, one inside itself, one that moved while read, one whose predicate raised
-- and a complement reads its inner walk in Kleene's strong logic, where that
third answer stays the third: `complement(t)` refuses such a value with the
stop's code rather than admitting it
([04-walk.md](04-walk.md#a-stop-is-not-a-non-member)).
**[DEVIATION: membership-has-a-third-answer]**

IMPLEMENTED-BY: valgebra-py/src/check/walk.rs::member, valgebra-py/src/check/walk.rs::check_union, valgebra-py/src/check/walk.rs::settled_union, valgebra-py/src/check/walk.rs::check_complement

HELD-BY: test_a_complement_refuses_a_value_its_inner_walk_stopped_on, test_a_complement_refuses_a_document_its_inner_walk_stopped_on, a_complement_refuses_a_value_its_inner_walk_stopped_on

**The integer and float carriers are `i64` and `f64`.** Membership is exact --
the walk reads the Python object -- and a relation *declines* where deciding it
takes a value the carrier cannot spell. A rounded carrier would be wrong in one
direction, so it refuses.

What that bounds is narrower than the whole question, and the narrowing is the
same under both operators. An **inclusion** is settled by the operands rather
than by the sets they name: two steps by the oracle's `%`, where the base and
both steps meet in one kind of number, and two order bounds by the oracle's
comparison. So `MultipleOf(5000)` is proved below `MultipleOf(2500)` and
`Ge(2**70 + 1)` below `Ge(2**70)`, at any size, with no interval and no residue
materialized to say so. What wants the representation is the **refutation**,
which needs a value the two schemas disagree on, and the meet, whose period is
the one the two steps share.
**[DEVIATION: the-carriers-are-i64-and-f64]**

HELD-BY: a_step_past_the_period_bound_is_refused, test_every_edge_of_the_integer_carrier_has_a_row, test_the_relation_is_the_one_recorded, test_a_multiple_is_a_remainder_of_zero, test_a_bound_past_the_carrier_is_proved_by_the_order, steps_entail_only_where_the_remainder_reads_them_alike

**A class is described by what the frontend can read of it.** A class laying
down a layout the frontend cannot read is its `isinstance` test and its kind and
nothing else, so every relation about the structure its instances have is a
question no rule can answer. The record half carries no class for the same
reason the halves are split -- each is then a set the rules already know -- and
it is decided against another record, by width and by depth, record
containment [Cas23, Lem. 2.3] read with an unnamed attribute free, as an open
record leaves it [Cas23, §3.2], and against the
lattice bounds. Against every other node it **declines** in both directions: a
subclass of any kind may carry an attribute, so without an oracle that
enumerates a kind's values there is no witness either way, and a refutation
would report a value nobody has. **[DEVIATION: a-class-is-what-can-be-read]**

IMPLEMENTED-BY: decision/records.rs::attr_record_subtype

HELD-BY: an_attribute_record_relates_to_every_other_node, a_class_met_with_its_attributes_has_a_value_when_its_fields_do, a_refutation_about_a_class_and_its_record_stands_on_an_instance_it_permits

Two places agree with their sources and are named so a reader does not take
them for departures: the gradual atom, which is the top spelled rather than a
node beside `Int` ([above](#gradual-typing)); and the necessary-property suite,
which is named for what all six of its properties are and names the two that are
also metamorphic relations.

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

**No result here is implemented whole.** What is implemented is named per entry,
and the shape of every gap is the same: a construction the paper gives over its
own type language, carried out here over the fragment this IR can spell. The
word component is the classical construction, a complement by exchanging the
accepting states of an automaton `regex-automata` determinized once per pattern,
a product and an emptiness test, on the regular languages a `str` refinement
denotes. Hosoya, Vouillon and Pierce describe it [HVP05, §4.1] and set it aside:
an early XDuce used it, and it realized every exponential blow-up the types
held, so their paper decides inclusion top-down instead. Neither that algorithm
nor the tree-automata inclusion it decides is built, and neither is the
tree-logic engine above. The structural procedure is sound and budget-bounded
and exact on a published fragment, not a decision procedure for the whole
relation. And the interning that shares a name with hash consing is a table of
nodes, not a decision procedure. Where a page here says otherwise, the page is
wrong.

**And two components are attributed to no source at all**, which is a
different and larger gap than any of the above: see [Two representations with
no paper here](#two-representations-with-no-paper-here).

The departures are tagged rows in [the ledger above](#where-this-tree-departs-from-its-sources),
each with the tests that hold its cost.

Tooling and toolchain facts are [11-references.md](11-references.md), not this
page.
