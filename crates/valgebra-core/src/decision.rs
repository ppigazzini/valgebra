//! The structural rules: subtyping, equivalence and disjointness, decided by
//! matching constructors against each other.
//!
//! One module per surface beside this one -- the emptiness every relation
//! reduces to, the readings a pair no rule decides is given, the oracle, and
//! the four constructor surfaces the shelf keeps a paper for each of. What
//! stays here is the coinductive procedure that ties them together: the trail,
//! the hypothesis a recursive pair is proved under, the shape match, and the
//! hand-off to the set representation where the rules decline.

mod constraints;
mod emptiness;
mod literals;
mod products;
mod readings;
mod records;

use std::cell::Cell;

use crate::descr::lower::{Constants, lower_unfolded};
use crate::ir::{Constraint, Polarity, Schema, SeqShape};
use crate::kind::{Region, Regions};
use crate::verdict::{Relation, Verdict};

use crate::oracle::has_complementary_pair;
use constraints::constraint_entailed;
use emptiness::class_with_attributes;
use literals::{finite_set, finite_set_below};
use products::{linear_subtype, seq_splits_across_union};
use readings::outside_every_kind;
use records::{attr_record_subtype, keyed_map_subtype};

pub use crate::oracle::{LeafRelations, NoLeafRelations};

/// The most decision steps one top-level query may take before it stops and
/// returns the conservative answer. Subtyping distributes over unions and
/// intersections and emptiness recurses the structural fragment, so a deeply
/// nested Boolean combination can demand work exponential in its depth; a memo
/// over goals is what would collapse that, and until one is written the
/// procedure bounds its own work.
///
/// The trail carries part of the termination argument already: a goal that comes
/// back returns against its hypothesis rather than unfolding again, which is
/// what decides a recursive schema. It does not carry all of it, because not
/// every goal is a subterm of the query -- [`seq_splits_across_union`] builds a
/// sequence per branch expansion -- so the set of goals the trail draws from is
/// not obviously finite, and this counter is what stands where that argument
/// would go. One budget is threaded through a whole top-level query —
/// subtyping and the emptiness checks it calls into share it, and the two
/// directions of an equivalence share it — so the bound cannot be escaped through
/// a side door or spent twice.
///
/// **This bound is debt, and a memo is not what would pay it.** Regularity
/// bounds the number of distinct subtyping goals a query can reach, so a table
/// over goals would terminate by a theorem rather than by a ceiling -- but
/// counted over the three decision workloads and over a record of thirty-two
/// fields sharing one interned inner schema, the goals a query *repeats* number
/// zero. The trail absorbs recursion, and the field and position caches absorb
/// the shape where one goal is asked once per field. So the ceiling stands in
/// for a termination argument and for nothing else, and there is no speed-up
/// behind it waiting to be collected. What would reopen the question is a shape
/// where one goal is reached by two rules with no cache between them.
///
/// The ceiling is far above any schema a real
/// annotation produces, so a legitimate relation is always decided; only an
/// adversarial schema built to blow up the decision reaches it, and there a
/// `false` ("not proven") is sound by the conservative contract. A complete,
/// work-sharing decision is the interning-based procedure.
pub(crate) const DECISION_BUDGET: u32 = 1_000_000;

/// Spend one unit of `budget`; returns `false` when it is already exhausted, the
/// signal a budgeted decision uses to stop and report the conservative answer.
pub(super) fn spend(budget: &Cell<u32>) -> bool {
    match budget.get().checked_sub(1) {
        Some(remaining) => {
            budget.set(remaining);
            true
        }
        None => false,
    }
}

impl SeqShape {
    /// Whether every sequence this shape admits is also admitted by `other`.
    ///
    /// The whole rule is [`linear_subtype`]; this is where the elements are
    /// borrowed out of the two shapes. There is no alternation to distribute
    /// over and no shape to first prove linear, because a shape is the linear
    /// form.
    fn shape_subtype(
        &self,
        other: &SeqShape,
        cx: SubtypeCx<'_>,
        assumptions: &mut Vec<(Schema, Schema)>,
    ) -> Relation {
        if self == other {
            return Relation::Holds;
        }
        linear_subtype(
            &self.prefix,
            self.tail.as_deref(),
            &other.prefix,
            other.tail.as_deref(),
            cx,
            assumptions,
        )
    }
}

impl Schema {
    /// Whether the descriptor proves every value of this schema is one of
    /// `other`, by proving the difference empty.
    ///
    /// `a ≤ b` is `a ∧ ¬b = ∅`, which is the whole of the test. A **widening and
    /// nothing else**, asked where [`denotes_no_value`](Self::denotes_no_value)
    /// is asked and for the same reasons.
    ///
    /// An empty left side is below every set, and it says so without lowering
    /// the right one at all -- otherwise a schema the descriptor proves empty
    /// would be below nothing whose descriptor it could not build, which is a
    /// pair of answers that contradict each other.
    pub(crate) fn descriptor_contained_in(
        &self,
        other: &Schema,
        pool: &dyn Constants,
        defs: &[Schema],
    ) -> Relation {
        // The two sides are unfolded in opposite directions, which is what makes
        // a difference over a recursive schema sound *in one direction*: the
        // left grows and the right shrinks, so a difference proved empty here
        // was empty before.
        //
        // **The other direction does not carry.** `self⁺ ∧ ¬other⁻` contains
        // the real difference and is not contained by it, so a value found in
        // the widened reading need be no value of `self ∧ ¬other` at all, and
        // the inhabited answer refutes nothing. Where a reference was cut, an
        // empty difference proves the inclusion and an inhabited one is a
        // decline.
        //
        // A side this reading cannot lower, and a difference it cannot build,
        // are declines rather than refutations: nothing about the inclusion is
        // known from a set that was never constructed.
        let cut = !defs.is_empty() && (self.has_reference() || other.has_reference());
        let Some(mine) = lower_unfolded(self, defs, Polarity::Widen, pool) else {
            return Relation::Unknown;
        };
        if mine.emptiness() == Verdict::Empty {
            return Relation::Holds;
        }
        let Some(theirs) = lower_unfolded(other, defs, Polarity::Narrow, pool) else {
            return Relation::Unknown;
        };
        mine.intersect(&theirs.complement())
            .map_or(Relation::Unknown, |difference| {
                let emptiness = difference.emptiness();
                if cut {
                    Relation::proven(emptiness == Verdict::Empty)
                } else {
                    Relation::of_difference(emptiness)
                }
            })
    }

    /// The decision steps [`is_subtype_of`](Self::is_subtype_of) spends, by the
    /// same argument as [`empty_steps`](Self::empty_steps).
    #[cfg(test)]
    pub(crate) fn subtype_steps(&self, other: &Schema) -> u32 {
        self.subtype_steps_under(other, &NoLeafRelations)
    }

    /// The same count where the rules can look a constant or a class up, for a
    /// rule whose work depends on what the oracle answers.
    #[cfg(test)]
    pub(crate) fn subtype_steps_under(&self, other: &Schema, oracle: &dyn LeafRelations) -> u32 {
        let budget = Cell::new(DECISION_BUDGET);
        self.is_subtype_rec(
            other,
            SubtypeCx {
                oracle,
                defs: &[],
                budget: &budget,
            },
            &mut Vec::new(),
        );
        DECISION_BUDGET - budget.get()
    }

    /// Whether every value of `self` is also a value of `other` — set inclusion,
    /// the semantic-subtyping relation. Complete on the scalar fragment via
    /// `self ∧ ¬other = ∅`, and decided structurally past it by recursion on
    /// matching constructors (the lattice rules, set/frozenset element
    /// inclusion, and sequence inclusion on the prefix-and-tail form). Every
    /// rule is **sound** — it never reports a subtype it cannot justify — and
    /// conservative where it cannot decide: there it returns `false` rather than
    /// guess. A `false` is then asked again of the two *sets*, under a bound on
    /// what building them may cost, which decides
    /// what no rule about shapes reaches: a container meet, a double complement,
    /// one regular language inside another, a kind against its own literals, one
    /// step dividing another.
    ///
    /// The decision is bounded twice over: an adversarial schema that would take
    /// more than a fixed number of steps stops, and a descriptor too costly to
    /// build refuses. A `false` can therefore mean "not proven a subtype within
    /// the work bound". A real schema decides far inside both.
    #[must_use]
    pub fn is_subtype_of(&self, other: &Schema) -> bool {
        self.is_subtype_of_under(other, &NoLeafRelations, &[])
    }

    /// [`is_subtype_of`](Self::is_subtype_of) with a [`LeafRelations`] oracle deciding
    /// the leaf relations the structural rules cannot (an `Instance` class or a
    /// `Literal` value), and the `defs` that resolve recursive references so
    /// subtyping is decided between recursive schemas too. The oracle's `None`
    /// and an unresolved reference both keep the conservative `false`.
    #[must_use]
    pub fn is_subtype_of_under(
        &self,
        other: &Schema,
        oracle: &dyn LeafRelations,
        defs: &[Schema],
    ) -> bool {
        self.subtype_relation_under(other, oracle, defs).holds()
    }

    /// The inclusion in three values: proven, refuted, or neither.
    ///
    /// [`is_subtype_of_under`](Self::is_subtype_of_under) is this reduced to the
    /// two a boundary reports, and the reduction loses the distinction a caller
    /// most often wants: whether a `false` is a value outside the supertype or
    /// a question no rule here answers. Both deciders are asked, the second
    /// only where the first declines, and a refutation from either stands on a
    /// value of the subject -- which the rules' answer is read against, and
    /// which the set reading proves by finding the difference inhabited.
    #[must_use]
    pub fn subtype_relation_under(
        &self,
        other: &Schema,
        oracle: &dyn LeafRelations,
        defs: &[Schema],
    ) -> Relation {
        let budget = Cell::new(DECISION_BUDGET);
        self.subtype_relation(other, oracle, defs, &budget)
            .or_else(|| self.descriptor_contained_in(other, oracle, defs))
    }

    /// The three-valued subtyping answer under an oracle, the definitions, and a
    /// work budget the caller owns.
    ///
    /// The structural procedure alone: the descriptor's own reading is a second
    /// decider the public relation asks after this one. Held here so a test can
    /// count what the rules decline rather than reading `false` for both halves
    /// of the conservative contract.
    pub(crate) fn subtype_relation(
        &self,
        other: &Schema,
        oracle: &dyn LeafRelations,
        defs: &[Schema],
        budget: &Cell<u32>,
    ) -> Relation {
        let cx = SubtypeCx {
            oracle,
            defs,
            budget,
        };
        self.is_subtype_rec(other, cx, &mut Vec::new())
    }

    /// Whether this schema has a value by its own shape, read without descent.
    ///
    /// The witness guard runs on every refutation and at every level of one, so
    /// the shapes whose inhabitance is their own form answer here rather than
    /// through the general fold: a scalar atom is a value, a container that
    /// admits an empty one has that value whatever its elements say, and a
    /// union has one where a member does. `false` is "not seen from here" and
    /// costs the descent, never an answer -- and `laws.rs` holds this to the
    /// fold in the direction it claims.
    pub(crate) fn holds_a_value_shallowly(&self) -> bool {
        match self {
            Schema::Anything(_)
            | Schema::Bool
            | Schema::Int
            | Schema::Float
            | Schema::Str
            | Schema::Bytes
            | Schema::NoneType
            | Schema::Coll { .. } => true,
            Schema::Seq { shape, .. } => shape.prefix.is_empty(),
            Schema::KeyedMap { fields, .. } => fields.iter().all(|f| !f.required),
            Schema::Union(members) => members.iter().any(Schema::holds_a_value_shallowly),
            _ => false,
        }
    }

    /// One rule's answer about this subject, with a refutation believed only
    /// where the subject has a value to stand on.
    ///
    /// Every refutation the rules reach is a mismatch of shapes: a value of the
    /// subject the other schema rejects. A subject with no value has no such
    /// value, so the reading is what turns a mismatch into a claim -- and it is
    /// taken about the subject of *this* comparison, at every level, because a
    /// composition carries a part's refutation up and the part is a subject of
    /// its own. A list of an empty element is the empty list and is below a
    /// list of anything; the mismatch its element reports is about no value.
    ///
    /// It costs the *proving* half of the decision surface 1.4%, measured by
    /// disabling it: sixteen instructions per query that never refutes, which
    /// is the price of reading a refutation honestly on the queries that do. An
    /// `#[inline]` recovers none of it -- the compiler is already free to,
    /// within the crate -- so the reading stands as the cost.
    fn witnessed(&self, answer: Relation, cx: SubtypeCx<'_>) -> Relation {
        if answer == Relation::Fails {
            if self.holds_a_value_shallowly() {
                return answer;
            }
            return Relation::of_mismatch(self.verdict_rec(
                cx.oracle,
                cx.defs,
                &mut Vec::new(),
                cx.budget,
            ));
        }
        answer
    }

    /// The rules' answer about this pair, with every refutation read against
    /// the subject it is about. See [`witnessed`](Self::witnessed).
    fn is_subtype_rec(
        &self,
        other: &Schema,
        cx: SubtypeCx<'_>,
        assumptions: &mut Vec<(Schema, Schema)>,
    ) -> Relation {
        let answer = self.subtype_by_rules(other, cx, assumptions);
        self.witnessed(answer, cx)
    }

    fn subtype_by_rules(
        &self,
        other: &Schema,
        cx: SubtypeCx<'_>,
        assumptions: &mut Vec<(Schema, Schema)>,
    ) -> Relation {
        // Bound the total work: the distribution rules below can demand effort
        // exponential in the schema depth, so once the shared budget is spent the
        // decision stops and returns the conservative `false` rather than running
        // unbounded. A real annotation decides in a few steps; only an adversarial
        // schema reaches the ceiling.
        // Reflexivity, by identity. The checks below are ordered by what they
        // cost, and each answers the whole query on its own, so a cheaper one
        // never hides a verdict a later one would reach.
        if core::ptr::eq(self, other) {
            return Relation::Holds;
        }
        if !spend(cx.budget) {
            // The work bound stopping a descent proves nothing about the pair.
            return Relation::Unknown;
        }
        // Scalar fragment: exact via the region partition. Subtyping there is
        // set inclusion between the two region sets, and nothing else. A
        // non-scalar node yields `Unknown` from its own discriminant, so this
        // costs one match off the fragment.
        //
        // The subtype's own regions are read only once the supertype's are
        // known. Reading a region set is one match on an atom and a fold over
        // every member of a union, so on a Boolean schema it walks the subtree;
        // asking for a set that a mismatched partner has already made unusable
        // walks it for nothing. Both are pure, so which is asked first is free
        // to choose, and the answer is the same either way.
        let supertype_regions = other.region_set();
        if let Regions::Known(b) = supertype_regions
            && let Regions::Known(a) = self.region_set()
        {
            // Exact in both directions: on the scalar fragment the region sets
            // *are* the schemas, so a region outside the supertype's set is a
            // value outside it.
            return Relation::decided(a.subset_of(b));
        }
        // Reflexivity for two equal spellings that are not the same node.
        if self == other {
            return Relation::Holds;
        }
        // Coinductive hypothesis: a goal already being proven on this path is
        // assumed to hold, so two recursive types are compared at their greatest
        // fixpoint rather than unfolded forever. The scan is empty for a
        // non-recursive query; under recursion the stack holds one entry per
        // reference goal still being unfolded. Every recorded goal has a `Ref` on
        // one side, so the structural compare rejects a mismatched goal on the
        // discriminant before walking either subtree.
        if assumptions.iter().any(|(a, b)| a == self && b == other) {
            return Relation::Holds;
        }
        self.subtype_decide(other, supertype_regions, cx, assumptions)
    }

    /// Whether one of the lattice bounds settles `self ⊆ other`: `self` denotes
    /// the empty set, or `other` denotes the whole universe.
    ///
    /// Decided by emptiness rather than by matching the `Nothing` or `Anything`
    /// atom, so a record with an uninhabited required field and a union covering
    /// the universe are both recognised. Asking the atom alone is a rule that
    /// confirms itself: the pattern matches only the shape it is written for.
    ///
    /// The universe side asks whether the complement is empty, which is the same
    /// question one De Morgan step away and needs no separate procedure.
    fn bounds_the_pair(
        &self,
        other: &Schema,
        supertype_regions: Regions,
        cx: SubtypeCx<'_>,
    ) -> bool {
        // `other` covers the universe exactly when its region set is the whole
        // partition. The set is the caller's -- `is_subtype_rec` reads it for the
        // exact scalar rule and nothing between there and here changes `other` --
        // so it arrives as an argument rather than being derived twice. The
        // caller reads it again before the rules, because it is a comparison;
        // this half is the walk, and the caller asks it last.
        if supertype_regions == Regions::Known(Region::ALL) {
            return true;
        }
        if self.is_empty_rec(cx.oracle, cx.defs, &mut Vec::new(), cx.budget) {
            return true;
        }
        // The regions again, read without the stop the fast fold takes at the
        // first opaque member. `~bool | bool` covers the universe and so does
        // the same union with a set beside it, which the fast reading declines
        // because it stops on the set and a member behind it cannot reopen the
        // answer -- so the same three members read as the universe in one order
        // and as unknown in another. A fuzz run found the pair that shows it.
        //
        // Reading on is what the fast fold cannot afford: it is asked of every
        // pair, and a union of a thousand literals is a thousand opaque members
        // to walk rather than one to stop at. Here the pair has had every rule
        // and none of them answered, so the walk is spent on a query that was
        // going to be declined, and a query that decides never reaches it.
        other.covers_the_universe()
    }

    /// Whether `self` and `other` share no value, which is what decides `self ⊆
    /// ¬other`.
    ///
    /// This is the semantic subtyping reduction `[[s ∧ ¬t]] = ∅` applied where
    /// the structural arms have nothing to say: a complement offers no shape on
    /// the right to recurse into, so the question goes to emptiness, which
    /// already decides kind disjointness and the scalar regions. Without it a
    /// container is never seen below the complement of a scalar.
    fn shares_no_value_with(&self, other: &Schema, cx: SubtypeCx<'_>) -> bool {
        // Kind disjointness reads two discriminants and settles most pairs: a
        // list never shares a value with an int. The general question below owns
        // the rest, and has to build the meet to ask it.
        if self.disjoint_with(other, cx.oracle) {
            return true;
        }
        // Built through the meet constructor rather than as a raw node: the
        // emptiness rule that decides most pairs compares the *members* of one
        // intersection pairwise, and a member that is itself an intersection
        // hides its own members from it. `tuple[int, str]` against
        // `dict[str, int] & ~{"a": int}` was the shape that showed it -- the
        // tuple and the dict never met, so `A <= ~B` declined for a pair whose
        // meet `is_empty` decides, and the two relations disagreed about one
        // question asked two ways.
        Schema::meet_within([self.clone(), other.clone()], cx.defs).is_empty_rec(
            cx.oracle,
            cx.defs,
            &mut Vec::new(),
            cx.budget,
        )
    }

    /// Whether `self` reduces to something below `other` by a rule that reads
    /// only the left side: a reference unfolds to its definition, and a
    /// refinement drops to its base.
    ///
    /// Both rules are sound alone, and the `A ⊆ (Y ∪ Z)` rule beside them is
    /// *lossy* -- it commits to a single branch, so a subject that lands in the
    /// union only once it has been reduced gets no answer from it. A match picks
    /// one arm, but the relation is the disjunction of every sound rule that
    /// applies, so where both do, both are asked. That is what decides a
    /// recursive schema against its own body, and a refinement of a union
    /// against that union.
    ///
    /// The reference case records its goal before descending, so a cycle back to
    /// it meets the coinductive hypothesis rather than unfolding forever.
    fn left_reduces_below(
        &self,
        other: &Schema,
        cx: SubtypeCx<'_>,
        assumptions: &mut Vec<(Schema, Schema)>,
    ) -> Relation {
        match self {
            Schema::Ref(id) => match cx.defs.get(id.get()) {
                Some(def) => {
                    assumptions.push((self.clone(), other.clone()));
                    let holds = def.is_subtype_rec(other, cx, assumptions);
                    assumptions.pop();
                    holds
                }
                // A reference into a table that does not hold it: the schema
                // names a set this query cannot read, which is not a refutation.
                None => Relation::Unknown,
            },
            // A refinement carries its base's *proof* and not its refutation:
            // the constraints can exclude every value that would stand against
            // the inclusion, which is how a bounded-length list lands inside a
            // fixed-length one whose base it is nowhere near.
            Schema::Refine { base, .. } => base.is_subtype_rec(other, cx, assumptions).proof_only(),
            // Nothing on the left reduces, so this rule has nothing to say --
            // which is not the same as the relation failing.
            _ => Relation::Unknown,
        }
    }

    /// `A ⊆ (Y ∪ Z)`: every rule that can place a subject inside a union.
    ///
    /// A branch equal to the subject settles it, which is set containment and is
    /// linear in the branches. Failing that, the subject may land in one branch,
    /// or -- for a fixed-arity sequence -- split across them, which no
    /// single-branch rule sees, or be one a left-side rule reduces to something
    /// the union contains.
    ///
    /// Every rule here proves inclusion when it fires, so a union none of them
    /// places the subject in is a pair this procedure declines -- with one
    /// exception, which is the reduction below. A reference denotes exactly its
    /// definition, so a definition that holds a value outside the union names a
    /// value of the reference outside it, and that is a refutation rather than
    /// a decline. The arm for a non-union supertype already reads it that way;
    /// this reads it the same, which is the only difference a union makes.
    fn below_a_union(
        &self,
        other: &Schema,
        members: &[Schema],
        cx: SubtypeCx<'_>,
        assumptions: &mut Vec<(Schema, Schema)>,
    ) -> Relation {
        if members.contains(self)
            || Relation::any(
                members
                    .iter()
                    .map(|m| self.is_subtype_rec(m, cx, assumptions)),
            )
            .holds()
            || seq_splits_across_union(self, members, cx, assumptions)
        {
            return Relation::Holds;
        }
        // Asked once, and read twice: for the proof it may carry, and -- where
        // it carries none -- for the refutation, which only a reference's
        // reduction can give. A refinement's reduction drops its refutation
        // before this sees it, since the constraints may exclude the very value
        // that stood against the inclusion.
        let reduced = self.left_reduces_below(other, cx, assumptions);
        if reduced.holds()
            // Last, and only for the one subject the oracle can answer about
            // here: an `Instance` whose *values* the bindings can enumerate
            // is below a union when each of them is, which is what makes an
            // enumeration and the union of its members one set. Asked here
            // because a union on the right never reaches the leaf arm, and
            // asked for nothing else because every other subject would pay a
            // call that always declines -- ten percent of the decision
            // workload, measured.
            || (matches!(self, Schema::Instance(_))
                && cx.oracle.leaf_subtype(self, other).unwrap_or(false))
        {
            return Relation::Holds;
        }
        reduced.refutation_only()
    }

    /// `(A ∩ B) ⊆ C`: a meet is below whatever one of its conjuncts is below.
    ///
    /// When `C` is a union the meet may instead land in one branch, so that
    /// sound rule is tried here too -- ahead of the plain `_ ⊆ (Y ∪ Z)` rule, so
    /// a meet that contains its own supertype (a reference beside that union)
    /// decides, which is what lets such a meet be recognised as a subtype of
    /// itself.
    ///
    /// Both halves are sound and incomplete: a conjunct that does not contain
    /// the supertype says nothing about the meet, so a disjunction of them that
    /// finds no proof is unproven rather than refuted.
    fn meet_below(
        &self,
        other: &Schema,
        members: &[Schema],
        cx: SubtypeCx<'_>,
        assumptions: &mut Vec<(Schema, Schema)>,
    ) -> Relation {
        // Every rule here proves and none refutes: a member that is *not* below
        // the supertype says nothing about the meet, which is a smaller set
        // than that member. So a pair none of them places is handed to the
        // reading every pair with no rule gets, rather than ending the match.
        let placed = Relation::proven(
            Relation::any(
                members
                    .iter()
                    .map(|m| m.is_subtype_rec(other, cx, assumptions)),
            )
            .holds()
                || matches!(other, Schema::Union(branches)
                    if Relation::any(
                        branches
                            .iter()
                            .map(|b| self.is_subtype_rec(b, cx, assumptions)),
                    )
                    .holds()),
        );
        if placed.holds() {
            return placed;
        }
        // One shape does refute, and it is the meet a dataclass lowers to. A
        // class together with the attributes its instances carry is below
        // another class exactly as its class is: the value the meet holds is a
        // *direct* instance of `C` carrying those attributes, so `type(v) is C`
        // settles `isinstance(v, D)` through the order alone, and a class
        // deriving from both changes nothing about that value. The guard around
        // this rule reads whether the meet has it.
        if !class_with_attributes(members) {
            return placed;
        }
        let Some(class) = members.iter().find_map(|m| match m {
            Schema::Instance(ix) => Some(*ix),
            _ => None,
        }) else {
            return placed;
        };
        if matches!(other, Schema::Instance(_))
            && cx.oracle.leaf_subtype(&Schema::Instance(class), other) == Some(false)
        {
            return Relation::Fails;
        }
        // And against a supertype every value of which has one kind: the same
        // direct instance either has that kind or does not, and the oracle
        // reads `type(v) is C` rather than the subtree beneath `C`, which is
        // what makes the answer a value rather than an open-world guess.
        if outside_every_kind(other, class, cx) {
            return Relation::Fails;
        }
        placed
    }

    /// The structural subtyping decision: the lattice, recursion, and
    /// constructor-matching rules.
    ///
    /// Reached from [`is_subtype_rec`](Self::is_subtype_rec) after the
    /// coinductive, scalar and identity fast paths, which are the three that
    /// answer without reading a shape.
    fn subtype_decide(
        &self,
        other: &Schema,
        supertype_regions: Regions,
        cx: SubtypeCx<'_>,
        assumptions: &mut Vec<(Schema, Schema)>,
    ) -> Relation {
        // Two finite sets of constants, before the lattice rules distribute
        // them: inclusion between them is membership, and membership is a
        // lookup. What the distribution below would make of the same pair is a
        // decision step per pair of members, which is the product the budget
        // binds; this is a walk of the two lists, which is a sum.
        if let (Some(subject), Some(supertype)) = (finite_set(self), finite_set(other)) {
            let answer = finite_set_below(subject, supertype, other, cx.oracle);
            if answer != Relation::Unknown {
                return answer;
            }
        }
        // `A ⊆ U`, read off the supertype's regions alone: the caller already
        // holds them for the scalar rule, so this is a comparison rather than a
        // walk, and it belongs ahead of the rules because a universe on the
        // right answers every pair. The other lattice bound is a walk and is
        // asked in `or_bounded`, after the rules have had the pair.
        if supertype_regions == Regions::Known(Region::ALL) {
            return Relation::Holds;
        }
        let answer = self.subtype_by_shape(other, cx, assumptions);
        self.or_bounded(answer, other, supertype_regions, cx)
    }

    /// The arms that match a pair by the shapes on its two sides.
    ///
    /// Split from its caller because the two lattice bounds around it are asked
    /// at different points and the whole read past what one function may be.
    /// No `#[inline]`: one was tried and moved no workload, and the compiler is
    /// already free to within the crate.
    fn subtype_by_shape(
        &self,
        other: &Schema,
        cx: SubtypeCx<'_>,
        assumptions: &mut Vec<(Schema, Schema)>,
    ) -> Relation {
        match (self, other) {
            // (X ∪ Y) ⊆ Z iff X ⊆ Z and Y ⊆ Z; A ⊆ (Y ∩ Z) iff A ⊆ Y and A ⊆ Z.
            (Schema::Union(members), _) => Relation::all(
                members
                    .iter()
                    .map(|m| m.is_subtype_rec(other, cx, assumptions)),
            ),
            (_, Schema::Intersection(members)) => Relation::all(
                members
                    .iter()
                    .map(|m| self.is_subtype_rec(m, cx, assumptions)),
            ),
            (Schema::Intersection(members), _) => self
                .meet_below(other, members, cx, assumptions)
                .or_else(|| self.unstructured(other, cx, assumptions)),
            (_, Schema::Union(members)) => self
                .below_a_union(other, members, cx, assumptions)
                .or_else(|| self.unstructured(other, cx, assumptions)),
            // Unfold a recursive reference — after the lattice rules, so an
            // intersection or union meeting a reference decomposes first (which
            // lets a recursive member be compared against the reference rather
            // than the reference being unfolded past it). Where the union rule
            // above ran first and declined, it has already asked this one.
            (Schema::Ref(_), _) => self.left_reduces_below(other, cx, assumptions),
            (_, Schema::Ref(id)) => match cx.defs.get(id.get()) {
                Some(def) => {
                    assumptions.push((self.clone(), other.clone()));
                    let holds = self.is_subtype_rec(def, cx, assumptions);
                    assumptions.pop();
                    holds
                }
                // As in `left_reduces_below`: a reference the table does not
                // resolve names a set this query cannot read.
                None => Relation::Unknown,
            },
            // Set and frozenset inclusion reduces to element inclusion.
            (
                Schema::Coll {
                    container: a_kind,
                    element: a,
                },
                Schema::Coll {
                    container: b_kind,
                    element: b,
                },
            ) if a_kind == b_kind => a.is_subtype_rec(b, cx, assumptions),
            // Same-kind sequence inclusion is language inclusion on the shapes.
            (
                Schema::Seq {
                    container: ka,
                    shape: sa,
                },
                Schema::Seq {
                    container: kb,
                    shape: sb,
                },
            ) if ka == kb => sa.shape_subtype(sb, cx, assumptions),
            // Record and mapping inclusion.
            (
                Schema::KeyedMap {
                    fields: fa,
                    defaults: da,
                },
                Schema::KeyedMap {
                    fields: fb,
                    defaults: db,
                },
            ) => keyed_map_subtype(fa, da, fb, db, cx, assumptions),
            // Record inclusion on attributes: width and depth, no class in it.
            // The nominal half of a dataclass is a separate conjunct, and the
            // lattice rules below relate the meet to the meet.
            (Schema::AttrRecord { fields: fa }, Schema::AttrRecord { fields: fb }) => {
                attr_record_subtype(fa, fb, cx, assumptions)
            }
            // Complement is contravariant: ¬A ⊆ ¬B exactly when B ⊆ A.
            (Schema::Complement(a), Schema::Complement(b)) => b.is_subtype_rec(a, cx, assumptions),
            // `A ⊆ ¬B` is disjointness, which the rules prove or fail to
            // prove -- they are sound and incomplete, so not proving it is not
            // refuting it. There is one shape where the pair *is* refuted, and
            // it is the opposite proof: `A ⊆ B` holding means every value of
            // `A` is in `B`, so a value of `A` is a value outside `¬B`. That
            // needs `A` to have one, which the reading around this settles.
            (_, Schema::Complement(inner)) => {
                if self.shares_no_value_with(inner, cx) {
                    return Relation::Holds;
                }
                if self.is_subtype_rec(inner, cx, assumptions).holds() {
                    Relation::of_mismatch(self.verdict_of(cx))
                } else {
                    Relation::Unknown
                }
            }
            (
                Schema::Refine {
                    base: narrow_base,
                    constraints: narrow_cons,
                },
                Schema::Refine {
                    base: wide_base,
                    constraints: wide_cons,
                },
            ) => refinement_subtype(
                narrow_base,
                narrow_cons,
                wide_base,
                wide_cons,
                cx,
                assumptions,
            )
            // The constraint rule only proves, so a pair it leaves unproven is
            // handed to the reading every pair with no rule gets, rather than
            // ending the match here. A refinement against a *non*-refinement
            // already reaches that reading, and it is written for a refinement
            // supertype -- so a refinement on both sides was the one pair kept
            // from it, and a list refinement against an integer one walked the
            // sets to learn that a list is not an integer.
            .or_else(|| self.unstructured(other, cx, assumptions)),
            // Against a non-refinement, a refinement inherits its base's
            // supertypes -- and inherits nothing else, so a pair this leaves
            // unproven is a pair with no rule of its own.
            (Schema::Refine { .. }, _) => self
                .left_reduces_below(other, cx, assumptions)
                .or_else(|| self.unstructured(other, cx, assumptions)),
            _ => self.unstructured(other, cx, assumptions),
        }
    }

    /// `∅ ⊆ B`, asked where the rules declined.
    ///
    /// The remaining lattice bound, and the one that costs a walk: whether the
    /// *subject* is empty. It is asked last rather than first because a proof
    /// stands without it and a refutation is read against that same emptiness
    /// by [`witnessed`](Self::witnessed) on the way out -- so asking first
    /// walked the subject on every pair the rules were about to decide anyway,
    /// which is a sixth of the repeating workload.
    ///
    /// The answer is identical either way. Both readings are sound, and an
    /// empty subject is below everything whichever of them says so.
    fn or_bounded(
        &self,
        answer: Relation,
        other: &Schema,
        supertype_regions: Regions,
        cx: SubtypeCx<'_>,
    ) -> Relation {
        match answer {
            Relation::Unknown if self.bounds_the_pair(other, supertype_regions, cx) => {
                Relation::Holds
            }
            answer => answer,
        }
    }

    /// What a pair no structural rule decided is worth, asked in one place.
    ///
    /// Three readings, in this order. Two schemas that **share no value**:
    /// every value of the subject is outside the supertype, which is a
    /// refutation on the same reading as a mismatched arity, and the
    /// disjointness here is the cheap one -- two discriminants that cannot
    /// overlap, a list beside a tuple, a set beside a mapping. Then the
    /// **oracle**, the only reader of a class or a constant, whose `None` is
    /// the decline it says it is. Then the supertype's own shape: a
    /// **refinement** is a subset of its base, so a subject outside the base is
    /// outside the refinement -- the value that refutes the one refutes the
    /// other, and it is the same value. Its proof does not carry, since being
    /// inside the base says nothing about the constraints, so only the
    /// refutation is taken.
    ///
    /// The oracle is asked before the refinement because it *proves* where the
    /// refinement reading only refutes: a literal below a predicate refinement
    /// is settled by running the predicate on the constant, and a reading that
    /// answered first would take that proof away.
    ///
    /// Every arm above that answers for a shape and then *declines* ends here
    /// rather than ending the match, because a decline is not an answer: such a
    /// pair has had no rule, and this is what a pair with no rule is worth.
    fn unstructured(
        &self,
        other: &Schema,
        cx: SubtypeCx<'_>,
        assumptions: &mut Vec<(Schema, Schema)>,
    ) -> Relation {
        if self.disjoint_with(other, cx.oracle) {
            return Relation::Fails;
        }
        if self.spills_past(other, cx.oracle) {
            return Relation::Fails;
        }
        if self.outside_the_class(other, cx.oracle) {
            return Relation::Fails;
        }
        if self.shorter_than(other) {
            return Relation::Fails;
        }
        match cx.oracle.leaf_subtype(self, other) {
            Some(true) => Relation::Holds,
            Some(false) => Relation::Fails,
            None => match other {
                Schema::Refine { base, .. } => {
                    self.is_subtype_rec(base, cx, assumptions).refutation_only()
                }
                _ => Relation::Unknown,
            },
        }
    }

    /// This schema's emptiness verdict, under the query's oracle and budget.
    ///
    /// The fold that reads it takes four arguments the query already holds, and
    /// three call sites were spelling them out; the trail it starts is empty
    /// because a verdict is a question about one schema rather than about a
    /// goal.
    fn verdict_of(&self, cx: SubtypeCx<'_>) -> Verdict {
        self.verdict_rec(cx.oracle, cx.defs, &mut Vec::new(), cx.budget)
    }

    /// Whether `self` and `other` denote the same set — mutual inclusion.
    ///
    /// Like the relations it composes, the decision is bounded; a `false` can mean
    /// "not proven equivalent within the work bound" for an adversarial schema.
    #[must_use]
    pub fn is_equivalent(&self, other: &Schema) -> bool {
        self.is_equivalent_under(other, &NoLeafRelations, &[])
    }

    /// [`is_equivalent`](Self::is_equivalent) under a [`LeafRelations`] oracle and
    /// the recursive definitions.
    #[must_use]
    pub fn is_equivalent_under(
        &self,
        other: &Schema,
        oracle: &dyn LeafRelations,
        defs: &[Schema],
    ) -> bool {
        // Both inclusion directions share one budget, so equivalence cannot spend
        // twice the ceiling, and its verdict does not depend on which direction
        // happened to allocate a fresh allowance first.
        let budget = Cell::new(DECISION_BUDGET);
        let cx = SubtypeCx {
            oracle,
            defs,
            budget: &budget,
        };
        let within = |sub: &Schema, sup: &Schema| {
            sub.is_subtype_rec(sup, cx, &mut Vec::new())
                .or_else(|| sub.descriptor_contained_in(sup, oracle, defs))
        };
        // Equivalence is the meet of two inclusions, taken in the vocabulary
        // both of them answer in: mutual inclusion, and the conjunction's own
        // short circuit, rather than two booleans that have each forgotten
        // whether they were refuted or merely unproven.
        within(self, other).and(|| within(other, self)).holds()
    }
}

/// Threaded state for the subtyping decision: the leaf-relation oracle, the
/// definitions that resolve recursive references, and the remaining work budget
/// shared across the whole query. The budget counts decision steps down to zero,
/// at which point the procedure stops and returns the conservative `false`,
/// bounding the cost of a deeply nested Boolean combination.
#[derive(Clone, Copy)]
struct SubtypeCx<'a> {
    oracle: &'a dyn LeafRelations,
    defs: &'a [Schema],
    budget: &'a Cell<u32>,
}

/// Every unordered pair of distinct elements, each once.
///
/// The pairwise-disjointness law is asked of an intersection's members and of the
/// inners of a union's complements, and the scan was written out at both. One
/// function decides where the pairs come from, and it reads the tail through
/// `get`, so the law needs no panicking index.
pub(crate) fn unordered_pairs<T>(items: &[T]) -> impl Iterator<Item = (&T, &T)> {
    items.iter().enumerate().flat_map(|(i, a)| {
        items
            .get(i + 1..)
            .unwrap_or_default()
            .iter()
            .map(move |b| (a, b))
    })
}

/// Whether some two members are provably disjoint (distinct concrete kinds,
/// `bool ⊆ int` aside), so their intersection is empty. This decides the
/// structural-kind disjointness (a list is never a set) the scalar region
/// bitset cannot see, and the simplifier reads it too so both consume one
/// statement of the lattice law.
pub(crate) fn has_disjoint_pair(members: &[Schema], oracle: &dyn LeafRelations) -> bool {
    unordered_pairs(members).any(|(a, b)| a.disjoint_with(b, oracle))
}

/// A refinement below another: the base narrows and every constraint holds.
///
/// A refinement is a subset of its base. Against another refinement the base
/// must subtype and every constraint of the supertype must hold of every
/// subtype value: either it appears verbatim, or it is entailed by the
/// subtype's bounds (a tighter lower, upper or length bound entails a looser
/// one, decided through the ordering oracle). A bound the oracle cannot compare
/// and a non-order constraint stay on the verbatim path, so a constraint
/// neither written nor entailed leaves the pair unproven rather than refuted.
fn refinement_subtype(
    narrow_base: &Schema,
    narrow_cons: &[Constraint],
    wide_base: &Schema,
    wide_cons: &[Constraint],
    cx: SubtypeCx<'_>,
    assumptions: &mut Vec<(Schema, Schema)>,
) -> Relation {
    narrow_base
        .is_subtype_rec(wide_base, cx, assumptions)
        .proof_only()
        .and(|| {
            Relation::proven(wide_cons.iter().all(|constraint| {
                narrow_cons.contains(constraint)
                    || constraint_entailed(constraint, narrow_cons, cx.oracle)
            }))
        })
}

#[cfg(test)]
mod budget_tests;

#[cfg(test)]
mod tests;
