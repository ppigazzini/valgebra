//! The decision procedures over the IR: emptiness, subtyping, equivalence, and
//! disjointness, with the leaf-relation oracle and the scalar region partition.

mod constraints;
mod emptiness;
mod literals;
mod products;
mod records;

use std::borrow::Cow;
use std::cell::Cell;

use crate::descr::lower::{Constants, lower};
use crate::ir::{
    ClassIx, CollKind, ConstIx, Constraint, DefIx, OperandIx, Schema, SeqKind, SeqShape,
};
use crate::kind::{Kind, Region, Regions};
use crate::verdict::{Relation, Verdict};

use constraints::constraint_entailed;
use emptiness::class_with_attributes;
use literals::{finite_set, finite_set_below, literal_constants};
use products::{linear_subtype, seq_splits_across_union};
use records::{attr_record_subtype, keyed_map_subtype};

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
    /// Whether this schema and `other` are *provably* disjoint: no value belongs
    /// to both. Sound, not complete — it returns true only when the concrete
    /// types cannot overlap (distinct builtin scalars, distinct container kinds,
    /// a refinement's base versus another), and false (conservatively) for the
    /// cases it cannot decide in the core: `Literal` and `Instance` (a class may
    /// subclass a builtin), `Any`, references, and combinators.
    #[must_use]
    pub fn disjoint(&self, other: &Schema) -> bool {
        self.disjoint_with(other, &NoLeafRelations)
    }

    /// [`disjoint`](Self::disjoint) with an oracle that can kind a `Literal` and
    /// settle a pair of them, neither of which the core can read.
    pub(crate) fn disjoint_with(&self, other: &Schema, oracle: &dyn LeafRelations) -> bool {
        if matches!(self, Schema::Nothing) || matches!(other, Schema::Nothing) {
            return true;
        }
        // Two sets of literals are compared as *sets*. The recursion below is a
        // member-by-member walk, so a union of literals against another is
        // quadratic in the oracle: two twenty-thousand-member unions were four
        // hundred million calls into the bindings, and six seconds. The oracle
        // can answer the whole question at once where it holds the constants,
        // and declines -- leaving the walk to run -- where it cannot.
        if let (Some(left), Some(right)) = (literal_constants(self), literal_constants(other))
            && let Some(answer) = oracle.literal_sets_disjoint(&left, &right)
        {
            return answer;
        }
        // A union shares no value with a schema when none of its members does.
        // The frontend builds `Literal[...]` as a union of its constants, so a
        // single-constant literal arrives wrapped and the pair rule below would
        // never see it.
        match (self, other) {
            (Schema::Union(members), _) => {
                return !members.is_empty()
                    && members.iter().all(|m| m.disjoint_with(other, oracle));
            }
            (_, Schema::Union(members)) => {
                return !members.is_empty()
                    && members.iter().all(|m| self.disjoint_with(m, oracle));
            }
            // Two literals pin `type(x)` exactly, so the kind rule below -- which
            // exempts `bool`/`int` because `bool` subclasses `int` -- is the wrong
            // question for them. Ask the oracle about the constants instead.
            (Schema::Literal(a), Schema::Literal(b)) => {
                return oracle.literals_disjoint(*a, *b).unwrap_or(false);
            }
            _ => {}
        }
        // Two sequences of one container meet in the sequences of their
        // elements' meet -- `a* ∩ b* = (a ∩ b)*`, reading a homogeneous
        // sequence as a regular expression over its element. Elements that
        // share no value leave `∅*`, which is not empty: it holds the empty
        // sequence, so the two are *not* disjoint on this reading alone. A
        // length bound that rules the empty sequence out leaves nothing at all,
        // and that pair is disjoint.
        //
        // The bound is read first: only a refinement carries one, so every
        // other node answers it with one discriminant test, and this is asked
        // of every pair the walk reaches. The two tags are the ones the rule
        // below reads, taken once for both.
        let (tag, other_tag) = (self.type_tag_with(oracle), other.type_tag_with(oracle));
        if (self.holds_an_element() || other.holds_an_element())
            && tag == other_tag
            && let (Some(mine), Some(theirs)) = (self.star_element(), other.star_element())
            && mine.disjoint_with(theirs, oracle)
        {
            return true;
        }
        match (tag, other_tag) {
            // Distinct concrete types are disjoint, except bool ⊆ int.
            (Some(a), Some(b)) => !a.shares_values_with(b),
            // One side has no tag of its own, which a class never has: only the
            // bindings read a class, so the kind goes to them as a question.
            (None, Some(kind)) => self.class_excludes(kind, oracle),
            (Some(kind), None) => other.class_excludes(kind, oracle),
            (None, None) => false,
        }
    }

    /// The element of a schema whose values are that element repeated: a
    /// homogeneous sequence, or a set.
    ///
    /// `None` for a shape that is not a plain star. A sequence with a prefix
    /// denotes more than one element type repeated, so the `(a ∩ b)*` reading
    /// above is not the whole of its meet with another.
    fn star_element(&self) -> Option<&Schema> {
        match self {
            Schema::Refine { base, .. } => base.star_element(),
            Schema::Seq { shape, .. } if shape.prefix.is_empty() => shape.tail.as_deref(),
            Schema::Coll { element, .. } => Some(element),
            _ => None,
        }
    }

    /// Whether a length bound puts at least one element in every value of this
    /// schema.
    ///
    /// What rules the empty sequence out of the meet above. Only a refinement
    /// can say it: a *shape* that guarantees an element is one with a prefix,
    /// and a shape with a prefix is not a star, so it has no element for that
    /// reading to compare and could never reach this question anyway.
    fn holds_an_element(&self) -> bool {
        match self {
            Schema::Refine { base, constraints } => {
                constraints
                    .iter()
                    .any(|constraint| matches!(constraint, Constraint::MinLen(least) if *least > 0))
                    || base.holds_an_element()
            }
            _ => false,
        }
    }

    /// Whether the subject holds a sequence the supertype's length bound leaves
    /// out.
    ///
    /// A sequence type carrying no bound of its own holds the empty sequence,
    /// and a bound of one or more is exactly what excludes that value. So the
    /// empty sequence is a value of the subject outside the supertype, which is
    /// a refutation -- and one read off two nodes, where the descriptor decides
    /// the same pair by building both sets.
    ///
    /// The subject is read as a bare shape rather than through a refinement,
    /// because a constraint this does not read -- a predicate above all -- may
    /// exclude the empty sequence as well, and then there is no value left to
    /// stand on. The supertype's bound is asked first: only a refinement
    /// carries one, so every other node answers with one discriminant test.
    fn shorter_than(&self, other: &Schema) -> bool {
        other.holds_an_element()
            && matches!(self, Schema::Seq { .. } | Schema::Coll { .. })
            && self.star_element().is_some()
    }

    /// Whether this schema is a class the oracle says holds no value of `kind`.
    ///
    /// A class is the one atom the core cannot read at all, and the answer is
    /// the bindings' -- `false` for a class laid out as another kind, since a
    /// subclass inherits the layout and cannot lay down a second. A class
    /// laying down none is declined there rather than refuted: a subclass of it
    /// may derive from a builtin too, and its instances are then of that kind.
    fn class_excludes(&self, kind: Kind, oracle: &dyn LeafRelations) -> bool {
        matches!(self, Schema::Instance(class)
            if oracle.class_admits_kind(*class, kind) == Some(false))
    }

    /// A concrete type tag for nodes whose disjointness the core can decide
    /// soundly. `None` for nodes it cannot (`Literal`/`Instance`/`Any`/...).
    fn type_tag(&self) -> Option<Kind> {
        self.type_tag_with(&NoLeafRelations)
    }

    /// [`type_tag`](Self::type_tag) with the oracle that kinds a `Literal`.
    fn type_tag_with(&self, oracle: &dyn LeafRelations) -> Option<Kind> {
        Some(match self {
            Schema::Literal(constant) => return oracle.literal_kind(*constant),
            Schema::NoneType => Kind::NoneType,
            Schema::Bool => Kind::Bool,
            Schema::Int => Kind::Int,
            Schema::Float => Kind::Float,
            Schema::Str => Kind::Str,
            Schema::Bytes => Kind::Bytes,
            Schema::Seq {
                container: SeqKind::List,
                ..
            } => Kind::List,
            Schema::Seq {
                container: SeqKind::Tuple,
                ..
            } => Kind::Tuple,
            Schema::Coll { container, .. } => match container {
                CollKind::Set => Kind::Set,
                CollKind::FrozenSet => Kind::FrozenSet,
            },
            Schema::KeyedMap { .. } => Kind::Dict,
            // A refinement is a subset of its base, so its base's disjointness
            // is sound for it.
            Schema::Refine { base, .. } => return base.type_tag_with(oracle),
            _ => return None,
        })
    }

    /// The region a scalar atom denotes *exactly*, or `None` for every other node.
    ///
    /// Exactness is the whole condition. A region set is read back through
    /// [`Region::complement`], and the complement of an over-approximation is an
    /// under-approximation -- which would report an inhabited schema empty. So a
    /// node earns a region only when it denotes that region and nothing less:
    /// `str` is every string, while `list[int]` is a proper part of the lists
    /// and stays opaque.
    fn atom_region(&self) -> Option<Region> {
        Some(match self {
            Schema::NoneType => Kind::NoneType.region(),
            Schema::Bool => Kind::Bool.region(),
            // `bool` subclasses `int`, so an `int` schema admits both regions.
            // This is the one place a schema's regions are not its kind's.
            Schema::Int => Kind::Bool.region().union(Kind::Int.region()),
            Schema::Float => Kind::Float.region(),
            Schema::Str => Kind::Str.region(),
            Schema::Bytes => Kind::Bytes.region(),
            _ => return None,
        })
    }

    /// The value-universe regions this schema denotes, as a set over the
    /// [`Kind`] partition, or [`Regions::Unknown`] when the schema is not
    /// *scalar-decidable* — built only from the scalar atoms, `Nothing`,
    /// `Anything`, and the `Union`/`Intersection`/`Complement` combinators. On
    /// that fragment the set is **exact**, so emptiness and subtyping are decided
    /// completely; elsewhere the caller stays conservative. The gradual `Any`,
    /// literals, instances, refinements, content-bearing containers, and
    /// references are not scalar-decidable, so any combination holding one is
    /// `Unknown`.
    pub(crate) fn region_set(&self) -> Regions {
        if let Some(region) = self.atom_region() {
            return Regions::Known(region);
        }
        Regions::Known(match self {
            Schema::Nothing => Region::EMPTY,
            Schema::Anything(_) => Region::ALL,
            Schema::Union(members) => {
                let mut acc = Regions::UNION_UNIT;
                for member in members.iter() {
                    acc = acc.union(member.region_set());
                    if acc.is_absorbing() {
                        return acc;
                    }
                }
                return acc;
            }
            Schema::Intersection(members) => {
                let mut acc = Regions::MEET_UNIT;
                for member in members.iter() {
                    acc = acc.intersect(member.region_set());
                    if acc.is_absorbing() {
                        return acc;
                    }
                }
                return acc;
            }
            Schema::Complement(inner) => match inner.region_set() {
                Regions::Known(regions) => regions.complement(),
                Regions::Unknown => return Regions::Unknown,
            },
            // A refinement with no constraint denotes exactly its base, so it
            // earns the base's regions. One with a constraint narrows, and a
            // narrowed region set read back through a complement would report an
            // inhabited schema empty -- which is why the rest are unknown.
            Schema::Refine { base, constraints } if constraints.is_empty() => {
                return base.region_set();
            }
            _ => return Regions::Unknown,
        })
    }

    /// Whether this schema's regions cover the universe, read without the early
    /// stop [`region_set`](Self::region_set) takes at the first opaque member.
    ///
    /// `~bool | bool` covers the universe and so does the same union with a set
    /// beside it; the fast reading declines the second, because it stops on the
    /// member the partition cannot read and nothing behind that member can
    /// reopen the answer -- so the same three members read as the universe in
    /// one order and as unknown in another. Here every member is read and what
    /// they cover *between* them is carried, which no pairwise step can do, and
    /// the walk stops where the cover is complete because no member can add to
    /// it. Order stops mattering in both directions.
    ///
    /// Reading on is what the fast fold cannot afford: it is taken of every
    /// pair, and a union of a thousand literals is a thousand opaque members to
    /// walk where the stop reads one. So this is asked once, by
    /// [`bounds_the_pair`](Self::bounds_the_pair), for a pair every rule has
    /// already declined.
    ///
    /// It answers a `bool` rather than a region set because that is the whole
    /// of what its one caller asks, and a set would carry a part it cannot
    /// mean: what the members cover between them is a *lower* bound, exact only
    /// where it reaches the universe.
    ///
    /// It is the reading `simplify::finish_union` already takes -- it folds the
    /// regions of the members it can read and ignores the rest, "keeping the
    /// decision independent of grouping" -- so this is the decision agreeing
    /// with the construction rather than a new claim about a union.
    fn covers_the_universe(&self) -> bool {
        let Schema::Union(members) = self else {
            // Nothing else folds a member list, so nothing else has a stop to
            // read past. A meet covers the universe only when every member
            // does, and the rule for a meet on the right asks each on its own;
            // a complement covers it only when the schema under it holds no
            // value, which the complement rule already asks.
            return self.region_set() == Regions::Known(Region::ALL);
        };
        let mut covered = Region::EMPTY;
        for member in members.iter() {
            // A member this partition cannot read contributes nothing and
            // hides nothing: it is passed over rather than stopped on. Asking
            // it this question again is redundant, and the sweep says so -- a
            // member that covers the universe by itself is a branch the union
            // rule takes before this reading is reached at all.
            let Regions::Known(regions) = member.region_set() else {
                continue;
            };
            covered = covered.union(regions);
            if covered == Region::ALL {
                return true;
            }
        }
        false
    }

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
        // a difference over a recursive schema sound: the left grows and the
        // right shrinks, so a difference proved empty here was empty before.
        //
        // A side this reading cannot lower, and a difference it cannot build,
        // are declines rather than refutations: nothing about the inclusion is
        // known from a set that was never constructed.
        let Some(mine) = lower(&unfolded_for(self, defs, true), pool) else {
            return Relation::Unknown;
        };
        if mine.emptiness() == Verdict::Empty {
            return Relation::Holds;
        }
        let Some(theirs) = lower(&unfolded_for(other, defs, false), pool) else {
            return Relation::Unknown;
        };
        mine.intersect(&theirs.complement())
            .map_or(Relation::Unknown, |difference| {
                Relation::of_difference(difference.emptiness())
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

    /// The structural subtyping decision: the lattice, recursion, and
    /// constructor-matching rules. Reached from [`is_subtype_rec`] after the
    /// coinductive, scalar, identity, and memo fast paths.
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

    /// Whether this schema's kind holds a value the supertype's class does not.
    ///
    /// A subject every value of which has one kind is refuted against a class
    /// that kind's own builtin does not derive from: a value of the kind built
    /// as that builtin has `type(v)` equal to it, so the order settles
    /// `isinstance`, and no class deriving from both changes that particular
    /// value. The dual of the reading a meet of a class and its attributes
    /// gets, with the kind and the class swapping sides.
    fn outside_the_class(&self, other: &Schema, oracle: &dyn LeafRelations) -> bool {
        let Schema::Instance(class) = other else {
            return false;
        };
        // A complement holds a value of every kind its inner schema's own kind
        // is not, so the reading is asked of those instead of of one. Any of
        // them that the class's order refutes names the witness: a value built
        // as that kind's builtin is outside the class, and outside the inner
        // schema because its kind is not that one.
        //
        // The search stops at the first, and it is only reached by a complement
        // against a class -- everything else asks the one question it did.
        if let Schema::Complement(inner) = self {
            return inner.type_tag_with(oracle).is_some_and(|excluded| {
                Kind::ALL.iter().any(|kind| {
                    !kind.shares_values_with(excluded)
                        && oracle.kind_derives_from(*kind, *class) == Some(false)
                })
            });
        }
        self.type_tag_with(oracle)
            .is_some_and(|kind| oracle.kind_derives_from(kind, *class) == Some(false))
    }

    /// Whether the subject holds a region the supertype's kind cannot.
    ///
    /// The scalar rule decides a pair only where *both* region sets are known,
    /// and a container's is not: `list[int]` is a proper part of the lists, so
    /// it earns no region. But a container has a *kind*, and a kind bounds the
    /// regions its values can be in -- so a subject whose regions are exact and
    /// reach outside that bound holds a value the supertype rejects.
    ///
    /// Exactness on the subject's side is what makes it a refutation rather
    /// than a guess, and every region is inhabited -- `None`, `False`, `0`,
    /// `0.0`, `""`, `b""`, `[]` -- so a region outside the bound names a value.
    /// The bound is [`Kind::admitted_regions`] rather than the kind's own
    /// region, which is what keeps `bool` inside a bounded `int`.
    /// The complement of a scalar is the shape this reads: `¬int` is every
    /// region but two, and a list is one of them.
    ///
    /// Asked where no rule decided, so a pair the scalar rule already answered
    /// never reaches it, and a subject with no exact region set pays one match.
    fn spills_past(&self, other: &Schema, oracle: &dyn LeafRelations) -> bool {
        let Regions::Known(mine) = self.region_set() else {
            return false;
        };
        other
            .type_tag_with(oracle)
            .is_some_and(|kind| !mine.subset_of(kind.admitted_regions()))
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

/// Resolves the leaf relations the structural subtyping decision cannot: those
/// that depend on the Python class hierarchy (an `Instance`) or on a concrete
/// value (a `Literal`). The bindings implement it with `issubclass` and
/// membership; the core defaults to [`NoLeafRelations`].
///
/// It carries [`Constants`] because both are the same question asked twice: the
/// object table lives in the bindings, and an implementor that can answer what a
/// `Literal` is a subtype of can also say what it *names*. Requiring it here is
/// what lets a relation lower its two schemas -- a descriptor holding a constant
/// as a value decides `MultipleOf(4) ≤ MultipleOf(2)`, which no rule about
/// constraints reaches -- rather than declining every schema that names one.
pub trait LeafRelations: Constants {
    /// Whether leaf schema `sub` is a subtype of `sup`, or `None` to leave the
    /// relation conservatively undecided.
    fn leaf_subtype(&self, sub: &Schema, sup: &Schema) -> Option<bool>;

    /// Order the two pool values behind refinement bounds at indices `left` and
    /// `right`, or `None` when the core cannot or the values are not comparable.
    /// The default decides nothing, so bound satisfiability stays conservative.
    fn compare(&self, _left: OperandIx, _right: OperandIx) -> Option<core::cmp::Ordering> {
        None
    }

    /// Whether two *sets* of literal constants share no value, or `None` to
    /// leave it to the pairwise question.
    ///
    /// The same relation [`literals_disjoint`](Self::literals_disjoint) answers
    /// for one pair, asked of every pair at once. The core walks members against
    /// members, which is quadratic in the oracle for two wide literal unions --
    /// a shape a contract really writes, since an enumeration of codes is one.
    /// An implementor that can hash its constants answers in one pass; the
    /// default declines, and the walk stands.
    fn literal_sets_disjoint(&self, _left: &[ConstIx], _right: &[ConstIx]) -> Option<bool> {
        None
    }

    /// Whether no integer lies between the pool values at `lo` and `hi`, under the
    /// strictness of each bound (`lo_strict` excludes `lo`, `hi_strict` excludes
    /// `hi`). The core asks this only for an integer-discrete refinement base, so a
    /// `Some(true)` proves the interval admits no integer and the refinement is
    /// empty. `None` leaves the discreteness rule conservative — the default, so a
    /// core with no value oracle never decides on integer adjacency.
    fn no_int_between(
        &self,
        _lo: OperandIx,
        _lo_strict: bool,
        _hi: OperandIx,
        _hi_strict: bool,
    ) -> Option<bool> {
        None
    }

    /// Whether an atom the core cannot read denotes a *set* -- the same values
    /// however often it is asked -- or `None` when the bindings cannot say.
    ///
    /// Asked of a class: `isinstance` against a metaclass that overrides
    /// `__instancecheck__` runs user code, so two occurrences of one class can
    /// disagree and `A ∩ ¬A` is not empty. Telling a pure class from a hooked one
    /// needs the class object, which only the bindings hold. The default decides
    /// nothing, so a core with no oracle treats every such atom as one it cannot
    /// reason about -- the conservative direction, which declines a law rather
    /// than applying it where it does not hold.
    fn atom_denotes_a_set(&self, _atom: &Schema) -> Option<bool> {
        None
    }

    /// The kind of the pooled constant behind a [`Schema::Literal`], or `None`
    /// when the bindings decline to kind it.
    ///
    /// A literal denotes `{x | type(x) is type(c) and x == c}`, so its kind is
    /// the kind of `c`'s type and the core cannot read it. Answering places the
    /// literal in the partition, which is what decides it against another kind.
    /// The default declines, so a core with no value oracle stays conservative.
    fn literal_kind(&self, _constant: ConstIx) -> Option<Kind> {
        None
    }

    /// Whether a value of `kind` can be an instance of the class behind a
    /// [`Schema::Instance`], or `None` when the bindings decline to say.
    ///
    /// A class is the one atom the core cannot read at all, and disjointness is
    /// the question it most often needs answered about one: a value of a kind
    /// the class cannot hold is a value the class's set does not contain. The
    /// question is asked this way round, rather than as "what kind is this
    /// class", because a class need not have one -- a class deriving from no
    /// builtin lays down no kind, and *that* is the answer that decides, since
    /// its instances are none of the kinds the partition names.
    ///
    /// `Some(false)` is the only answer that refutes, so an implementor may
    /// answer `Some(true)` for every kind its class could conceivably hold. It
    /// must decline for a class whose `isinstance` runs user code, where
    /// membership is not a property of the value's type at all. The default
    /// declines, so a core with no oracle keeps every class conservative.
    fn class_admits_kind(&self, _class: ClassIx, _kind: Kind) -> Option<bool> {
        None
    }

    /// Whether a value whose *type is* the pooled class has `kind`.
    ///
    /// The narrower question beside [`class_admits_kind`](Self::class_admits_kind),
    /// and the one a refutation can stand on. That one reads the whole subtree
    /// -- a subclass may derive from a builtin, so a plain class *admits* every
    /// kind -- and declines for a plain class because a claim there would be
    /// unsound. This one asks about `type(v) is C` alone, where no subclass can
    /// interfere: a direct instance of a class laying down no builtin layout is
    /// a plain object and has none of the kinds the partition names.
    ///
    /// Both answers are used, so an implementor must not widen either. It must
    /// decline for a class whose `isinstance` runs user code, and for a class
    /// it cannot read.
    fn direct_instance_of_kind(&self, _class: ClassIx, _kind: Kind) -> Option<bool> {
        None
    }

    /// Whether every value of `kind` is an instance of the pooled class.
    ///
    /// The dual of [`direct_instance_of_kind`](Self::direct_instance_of_kind),
    /// asked of the kind's own builtin: a value of `kind` built as that builtin
    /// has `type(v)` equal to it, so the order between the builtin and the
    /// class settles `isinstance`. `Some(false)` refutes -- the kind holds a
    /// value the class does not -- and it is the only answer that does.
    ///
    /// It must decline for a class whose `isinstance` runs user code. The
    /// default declines, so a core with no oracle keeps every class
    /// conservative.
    fn kind_derives_from(&self, _kind: Kind, _class: ClassIx) -> Option<bool> {
        None
    }

    /// Whether the two pooled constants behind a pair of [`Schema::Literal`]s
    /// denote disjoint singletons, or `None` when it cannot be settled soundly.
    ///
    /// Two literals share no value when their constants have different types --
    /// a literal pins `type(x)` exactly, so `Literal[1]` and `Literal[True]` are
    /// disjoint although `1 == True` -- or when the types are the same and the
    /// values differ under an equality the bindings trust. They must decline for
    /// a type carrying user-defined equality, where two distinct constants may
    /// still admit one value. The default declines.
    fn literals_disjoint(&self, _left: ConstIx, _right: ConstIx) -> Option<bool> {
        None
    }
}

/// The trivial [`LeafRelations`] that decides nothing — the core default, under
/// which `Instance` and `Literal` relations stay conservative.
pub struct NoLeafRelations;

impl LeafRelations for NoLeafRelations {
    fn leaf_subtype(&self, _sub: &Schema, _sup: &Schema) -> Option<bool> {
        None
    }
}

/// An oracle that decides nothing reads no pool either, so a schema naming a
/// constant refuses to lower here and is decided by the rules alone.
impl Constants for NoLeafRelations {}

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

/// Whether the intersection contains a schema and its complement (`A ∩ ¬A = ∅`).
///
/// The law is a law **about sets**, and it is applied only where both sides are
/// one. Two atoms are not: the gradual `Any`, whose complement is not its set
/// complement, and an atom that runs a callback -- a predicate is arbitrary code
/// evaluated once per occurrence, so nothing makes the two occurrences agree.
/// A predicate that alternates puts a value in `A` and in `¬A` at once, and the
/// law would report the meet empty with that value as a witness against it.
///
/// This is the completeness law `simplify` applies, decided structurally on the
/// (small) member list. Shared with the simplifier so both read the same lattice
/// law -- and so the simplifier does not rewrite to `nothing` what the decision
/// declines to call empty.
pub(crate) fn has_complementary_pair(members: &[Schema], oracle: &dyn LeafRelations) -> bool {
    has_complementary_pair_within(members, oracle, &[])
}

/// The same, with the definitions a reference in `members` may name.
///
/// A `Ref` is not a set on its own evidence -- what it names is elsewhere -- so
/// with no definitions to read, the fold declines for every recursive schema and
/// `json & ~json` stands. Given them, the reference is resolved and the law
/// applies to a fixpoint like any other set.
pub(crate) fn has_complementary_pair_within(
    members: &[Schema],
    oracle: &dyn LeafRelations,
    definitions: &[Schema],
) -> bool {
    members.iter().any(|member| match member {
        Schema::Complement(inner) => {
            denotes_a_set_within(inner, oracle, definitions)
                && members.iter().any(|other| other == &**inner)
        }
        _ => false,
    })
}

/// How many times a reference is unfolded before the descriptor is asked.
///
/// One. A single unfolding puts the fixpoint's own body in front of the
/// representation, which settles every relation that turns on *what kinds* a
/// recursive schema admits -- a meet with a disjoint kind, an inclusion in a
/// wider union -- and that is the whole of what the structural rules cannot
/// read. Each further unfolding multiplies the schema the descriptor must build
/// against a bound of 64 nodes, for relations nobody has asked for.
const UNFOLDS: u32 = 1;

/// A schema the descriptor can hold, standing in for one that may recurse.
///
/// Borrowed where there is no reference, which is the common case and the one
/// that must cost nothing.
fn unfolded_for<'a>(schema: &'a Schema, defs: &[Schema], positive: bool) -> Cow<'a, Schema> {
    // The empty check first: it is one comparison, and it is true for every
    // schema that carries no fixpoint at all -- which is almost all of them, and
    // all of the ones on the workload the decision budget is measured over. The
    // walk that follows costs a pass over the tree, and paying it per relation
    // for a schema with no definitions was ten percent of that workload.
    if !defs.is_empty() && schema.has_reference() {
        Cow::Owned(schema.unfolded(defs, UNFOLDS, positive))
    } else {
        Cow::Borrowed(schema)
    }
}

/// Whether a schema denotes a *set*: the same values however often it is asked.
///
/// Sound rather than complete, and conservative in the direction that declines.
/// A callback is the atom this rules out: `Predicate` runs user code, so two
/// occurrences of one schema can disagree, and a law that assumes they agree is
/// not a law about this. The gradual `Any` is ruled out because its complement
/// is not its set complement.
///
/// A class is referred to the `oracle`: `isinstance` against a metaclass that
/// overrides `__instancecheck__` is a callback too, and telling a pure class from
/// a hooked one needs the class object, which only the bindings hold.
///
/// A **reference** is read where `definitions` holds what it names, and refused
/// where it does not -- a callback may hide behind a body that is not in hand.
/// Given the body, a reference met again while that body is being walked is
/// *assumed* to be a set: the greatest-fixpoint reading the rest of the
/// recursion uses, and the only one that terminates.
pub(crate) fn denotes_a_set_within(
    schema: &Schema,
    oracle: &dyn LeafRelations,
    definitions: &[Schema],
) -> bool {
    // The root is held beside the worklist rather than inside it. A one-element
    // `vec![...]` is a heap allocation, and most schemas asked this question
    // answer from the root alone -- an atom has no children to defer, and the
    // two refusals below return before reaching any. Seeding the loop this way
    // leaves the worklist empty until a node actually has children, so the
    // common call allocates nothing; the order is the stack's either way, since
    // the root is the only thing the vector held.
    let mut pending: Vec<&Schema> = Vec::new();
    let mut root = Some(schema);
    let mut open: Vec<DefIx> = Vec::new();
    while let Some(node) = root.take().or_else(|| pending.pop()) {
        match node {
            Schema::Ref(index) => {
                if open.contains(index) {
                    continue;
                }
                let Some(body) = definitions.get(index.get()) else {
                    return false;
                };
                open.push(*index);
                pending.push(body);
            }
            Schema::SelfRef(_) => return false,
            // Only the bindings hold the class, so only they can tell a pure one
            // from a hooked one. No answer is the conservative answer.
            Schema::Instance(_) => {
                if oracle.atom_denotes_a_set(node) != Some(true) {
                    return false;
                }
            }
            Schema::Refine { base, constraints } => {
                if constraints
                    .iter()
                    .any(|constraint| matches!(constraint, Constraint::Predicate(_)))
                {
                    return false;
                }
                pending.push(base);
            }
            Schema::Seq { shape, .. } => {
                pending.extend(shape.prefix.iter());
                pending.extend(shape.tail.as_deref());
            }
            Schema::Coll { element: inner, .. } | Schema::Complement(inner) => {
                pending.push(inner);
            }
            Schema::Union(members) | Schema::Intersection(members) => {
                pending.extend(members.iter());
            }
            Schema::KeyedMap { fields, defaults } => {
                pending.extend(fields.iter().map(|field| &field.schema));
                for clause in defaults.iter() {
                    pending.push(&clause.key);
                    pending.push(&clause.value);
                }
            }
            Schema::AttrRecord { fields } => {
                pending.extend(fields.iter().map(|field| &field.schema));
            }
            _ => {}
        }
    }
    true
}

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
/// Whether a direct instance of `class` has none of the kinds `other` admits.
///
/// The kind half of the reading a meet of a class and its attributes gets,
/// asked of a union by distributing over its branches: a union is a value's
/// choice among them, so a value in no branch is outside the union. The
/// witness is one value -- a direct instance of the class carrying the
/// attributes -- and it is the same value in every branch, which is what lets
/// the branches be read one at a time.
///
/// A branch with no kind of its own ends it, since the witness may be in that
/// branch for all this reads, and an empty union is refuted by the rule that
/// reads it as nothing rather than here.
///
/// A reference is unfolded once, to the kind of what it names: it denotes that
/// set, so it has that set's kind, and without the unfolding a recursive record
/// reaches this reading as a node carrying no kind at all.
///
/// Once, and not through what it finds. The walk then descends only into union
/// branches, which is the schema's own finite tree, so it ends without a trail
/// to keep or a budget to spend. Unfolding onwards would buy a reference naming
/// a union, and would owe termination an argument about cycles -- and the only
/// way to build a cycle here is a chain of references and unions with no
/// constructor between them, which is the shape a repeated *goal* is assumed
/// through long before this reading is reached.
fn outside_every_kind(other: &Schema, class: ClassIx, cx: SubtypeCx<'_>) -> bool {
    let outside = |kind| cx.oracle.direct_instance_of_kind(class, kind) == Some(false);
    match other {
        Schema::Union(branches) => {
            !branches.is_empty()
                && branches
                    .iter()
                    .all(|branch| outside_every_kind(branch, class, cx))
        }
        Schema::Ref(at) => cx
            .defs
            .get(at.get())
            .and_then(|body| body.type_tag_with(cx.oracle))
            .is_some_and(outside),
        _ => other.type_tag_with(cx.oracle).is_some_and(outside),
    }
}

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
