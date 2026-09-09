//! The decision procedures over the IR: emptiness, subtyping, equivalence, and
//! disjointness, with the leaf-relation oracle and the scalar region partition.

use crate::descr::lower::{Constants, lower};
use crate::ir::{
    CollKind, ConstIx, Constraint, DefIx, Field, MapClause, OperandIx, Schema, SeqKind, SeqShape,
};
use rustc_hash::FxHashMap;
use std::borrow::Cow;
use std::cell::Cell;

/// What a decision could establish about a *relation* between two schemas.
///
/// The companion of [`Verdict`], which says what could be established about one
/// schema's set. A `bool` answer to "is this a subtype of that" conflates the
/// two things a caller most needs to tell apart: a value of the left that is
/// outside the right, which refutes the relation, and a rule that declined --
/// an oracle with no answer, a constructor pair no rule relates, a descent the
/// work bound stopped. Both read as `false`, so the conservative half of the
/// procedure is invisible from outside and cannot be counted, listed, or held
/// to a ledger.
///
/// The public relations still answer `bool`, because that is what the contract
/// promises: [`Relation::Holds`] is `true` and both other answers are `false`.
/// What the three values buy is that "not proven" is a value the tests can
/// count and the pages can enumerate.
///
/// A rule that reports [`Relation::Fails`] is claiming a proof. Where a rule is
/// sound but incomplete -- it establishes the relation when it fires and says
/// nothing when it does not -- the answer is [`Relation::Unknown`], and the
/// combinators below propagate that: a conjunction is unknown when a conjunct
/// is, and a disjunction is unknown when no disjunct holds and one is unknown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Relation {
    /// Proven: every value of the subject belongs to the other schema.
    Holds,
    /// Refuted: a value of the subject is outside the other schema.
    ///
    /// A witness is what makes this answer. The descriptor's reading produces
    /// one directly, by proving the difference holds a value; the structural
    /// rules produce one by finding a mismatch of shapes, which stands on
    /// *some* value of the subject and so is read against the subject's own
    /// emptiness before it is believed.
    Fails,
    /// Neither, and the procedure says so rather than answering `false`.
    Unknown,
}

impl Relation {
    /// The relation as the boundary reports it: only a proof is `true`.
    #[must_use]
    #[inline]
    pub fn holds(self) -> bool {
        matches!(self, Relation::Holds)
    }

    /// A relation from a decision that is exact in both directions.
    ///
    /// For a rule whose `false` is a refutation rather than a decline -- set
    /// inclusion between two region sets, an arity that cannot match. A rule
    /// that is sound one way and silent the other spells its own arms.
    #[must_use]
    #[inline]
    fn decided(held: bool) -> Relation {
        if held {
            Relation::Holds
        } else {
            Relation::Fails
        }
    }

    /// A relation from a rule that proves inclusion and declines otherwise.
    #[must_use]
    #[inline]
    fn proven(held: bool) -> Relation {
        if held {
            Relation::Holds
        } else {
            Relation::Unknown
        }
    }

    /// Both must hold. The first answer that is not a proof is the answer:
    /// once the conjunction cannot hold, what is left to learn is whether it is
    /// refuted or merely unproven, and paying for that would cost the descent
    /// its early exit. Reporting the weaker of the two is the safe direction.
    #[must_use]
    #[inline]
    fn and(self, other: impl FnOnce() -> Relation) -> Relation {
        match self {
            Relation::Holds => other(),
            answer => answer,
        }
    }

    /// The proof this answer carries, with its refutation dropped.
    ///
    /// For a step that is sound in one direction only. Reducing a refinement to
    /// its base is the case: a refinement is a *subset* of its base, so the
    /// base's inclusion carries it -- but the base's refutation does not, since
    /// the value that puts the base outside may be one the constraints exclude.
    /// `list[int]` is not below the empty list and `list[int]` of length at most
    /// zero is, and both are read from the same base.
    #[must_use]
    #[inline]
    fn proof_only(self) -> Relation {
        match self {
            Relation::Holds => Relation::Holds,
            _ => Relation::Unknown,
        }
    }

    /// The refutation a mismatch carries, read against what the subject holds.
    ///
    /// A rule refutes by finding a mismatch -- two arities that cannot align, a
    /// key one side requires and the other does not guarantee -- and names the
    /// value that stands against the inclusion only implicitly: it is *some*
    /// value of the subject, shaped the way the subject says. A subject with no
    /// value names none, and the empty set is below every set, the shape it can
    /// never take included. So the mismatch is read against the subject: proved
    /// inhabited it refutes, proved empty it establishes the opposite, and
    /// undecided it decides nothing.
    #[must_use]
    #[inline]
    const fn of_mismatch(subject: Verdict) -> Relation {
        match subject {
            Verdict::Inhabited => Relation::Fails,
            Verdict::Empty => Relation::Holds,
            Verdict::Unknown => Relation::Unknown,
        }
    }

    /// The second decider, asked only where the first declines.
    ///
    /// Two procedures answer inclusion here: the structural rules and the
    /// difference the descriptor builds. A proof from either is a proof, and a
    /// *refutation* from either is a refutation -- so the only answer worth a
    /// second opinion is `Unknown`. Reaching for the second decider after the
    /// first has refuted would be asking a question already answered, and
    /// discarding the answer it gave.
    #[must_use]
    #[inline]
    fn or_else(self, second: impl FnOnce() -> Relation) -> Relation {
        match self {
            Relation::Unknown => second(),
            answer => answer,
        }
    }

    /// The inclusion a difference decides: `a <= b` is `a & ~b = {}`.
    ///
    /// The whole of the descriptor's reading, named where the two vocabularies
    /// meet. An empty difference proves the inclusion; a difference proved to
    /// hold a value refutes it, because that value is in `a` and outside `b`;
    /// and a difference the descriptor cannot decide decides nothing.
    #[must_use]
    #[inline]
    const fn of_difference(difference: Verdict) -> Relation {
        match difference {
            Verdict::Empty => Relation::Holds,
            Verdict::Inhabited => Relation::Fails,
            Verdict::Unknown => Relation::Unknown,
        }
    }

    /// Every item, with [`Relation::and`]'s short circuit.
    #[inline]
    fn all(items: impl IntoIterator<Item = Relation>) -> Relation {
        for answer in items {
            if answer != Relation::Holds {
                return answer;
            }
        }
        Relation::Holds
    }

    /// Any item, with [`Relation::or`]'s propagation of a decline.
    #[inline]
    fn any(items: impl IntoIterator<Item = Relation>) -> Relation {
        let mut declined = false;
        for answer in items {
            match answer {
                Relation::Holds => return Relation::Holds,
                Relation::Unknown => declined = true,
                Relation::Fails => {}
            }
        }
        if declined {
            Relation::Unknown
        } else {
            Relation::Fails
        }
    }
}

/// The most decision steps one top-level query may take before it stops and
/// returns the conservative answer. Subtyping distributes over unions and
/// intersections and emptiness recurses the structural fragment, so a deeply
/// nested Boolean combination can demand work exponential in its depth; without
/// interning to share equal subtrees there is no cheap memo, so the procedure
/// bounds its own work.
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
/// **This bound is debt.** Regularity bounds the number of distinct subtyping
/// goals a query can reach, so a memo over goals terminates by a theorem rather
/// than by a ceiling -- but a memo needs a cheap key, and a key is cheap only
/// when structurally equal subtrees are one node. Sharing the nodes is what
/// makes the memo possible and this constant removable; until then the ceiling
/// stands in for the argument.
///
/// The ceiling is far above any schema a real
/// annotation produces, so a legitimate relation is always decided; only an
/// adversarial schema built to blow up the decision reaches it, and there a
/// `false` ("not proven") is sound by the conservative contract. A complete,
/// work-sharing decision is the interning-based procedure.
pub(crate) const DECISION_BUDGET: u32 = 1_000_000;

/// Spend one unit of `budget`; returns `false` when it is already exhausted, the
/// signal a budgeted decision uses to stop and report the conservative answer.
fn spend(budget: &Cell<u32>) -> bool {
    match budget.get().checked_sub(1) {
        Some(remaining) => {
            budget.set(remaining);
            true
        }
        None => false,
    }
}

/// What a decision could establish about a set.
///
/// A `bool` answer conflates two different things. `is_empty` returning `false`
/// means "not proven empty", which covers a schema proven to admit values and a
/// schema the procedure gave up on -- an opaque leaf, or a descent the work bound
/// stopped. The caller cannot tell them apart, and neither can an instrument
/// watching from outside, so a budget exhaustion at a realistic size reads as a
/// confident answer.
///
/// The public relations still answer `bool`, because that is what soundness
/// promises: `Unknown` and `Inhabited` both mean "not proven empty". What the
/// three values buy is that the difference is now *visible* -- to a test, to a
/// gate, and to the memoisation that will make it rarer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// Proven to admit no value.
    Empty,
    /// Proven to admit at least one value.
    Inhabited,
    /// Neither: an opaque leaf the core cannot read, or a descent the work bound
    /// stopped before it reached one.
    Unknown,
}

impl Verdict {
    /// Whether this verdict proves emptiness. The reduction the public relations
    /// make, named once: `Unknown` is not a proof, so it answers with
    /// `Inhabited`.
    pub(crate) const fn is_empty(self) -> bool {
        matches!(self, Verdict::Empty)
    }

    /// The verdict for a value that must satisfy **every** part: a product, a
    /// meet of positions, a record's required fields.
    ///
    /// One empty part empties the whole, whatever the others are, so `Empty`
    /// absorbs. Otherwise every part must be proven inhabited for the whole to
    /// be, and one `Unknown` leaves it unknown. An empty iterator is `Inhabited`:
    /// nothing is required, so the empty value satisfies it.
    pub(crate) fn every(parts: impl Iterator<Item = Verdict>) -> Verdict {
        let mut verdict = Verdict::Inhabited;
        for part in parts {
            match part {
                Verdict::Empty => return Verdict::Empty,
                Verdict::Unknown => verdict = Verdict::Unknown,
                Verdict::Inhabited => {}
            }
        }
        verdict
    }

    /// The verdict for a value that may satisfy **any** part: a union.
    ///
    /// The dual of [`every`](Self::every). One inhabited part inhabits the whole,
    /// so `Inhabited` absorbs; every part must be proven empty for the whole to
    /// be. An empty iterator is `Empty`, which is what a union of no members
    /// denotes.
    pub(crate) fn any(parts: impl Iterator<Item = Verdict>) -> Verdict {
        let mut verdict = Verdict::Empty;
        for part in parts {
            match part {
                Verdict::Inhabited => return Verdict::Inhabited,
                Verdict::Unknown => verdict = Verdict::Unknown,
                Verdict::Empty => {}
            }
        }
        verdict
    }
}

/// The region set a schema denotes, or `Unknown` where it is not
/// scalar-decidable.
///
/// A monoid under each lattice operation, with `Unknown` **absorbing** both: an
/// opaque member makes the whole combination opaque, whatever the others say.
/// Naming the absorbing element is what lets a fold over members stop at it --
/// past that point no later member can change the result, and a walk that
/// continues spends the decision budget on an answer already fixed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Regions {
    /// The exact set of regions, on the scalar-decidable fragment.
    Known(Region),
    /// Not scalar-decidable, so the regions are unknown rather than empty.
    Unknown,
}

impl Regions {
    /// The identity of [`union`](Self::union): the empty region set.
    pub(crate) const UNION_UNIT: Regions = Regions::Known(Region::EMPTY);
    /// The identity of [`intersect`](Self::intersect): every region.
    pub(crate) const MEET_UNIT: Regions = Regions::Known(Region::ALL);

    /// Every region in either set, opaque if either side is.
    ///
    /// A member naming every region would settle a union whatever the others
    /// are, and saying so here was tried: it makes the two folds that walk a
    /// union disagree, because one stops at the first opaque member and the
    /// other does not. Reading on instead of stopping costs a recursive region
    /// walk per member and half again the decision budget. So the accumulator
    /// stays opaque once any member is, and a universe spelled with an opaque
    /// member beside `Anything` is left undecided rather than paid for.
    pub(crate) fn union(self, other: Regions) -> Regions {
        match (self, other) {
            (Regions::Known(a), Regions::Known(b)) => Regions::Known(a.union(b)),
            _ => Regions::Unknown,
        }
    }

    /// Every region in both sets, opaque if either side is.
    ///
    /// Opaque for the same reason [`union`](Self::union) is.
    pub(crate) fn intersect(self, other: Regions) -> Regions {
        match (self, other) {
            (Regions::Known(a), Regions::Known(b)) => Regions::Known(a.intersect(b)),
            _ => Regions::Unknown,
        }
    }

    /// Whether this value absorbs both operations, so no further member can
    /// change the result and a fold over them may stop here.
    ///
    pub(crate) const fn is_absorbing(self) -> bool {
        matches!(self, Regions::Unknown)
    }

    /// The verdict a known region set settles by itself.
    ///
    /// A region set is held only when the schema names it *exactly*, so an empty
    /// one is a proof of emptiness and a non-empty one is a proof of inhabitance.
    /// That is the whole payoff of the exactness condition
    /// [`Schema::atom_region`] carries: on the scalar-decidable fragment the fold
    /// answers both directions, not just the one.
    pub(crate) const fn verdict(self) -> Verdict {
        match self {
            Regions::Known(regions) if regions.is_empty() => Verdict::Empty,
            Regions::Known(_) => Verdict::Inhabited,
            Regions::Unknown => Verdict::Unknown,
        }
    }

    /// The regions, where they are known.
    pub(crate) const fn known(self) -> Option<Region> {
        match self {
            Regions::Known(regions) => Some(regions),
            Regions::Unknown => None,
        }
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
        match (self.type_tag_with(oracle), other.type_tag_with(oracle)) {
            // Distinct concrete types are disjoint, except bool ⊆ int.
            (Some(a), Some(b)) => {
                a != b && !matches!((a, b), (Kind::Bool, Kind::Int) | (Kind::Int, Kind::Bool))
            }
            _ => false,
        }
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

    /// Whether this schema is provably empty — denotes no value. Complete on the
    /// scalar fragment (every Boolean combination of scalar atoms) and on the
    /// structural fragment reached here — a sequence whose regex matches no
    /// sequence, a keyed map with an impossible required field, and a union of
    /// empties — and sound everywhere else: it never reports a non-empty schema
    /// as empty. A set or frozenset is never empty (the empty collection is
    /// always a member). To resolve recursive references, use
    /// [`is_empty_under`](Self::is_empty_under).
    ///
    /// Where the rules decline, the question is asked again of the *set* the
    /// schema denotes, under a bound on what building it may cost, which
    /// decides a container meet, a double complement and a kind against its own
    /// literals. What is left undecided is what no bounded descriptor holds: an
    /// unresolved recursive reference, a predicate, and a schema past one of the
    /// build's bounds.
    ///
    /// The decision is bounded: a deeply nested adversarial schema that would take
    /// more than a fixed number of steps stops and returns `false`, so a `false`
    /// means "not proven empty within the work bound", not necessarily "non-empty".
    /// A real schema decides far inside the bound. The scalar-region check is folded
    /// bottom-up from each node's children, so nested Boolean structure is decided
    /// in time linear in its size rather than by re-walking each subtree per level.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.is_empty_with(&NoLeafRelations, &[])
    }

    /// Like [`is_empty`](Self::is_empty), but resolving recursive references
    /// through `defs`, so an uninhabited recursive schema — a mandatory
    /// self-reference with no base case — is detected. A reference `defs` does
    /// not resolve stays conservative (never reported empty).
    #[must_use]
    pub fn is_empty_under(&self, defs: &[Schema]) -> bool {
        self.is_empty_with(&NoLeafRelations, defs)
    }

    /// Like [`is_empty_under`](Self::is_empty_under), but with an `oracle` that
    /// can order the pool values behind refinement bounds, so an unsatisfiable
    /// bound conjunction (a lower bound above an upper bound) is detected.
    #[must_use]
    pub fn is_empty_with(&self, oracle: &dyn LeafRelations, defs: &[Schema]) -> bool {
        self.is_empty_rec(oracle, defs, &mut Vec::new(), &Cell::new(DECISION_BUDGET))
            || self.denotes_no_value(oracle, defs)
    }

    /// Whether the descriptor proves this schema admits no value.
    ///
    /// Asked **after** the structural rules and only where they decline, which
    /// is the ordering the cost forces: building a descriptor determinises
    /// automata and takes products, and beside a verdict the rules already
    /// reached that work is discarded. Since it can only turn a `false` into a
    /// `true`, both orders give the same answers and only one is affordable.
    ///
    /// Asking costs a bounded amount whether or not it answers: the build is
    /// held to the nodes it may read, the nesting it may descend and the work it
    /// may spend, and past any of those it refuses. A schema the descriptor
    /// cannot hold -- a recursive one -- refuses the same way. Either way the
    /// caller keeps the verdict the rules reached.
    fn denotes_no_value(&self, pool: &dyn Constants, defs: &[Schema]) -> bool {
        lower(&unfolded_for(self, defs, true), pool)
            .is_some_and(|set| set.emptiness() == Verdict::Empty)
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

    /// The decision steps [`is_empty`](Self::is_empty) spends on this schema.
    ///
    /// The instrument for the tests that pin a complexity bound. A wall-clock
    /// assertion measures the machine as much as the algorithm: it passes on a
    /// quiet laptop and fails on a loaded runner for reasons that have nothing
    /// to do with the code. The step count is the quantity the bound is actually
    /// about, and it is the same number on every machine.
    #[cfg(test)]
    pub(crate) fn empty_steps(&self) -> u32 {
        let budget = Cell::new(DECISION_BUDGET);
        self.is_empty_rec(&NoLeafRelations, &[], &mut Vec::new(), &budget);
        DECISION_BUDGET - budget.get()
    }

    /// The decision steps [`is_subtype_of`](Self::is_subtype_of) spends, by the
    /// same argument as [`empty_steps`](Self::empty_steps).
    #[cfg(test)]
    pub(crate) fn subtype_steps(&self, other: &Schema) -> u32 {
        let budget = Cell::new(DECISION_BUDGET);
        self.is_subtype_rec(
            other,
            SubtypeCx {
                oracle: &NoLeafRelations,
                defs: &[],
                budget: &budget,
            },
            &mut Vec::new(),
        );
        DECISION_BUDGET - budget.get()
    }

    fn is_empty_rec(
        &self,
        oracle: &dyn LeafRelations,
        defs: &[Schema],
        visiting: &mut Vec<DefIx>,
        budget: &Cell<u32>,
    ) -> bool {
        self.verdict_rec(oracle, defs, visiting, budget).is_empty()
    }

    /// What this schema's emptiness can be proven to be, under the leaf oracle
    /// and the recursive definitions.
    ///
    /// The three-valued answer behind [`is_empty`](Self::is_empty), which reduces
    /// it: an `Unknown` is not a proof of emptiness, so the public relation
    /// answers `false` for it exactly as it does for `Inhabited`. What it is for
    /// is telling those two apart from outside -- an exhausted budget is
    /// `Unknown`, and a test can say so.
    #[must_use]
    pub fn verdict(&self) -> Verdict {
        self.verdict_rec(
            &NoLeafRelations,
            &[],
            &mut Vec::new(),
            &Cell::new(DECISION_BUDGET),
        )
    }

    fn verdict_rec(
        &self,
        oracle: &dyn LeafRelations,
        defs: &[Schema],
        visiting: &mut Vec<DefIx>,
        budget: &Cell<u32>,
    ) -> Verdict {
        self.empty_and_region(oracle, defs, visiting, budget).0
    }

    /// The emptiness verdict and the value-region bitset of `self`, decided in a
    /// single bottom-up pass: a Boolean node folds its children's already-computed
    /// regions in O(1) each instead of re-deriving its region by re-walking the
    /// whole subtree with [`region_set`](Self::region_set). The returned bitset is
    /// exactly what `region_set` would return (`None` off the scalar-decidable
    /// fragment), so emptiness on the scalar fragment is decided identically, and
    /// a deeply nested intersection is decided in time linear in its size: each
    /// level folds its children's regions without re-walking the levels below.
    ///
    /// The work is bounded by the shared `budget`, so the region computation cannot
    /// run unbounded down a side door any more than the rest of the decision can;
    /// on exhaustion it returns the conservative "not proven empty" with an unknown
    /// region.
    fn empty_and_region(
        &self,
        oracle: &dyn LeafRelations,
        defs: &[Schema],
        visiting: &mut Vec<DefIx>,
        budget: &Cell<u32>,
    ) -> (Verdict, Regions) {
        // Bound the work, sharing the budget with the caller (the subtyping
        // decision passes its own `cx.budget` in), so emptiness cannot escape the
        // ceiling subtyping advertises. Exhaustion proves nothing either way,
        // which is what `Unknown` says and what a `false` could not.
        if !spend(budget) {
            return (Verdict::Unknown, Regions::Unknown);
        }
        // A scalar atom names its region exactly, so the region settles it. Read
        // before the match so the mapping from atom to region is written once,
        // beside the exactness it depends on.
        if let Some(region) = self.atom_region() {
            let regions = Regions::Known(region);
            return (regions.verdict(), regions);
        }
        match self {
            // The lattice bounds carry a known region, which settles them the
            // same way.
            Schema::Nothing => (Verdict::Empty, Regions::Known(Region::EMPTY)),
            Schema::Anything(_) => (Verdict::Inhabited, Regions::Known(Region::ALL)),
            Schema::Ref(id) => {
                // A reference reached again while resolving it is a cycle: this
                // occurrence demands an infinite unfolding, so on its own it has
                // no finite inhabitant. A union base case or an optional or
                // starred position escapes before reaching here.
                if visiting.contains(id) {
                    return (Verdict::Empty, Regions::Unknown);
                }
                match defs.get(id.get()) {
                    Some(def) => {
                        visiting.push(*id);
                        let verdict = def.verdict_rec(oracle, defs, visiting, budget);
                        visiting.pop();
                        (verdict, Regions::Unknown)
                    }
                    // A reference no definition resolves says nothing about the
                    // set it names.
                    None => (Verdict::Unknown, Regions::Unknown),
                }
            }
            // A sequence admits no value when a prefix element admits none. A
            // tail repeats zero times, so a shape whose prefix is all inhabited
            // admits at least the sequence that stops at the prefix.
            Schema::Seq { shape, .. } => {
                let prefix = shape
                    .prefix
                    .iter()
                    .map(|element| element.verdict_rec(oracle, defs, visiting, budget));
                (Verdict::every(prefix), Regions::Unknown)
            }
            // A refinement is a subset of its base: an empty base empties it, and
            // so does an unsatisfiable bound conjunction (decided by the oracle).
            // A refinement with no constraint denotes exactly its base, so it
            // earns the base's verdict *and* its regions. One with a constraint
            // narrows, and a narrowed region set is an over-approximation whose
            // complement would report an inhabited schema empty -- which is why
            // every other refinement stays unknown.
            //
            // The two spellings of the universe were decided differently without
            // this: `Anything` is below `Refine { base: Anything }` through the
            // refinement rule, and the gradual `Any` is below `Anything` but was
            // not below the refinement, because only a region set says so.
            Schema::Refine { base, constraints } if constraints.is_empty() => {
                base.empty_and_region(oracle, defs, visiting, budget)
            }
            Schema::Refine { base, constraints } => {
                let int_discrete = bounded_to_the_integers([base.as_ref()]);
                let empty = base.is_empty_rec(oracle, defs, visiting, budget)
                    || bounds_unsatisfiable(constraints.iter(), oracle, int_discrete);
                // A satisfiable bound over an inhabited base is not a proof of
                // inhabitance: the constraints narrow the base, and whether any
                // value survives them is what the bounds check declines to say.
                (
                    if empty {
                        Verdict::Empty
                    } else {
                        Verdict::Unknown
                    },
                    Regions::Unknown,
                )
            }
            Schema::Intersection(members) => {
                intersection_verdict(members, oracle, defs, visiting, budget)
            }
            // A set or frozenset admits the empty collection whatever its
            // element schema is, so it is *proven* inhabited -- which two values
            // could not say apart from the opaque wildcard below, where the same
            // `false` meant only that nothing proved emptiness.
            Schema::Coll { .. } => (Verdict::Inhabited, Regions::Unknown),
            // A map is emptied by a required field that admits nothing, and
            // admits the empty dict when it requires nothing at all.
            Schema::KeyedMap { fields, .. } => {
                let required = fields.iter().filter(|field| field.required);
                let verdict = Verdict::every(
                    required.map(|field| field.schema.verdict_rec(oracle, defs, visiting, budget)),
                );
                (verdict, Regions::Unknown)
            }
            // An attribute record carries no class, so its fields decide it in
            // both directions -- the same rule as a keyed map, and for the same
            // reason. A required field admitting nothing empties it, because no
            // value can carry that attribute; required fields that are all
            // *proven* inhabited make it inhabited, because an object carrying
            // one witness per attribute is a value of the record. That second
            // half is what the class half used to take away: an `isinstance`
            // atom is opaque, so a node holding both could never be more than
            // unknown.
            Schema::AttrRecord { fields } => {
                let required = fields.iter().filter(|field| field.required);
                let verdict = Verdict::every(
                    required.map(|field| field.schema.verdict_rec(oracle, defs, visiting, budget)),
                );
                (verdict, Regions::Unknown)
            }
            // A union is empty when every member is and inhabited when any member
            // is; its region is the union of the members' regions, again folded
            // from the children.
            Schema::Union(members) => {
                let mut verdict = Verdict::Empty;
                let mut region = Regions::UNION_UNIT;
                for m in members.iter() {
                    let (member, member_region) =
                        m.empty_and_region(oracle, defs, visiting, budget);
                    verdict = Verdict::any([verdict, member].into_iter());
                    region = region.union(member_region);
                    // A member that is not proven empty and an opaque region are
                    // both absorbing: no later member can make the union empty,
                    // and none can make the region known. The stop is on "not
                    // proven empty" rather than "proven inhabited", which is
                    // where the old two-valued fold stopped -- reading further to
                    // strengthen an `Unknown` into an `Inhabited` would spend
                    // budget on a question the caller does not ask.
                    if verdict != Verdict::Empty && region.is_absorbing() {
                        break;
                    }
                }
                (verdict, region)
            }
            // A complement's region is the partition minus its inner's region; it is
            // empty exactly when that region is empty (`¬⊤ = ∅`).
            Schema::Complement(inner) => {
                let (_, inner_region) = inner.empty_and_region(oracle, defs, visiting, budget);
                let region = match inner_region {
                    Regions::Known(regions) => Regions::Known(regions.complement()),
                    Regions::Unknown => Regions::Unknown,
                };
                (region.verdict(), region)
            }
            // The gradual `Any`, literals, and instances are not scalar-decidable
            // and the core cannot read them: a literal's constant may be `nan`,
            // which is equal to nothing and denotes the empty set, and a class may
            // have no instances. Neither direction is proven.
            _ => (Verdict::Unknown, Regions::Unknown),
        }
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
    /// two a boundary reports. Held inside the crate rather than published,
    /// because the reduction is what makes the third value safe to ignore: a
    /// refutation the *set* reading makes is a difference it proved inhabited,
    /// and its components do not relate a class to the shape its instances
    /// have, so it reports a named tuple of two integers outside `tuple[int,
    /// int]`. A boundary that reported that as a refutation would state
    /// something false; one that reports it as "not proven" states what the
    /// procedure knows.
    fn subtype_relation_under(
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
        self.witnessed(self.is_subtype_rec(other, cx, &mut Vec::new()), cx)
    }

    /// One rule's answer about this subject, with a refutation believed only
    /// where the subject has a value to stand on.
    ///
    /// Every refutation the rules reach is a mismatch of shapes, and every
    /// composition that carries one preserves that: a conjunction hands on the
    /// refutation of a part, and the one disjunction that could invent a
    /// refutation out of parts each failing for a different reason is read for
    /// its proof alone. So one reading of the subject at the top settles them
    /// all, and it is taken once per query rather than once per rule.
    fn witnessed(&self, answer: Relation, cx: SubtypeCx<'_>) -> Relation {
        if answer == Relation::Fails {
            return Relation::of_mismatch(self.verdict_rec(
                cx.oracle,
                cx.defs,
                &mut Vec::new(),
                cx.budget,
            ));
        }
        answer
    }

    fn is_subtype_rec(
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
    fn bounds_the_pair(&self, supertype_regions: Regions, cx: SubtypeCx<'_>) -> bool {
        // `other` covers the universe exactly when its region set is the whole
        // partition. The set is the caller's -- `is_subtype_rec` reads it for the
        // exact scalar rule and nothing between there and here changes `other` --
        // so it arrives as an argument rather than being derived twice.
        if supertype_regions == Regions::Known(Region::ALL) {
            return true;
        }
        self.is_empty_rec(cx.oracle, cx.defs, &mut Vec::new(), cx.budget)
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
    /// Every rule here proves inclusion when it fires and says nothing when it
    /// does not, so a union none of them places the subject in is a pair this
    /// procedure declines rather than one it refutes.
    fn below_a_union(
        &self,
        other: &Schema,
        members: &[Schema],
        cx: SubtypeCx<'_>,
        assumptions: &mut Vec<(Schema, Schema)>,
    ) -> Relation {
        Relation::proven(
            members.contains(self)
                || Relation::any(
                    members
                        .iter()
                        .map(|m| self.is_subtype_rec(m, cx, assumptions)),
                )
                .holds()
                || seq_splits_across_union(self, members, cx, assumptions)
                || self.left_reduces_below(other, cx, assumptions).holds()
                // Last, and only for the one subject the oracle can answer about
                // here: an `Instance` whose *values* the bindings can enumerate
                // is below a union when each of them is, which is what makes an
                // enumeration and the union of its members one set. Asked here
                // because a union on the right never reaches the leaf arm, and
                // asked for nothing else because every other subject would pay a
                // call that always declines -- ten percent of the decision
                // workload, measured.
                || (matches!(self, Schema::Instance(_))
                    && cx.oracle.leaf_subtype(self, other).unwrap_or(false)),
        )
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
        Relation::proven(
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
        )
    }

    fn subtype_decide(
        &self,
        other: &Schema,
        supertype_regions: Regions,
        cx: SubtypeCx<'_>,
        assumptions: &mut Vec<(Schema, Schema)>,
    ) -> Relation {
        match (self, other) {
            // Every lattice bound, in one arm: `∅ ⊆ B`, `A ⊆ U`, and `A ⊆ ∅`
            // when A is empty. All three are the same question asked of
            // emptiness, so one guard answers them and a per-atom arm beside it
            // would be dead code -- the mutation sweep says so, by surviving its
            // deletion.
            _ if self.bounds_the_pair(supertype_regions, cx) => Relation::Holds,
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
            (Schema::Intersection(members), _) => self.meet_below(other, members, cx, assumptions),
            (_, Schema::Union(members)) => self.below_a_union(other, members, cx, assumptions),
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
            // Disjointness is proven or not proven, never refuted here: the
            // rules that establish it are sound and incomplete.
            (_, Schema::Complement(inner)) => {
                Relation::proven(self.shares_no_value_with(inner, cx))
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
            ),
            // Against a non-refinement, a refinement inherits its base's supertypes.
            (Schema::Refine { .. }, _) => self.left_reduces_below(other, cx, assumptions),
            // A leaf the structural rules cannot relate (an instance or literal):
            // defer to the oracle, conservative when it declines.
            // A leaf pair the structural rules cannot relate: the oracle
            // answers, and its `None` is the decline it says it is.
            _ => match cx.oracle.leaf_subtype(self, other) {
                Some(true) => Relation::Holds,
                Some(false) => Relation::Fails,
                None => Relation::Unknown,
            },
        }
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
            sub.witnessed(sub.is_subtype_rec(sup, cx, &mut Vec::new()), cx)
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

/// Whether the values a meet admits are bounded to the integers, so a bound
/// conjunction over it may count them.
///
/// Asked by both meets the IR carries -- a refinement's constraint conjunction and
/// an [`Schema::Intersection`] of refinements -- so a rule that fires for one
/// fires for the other. An intersection is a subset of every member, so one
/// member bounded to the integers bounds the whole meet; a lone base bounds it by
/// being one. `bool` counts because it subclasses `int`, and a float base does
/// not because the reals between two bounds are dense.
///
/// Sound and not complete for `bool`: the rule counts the integers in the
/// interval rather than the two values a boolean has, so an interval holding an
/// integer that is neither zero nor one stays conservatively inhabited.
fn bounded_to_the_integers<'a>(bases: impl IntoIterator<Item = &'a Schema>) -> bool {
    bases
        .into_iter()
        .any(|base| matches!(base.type_tag(), Some(Kind::Int | Kind::Bool)))
}

/// The verdict and region set of an intersection.
///
/// Five rules can prove a meet empty and they are asked in one place: an empty
/// member, cancelling scalar regions, a member beside its own complement, two
/// members of disjoint kinds, refinement bounds that cannot hold together, and a
/// required key two record members cannot agree on.
///
/// Inhabitance has no such rule. Proving a meet inhabited means finding a value
/// in *every* member, which none of the five does, so it is the region that
/// settles it -- exactly, over the whole scalar fragment -- and past that the
/// meet is opaque however inhabited its members are.
fn intersection_verdict(
    members: &[Schema],
    oracle: &dyn LeafRelations,
    defs: &[Schema],
    visiting: &mut Vec<DefIx>,
    budget: &Cell<u32>,
) -> (Verdict, Regions) {
    let mut any_empty = false;
    let mut region = Regions::MEET_UNIT;
    for m in members {
        let (verdict, member_region) = m.empty_and_region(oracle, defs, visiting, budget);
        any_empty |= verdict.is_empty();
        region = region.intersect(member_region);
        // Both accumulators absorb: an empty member empties the meet whatever the
        // rest are, and an opaque region stays opaque. No later member can change
        // either, so the walk stops rather than spending the budget on an answer
        // already fixed.
        if any_empty && region.is_absorbing() {
            break;
        }
    }
    let empty = any_empty
        || region.known().is_some_and(Region::is_empty)
        || has_complementary_pair(members, oracle)
        || has_disjoint_pair(members, oracle)
        || intersection_bounds_unsatisfiable(members, oracle)
        || keyed_map_meet_empty(members, oracle, defs, budget);
    let verdict = if empty {
        Verdict::Empty
    } else {
        region.verdict()
    };
    (verdict, region)
}

/// Whether a refinement's bound and length constraints cannot hold together: a
/// required minimum length above the allowed maximum, or a numeric lower bound
/// above the upper bound (or equal with a strict end). Sound: it reports
/// unsatisfiable only when the ordering the oracle returns forces it, and stays
/// conservative when the oracle cannot compare two bounds.
fn bounds_unsatisfiable<'a>(
    constraints: impl Iterator<Item = &'a Constraint> + Clone,
    oracle: &dyn LeafRelations,
    int_discrete: bool,
) -> bool {
    use core::cmp::Ordering;
    let min_len = constraints
        .clone()
        .filter_map(|c| match c {
            Constraint::MinLen(n) => Some(*n),
            _ => None,
        })
        .max();
    let max_len = constraints
        .clone()
        .filter_map(|c| match c {
            Constraint::MaxLen(n) => Some(*n),
            _ => None,
        })
        .min();
    if let (Some(lo), Some(hi)) = (min_len, max_len)
        && lo > hi
    {
        return true;
    }
    let mut lower: Option<(OperandIx, bool)> = None;
    let mut upper: Option<(OperandIx, bool)> = None;
    for constraint in constraints {
        match constraint {
            Constraint::Ge(i) => lower = Some(tighter_bound(lower, (*i, false), oracle, true)),
            Constraint::Gt(i) => lower = Some(tighter_bound(lower, (*i, true), oracle, true)),
            Constraint::Le(i) => upper = Some(tighter_bound(upper, (*i, false), oracle, false)),
            Constraint::Lt(i) => upper = Some(tighter_bound(upper, (*i, true), oracle, false)),
            _ => {}
        }
    }
    if let (Some((lo, lo_strict)), Some((hi, hi_strict))) = (lower, upper) {
        match oracle.compare(lo, hi) {
            Some(Ordering::Greater) => return true,
            Some(Ordering::Equal) => return lo_strict || hi_strict,
            _ => {}
        }
        // An integer-discrete base bounds the integers in the interval, so the
        // refinement is empty when no integer lies between the bounds even though
        // the endpoints themselves are ordered `lo < hi` — `Annotated[int, Gt(0),
        // Lt(1)]` admits no value. The oracle answers only for a real numeric
        // pair and stays `None` otherwise, so floats and incomparable bounds keep
        // the interval conservatively non-empty.
        if int_discrete && oracle.no_int_between(lo, lo_strict, hi, hi_strict) == Some(true) {
            return true;
        }
    }
    false
}

/// Whether a single supertype refinement constraint is *entailed* by the subtype's
/// constraint set: every value satisfying all of `narrow` also satisfies `wide`.
/// Order and length bounds entail by value (a tighter lower bound entails a looser
/// one, dually for upper and length), decided through the ordering `oracle`; the
/// remaining kinds (`MultipleOf`, `Predicate`, `Regex`) have no sound value
/// entailment and require the constraint to appear verbatim, handled by the
/// caller's syntactic-containment check. A bound the oracle cannot compare is not
/// entailed (conservative).
fn constraint_entailed(
    wide: &Constraint,
    narrow: &[Constraint],
    oracle: &dyn LeafRelations,
) -> bool {
    use core::cmp::Ordering;
    let ge = |o: Option<Ordering>| matches!(o, Some(Ordering::Greater | Ordering::Equal));
    let gt = |o: Option<Ordering>| matches!(o, Some(Ordering::Greater));
    let le = |o: Option<Ordering>| matches!(o, Some(Ordering::Less | Ordering::Equal));
    let lt = |o: Option<Ordering>| matches!(o, Some(Ordering::Less));
    match wide {
        // x >= w holds if the subtype forces a lower bound at value >= w.
        Constraint::Ge(w) => narrow.iter().any(|c| match c {
            Constraint::Ge(n) | Constraint::Gt(n) => ge(oracle.compare(*n, *w)),
            _ => false,
        }),
        // x > w holds from Gt(n>=w), or Ge(n>w).
        Constraint::Gt(w) => narrow.iter().any(|c| match c {
            Constraint::Gt(n) => ge(oracle.compare(*n, *w)),
            Constraint::Ge(n) => gt(oracle.compare(*n, *w)),
            _ => false,
        }),
        // x <= w holds if the subtype forces an upper bound at value <= w.
        Constraint::Le(w) => narrow.iter().any(|c| match c {
            Constraint::Le(n) | Constraint::Lt(n) => le(oracle.compare(*n, *w)),
            _ => false,
        }),
        // x < w holds from Lt(n<=w), or Le(n<w).
        Constraint::Lt(w) => narrow.iter().any(|c| match c {
            Constraint::Lt(n) => le(oracle.compare(*n, *w)),
            Constraint::Le(n) => lt(oracle.compare(*n, *w)),
            _ => false,
        }),
        // Length bounds compare by their raw counts.
        Constraint::MinLen(w) => narrow
            .iter()
            .any(|c| matches!(c, Constraint::MinLen(n) if n >= w)),
        Constraint::MaxLen(w) => narrow
            .iter()
            .any(|c| matches!(c, Constraint::MaxLen(n) if n <= w)),
        // No sound value entailment without an exact match (handled by the caller).
        Constraint::MultipleOf(_) | Constraint::Predicate(_) | Constraint::Regex(_) => false,
    }
}

/// Keep the tighter of two one-sided bounds: the greater value for a lower bound,
/// the lesser for an upper bound; on equal values the strict end wins, and on an
/// incomparable pair the current bound is kept (conservative).
fn tighter_bound(
    current: Option<(OperandIx, bool)>,
    candidate: (OperandIx, bool),
    oracle: &dyn LeafRelations,
    is_lower: bool,
) -> (OperandIx, bool) {
    use core::cmp::Ordering;
    let Some(current) = current else {
        return candidate;
    };
    match oracle.compare(candidate.0, current.0) {
        Some(Ordering::Equal) => (current.0, current.1 || candidate.1),
        Some(Ordering::Greater) => {
            if is_lower {
                candidate
            } else {
                current
            }
        }
        Some(Ordering::Less) => {
            if is_lower {
                current
            } else {
                candidate
            }
        }
        None => current,
    }
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

/// Whether the keyed maps meeting in an intersection admit no dict between them.
///
/// ICFP formula (12) meets two record atoms pointwise -- field by field, clause
/// by clause -- and formula (11) makes the result empty when a field's type is.
/// A dict in the meet carries every key some side requires, with a value in every
/// type its sides give that key, so two rules follow:
///
/// - a key required somewhere whose types meet to nothing admits no dict;
/// - a key required somewhere and absent from a *closed* map admits none either,
///   since a closed map is exactly its declared keys.
///
/// Only a **required** key can empty a meet. Footnote 11 of the same paper is the
/// guard: the meet of two mappings "is never empty since it always contains at
/// least the empty record expression", and two optional fields are the same case
/// -- the empty dict satisfies both.
///
/// A map with clauses is not read as closed here. Deciding whether a clause
/// admits a given name means comparing a bare `String` against a key schema,
/// which the core cannot do, so any clause at all leaves the map open and the
/// second rule declines.
fn keyed_map_meet_empty(
    members: &[Schema],
    oracle: &dyn LeafRelations,
    defs: &[Schema],
    budget: &Cell<u32>,
) -> bool {
    let maps: Vec<(&[Field], bool)> = members
        .iter()
        .filter_map(|member| match member {
            Schema::KeyedMap { fields, defaults } => Some((&fields[..], defaults.is_empty())),
            _ => None,
        })
        .collect();
    if maps.len() < 2 {
        return false;
    }
    // Every type the maps give a key, and whether any of them requires it.
    let mut keys: FxHashMap<&str, (Vec<&Schema>, bool)> = FxHashMap::default();
    for (fields, _) in &maps {
        for field in *fields {
            let entry = keys.entry(&*field.name).or_default();
            entry.0.push(&field.schema);
            entry.1 |= field.required;
        }
    }
    keys.iter()
        .filter(|(_, (_, required))| *required)
        .any(|(name, (types, _))| {
            let types_cannot_hold = types.len() > 1 && {
                let meet = Schema::Intersection(types.iter().copied().cloned().collect());
                meet.is_empty_rec(oracle, defs, &mut Vec::new(), budget)
            };
            types_cannot_hold
                || maps.iter().any(|(fields, closed)| {
                    *closed && !fields.iter().any(|field| *field.name == **name)
                })
        })
}

/// The element schemas of a *fixed-arity* sequence, or `None` when the shape has
/// a tail.
///
/// Lemma 6.5 decomposes a product, and a sequence is a product only when its
/// component count is fixed: a repeated tail admits sequences of every length,
/// so there is no tuple of components to split over. An empty prefix with no
/// tail is the nullary product.
fn fixed_components(shape: &SeqShape) -> Option<Vec<Schema>> {
    shape.tail.is_none().then(|| shape.prefix.to_vec())
}

/// Whether the product `components` is contained in the union of the products in
/// `branches`, all of the same arity.
///
/// JACM Lemma 6.5 characterises this by splitting the negative set every way,
/// which is `2^|N|` subsets *and* needs a hypothesis retracted whenever a split
/// fails. ICFP §2.1.4 gives the equivalent formulation that does not backtrack,
/// and this is that function at `n` components rather than two:
///
/// ```text
/// Phi(P, [])       = false
/// Phi(P, [B, ..R]) = for every i:  P[i] <= B[i]  or  Phi(P with P[i] := P[i] \ B[i], R)
/// ```
///
/// Narrowing a component to the empty set makes the whole product empty, and the
/// empty set is below everything -- which is the base case that makes a value
/// split across branches decide, since no single branch contains it.
fn product_subtype(
    components: &[Schema],
    branches: &[&[Schema]],
    cx: SubtypeCx<'_>,
    assumptions: &mut Vec<(Schema, Schema)>,
) -> bool {
    if !spend(cx.budget) {
        return false;
    }
    if components
        .iter()
        .any(|c| c.is_empty_rec(cx.oracle, cx.defs, &mut Vec::new(), cx.budget))
    {
        return true;
    }
    let Some((branch, rest)) = branches.split_first() else {
        return false;
    };
    components
        .iter()
        .zip(branch.iter())
        .enumerate()
        .all(|(position, (mine, theirs))| {
            mine.is_subtype_rec(theirs, cx, assumptions).holds() || {
                // The component this branch does not cover, narrowed by what the
                // branch takes away, with the rest of the tuple as it was. Built
                // by mapping rather than by writing at an index: the position
                // comes from the same enumeration as the component, so an index
                // write cannot go out of range and cannot be seen not to.
                let narrowed: Vec<Schema> = components
                    .iter()
                    .enumerate()
                    .map(|(index, component)| {
                        if index == position {
                            Schema::meet([mine.clone(), theirs.clone().complement()])
                        } else {
                            component.clone()
                        }
                    })
                    .collect();
                product_subtype(&narrowed, rest, cx, assumptions)
            }
        })
}

/// Whether a fixed-arity sequence is covered by the sequence branches of a union.
///
/// A value that splits across branches -- `tuple[int|str, int]` covered by
/// `tuple[int, int] | tuple[str, int]` -- lands in no single branch, so the rule
/// that tries each branch alone cannot see it. Branches of another container or
/// another arity share no value with `self` by shape, so they drop out rather
/// than blocking the decomposition.
fn seq_splits_across_union(
    schema: &Schema,
    members: &[Schema],
    cx: SubtypeCx<'_>,
    assumptions: &mut Vec<(Schema, Schema)>,
) -> bool {
    let Schema::Seq { container, shape } = schema else {
        return false;
    };
    let Some(components) = fixed_components(shape) else {
        return false;
    };
    let branches: Vec<Vec<Schema>> = members
        .iter()
        .filter_map(|member| match member {
            Schema::Seq {
                container: their_kind,
                shape: their_shape,
            } if their_kind == container => fixed_components(their_shape),
            _ => None,
        })
        .filter(|branch| branch.len() == components.len())
        .collect();
    if branches.is_empty() {
        return false;
    }
    let branches: Vec<&[Schema]> = branches.iter().map(Vec::as_slice).collect();
    product_subtype(&components, &branches, cx, assumptions)
}

/// Whether two members are provably disjoint (distinct concrete kinds, `bool ⊆
/// int` aside), so the intersection is empty. This decides the structural-kind
/// disjointness (a list is never a set) the scalar region bitset cannot see.
/// Shared with the simplifier so both read the same lattice law.
/// The constants of a schema that is a literal, or a union of nothing but
/// literals; `None` for anything else.
///
/// What makes the set question askable: a union carrying one non-literal member
/// has no set of constants standing for it, and the member walk is then the
/// only reading.
fn literal_constants(schema: &Schema) -> Option<Vec<ConstIx>> {
    match schema {
        Schema::Literal(index) => Some(vec![*index]),
        Schema::Union(members) if !members.is_empty() => members
            .iter()
            .map(|member| match member {
                Schema::Literal(index) => Some(*index),
                _ => None,
            })
            .collect(),
        _ => None,
    }
}

pub(crate) fn has_disjoint_pair(members: &[Schema], oracle: &dyn LeafRelations) -> bool {
    unordered_pairs(members).any(|(a, b)| a.disjoint_with(b, oracle))
}

/// Whether the refinement constraints of the intersection's **directly refined
/// members** cannot hold together. A value in the intersection satisfies every
/// member, so the constraints of each top-level `Refine` member apply to it at
/// once. This gathers only those top-level constraints — a refinement nested
/// inside a member (say under a union arm) is not collected here; the decision
/// stays sound, since missing a contradiction only forgoes reporting emptiness,
/// never reports a non-empty intersection empty.
fn intersection_bounds_unsatisfiable(members: &[Schema], oracle: &dyn LeafRelations) -> bool {
    // Gather the top-level refine members' constraints by reference — no clone, so
    // a `Regex` constraint's pattern string is not copied per intersection node.
    let merged: Vec<&Constraint> = members
        .iter()
        .filter_map(|m| match m {
            Schema::Refine { constraints, .. } => Some(&constraints[..]),
            _ => None,
        })
        .flatten()
        .collect();
    let int_discrete = bounded_to_the_integers(members);
    !merged.is_empty() && bounds_unsatisfiable(merged.iter().copied(), oracle, int_discrete)
}

/// Whether the language `pa · ta*` is included in `pb · tb*` — a fixed prefix
/// optionally followed by a repeated tail, which is every [`SeqShape`].
/// `ta`/`tb` of `None` mean no repeated tail.
fn linear_subtype(
    pa: &[Schema],
    ta: Option<&Schema>,
    pb: &[Schema],
    tb: Option<&Schema>,
    cx: SubtypeCx<'_>,
    assumptions: &mut Vec<(Schema, Schema)>,
) -> Relation {
    // A repeated tail with an empty element language never repeats, so the left
    // side is then just its fixed prefix. Emptiness is decided with the same
    // oracle and definitions as the rest of the decision, so a tail empty only
    // under a refinement bound or an uninhabited recursive reference is
    // recognised here too, consistent with the context-aware recursion around it.
    let ta =
        ta.filter(|element| !element.is_empty_rec(cx.oracle, cx.defs, &mut Vec::new(), cx.budget));
    // A's fixed prefix must align with B: against B's prefix where they overlap,
    // then against B's repeated tail past it (which B must therefore have). A
    // prefix shorter than B's cannot align at all, which is a refutation by
    // shape: the lengths one side admits are not among the other's.
    if pa.len() < pb.len() {
        return Relation::Fails;
    }
    let aligns = |assumptions: &mut Vec<(Schema, Schema)>| {
        Relation::all(pa.iter().enumerate().map(|(i, element)| match pb.get(i) {
            Some(expected) => element.is_subtype_rec(expected, cx, assumptions),
            // Past B's prefix, B must repeat -- a fixed-length B admits no such
            // position at all.
            None => tb.map_or(Relation::Fails, |tail| {
                element.is_subtype_rec(tail, cx, assumptions)
            }),
        }))
    };
    match (ta, tb) {
        (None, None) if pa.len() != pb.len() => Relation::Fails,
        (None, None | Some(_)) => aligns(assumptions),
        // A repeats without bound but B is finite-length: impossible.
        (Some(_), None) => Relation::Fails,
        // A's repeated element must also land in B's repeated tail.
        (Some(a), Some(tail)) => {
            aligns(assumptions).and(|| a.is_subtype_rec(tail, cx, assumptions))
        }
    }
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

/// Whether keyed-map `a` (fields `fa`, default clauses `da`) is a subtype of
/// keyed-map `b`. Sound everywhere; complete on three shapes, conservative
/// (returns `false`) outside them:
///
/// 1. **Closed record ≤ anything** (`da` empty): holds by width and depth (each
///    field of `a` maps into a like-named field of `b` with a subtype schema) and
///    by required-ness (every field `b` requires is required in `a`).
/// 2. **Pure mapping ≤ pure mapping** (`fa` and `fb` empty): every clause of `a`
///    is subsumed by a clause of `b` with both key and value narrower.
/// 3. **Mixed record-and-catch-all ≤ mixed** (general): each shared field narrows
///    and respects required-ness; each field `a` declares that `b` does not is
///    covered by `b`'s catch-all; each field `b` requires that `a` lacks is
///    governed by `a`'s catch-all — decidable when it is **optional** and every
///    catch-all value of `a` fits it, and *refuted* when `a` carries no
///    catch-all at all, since a closed record admits no value with a key it does
///    not declare; and every catch-all clause of `a` is subsumed by one of `b`.
///
/// Sound throughout — a required supertype field a subject with a catch-all
/// cannot guarantee present, or a clause an oracle cannot relate, is undecided
/// rather than an unsound proof, and a refutation stands on a value of the
/// subject, which the query's witness guard reads against its emptiness.
fn keyed_map_subtype(
    fa: &[Field],
    da: &[MapClause],
    fb: &[Field],
    db: &[MapClause],
    cx: SubtypeCx<'_>,
    assumptions: &mut Vec<(Schema, Schema)>,
) -> Relation {
    // Index both field lists by name once, so the cross-list lookups below are O(1)
    // each rather than a fresh linear scan per field (O(fields²) per comparison).
    let a_by_name = field_index(fa);
    let b_by_name = field_index(fb);
    {
        // One rule for every shape a keyed map takes. The closed record and the
        // pure mapping are not special cases needing a branch of their own: each
        // is this rule with one of the two lists empty, and a dedicated branch
        // for either can only answer the same question less well. The closed
        // record did -- it read a field the supertype covered through a catch-all
        // as undecided, so `{"x": int}` was not seen below `dict[str, int]`.
        //
        // Every supertype field is checked against `a`: a field `a` declares is
        // matched field-wise; a field `a` lacks is governed by `a`'s catch-all.
        let fields_ok = Relation::all(fb.iter().map(|b_field| {
            match a_by_name.get(&*b_field.name) {
                // Shared field: it must narrow in depth, and a field `b` requires
                // must be required in `a` too. A key the supertype requires and
                // the subtype does not is a value of the subtype -- the one
                // leaving that key out -- that the supertype rejects.
                Some(a_field) => a_field
                    .schema
                    .is_subtype_rec(&b_field.schema, cx, assumptions)
                    .and(|| Relation::decided(!b_field.required || a_field.required)),
                // A field `b` requires that `a` lacks. A catch-all guarantees a
                // key's value type and never its presence, so a subject carrying
                // one may or may not place the key and the relation is undecided
                // -- but a subject carrying *no* catch-all is closed, and a
                // closed record admits no value with a key it does not declare.
                // Every value it has is one `b` rejects, which is a refutation
                // by the same reading as a key `a` declares optional two arms
                // above. The empty subject is not an exception: it has no such
                // value, and the witness guard at the top of the query reads
                // this refutation against its emptiness before believing it.
                None if b_field.required => {
                    if da.is_empty() {
                        Relation::Fails
                    } else {
                        Relation::Unknown
                    }
                }
                None => Relation::all(da.iter().map(|clause| {
                    clause
                        .value
                        .is_subtype_rec(&b_field.schema, cx, assumptions)
                })),
            }
        }));
        // Each field `a` declares that `b` does not is read by `b` through its
        // catch-all, so a `str`/`anything`-keyed clause of `b` must cover it.
        let extra_covered = Relation::all(
            fa.iter()
                .filter(|a_field| !b_by_name.contains_key(&*a_field.name))
                .map(|a_field| {
                    // The clause has to be one that admits the *name*, and this
                    // reads only the two spellings that plainly do, so a clause
                    // it cannot read leaves the field unproven rather than
                    // uncovered.
                    Relation::proven(db.iter().any(|clause| {
                        matches!(clause.key, Schema::Str | Schema::Anything(_))
                            && a_field
                                .schema
                                .is_subtype_rec(&clause.value, cx, assumptions)
                                .holds()
                    }))
                }),
        );
        // Every catch-all clause of `a` (governing its non-field keys) is subsumed
        // by a clause of `b` with both key and value narrower.
        let defaults = Relation::all(da.iter().map(|mine| {
            // One clause of `b` subsuming this one settles it; none of them
            // doing so is a decline, since a clause pair the rules cannot
            // relate is not a pair they have refuted.
            Relation::proven(db.iter().any(|theirs| {
                mine.key
                    .is_subtype_rec(&theirs.key, cx, assumptions)
                    .and(|| mine.value.is_subtype_rec(&theirs.value, cx, assumptions))
                    .holds()
            }))
        }));
        fields_ok.and(|| extra_covered).and(|| defaults)
    }
}

/// Whether the attribute record `fa` is a subtype of `fb`: width and depth.
///
/// Every attribute the supertype declares must be one the subtype declares, and
/// no less narrowly. There is no class in it -- a record denotes every value
/// carrying its attributes, whatever the value is -- so the nominal question the
/// old node asked first is now a conjunct of its own, and two records that came
/// from unrelated classes still relate.
///
/// The rule is set inclusion read off the denotation: an attribute the supertype
/// does not name constrains nothing, and one it names constrains every value of
/// the subtype exactly when the subtype names it too, at least as narrowly.
fn attr_record_subtype(
    fa: &[Field],
    fb: &[Field],
    cx: SubtypeCx<'_>,
    assumptions: &mut Vec<(Schema, Schema)>,
) -> Relation {
    let a_by_name = field_index(fa);
    Relation::all(fb.iter().map(|b| {
        match a_by_name.get(&*b.name) {
            // An attribute the supertype names and the subtype does not: the
            // subtype holds values without it, and those are outside the
            // supertype. So is a supertype attribute the subtype only *may*
            // carry.
            None => Relation::Fails,
            Some(a) if b.required && !a.required => Relation::Fails,
            Some(a) => a.schema.is_subtype_rec(&b.schema, cx, assumptions),
        }
    }))
}

/// Index a field list by name for O(1) cross-list lookup during subtyping.
///
/// Unique field names are a hard caller invariant: `collect` into a map keeps the
/// last entry per key, so a duplicate name would silently shadow an earlier field
/// and could make the `required`/width checks that consume this index unsound. The
/// frontend rejects duplicates; the `debug_assert` makes that dependency explicit
/// and catches a malformed IR in debug rather than deciding on a shadowed field.
fn field_index(fields: &[Field]) -> FxHashMap<&str, &Field> {
    let index: FxHashMap<&str, &Field> = fields.iter().map(|f| (&*f.name, f)).collect();
    debug_assert_eq!(
        index.len(),
        fields.len(),
        "record has duplicate field names; the frontend must reject them"
    );
    index
}

/// A set of value-universe regions: which of the mutually-disjoint parts the
/// universe is cut into a schema's denotation can reach.
///
/// The value universe is partitioned so a Boolean combination of scalar atoms
/// denotes a set the lattice operations compute exactly. Which region a kind
/// falls in is [`Kind::region`]: the six scalar kinds take one each, and every
/// container kind shares [`Region::NON_SCALAR`]. That region exists so the
/// complement of a scalar includes every non-scalar value, which keeps emptiness
/// sound — the meet of all six scalar complements is the inhabited non-scalar
/// region, not the empty set.
///
/// **A set holds a region only when a schema names it exactly.** The set is read
/// back through [`complement`](Region::complement), and the complement of an
/// over-approximation is an under-approximation, which would report an inhabited
/// schema empty. So `str` earns a region and `list[int]` does not: a list schema
/// is a proper part of the lists, and the fold keeps it opaque.
///
/// **It is a set, and its operations are the set's.** The bits were an
/// `Option<u8>` combined at each call site with `|`, `&`, `!`, `|=`, and `<<`,
/// where "did this schema's regions cancel" read as `== Some(0)` and "is every
/// region of this one also a region of that" read as `a & !b == 0`. Naming the
/// operations puts each of those decisions in one place with one test, and takes
/// the raw operators out of the fold entirely.
///
/// 6 scalar regions plus the non-scalar region is 7 of `u8`'s 8 bits. The
/// representation is sized to the partition, so an 8th region still fits and a
/// 9th would overflow `1 << 8` at compile time — the width is the guard against
/// a silent wrap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub(crate) struct Region(u8);

impl Region {
    /// No region: the empty set of values.
    pub(crate) const EMPTY: Region = Region(0);
    /// Every region: the whole value universe.
    pub(crate) const ALL: Region = Region((1 << 7) - 1);

    /// Everything that is no scalar kind -- containers, instances, callables,
    /// and the rest -- in one region. No schema names it alone, which is why
    /// one bit is enough for every kind that falls in it.
    const NON_SCALAR: Region = Region(1 << 6);

    /// Every region in either set.
    #[inline]
    pub(crate) const fn union(self, other: Region) -> Region {
        Region(self.0 | other.0)
    }

    /// Every region in both sets.
    #[inline]
    pub(crate) const fn intersect(self, other: Region) -> Region {
        Region(self.0 & other.0)
    }

    /// Every region this set does not hold. Bounded to the partition, so the
    /// unused eighth bit never appears in a result.
    #[inline]
    pub(crate) const fn complement(self) -> Region {
        Region(Region::ALL.0 & !self.0)
    }

    /// Whether this set holds no region at all — the schema denotes no value.
    #[inline]
    pub(crate) const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Whether every region of `self` is also a region of `other`: set inclusion,
    /// which on the scalar-decidable fragment *is* the subtyping relation.
    #[inline]
    pub(crate) const fn subset_of(self, other: Region) -> bool {
        self.intersect(other.complement()).is_empty()
    }
}

/// A concrete runtime kind: the type a value's `type(x)` is.
///
/// The kinds partition the value universe, so two schemas carrying different
/// kinds share no value -- `bool` and `int` aside, since `bool` subclasses `int`.
/// A schema the core cannot kind has none, and disjointness stays conservative.
///
/// Public because the core cannot see a Python object: a `Literal`'s kind is a
/// fact about a pooled constant, which only the bindings can read, and they
/// answer in this vocabulary through [`LeafRelations::literal_kind`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Kind {
    /// `None`.
    NoneType,
    Bool,
    Int,
    Float,
    Str,
    Bytes,
    List,
    Tuple,
    Set,
    FrozenSet,
    Dict,
}

impl Kind {
    /// Every kind, in one place, so a walk over the partition cannot miss one.
    ///
    /// A `match` is exhaustive and a list is not, so this array carries a test
    /// that counts it against the variants rather than a promise that it is
    /// complete.
    pub const ALL: [Kind; 11] = [
        Kind::NoneType,
        Kind::Bool,
        Kind::Int,
        Kind::Float,
        Kind::Str,
        Kind::Bytes,
        Kind::List,
        Kind::Tuple,
        Kind::Set,
        Kind::FrozenSet,
        Kind::Dict,
    ];
}

impl Kind {
    /// The [`Region`] this kind falls in.
    ///
    /// The one place that says where a kind lands, so adding a kind is a change
    /// in one file that the compiler makes you finish. The two vocabularies were
    /// separate lists -- the kinds here and a set of region constants beside
    /// them -- with nothing tying `List` to the region a list belongs to.
    ///
    /// The six scalar kinds each get a region of their own, because a schema can
    /// name one exactly: `str` denotes every string and nothing else, so the
    /// complement of `str` is exactly the other six regions. The five container
    /// kinds share the non-scalar region, because no schema names one exactly --
    /// `list[int]` is a proper part of the lists, so the fold keeps a container
    /// opaque rather than claiming a region for it (see [`Regions`]).
    pub(crate) const fn region(self) -> Region {
        match self {
            Kind::NoneType => Region(1 << 0),
            Kind::Bool => Region(1 << 1),
            Kind::Int => Region(1 << 2),
            Kind::Float => Region(1 << 3),
            Kind::Str => Region(1 << 4),
            Kind::Bytes => Region(1 << 5),
            Kind::List | Kind::Tuple | Kind::Set | Kind::FrozenSet | Kind::Dict => {
                Region::NON_SCALAR
            }
        }
    }
}

#[cfg(test)]
mod budget_tests;

#[cfg(test)]
mod region_tests;

#[cfg(test)]
mod tests;
