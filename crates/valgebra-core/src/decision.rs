//! The decision procedures over the IR: emptiness, subtyping, equivalence, and
//! disjointness, with the leaf-relation oracle and the scalar region partition.

use crate::ir::{
    ClassIx, ConstIx, Constraint, DefIx, Field, MapClause, OperandIx, Schema, SeqKind, SeqShape,
};
use rustc_hash::FxHashMap;
use std::cell::Cell;

/// The most decision steps one top-level query may take before it stops and
/// returns the conservative answer. Subtyping distributes over unions and
/// intersections and emptiness recurses the structural fragment, so a deeply
/// nested Boolean combination can demand work exponential in its depth; without
/// interning to share equal subtrees there is no cheap memo, so the procedure
/// bounds its own work. One budget is threaded through a whole top-level query —
/// subtyping and the emptiness checks it calls into share it, and the two
/// directions of an equivalence share it — so the bound cannot be escaped through
/// a side door or spent twice. The ceiling is far above any schema a real
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
    ) -> bool {
        self == other
            || linear_subtype(
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
            Schema::Set(_) => Kind::Set,
            Schema::FrozenSet(_) => Kind::FrozenSet,
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
            Schema::Anything => Region::ALL,
            Schema::Union(members) => {
                let mut acc = Regions::UNION_UNIT;
                for member in members {
                    acc = acc.union(member.region_set());
                    if acc.is_absorbing() {
                        return acc;
                    }
                }
                return acc;
            }
            Schema::Intersection(members) => {
                let mut acc = Regions::MEET_UNIT;
                for member in members {
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
    /// always a member). The gradual `Any`, instances, literals, refinements,
    /// and unresolved recursive references are not decided, so a combination
    /// containing one is never reported empty. To resolve recursive references,
    /// use [`is_empty_under`](Self::is_empty_under).
    ///
    /// The decision is bounded: a deeply nested adversarial schema that would take
    /// more than a fixed number of steps stops and returns `false`, so a `false`
    /// means "not proven empty within the work bound", not necessarily "non-empty".
    /// A real schema decides far inside the bound. The scalar-region check is folded
    /// bottom-up from each node's children, so nested Boolean structure is decided
    /// in time linear in its size rather than by re-walking each subtree per level.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.is_empty_rec(
            &NoLeafRelations,
            &[],
            &mut Vec::new(),
            &Cell::new(DECISION_BUDGET),
        )
    }

    /// Like [`is_empty`](Self::is_empty), but resolving recursive references
    /// through `defs`, so an uninhabited recursive schema — a mandatory
    /// self-reference with no base case — is detected. A reference `defs` does
    /// not resolve stays conservative (never reported empty).
    #[must_use]
    pub fn is_empty_under(&self, defs: &[Schema]) -> bool {
        self.is_empty_rec(
            &NoLeafRelations,
            defs,
            &mut Vec::new(),
            &Cell::new(DECISION_BUDGET),
        )
    }

    /// Like [`is_empty_under`](Self::is_empty_under), but with an `oracle` that
    /// can order the pool values behind refinement bounds, so an unsatisfiable
    /// bound conjunction (a lower bound above an upper bound) is detected.
    #[must_use]
    pub fn is_empty_with(&self, oracle: &dyn LeafRelations, defs: &[Schema]) -> bool {
        self.is_empty_rec(oracle, defs, &mut Vec::new(), &Cell::new(DECISION_BUDGET))
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
            Schema::Anything => (Verdict::Inhabited, Regions::Known(Region::ALL)),
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
            Schema::Set(_) | Schema::FrozenSet(_) => (Verdict::Inhabited, Regions::Unknown),
            // A map is emptied by a required field that admits nothing, and
            // admits the empty dict when it requires nothing at all.
            Schema::KeyedMap { fields, .. } => {
                let required = fields.iter().filter(|field| field.required);
                let verdict = Verdict::every(
                    required.map(|field| field.schema.verdict_rec(oracle, defs, visiting, budget)),
                );
                (verdict, Regions::Unknown)
            }
            // A structural-attribute schema requires every field, so an empty
            // required field's schema empties it — the same rule as a keyed map.
            // An uninhabited dataclass-style schema is detected here; the nominal
            // `isinstance` part stays opaque, so it never narrows to empty.
            Schema::Attrs { fields, .. } => {
                let required = fields.iter().filter(|field| field.required);
                let attributes = Verdict::every(
                    required.map(|field| field.schema.verdict_rec(oracle, defs, visiting, budget)),
                );
                // Inhabited attributes are not an inhabited schema: a value must
                // also be an instance of the class, which the core cannot decide.
                let verdict = if attributes.is_empty() {
                    Verdict::Empty
                } else {
                    Verdict::Unknown
                };
                (verdict, Regions::Unknown)
            }
            // A union is empty when every member is and inhabited when any member
            // is; its region is the union of the members' regions, again folded
            // from the children.
            Schema::Union(members) => {
                let mut verdict = Verdict::Empty;
                let mut region = Regions::UNION_UNIT;
                for m in members {
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
    /// conservative where it cannot decide (`Or` regexes, recursive references,
    /// instances, literals): there it returns `false` rather than guess.
    ///
    /// The decision is bounded: an adversarial schema that would take more than a
    /// fixed number of steps stops and returns `false`, so a `false` can mean
    /// "not proven a subtype within the work bound". A real schema decides far
    /// inside the bound.
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
        let budget = Cell::new(DECISION_BUDGET);
        self.is_subtype_rec(
            other,
            SubtypeCx {
                oracle,
                defs,
                budget: &budget,
            },
            &mut Vec::new(),
        )
    }

    fn is_subtype_rec(
        &self,
        other: &Schema,
        cx: SubtypeCx<'_>,
        assumptions: &mut Vec<(Schema, Schema)>,
    ) -> bool {
        // Bound the total work: the distribution rules below can demand effort
        // exponential in the schema depth, so once the shared budget is spent the
        // decision stops and returns the conservative `false` rather than running
        // unbounded. A real annotation decides in a few steps; only an adversarial
        // schema reaches the ceiling.
        // Reflexivity, by identity. The checks below are ordered by what they
        // cost, and each answers the whole query on its own, so a cheaper one
        // never hides a verdict a later one would reach.
        if core::ptr::eq(self, other) {
            return true;
        }
        if !spend(cx.budget) {
            return false;
        }
        // Scalar fragment: exact via the region partition. Subtyping there is
        // set inclusion between the two region sets, and nothing else. A
        // non-scalar node yields `Unknown` from its own discriminant, so this
        // costs one match off the fragment.
        let supertype_regions = other.region_set();
        if let (Regions::Known(a), Regions::Known(b)) = (self.region_set(), supertype_regions) {
            return a.subset_of(b);
        }
        // Reflexivity for two equal spellings that are not the same node.
        if self == other {
            return true;
        }
        // Coinductive hypothesis: a goal already being proven on this path is
        // assumed to hold, so two recursive types are compared at their greatest
        // fixpoint rather than unfolded forever. The scan is empty for a
        // non-recursive query; under recursion the stack holds one entry per
        // reference goal still being unfolded. Every recorded goal has a `Ref` on
        // one side, so the structural compare rejects a mismatched goal on the
        // discriminant before walking either subtree.
        if assumptions.iter().any(|(a, b)| a == self && b == other) {
            return true;
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
        Schema::Intersection(vec![self.clone(), other.clone()]).is_empty_rec(
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
    ) -> bool {
        match self {
            Schema::Ref(id) => match cx.defs.get(id.get()) {
                Some(def) => {
                    assumptions.push((self.clone(), other.clone()));
                    let holds = def.is_subtype_rec(other, cx, assumptions);
                    assumptions.pop();
                    holds
                }
                None => false,
            },
            Schema::Refine { base, .. } => base.is_subtype_rec(other, cx, assumptions),
            _ => false,
        }
    }

    fn subtype_decide(
        &self,
        other: &Schema,
        supertype_regions: Regions,
        cx: SubtypeCx<'_>,
        assumptions: &mut Vec<(Schema, Schema)>,
    ) -> bool {
        match (self, other) {
            // Every lattice bound, in one arm: `∅ ⊆ B`, `A ⊆ U`, and `A ⊆ ∅`
            // when A is empty. All three are the same question asked of
            // emptiness, so one guard answers them and a per-atom arm beside it
            // would be dead code -- the mutation sweep says so, by surviving its
            // deletion.
            _ if self.bounds_the_pair(supertype_regions, cx) => true,
            // (X ∪ Y) ⊆ Z iff X ⊆ Z and Y ⊆ Z; A ⊆ (Y ∩ Z) iff A ⊆ Y and A ⊆ Z.
            (Schema::Union(members), _) => members
                .iter()
                .all(|m| m.is_subtype_rec(other, cx, assumptions)),
            (_, Schema::Intersection(members)) => members
                .iter()
                .all(|m| self.is_subtype_rec(m, cx, assumptions)),
            // (A ∩ B) ⊆ C if some conjunct already is. When C is a union, the meet
            // may instead land in one branch, so that sound rule is tried too —
            // ahead of the plain `_ ⊆ (Y ∪ Z)` rule, so a meet that contains its
            // own supertype (a reference beside that union) decides, which is what
            // lets such a meet be recognised as a subtype of itself.
            (Schema::Intersection(members), _) => {
                members
                    .iter()
                    .any(|m| m.is_subtype_rec(other, cx, assumptions))
                    || matches!(other, Schema::Union(branches)
                        if branches.iter().any(|b| self.is_subtype_rec(b, cx, assumptions)))
            }
            // A ⊆ (Y ∪ Z) if A lands in one branch (sound, conservative), or -- for
            // a fixed-arity sequence -- if it splits across the branches, which
            // no single-branch rule can see. Failing both, the subject may still
            // be one a left-side rule reduces to something the union contains.
            (_, Schema::Union(members)) => {
                // A branch equal to the subject settles it, which is set
                // containment and is linear in the branches. The recursion below
                // decides the rest, at the cost of a full step per branch.
                members.contains(self)
                    || members
                        .iter()
                        .any(|m| self.is_subtype_rec(m, cx, assumptions))
                    || seq_splits_across_union(self, members, cx, assumptions)
                    || self.left_reduces_below(other, cx, assumptions)
            }
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
                None => false,
            },
            // Set and frozenset inclusion reduces to element inclusion.
            (Schema::Set(a), Schema::Set(b)) | (Schema::FrozenSet(a), Schema::FrozenSet(b)) => {
                a.is_subtype_rec(b, cx, assumptions)
            }
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
            // Two structural-attribute schemas relate nominally and by attribute.
            (
                Schema::Attrs {
                    class_index: ca,
                    fields: fa,
                },
                Schema::Attrs {
                    class_index: cb,
                    fields: fb,
                },
            ) => attrs_subtype(*ca, fa, *cb, fb, cx, assumptions),
            // An attribute schema is its class's isinstance atom narrowed by an
            // attribute record, so it is below that atom and below any atom the
            // atom is below.
            (Schema::Attrs { class_index, .. }, Schema::Instance(_)) => {
                Schema::Instance(*class_index).is_subtype_rec(other, cx, assumptions)
            }
            // Complement is contravariant: ¬A ⊆ ¬B exactly when B ⊆ A.
            (Schema::Complement(a), Schema::Complement(b)) => b.is_subtype_rec(a, cx, assumptions),
            (_, Schema::Complement(inner)) => self.shares_no_value_with(inner, cx),
            // A refinement is a subset of its base. Against another refinement the
            // base must subtype and every constraint of the supertype must hold of
            // every subtype value: either it appears verbatim, or it is entailed by
            // the subtype's bounds (a tighter lower/upper/length bound entails a
            // looser one, decided through the ordering oracle). A bound the oracle
            // cannot compare, and a non-order constraint, stay on the verbatim path.
            (
                Schema::Refine {
                    base: narrow_base,
                    constraints: narrow_cons,
                },
                Schema::Refine {
                    base: wide_base,
                    constraints: wide_cons,
                },
            ) => {
                narrow_base.is_subtype_rec(wide_base, cx, assumptions)
                    && wide_cons.iter().all(|constraint| {
                        narrow_cons.contains(constraint)
                            || constraint_entailed(constraint, narrow_cons, cx.oracle)
                    })
            }
            // Against a non-refinement, a refinement inherits its base's supertypes.
            (Schema::Refine { .. }, _) => self.left_reduces_below(other, cx, assumptions),
            // A leaf the structural rules cannot relate (an instance or literal):
            // defer to the oracle, conservative when it declines.
            _ => cx.oracle.leaf_subtype(self, other).unwrap_or(false),
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
        self.is_subtype_rec(other, cx, &mut Vec::new())
            && other.is_subtype_rec(self, cx, &mut Vec::new())
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
pub trait LeafRelations {
    /// Whether leaf schema `sub` is a subtype of `sup`, or `None` to leave the
    /// relation conservatively undecided.
    fn leaf_subtype(&self, sub: &Schema, sup: &Schema) -> Option<bool>;

    /// Order the two pool values behind refinement bounds at indices `left` and
    /// `right`, or `None` when the core cannot or the values are not comparable.
    /// The default decides nothing, so bound satisfiability stays conservative.
    fn compare(&self, _left: OperandIx, _right: OperandIx) -> Option<core::cmp::Ordering> {
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
    members.iter().any(|member| match member {
        Schema::Complement(inner) => {
            denotes_a_set(inner, oracle) && members.iter().any(|other| other == &**inner)
        }
        _ => false,
    })
}

/// Whether a schema denotes a *set*: the same values however often it is asked.
///
/// Sound rather than complete, and conservative in the direction that declines.
/// A callback is the atom this rules out: `Predicate` runs user code, so two
/// occurrences of one schema can disagree, and a law that assumes they agree is
/// not a law about this. A reference is ruled out for the same reason at one
/// remove -- what it names is not in hand here, so a callback may hide behind it.
/// The gradual `Any` is ruled out because its complement is not its set
/// complement.
///
/// A class is referred to the `oracle`: `isinstance` against a metaclass that
/// overrides `__instancecheck__` is a callback too, and telling a pure class from
/// a hooked one needs the class object, which only the bindings hold.
pub(crate) fn denotes_a_set(schema: &Schema, oracle: &dyn LeafRelations) -> bool {
    let mut pending = vec![schema];
    while let Some(node) = pending.pop() {
        match node {
            Schema::Dynamic | Schema::Ref(_) | Schema::SelfRef(_) => return false,
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
            Schema::Set(inner) | Schema::FrozenSet(inner) | Schema::Complement(inner) => {
                pending.push(inner);
            }
            Schema::Union(members) | Schema::Intersection(members) => pending.extend(members),
            Schema::KeyedMap { fields, defaults } => {
                pending.extend(fields.iter().map(|field| &field.schema));
                for clause in defaults {
                    pending.push(&clause.key);
                    pending.push(&clause.value);
                }
            }
            Schema::Attrs { fields, .. } => {
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
            Schema::KeyedMap { fields, defaults } => Some((fields.as_slice(), defaults.is_empty())),
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
            let entry = keys.entry(field.name.as_str()).or_default();
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
                    *closed && !fields.iter().any(|field| field.name == **name)
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
    shape.tail.is_none().then(|| shape.prefix.clone())
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
            mine.is_subtype_rec(theirs, cx, assumptions) || {
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
            Schema::Refine { constraints, .. } => Some(constraints.as_slice()),
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
) -> bool {
    // A repeated tail with an empty element language never repeats, so the left
    // side is then just its fixed prefix. Emptiness is decided with the same
    // oracle and definitions as the rest of the decision, so a tail empty only
    // under a refinement bound or an uninhabited recursive reference is
    // recognised here too, consistent with the context-aware recursion around it.
    let ta =
        ta.filter(|element| !element.is_empty_rec(cx.oracle, cx.defs, &mut Vec::new(), cx.budget));
    // A's fixed prefix must align with B: against B's prefix where they overlap,
    // then against B's repeated tail past it (which B must therefore have).
    let prefix_aligns = pa.len() >= pb.len()
        && pa.iter().enumerate().all(|(i, element)| match pb.get(i) {
            Some(expected) => element.is_subtype_rec(expected, cx, assumptions),
            None => tb.is_some_and(|tail| element.is_subtype_rec(tail, cx, assumptions)),
        });
    match (ta, tb) {
        (None, None) => pa.len() == pb.len() && prefix_aligns,
        (None, Some(_)) => prefix_aligns,
        // A repeats without bound but B is finite-length: impossible.
        (Some(_), None) => false,
        // A's repeated element must also land in B's repeated tail.
        (Some(a), Some(tail)) => prefix_aligns && a.is_subtype_rec(tail, cx, assumptions),
    }
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
///    covered by `b`'s catch-all; each field `b` declares that `a` lacks is
///    governed by `a`'s catch-all — decidable only when it is **optional** (a
///    catch-all guarantees a key's value type, never its presence, so a required
///    such field stays `false`) and every catch-all value of `a` fits it; and
///    every catch-all clause of `a` is subsumed by one of `b`.
///
/// Sound throughout — a required supertype field the subtype cannot guarantee
/// present, or a clause an oracle cannot relate, is reported `false`, never an
/// unsound `true`.
fn keyed_map_subtype(
    fa: &[Field],
    da: &[MapClause],
    fb: &[Field],
    db: &[MapClause],
    cx: SubtypeCx<'_>,
    assumptions: &mut Vec<(Schema, Schema)>,
) -> bool {
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
        let fields_ok = fb.iter().all(|b_field| {
            match a_by_name.get(b_field.name.as_str()) {
                // Shared field: it must narrow in depth, and a field `b` requires
                // must be required in `a` too.
                Some(a_field) => {
                    a_field
                        .schema
                        .is_subtype_rec(&b_field.schema, cx, assumptions)
                        && (!b_field.required || a_field.required)
                }
                // A field `b` declares that `a` lacks: a catch-all guarantees a
                // key's value type but never its presence, so a *required* such
                // field stays undecided; an *optional* one holds when every value
                // `a`'s catch-all could place at that key fits `b`'s field schema.
                None => {
                    !b_field.required
                        && da.iter().all(|clause| {
                            clause
                                .value
                                .is_subtype_rec(&b_field.schema, cx, assumptions)
                        })
                }
            }
        });
        // Each field `a` declares that `b` does not is read by `b` through its
        // catch-all, so a `str`/`anything`-keyed clause of `b` must cover it.
        let extra_covered = fa
            .iter()
            .filter(|a_field| !b_by_name.contains_key(a_field.name.as_str()))
            .all(|a_field| {
                db.iter().any(|clause| {
                    matches!(clause.key, Schema::Str | Schema::Anything)
                        && a_field
                            .schema
                            .is_subtype_rec(&clause.value, cx, assumptions)
                })
            });
        // Every catch-all clause of `a` (governing its non-field keys) is subsumed
        // by a clause of `b` with both key and value narrower.
        let defaults = da.iter().all(|mine| {
            db.iter().any(|theirs| {
                mine.key.is_subtype_rec(&theirs.key, cx, assumptions)
                    && mine.value.is_subtype_rec(&theirs.value, cx, assumptions)
            })
        });
        fields_ok && extra_covered && defaults
    }
}

/// Whether attribute schema `(ca, fa)` is a subtype of `(cb, fb)`.
///
/// Two halves. **Nominally**, the subtype's class must be the supertype's or below
/// it -- the question the leaf oracle already answers for an isinstance atom, so
/// it is asked rather than left to a fallthrough that cannot match this variant.
/// **By attribute**, every attribute the supertype declares must be carried by the
/// subtype with a narrower schema; all attributes of an attribute schema are
/// required, so width and depth are the whole of it.
///
/// Sound on the same assumption the isinstance atom already makes: that a
/// subclass's instances are instances of the base.
fn attrs_subtype(
    ca: ClassIx,
    fa: &[Field],
    cb: ClassIx,
    fb: &[Field],
    cx: SubtypeCx<'_>,
    assumptions: &mut Vec<(Schema, Schema)>,
) -> bool {
    let nominal = ca == cb
        || cx
            .oracle
            .leaf_subtype(&Schema::Instance(ca), &Schema::Instance(cb))
            == Some(true);
    if !nominal {
        return false;
    }
    let a_by_name = field_index(fa);
    fb.iter().all(|b| {
        a_by_name
            .get(b.name.as_str())
            .is_some_and(|a| a.schema.is_subtype_rec(&b.schema, cx, assumptions))
    })
}

/// Index a field list by name for O(1) cross-list lookup during subtyping.
///
/// Unique field names are a hard caller invariant: `collect` into a map keeps the
/// last entry per key, so a duplicate name would silently shadow an earlier field
/// and could make the `required`/width checks that consume this index unsound. The
/// frontend rejects duplicates; the `debug_assert` makes that dependency explicit
/// and catches a malformed IR in debug rather than deciding on a shadowed field.
fn field_index(fields: &[Field]) -> FxHashMap<&str, &Field> {
    let index: FxHashMap<&str, &Field> = fields.iter().map(|f| (f.name.as_str(), f)).collect();
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
mod budget_tests {
    use super::{DECISION_BUDGET, spend};
    use std::cell::Cell;

    /// The work ceiling one decision query may spend. An exhausted budget must
    /// answer `false`, because that is the whole signal a budgeted decision has
    /// for stopping and reporting the conservative answer.
    ///
    /// Driven here rather than through a decision, deliberately: a budget that
    /// never exhausts makes an adversarial schema run without bound, so the
    /// experiment would not finish and a timeout is a rig fault, not a
    /// detection. One unit of the counter is the whole of what there is to test.
    #[test]
    fn an_exhausted_budget_refuses_to_spend() {
        let budget = Cell::new(2u32);
        assert!(spend(&budget));
        assert_eq!(budget.get(), 1);
        assert!(spend(&budget));
        assert_eq!(budget.get(), 0);
        // Exhausted: refuses, and does not wrap round to a fresh budget.
        assert!(!spend(&budget));
        assert_eq!(budget.get(), 0);
        assert!(!spend(&budget));
        assert_eq!(budget.get(), 0);

        // A budget of zero refuses on its first call.
        assert!(!spend(&Cell::new(0)));
        // And the shipped ceiling admits a real query: a budget of one spends
        // once and then refuses, so a ceiling of zero would refuse every query
        // before it started.
        let shipped = Cell::new(DECISION_BUDGET);
        assert!(spend(&shipped));
        assert_eq!(shipped.get(), DECISION_BUDGET - 1);
    }
}

#[cfg(test)]
mod region_tests {
    use super::{Kind, Region, Regions};

    /// Every region operation is a set operation, and each is pinned here rather
    /// than inside the folds that use it. The bits used to be combined at each
    /// call site with a raw `|`, `&`, `!`, or `|=`, where the wrong operator is a
    /// one-character defect no test in that fold could distinguish; concentrating
    /// them into five methods is only worth it if the five are tested, so they
    /// are, over the boundary cases the folds start and end at.
    #[test]
    fn the_region_operations_are_the_set_operations() {
        let a = Kind::Bool.region().union(Kind::Int.region());
        let b = Kind::Bool.region().union(Kind::Str.region());

        // Union and intersection, distinguished: a wrong operator swaps these.
        assert_eq!(
            a.union(b),
            Kind::Bool
                .region()
                .union(Kind::Int.region())
                .union(Kind::Str.region())
        );
        assert_eq!(a.intersect(b), Kind::Bool.region());
        assert_ne!(a.union(b), a.intersect(b));

        // The bounds are the identities of their operations, and absorb the other.
        assert_eq!(a.union(Region::EMPTY), a);
        assert_eq!(a.intersect(Region::ALL), a);
        assert_eq!(a.union(Region::ALL), Region::ALL);
        assert_eq!(a.intersect(Region::EMPTY), Region::EMPTY);

        // The complement is bounded to the partition, so no result ever carries
        // the unused eighth bit, and it is an involution.
        assert_eq!(Region::EMPTY.complement(), Region::ALL);
        assert_eq!(Region::ALL.complement(), Region::EMPTY);
        assert_eq!(a.complement().complement(), a);
        assert_eq!(a.intersect(a.complement()), Region::EMPTY);
        assert_eq!(a.union(a.complement()), Region::ALL);

        // Emptiness is the fold's own verdict, so it is pinned on both sides.
        assert!(Region::EMPTY.is_empty());
        assert!(!Region::ALL.is_empty());
        assert!(!a.is_empty());
        assert!(a.intersect(Kind::Str.region()).is_empty());

        // Inclusion is the subtyping relation on the scalar fragment, and it is
        // not symmetric: `bool` is below `int`, and `int` is not below `bool`.
        assert!(Kind::Bool.region().subset_of(a));
        assert!(!a.subset_of(Kind::Bool.region()));
        assert!(a.subset_of(a));
        assert!(Region::EMPTY.subset_of(a));
        assert!(a.subset_of(Region::ALL));
        assert!(!Kind::Str.region().subset_of(a));
    }

    /// The region set an emptiness fold accumulates is a monoid under each lattice
    /// operation, and `Unknown` absorbs both. The absorbing element is what lets a
    /// fold over members stop: once it appears, no later member can change the
    /// result, and the walk that continues spends the decision budget on an answer
    /// already fixed.
    #[test]
    fn the_region_set_is_a_monoid_with_an_absorbing_element() {
        let known = |r| Regions::Known(r);
        let unknown = Regions::Unknown;
        let bool_int = known(Kind::Bool.region().union(Kind::Int.region()));

        // Each operation has its identity.
        assert_eq!(bool_int.union(Regions::UNION_UNIT), bool_int);
        assert_eq!(bool_int.intersect(Regions::MEET_UNIT), bool_int);

        // Unknown absorbs both operations, from either side.
        for combine in [Regions::union, Regions::intersect] {
            assert_eq!(combine(unknown, bool_int), unknown);
            assert_eq!(combine(bool_int, unknown), unknown);
            assert_eq!(combine(unknown, unknown), unknown);
        }

        // Only the absorbing element reports itself as one, so a fold cannot stop
        // on a known region it still has to combine.
        assert!(unknown.is_absorbing());
        assert!(!Regions::UNION_UNIT.is_absorbing());
        assert!(!Regions::MEET_UNIT.is_absorbing());
        assert!(!bool_int.is_absorbing());

        // The operations are the region's own where both sides are known.
        assert_eq!(
            known(Kind::Bool.region()).union(known(Kind::Str.region())),
            known(Kind::Bool.region().union(Kind::Str.region()))
        );
        assert_eq!(
            bool_int.intersect(known(Kind::Bool.region())),
            known(Kind::Bool.region())
        );
    }

    /// The six scalar regions and the non-scalar remainder partition the
    /// universe: they are pairwise disjoint and together cover it. Emptiness
    /// soundness rests on the cover — the meet of all six scalar complements must
    /// be the non-empty non-scalar region, not the empty set.
    #[test]
    fn every_kind_lands_in_the_partition_and_the_scalars_land_apart() {
        // The claim the region fold rests on, read off the kinds rather than off
        // a second list beside them. Six scalar kinds, each with a region of its
        // own; five container kinds sharing the one that is left. A kind added
        // without a region fails to compile, and one given a region that
        // collapses or collides fails here.
        let scalars = [
            Kind::NoneType,
            Kind::Bool,
            Kind::Int,
            Kind::Float,
            Kind::Str,
            Kind::Bytes,
        ];
        let containers = [
            Kind::List,
            Kind::Tuple,
            Kind::Set,
            Kind::FrozenSet,
            Kind::Dict,
        ];

        // Non-empty and pairwise disjoint together force six distinct bits, which
        // is what makes the partition a partition. Either half alone is satisfied
        // by a region that collapsed to nothing.
        for (i, one) in scalars.iter().enumerate() {
            assert!(!one.region().is_empty(), "{one:?} has no region");
            for other in &scalars[i + 1..] {
                assert!(
                    one.region().intersect(other.region()).is_empty(),
                    "{one:?} and {other:?} share a region"
                );
            }
        }

        // The container kinds share one region, and it is none of the scalars'.
        // No schema names it alone -- `list[int]` is a proper part of the lists --
        // which is why one bit carries all five.
        let non_scalar = Kind::List.region();
        for kind in containers {
            assert_eq!(kind.region(), non_scalar, "{kind:?} left the shared region");
        }
        for kind in scalars {
            assert!(kind.region().intersect(non_scalar).is_empty());
        }

        // The scalars do not cover the universe: what is left is the region a
        // complement must keep, which is what makes the meet of all six scalar
        // complements inhabited rather than empty.
        let union = scalars
            .iter()
            .fold(Region::EMPTY, |acc, kind| acc.union(kind.region()));
        assert_ne!(union, Region::ALL);
        assert_eq!(union.complement(), non_scalar);
        let all_complements = scalars.iter().fold(Region::ALL, |acc, kind| {
            acc.intersect(kind.region().complement())
        });
        assert!(!all_complements.is_empty());
        assert_eq!(all_complements, union.complement());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The structural inclusion procedure alone, with no descriptor beside it.
    ///
    /// `is_subtype_of_under` asks the descriptor where the rules decline, which
    /// is what widens the public relations -- and what makes a defect in a rule
    /// invisible through them, since the answer comes out right for the other
    /// reason. A rule is pinned by asking it on its own.
    fn structural(sub: &Schema, sup: &Schema) -> bool {
        let budget = Cell::new(DECISION_BUDGET);
        sub.is_subtype_rec(
            sup,
            SubtypeCx {
                oracle: &NoLeafRelations,
                defs: &[],
                budget: &budget,
            },
            &mut Vec::new(),
        )
    }

    /// Every set is below the universe, however the universe is spelled.
    ///
    /// A refinement with no constraint denotes exactly its base, and until it
    /// said so the two spellings were decided differently: `Anything` reached
    /// `Refine { base: Anything }` through the refinement rule, and the gradual
    /// `Any` reached `Anything` through the region bound but not the refinement,
    /// because only a region set carries that bound. The fuzzer found it as a
    /// union holding both.
    #[test]
    fn every_set_is_below_the_universe_however_it_is_spelled() {
        let bare = Schema::Refine {
            base: Box::new(Schema::Anything),
            constraints: Vec::new(),
        };
        // The complement of the universe is empty, which is the same fact read
        // through the *other* fold: `region_set` decides the inclusion above,
        // and `empty_and_region` decides this. Both carry the rule, so both are
        // asked -- a fix in one of two folds is half a fix.
        assert!(Schema::Complement(Box::new(bare.clone())).is_empty());
        assert!(!bare.is_empty(), "and the universe itself is not");

        for universe in [Schema::Anything, bare.clone(), Schema::Union(vec![bare])] {
            for sub in [
                Schema::Dynamic,
                Schema::Anything,
                Schema::Int,
                Schema::Union(vec![Schema::Dynamic, Schema::Anything]),
            ] {
                assert!(
                    sub.is_subtype_of(&universe),
                    "{sub:?} is below the universe {universe:?}"
                );
            }
        }
    }

    /// The structural inclusion rules, held to their own work.
    ///
    /// The descriptor is asked after these rules and decides much of what they
    /// do, so a defect in one is invisible through `is_subtype_of` -- the answer
    /// comes out right for the other reason. `is_subtype_of_under` is the
    /// structural procedure alone, which is where each rule has to be pinned.
    #[test]
    fn the_structural_inclusion_rules_decide_without_the_descriptor() {
        // Built raw rather than through the smart constructors, which fold a
        // meet of two kinds before it ever reaches the rule under test.
        let meet = |members: [Schema; 2]| Schema::Intersection(members.to_vec());
        let joined = Schema::union([Schema::Int, Schema::Float]);

        // `A ⊆ (Y ∩ Z)` needs both conjuncts.
        assert!(structural(
            &Schema::Int,
            &meet([Schema::Int, joined.clone()])
        ));
        assert!(!structural(
            &Schema::Int,
            &meet([Schema::Int, Schema::Float])
        ));
        // `(A ∩ B) ⊆ C` when some conjunct already is, and when the meet lands
        // in one branch of a union supertype -- the second half of that arm.
        assert!(structural(&meet([Schema::Int, Schema::Str]), &Schema::Int));
        assert!(structural(
            &meet([Schema::Int, Schema::Bytes]),
            &Schema::union([Schema::Int, Schema::Str])
        ));
        // An inhabited meet that no branch covers: an empty one would be a
        // subtype of everything and would say nothing about the rule.
        assert!(!structural(
            &meet([Schema::Int, joined.clone()]),
            &Schema::union([Schema::Float, Schema::Str])
        ));

        // Set and frozenset inclusion reduces to element inclusion, and the two
        // kinds do not cross.
        let ints = Schema::Set(Box::new(Schema::Int));
        assert!(structural(&ints, &Schema::Set(Box::new(joined.clone()))));
        assert!(!structural(
            &ints,
            &Schema::FrozenSet(Box::new(joined.clone()))
        ));
        assert!(structural(
            &Schema::FrozenSet(Box::new(Schema::Int)),
            &Schema::FrozenSet(Box::new(joined.clone()))
        ));

        // Each rule above is a *shortcut*: delete it and the general reduction
        // `A ⊆ B` to `A ∩ ¬B = ∅` decides the same thing. What the shortcut is
        // for is the case where that reduction declines -- and an atom carrying
        // a callback is exactly one, since the complement law does not hold of
        // it. So each rule is pinned over a schema the reduction cannot read.
        let opaque = Schema::Refine {
            base: Box::new(Schema::Int),
            constraints: vec![Constraint::Predicate(PredIx::new(0))],
        };
        assert!(structural(&opaque, &meet([opaque.clone(), opaque.clone()])));
        assert!(structural(&meet([opaque.clone(), Schema::Str]), &opaque));
        assert!(structural(
            &meet([opaque.clone(), Schema::Str]),
            &Schema::union([opaque.clone(), Schema::Float])
        ));

        // Complement is contravariant: the inclusion under it runs the other way.
        let wider = Schema::Complement(Box::new(Schema::Int));
        let narrower = Schema::Complement(Box::new(joined));
        assert!(structural(&narrower, &wider));
        assert!(!structural(&wider, &narrower));
        // And over an atom the reduction cannot read, where nothing else does.
        assert!(structural(
            &Schema::Complement(Box::new(opaque.clone())),
            &Schema::Complement(Box::new(meet([opaque.clone(), Schema::Str])))
        ));
    }

    /// A fixed-arity sequence splits across the branches of a union, decided by
    /// the product rule rather than by the descriptor beside it.
    #[test]
    fn a_product_splits_across_a_union_without_the_descriptor() {
        let pair = |a: Schema, b: Schema| Schema::tuple(SeqShape::fixed([a, b]));
        let subject = pair(Schema::union([Schema::Int, Schema::Str]), Schema::Int);
        let split = Schema::union([
            pair(Schema::Int, Schema::Int),
            pair(Schema::Str, Schema::Int),
        ]);

        assert!(structural(&subject, &split), "the branches cover it");
        assert!(
            !structural(
                &subject,
                &Schema::union([
                    pair(Schema::Int, Schema::Int),
                    pair(Schema::Bytes, Schema::Int),
                ])
            ),
            "and branches that do not cover it decide no"
        );
        // A repeated tail is not a product, so there is nothing to split.
        let variadic = Schema::tuple(SeqShape::homogeneous(Schema::union([
            Schema::Int,
            Schema::Str,
        ])));
        assert!(!structural(
            &variadic,
            &Schema::union([
                Schema::tuple(SeqShape::homogeneous(Schema::Int)),
                Schema::tuple(SeqShape::homogeneous(Schema::Str)),
            ])
        ));
    }

    /// A callback hides behind every container, and the walk that looks for one
    /// must enter each.
    ///
    /// `A ∩ ¬A = ∅` is declined for an atom that is not a set, and a predicate
    /// nested inside a container makes the whole container one -- so each way a
    /// schema holds another is a way the search must descend.
    #[test]
    fn a_callback_is_found_through_every_container() {
        let predicate = Schema::Refine {
            base: Box::new(Schema::Int),
            constraints: vec![Constraint::Predicate(PredIx::new(0))],
        };
        let field = |schema: Schema| Field {
            name: "x".to_owned(),
            schema,
            required: true,
        };
        let wrappers: [Schema; 11] = [
            predicate.clone(),
            Schema::Set(Box::new(predicate.clone())),
            Schema::FrozenSet(Box::new(predicate.clone())),
            Schema::Complement(Box::new(predicate.clone())),
            Schema::union([predicate.clone(), Schema::Int]),
            Schema::list(SeqShape::fixed([predicate.clone()])),
            Schema::list(SeqShape::homogeneous(predicate.clone())),
            Schema::KeyedMap {
                fields: vec![field(predicate.clone())],
                defaults: Vec::new(),
            },
            Schema::KeyedMap {
                fields: Vec::new(),
                defaults: vec![MapClause {
                    key: Schema::Str,
                    value: predicate.clone(),
                }],
            },
            Schema::KeyedMap {
                fields: Vec::new(),
                defaults: vec![MapClause {
                    key: predicate.clone(),
                    value: Schema::Str,
                }],
            },
            Schema::Attrs {
                class_index: ClassIx::new(0),
                fields: vec![field(predicate.clone())],
            },
        ];
        for wrapper in wrappers {
            assert!(
                !denotes_a_set(&wrapper, &NoLeafRelations),
                "a predicate inside {wrapper:?} is still a predicate"
            );
            let meet = Schema::Intersection(vec![
                wrapper.clone(),
                Schema::Complement(Box::new(wrapper.clone())),
            ]);
            assert!(
                !meet.is_empty_under(&[]),
                "so the law must decline {wrapper:?}"
            );
        }

        // Without one, the same shapes are sets and the law still decides.
        let plain = Schema::Set(Box::new(Schema::Int));
        assert!(denotes_a_set(&plain, &NoLeafRelations));
        assert!(
            Schema::Intersection(vec![plain.clone(), Schema::Complement(Box::new(plain))])
                .is_empty_under(&[])
        );
    }

    /// An oracle that reports every atom it is asked about as a set, standing
    /// for the bindings' answer about a pure class.
    struct Pure;

    impl LeafRelations for Pure {
        fn leaf_subtype(&self, _sub: &Schema, _sup: &Schema) -> Option<bool> {
            None
        }

        fn atom_denotes_a_set(&self, _atom: &Schema) -> Option<bool> {
            Some(true)
        }
    }

    /// A class is referred to the oracle, and a core with none declines it.
    ///
    /// The default answers nothing, which is what makes an unwired core
    /// conservative rather than wrong: `Instance ∩ ¬Instance` is not decided
    /// empty until the bindings say the class is pure.
    #[test]
    fn a_class_without_an_oracle_is_not_a_set() {
        let class = Schema::Instance(ClassIx::new(0));
        assert_eq!(NoLeafRelations.atom_denotes_a_set(&class), None);
        assert!(!denotes_a_set(&class, &NoLeafRelations));

        assert!(denotes_a_set(&class, &Pure));
    }

    /// The structural region rules decide without the descriptor, and stay
    /// pinned where it would otherwise answer for them.
    ///
    /// `is_empty` asks the descriptor after these rules, so a defect in them is
    /// invisible through it -- the answer comes out right for the other reason.
    /// `is_empty_under` is the structural procedure alone, which is where a rule
    /// has to be held to its own work.
    #[test]
    fn the_scalar_regions_decide_a_disjoint_meet_without_the_descriptor() {
        for (left, right) in [
            (Schema::NoneType, Schema::Str),
            (Schema::Str, Schema::Float),
            (Schema::NoneType, Schema::Bytes),
        ] {
            let meet = Schema::meet([left.clone(), right.clone()]);
            assert!(
                meet.is_empty_under(&[]),
                "{left:?} and {right:?} share no region"
            );
        }
    }

    /// A sequence with an uninhabited prefix element admits nothing, decided by
    /// the structural rule rather than by the descriptor beside it.
    #[test]
    fn an_uninhabited_prefix_empties_a_sequence_without_the_descriptor() {
        let empty_element = Schema::list(SeqShape::fixed([Schema::Nothing]));
        assert!(empty_element.is_empty_under(&[]));

        // The tail is not a prefix: a sequence may stop before it, so an
        // uninhabited tail leaves the sequence that ends at the prefix.
        let empty_tail = Schema::list(SeqShape::homogeneous(Schema::Nothing));
        assert!(!empty_tail.is_empty_under(&[]));
    }

    use crate::ir::{ClassIx, ConstIx, PredIx};
    use proptest::prelude::*;

    /// A generator mixing the scalar-decidable atoms with opaque leaves (the
    /// gradual `Any`, a literal, a content-bearing set) under the Boolean
    /// combinators, so both the region-carrying and the region-`None` paths and
    /// their propagation through every combinator are exercised.
    fn schema() -> impl Strategy<Value = Schema> {
        let leaf = prop_oneof![
            Just(Schema::Anything),
            Just(Schema::Nothing),
            Just(Schema::NoneType),
            Just(Schema::Bool),
            Just(Schema::Int),
            Just(Schema::Float),
            Just(Schema::Str),
            Just(Schema::Bytes),
            Just(Schema::Dynamic),
            Just(Schema::Literal(ConstIx::new(0))),
            Just(Schema::Set(Box::new(Schema::Int))),
        ];
        leaf.prop_recursive(4, 24, 3, |inner| {
            prop_oneof![
                proptest::collection::vec(inner.clone(), 1..4).prop_map(Schema::Union),
                proptest::collection::vec(inner.clone(), 1..4).prop_map(Schema::Intersection),
                inner.prop_map(|s| Schema::Complement(Box::new(s))),
            ]
        })
    }

    proptest! {
        /// Asking whether a schema covers the universe by reading its region is the
        /// same question as asking whether its complement is empty. The lattice
        /// bound `A subset-of U` is decided the second way in principle and the
        /// first way in the code, because the first builds nothing; this holds the
        /// two together so the cheaper one cannot drift from the rule it stands in
        /// for.
        #[test]
        fn covering_the_universe_is_the_complement_being_empty(s in schema()) {
            let budget = Cell::new(DECISION_BUDGET);
            let via_complement = Schema::Complement(Box::new(s.clone()))
                .is_empty_rec(&NoLeafRelations, &[], &mut Vec::new(), &budget);
            let budget = Cell::new(DECISION_BUDGET);
            let (_, regions) =
                s.empty_and_region(&NoLeafRelations, &[], &mut Vec::new(), &budget);
            prop_assert_eq!(via_complement, regions == Regions::Known(Region::ALL));
        }

        /// The bottom-up region folded by `empty_and_region` is exactly the region
        /// `region_set` recomputes from scratch, for every schema. This pins the two
        /// region code paths together so a future change to one cannot silently
        /// diverge from the other (the emptiness decision relies on their agreement).
        #[test]
        fn empty_and_region_folds_the_same_region_as_region_set(s in schema()) {
            let folded = s
                .empty_and_region(
                    &NoLeafRelations,
                    &[],
                    &mut Vec::new(),
                    &Cell::new(DECISION_BUDGET),
                )
                .1;
            prop_assert_eq!(folded, s.region_set());
        }
    }

    /// A union fold stops only once a member is known inhabited. The stopping rule
    /// reads the *inhabited* accumulator, not the empty one: a member that is
    /// empty and opaque leaves the verdict open, and breaking there would report a
    /// union empty on the strength of the members walked so far.
    ///
    /// The witness needs a member that is empty with an unknown region -- a record
    /// with an uninhabited required field -- followed by an inhabited one, because
    /// a member that is empty with a *known* region cannot make the accumulator
    /// absorbing on its own.
    /// The unordered pairs of a slice: every distinct pair once, in neither order
    /// twice, and none of an element with itself. Both disjointness laws scan them
    /// -- over members, and over the inners of complements -- so the scan is one
    /// function and the law it serves is decided in one place.
    #[test]
    fn unordered_pairs_yields_each_distinct_pair_once() {
        let pairs: Vec<(i32, i32)> = unordered_pairs(&[1, 2, 3]).map(|(a, b)| (*a, *b)).collect();
        assert_eq!(pairs, [(1, 2), (1, 3), (2, 3)]);
        // The degenerate lengths a member list can reach: no pair to compare.
        assert_eq!(unordered_pairs::<i32>(&[]).count(), 0);
        assert_eq!(unordered_pairs(&[1]).count(), 0);
        // n elements give n*(n-1)/2 pairs, so nothing is visited twice.
        assert_eq!(unordered_pairs(&[1, 2, 3, 4, 5]).count(), 10);
    }

    #[test]
    fn a_union_fold_stops_only_once_a_member_is_inhabited() {
        let uninhabited = Schema::record(
            vec![Field {
                name: "a".to_owned(),
                schema: Schema::Nothing,
                required: true,
            }],
            crate::ir::Openness::Closed,
        );
        assert!(uninhabited.is_empty());
        assert_eq!(uninhabited.region_set(), Regions::Unknown);

        // The union is inhabited by its second member, which the fold reaches only
        // by not stopping at the first.
        let union = Schema::union([uninhabited.clone(), Schema::Int]);
        assert!(!union.is_empty());
        // Every member uninhabited is still empty, so the stop does not hide that.
        assert!(Schema::union([uninhabited.clone(), uninhabited]).is_empty());
    }

    #[test]
    fn is_empty_decides_complement_and_disjoint_intersections() {
        let list = |e| Schema::list(SeqShape::homogeneous(e));
        let not = |s| Schema::Complement(Box::new(s));

        // A ∩ ¬A is empty for a structural A the scalar region bitset cannot see.
        let a = list(Schema::Int);
        assert!(Schema::Intersection(vec![a.clone(), not(a)]).is_empty());

        // The gradual `Any` is exempt: `Any ∩ ¬Any` is not empty.
        assert!(!Schema::Intersection(vec![Schema::Dynamic, not(Schema::Dynamic)]).is_empty());

        // Disjoint structural kinds: a list is never a set.
        assert!(
            Schema::Intersection(vec![list(Schema::Int), Schema::Set(Box::new(Schema::Int))])
                .is_empty()
        );

        // A refined int is still an int, disjoint from str.
        assert!(
            Schema::Intersection(vec![
                Schema::Refine {
                    base: Box::new(Schema::Int),
                    constraints: vec![Constraint::Ge(OperandIx::new(0))],
                },
                Schema::Str,
            ])
            .is_empty()
        );

        // Sanity: two same-kind lists share the empty list, so not empty.
        assert!(!Schema::Intersection(vec![list(Schema::Int), list(Schema::Bool)]).is_empty());
    }

    /// The two rules that read only the left side -- a reference unfolds, a
    /// refinement drops to its base -- and the union rule beside them.
    ///
    /// A union on the right is tried branch by branch, which commits to a
    /// branch. Both readings are sound and neither subsumes the other, so where
    /// both apply both are asked; these are the cases that separate them.
    #[test]
    fn a_reference_and_a_refinement_are_read_beside_the_union_rule() {
        let defs = vec![Schema::union([Schema::Int, Schema::Str])];
        let reference = Schema::Ref(DefIx::new(0));
        let body = Schema::union([Schema::Int, Schema::Str]);
        let refined = |base: Schema| Schema::Refine {
            base: Box::new(base),
            constraints: vec![Constraint::Ge(OperandIx::new(0))],
        };

        // Against a union: only the left-side rule decides these. The reference
        // is in no branch of its own body, and the refinement is in neither
        // branch of the union it refines.
        assert!(reference.is_subtype_of_under(&body, &NoLeafRelations, &defs));
        assert!(refined(body.clone()).is_subtype_of(&body));

        // Against a non-union: the same two rules, reached through their own
        // arms rather than through the union rule.
        assert!(Schema::Ref(DefIx::new(0)).is_subtype_of_under(
            &Schema::union([Schema::Int, Schema::Str, Schema::Bytes]),
            &NoLeafRelations,
            &defs
        ));
        assert!(refined(Schema::Int).is_subtype_of(&Schema::Int));
        let int_def = vec![Schema::Int];
        assert!(Schema::Ref(DefIx::new(0)).is_subtype_of_under(
            &Schema::Int,
            &NoLeafRelations,
            &int_def
        ));

        // The union rule is not replaced by them. A branch equal to the subject
        // settles it, and the base rule would answer no: `int` is not below the
        // refinement, so a subject that IS the refinement is decided only by the
        // branch it equals.
        let narrowed = refined(Schema::Int);
        assert!(narrowed.is_subtype_of(&Schema::union([narrowed.clone(), Schema::Str])));
        // And a subject that neither rule reaches still lands in its branch.
        assert!(Schema::Int.is_subtype_of(&Schema::union([Schema::Int, Schema::Str])));

        // Sound in the other direction: unfolding a reference is not a licence.
        assert!(!Schema::Ref(DefIx::new(0)).is_subtype_of_under(
            &Schema::union([Schema::Int, Schema::Bytes]),
            &NoLeafRelations,
            &defs
        ));
        // A reference no definition resolves decides nothing.
        assert!(!Schema::Ref(DefIx::new(9)).is_subtype_of_under(&body, &NoLeafRelations, &defs));
    }

    /// A field, spelled once rather than at each of the many sites below.
    fn field(name: &str, schema: Schema, required: bool) -> Field {
        Field {
            name: name.to_owned(),
            schema,
            required,
        }
    }

    /// A record: declared fields and no catch-all clause, so it is closed.
    fn closed(fields: Vec<Field>) -> Schema {
        Schema::KeyedMap {
            fields,
            defaults: Vec::new(),
        }
    }

    /// A record with a catch-all clause, which is what makes it open.
    fn open(fields: Vec<Field>) -> Schema {
        Schema::KeyedMap {
            fields,
            defaults: vec![MapClause {
                key: Schema::Str,
                value: Schema::Anything,
            }],
        }
    }

    fn meet_is_empty(members: &[Schema]) -> bool {
        keyed_map_meet_empty(members, &NoLeafRelations, &[], &Cell::new(DECISION_BUDGET))
    }

    /// The two rules ICFP formulae (11) and (12) give for a meet of record atoms,
    /// and the footnote-11 guard that stops each from firing where it must not.
    ///
    /// Driven against the rule rather than through `is_empty`, because the
    /// question is which of these shapes the rule answers for: reached through
    /// the decision procedure, an intersection has half a dozen other reasons to
    /// be reported empty, and a test that only watches the verdict cannot tell
    /// which one spoke.
    #[test]
    fn a_record_meet_is_empty_only_where_a_required_key_cannot_hold() {
        // Rule one: a key required somewhere whose types meet to nothing.
        assert!(meet_is_empty(&[
            closed(vec![field("a", Schema::Int, true)]),
            closed(vec![field("a", Schema::Str, true)]),
        ]));
        // The types must actually meet to nothing. `bool` is below `int`, so
        // these two agree on every bool.
        assert!(!meet_is_empty(&[
            closed(vec![field("a", Schema::Int, true)]),
            closed(vec![field("a", Schema::Bool, true)]),
        ]));

        // Rule two: a key required somewhere and absent from a closed map.
        assert!(meet_is_empty(&[
            closed(vec![field("a", Schema::Int, true)]),
            closed(vec![field("b", Schema::Int, true)]),
        ]));
        // The map that lacks the key must be closed. A clause admits keys the
        // field list does not name, so an open map is no obstacle.
        assert!(!meet_is_empty(&[
            closed(vec![field("a", Schema::Int, true)]),
            open(vec![]),
        ]));
        // And a closed map that declares the key is no obstacle either, which is
        // the same walk reading the field list the other way.
        assert!(!meet_is_empty(&[
            closed(vec![field("a", Schema::Int, true)]),
            closed(vec![field("a", Schema::Anything, true)]),
        ]));

        // Footnote 11: only a REQUIRED key can empty a meet. Two optional fields
        // whose types share nothing still admit the empty dict, and so does a
        // key absent from a closed map when nothing requires it.
        assert!(!meet_is_empty(&[
            closed(vec![field("a", Schema::Int, false)]),
            closed(vec![field("a", Schema::Str, false)]),
        ]));
        assert!(!meet_is_empty(&[
            closed(vec![field("a", Schema::Int, false)]),
            closed(vec![field("b", Schema::Int, false)]),
        ]));
        // Required on ONE side is enough: the meet must satisfy both maps, so a
        // key one of them demands is a key every value in the meet carries.
        assert!(meet_is_empty(&[
            closed(vec![field("a", Schema::Int, true)]),
            closed(vec![field("a", Schema::Str, false)]),
        ]));

        // The first rule is about types from DIFFERENT maps meeting to nothing.
        // A required key whose type is uninhabited in a single map empties that
        // map on its own, and the keyed-map node's own rule decides it -- so
        // this one declines rather than answering a question already answered.
        assert!(!meet_is_empty(&[
            closed(vec![field("a", Schema::Nothing, true)]),
            open(vec![]),
        ]));
        assert!(
            Schema::Intersection(vec![
                closed(vec![field("a", Schema::Nothing, true)]),
                open(vec![]),
            ])
            .is_empty()
        );

        // Fewer than two maps is not a meet of maps. One map alone is decided by
        // the node's own rule, and a meet with a non-map member says nothing
        // about the keys.
        assert!(!meet_is_empty(&[closed(vec![field(
            "a",
            Schema::Nothing,
            true
        )])]));
        assert!(!meet_is_empty(&[
            closed(vec![field("a", Schema::Int, true)]),
            Schema::Str,
        ]));
        // Three maps: the pair that cannot hold together need not be the first
        // two, so the scan runs over all of them rather than stopping at a count.
        assert!(meet_is_empty(&[
            open(vec![]),
            closed(vec![field("a", Schema::Int, true)]),
            closed(vec![field("a", Schema::Str, true)]),
        ]));
    }

    /// A fixed-arity sequence is a product, and a product is decided against a
    /// union of products by the backtrack-free `Phi` -- so a value that lands in
    /// no single branch is still decided, which is the whole reason the rule
    /// exists.
    #[test]
    fn a_fixed_sequence_splits_across_the_branches_that_share_its_shape() {
        let tuple = |elements: [Schema; 2]| Schema::tuple(SeqShape::fixed(elements));
        let int_or_str = Schema::union([Schema::Int, Schema::Str]);

        // The split: neither branch contains the subject, and together they do.
        let subject = tuple([int_or_str.clone(), Schema::Int]);
        let split = Schema::union([
            tuple([Schema::Int, Schema::Int]),
            tuple([Schema::Str, Schema::Int]),
        ]);
        assert!(subject.is_subtype_of(&split));
        // Sound in the other direction: branches that do not cover it decide no.
        assert!(!subject.is_subtype_of(&Schema::union([
            tuple([Schema::Int, Schema::Int]),
            tuple([Schema::Bytes, Schema::Int]),
        ])));

        // A branch of another container kind shares no value with the subject, so
        // it drops out rather than being read as a component-wise cover.
        let list_branches = Schema::union([
            Schema::list(SeqShape::fixed([Schema::Int, Schema::Int])),
            Schema::list(SeqShape::fixed([Schema::Str, Schema::Int])),
        ]);
        assert!(!subject.is_subtype_of(&list_branches));
        // A branch of another arity drops out the same way.
        assert!(!subject.is_subtype_of(&Schema::union([
            Schema::tuple(SeqShape::fixed([Schema::Int])),
            Schema::tuple(SeqShape::fixed([Schema::Str])),
        ])));
        // With no branch of the subject's shape at all there is nothing to split
        // over, and the rule declines rather than deciding on an empty product.
        assert!(!subject.is_subtype_of(&Schema::union([Schema::Int, Schema::Str])));

        // The subject must be a product. A repeated tail admits every length, so
        // there is no tuple of components to split.
        let variadic = Schema::tuple(SeqShape::homogeneous(int_or_str.clone()));
        assert!(!variadic.is_subtype_of(&Schema::union([
            Schema::tuple(SeqShape::homogeneous(Schema::Int)),
            Schema::tuple(SeqShape::homogeneous(Schema::Str)),
        ])));
        // Nor is a branch with a tail a product, so it drops out of the branches.
        assert!(!subject.is_subtype_of(&Schema::union([
            Schema::tuple(SeqShape::prefix_tail([Schema::Int], Schema::Int)),
            tuple([Schema::Str, Schema::Int]),
        ])));

        // The empty prefix is the nullary product, and it is covered by itself.
        let nullary = Schema::tuple(SeqShape::fixed([]));
        assert!(nullary.is_subtype_of(&Schema::union([nullary.clone(), Schema::Int])));
    }

    /// An oracle that kinds a pooled constant by its index and settles a pair of
    /// them, standing in for the bindings' reading of two Python objects.
    ///
    /// Index 0 is an `int`, 1 a `str`, 2 a second `int`. Two constants are
    /// disjoint when their kinds differ or their indices do -- the same rule the
    /// bindings apply to a builtin scalar, whose equality is Python's own.
    struct Kinded;
    impl LeafRelations for Kinded {
        fn leaf_subtype(&self, _: &Schema, _: &Schema) -> Option<bool> {
            None
        }
        fn literal_kind(&self, constant: ConstIx) -> Option<Kind> {
            match constant.get() {
                0 | 2 => Some(Kind::Int),
                1 => Some(Kind::Str),
                _ => None,
            }
        }
        fn literals_disjoint(&self, left: ConstIx, right: ConstIx) -> Option<bool> {
            Some(left != right)
        }
    }

    /// What a literal contributes to disjointness, and where each answer comes
    /// from. Three rules meet at a literal and they are asked in this order: two
    /// literals go to the constants, a literal against anything else goes to the
    /// kind, and an unkinded constant declines.
    #[test]
    fn a_literal_is_disjoint_by_its_constants_then_by_its_kind() {
        let lit = |i: usize| Schema::Literal(ConstIx::new(i));

        // Two literals: the constants settle it, and the kind rule never runs --
        // which matters because the kind rule exempts bool/int and would answer
        // differently for two int constants.
        assert!(lit(0).disjoint_with(&lit(1), &Kinded));
        assert!(lit(0).disjoint_with(&lit(2), &Kinded));
        assert!(!lit(0).disjoint_with(&lit(0), &Kinded));
        // An oracle that declines leaves the pair conservative rather than
        // falling through to a rule that would answer for it.
        assert!(!lit(0).disjoint_with(&lit(1), &NoLeafRelations));

        // A literal against a kind: the constant's kind places it in the
        // partition. Without it the literal is opaque and nothing is decided.
        assert!(lit(0).disjoint_with(&Schema::Str, &Kinded));
        assert!(!lit(0).disjoint_with(&Schema::Int, &Kinded));
        assert!(!lit(0).disjoint_with(&Schema::Str, &NoLeafRelations));
        // Read the other way round too: the arms are ordered, and only one of
        // them puts the literal on the left.
        assert!(Schema::Str.disjoint_with(&lit(0), &Kinded));

        // A union is disjoint from a schema when every member is, which is how a
        // `Literal[...]` -- built as a union of its constants -- is reached at
        // all. One overlapping member is enough to decline.
        let table = Schema::union([lit(0), lit(2)]);
        assert!(table.disjoint_with(&Schema::Str, &Kinded));
        assert!(!table.disjoint_with(&Schema::Int, &Kinded));
        assert!(Schema::Str.disjoint_with(&table, &Kinded));
        assert!(!Schema::Int.disjoint_with(&table, &Kinded));
        // An empty union denotes nothing, so it is disjoint from everything --
        // by the bottom rule above these arms, not by "every member is".
        assert!(Schema::union(Vec::new()).disjoint_with(&Schema::Int, &Kinded));
    }

    /// An oracle treating each pool index as its own value, so comparing indices
    /// orders the bound values they stand for.
    struct ByIndex;
    impl LeafRelations for ByIndex {
        fn leaf_subtype(&self, _: &Schema, _: &Schema) -> Option<bool> {
            None
        }
        fn compare(&self, a: OperandIx, b: OperandIx) -> Option<core::cmp::Ordering> {
            Some(a.get().cmp(&b.get()))
        }
    }

    #[test]
    fn constraint_entailment_covers_every_ordering_arm() {
        let o = &ByIndex;
        // Ge(w): a tighter-or-equal lower bound, from Ge or Gt, entails a looser one.
        assert!(constraint_entailed(
            &Constraint::Ge(OperandIx::new(3)),
            &[Constraint::Ge(OperandIx::new(5))],
            o
        ));
        assert!(constraint_entailed(
            &Constraint::Ge(OperandIx::new(3)),
            &[Constraint::Gt(OperandIx::new(5))],
            o
        ));
        assert!(!constraint_entailed(
            &Constraint::Ge(OperandIx::new(5)),
            &[Constraint::Ge(OperandIx::new(3))],
            o
        ));
        // Gt(w): Gt(n) with n >= w, or Ge(n) with n > w.
        assert!(constraint_entailed(
            &Constraint::Gt(OperandIx::new(3)),
            &[Constraint::Gt(OperandIx::new(3))],
            o
        ));
        assert!(constraint_entailed(
            &Constraint::Gt(OperandIx::new(3)),
            &[Constraint::Ge(OperandIx::new(5))],
            o
        ));
        assert!(!constraint_entailed(
            &Constraint::Gt(OperandIx::new(5)),
            &[Constraint::Ge(OperandIx::new(5))],
            o
        ));
        // Le(w): Le(n) or Lt(n) with n <= w.
        assert!(constraint_entailed(
            &Constraint::Le(OperandIx::new(5)),
            &[Constraint::Le(OperandIx::new(3))],
            o
        ));
        assert!(constraint_entailed(
            &Constraint::Le(OperandIx::new(5)),
            &[Constraint::Lt(OperandIx::new(3))],
            o
        ));
        assert!(!constraint_entailed(
            &Constraint::Le(OperandIx::new(3)),
            &[Constraint::Le(OperandIx::new(5))],
            o
        ));
        // Lt(w): Lt(n) with n <= w, or Le(n) with n < w.
        assert!(constraint_entailed(
            &Constraint::Lt(OperandIx::new(5)),
            &[Constraint::Lt(OperandIx::new(5))],
            o
        ));
        assert!(constraint_entailed(
            &Constraint::Lt(OperandIx::new(5)),
            &[Constraint::Le(OperandIx::new(3))],
            o
        ));
        assert!(!constraint_entailed(
            &Constraint::Lt(OperandIx::new(5)),
            &[Constraint::Le(OperandIx::new(5))],
            o
        ));
        // Length bounds compare by their raw counts, no oracle needed.
        assert!(constraint_entailed(
            &Constraint::MinLen(3),
            &[Constraint::MinLen(5)],
            o
        ));
        assert!(!constraint_entailed(
            &Constraint::MinLen(5),
            &[Constraint::MinLen(3)],
            o
        ));
        assert!(constraint_entailed(
            &Constraint::MaxLen(5),
            &[Constraint::MaxLen(3)],
            o
        ));
        assert!(!constraint_entailed(
            &Constraint::MaxLen(3),
            &[Constraint::MaxLen(5)],
            o
        ));
        // A multiple-of or predicate bound has no order entailment.
        assert!(!constraint_entailed(
            &Constraint::MultipleOf(OperandIx::new(0)),
            &[Constraint::MultipleOf(OperandIx::new(0))],
            o
        ));
    }

    /// An oracle that orders pool indices as the values they stand for and reports
    /// integer adjacency by the same arithmetic the binding uses, so the
    /// discreteness rule can be driven without an interpreter.
    struct Adjacent;
    impl LeafRelations for Adjacent {
        fn leaf_subtype(&self, _: &Schema, _: &Schema) -> Option<bool> {
            None
        }
        fn compare(&self, a: OperandIx, b: OperandIx) -> Option<core::cmp::Ordering> {
            Some(a.get().cmp(&b.get()))
        }
        fn no_int_between(
            &self,
            lo: OperandIx,
            lo_strict: bool,
            hi: OperandIx,
            hi_strict: bool,
        ) -> Option<bool> {
            let least = lo.get() + usize::from(lo_strict);
            let greatest = hi.get().checked_sub(usize::from(hi_strict))?;
            Some(least > greatest)
        }
    }

    /// One set spelled two ways gets one verdict. The bounds of a refinement and
    /// the bounds gathered across an intersection are the same conjunction, so a
    /// rule that fires for one fires for the other: an intersection is a subset of
    /// every member, and a member bounded to the integers bounds the meet.
    #[test]
    fn both_meets_ask_the_same_question_of_their_base() {
        let refine = |constraints| Schema::Refine {
            base: Box::new(Schema::Int),
            constraints,
        };
        let (gt0, lt1) = (
            Constraint::Gt(OperandIx::new(0)),
            Constraint::Lt(OperandIx::new(1)),
        );
        let on_one = refine(vec![gt0.clone(), lt1.clone()]);
        let across = Schema::meet([refine(vec![gt0]), refine(vec![lt1])]);
        assert!(on_one.is_empty_with(&Adjacent, &[]));
        assert_eq!(
            on_one.is_empty_with(&Adjacent, &[]),
            across.is_empty_with(&Adjacent, &[])
        );
    }

    /// An attribute schema is its class's isinstance atom narrowed by an attribute
    /// record, so it is below that atom. Without the rule the pair reaches the
    /// leaf oracle, which matches a literal and an instance on the subtype side
    /// and cannot answer for this variant.
    #[test]
    fn an_attribute_schema_is_below_its_own_class() {
        let attrs = Schema::Attrs {
            class_index: ClassIx::new(0),
            fields: vec![Field {
                name: "a".to_owned(),
                schema: Schema::Int,
                required: true,
            }],
        };
        assert!(attrs.is_subtype_of(&Schema::Instance(ClassIx::new(0))));
        // A different class is a nominal question, and the core's default oracle
        // decides nothing, so it stays conservative.
        assert!(!attrs.is_subtype_of(&Schema::Instance(ClassIx::new(1))));
    }

    /// A boolean base is bounded to the integers, so it counts them too. Sound but
    /// not complete: the rule sees the integers in the interval, not the two
    /// values `bool` actually has, so an interval holding an integer that is
    /// neither 0 nor 1 stays conservatively non-empty.
    #[test]
    fn a_boolean_base_counts_integers() {
        let refine = |base, constraints| Schema::Refine {
            base: Box::new(base),
            constraints,
        };
        let open_unit = vec![
            Constraint::Gt(OperandIx::new(0)),
            Constraint::Lt(OperandIx::new(1)),
        ];
        assert!(refine(Schema::Bool, open_unit).is_empty_with(&Adjacent, &[]));
        // A dense base is not bounded to the integers and stays inhabited.
        let dense = vec![
            Constraint::Gt(OperandIx::new(0)),
            Constraint::Lt(OperandIx::new(1)),
        ];
        assert!(!refine(Schema::Float, dense).is_empty_with(&Adjacent, &[]));
    }

    #[test]
    fn tighter_refinement_bounds_subtype_looser_ones_through_the_oracle() {
        // The entailment feeds the refinement subtype rule: a tighter bound makes a
        // refinement a subtype of a refinement with a looser one, even when the
        // constraints are not identical (so the verbatim path does not apply).
        let refine = |constraints| Schema::Refine {
            base: Box::new(Schema::Int),
            constraints,
        };
        let tight = refine(vec![Constraint::Ge(OperandIx::new(5))]);
        let loose = refine(vec![Constraint::Ge(OperandIx::new(3))]);
        assert!(tight.is_subtype_of_under(&loose, &ByIndex, &[]));
        assert!(!loose.is_subtype_of_under(&tight, &ByIndex, &[]));
    }

    #[test]
    fn scalar_type_tags_decide_disjointness() {
        // Distinct concrete scalars are provably disjoint; bool is a subtype of
        // int, so the two overlap. This pins each scalar's own type tag: dropping
        // one would make its disjointness with every other scalar undecidable.
        assert!(Schema::Bool.disjoint(&Schema::Str));
        assert!(Schema::Int.disjoint(&Schema::Str));
        assert!(Schema::Float.disjoint(&Schema::Int));
        assert!(Schema::Bytes.disjoint(&Schema::Str));
        assert!(!Schema::Bool.disjoint(&Schema::Int));
        assert!(!Schema::Int.disjoint(&Schema::Int));
    }

    #[test]
    fn equal_bounds_keep_the_strict_end_when_narrowing() {
        // Narrowing two equal lower bounds keeps the strict one: Ge(5) ∩ Gt(5) is
        // Gt(5), so Ge(5) ∩ Gt(5) ∩ Le(5) is x > 5 ∧ x <= 5 — empty. If the strict
        // ends were combined the other way (both-strict rather than either-strict)
        // the lower bound would relax to Ge(5) and the range {5} would look
        // inhabited, so this pins the strictness combination.
        let refine = |constraints| Schema::Refine {
            base: Box::new(Schema::Int),
            constraints,
        };
        let empty = Schema::Intersection(vec![
            refine(vec![Constraint::Ge(OperandIx::new(5))]),
            refine(vec![Constraint::Gt(OperandIx::new(5))]),
            refine(vec![Constraint::Le(OperandIx::new(5))]),
        ]);
        assert!(empty.is_empty_with(&ByIndex, &[]));
        // Both bounds non-strict: the singleton {5} is inhabited.
        let inhabited = Schema::Intersection(vec![
            refine(vec![Constraint::Ge(OperandIx::new(5))]),
            refine(vec![Constraint::Le(OperandIx::new(5))]),
        ]);
        assert!(!inhabited.is_empty_with(&ByIndex, &[]));
    }

    #[test]
    fn a_union_of_disjoint_complements_simplifies_to_the_top() {
        // De Morgan: ¬A ∪ ¬B = ¬(A ∩ B), which is ⊤ when A and B are disjoint. int
        // and str are disjoint, so their complements cover the universe.
        let disjoint = Schema::Union(vec![
            Schema::Complement(Box::new(Schema::Int)),
            Schema::Complement(Box::new(Schema::Str)),
        ]);
        assert_eq!(disjoint.simplify(), Schema::Anything);
        // bool is a subtype of int, so int and bool overlap and their complements
        // do not cover the universe.
        let overlapping = Schema::Union(vec![
            Schema::Complement(Box::new(Schema::Int)),
            Schema::Complement(Box::new(Schema::Bool)),
        ]);
        assert_ne!(overlapping.simplify(), Schema::Anything);
    }
}
