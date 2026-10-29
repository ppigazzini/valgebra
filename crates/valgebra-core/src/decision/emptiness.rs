//! Emptiness: whether a schema denotes no value, as a three-valued verdict.
//!
//! The question every relation reduces to. Subtyping is the emptiness of a
//! difference and equivalence is two of those, so this is the module the other
//! two rest on. It reads the simulation of [FCB08, Def. 6.9] by recursion over
//! the constructors rather than as a saturation, a node met again on its own
//! path taken as empty [LN26, §5.1], which is the coinductive hypothesis -- a
//! union is empty when every member is, a meet when its members share no value,
//! a product when a component is -- with the value-region partition folded in
//! the same pass, so a Boolean node reads its children's regions in constant
//! time.
//!
//! The answer is three-valued because a refutation needs it: `Unknown` is not
//! a proof of emptiness and not a proof of inhabitation, and the boolean the
//! public entry points return folds it into "not proved empty". Where the rules
//! here decline, the question is asked once more of the *set* the schema
//! denotes, under the lowering's bound on what building it may cost.

use std::cell::RefCell;

use crate::descr::classes::Reach;
use crate::descr::lower::{Constants, an_empty_reading_stands, read_widened};
use crate::ir::{Attribute, ClassIx, Constraint, Constraints, Schema};
use crate::kind::{Kind, Region, Regions};
use crate::oracle::denotes_a_set_within;
use crate::verdict::Verdict;

use super::constraints::{
    Density, Multiples, bounds_unsatisfiable, longest, shortest, tightest_bounds,
};
use super::literals::{FiniteMeet, a_meet_read_by_its_constants};
use super::records::{keyed_map_meet_verdict, keyed_map_minus_keyed_maps};
use super::{
    Budget, Cx, DECISION_BUDGET, LeafRelations, NoLeafRelations, descending,
    has_complementary_pair, has_disjoint_pair, pushed_complement_is_placed, spend,
};

impl Schema {
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
    /// decides a container meet, a complement nested inside another, and a kind
    /// against its own literals. What is left undecided is what no bounded descriptor holds: an
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
        // The rules answer in three values and the descriptor is asked where
        // they reach the third -- which is what makes the pair one procedure
        // rather than a second opinion. A proof of *inhabitation* is an answer
        // like a proof of emptiness: a value of the schema is a value of it,
        // whatever a second reading would say, so the descriptor is not asked
        // to overturn what it cannot. Reading the bool instead of the verdict
        // asked it on every inhabited schema, and lowering one determinizes
        // automata and takes products -- a third of the decision workload,
        // spent on a question already answered.
        match self.verdict_rec(&mut Cx::new(oracle, defs, &Budget::new(DECISION_BUDGET))) {
            Verdict::Empty => true,
            Verdict::Inhabited => false,
            // A string subject's texts last, as the relation below `Nothing`
            // reads them, so the two answer alike.
            Verdict::Unknown => {
                self.denotes_no_value(oracle, defs)
                    || (matches!(self, Schema::Refine { .. } | Schema::Intersection(_))
                        && self.decided_by_its_texts(&Schema::Nothing, oracle).holds())
            }
        }
    }

    /// Whether the descriptor proves this schema admits no value.
    ///
    /// Asked **after** the structural rules and only where they decline, which
    /// is the ordering the cost forces: building a descriptor determinizes
    /// automata and takes products, and beside a verdict the rules already
    /// reached that work is discarded. Since it can only turn a `false` into a
    /// `true`, both orders give the same answers and only one is affordable.
    ///
    /// Asking costs a bounded amount whether or not it answers: the build is
    /// held to the nodes it may read, the nesting it may descend and the work it
    /// may spend, and past any of those it refuses. A schema the descriptor
    /// cannot hold -- a recursive one -- refuses the same way. Either way the
    /// caller keeps the verdict the rules reached.
    ///
    /// A literal key read as a label under a complement can narrow the set it
    /// should widen, and a proof over one stands only where
    /// [`an_empty_reading_stands`] says it does.
    ///
    /// A conjunct at the top the lowering refuses -- a predicate, a member
    /// nested past the depth bound -- is left out ([`read_widened`]), which
    /// reads a superset, so emptiness is monotone in a meet's members. A
    /// superset the reading can only find inhabited proves nothing here, and
    /// is not built. A proof the label check refuses is no answer, and the
    /// reading that leaves a scalar past the depth out is asked after it.
    pub(super) fn denotes_no_value(&self, pool: &dyn Constants, defs: &[Schema]) -> bool {
        read_widened(
            self,
            defs,
            pool,
            |read| !the_rules_find_a_value(read),
            |set, kept| {
                (set.emptiness() == Verdict::Empty
                    && match kept {
                        None => an_empty_reading_stands(&[self], &[], defs, pool),
                        Some(kept) => {
                            let kept: Vec<&Schema> = kept.iter().collect();
                            an_empty_reading_stands(&kept, &[], defs, pool)
                        }
                    })
                .then_some(())
            },
        )
        .is_some()
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
        let budget = Budget::new(DECISION_BUDGET);
        self.is_empty_rec(&mut Cx::new(&NoLeafRelations, &[], &budget));
        DECISION_BUDGET - budget.left()
    }

    /// The emptiness verdict where the rules can look a constant or a class up.
    #[cfg(test)]
    pub(crate) fn verdict_under(&self, oracle: &dyn LeafRelations) -> Verdict {
        self.verdict_rec(&mut Cx::new(oracle, &[], &Budget::new(DECISION_BUDGET)))
    }

    /// The rules' emptiness verdict with references resolved through `defs`.
    #[cfg(test)]
    pub(crate) fn verdict_under_defs(&self, defs: &[Schema]) -> Verdict {
        self.verdict_rec(&mut Cx::new(
            &NoLeafRelations,
            defs,
            &Budget::new(DECISION_BUDGET),
        ))
    }

    pub(super) fn is_empty_rec(&self, cx: &mut Cx<'_>) -> bool {
        self.verdict_rec(cx).is_empty()
    }

    /// What this schema's emptiness can be proven to be, under the leaf oracle
    /// and the recursive definitions.
    ///
    /// The three-valued answer behind [`is_empty`](Self::is_empty), which reduces
    /// it: an `Unknown` is not a proof of emptiness, so the public relation
    /// answers `false` for it exactly as it does for `Inhabited`. What it is for
    /// is telling those two apart from outside -- an exhausted budget is
    /// `Unknown`, and a test can say so.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn verdict(&self) -> Verdict {
        self.verdict_rec(&mut Cx::new(
            &NoLeafRelations,
            &[],
            &Budget::new(DECISION_BUDGET),
        ))
    }

    pub(super) fn verdict_rec(&self, cx: &mut Cx<'_>) -> Verdict {
        self.empty_and_region(cx).0
    }

    /// The emptiness verdict and the value-region bitset of `self`, decided in a
    /// single bottom-up pass: a Boolean node folds its children's already-computed
    /// regions in O(1) each instead of re-deriving its region by re-walking the
    /// whole subtree with [`region_set`](Self::region_set). The returned bitset is
    /// exactly what `region_set` would return ([`Regions::Unknown`] off the
    /// scalar-decidable fragment), so emptiness on the scalar fragment is
    /// decided identically, and a deeply nested intersection is decided in time
    /// linear in its size: each level folds its children's regions without
    /// re-walking the levels below.
    ///
    /// The work is bounded by the shared `budget`, so the region computation cannot
    /// run unbounded down a side door any more than the rest of the decision can;
    /// on exhaustion it returns the conservative "not proven empty" with an unknown
    /// region.
    pub(super) fn empty_and_region(&self, cx: &mut Cx<'_>) -> (Verdict, Regions) {
        // One level of the recursion, and a level past the depth bound is a
        // frame the stack may not have, which proves nothing.
        descending(cx.budget, (Verdict::Unknown, Regions::Unknown), || {
            self.empty_and_region_at(cx)
        })
    }

    /// [`empty_and_region`](Self::empty_and_region) one level down, inside the
    /// level it holds.
    ///
    /// Each arm is a rule of the coinductive proof of emptiness
    /// [FCB08, Def. 6.9], read over the constructors, with a reference met
    /// again on its own path read as empty, as a node met during its own
    /// traversal is [LN26, §5.1].
    fn empty_and_region_at(&self, cx: &mut Cx<'_>) -> (Verdict, Regions) {
        // Bound the work, sharing the budget with the caller (the subtyping
        // decision passes its own `cx.budget` in), so emptiness cannot escape the
        // ceiling subtyping advertises. Exhaustion proves nothing either way,
        // which is what `Unknown` says and what a `false` could not.
        if !spend(cx.budget) {
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
                // occurrence demands an infinite unfolding, so on its own it
                // has no finite inhabitant [LN26, §5.1]. A union base case or
                // an optional or starred position escapes before reaching here.
                if cx.visiting.contains(id) {
                    return (Verdict::Empty, Regions::Unknown);
                }
                match cx.defs.get(id.get()) {
                    Some(def) => {
                        cx.visiting.push(*id);
                        let verdict = def.verdict_rec(cx);
                        cx.visiting.pop();
                        (verdict, Regions::Unknown)
                    }
                    // A reference no definition resolves says nothing about the
                    // set it names.
                    None => (Verdict::Unknown, Regions::Unknown),
                }
            }
            // A sequence admits no value when a prefix element admits none, a
            // product being empty where a component is [FCB08, Lem. 6.5]. A
            // tail repeats zero times, so a shape whose prefix is all inhabited
            // admits at least the sequence that stops at the prefix.
            Schema::Seq { shape, .. } => {
                let prefix = shape.prefix.iter().map(|element| element.verdict_rec(cx));
                (Verdict::every(prefix), Regions::Unknown)
            }
            // A refinement is a subset of its base: an empty base empties it, and
            // so does an unsatisfiable bound conjunction (decided by the oracle).
            // A refinement with no constraint denotes exactly its base, so it
            // earns the base's verdict *and* its regions. One with a constraint
            // narrows, and a narrowed region set is an over-approximation whose
            // complement would report an inhabited schema empty -- which is why
            // every other refinement stays unknown.
            Schema::Refine { base, constraints } if constraints.is_empty() => {
                base.empty_and_region(cx)
            }
            Schema::Refine { base, constraints } => {
                (refinement_verdict(base, constraints, cx), Regions::Unknown)
            }
            Schema::Intersection(members) => intersection_verdict(members, cx),
            // A set or frozenset admits the empty collection whatever its
            // element schema is, so it is *proven* inhabited -- which two values
            // could not say apart from the opaque wildcard below, where the same
            // `false` meant only that nothing proved emptiness.
            Schema::Coll { .. } => (Verdict::Inhabited, Regions::Unknown),
            // A map is emptied by a required field that admits nothing, and
            // admits the empty dict when it requires nothing at all
            // [Cas23, (11)].
            Schema::KeyedMap { fields, .. } => {
                let required = fields.iter().filter(|field| field.required);
                let verdict = Verdict::every(required.map(|field| field.schema.verdict_rec(cx)));
                (verdict, Regions::Unknown)
            }
            // An attribute record carries no class, so its attributes decide it
            // in both directions -- the same rule as a keyed map's required
            // fields [Cas23, (11)], and for the same reason. An attribute
            // admitting nothing empties it, because no value can carry that
            // attribute; attributes that are all *proven* inhabited make it
            // inhabited, because an object carrying one witness per attribute
            // is a value of the record. That second half is what a class in the
            // same node would take away: an `isinstance` atom is opaque, so a
            // node holding both can never be more than unknown.
            Schema::AttrRecord { fields } => {
                let verdict =
                    Verdict::every(fields.iter().map(|field| field.schema.verdict_rec(cx)));
                (verdict, Regions::Unknown)
            }
            // A union is empty when every member is and inhabited when any
            // member is [FCB08, Def. 4.1]; its region is the union of the
            // members' regions, again folded from the children.
            Schema::Union(members) => {
                let mut verdict = Verdict::Empty;
                let mut region = Regions::UNION_UNIT;
                for m in members.iter() {
                    let (member, member_region) = m.empty_and_region(cx);
                    verdict = Verdict::any([verdict, member].into_iter());
                    region = region.union(member_region);
                    // A member that is not proven empty and an opaque region are
                    // both absorbing: no later member can make the union empty,
                    // and none can make the region known. The stop is on "not
                    // proven empty" rather than "proven inhabited": reading
                    // further to strengthen an `Unknown` into an `Inhabited`
                    // would spend budget on a question the caller does not ask.
                    if verdict != Verdict::Empty && region.is_absorbing() {
                        break;
                    }
                }
                (verdict, region)
            }
            // A complement's region is the partition minus its inner's region;
            // it is empty exactly when that region is empty (`¬⊤ = ∅`,
            // [FCB08, Def. 4.1]).
            //
            // The region that settles it is read by the **fast** fold, which
            // stops at the first member a union carries that the partition
            // cannot read. So `list | object` folds to an unknown region and
            // `¬(list | object)` folds to one with it, though the inner is the
            // universe and the complement is the empty set. Where the fast
            // reading leaves the verdict open the *complete* one is asked --
            // every member read and what they cover between them carried -- and
            // only there, so a complement the regions already settle pays
            // nothing for it. The reading is a member list walked against a
            // bitset; the set representation answers the same question by
            // lowering both sides, which is about three orders of magnitude
            // dearer (`scripts/perf_gate.py --decision-sets`).
            //
            // **The verdict moves and the region does not.** They are two
            // answers, and only the first of them is this reading's to give: a
            // region is what the *partition* can say about the schema, it is
            // folded into every parent, and `region_set` recomputes it from
            // scratch, so a region here that the recomputation does not reach
            // is a divergence between two paths a test holds together
            // (`empty_and_region_folds_the_same_region_as_region_set`). The
            // emptiness verdict has no such second reader.
            //
            // Where the regions leave it open, the complement is empty where
            // its inner covers the universe by the region walk past an opaque
            // member, or where its inner is a union whose complements, pushed
            // one level in [FCB08, Lem. 6.2], meet in nothing as they are
            // built: `¬(¬s ∪ ¬(j ∩ ¬s))` is `s ∩ j ∩ ¬s`. That is the relation's
            // reading of the same subject below `nothing`
            // (`pushed_complement_is_placed`), so `is_empty` and a relation to
            // the empty set read it alike; the fold alone, which spends no
            // step, since a step spent here is one a relation asking this
            // emptiness inside its own query no longer has.
            //
            // And it holds a value where its inner is a meet one of whose
            // members is a complement of a set that holds one: pushed in, the
            // complement of a meet is the union of its members' complements
            // [FCB08, Lem. 6.2], which holds that member's inner. A record has
            // no region, so `¬(¬{"a": None} ∩ ¬{})` is read this way or not at
            // all. The member's inner is read by its own shape, which spends
            // no step, as the fold above spends none.
            Schema::Complement(inner) => {
                let (_, inner_region) = inner.empty_and_region(cx);
                let region = match inner_region {
                    Regions::Known(regions) => Regions::Known(regions.complement()),
                    Regions::Unknown => Regions::Unknown,
                };
                let verdict = match region.verdict() {
                    Verdict::Unknown
                        if inner.covers_the_universe()
                            || (matches!(&**inner, Schema::Union(_))
                                && pushed_complement_is_placed(
                                    inner,
                                    &Schema::Nothing,
                                    cx.defs,
                                )) =>
                    {
                        Verdict::Empty
                    }
                    Verdict::Unknown => a_complemented_member_holds_a_value(inner),
                    settled => settled,
                };
                (verdict, region)
            }
            // A literal denotes `{x | type(x) is type(c) and x == c}`, the
            // model's singleton [FCB08, §4.2] widened to the values `==` joins
            // to `c`, which holds `c` itself exactly when `c` equals itself. A
            // constant that does not -- a `nan` -- denotes the empty set, and
            // every other constant is a value, which is what makes a refutation
            // naming one a refutation at all.
            //
            // The core cannot read a constant, and this asks the oracle the
            // question it already answers: whether two constants denote
            // disjoint singletons, asked of one constant *against itself*. A
            // set disjoint from itself is the empty one, and a set that shares
            // a value with itself has a value. The oracle declines for a type
            // whose equality it does not trust, and the literal stays unknown
            // there.
            Schema::Literal(constant) => (
                match cx.oracle.literals_disjoint(*constant, *constant) {
                    Some(true) => Verdict::Empty,
                    Some(false) => Verdict::Inhabited,
                    None => Verdict::Unknown,
                },
                // A literal is a *subset* of its kind's region rather than the
                // whole of it, so it has no region set of its own: one would
                // read `int <= Literal[1]` as proven.
                Regions::Unknown,
            ),
            // A class is a set of objects the core cannot read, so this asks
            // the oracle the one question it already answers about one:
            // whether the atom denotes a set at all -- a class whose metaclass
            // leaves `isinstance` alone. Such a class reads as **inhabited**,
            // which is the open world the set representation already works in:
            // it lowers the same class to an atom holding an object, and every
            // refutation this library makes about a class rests on that. A
            // class no value can instantiate is where the assumption is wrong,
            // and it is as wrong in the descriptor, which makes the same claim:
            // answering here spares the rules the cost of deferring to it.
            //
            // A class with a hooked metaclass is not a set here, and stays
            // unknown as it is everywhere else.
            Schema::Instance(_) => (
                match cx.oracle.atom_denotes_a_set(self) {
                    Some(true) => Verdict::Inhabited,
                    _ => Verdict::Unknown,
                },
                Regions::Unknown,
            ),
            // The marker a `recursive` body carries where its back edge will go.
            // It denotes the empty set, and every decider reads it as a schema
            // it cannot relate, so neither direction is proven of it.
            _ => (Verdict::Unknown, Regions::Unknown),
        }
    }
}

/// `Inhabited` where `inner` is a meet a member of which is the complement of
/// a set that has a value by its own shape ([`Schema::holds_a_value_shallowly`]),
/// which the complement of the meet then holds; `Unknown` otherwise.
///
/// Read without descent: the fold that reached this complement has read the
/// set already and kept no verdict of it, and reading it again would spend the
/// steps a relation asking this emptiness inside its own query has left.
pub(super) fn a_complemented_member_holds_a_value(inner: &Schema) -> Verdict {
    let Schema::Intersection(members) = inner else {
        return Verdict::Unknown;
    };
    if members
        .iter()
        .any(|member| matches!(member, Schema::Complement(set) if set.holds_a_value_shallowly()))
    {
        Verdict::Inhabited
    } else {
        Verdict::Unknown
    }
}

/// Whether the rules find a value of the meet of `members`, so that a widened
/// reading of them, a superset, holds one too and proves nothing empty.
///
/// The reading the set representation's thinned reading is spared by: asked of
/// the conjuncts it would keep, where a build costs what the decision workloads
/// spend in thousands of queries and the rules' verdict is a fraction of one.
/// Read with no oracle, since a widened reading's conjuncts are the schema's
/// with every reference cut, and a literal or a class it would need is one the
/// reading builds as before.
pub(super) fn the_rules_find_a_value(members: &[Schema]) -> bool {
    let budget = Budget::new(DECISION_BUDGET);
    let cx = &mut Cx::new(&NoLeafRelations, &[], &budget);
    let verdict = match members {
        [only] => only.verdict_rec(cx),
        members => Schema::Intersection(members.to_vec().into()).verdict_rec(cx),
    };
    verdict == Verdict::Inhabited
}

/// Whether the values a meet admits are bounded to the integers, so a bound
/// conjunction over it may count them.
///
/// Asked by both meets the IR carries -- a refinement's constraint conjunction and
/// an [`Schema::Intersection`] of refinements -- so a rule that fires for one
/// fires for the other. An intersection is a subset of every member, so one
/// member bounded to the integers bounds the whole meet; a lone base bounds it by
/// being one. `bool` counts because it subclasses `int`. A float base does not:
/// the floats between two bounds are the set of floats' to count.
///
/// Sound and not complete for `bool`: the rule counts the integers in the
/// interval rather than the two values a boolean has, so an interval holding an
/// integer that is neither zero nor one stays conservatively inhabited.
fn density_of<'a>(bases: impl IntoIterator<Item = &'a Schema>) -> Density {
    if bases
        .into_iter()
        .any(|base| matches!(base.type_tag(), Some(Kind::Int | Kind::Bool)))
    {
        Density::Discrete
    } else {
        Density::Dense
    }
}

/// The verdict and region set of an intersection.
///
/// Seven rules can prove a meet empty and they are asked in one place: an empty
/// member, canceling scalar regions, a member beside its own complement
/// [FCB08, Def. 4.1], a pattern beside the complement of one with the same
/// constraints, two members of disjoint kinds [FCB08, (5)], refinement bounds
/// that cannot hold together, and a required key two record members cannot
/// agree on, the meet of two map atoms being the atom of the meets
/// [Cas23, (12)] and empty where a required label is empty [Cas23, (11)].
///
/// Proving a meet inhabited means finding a value in *every* member, which none
/// of the seven does. The region settles it exactly over the whole scalar
/// fragment, and past that four shapes do: a direct instance of one class
/// carrying the attribute record it is met with ([`class_with_attributes`]),
/// closed maps whose required keys meet in a value
/// ([`keyed_map_meet_verdict`]), a table of constants whose constants are
/// walked through the other members ([`a_meet_read_by_its_constants`]), and a
/// reference among the members unfolded into the meet
/// ([`meet_through_a_reference`]); the last two decide it either way. Any other
/// meet is opaque however inhabited its members are.
fn intersection_verdict(members: &[Schema], cx: &mut Cx<'_>) -> (Verdict, Regions) {
    let mut any_empty = false;
    let mut region = Regions::MEET_UNIT;
    // What the members say together, which only the shape below can read as the
    // meet's own: in general two inhabited members meet in nothing.
    let mut members_hold = Verdict::Inhabited;
    for m in members {
        let (verdict, member_region) = m.empty_and_region(cx);
        any_empty |= verdict.is_empty();
        members_hold = Verdict::every([members_hold, verdict].into_iter());
        region = region.intersect(member_region);
        // Both accumulators absorb: an empty member empties the meet whatever the
        // rest are, and an opaque region stays opaque. No later member can change
        // either, so the walk stops rather than spending the budget on an answer
        // already fixed.
        if any_empty && region.is_absorbing() {
            break;
        }
    }
    let mut maps_hold = Verdict::Unknown;
    let empty = any_empty
        || region.known().is_some_and(Region::is_empty)
        || has_complementary_pair(members, cx.oracle, &[])
        || has_a_pattern_and_its_complement(members, cx)
        || has_disjoint_pair(members, cx.oracle)
        || intersection_bounds_unsatisfiable(members, cx.oracle)
        || {
            maps_hold = keyed_map_meet_verdict(members, cx);
            maps_hold.is_empty()
        };
    let verdict = if empty {
        Verdict::Empty
    } else if let Some((class, fields)) = class_with_attributes(members) {
        if members_hold == Verdict::Inhabited && !a_direct_instance_carries(class, fields, cx) {
            Verdict::Unknown
        } else {
            members_hold
        }
    } else {
        match region.verdict() {
            // A meet of closed maps is inhabited where the keys they require
            // meet in a value (`keyed_map_meet_verdict`), whatever its members
            // read as alone: a member's fold stops at the first branch of a
            // union it cannot settle, where the meet of a key's types is
            // distributed over that union and reads every branch.
            Verdict::Unknown => maps_hold,
            settled => settled,
        }
    };
    let verdict = match verdict {
        // The complement law again, with the definitions: a reference is a set
        // where its definition is one, which a walk of the definition says, and
        // the walk is paid only by a meet the readings above left open.
        Verdict::Unknown
            if !cx.defs.is_empty() && has_complementary_pair(members, cx.oracle, cx.defs) =>
        {
            Verdict::Empty
        }
        // Each reading is sound on its own, so one that cannot read the meet
        // hands it on: a narrowing or a distribution that ends undecided leaves
        // the difference of the records to be asked, as one that does not
        // apply does. A narrowed meet is not distributed again: it is the same
        // set, read with its own unions distributed already, and distributing
        // the meet reads the branches it dropped, which meet nothing, at the
        // distribution's cost.
        Verdict::Unknown => the_empty_sequence_in_every_member(members)
            .or_else(|| a_meet_read_by_its_constants(members, cx.oracle, cx.defs))
            .or_else(|| match narrowed_by_its_members(members, cx) {
                Some(narrowed) => narrowed,
                None => meet_through_a_reference(members, cx),
            })
            .or_else(|| a_difference_of_sets(members, cx)),
        settled => settled,
    };
    (verdict, region)
}

/// `Inhabited` where every member of the meet is a sequence of one kind that
/// holds the empty sequence, which is then a value of every member;
/// `Unknown` otherwise.
///
/// A sequence shape with no prefix holds the empty sequence whatever its
/// element admits, and a refinement of one keeps it under length bounds that
/// admit zero ([`Schema::admits_the_empty_sequence`]), so the meet holds it with
/// no element read: `[] & list[t]` has the empty list as a value where `t` is a
/// definition the recursion is resolving and reads as empty.
fn the_empty_sequence_in_every_member(members: &[Schema]) -> Verdict {
    let Some(kind) = members.iter().find_map(|member| match member {
        Schema::Seq { container, .. } => Some(*container),
        _ => None,
    }) else {
        return Verdict::Unknown;
    };
    if members
        .iter()
        .all(|member| member.admits_the_empty_sequence(kind))
    {
        Verdict::Inhabited
    } else {
        Verdict::Unknown
    }
}

/// The verdict of the meet `members` with every branch of a union member
/// dropped that shares no value with another member, or `None` where no
/// branch is.
///
/// `A ∩ (B ∪ C)` is `(A ∩ B) ∪ (A ∩ C)` [FCB08, Def. 4.1], which is `A ∩ C`
/// where `A ∩ B` is empty, so the narrowed meet is the same set and its verdict
/// is the meet's, either way. A branch is dropped where a complement says so
/// ([`shares_nothing_beside`]), which is what the guard a translation from
/// JSON Schema writes needs: `~K | X` met with `K` is `X` met with `K`,
/// with no clause distributed, where three guards distributed are eight
/// clauses and past [`MAX_DISTRIBUTED_TERMS`]. A union is narrowed only by the
/// members beside it, never by itself: a branch meets its own union in itself.
///
/// Out of line: asked of every meet the rules decline, it answers `None` at the
/// first look where no member is a union.
#[inline(never)]
fn narrowed_by_its_members(members: &[Schema], cx: &mut Cx<'_>) -> Option<Verdict> {
    let drops = |at: usize, branch: &Schema| shares_nothing_beside(branch, members, at, cx);
    let narrows = members.iter().enumerate().any(|(at, member)| {
        matches!(member, Schema::Union(branches) if branches.iter().any(|b| drops(at, b)))
    });
    if !narrows {
        return None;
    }
    let kept: Vec<Schema> = members
        .iter()
        .enumerate()
        .map(|(at, member)| match member {
            Schema::Union(branches) => {
                Schema::union_within(branches.iter().filter(|b| !drops(at, b)).cloned(), cx.defs)
            }
            other => other.clone(),
        })
        .collect();
    Some(Schema::meet_within(kept, cx.defs).empty_and_region(cx).0)
}

/// Whether `branch` shares no value with the members of a meet other than the
/// one at `at`, by a complement and nothing else: it is the complement of one of
/// them, of a meet whose conjuncts are among them, or of a union holding one of
/// them; or one of them is its complement, or the complement of a union holding
/// it, which a guard's own union flattens `¬¬(a ∪ b)` into. Each of those must
/// be a set. The kinds are left to the rules the narrowed meet reaches: a branch
/// dropped for its kind can take with it the half of a complementary pair the
/// meet's constructor would fold.
fn shares_nothing_beside(branch: &Schema, members: &[Schema], at: usize, cx: &Cx<'_>) -> bool {
    let beside = || {
        members
            .iter()
            .enumerate()
            .filter(move |(i, _)| *i != at)
            .map(|(_, m)| m)
    };
    let a_set = |schema: &Schema| denotes_a_set_within(schema, cx.oracle, cx.defs);
    if let Schema::Complement(inner) = branch {
        let held = match &**inner {
            Schema::Intersection(parts) => parts.iter().all(|part| beside().any(|m| m == part)),
            Schema::Union(parts) => beside().any(|m| m == &**inner || parts.contains(m)),
            single => beside().any(|m| m == single),
        };
        if held && a_set(inner) {
            return true;
        }
    }
    beside().any(|member| {
        let complements = matches!(member, Schema::Complement(inner)
            if **inner == *branch
                || matches!(&**inner, Schema::Union(parts) if parts.contains(branch)));
        complements && a_set(member)
    })
}

/// A meet the rules declined, read as a set minus sets where it is one: a
/// table met with complements of tables ([`FiniteMeet`]), or keyed maps met
/// with complements of keyed maps ([`keyed_map_minus_keyed_maps`]), beside
/// which a member of another shape is left out.
///
/// Out of line, and asked only of a declined meet no reference unfolds, so the
/// meets the rules decide carry neither reading on their path; and a meet
/// holding no keyed map and a member of no shape the table reads -- a class
/// with its attributes, say -- leaves at the first look.
#[inline(never)]
fn a_difference_of_sets(members: &[Schema], cx: &mut Cx<'_>) -> Verdict {
    let shaped = members.iter().all(|member| {
        matches!(
            member,
            Schema::Literal(_) | Schema::Union(_) | Schema::Complement(_) | Schema::KeyedMap { .. }
        )
    });
    if shaped && let Some(meet) = FiniteMeet::of(members) {
        return meet.verdict(cx.oracle);
    }
    if shaped
        || members
            .iter()
            .any(|member| matches!(member, Schema::KeyedMap { .. }))
    {
        return keyed_map_minus_keyed_maps(members, cx);
    }
    Verdict::Unknown
}

/// Whether `members` hold a pattern refinement and the complement of one with
/// the same constraints whose base holds every string.
///
/// A pattern admits no value but a `str`, so the two denote one set -- the
/// strings the constraints admit -- whatever their bases hold beside strings,
/// and `Refine(B, C) ∧ ¬Refine(B', C)` is `B ∧ C ∧ ¬B'`, empty where `C ⊆ str
/// ⊆ B'`. The spelling check beside it reads `Refine(str, C)` and
/// `Refine(Anything, C)` as two sets, while the set reading lowers both to the
/// same language, so without this a union of one and the other's complement
/// would be the universe to the set reading and short of it to the rules. The
/// constraints are read as a set first: a predicate among them may answer the
/// two occurrences differently.
fn has_a_pattern_and_its_complement(members: &[Schema], cx: &mut Cx<'_>) -> bool {
    members.iter().any(|member| {
        let Schema::Complement(inner) = member else {
            return false;
        };
        let Schema::Refine { base, constraints } = &**inner else {
            return false;
        };
        constraints.iter().any(|c| matches!(c, Constraint::Regex(_)))
            && members.iter().any(|other| {
                matches!(other, Schema::Refine { constraints: ours, .. } if ours == constraints)
            })
            && denotes_a_set_within(inner, cx.oracle, cx.defs)
            && Schema::Str.is_subtype_rec(base, cx).holds()
    })
}

/// A meet the rules leave undecided, decided by unfolding a reference among
/// its members into it, or `Unknown` where no member is a reference a
/// definition resolves.
///
/// A meet is a set of values, and a reference denotes its definition, so `t ∧
/// X` is the definition met with `X` and a definition that is a union is the
/// union of its branches met with `X` [FCB08, Lem. 6.2], empty where every
/// branch's meet is, inhabited where one is. A branch's meet is a meet like any
/// other, so the rules read it as they read this one -- a branch of another
/// kind than `X` meets nothing, two records meet key by key, and a key's meet
/// reaches this again where its types name the reference. A union among a
/// meet's members is distributed the same way where it holds a reference
/// itself, since the reference is as often written inside one -- `{"next": Node
/// | None}` -- as around it, and below an unfolding, where the key of a
/// branch's meet is as often a union of the definition's own -- `t ∧ (None |
/// {"a": None})` asks `None ∧ (None | {"a": None})`. Elsewhere a union is
/// distributed only where the meet's normal form is a few clauses
/// ([`normal_form_within`]) and is left to the rules past them.
///
/// The meet is the goal, its members sorted and without repeats -- the members
/// of a meet are a set -- held open while the path below decides it
/// ([`OpenMeet`]) and read as empty where the path reaches it again
/// [LN26, §5.1]. That is the least fixpoint read inductively, as a reference
/// reached again on `visiting` is: a value is a finite tree, so a value of the
/// meet that needed a value of the same meet below it needs a smaller one, and
/// a smallest value needs none. The goal is popped when it is decided and
/// remembered nowhere, since what it answered rested on the goals open above
/// it. Its members are drawn from the schema's subterms and the definitions',
/// so a query has finitely many, as normal forms over the atoms a type mentions
/// are [FCB08, §6.9], and a path holds each at most once.
///
/// **A meet distributed for its normal form alone is no goal.** No reference
/// reaches it, so the path below reaches it again only through a reference
/// that opens a goal of its own, and a goal it held open would have every meet
/// below it distributed past the bound: `set[str] | set[bytes]` met with an
/// intersection-of-unions tower three levels deep reads in 334 steps without
/// one and in 87 640 with one, a factor every level multiplies.
///
/// **A branch drops every copy of the member it distributes.** A branch is
/// inside that member, so meeting it with another copy says nothing more, and a
/// copy left in would make the branch's meet the meet already open wherever the
/// branch is a member beside it: `float` distributed out of `int | float` in a
/// meet holding `float` and a second `int | float`. Reached again with no
/// constructor between, the cycle needs the same value rather than a smaller
/// one, so reading it as empty is not the least fixpoint at all. With the copies
/// gone, a distribution leaves strictly smaller members and an unfolding is
/// guarded by the contractivity every definition is built with, so a path
/// reaches an open meet again only under a constructor.
///
/// **The meet is its own goal and never the reference's.** Under `t = None |
/// {"a": t}` the meet `t ∧ {"a": None | {"a": None}}` asks of its key `t ∧
/// (None | {"a": None})`, which holds `None`. Reading that `t` as the
/// reference already being resolved -- pushing it on `visiting` here -- reads
/// the key, and the meet holding `{"a": None}`, empty.
///
/// Asked only where the rules decline, so a meet they decide keeps its answer.
fn meet_through_a_reference(members: &[Schema], cx: &mut Cx<'_>) -> Verdict {
    let defs = cx.defs;
    let unfolding = members
        .iter()
        .enumerate()
        .find_map(|(at, member)| match member {
            Schema::Ref(id) => defs.get(id.get()).map(|body| match body {
                Schema::Union(branches) => (at, &branches[..]),
                body => (at, std::slice::from_ref(body)),
            }),
            _ => None,
        });
    let (at, branches, unfolds) = if let Some((at, branches)) = unfolding {
        (at, branches, true)
    } else {
        let Some((at, branches)) =
            members
                .iter()
                .enumerate()
                .find_map(|(at, member)| match member {
                    Schema::Union(branches) => Some((at, &branches[..])),
                    _ => None,
                })
        else {
            return Verdict::Unknown;
        };
        let holds_a_reference = branches
            .iter()
            .any(|branch| matches!(branch, Schema::Ref(_)));
        let unfolds = holds_a_reference || OpenMeet::any_open(cx.budget);
        if !(unfolds || normal_form_within(members, MAX_DISTRIBUTED_TERMS)) {
            return Verdict::Unknown;
        }
        (at, branches, unfolds)
    };
    let _open = if unfolds {
        let mut goal = members.to_vec();
        goal.sort();
        goal.dedup();
        let Some(open) = OpenMeet::open(cx.budget, goal) else {
            return Verdict::Empty;
        };
        Some(open)
    } else {
        None
    };
    let mut verdict = Verdict::Empty;
    for branch in branches {
        let mut meet = match branch {
            Schema::Intersection(inner) => inner.to_vec(),
            branch => vec![branch.clone()],
        };
        let distributed = members.get(at);
        meet.extend(
            members
                .iter()
                .filter(|member| Some(*member) != distributed)
                .cloned(),
        );
        let (held, _) = Schema::Intersection(meet.into()).empty_and_region(cx);
        verdict = Verdict::any([verdict, held].into_iter());
        if verdict == Verdict::Inhabited {
            break;
        }
    }
    verdict
}

/// The clauses a meet with no reference may be read as before it is distributed.
///
/// Four, the bound ty's intersection builder caps its normal form at where it
/// is bounded (`MAX_INTERSECTION_DNF_TERMS`): two keywords guarded to their
/// kind, `~K | X`, are four clauses, and a branch a guard's kind empties at
/// once costs a step.
pub(super) const MAX_DISTRIBUTED_TERMS: usize = 4;

/// Whether the normal form of the meet `members` spans at most `bound`
/// clauses: the product of the widths of the unions among its members, each
/// branch of them one clause.
///
/// A meet the rules decline is the union of the meets its union members'
/// branches make with the rest [FCB08, Def. 4.1], which is how every source
/// decides it, on the normal form, and a branch's meet is strictly smaller, so
/// the distribution ends. What it costs is the clauses: a meet of `k` unions of
/// two is `2^k` of them, each a decision, so a decline that spends a step
/// distributed spends a product, which [`MAX_DISTRIBUTED_TERMS`] bounds.
///
/// **A branch that is a meet or a union is not one clause.** Distributed, its
/// members join the branch's meet and its own unions are distributed in turn,
/// so the width counts fewer clauses than the distribution reads, and the meet
/// is left to the rules: a tower whose every level is `(N | A) & (N | B)`,
/// counted by its widths, is four clauses at each level and a product of them
/// over its depth.
fn normal_form_within(members: &[Schema], bound: usize) -> bool {
    members
        .iter()
        .try_fold(1usize, |clauses, member| match member {
            Schema::Union(branches)
                if branches.iter().all(|branch| {
                    !matches!(branch, Schema::Union(_) | Schema::Intersection(_))
                }) =>
            {
                clauses
                    .checked_mul(branches.len())
                    .filter(|&clauses| clauses <= bound)
            }
            Schema::Union(_) => None,
            _ => Some(clauses),
        })
        .is_some()
}

thread_local! {
    /// The meets open on this thread's emptiness paths, each beside the address
    /// of the budget of the query that opened it.
    ///
    /// A meet goal is a hypothesis of one query's path and of no other, and a
    /// query can start inside another on the same thread -- a class relation
    /// the oracle asks of Python can ask one -- so each is read only by the
    /// query whose budget opened it, which is live, and so at an address no
    /// other live budget has. Held here rather than on the budget or the
    /// query's [`Cx`], beside `visiting`: a list on either is built and dropped
    /// by every query, and this one is reached only where a meet is unfolded.
    /// On the context its teardown would read 10.6% more instructions on the
    /// decision workload, whose queries take a few hundred each.
    static OPEN_MEETS: RefCell<Vec<(usize, Vec<Schema>)>> = const { RefCell::new(Vec::new()) };
}

/// A meet held open on [`OPEN_MEETS`] while the path below decides it, and
/// closed when that is decided, however the path ends.
struct OpenMeet;

impl OpenMeet {
    /// Whether the query `budget` belongs to holds a meet open.
    fn any_open(budget: &Budget) -> bool {
        let query = std::ptr::from_ref(budget) as usize;
        OPEN_MEETS.with_borrow(|open| open.iter().any(|(owner, _)| *owner == query))
    }

    /// Open `goal` for the query `budget` belongs to, or `None` where that
    /// query holds it open already, a node met during its own traversal being
    /// empty [LN26, §5.1].
    fn open(budget: &Budget, goal: Vec<Schema>) -> Option<Self> {
        let query = std::ptr::from_ref(budget) as usize;
        OPEN_MEETS.with_borrow_mut(|open| {
            if open
                .iter()
                .any(|(owner, held)| *owner == query && *held == goal)
            {
                return None;
            }
            open.push((query, goal));
            Some(Self)
        })
    }
}

impl Drop for OpenMeet {
    fn drop(&mut self) {
        OPEN_MEETS.with_borrow_mut(|open| {
            open.pop();
        });
    }
}

/// Whether a direct instance of `class` can carry the record it is met with.
///
/// The value [`class_with_attributes`] reads the meet as holding is a direct
/// instance of the class with a witness in each field, and the class decides
/// whether one can exist: a name its body defines cannot be missing, a property
/// answers with its getter, and an instance without a dictionary holds nothing
/// its slots do not name. Where it cannot, a subclass may carry the record and
/// may not exist, so the caller reads the meet as unknown -- the descriptor's
/// atom rule reads the same [`Reach`] off the same snapshot, and the two
/// deciders keep one answer.
///
/// Asked only of a meet whose members are inhabited, so every attribute is
/// already known to admit a value, and the class answers only whether a direct
/// instance carries each name.
fn a_direct_instance_carries(class: ClassIx, fields: &[Attribute], cx: &Cx<'_>) -> bool {
    fields.iter().all(|field| {
        matches!(
            cx.oracle.attribute_reach(class, &field.name),
            Some(Reach::Anything | Reach::AnyValue)
        )
    })
}

/// Whether this meet is one class together with the attributes its instances
/// carry, which is the shape a dataclass lowers to.
///
/// The descriptor's atom rule already reads that pair -- a class with fields,
/// inhabited when its fields are -- and the meet above says it in the
/// vocabulary the structural procedure uses. Neither member's region says
/// anything, so without this the meet of two opaque halves is opaque, and every
/// refutation about a dataclass would be dropped by the guard that believes one.
///
/// **The witness is a *direct* instance of the class**, carrying whatever the
/// fields admit: `type(v) is C`, so nothing is asked about which classes derive
/// from which. That is why exactly one class is read. Two would need a class
/// deriving from both, which is the open world the atom rule declines, and two
/// attribute records would need their fields not to contradict each other --
/// `{x: int}` and `{x: str}` are each inhabited and meet in nothing. Anything
/// else in the meet leaves the answer to the regions.
///
/// The class and the record's fields come back with the answer because the
/// callers want them: one reads the class out of the shape, and the meet's
/// reading asks whether the class carries the fields. A guard returning a
/// `bool` would make each search for what the match has already found -- a
/// search that cannot fail, and whose failure arm no input reaches.
pub(super) fn class_with_attributes(members: &[Schema]) -> Option<(ClassIx, &[Attribute])> {
    match members {
        [Schema::Instance(class), Schema::AttrRecord { fields }]
        | [Schema::AttrRecord { fields }, Schema::Instance(class)] => Some((*class, fields)),
        _ => None,
    }
}

/// What a refinement's emptiness is, read from its base and its constraints.
///
/// A refinement is a subset of its base, so an empty base empties it and so
/// does a bound conjunction nothing satisfies. Inhabitance is the harder half:
/// a satisfiable bound over an inhabited base proves nothing in general, since
/// whether any value of the base survives the constraints is what the bounds
/// check declines to say.
///
/// One shape is the exception, and it is the shape a length bound is written
/// for: a container that repeats one element, where a value of any length is
/// built by repeating one. Its element decides it -- a shortest length of zero
/// is met by the empty container whatever the element admits, and a longer one
/// by as many copies of an element as it asks for. Which is also the reading
/// that gives `mu t. list[t] & MinLen(1)` the answer it has: each unfolding
/// demands one more element and the cycle has no finite value.
fn refinement_verdict(base: &Schema, constraints: &Constraints, cx: &mut Cx<'_>) -> Verdict {
    let density = density_of([base]);
    if base.is_empty_rec(cx)
        || bounds_unsatisfiable(constraints.iter(), cx.oracle, density)
        || lengths_miss_the_shape(base, constraints, cx)
    {
        return Verdict::Empty;
    }
    // The integers are their own case: they are unbounded both ways and
    // discrete, so an order bound over them is met by an integer the oracle can
    // be asked to name. Read before the length bounds below, which are a
    // different question about a different base.
    if matches!(base, Schema::Int) {
        return bounded_integer_verdict(constraints, cx.oracle);
    }
    let lengths_only = constraints
        .iter()
        .all(|c| matches!(c, Constraint::MinLen(_) | Constraint::MaxLen(_)));
    if !lengths_only {
        return Verdict::Unknown;
    }
    match base {
        // A string and a bytes take any length, so a bound the lengths
        // themselves satisfy -- which the check above has already read -- is met
        // by a value of the shortest length it admits.
        Schema::Str | Schema::Bytes => Verdict::Inhabited,
        // A container takes any length too: a bound of zero is met by the empty
        // container whatever the element admits. Past zero the two container
        // families part. A sequence repeats one element, so a bound of any size
        // is met by that many copies of a single witness. A **set holds each
        // member once**, so a bound of `n` asks the element for `n` values that
        // differ -- `set[None]` with `MinLen(2)` denotes no set at all, and
        // reading it as inhabited let a kind mismatch refute an inclusion that
        // holds vacuously.
        _ => match repeated_element(base) {
            Some(_) if shortest(constraints.iter()) == 0 => Verdict::Inhabited,
            Some(element) if repeats_a_member(base) => element.verdict_rec(cx),
            Some(element) => distinct_members(element, shortest(constraints.iter()), cx.oracle),
            None => Verdict::Unknown,
        },
    }
}

/// Whether an order bound over `int` is met by an integer, asked of the oracle.
///
/// The question is the one the oracle already answers for emptiness -- whether
/// an integer lies between two bounds -- read for its other answer. A
/// `Some(false)` there *is* the witness: an integer in the interval is a value
/// of the refinement.
///
/// One bound is the degenerate interval on the bound itself, because the
/// integers are unbounded the other way: an integer at `c` means one at or past
/// `c` exists, and one below it too. A bound the oracle cannot place on the
/// integer line -- `Ge(inf)`, where the question raises -- declines, and so
/// does one that lies strictly between two integers: `Ge(0.5)` has values and
/// this question cannot name one, which is the miss it is worth having.
///
/// Every comparison has to be answered. Two bounds on one side that the oracle
/// cannot order leave it unknown which is the tighter, and naming a value under
/// the looser one would name a value the tighter excludes -- sound for proving
/// the refinement *empty*, which is what the fold above does with the same
/// pair, and unsound for proving it has a value.
fn bounded_integer_verdict(constraints: &Constraints, oracle: &dyn LeafRelations) -> Verdict {
    // One integer step beside integer bounds is read exactly: the set is the
    // multiples of the step between the bounds, and a range of them is empty
    // or holds a value.
    if let Some(multiples) = Multiples::of(constraints, oracle) {
        return if multiples.is_empty() {
            Verdict::Empty
        } else {
            Verdict::Inhabited
        };
    }
    // A constraint that is not an order bound narrows the set by something this
    // cannot read, so it names no value.
    if !constraints.iter().all(|c| {
        matches!(
            c,
            Constraint::Ge(_) | Constraint::Gt(_) | Constraint::Le(_) | Constraint::Lt(_)
        )
    }) {
        return Verdict::Unknown;
    }
    let (lower, upper, ordered) = tightest_bounds(constraints.iter(), oracle);
    if !ordered {
        return Verdict::Unknown;
    }
    let named = match (lower, upper) {
        // Every integer, and there is one.
        (None, None) => return Verdict::Inhabited,
        (Some((lo, lo_strict)), Some((hi, hi_strict))) => {
            oracle.no_int_between(lo, lo_strict, hi, hi_strict)
        }
        (Some((c, _)), None) | (None, Some((c, _))) => oracle.no_int_between(c, false, c, false),
    };
    match named {
        Some(false) => Verdict::Inhabited,
        _ => Verdict::Unknown,
    }
}

/// Whether the container admits one value more than once.
///
/// A list and a tuple do; a set and a frozenset hold each member once, which is
/// what makes a length bound over one a question about the element's values
/// rather than about the container.
fn repeats_a_member(base: &Schema) -> bool {
    matches!(base, Schema::Seq { .. })
}

/// Whether a set of `wanted` members of this schema exists.
///
/// The question is how many values the schema denotes, and it is answered from
/// bounds rather than from a count: a schema is read as denoting at least
/// `least` values and at most `most`, and each side decides one answer. Where
/// the two do not reach the bound, the answer is a decline -- a union may name
/// one value twice, so its members' counts sum to an upper bound and the
/// largest of them is a lower one, and neither is the count.
fn distinct_members(element: &Schema, wanted: usize, oracle: &dyn LeafRelations) -> Verdict {
    let (least, most) = value_count_bounds(element, oracle);
    if most < wanted {
        return Verdict::Empty;
    }
    if least >= wanted {
        return Verdict::Inhabited;
    }
    Verdict::Unknown
}

/// How many values a schema denotes, counted up to `==`, as a lower and an
/// upper bound: the members a set drawn from it can hold apart, since a set
/// keeps one of two equal values. A literal holding a value counts one, its
/// constant's class being one member however many values `==` joins to it, the
/// model's singleton [FCB08, §4.2] read as a set reads it.
///
/// [`usize::MAX`] stands for "more than any bound a caller writes". The pair is
/// read in one direction each: the upper bound proves a set empty, the lower
/// one proves it inhabited, and a schema this cannot count reports `(0, MAX)`,
/// which proves neither.
fn value_count_bounds(schema: &Schema, oracle: &dyn LeafRelations) -> (usize, usize) {
    match schema {
        Schema::Anything(_) | Schema::Int | Schema::Float | Schema::Str | Schema::Bytes => {
            (usize::MAX, usize::MAX)
        }
        Schema::Nothing => (0, 0),
        Schema::NoneType => (1, 1),
        // A literal is one value only where its constant's type carries the
        // equality this oracle can read. `Literal[c]` denotes `{x | type(x) is
        // type(c) and x == c}`, so a constant whose `__eq__` answers `True` for
        // its siblings denotes more than one value and a constant equal to
        // nothing -- `nan` -- denotes none. The question that separates the
        // three is the one the oracle already answers for a pair, asked of the
        // constant against itself: `Some(false)` is "it is a value, and equality
        // here is one I can read", `Some(true)` is the empty singleton, and a
        // decline leaves the count unread.
        Schema::Literal(index) => match oracle.literals_disjoint(*index, *index) {
            Some(false) => (1, 1),
            Some(true) => (0, 0),
            None => (0, usize::MAX),
        },
        Schema::Bool => (2, 2),
        // Every collection of an inhabited element is a value, and the empty
        // one is a value whatever the element is -- so one at least, and no
        // upper bound this reads.
        Schema::Coll { .. } => (1, usize::MAX),
        Schema::Union(members) => members
            .iter()
            .map(|member| value_count_bounds(member, oracle))
            .fold((0, 0), |(least, most), (member_least, member_most)| {
                (least.max(member_least), most.saturating_add(member_most))
            }),
        _ => (0, usize::MAX),
    }
}

/// Whether a sequence base's own lengths and the refinement's length bounds
/// share none.
///
/// A sequence is never shorter than its fixed positions, and never longer where
/// its tail admits nothing -- no tail, or one that is empty, which repeats zero
/// times. So `MinLen(5)` over a list of exactly three elements, or `MaxLen(1)`
/// over one of at least two, admits no sequence, whatever the conjunction's
/// other constraints say.
fn lengths_miss_the_shape(base: &Schema, constraints: &Constraints, cx: &mut Cx<'_>) -> bool {
    let Schema::Seq { shape, .. } = base else {
        return false;
    };
    let fixed = shape.prefix.len();
    if longest(constraints.iter()).is_some_and(|most| most < fixed) {
        return true;
    }
    let open = shape
        .tail
        .as_deref()
        .is_some_and(|tail| !tail.is_empty_rec(cx));
    !open && shortest(constraints.iter()) > fixed
}

/// The one element schema a container repeats, for the containers that repeat
/// one: a sequence with no fixed position, a set, a frozenset.
///
/// A value of such a container is any number of values of that element, which
/// is what lets a length bound be met by building one.
fn repeated_element(base: &Schema) -> Option<&Schema> {
    match base {
        Schema::Seq { shape, .. } if shape.prefix.is_empty() => shape.tail.as_deref(),
        Schema::Coll { element, .. } => Some(element),
        _ => None,
    }
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
    let density = density_of(members);
    !merged.is_empty() && bounds_unsatisfiable(merged.iter().copied(), oracle, density)
}
