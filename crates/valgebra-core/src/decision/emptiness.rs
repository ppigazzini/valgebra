//! Emptiness: whether a schema denotes no value, as a three-valued verdict.
//!
//! The question every relation reduces to. Subtyping is the emptiness of a
//! difference and equivalence is two of those, so this is the module the other
//! two rest on; it is JACM Definition 6.9 read *inductively* over the
//! constructors rather than as a saturation -- a union is empty when every
//! member is, a meet when its members share no value, a product when a
//! component is -- with the value-region partition folded in the same pass, so
//! a Boolean node reads its children's regions in constant time.
//!
//! The answer is three-valued because a refutation needs it: `Unknown` is not
//! a proof of emptiness and not a proof of inhabitation, and the boolean the
//! public entry points return folds it into "not proved empty". Where the rules
//! here decline, the question is asked once more of the *set* the schema
//! denotes, under the lowering's bound on what building it may cost.

use std::cell::Cell;

use crate::descr::lower::{Constants, lower_unfolded};
use crate::ir::{Constraint, Constraints, DefIx, Polarity, Schema};
use crate::kind::{Kind, Region, Regions};
use crate::verdict::Verdict;

use super::constraints::{Density, bounds_unsatisfiable, shortest, tightest_bounds};
use super::records::keyed_map_meet_empty;
use super::{
    DECISION_BUDGET, LeafRelations, NoLeafRelations, has_complementary_pair, has_disjoint_pair,
    spend,
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
        // The rules answer in three values and the descriptor is asked where
        // they reach the third -- which is what makes the pair one procedure
        // rather than a second opinion. A proof of *inhabitation* is an answer
        // like a proof of emptiness: a value of the schema is a value of it,
        // whatever a second reading would say, so the descriptor is not asked
        // to overturn what it cannot. Reading the bool instead of the verdict
        // asked it on every inhabited schema, and lowering one determinises
        // automata and takes products -- a third of the decision workload,
        // spent on a question already answered.
        match self.verdict_rec(oracle, defs, &mut Vec::new(), &Cell::new(DECISION_BUDGET)) {
            Verdict::Empty => true,
            Verdict::Inhabited => false,
            Verdict::Unknown => self.denotes_no_value(oracle, defs),
        }
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
        lower_unfolded(self, defs, Polarity::Widen, pool)
            .is_some_and(|set| set.emptiness() == Verdict::Empty)
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

    /// The emptiness verdict where the rules can look a constant or a class up.
    #[cfg(test)]
    pub(crate) fn verdict_under(&self, oracle: &dyn LeafRelations) -> Verdict {
        self.verdict_rec(oracle, &[], &mut Vec::new(), &Cell::new(DECISION_BUDGET))
    }

    pub(super) fn is_empty_rec(
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

    pub(super) fn verdict_rec(
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
    pub(super) fn empty_and_region(
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
            Schema::Refine { base, constraints } => (
                refinement_verdict(base, constraints, oracle, defs, visiting, budget),
                Regions::Unknown,
            ),
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
            // half is what a class in the same node would take away: an
            // `isinstance` atom is opaque, so a node holding both can never be
            // more than unknown.
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
            // A literal denotes `{x | type(x) is type(c) and x == c}`, which
            // holds `c` itself exactly when `c` equals itself. A constant that
            // does not -- a `nan` -- denotes the empty set, and every other
            // constant is a value, which is what makes a refutation naming one
            // a refutation at all.
            //
            // The core cannot read a constant, and this asks the oracle the
            // question it already answers: whether two constants denote
            // disjoint singletons, asked of one constant *against itself*. A
            // set disjoint from itself is the empty one, and a set that shares
            // a value with itself has a value. The oracle declines for a type
            // whose equality it does not trust, and the literal stays unknown
            // there, which is where it was for every constant before.
            Schema::Literal(constant) => (
                match oracle.literals_disjoint(*constant, *constant) {
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
            // and it was already wrong there -- the descriptor was making the
            // claim and the rules were paying to defer to it.
            //
            // A class with a hooked metaclass is not a set here, and stays
            // unknown as it is everywhere else.
            Schema::Instance(_) => (
                match oracle.atom_denotes_a_set(self) {
                    Some(true) => Verdict::Inhabited,
                    _ => Verdict::Unknown,
                },
                Regions::Unknown,
            ),
            // The gradual `Any` is not scalar-decidable and neither direction
            // is proven of it.
            _ => (Verdict::Unknown, Regions::Unknown),
        }
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
    // What the members say together, which only the shape below can read as the
    // meet's own: in general two inhabited members meet in nothing.
    let mut members_hold = Verdict::Inhabited;
    for m in members {
        let (verdict, member_region) = m.empty_and_region(oracle, defs, visiting, budget);
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
    let empty = any_empty
        || region.known().is_some_and(Region::is_empty)
        || has_complementary_pair(members, oracle)
        || has_disjoint_pair(members, oracle)
        || intersection_bounds_unsatisfiable(members, oracle)
        || keyed_map_meet_empty(members, oracle, defs, budget);
    let verdict = if empty {
        Verdict::Empty
    } else if class_with_attributes(members) {
        members_hold
    } else {
        region.verdict()
    };
    (verdict, region)
}

/// Whether this meet is one class together with the attributes its instances
/// carry, which is the shape a dataclass lowers to.
///
/// The descriptor's atom rule already reads that pair -- a class with fields,
/// inhabited when its fields are -- and the meet above says it in the
/// vocabulary the structural procedure uses. Neither member's region says
/// anything, so without this the meet of two opaque halves is opaque, and every
/// refutation about a dataclass was dropped by the guard that believes one.
///
/// **The witness is a *direct* instance of the class**, carrying whatever the
/// fields admit: `type(v) is C`, so nothing is asked about which classes derive
/// from which. That is why exactly one class is read. Two would need a class
/// deriving from both, which is the open world the atom rule declines, and two
/// attribute records would need their fields not to contradict each other --
/// `{x: int}` and `{x: str}` are each inhabited and meet in nothing. Anything
/// else in the meet leaves the answer to the regions.
pub(super) fn class_with_attributes(members: &[Schema]) -> bool {
    matches!(
        members,
        [Schema::Instance(_), Schema::AttrRecord { .. }]
            | [Schema::AttrRecord { .. }, Schema::Instance(_)]
    )
}

/// Whether a refinement's bound and length constraints cannot hold together: a
/// required minimum length above the allowed maximum, or a numeric lower bound
/// above the upper bound (or equal with a strict end). Sound: it reports
/// unsatisfiable only when the ordering the oracle returns forces it, and stays
/// conservative when the oracle cannot compare two bounds.
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
fn refinement_verdict(
    base: &Schema,
    constraints: &Constraints,
    oracle: &dyn LeafRelations,
    defs: &[Schema],
    visiting: &mut Vec<DefIx>,
    budget: &Cell<u32>,
) -> Verdict {
    let density = density_of([base]);
    if base.is_empty_rec(oracle, defs, visiting, budget)
        || bounds_unsatisfiable(constraints.iter(), oracle, density)
    {
        return Verdict::Empty;
    }
    // The integers are their own case: they are unbounded both ways and
    // discrete, so an order bound over them is met by an integer the oracle can
    // be asked to name. Read before the length bounds below, which are a
    // different question about a different base.
    if matches!(base, Schema::Int) {
        return bounded_integer_verdict(constraints, oracle);
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
            Some(element) if repeats_a_member(base) => {
                element.verdict_rec(oracle, defs, visiting, budget)
            }
            Some(element) => distinct_members(element, shortest(constraints.iter())),
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

/// The one element schema a container repeats, for the containers that repeat
/// one: a sequence with no fixed position, a set, a frozenset.
///
/// A value of such a container is any number of values of that element, which
/// is what lets a length bound be met by building one.
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
fn distinct_members(element: &Schema, wanted: usize) -> Verdict {
    let (least, most) = value_count_bounds(element);
    if most < wanted {
        return Verdict::Empty;
    }
    if least >= wanted {
        return Verdict::Inhabited;
    }
    Verdict::Unknown
}

/// How many values a schema denotes, as a lower and an upper bound.
///
/// [`usize::MAX`] stands for "more than any bound a caller writes". The pair is
/// read in one direction each: the upper bound proves a set empty, the lower
/// one proves it inhabited, and a schema this cannot count reports `(0, MAX)`,
/// which proves neither.
fn value_count_bounds(schema: &Schema) -> (usize, usize) {
    match schema {
        Schema::Anything(_) | Schema::Int | Schema::Float | Schema::Str | Schema::Bytes => {
            (usize::MAX, usize::MAX)
        }
        Schema::Nothing => (0, 0),
        Schema::NoneType | Schema::Literal(_) => (1, 1),
        Schema::Bool => (2, 2),
        // Every collection of an inhabited element is a value, and the empty
        // one is a value whatever the element is -- so one at least, and no
        // upper bound this reads.
        Schema::Coll { .. } => (1, usize::MAX),
        Schema::Union(members) => members.iter().map(value_count_bounds).fold(
            (0, 0),
            |(least, most), (member_least, member_most)| {
                (least.max(member_least), most.saturating_add(member_most))
            },
        ),
        _ => (0, usize::MAX),
    }
}

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
