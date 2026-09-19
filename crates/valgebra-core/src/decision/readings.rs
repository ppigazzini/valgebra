//! The readings a pair with no rule of its own is given: what a schema can be
//! seen to hold or exclude without building the set it denotes.
//!
//! A refutation is a claim that a value exists in the subject and outside the
//! supertype, so every reading here that refutes names the value it stands on:
//! two schemas that share no value, a kind the other side cannot hold, a class
//! outside every kind the supertype admits, a sequence shorter than any the
//! other side has. Each is read off the constructors, and off the seven-bit
//! region summary the partition derives -- the summary is all a bitset needs to
//! decide a scalar, and the eleven-part partition itself is the descriptor's,
//! which reasons a component at a time.
//!
//! None of these is a rule: a rule proves an inclusion, and these answer the
//! question a rule left, in the direction a sound procedure may -- a reading
//! that cannot name a value declines, and the pair stays undecided.

use crate::ir::{ClassIx, CollKind, Constraint, Schema, SeqKind};
use crate::kind::{Kind, Region, Regions};

use super::literals::literal_constants;
use super::{LeafRelations, NoLeafRelations, SubtypeCx};

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
        // Each table is read only where the one before it answered: a tuple of
        // the two builds both, and the second is wasted whenever the first says
        // this pair is not two tables of constants.
        //
        // And neither is read until both sides are a shape that could *be* one.
        // A table is a literal or a union of them, so every other node answers
        // the question with a discriminant test -- while building the left
        // table walks the left union's members, and a union of nothing but
        // literals against a node that is not a table pays that walk to reach a
        // reading the right side was never going to answer.
        if matches!(self, Schema::Literal(_) | Schema::Union(_))
            && matches!(other, Schema::Literal(_) | Schema::Union(_))
            && let Some(left) = literal_constants(self)
            && let Some(right) = literal_constants(other)
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
    pub(super) fn shorter_than(&self, other: &Schema) -> bool {
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
    pub(super) fn type_tag(&self) -> Option<Kind> {
        self.type_tag_with(&NoLeafRelations)
    }

    /// [`type_tag`](Self::type_tag) with the oracle that kinds a `Literal`.
    pub(super) fn type_tag_with(&self, oracle: &dyn LeafRelations) -> Option<Kind> {
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
    pub(super) fn atom_region(&self) -> Option<Region> {
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
    pub(super) fn covers_the_universe(&self) -> bool {
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

    /// Whether this schema's kind holds a value the supertype's class does not.
    ///
    /// A subject every value of which has one kind is refuted against a class
    /// that kind's own builtin does not derive from: a value of the kind built
    /// as that builtin has `type(v)` equal to it, so the order settles
    /// `isinstance`, and no class deriving from both changes that particular
    /// value. The dual of the reading a meet of a class and its attributes
    /// gets, with the kind and the class swapping sides.
    pub(super) fn outside_the_class(&self, other: &Schema, oracle: &dyn LeafRelations) -> bool {
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
    pub(super) fn spills_past(&self, other: &Schema, oracle: &dyn LeafRelations) -> bool {
        let Regions::Known(mine) = self.region_set() else {
            return false;
        };
        other
            .type_tag_with(oracle)
            .is_some_and(|kind| !mine.subset_of(kind.admitted_regions()))
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
pub(super) fn outside_every_kind(other: &Schema, class: ClassIx, cx: SubtypeCx<'_>) -> bool {
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
