//! The partition of the value universe, and the summary derived from it.
//!
//! [`Kind`] is the eleven-part partition the semantic-subtyping literature
//! kinds the atoms by, and the one a descriptor is indexed with: `Descr::kinds`
//! is an array over it, which is what makes that structure "a record mapping
//! each atomic type to its representation". [`Region`] is the seven-bit
//! *summary* derived from it, and [`Regions`] is a region set that may be
//! opaque where a schema is not scalar-decidable.
//!
//! All three sit **below both deciders**, which is why they are here rather
//! than in either. The structural rules reason in the summary, because that is
//! all a bitset needs to decide a scalar; the descriptor reasons in the
//! partition, because a component per kind is what closes each under
//! complement. Neither owns the frame they share, and a test holds the
//! direction: `crates/valgebra-core/src/descr/` imports nothing from
//! `decision.rs`. `docs/dev/02-decision.md` describes the two vocabularies and
//! why a change to one is a change to the other.

use crate::verdict::Verdict;

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

/// A concrete runtime kind: the type a value's `type(x)` is.
///
/// The kinds partition the value universe, so two schemas carrying different
/// kinds share no value -- `bool` and `int` aside, since `bool` subclasses `int`.
/// A schema the core cannot kind has none, and disjointness stays conservative.
///
/// Public because the core cannot see a Python object: a `Literal`'s kind is a
/// fact about a pooled constant, which only the bindings can read, and they
/// answer in this vocabulary through
/// [`LeafRelations::literal_kind`](crate::LeafRelations::literal_kind).
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
    /// Whether a value can have both this kind and `other`.
    ///
    /// The partition is a partition but for one pair: `bool` subclasses `int`,
    /// so every boolean is an integer and the two kinds share values. Two kinds
    /// that share none are disjoint sets, which is the reading a kind is for.
    #[must_use]
    pub fn shares_values_with(self, other: Kind) -> bool {
        self == other
            || matches!(
                (self, other),
                (Kind::Bool, Kind::Int) | (Kind::Int, Kind::Bool)
            )
    }

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
    /// in one file that the compiler makes you finish. Held as two lists -- the
    /// kinds here and a set of region constants beside them -- nothing ties
    /// `List` to the region a list belongs to.
    ///
    /// The six scalar kinds each get a region of their own, because a schema can
    /// name one exactly: `str` denotes every string and nothing else, so the
    /// complement of `str` is exactly the other six regions. The five container
    /// kinds share the non-scalar region, because no schema names one exactly --
    /// `list[int]` is a proper part of the lists, so the fold keeps a container
    /// opaque rather than claiming a region for it (see [`Regions`]).
    /// The regions a value of a schema *tagged* this kind may be in.
    ///
    /// A kind's own region, with one exception, and it is the exception
    /// [`Schema::atom_region`] already names from the other side: `bool`
    /// subclasses `int`, so a schema whose values are integers admits both
    /// regions and a refinement over `int` holds `False`. Reading `Int` as its
    /// own region alone would refute `bool` against a bounded `int`, which is
    /// a value the pair has.
    pub(crate) const fn admitted_regions(self) -> Region {
        match self {
            Kind::Int => Kind::Bool.region().union(Kind::Int.region()),
            _ => self.region(),
        }
    }

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
mod tests;
