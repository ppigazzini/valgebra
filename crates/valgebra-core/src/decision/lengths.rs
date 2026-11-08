//! A Boolean term of length bounds over one base, read as the lengths it holds.
//!
//! A refinement of `str`, `bytes`, or a sequence with no prefix by `MinLen` and
//! `MaxLen` alone holds the values of its base whose length lies in an
//! interval. A union, a meet and a complement of such refinements over one base
//! hold the values of the base whose length lies in the union, the meet or the
//! complement of the intervals, and a complement holds every value outside the
//! base besides [FCB08, Def. 4.1]. So such a term denotes a set of lengths and
//! one bit, read by a fold over the term, and two of them over one base are
//! related as their lengths are: `a ⊆ b` is `a ∧ ¬b = ∅` [FCB08, §4.1], a
//! difference of interval sets.
//!
//! The descriptor holds the same sets as automata with a state per element a
//! bound counts, and refuses a bound past its state bound: `str` past 584 code
//! points, `bytes` past 4,094 bytes, a list past 2,046 elements. Over the
//! Boolean terms of the bounds a translation from JSON Schema writes and the
//! fuzzing harness draws, every relation declined at depths 2 to 7 over `str`
//! and `bytes` had met that refusal. Read here a bound is two integers whatever
//! it counts, so a term the descriptor refuses for its size is decided, and one
//! it builds is decided without the build. The intervals are the descriptor's
//! own [`IntervalSet`], so the operations are the ones the integer component is
//! held to.

use crate::descr::budget;
use crate::descr::interval::IntervalSet;
use crate::descr::lower::WORK;
use crate::ir::{Constraint, LONGEST, REACHED, Schema};
use crate::joined_in_rounds;
use crate::verdict::Relation;

use super::MAX_DECISION_DEPTH;
use super::constraints::{longest, shortest};
use super::emptiness::the_rules_find_a_value;

/// What a term of length bounds over one base holds: the values of `base`
/// whose length is in `lengths`, and every value outside `base` where
/// `beyond`. A term naming no base -- the universe, nothing -- has none.
pub(super) struct ByLength<'a> {
    base: Option<&'a Schema>,
    lengths: IntervalSet,
    beyond: bool,
}

impl Schema {
    /// This schema read as the lengths it holds, or `None` where it is no
    /// Boolean term of length bounds over one base whose values hold every
    /// length.
    ///
    /// Each node read spends a unit of the lowering's allowance ([`WORK`]),
    /// and the term nests at most [`MAX_DECISION_DEPTH`] levels, so a term
    /// the reading refuses costs what the lowering's refusal does: a schema
    /// shares its subterms, and a tree read whole is exponential in a term
    /// built of shared ones.
    pub(super) fn read_as_lengths(&self) -> Option<ByLength<'_>> {
        let read = budget::under(WORK, || lengths_of(self, MAX_DECISION_DEPTH))?;
        read.base.is_some_and(holds_every_length).then_some(read)
    }

    /// The relation of two terms of length bounds over one base, read as
    /// their lengths ([`ByLength::relation_to`]); `None` where either is no
    /// such term, or the two count over different bases.
    pub(super) fn related_by_lengths(&self, other: &Schema) -> Option<Relation> {
        let mine = budget::under(WORK, || lengths_of(self, MAX_DECISION_DEPTH))?;
        let theirs = budget::under(WORK, || lengths_of(other, MAX_DECISION_DEPTH))?;
        if !agree(mine.base, theirs.base) {
            return None;
        }
        let base = one_base(mine.base, theirs.base)?;
        holds_every_length(base).then(|| mine.relation_to(&theirs))
    }
}

impl ByLength<'_> {
    /// Whether this term holds more words than `constants` constants are: a
    /// length of a `str` or a `bytes` holds the alphabet to its power in words,
    /// so where the term holds a length past what that many constants can name,
    /// one of its words is no constant of them. A code point is a `str`'s
    /// letter, lone surrogates included, and a byte a `bytes`'s.
    pub(super) fn holds_more_words_than(&self, constants: usize) -> bool {
        let letters: u128 = match self.base {
            Some(Schema::Str) => 0x11_0000,
            Some(Schema::Bytes) => 256,
            _ => return false,
        };
        let Ok(constants) = u128::try_from(constants) else {
            return false;
        };
        let mut length: i64 = 1;
        let mut words = letters;
        while words <= constants {
            length += 1;
            words = words.saturating_mul(letters);
        }
        !self
            .lengths
            .intersect(&lengths_held())
            .intersect(&lengths_up_to(REACHED))
            .intersect(&IntervalSet::between(Some(length), None))
            .is_empty()
    }

    /// Nothing: no length, and no value beyond a base.
    fn nothing() -> Self {
        ByLength {
            base: None,
            lengths: IntervalSet::empty(),
            beyond: false,
        }
    }

    /// The universe: every length, and every value beyond a base.
    fn anything() -> Self {
        ByLength {
            base: None,
            lengths: IntervalSet::all(),
            beyond: true,
        }
    }

    /// Whether this term holds no value: no length a value of its base has,
    /// and nothing beyond it.
    pub(super) fn holds_none(&self) -> bool {
        !self.beyond && self.lengths.intersect(&lengths_held()).is_empty()
    }

    /// How this term relates to `other`, a term over the same base or over
    /// none, read off what this holds and `other` does not, since `a ⊆ b` is
    /// `a ∧ ¬b = ∅` [FCB08, §4.1].
    ///
    /// `Holds` where the lengths this holds and `other` does not are no
    /// length a value has, and nothing beyond the base is this term's alone.
    /// `Fails` on a value of this term outside `other`: one beyond the base,
    /// which every base leaves -- `None` is outside each -- or a value of the
    /// base whose length is in the difference, at most [`REACHED`], the
    /// longest a container every interpreter builds. A difference holding
    /// only lengths past it is declined, as a length bound past it is.
    fn relation_to(&self, other: &ByLength<'_>) -> Relation {
        if self.beyond && !other.beyond {
            return Relation::Fails;
        }
        let outside = self
            .lengths
            .intersect(&other.lengths.complement())
            .intersect(&lengths_held());
        if outside.is_empty() {
            Relation::Holds
        } else if outside.intersect(&lengths_up_to(REACHED)).is_empty() {
            Relation::Unknown
        } else {
            Relation::Fails
        }
    }
}

/// The lengths a value has: none past [`LONGEST`].
fn lengths_held() -> IntervalSet {
    lengths_up_to(LONGEST)
}

/// The lengths from zero to `most`.
fn lengths_up_to(most: usize) -> IntervalSet {
    IntervalSet::between(Some(0), i64::try_from(most).ok())
}

/// The term `schema` read as its lengths, with `depth` levels left; `None`
/// where a node is no term of length bounds, two count over different bases,
/// or the allowance or the depth is spent.
///
/// Each connective is the set operation it denotes [FCB08, Def. 4.1], taken
/// on the lengths: a union joins them, a meet meets them, and a complement
/// takes the lengths its inner term leaves and every value beyond the base.
fn lengths_of(schema: &Schema, depth: u32) -> Option<ByLength<'_>> {
    let depth = depth.checked_sub(1)?;
    if !budget::spend() {
        return None;
    }
    match schema {
        Schema::Refine { base, constraints }
            if counts_lengths(base)
                && constraints.iter().all(|constraint| {
                    matches!(constraint, Constraint::MinLen(_) | Constraint::MaxLen(_))
                }) =>
        {
            let least = i64::try_from(shortest(constraints.iter())).ok()?;
            let most = longest(constraints.iter())
                .map(i64::try_from)
                .transpose()
                .ok()?;
            Some(ByLength {
                base: Some(base),
                lengths: IntervalSet::between(Some(least), most),
                beyond: false,
            })
        }
        bare if counts_lengths(bare) => Some(ByLength {
            base: Some(bare),
            lengths: IntervalSet::all(),
            beyond: false,
        }),
        Schema::Nothing => Some(ByLength::nothing()),
        Schema::Anything(_) => Some(ByLength::anything()),
        Schema::Complement(inner) => {
            let inner = lengths_of(inner, depth)?;
            Some(ByLength {
                base: inner.base,
                lengths: inner.lengths.complement(),
                beyond: !inner.beyond,
            })
        }
        Schema::Union(members) => joined(members, depth, ByLength::nothing, |a, b| {
            agree(a.base, b.base).then(|| ByLength {
                base: one_base(a.base, b.base),
                lengths: a.lengths.union(&b.lengths),
                beyond: a.beyond || b.beyond,
            })
        }),
        Schema::Intersection(members) => joined(members, depth, ByLength::anything, |a, b| {
            agree(a.base, b.base).then(|| ByLength {
                base: one_base(a.base, b.base),
                lengths: a.lengths.intersect(&b.lengths),
                beyond: a.beyond && b.beyond,
            })
        }),
        _ => None,
    }
}

/// The members of a union or a meet read as lengths and joined two at a time,
/// each round halving the list; `identity` where there are none.
///
/// A fold joining the whole so far with each member in turn costs a union of
/// disjoint bounds the square of its width, since the whole grows by a span a
/// member: four thousand of them took 127 ms where the descriptor's refusal
/// took 2. Joined in rounds, each span is merged once a round, and there are
/// as many rounds as the width has binary digits.
fn joined<'a>(
    members: &'a [Schema],
    depth: u32,
    identity: impl FnOnce() -> ByLength<'a>,
    join: impl Fn(ByLength<'a>, ByLength<'a>) -> Option<ByLength<'a>>,
) -> Option<ByLength<'a>> {
    joined_in_rounds(members, |member| lengths_of(member, depth), identity, join)
}

/// The base two parts of a term count over: the one either names, or `None`
/// where neither does. Two that name different bases are no term over one,
/// and the caller's fold stops at them (`agree`).
fn one_base<'a>(a: Option<&'a Schema>, b: Option<&'a Schema>) -> Option<&'a Schema> {
    a.or(b)
}

/// Whether two parts of a term count over one base, or one of them over none.
fn agree(a: Option<&Schema>, b: Option<&Schema>) -> bool {
    a.zip(b).is_none_or(|(a, b)| a == b)
}

/// Whether `schema` is a base whose values' length a length bound reads:
/// `str`, `bytes`, or a list or a tuple of one element type with no prefix.
/// A set's length is bounded by the values its element holds apart, a
/// mapping's by its keys, and a prefix by its positions, so none of those is
/// read here.
fn counts_lengths(schema: &Schema) -> bool {
    match schema {
        Schema::Str | Schema::Bytes => true,
        Schema::Seq { shape, .. } => shape.prefix.is_empty() && shape.tail.is_some(),
        _ => false,
    }
}

/// Whether `base` holds a value of every length: a word does, and a sequence
/// does where its element holds a value, found by the rules. A sequence of
/// an empty element holds the empty one alone, which this reading does not
/// count.
fn holds_every_length(base: &Schema) -> bool {
    match base {
        Schema::Seq { shape, .. } => shape
            .tail
            .as_deref()
            .is_some_and(|element| the_rules_find_a_value(core::slice::from_ref(element))),
        _ => true,
    }
}
