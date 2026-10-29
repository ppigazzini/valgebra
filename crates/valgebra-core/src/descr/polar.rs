//! A union of summands under a polarity: the device the four lattices built
//! from parts share.
//!
//! A kind's lines, the record atoms, the powerset lines and the map atoms are
//! each a set held as a finite union of summands, and each is closed under the
//! three operations the same way. A meet is a pairwise product. A complement is
//! De Morgan over the summands, `¬⋁ᵢSᵢ = ⋀ᵢ¬Sᵢ`, where each `¬Sᵢ` is itself a
//! finite union. Both multiply, so both are held to a width bound, and past it
//! there is no sound set to substitute: one too wide is complemented into one
//! too narrow, so the operation refuses.
//!
//! **The polarity is what keeps a complement total**, which the
//! [`Guard`](super::Guard) contract asks of every component. Rebuilding a
//! complement as a union is a product and can pass the bound; flipping a flag
//! cannot. The product is paid by the operation that needs the summands, where
//! a refusal is already allowed. So a union here is the summands held, or the
//! summands *not* held, and what a summand is -- and what being outside one
//! means -- is the one thing each lattice says for itself ([`Summand`]).

use std::borrow::Cow;

use super::budget;
use crate::verdict::Verdict;

/// One summand of a union a [`PolarUnion`] holds: the operations the shared
/// device reads off a part, and the bound on how many parts one union may hold.
pub(super) trait Summand: Clone + Ord + Sized {
    /// What an operation is told beyond its operands: the kind a line serves,
    /// whose whole a complement is taken against. `()` where the summand
    /// carries its universe itself.
    type Within: Copy;

    /// The summands a summand's complement is the union of.
    type Complement: AsRef<[Self]>;

    /// The most summands one union may hold.
    const MAX: usize;

    /// The summand holding every value of the universe `within` names.
    fn top(within: Self::Within) -> Self;

    /// The values both hold, or `None` where a part of the meet refuses.
    fn meet(&self, other: &Self) -> Option<Self>;

    /// The values of the universe this summand does not hold, as a union.
    fn complement(&self, within: Self::Within) -> Self::Complement;

    /// Bring one summand into a list kept in the lattice's canonical form:
    /// dropped where it is proved to hold nothing or the list holds it
    /// already, merged into a summand of the list where the lattice merges
    /// two into one, and kept, in the shape this lattice compares them in,
    /// otherwise. `None` where bringing it into that shape refuses. The list
    /// is left unsorted; [`compacted`](Summand::compacted) puts it in order.
    ///
    /// The lattice's own reading of what makes two spellings of a union one,
    /// which is why it is not written once: each lattice has its own canonical
    /// summand, and one merges two summands into one. "Each once" is by
    /// equality rather than by position after a sort, because a guard's
    /// equality can be coarser than its order -- an integer set compares two
    /// spellings of one set equal and sorts them apart -- so two equal
    /// summands need not be neighbors once sorted.
    ///
    /// One summand at a time, as a union is built, which is what keeps a
    /// product at its bound from reading its whole list again for every pair:
    /// [`product`] admits each pair as it is met, the way a type checker's
    /// union builder checks a member against the ones it holds as it adds it
    /// rather than simplifying the whole union again.
    fn admit(kept: &mut Vec<Self>, summand: Self) -> Option<()>;

    /// A list a meet or a union has just built, in the lattice's canonical
    /// form: every summand admitted ([`admit`](Summand::admit)), and the list
    /// put in order. `None` where bringing one into shape refuses. [`tidy`]
    /// holds what this returns to [`MAX`](Summand::MAX).
    fn compacted(summands: Vec<Self>) -> Option<Vec<Self>> {
        let mut kept = Vec::with_capacity(summands.len());
        for summand in summands {
            Self::admit(&mut kept, summand)?;
        }
        kept.sort();
        Some(kept)
    }
}

/// A set held as a union of summands, or as the complement of one.
///
/// No summands is the empty set, and no summands *negated* is the whole
/// universe, which is what makes the two bounds cost nothing to hold.
///
/// **Not canonical in general.** The polarity carries what a product could not
/// lay out, so one set has a positive spelling and a negated one, and a union
/// whose summands overlap has several positive ones. Equality is therefore
/// finer than agreeing on values, and a law over one of these is asked of the
/// values rather than of the forms.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct PolarUnion<S> {
    summands: Vec<S>,
    /// Whether the summands are the values held or the values *not* held.
    negated: bool,
}

impl<S: Summand> PolarUnion<S> {
    /// The union of `summands`, held as they are.
    ///
    /// The caller's list is its canonical form already -- a constructor's one
    /// summand, or the output of [`tidy`] -- so nothing is compacted here.
    pub(super) const fn of(summands: Vec<S>) -> Self {
        Self {
            summands,
            negated: false,
        }
    }

    /// A union held under the polarity given, as a test spells one.
    #[cfg(test)]
    pub(super) const fn with_polarity(summands: Vec<S>, negated: bool) -> Self {
        Self { summands, negated }
    }

    /// The summands as held, whichever polarity reads them.
    #[cfg(test)]
    pub(super) fn summands(&self) -> &[S] {
        &self.summands
    }

    /// Whether the summands are the values *not* held.
    #[cfg(test)]
    pub(super) const fn is_negated(&self) -> bool {
        self.negated
    }

    /// The summands of the values this holds, complementing a negated form,
    /// or `None` where that complement passes the bound.
    ///
    /// Borrowed where the list is already positive, which is the common case
    /// and the one asked most often: a union is asked whether it is empty once
    /// per question about it, and copying the list to read whether it holds a
    /// summand is the whole cost of asking.
    pub(super) fn positive(&self, within: S::Within) -> Option<Cow<'_, [S]>> {
        if self.negated {
            complement_all(&self.summands, within).map(Cow::Owned)
        } else {
            Some(Cow::Borrowed(&self.summands))
        }
    }

    /// What is known about this holding a value, given what is known of each
    /// summand.
    ///
    /// A union's verdict: empty when every summand is proved empty, inhabited
    /// as soon as one is. A negated form is expanded first, and a refusal there
    /// is *unknown* rather than inhabited -- past the bound there is no union
    /// to read, so nothing is proved either way.
    pub(super) fn verdict(&self, within: S::Within, of: impl Fn(&S) -> Verdict) -> Verdict {
        match self.positive(within) {
            Some(summands) => Verdict::any(summands.iter().map(of)),
            None => Verdict::Unknown,
        }
    }

    /// Whether a value is held, given which summands hold it.
    ///
    /// Asked of the summands as they are: a value in some summand of a negated
    /// form is a value the set does *not* hold, so the polarity flips the
    /// answer and nothing is expanded.
    #[cfg(test)]
    pub(super) fn holds(&self, held_by: impl FnMut(&S) -> bool) -> bool {
        self.summands.iter().any(held_by) != self.negated
    }

    /// The values in either, or `None` past the bound.
    ///
    /// A negated side is joined through De Morgan rather than rebuilt: `A ∪ B`
    /// is `¬(¬A ∩ ¬B)`, and the meet is the operation that can drop a summand
    /// mid-way. Expanding the negation first asks the bound about a union that
    /// is only an intermediate, and a refusal there refuses a join whose
    /// answer fits.
    pub(super) fn union(&self, other: &Self, within: S::Within) -> Option<Self> {
        if self.negated || other.negated {
            let met = self
                .complement(within)
                .intersect(&other.complement(within), within)?;
            return Some(met.complement(within));
        }
        let mut summands = self.summands.clone();
        summands.extend(other.summands.iter().cloned());
        Some(Self::of(tidy(summands)?))
    }

    /// The values in both, or `None` past the bound or where a summand's meet
    /// refuses.
    ///
    /// A negated side is removed one summand at a time rather than rebuilt into
    /// a union first. `¬⋁ᵢSᵢ` is `⋀ᵢ¬Sᵢ` [FCB08, Lem. 6.2], so both orders
    /// compute this set; what they differ in is the widest intermediate they
    /// ask the bound about. Rebuilding first multiplies every `¬Sᵢ` together
    /// with nothing to narrow the product, while meeting each factor into what
    /// is already held drops the summands holding nothing before the next
    /// factor multiplies them. A bound reached under one spelling of a
    /// difference and not the other would make a relation's answer a property
    /// of how it was written.
    pub(super) fn intersect(&self, other: &Self, within: S::Within) -> Option<Self> {
        let mut summands = match (self.negated, other.negated) {
            // The meet of two unions is the union of the pairwise meets
            // [FCB08, Lem. 6.2].
            (false, false) => product(&self.summands, &other.summands)?,
            (false, true) => self.summands.clone(),
            (true, false) => other.summands.clone(),
            // Two negated sides leave nothing positive to start from, so the
            // meet starts at the whole universe and both sides narrow it.
            (true, true) => vec![S::top(within)],
        };
        // Each negated summand is subtracted from what is held, the way the
        // proof of [Cas23, Thm. 4.2] subtracts the negated atoms one at a time.
        for negated in [self, other].into_iter().filter(|side| side.negated) {
            for summand in &negated.summands {
                summands = product(&summands, summand.complement(within).as_ref())?;
            }
        }
        Some(Self::of(summands))
    }

    /// The values this does not hold.
    ///
    /// Total, which is what the [`Guard`](super::Guard) contract asks. A
    /// negated form's complement is its summands held positively. A positive
    /// one is rebuilt where the rebuild is one product, which is what keeps
    /// the cheap forms canonical -- complementing the whole universe gives back
    /// exactly the empty set rather than a second spelling of it. Past that
    /// the negation is carried: rebuilding a wide union's complement here
    /// spends the build's allowance on an intermediate the meet it is headed
    /// for would have pruned, and a meet against a negated side removes one
    /// summand at a time instead.
    #[must_use]
    pub(super) fn complement(&self, within: S::Within) -> Self {
        if self.negated {
            return Self::of(self.summands.clone());
        }
        let carried = || Self {
            summands: self.summands.clone(),
            negated: true,
        };
        if self.summands.len() > 1 {
            return carried();
        }
        complement_all(&self.summands, within).map_or_else(carried, Self::of)
    }
}

/// The summands a union of summands complements into, or `None` past the
/// bound.
///
/// De Morgan over the summands: `¬⋁ᵢSᵢ` is `⋀ᵢ¬Sᵢ` [FCB08, Lem. 6.2], and each
/// `¬Sᵢ` is the union [`Summand::complement`] gives, so the fold is a product
/// rather than a subtraction. It starts from the whole universe, which is what
/// complementing no summands yields.
pub(super) fn complement_all<S: Summand>(summands: &[S], within: S::Within) -> Option<Vec<S>> {
    let mut whole = vec![S::top(within)];
    for summand in summands {
        whole = product(&whole, summand.complement(within).as_ref())?;
    }
    Some(whole)
}

/// The summands of a meet, which is a meet of every pair, the meet of two
/// unions in normal form [FCB08, Lem. 6.2].
///
/// Every pair charges the build's allowance, because this is the loop that
/// multiplies: the pairs are the product of the two counts, a fold over several
/// unions raises that to a power, and a meet of two guards descends a level of
/// nesting for each pair. The width bound stops the result from being too
/// wide, and the allowance stops the *work* from being too much before the
/// width is known. See [`budget`].
///
/// **Each pair is admitted as it is met** ([`Summand::admit`]), so the list is
/// in canonical form at every pair and the bound is read off it there: a list
/// past [`MAX`](Summand::MAX) once its repeats and the summands holding nothing
/// are gone refuses at the pair that takes it past, and the result is what
/// [`tidy`] would make of every pair at the end -- the same summands, kept
/// once, in order. A unit of the allowance then buys one pair and one reading
/// of the list. Compacting the whole list again whenever it reaches the bound
/// would read every kept summand against every other each time a few slots
/// come free: a product of map atoms standing at the bound would spend the
/// width squared in comparisons of trees every few pairs, sixty-five thousand
/// for two hundred and fifty-six atoms.
pub(super) fn product<S: Summand>(left: &[S], right: &[S]) -> Option<Vec<S>> {
    let mut kept = Vec::new();
    for mine in left {
        for theirs in right {
            if !budget::spend() {
                return None;
            }
            S::admit(&mut kept, mine.meet(theirs)?)?;
            if kept.len() > S::MAX {
                return None;
            }
        }
    }
    kept.sort();
    kept.dedup();
    Some(kept)
}

/// Bring a list into the lattice's canonical form, and refuse a union past
/// the bound.
///
/// Dropping a summand proved empty is not an optimization: it contributes no
/// value to the union, so removing it leaves the same set and keeps the count
/// from growing on shapes that describe nothing. One merely *unknown* stays,
/// because it may yet hold a value. The order is what makes two equal unions
/// compare equal, as far as equality here goes.
pub(super) fn tidy<S: Summand>(summands: Vec<S>) -> Option<Vec<S>> {
    let kept = S::compacted(summands)?;
    (kept.len() <= S::MAX).then_some(kept)
}

#[cfg(test)]
mod tests;
