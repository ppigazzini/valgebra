//! Sets of integers that repeat: an interval set per residue class.
//!
//! A refinement over `int` can bound (`Ge(0)`), pin (`Literal[3]`) and *step*
//! (`MultipleOf(3)`). Bounds and points are intervals; a step is not, and no
//! finite union of intervals is one. What holds all three, and is closed under
//! union, intersection and complement, is the class of **eventually periodic**
//! sets -- finite unions of an interval met with a residue class -- which is
//! what one variable of Presburger arithmetic defines.
//!
//! The representation makes the three operations pointwise. A set is a modulus
//! `m` together with an [`IntervalSet`] per residue `r` in `0..m`, and the
//! interval set is read in the residue class's *own* coordinates: it holds `k`
//! exactly when the set holds `r + m*k`. A residue class is order-isomorphic to
//! the integers, so nothing is lost, and every operation becomes the interval
//! operation applied residue by residue.
//!
//! An operation between two periods reads both tables at the least common
//! multiple, which is where the classes line up. Lifting is exact -- it splits
//! each class into `t` classes and re-indexes -- so two sets compared there are
//! equal exactly when they hold the same integers, and an answer built there
//! holds the integers it should whichever periods it came from.
//!
//! The form is **not canonical**: a set can be written at every multiple of the
//! period it needs, and no reduction picks one of those spellings out. Reducing
//! the multiples of two met with `0..=8000` to a period of one is exact and
//! turns one interval into four thousand, and the interval count has no bound
//! the period has. That is why equality lifts, and why the *order* beside it
//! cannot: see [`IntSet`] for the split and what it costs.

use std::borrow::Cow;

use super::interval::IntervalSet;

/// A set of integers, held as an interval set per residue class.
///
/// **Equality is on the integers and the order is on the table.** The split is
/// deliberate, and neither half can be moved to the other.
///
/// Equality lifts both tables to the period the pair shares, so the multiples
/// of two and the same set written with a period of four are one set. A
/// descriptor's components are compared as sets, which is what lets a law about
/// the algebra be checked on the forms themselves rather than over whatever
/// values a corpus can list, so equality has to answer for the integers.
///
/// An order read the same way is not an order. Each pair would be read at its
/// own period, and three sets then come out in a cycle: the integers sort
/// before `{-2}` at the period those two share, `{-2}` sorts before the
/// multiples of three at the period *those* two share, and the integers sort
/// after the multiples of three at the period they share. Antisymmetry holds of
/// that triple and so does agreement with equality; a sort finds the cycle and
/// panics on it.
///
/// What would settle both at once is a canonical spelling, and the module
/// header says why there is none to have. So the two disagree, on exactly one
/// thing: two spellings of one set are one set to equality and two positions to
/// the order. The cost is a `dedup` that follows a sort keeping a pair that a
/// scan for equality folds -- a row in a table of guards, never an answer.
#[derive(Debug, Clone)]
pub struct IntSet {
    /// The period. At least one; a modulus of one is a set with no step, whose
    /// single class is the integers themselves.
    modulus: i64,
    /// One interval set per residue, indexed by the residue. `classes[r]` holds
    /// `k` exactly when this set holds `r + modulus * k`.
    classes: Vec<IntervalSet>,
}

/// The largest period this representation materialises.
///
/// A class per residue is what makes the three operations pointwise, and it is
/// also what makes the cost linear in the period: `MultipleOf(n)` holds `n`
/// interval sets, and two coprime steps meet at their product. The bound is
/// generous against what a real annotation asks for -- a step is a divisibility
/// check, and the ones people write are small -- and it is a **limit of the
/// representation**, not an approximation: a step beyond it stays opaque rather
/// than being rounded to one this can hold.
///
/// The bound is on one step *and on their composition*, and the second is the
/// half that is easy to lose. `MultipleOf` lowers to an `IntSet`
/// ([`lower`](super::lower)), so two steps a caller may write independently --
/// each far inside the bound -- meet at their least common multiple, which need
/// not be: 64 and 81 meet at 5,184. That composition is refused by the
/// operations rather than rounded, because there is no sound set to substitute
/// for a period this cannot hold. One too wide is complemented into one too
/// narrow, so a rounded answer is wrong in one direction or the other, and a
/// refusal is what every caller of a descriptor operation already handles.
pub const MAX_PERIOD: i64 = 4096;

impl IntSet {
    /// A set with one class per residue, built by `of`, for a period the
    /// representation holds.
    ///
    /// Only the constructors whose period is one reach this directly; anything
    /// deriving a period from another set's goes through
    /// [`try_build`](Self::try_build), which refuses rather than asserting.
    fn build(modulus: i64, of: impl Fn(i64) -> IntervalSet) -> IntSet {
        debug_assert!(modulus >= 1, "a modulus is at least one");
        debug_assert!(
            modulus <= MAX_PERIOD,
            "a period of {modulus} materialises that many classes, past the \
             {MAX_PERIOD} this representation holds"
        );
        build_unchecked(modulus, of)
    }

    /// The same, for a period that may be past the bound: `None` where it is.
    ///
    /// The refusal is the whole point. Clamping the period here would build a
    /// table for a *different* set and hand it back as this one -- the shape of
    /// wrong answer this representation exists to avoid -- and asserting would
    /// turn a composition a caller is entitled to write into a panic on the
    /// debug builds every contributor runs.
    fn try_build(modulus: i64, of: impl Fn(i64) -> IntervalSet) -> Option<IntSet> {
        (1..=MAX_PERIOD)
            .contains(&modulus)
            .then(|| build_unchecked(modulus, of))
    }

    /// The empty set.
    #[must_use]
    pub fn empty() -> IntSet {
        IntSet::build(1, |_| IntervalSet::empty())
    }

    /// Every integer.
    #[must_use]
    pub fn all() -> IntSet {
        IntSet::build(1, |_| IntervalSet::all())
    }

    /// The one-element set.
    #[must_use]
    pub fn just(value: i64) -> IntSet {
        IntSet::build(1, |_| IntervalSet::just(value))
    }

    /// The integers from `lo` to `hi` inclusive, with `None` unbounded.
    #[must_use]
    pub fn between(lo: Option<i64>, hi: Option<i64>) -> IntSet {
        IntSet::build(1, |_| IntervalSet::between(lo, hi))
    }

    /// The multiples of `step`, or `None` for a step past [`MAX_PERIOD`].
    ///
    /// The one constructor that needs a modulus: `MultipleOf(3)` is the residue
    /// class of zero, which no union of intervals holds. A step of zero divides
    /// nothing and a negative step names the same multiples as its magnitude, so
    /// both are read as their absolute value, and zero gives the singleton `{0}`
    /// -- the only integer that is a multiple of nothing.
    ///
    /// The refusal is in the return type rather than in an assertion, because it
    /// is a decision the caller has to make. There is no sound approximation to
    /// substitute: a set that is too wide or too narrow is complemented into one
    /// that is wrong the other way, so a step this cannot hold must stay opaque
    /// in whatever lowers it.
    #[must_use]
    pub fn multiple_of(step: i64) -> Option<IntSet> {
        match step.checked_abs() {
            Some(0) | None => Some(IntSet::just(0)),
            Some(step) if step > MAX_PERIOD => None,
            Some(step) => Some(IntSet::build(step, |residue| {
                if residue == 0 {
                    IntervalSet::all()
                } else {
                    IntervalSet::empty()
                }
            })),
        }
    }

    /// Whether this set holds no integer.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.classes.iter().all(IntervalSet::is_empty)
    }

    /// Whether this set holds `value`.
    #[must_use]
    pub fn holds(&self, value: i64) -> bool {
        // `value = residue + modulus * k`, and the class holds that `k`.
        let residue = value.rem_euclid(self.modulus);
        self.classes
            .get(usize::try_from(residue).unwrap_or(0))
            .is_some_and(|class| class.holds(value.div_euclid(self.modulus)))
    }

    /// This set's table read at `modulus`, which its period must divide, or
    /// `None` for a period past [`MAX_PERIOD`].
    ///
    /// A class `r` mod `m` splits into `t = modulus / m` classes `r + m*j`, and
    /// each keeps the integers of the original that land in it: `r + m*k` is in
    /// the new class `r + m*j` exactly when `k = j + t*k'`, so the new class's
    /// interval set is the preimage of the old one under that map.
    ///
    /// A table and not an [`IntSet`], because a table at a period this set does
    /// not carry is a *second spelling* of it, and a second spelling is a set
    /// the order puts somewhere else: a caller that wants one says so. Borrowed
    /// where the period is the one this set already has, which is every
    /// comparison and every operation between two sets of one period.
    fn table_at(&self, modulus: i64) -> Option<Cow<'_, [IntervalSet]>> {
        if modulus == self.modulus {
            return Some(Cow::Borrowed(&self.classes));
        }
        let stride = modulus / self.modulus;
        (1..=MAX_PERIOD).contains(&modulus).then(|| {
            Cow::Owned(
                (0..modulus)
                    .map(|residue| {
                        let old = residue.rem_euclid(self.modulus);
                        let step = (residue - old) / self.modulus;
                        self.classes
                            .get(usize::try_from(old).unwrap_or(0))
                            .map_or_else(IntervalSet::empty, |class| class.preimage(step, stride))
                    })
                    .collect(),
            )
        })
    }

    /// The period two sets share, where their classes line up.
    fn common(&self, other: &IntSet) -> i64 {
        lcm(self.modulus, other.modulus)
    }

    /// Combine two sets residue by residue, after lifting both to one period,
    /// or `None` where that period is past [`MAX_PERIOD`].
    fn zip(
        &self,
        other: &IntSet,
        op: fn(&IntervalSet, &IntervalSet) -> IntervalSet,
    ) -> Option<IntSet> {
        let modulus = self.common(other);
        let (mine, theirs) = (self.table_at(modulus)?, other.table_at(modulus)?);
        let combined = IntSet::try_build(modulus, |residue| {
            let index = usize::try_from(residue).unwrap_or(0);
            match (mine.get(index), theirs.get(index)) {
                (Some(a), Some(b)) => op(a, b),
                _ => IntervalSet::empty(),
            }
        })?;
        Some(combined.without_a_step())
    }

    /// This set written with no step where its classes do not need one.
    ///
    /// The one reduction that is worth making, and the module header says why
    /// no general one is: a step that cancels leaves a table saying the same
    /// thing in every residue, and dropping it costs nothing, because the two
    /// tables a period of one can say that with -- every integer, or none --
    /// are the two a period does not shorten. Carrying the period instead would
    /// put `multiple_of(64) | !multiple_of(64)` and [`all`](Self::all) in two
    /// places in a sorted table, and a third for every other step a caller
    /// joins with its own complement.
    fn without_a_step(self) -> IntSet {
        let Some(first) = self.classes.first() else {
            return self;
        };
        if self.modulus == 1 || !self.classes.iter().all(|class| class == first) {
            return self;
        }
        // Every residue holds the same `k`, so the set is the union of
        // `r + modulus * k` over all `r` -- which is that interval set scaled
        // back to a period of one only when it is one of the two sets scaling
        // cannot move: nothing, or everything.
        if first.is_empty() {
            IntSet::empty()
        } else if *first == IntervalSet::all() {
            IntSet::all()
        } else {
            self
        }
    }

    /// The integers in either set, or `None` past [`MAX_PERIOD`].
    #[must_use]
    pub fn union(&self, other: &IntSet) -> Option<IntSet> {
        self.zip(other, IntervalSet::union)
    }

    /// The integers in both sets, or `None` past [`MAX_PERIOD`].
    #[must_use]
    pub fn intersect(&self, other: &IntSet) -> Option<IntSet> {
        self.zip(other, IntervalSet::intersect)
    }

    /// Every integer this set does not hold.
    ///
    /// Residue by residue, which is the whole of it: the classes partition the
    /// integers, so complementing each one complements their union. That is why
    /// the period is carried rather than the pieces -- a union of
    /// interval-and-residue pairs would have to distribute a complement over
    /// every pair.
    #[must_use]
    pub fn complement(&self) -> IntSet {
        IntSet {
            modulus: self.modulus,
            classes: self.classes.iter().map(IntervalSet::complement).collect(),
        }
    }
}

impl PartialEq for IntSet {
    /// Two sets are equal when they hold the same integers, which is not the
    /// same as carrying the same period.
    ///
    /// Lifting both to the period where their classes line up settles it, and
    /// lifting is exact, so this is equality of the sets rather than of two
    /// spellings.
    fn eq(&self, other: &IntSet) -> bool {
        let modulus = self.common(other);
        // Past the bound neither table can be read in the other's coordinates.
        // Two sets that reach here carry two periods -- one period meets itself
        // under the bound every set is built to -- so they are two spellings,
        // read as two sets. Conservative rather than wrong: it can answer
        // `false` for two sets that hold the same integers, and only for a pair
        // whose periods are near-coprime and large. What it cannot do is answer
        // `true` for two sets that differ, which is the direction a decision
        // rests on.
        match (self.table_at(modulus), other.table_at(modulus)) {
            (Some(mine), Some(theirs)) => mine == theirs,
            _ => false,
        }
    }
}

impl Eq for IntSet {}

impl Ord for IntSet {
    /// Order two sets by the tables they carry.
    ///
    /// A total order, which is the whole of what a sort of guards needs and the
    /// one thing an order read off a *pair* cannot be. It is on the table and
    /// not on the integers, so it puts two spellings of one set in two places:
    /// [`IntSet`] says why that is the half that gives way.
    fn cmp(&self, other: &IntSet) -> core::cmp::Ordering {
        (self.modulus, &self.classes).cmp(&(other.modulus, &other.classes))
    }
}

impl PartialOrd for IntSet {
    fn partial_cmp(&self, other: &IntSet) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// The least common multiple of two positive integers.
///
/// Saturating, because the product of two periods can leave the range: a
/// saturated modulus is wrong, so the caller is kept from reaching one by the
/// frontend's bound on how large a step may be. The debug assertion says which
/// invariant is broken if it ever is.
fn lcm(a: i64, b: i64) -> i64 {
    let divisor = gcd(a, b);
    debug_assert!(divisor > 0, "a modulus is at least one");
    let reduced = a / divisor.max(1);
    reduced.checked_mul(b).unwrap_or_else(|| {
        debug_assert!(false, "the periods {a} and {b} have no representable lcm");
        a.max(b)
    })
}

/// One class per residue, with no check on the period.
///
/// The two constructors above differ only in how they answer a period the
/// representation cannot hold -- an assertion, or a refusal -- and share the
/// table they build once it is known to be one.
fn build_unchecked(modulus: i64, of: impl Fn(i64) -> IntervalSet) -> IntSet {
    IntSet {
        modulus,
        classes: (0..modulus).map(of).collect(),
    }
}

/// The greatest common divisor of two positive integers, by Euclid.
fn gcd(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a, b);
    while b != 0 {
        let remainder = a.rem_euclid(b);
        a = b;
        b = remainder;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::{IntSet, MAX_PERIOD, gcd, lcm};
    use proptest::prelude::*;

    /// Union and meet inside the test corpus, where the bound is out of reach.
    ///
    /// Every leaf's period is one or a step of at most five, so every period
    /// these compose divides sixty. A refusal here would be a broken generator
    /// rather than a law that failed, which is why it is spelled as one.
    fn or(a: &IntSet, b: &IntSet) -> IntSet {
        a.union(b).expect("a period inside the bound")
    }

    fn and(a: &IntSet, b: &IntSet) -> IntSet {
        a.intersect(b).expect("a period inside the bound")
    }

    /// The outermost endpoint the generator writes. Past it every set it builds
    /// is purely periodic, because union, intersection and complement move no
    /// endpoint and add none.
    const REACH: i64 = 9;

    /// The integers that decide whether two sets are equal.
    ///
    /// Past [`REACH`] both sets repeat with the period they carry, so agreeing
    /// over one whole common period beyond it is agreeing everywhere. The
    /// window has to be derived rather than fixed: a fixed one narrower than
    /// the period reads a set whose only members lie past it as empty, which is
    /// agreement on an accident.
    fn window(a: &IntSet, b: &IntSet) -> core::ops::RangeInclusive<i64> {
        let period = lcm(a.modulus, b.modulus);
        // A period past the bound cannot be materialised, so it cannot be
        // compared either. Clamping keeps the window finite where the assertion
        // is compiled out; reaching it at all is the bug the assertion names.
        debug_assert!(period <= MAX_PERIOD, "a period of {period} past the bound");
        let edge = REACH.saturating_add(period.min(MAX_PERIOD));
        -edge..=edge
    }

    fn same(a: &IntSet, b: &IntSet) -> bool {
        window(a, b).all(|n| a.holds(n) == b.holds(n))
    }

    /// Sets built from bounded endpoints and small steps, so agreement on the
    /// window is agreement everywhere.
    fn int_set() -> impl Strategy<Value = IntSet> {
        let leaf = prop_oneof![
            Just(IntSet::empty()),
            Just(IntSet::all()),
            (-9i64..=9).prop_map(IntSet::just),
            (-9i64..=9).prop_map(|lo| IntSet::between(Some(lo), None)),
            (-9i64..=9).prop_map(|hi| IntSet::between(None, Some(hi))),
            (-9i64..=9, -9i64..=9)
                .prop_map(|(a, b)| IntSet::between(Some(a.min(b)), Some(a.max(b)))),
            (1i64..=5).prop_map(|step| IntSet::multiple_of(step).expect("a small step")),
        ];
        leaf.prop_recursive(4, 24, 2, |inner| {
            prop_oneof![
                (inner.clone(), inner.clone()).prop_map(|(a, b)| or(&a, &b)),
                (inner.clone(), inner.clone()).prop_map(|(a, b)| and(&a, &b)),
                inner.prop_map(|a| a.complement()),
            ]
        })
    }

    proptest! {
        // A bounded shrink, so a broken invariant cannot turn a caught mutation
        // into a run that outlasts a sweep: every draw is larger under one, and
        // shrinking a counterexample redraws it thousands of times.
        #![proptest_config(ProptestConfig {
            max_shrink_time: 2_000,
            ..ProptestConfig::default()
        })]

        /// The Boolean algebra, checked against the integers.
        #[test]
        fn the_lattice_laws_hold_of_the_integers(
            a in int_set(),
            b in int_set(),
            c in int_set(),
        ) {
            prop_assert!(same(&or(&a, &b), &or(&b, &a)));
            prop_assert!(same(&and(&a, &b), &and(&b, &a)));
            prop_assert!(same(&or(&or(&a, &b), &c), &or(&a, &or(&b, &c))));
            prop_assert!(same(&and(&and(&a, &b), &c), &and(&a, &and(&b, &c))));
            prop_assert!(same(&or(&a, &and(&a, &b)), &a));
            prop_assert!(same(&and(&a, &or(&a, &b)), &a));
            prop_assert!(same(
                &and(&a, &or(&b, &c)),
                &or(&and(&a, &b), &and(&a, &c))
            ));
        }

        /// The complement laws, and De Morgan both ways.
        #[test]
        fn the_complement_laws_hold_of_the_integers(a in int_set(), b in int_set()) {
            prop_assert!(and(&a, &a.complement()).is_empty());
            prop_assert!(same(&or(&a, &a.complement()), &IntSet::all()));
            prop_assert!(same(&a.complement().complement(), &a));
            prop_assert!(same(
                &or(&a, &b).complement(),
                &and(&a.complement(), &b.complement())
            ));
            prop_assert!(same(
                &and(&a, &b).complement(),
                &or(&a.complement(), &b.complement())
            ));
        }

        /// Holding the same integers is being equal, across periods.
        ///
        /// The property the lifting exists for: two sets built with different
        /// steps are compared where their classes line up, so the multiples of
        /// two and the same set written with a period of four are one set.
        #[test]
        fn holding_the_same_integers_is_being_equal(a in int_set(), b in int_set()) {
            prop_assert_eq!(same(&a, &b), a == b);
        }

        /// Emptiness is a decision, which is what a bound conjunction needs: the
        /// structural procedure compares two bounds and declines what it cannot
        /// pair, while this answers from the set.
        #[test]
        fn emptiness_agrees_with_the_integers(a in int_set()) {
            prop_assert_eq!(a.is_empty(), window(&a, &a).all(|n| !a.holds(n)));
        }

        /// The order is total, and one position holds one set.
        ///
        /// Transitivity is the property no *pair* can see: an order read off a
        /// pair is antisymmetric while putting three sets in a cycle, because
        /// each pair is read at its own period. Three draws are what lets the
        /// law fail, and [`no_three_sets_are_ordered_in_a_cycle`] pins the
        /// witness so the seed does not decide.
        ///
        /// Only one direction of agreement with equality is a law here, and it
        /// is the one the order owes: two sets in one position are one set,
        /// because one table read at one period is one set. The converse is
        /// what the order gives up to be total, and
        /// [`two_spellings_of_one_set_are_one_set_in_two_places`] is where that
        /// is said.
        #[test]
        fn the_order_is_total(a in int_set(), b in int_set(), c in int_set()) {
            prop_assert_eq!(a.partial_cmp(&b), Some(a.cmp(&b)));
            prop_assert_eq!(a.cmp(&b), b.cmp(&a).reverse());
            if a.cmp(&b) == core::cmp::Ordering::Equal {
                prop_assert_eq!(&a, &b, "one position holds one set");
            }
            if a < b && b < c {
                prop_assert!(a < c, "the order takes a step it cannot take twice");
            }
        }

        /// Reading a set's table at a multiple of its period changes no
        /// integer.
        ///
        /// What every operation between two periods rests on: the two tables
        /// are combined residue by residue, so a lift that moved an integer
        /// would move it in the answer, and what equality rests on, which is
        /// why the set built back from the table is equal to the one it came
        /// from.
        #[test]
        fn a_table_at_a_multiple_period_holds_the_same_integers(
            a in int_set(),
            factor in 1i64..=6,
        ) {
            let modulus = a.modulus.saturating_mul(factor);
            // A period of at most sixty times six is inside the bound, so the
            // table is available for every draw; a refusal here would be the
            // generator, not the lift.
            let classes = a.table_at(modulus).expect("a period inside the bound");
            let lifted = IntSet { modulus, classes: classes.into_owned() };
            prop_assert!(same(&a, &lifted));
            prop_assert_eq!(&a, &lifted);
        }
    }

    /// A set whose members all lie past a period is not the empty set.
    ///
    /// The window has to reach past the period for the laws above to mean what
    /// they say. The positive multiples of sixty are the smallest thing the
    /// generator builds that a narrower window reads as empty: three, four and
    /// five meet at sixty, and the first member is the period itself.
    #[test]
    fn a_set_whose_members_start_past_a_period_is_seen() {
        let sixties = and(
            &and(&and(&step(3), &step(4)), &step(5)),
            &IntSet::between(Some(1), None),
        );

        assert!(
            sixties.holds(60) && !sixties.holds(0),
            "the positive multiples"
        );
        assert!(!sixties.is_empty());
        assert_ne!(sixties, IntSet::empty());
        assert!(!same(&sixties, &IntSet::empty()));
    }

    /// The multiples of a step inside the bound.
    fn step(n: i64) -> IntSet {
        IntSet::multiple_of(n).expect("a small step is inside the bound")
    }

    /// A step is the thing intervals cannot express, and the reason the period
    /// is carried at all.
    #[test]
    fn a_step_holds_its_multiples_and_nothing_between() {
        let evens = IntSet::multiple_of(2).expect("two is inside the bound");
        for n in -6i64..=6 {
            assert_eq!(evens.holds(n), n % 2 == 0, "{n}");
        }
        // The complement of a step is the other residues, which is again a step
        // set rather than a union of intervals.
        let odds = evens.complement();
        for n in -6i64..=6 {
            assert_eq!(odds.holds(n), n % 2 != 0, "{n}");
        }
        assert!(and(&evens, &odds).is_empty());
        assert_eq!(or(&evens, &odds), IntSet::all());

        // Two steps meet at their least common multiple, which is what a naive
        // pairwise rule over bounds cannot see.
        let sixes = and(&step(2), &step(3));
        for n in -12i64..=12 {
            assert_eq!(sixes.holds(n), n % 6 == 0, "{n}");
        }
        // And two steps that share no multiple but zero still meet there.
        assert!(!and(&step(2), &step(3)).is_empty());
    }

    /// A bound conjunction the structural procedure declines, decided here by
    /// the set: no integer is both at least five and below one.
    #[test]
    fn a_bound_conjunction_that_cannot_hold_is_empty() {
        let low = IntSet::between(Some(5), None);
        let high = IntSet::between(None, Some(1));
        assert!(and(&low, &high).is_empty());
        // Adjacent bounds leave exactly the integers between them, and none is
        // the empty set rather than a negative-width range.
        assert!(
            and(
                &IntSet::between(Some(2), None),
                &IntSet::between(None, Some(1))
            )
            .is_empty()
        );
        assert_eq!(
            and(
                &IntSet::between(Some(1), None),
                &IntSet::between(None, Some(1))
            ),
            IntSet::just(1)
        );
        // An even integer strictly between two consecutive even numbers: the
        // step and the bounds together empty a set neither empties alone.
        let between = and(&step(2), &IntSet::between(Some(3), Some(3)));
        assert!(between.is_empty());
    }

    /// A step of zero divides nothing, and a negative step names the multiples
    /// of its magnitude.
    #[test]
    fn a_degenerate_step_is_read_as_the_set_it_names() {
        assert_eq!(IntSet::multiple_of(0), Some(IntSet::just(0)));
        assert_eq!(IntSet::multiple_of(-3), IntSet::multiple_of(3));
        assert_eq!(IntSet::multiple_of(1), Some(IntSet::all()));
        // The magnitude of the smallest integer is not representable, and it is
        // read as the step that divides nothing rather than wrapping to itself.
        assert_eq!(IntSet::multiple_of(i64::MIN), Some(IntSet::just(0)));
    }

    /// A step past the bound has no representation, and the refusal says so
    /// rather than substituting a set that is wrong in one direction and,
    /// complemented, wrong in the other.
    #[test]
    fn a_step_past_the_period_bound_is_refused() {
        assert!(IntSet::multiple_of(MAX_PERIOD).is_some());
        assert!(IntSet::multiple_of(-MAX_PERIOD).is_some());
        assert!(IntSet::multiple_of(MAX_PERIOD + 1).is_none());
        assert!(IntSet::multiple_of(-(MAX_PERIOD + 1)).is_none());
        assert!(IntSet::multiple_of(i64::MAX).is_none());
    }

    /// Two steps inside the bound can meet past it, and the meet is refused.
    ///
    /// The half of the bound that is easy to lose: `multiple_of` refuses a step
    /// too large, but a caller writing two steps that are each far inside it --
    /// 64 and 81 -- asks for the period they share, 5,184, which is not. There
    /// is no set to substitute. A table built at the largest period this holds
    /// describes the multiples of something else, and handing that back as the
    /// answer is the wrong verdict this representation exists to avoid; so the
    /// operation refuses, exactly as the automaton components do, and the
    /// relation above stays undecided.
    #[test]
    fn two_steps_meeting_past_the_period_bound_are_refused() {
        for (a, b) in [(64, 81), (4093, 4096), (3, MAX_PERIOD), (63, 65)] {
            let (left, right) = (step(a), step(b));
            let shared = lcm(a, b);
            let past = shared > MAX_PERIOD;
            assert_eq!(
                left.intersect(&right).is_none(),
                past,
                "the meet of {a} and {b}, which share {shared}"
            );
            assert_eq!(
                left.union(&right).is_none(),
                past,
                "the join of {a} and {b}, which share {shared}"
            );
        }
    }

    /// A meet inside the bound still answers, which is what the refusal above
    /// must not cost: 63 and 64 share 4,032, and their multiples are decided.
    #[test]
    fn two_steps_meeting_inside_the_period_bound_still_answer() {
        let met = and(&step(63), &step(64));
        assert!(met.holds(0) && met.holds(4032) && met.holds(-4032));
        assert!(!met.holds(63) && !met.holds(64) && !met.holds(4031));
        assert!(!met.is_empty());
    }

    /// A step that cancels leaves no step behind.
    ///
    /// `a | !a` is the integers however `a` was written, and carrying `a`'s
    /// period into the answer would leave two spellings of one set. Since two
    /// spellings are two values, every step that cancels would put another
    /// copy of the integers in a table of guards, and two such sets built from
    /// *different* steps would never meet. Dropping a step nothing uses keeps
    /// them at a period of one, which is one value.
    #[test]
    fn a_step_that_cancels_is_not_carried() {
        for n in [2, 64, 81, MAX_PERIOD] {
            let set = step(n);
            let whole = or(&set, &set.complement());
            assert_eq!(whole, IntSet::all(), "the join over {n}");
            assert_eq!(whole.modulus, 1, "the join over {n} keeps a period");
            let nothing = and(&set, &set.complement());
            assert_eq!(nothing, IntSet::empty(), "the meet over {n}");
            assert_eq!(nothing.modulus, 1, "the meet over {n} keeps a period");
        }
        // And so two of them, from steps whose periods meet past the bound,
        // are still one set rather than a pair equality cannot align.
        let from_wide = or(&step(64), &step(64).complement());
        let from_tall = or(&step(81), &step(81).complement());
        assert_eq!(from_wide, from_tall);
    }

    /// The two number-theoretic helpers, driven directly: the periods meet at
    /// their least common multiple, and every other answer would either miss
    /// integers or carry classes that cannot occur.
    #[test]
    fn the_periods_meet_at_their_least_common_multiple() {
        assert_eq!(gcd(12, 18), 6);
        assert_eq!(gcd(7, 1), 1);
        assert_eq!(gcd(5, 5), 5);
        assert_eq!(lcm(4, 6), 12);
        assert_eq!(lcm(3, 3), 3);
        assert_eq!(lcm(1, 7), 7);
        // Coprime periods multiply, which is the case that grows fastest.
        assert_eq!(lcm(3, 5), 15);
    }

    /// Two spellings of one set are one set in two places.
    ///
    /// Where equality and the order part, said once so a reader who expects
    /// them to agree finds the reason rather than a surprise: the multiples of
    /// two and the same set written with a period of four are equal and take
    /// two positions. The order pays for being total, and it pays in the one
    /// place nothing is charged -- a table of guards holds a second row, and no
    /// question about the integers is answered differently.
    ///
    /// A pair whose periods meet past the bound is the other side of the same
    /// split: the order still answers, because it reads no period, while
    /// equality declines to call them one set.
    #[test]
    fn two_spellings_of_one_set_are_one_set_in_two_places() {
        let evens = IntSet::multiple_of(2).expect("a small step");
        let at_four = IntSet {
            modulus: 4,
            classes: evens
                .table_at(4)
                .expect("a period inside the bound")
                .into_owned(),
        };
        assert!(same(&evens, &at_four), "the two hold the same integers");
        assert_eq!(evens, at_four);
        assert_ne!(evens.cmp(&at_four), core::cmp::Ordering::Equal);

        let coarse = IntSet::multiple_of(MAX_PERIOD - 3).expect("a step within the bound");
        let fine = IntSet::multiple_of(MAX_PERIOD - 5).expect("a step within the bound");
        assert!(
            coarse.union(&fine).is_none(),
            "the periods meet past the bound"
        );
        assert_ne!(coarse, fine);
        assert_ne!(coarse.cmp(&fine), core::cmp::Ordering::Equal);
        assert_eq!(coarse, coarse.clone());
        assert_eq!(fine, fine.clone());
    }

    /// No three sets are ordered in a cycle.
    ///
    /// The witness a sort of record atoms found, pinned so no seed is asked to
    /// find it twice: read at the period each pair shares, the integers come
    /// before `{-2}`, `{-2}` comes before the multiples of three, and the
    /// integers come *after* them. Nothing is wrong with any one of the three
    /// readings; what is wrong is reading a pair, and an order on the tables
    /// has no pair to read.
    #[test]
    fn no_three_sets_are_ordered_in_a_cycle() {
        let corpus = [
            IntSet::all(),
            IntSet::empty(),
            IntSet::just(-2),
            IntSet::between(Some(0), None),
            step(2),
            step(3),
            step(4),
            step(3).complement(),
        ];
        for x in &corpus {
            for y in &corpus {
                for z in &corpus {
                    assert!(
                        !(x < y && y < z) || x < z,
                        "the order puts {x:?}, {y:?} and {z:?} in a cycle"
                    );
                }
            }
        }
    }
}
