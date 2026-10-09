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
    /// Reached only by a constructor that knows its period is inside the bound
    /// -- a period of one, or a step [`multiple_of`](Self::multiple_of) has
    /// checked; a period derived from another set's goes through
    /// [`from_table`](Self::from_table), which refuses rather than asserting.
    fn build(modulus: i64, of: impl Fn(i64) -> IntervalSet) -> IntSet {
        debug_assert!(modulus >= 1, "a modulus is at least one");
        debug_assert!(
            modulus <= MAX_PERIOD,
            "a period of {modulus} materialises that many classes, past the \
             {MAX_PERIOD} this representation holds"
        );
        IntSet {
            modulus,
            classes: (0..modulus).map(of).collect(),
        }
    }

    /// The set a finished table spells, or `None` where the table is not one
    /// class per residue of a period this representation holds.
    ///
    /// The refusal is the whole point. Clamping the period here would build a
    /// table for a *different* set and hand it back as this one -- the shape of
    /// wrong answer this representation exists to avoid -- and asserting would
    /// turn a composition a caller is entitled to write into a panic on the
    /// debug builds every contributor runs.
    fn from_table(modulus: i64, classes: Vec<IntervalSet>) -> Option<IntSet> {
        ((1..=MAX_PERIOD).contains(&modulus) && i64::try_from(classes.len()).ok() == Some(modulus))
            .then_some(IntSet { modulus, classes })
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
    /// class of zero, which no union of intervals holds. A negative step names
    /// the same multiples as its magnitude, so it is read as its absolute value,
    /// and zero gives the singleton `{0}` -- the only integer that is a multiple
    /// of nothing. `i64::MIN` has a magnitude no `i64` holds, and it is past the
    /// bound like every step that large: its multiples are `0` and `±2**63`, a
    /// set of values `{0}` would leave out.
    ///
    /// The refusal is in the return type rather than in an assertion, because it
    /// is a decision the caller has to make. There is no sound approximation to
    /// substitute: a set that is too wide or too narrow is complemented into one
    /// that is wrong the other way, so a step this cannot hold must stay opaque
    /// in whatever lowers it.
    #[must_use]
    pub fn multiple_of(step: i64) -> Option<IntSet> {
        match step.checked_abs() {
            Some(0) => Some(IntSet::just(0)),
            Some(step) if step <= MAX_PERIOD => Some(IntSet::build(step, |residue| {
                if residue == 0 {
                    IntervalSet::all()
                } else {
                    IntervalSet::empty()
                }
            })),
            // Past the bound, and `i64::MIN`, whose magnitude has no `i64`.
            _ => None,
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
        usize::try_from(residue)
            .ok()
            .and_then(|index| self.classes.get(index))
            .is_some_and(|class| class.holds(value.div_euclid(self.modulus)))
    }

    /// Every integer this set holds where there are at most `cap`, and `None`
    /// where there are more.
    pub(crate) fn values_up_to(&self, cap: usize) -> Option<Vec<i128>> {
        let mut values = Vec::new();
        for (residue, class) in (0_i128..).zip(&self.classes) {
            for k in class.values_up_to(cap)? {
                if values.len() == cap {
                    return None;
                }
                // A period of at most `MAX_PERIOD` times an `i64` with a step
                // of headroom stays far inside an `i128`.
                values.push(residue + i128::from(self.modulus) * k);
            }
        }
        Some(values)
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
        if !(1..=MAX_PERIOD).contains(&modulus) {
            return None;
        }
        // Each new class reads the old class its residue lands in, which a
        // table of one class per residue always has: a residue it lacks refuses
        // rather than reading that class as empty.
        (0..modulus)
            .map(|residue| {
                let old = residue.rem_euclid(self.modulus);
                let step = (residue - old) / self.modulus;
                let class = self.classes.get(usize::try_from(old).ok()?)?;
                Some(class.preimage(step, stride))
            })
            .collect::<Option<Vec<_>>>()
            .map(Cow::Owned)
    }

    /// The period two sets share, where their classes line up, or `None`
    /// where it is past the range of an `i64`.
    fn common(&self, other: &IntSet) -> Option<i64> {
        lcm(self.modulus, other.modulus)
    }

    /// Combine two sets residue by residue, after lifting both to one period,
    /// or `None` where that period is past [`MAX_PERIOD`].
    fn zip(
        &self,
        other: &IntSet,
        op: fn(&IntervalSet, &IntervalSet) -> IntervalSet,
    ) -> Option<IntSet> {
        let modulus = self.common(other)?;
        let (mine, theirs) = (self.table_at(modulus)?, other.table_at(modulus)?);
        // Two tables of one period pair up residue by residue, and the pairing
        // is a table of that period only where both hold one class per residue.
        if mine.len() != theirs.len() {
            return None;
        }
        let combined = IntSet::from_table(
            modulus,
            mine.iter()
                .zip(theirs.iter())
                .map(|(a, b)| op(a, b))
                .collect(),
        )?;
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
        let Some(modulus) = self.common(other) else {
            return false;
        };
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

/// The least common multiple of two positive integers, or `None` where it is
/// past the range of an `i64`.
///
/// No two periods here reach that: each is at most [`MAX_PERIOD`], so their
/// product is far inside the range. `None` is what a broken bound reads as,
/// and the callers refuse on it -- a combination declines, and two sets are
/// read as two -- rather than carrying a modulus no set has.
fn lcm(a: i64, b: i64) -> Option<i64> {
    let divisor = gcd(a, b);
    debug_assert!(divisor > 0, "a modulus is at least one");
    (a / divisor.max(1)).checked_mul(b)
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
mod tests;
