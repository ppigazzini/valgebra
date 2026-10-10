//! The Boolean-algebra laws, stated once and asked of every representation
//! that claims them.
//!
//! Each representation is a set closed under the three operations, and the
//! laws are what say so. One body states them, so every law reaches every
//! representation and none carries a subset of its own. What a representation
//! supplies is its operations and its reading of "the same set": equality
//! where its forms are canonical, agreement over a universe of values where a
//! bound or a polarity gives one set two spellings.
//!
//! An operation that refuses -- past a width bound, or past the build's
//! allowance -- answers `None`, and a law with a term that did not build is
//! not asked: the bound may turn an answer into a refusal and never one answer
//! into another, and that is a claim about the bound, which its own rows hold.
//! A representation whose operations cannot refuse on the corpus it draws says
//! so by refusing loudly in its own closures instead.

use proptest::prop_assert;
use proptest::test_runner::TestCaseError;

/// One representation's operations, as the laws ask them.
pub(crate) struct Algebra<T> {
    /// The values in either.
    pub(crate) join: fn(&T, &T) -> Option<T>,
    /// The values in both.
    pub(crate) meet: fn(&T, &T) -> Option<T>,
    /// The values in neither, which every representation keeps total.
    pub(crate) complement: fn(&T) -> T,
    /// Whether two forms hold the same values.
    pub(crate) same: fn(&T, &T) -> bool,
    /// Whether a form holds no value.
    pub(crate) holds_nothing: fn(&T) -> bool,
    /// Whether a form holds every value of its universe.
    pub(crate) holds_everything: fn(&T) -> bool,
}

impl<T> Algebra<T> {
    /// The lattice laws over three drawn sets: both operations are idempotent,
    /// commute and associate, each absorbs the other, and each distributes
    /// over the other.
    pub(crate) fn lattice_laws(&self, a: &T, b: &T, c: &T) -> Result<(), TestCaseError> {
        let (join, meet, same) = (self.join, self.meet, self.same);
        for (op, name) in [(join, "join"), (meet, "meet")] {
            if let Some(aa) = op(a, a) {
                prop_assert!(same(&aa, a), "{} is idempotent", name);
            }
            if let (Some(ab), Some(ba)) = (op(a, b), op(b, a)) {
                prop_assert!(same(&ab, &ba), "{} commutes", name);
            }
            if let (Some(ab), Some(bc)) = (op(a, b), op(b, c))
                && let (Some(left), Some(right)) = (op(&ab, c), op(a, &bc))
            {
                prop_assert!(same(&left, &right), "{} associates", name);
            }
        }
        if let Some(ab) = meet(a, b)
            && let Some(absorbed) = join(a, &ab)
        {
            prop_assert!(same(&absorbed, a), "a join absorbs a meet");
        }
        if let Some(ab) = join(a, b)
            && let Some(absorbed) = meet(a, &ab)
        {
            prop_assert!(same(&absorbed, a), "a meet absorbs a join");
        }
        for (outer, inner, name) in [
            (meet, join, "meet distributes over join"),
            (join, meet, "join distributes over meet"),
        ] {
            if let Some(bc) = inner(b, c)
                && let (Some(ab), Some(ac)) = (outer(a, b), outer(a, c))
                && let (Some(left), Some(right)) = (outer(a, &bc), inner(&ab, &ac))
            {
                prop_assert!(same(&left, &right), "{}", name);
            }
        }
        Ok(())
    }

    /// The complement laws over two drawn sets: a set and its complement share
    /// nothing and cover everything, the complement taken twice is the set,
    /// and De Morgan holds both ways.
    pub(crate) fn complement_laws(&self, a: &T, b: &T) -> Result<(), TestCaseError> {
        let (join, meet, complement, same) = (self.join, self.meet, self.complement, self.same);
        let (not_a, not_b) = (complement(a), complement(b));
        if let Some(met) = meet(a, &not_a) {
            prop_assert!((self.holds_nothing)(&met), "a value is in one of the two");
        }
        if let Some(joined) = join(a, &not_a) {
            prop_assert!((self.holds_everything)(&joined), "and in one of them");
        }
        prop_assert!(same(&complement(&not_a), a), "twice is nothing");
        if let (Some(joined), Some(met)) = (join(a, b), meet(&not_a, &not_b)) {
            prop_assert!(same(&complement(&joined), &met), "de Morgan, one way");
        }
        if let (Some(met), Some(joined)) = (meet(a, b), join(&not_a, &not_b)) {
            prop_assert!(same(&complement(&met), &joined), "and the other");
        }
        Ok(())
    }
}
