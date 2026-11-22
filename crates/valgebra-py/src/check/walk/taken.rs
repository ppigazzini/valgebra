//! The readings a walk took, noted for the tests that hold a value to the
//! reading chosen for it.
//!
//! A reading chosen for its cost answers as the reading beside it does -- a
//! list read through a snapshot or in place, a mapping by the loop written for
//! its two kinds or by the general one -- so no answer shows which one ran,
//! and a change that sends a value to the other passes every test of answers.
//! Each such site notes itself here behind `cfg(test)`, so the shipped
//! extension carries none of it, and `taken_tests.rs` asks how often a walk
//! took each.

use std::cell::RefCell;

use rustc_hash::FxHashMap;

/// A reading the walk takes in place of another that answers alike.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum Reading {
    /// A list copied into a tuple, whose elements are then read borrowed.
    Snapshot,
}

thread_local! {
    /// How many times this thread's walks took each reading.
    static TAKEN: RefCell<FxHashMap<Reading, usize>> = RefCell::default();
}

/// Note that the walk took `reading`.
pub(super) fn note(reading: Reading) {
    TAKEN.with_borrow_mut(|taken| *taken.entry(reading).or_default() += 1);
}

/// What `walk` answers, and how many times it took `reading`.
#[cfg(feature = "interpreter-tests")]
pub(super) fn counted<R>(reading: Reading, walk: impl FnOnce() -> R) -> (R, usize) {
    let taken = || TAKEN.with_borrow(|taken| taken.get(&reading).copied().unwrap_or(0));
    let before = taken();
    let answer = walk();
    (answer, taken() - before)
}
