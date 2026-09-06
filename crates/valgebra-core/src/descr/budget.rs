//! What a build is allowed to spend.
//!
//! The bounds on a descriptor -- `MAX_LINES`, `MAX_ATOMS`, `MAX_STATES` -- say
//! how large a result may be. They say nothing about the work spent reaching
//! one, and the two are not the same quantity: a union of four records nested
//! three deep, minus a union of its siblings, spends a third of a second and
//! then refuses because the result is too wide. A caller cannot buy safety by
//! accepting a smaller answer, because refusing costs what succeeding costs.
//!
//! This is the missing bound. A build runs [`under`] an allowance, every step
//! that multiplies charges one unit, and a step that finds the allowance spent
//! refuses. Refusing early is what makes the cost of asking bounded: the answer
//! is `None`, which every caller of a descriptor operation already handles,
//! and the caller keeps whatever it knew before it asked.
//!
//! **It is not semantic state.** Outside a build the allowance is unlimited and
//! nothing is charged, which is what leaves the laws measuring the algebra
//! rather than the meter. Inside one it can only turn an answer into a refusal;
//! it never turns one answer into a different answer. What it *can* change is
//! which of two equal spellings a build lands on -- a union of lines that cannot
//! be complemented within the allowance is carried as its own negation instead,
//! which is the same set written differently -- so a build is compared by the
//! values it admits, never by the shape it took.
//!
//! The allowance is per thread, and a build restores the one it found. Two
//! threads building at once each spend their own, which is the only sharing
//! rule a budget needs: nothing here crosses a thread.

use core::cell::Cell;

thread_local! {
    /// The units this thread's innermost build has left, or [`u64::MAX`] where
    /// no build is running -- an allowance no build can exhaust, which is how
    /// "unbudgeted" is spelled without a second state to test for.
    static LEFT: Cell<u64> = const { Cell::new(u64::MAX) };
}

/// Charge one unit, and say whether it was there to charge.
///
/// A step that multiplies -- a product of lines, a meet of two descriptors, a
/// guard recursing into the descriptor behind it -- calls this and refuses on
/// `false`. A step that is linear in what it already holds does not, because
/// bounding those bounds nothing the bounds above do not.
pub(crate) fn spend() -> bool {
    LEFT.with(|left| {
        let Some(rest) = left.get().checked_sub(1) else {
            return false;
        };
        left.set(rest);
        true
    })
}

/// The allowance restored when a build ends, however it ends.
struct Restore(u64);

impl Drop for Restore {
    fn drop(&mut self) {
        LEFT.with(|left| left.set(self.0));
    }
}

/// Run `build` with `units` of work, and restore the caller's allowance after.
///
/// Nests: a build inside a build gets its own allowance and gives back what it
/// found, so an inner one cannot spend an outer one's. That is deliberate --
/// the two are separate questions, and an inner build that refuses is an answer
/// the outer one goes on without.
pub fn under<T>(units: u64, build: impl FnOnce() -> T) -> T {
    let restore = LEFT.with(|left| Restore(left.replace(units)));
    let answer = build();
    drop(restore);
    answer
}

#[cfg(test)]
mod tests {
    use super::{spend, under};

    /// The allowance is what it was given, and refuses once it is gone.
    #[test]
    fn a_build_spends_what_it_was_given_and_no_more() {
        under(3, || {
            assert!(spend() && spend() && spend());
            assert!(!spend(), "the fourth unit was not there");
            assert!(!spend(), "and it stays spent");
        });
    }

    /// Outside a build nothing is charged, so the laws are not measuring this.
    #[test]
    fn nothing_is_charged_outside_a_build() {
        for _ in 0..1_000 {
            assert!(spend());
        }
    }

    /// A build gives back the allowance it found, so a spent inner build does
    /// not spend the outer one.
    #[test]
    fn a_build_restores_the_allowance_it_found() {
        under(2, || {
            under(1, || {
                assert!(spend());
                assert!(!spend());
            });
            assert!(spend() && spend(), "the outer allowance is untouched");
            assert!(!spend());
        });
    }

    /// And gives it back when the build unwinds rather than returns.
    #[test]
    fn a_build_restores_the_allowance_through_a_panic() {
        under(2, || {
            let panicked = std::panic::catch_unwind(|| {
                under(1, || {
                    assert!(spend());
                    panic!("the build gave up");
                });
            });
            assert!(panicked.is_err());
            assert!(spend() && spend());
            assert!(!spend());
        });
    }
}
