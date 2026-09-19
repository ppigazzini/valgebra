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
//! nothing is charged. Inside one it can only turn an answer into a refusal; it
//! never turns one answer into a different answer. What it *can* change is
//! which of two equal spellings a build lands on -- a union of lines that cannot
//! be complemented within the allowance is carried as its own negation instead,
//! which is the same set written differently -- so a build is compared by the
//! values it admits, never by the shape it took.
//!
//! **Every reader of a descriptor operation is a build**, including the laws.
//! The operations carry no ceiling of their own: a union's width bound is a
//! bound on the *result*, read once the parts holding nothing and the repeats
//! are gone, so it refuses a wide answer and never a long search for a narrow
//! one. This is what says how long a search may be. A reader without one costs
//! whatever its inputs ask, which for a drawn input is a cost that is a
//! property of the draw -- so the laws arm `law`, and `decision` builds its
//! difference under the same figure.
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

/// An allowance, for as long as this is held.
///
/// The build is the guard's scope, and it ends where the guard drops --
/// including where the scope unwinds, which is what makes a panicking build
/// give back what it found rather than leaving the next one short.
#[must_use = "the allowance lasts as long as the guard, so a dropped one arms nothing"]
pub(crate) struct Allowance(u64);

impl Drop for Allowance {
    fn drop(&mut self) {
        LEFT.with(|left| left.set(self.0));
    }
}

/// Arm `units` of work until the returned guard is dropped.
///
/// [`under`] is this with the build written as a closure, and the closure is
/// the better spelling wherever the build is an expression. It is not always
/// available: a property-test case leaves its body through a `return` the
/// macro writes, which a closure would swallow, so the case arms the guard
/// instead and the case *is* the build.
pub(crate) fn armed(units: u64) -> Allowance {
    LEFT.with(|left| Allowance(left.replace(units)))
}

/// Arm what a law's case may spend, which is what a relation's difference may.
///
/// **A bounded shrink is the other bound and does not stand in for this one.**
/// A law block's `max_shrink_time` bounds what a *failing* case costs, so a
/// broken invariant cannot outlast a sweep. This bounds what a *passing* case
/// costs. The two come apart under a mutation that deletes a pruning shortcut:
/// every case grows dear and none of them breaks, so there is no failure to
/// shrink, and a mutant the sweep cannot finish returns no verdict at all --
/// which `scripts/mutation_gate.py` reads as a rig fault rather than as a
/// survivor.
///
/// Arm it in a law that reads a refusal as a skip, which is every law over a
/// drawn descriptor or a drawn lattice: the allowance can only add to the draws
/// such a law skips. A law asserting that its operations *succeed* claims
/// something about a fragment rather than about a build, and is not armed.
#[cfg(test)]
pub(crate) fn law() -> Allowance {
    armed(crate::descr::lower::WORK)
}

/// Run `build` with `units` of work, and restore the caller's allowance after.
///
/// Nests: a build inside a build gets its own allowance and gives back what it
/// found, so an inner one cannot spend an outer one's. That is deliberate --
/// the two are separate questions, and an inner build that refuses is an answer
/// the outer one goes on without.
pub fn under<T>(units: u64, build: impl FnOnce() -> T) -> T {
    let _allowance = armed(units);
    build()
}

#[cfg(test)]
mod tests {
    use super::{armed, spend, under};

    /// The allowance is what it was given, and refuses once it is gone.
    #[test]
    fn a_build_spends_what_it_was_given_and_no_more() {
        under(3, || {
            assert!(spend() && spend() && spend());
            assert!(!spend(), "the fourth unit was not there");
            assert!(!spend(), "and it stays spent");
        });
    }

    /// A guard is the same build with its scope for a closure, and it ends
    /// where the scope does.
    ///
    /// The two spellings are one mechanism -- [`under`] is written in terms of
    /// this one -- so what this pins is the end: an allowance that outlived its
    /// scope would charge the next build in the thread, and one that never
    /// started would charge nothing.
    #[test]
    fn an_armed_allowance_ends_with_its_scope() {
        {
            let _allowance = armed(2);
            assert!(spend() && spend());
            assert!(!spend(), "the third unit was not there");
        }
        for _ in 0..1_000 {
            assert!(spend(), "and the scope gave the caller's allowance back");
        }
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
