//! What a build is allowed to spend.
//!
//! The bounds on a descriptor -- `MAX_LINES`, `MAX_ATOMS`, `MAX_STATES` -- say
//! how large a result may be. They say nothing about the work spent reaching
//! one, and the two are not the same quantity: a union of four records nested
//! three deep, minus a union of its siblings, builds the whole difference and
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
#[must_use = "a charge that ran out is the refusal; a caller that ignores it builds past the allowance"]
pub(crate) fn spend() -> bool {
    LEFT.with(|left| {
        let Some(rest) = left.get().checked_sub(1) else {
            return false;
        };
        left.set(rest);
        true
    })
}

/// The entries of a step's table one unit of the allowance reads: a residue, or
/// a span one holds.
///
/// A table is the one leaf a build holds whose size no schema node shows: a
/// `MultipleOf(4096)` holds 4,096 lists, and its period times its spans is what
/// each operation over it reads and what an automaton's edge keeps. Charged
/// nothing, one unit read all of that, so eight thousand of them could keep
/// gigabytes, past which an allocation ends the process. At this figure a build
/// reads about eight million entries, and a table under a thousand of them, which
/// is every step a schema writes below 1,024 with a few bounds, spends what it
/// spent uncharged.
pub(crate) const TABLE_ENTRIES: usize = 1024;

/// Charge what an operation over a step's table reads, `entries` being its
/// residues and spans, and say whether it was there to charge.
///
/// A table under [`TABLE_ENTRIES`] charges nothing: its operation is one the
/// step around it has paid for. Past the allowance the charge spends what is
/// left, so the operation after this one refuses as well.
#[must_use = "a charge that ran out is the refusal; a caller that ignores it builds past the allowance"]
pub(crate) fn spend_on_a_table(entries: usize) -> bool {
    let units = u64::try_from(entries / TABLE_ENTRIES).unwrap_or(u64::MAX);
    LEFT.with(|left| {
        let rest = left.get().checked_sub(units);
        left.set(rest.unwrap_or(0));
        rest.is_some()
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
///
/// **One allowance shared by every reading of a query is not the bound it
/// looks like.** A query asks a set reading of each goal its rules decline,
/// so a pool armed once per query, every reading spending from it, would bound
/// what a query that declines spends. Measured over the record families, the
/// queries they answer spend up to six hundred thousand units on their
/// readings, so a pool small enough to cut the queries that decline cuts
/// answers with them: sixteen readings' worth lost a `subset` on three of four
/// seeds. What a unit costs is the bound worth holding, and `polar::product`
/// holds it.
pub fn under<T>(units: u64, build: impl FnOnce() -> T) -> T {
    let _allowance = armed(units);
    build()
}

#[cfg(test)]
mod tests {
    use super::{TABLE_ENTRIES, armed, spend, spend_on_a_table, under};

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

    /// A small table charges nothing, a large one what it reads, and one past
    /// what is left spends the rest.
    #[test]
    fn a_table_is_charged_by_what_it_reads() {
        under(2, || {
            assert!(spend_on_a_table(TABLE_ENTRIES - 1));
            assert!(spend_on_a_table(TABLE_ENTRIES));
            assert!(spend(), "one unit went to the table, and one is left");
            assert!(!spend());
        });
        under(5, || {
            assert!(!spend_on_a_table(6 * TABLE_ENTRIES));
            assert!(!spend(), "a charge past the allowance spends it");
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
