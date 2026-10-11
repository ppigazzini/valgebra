use std::cell::Cell;

use super::{Summand, product};

thread_local! {
    /// The longest list [`Summand::compacted`] has been handed on this thread.
    static LONGEST: Cell<usize> = const { Cell::new(0) };
    /// How many lists it has been handed on this thread.
    static ASKED: Cell<usize> = const { Cell::new(0) };
}

/// A summand that is its number, and says what it is asked to compact: the
/// meet of two is the larger, none holds nothing, and a list of them is
/// compacted by putting it in order and dropping the repeats.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Counted(u8);

impl Summand for Counted {
    type Within = ();
    type Complement = Vec<Counted>;
    const MAX: usize = 4;

    fn top((): ()) -> Counted {
        Counted(0)
    }

    fn meet(&self, other: &Counted) -> Option<Counted> {
        Some(self.clone().max(other.clone()))
    }

    fn complement(&self, (): ()) -> Vec<Counted> {
        Vec::new()
    }

    fn compacted(mut summands: Vec<Counted>) -> Option<Vec<Counted>> {
        LONGEST.with(|longest| longest.set(longest.get().max(summands.len())));
        ASKED.with(|asked| asked.set(asked.get() + 1));
        summands.sort();
        summands.dedup();
        Some(summands)
    }
}

fn counted(values: &[u8]) -> Vec<Counted> {
    values.iter().copied().map(Counted).collect()
}

/// The pairs a product builds are compacted when they reach the bound, so the
/// list it holds stands at the bound at most, however many pairs it builds.
///
/// Nine pairs whose meets are three values: wider than the bound as built, and
/// within it once the repeats go. Compacting at every pair below the bound
/// instead stops compacting at the bound and hands the end all nine.
#[test]
fn a_product_holds_no_more_pairs_than_its_bound() {
    LONGEST.with(|longest| longest.set(0));
    let met = product(&counted(&[0, 1, 2]), &counted(&[0, 1, 2]));
    assert_eq!(met, Some(counted(&[0, 1, 2])));
    assert_eq!(LONGEST.with(Cell::get), Counted::MAX);
}

/// A product whose pairs never reach the bound is compacted once, at the end:
/// compacting is what the bound costs, and a narrow product pays none of it.
#[test]
fn a_product_under_its_bound_is_compacted_once() {
    ASKED.with(|asked| asked.set(0));
    let met = product(&counted(&[1, 2]), &counted(&[1, 2]));
    assert_eq!(met, Some(counted(&[1, 2])));
    assert_eq!(ASKED.with(Cell::get), 1);
}
