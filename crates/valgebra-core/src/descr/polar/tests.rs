use std::cell::Cell;

use super::{Summand, product};

thread_local! {
    /// The longest list [`Summand::admit`] has been handed on this thread.
    static LONGEST: Cell<usize> = const { Cell::new(0) };
    /// The summands of the lists it has been handed, summed, on this thread:
    /// what admitting every pair reads.
    static READ: Cell<usize> = const { Cell::new(0) };
}

/// A summand that is its number, and says what it is asked to admit: the meet
/// of two is the larger, none holds nothing, and a list keeps each number
/// once.
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

    fn admit(kept: &mut Vec<Counted>, summand: Counted) -> Option<()> {
        LONGEST.with(|longest| longest.set(longest.get().max(kept.len())));
        READ.with(|read| read.set(read.get() + kept.len()));
        if !kept.contains(&summand) {
            kept.push(summand);
        }
        Some(())
    }
}

fn counted(values: &[u8]) -> Vec<Counted> {
    values.iter().copied().map(Counted).collect()
}

/// A product admits each pair as it is met, so the list it holds stands at its
/// compacted width at every pair, however many pairs it builds.
///
/// Nine pairs whose meets are three values: wider than the bound as built, and
/// within it once the repeats go, so the list never holds more than three.
/// Building the pairs first and compacting after holds all nine at once.
#[test]
fn a_product_holds_no_more_pairs_than_its_bound() {
    LONGEST.with(|longest| longest.set(0));
    let met = product(&counted(&[0, 1, 2]), &counted(&[0, 1, 2]));
    assert_eq!(met, Some(counted(&[0, 1, 2])));
    assert_eq!(LONGEST.with(Cell::get), 3);
}

/// A unit of the allowance buys one pair and one reading of the list.
///
/// Sixteen pairs at a bound of four whose list stands at the bound from the
/// fourth pair on: each pair reads the list once, at most four summands, so
/// the product reads at most sixteen times four. Compacting the whole list
/// again at every pair at the bound reads it against itself, the width squared
/// a pair.
#[test]
fn a_product_at_its_bound_reads_each_pair_against_the_list_once() {
    READ.with(|read| read.set(0));
    let met = product(&counted(&[0, 1, 2, 3]), &counted(&[0, 1, 2, 3]));
    assert_eq!(met, Some(counted(&[0, 1, 2, 3])));
    assert!(READ.with(Cell::get) <= 16 * Counted::MAX);
}

/// The pair that takes the list past its bound refuses the product, and a
/// repeat at the bound refuses nothing.
#[test]
fn a_pair_past_the_bound_refuses_the_product() {
    assert_eq!(
        product(&counted(&[0, 1, 2, 3]), &counted(&[0, 1, 2, 3, 4])),
        None
    );
    assert_eq!(
        product(&counted(&[0, 1, 2, 3]), &counted(&[0, 1, 2, 3, 3])),
        Some(counted(&[0, 1, 2, 3]))
    );
}

/// A product is the compaction of its pairs: the same summands, each once, in
/// order, whichever way its factors were written.
#[test]
fn a_product_is_the_compaction_of_its_pairs() {
    let left = counted(&[3, 0, 2]);
    let right = counted(&[1, 3, 0]);
    let pairs: Vec<Counted> = left
        .iter()
        .flat_map(|mine| right.iter().filter_map(|theirs| mine.meet(theirs)))
        .collect();
    assert_eq!(product(&left, &right), Counted::compacted(pairs));
    assert_eq!(product(&left, &right), product(&right, &left));
}
