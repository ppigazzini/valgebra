use core::cell::Cell;

use crate::{joined_in_rounds, unordered_pairs};

/// A join that concatenates, and counts what it reads, so a test holds both
/// the order the items are joined in and the shape of the rounds.
fn joined_counting(width: u32, read: &Cell<usize>) -> Option<Vec<u32>> {
    let items: Vec<u32> = (0..width).collect();
    joined_in_rounds(
        &items,
        |item| Some(vec![*item]),
        Vec::new,
        |mut first, second| {
            read.set(read.get() + first.len() + second.len());
            first.extend(second);
            Some(first)
        },
    )
}

/// Each item is read once a round, which is what the rounds are for: eight
/// items in three rounds read twenty-four, where folding reads thirty-five.
#[test]
fn joined_in_rounds_reads_an_item_once_a_round() {
    let read = Cell::new(0);
    assert_eq!(joined_counting(8, &read), Some((0..8).collect()));
    assert_eq!(read.get(), 24, "three rounds of eight");
}

/// The items keep their order at every width, across the width where the fold
/// gives way to the rounds, and a list of one or none is no join at all.
#[test]
fn joined_in_rounds_keeps_the_order_at_every_width() {
    for width in 0..=33 {
        let read = Cell::new(0);
        let joined = joined_counting(width, &read);
        assert_eq!(joined, Some((0..width).collect()), "width {width}");
        if width < 2 {
            assert_eq!(read.get(), 0, "width {width} joins nothing");
        }
    }
    let three = Cell::new(0);
    joined_counting(3, &three);
    assert_eq!(three.get(), 5, "two and then three, as folding reads them");
    let four = Cell::new(0);
    joined_counting(4, &four);
    assert_eq!(
        four.get(),
        8,
        "two rounds of four, where folding reads nine"
    );
}

/// A join that refuses refuses the whole, at once.
#[test]
fn joined_in_rounds_stops_at_a_refusal() {
    let asked = Cell::new(0);
    let joined = joined_in_rounds(
        &[1, 2, 3, 4],
        |item: &i32| Some(*item),
        || 0,
        |first: i32, second| {
            asked.set(asked.get() + 1);
            (first != 3).then_some(first + second)
        },
    );
    assert_eq!(joined, None);
    assert_eq!(
        asked.get(),
        2,
        "the second pair refuses, and no round follows"
    );
}

/// Past three, a read that refuses spends no join and reads nothing past it,
/// which is what a union past the node bound relies on to refuse cheaply.
#[test]
fn joined_in_rounds_reads_before_it_joins() {
    let (joins, reads) = (Cell::new(0), Cell::new(0));
    let joined = joined_in_rounds(
        &[1, 2, 3, 4, 5],
        |item: &i32| {
            reads.set(reads.get() + 1);
            (*item != 4).then_some(*item)
        },
        || 0,
        |first, second| {
            joins.set(joins.get() + 1);
            Some(first + second)
        },
    );
    assert_eq!(joined, None);
    assert_eq!((joins.get(), reads.get()), (0, 4));
}

/// Every unordered pair of *distinct* items, each once.
///
/// The contract both callers rest on: each asks a question of two items,
/// and a pair of an item with itself is not that question -- two labels
/// that are one dict key, two members that share no value. An off-by-one
/// in the slice taken yields each item beside itself, which neither caller
/// distinguishes, since both questions are false on a self-pair.
#[test]
fn unordered_pairs_yields_each_distinct_pair_once() {
    let items = ["a", "b", "c"];
    let pairs: Vec<(&&str, &&str)> = unordered_pairs(&items).collect();
    assert_eq!(
        pairs,
        vec![(&"a", &"b"), (&"a", &"c"), (&"b", &"c")],
        "each pair once, in order, and never an item with itself"
    );
    for (left, right) in &pairs {
        assert_ne!(left, right, "a pair is of two distinct items");
    }
    // The degenerate sizes, where an off-by-one shows first.
    assert_eq!(unordered_pairs::<&str>(&[]).count(), 0, "no items, no pair");
    assert_eq!(unordered_pairs(&items[..1]).count(), 0, "one item, no pair");
    assert_eq!(unordered_pairs(&items[..2]).count(), 1);
    // n items give n*(n-1)/2 pairs, so nothing is visited twice.
    assert_eq!(unordered_pairs(&[1, 2, 3, 4, 5]).count(), 10);
}
