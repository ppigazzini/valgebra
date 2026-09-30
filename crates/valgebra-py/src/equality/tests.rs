use std::cell::Cell;

use super::*;

thread_local! {
    /// How many candidates `same_multiset` has looked at on this thread.
    static SCANNED: Cell<usize> = const { Cell::new(0) };
}

/// The recorder `same_multiset` reports each candidate it looks at to.
pub(super) fn scanned() {
    SCANNED.with(|count| count.set(count.get() + 1));
}

/// [`same_multiset`] under `==`, which cannot raise over integers.
fn matched(items: &[u8], others: &[u8]) -> bool {
    same_multiset(items, others, |a, b| Ok(a == b)).expect("an integer comparison does not raise")
}

#[test]
fn a_multiset_matches_a_permutation_and_refuses_a_repeat() {
    assert!(matched(&[1, 2, 3], &[3, 1, 2]));
    assert!(!matched(&[1, 1, 2], &[1, 2, 2]));
    assert!(!matched(&[1, 2], &[1, 2, 3]));
    assert!(matched(&[], &[]));
}

#[test]
fn every_stops_at_the_first_item_that_fails() {
    let mut asked = 0;
    let all = every([1, 2, 3], |n| {
        asked += 1;
        Ok(n != 2)
    });
    assert!(!all.expect("nothing raises"));
    assert_eq!(asked, 2, "the item after a failure is not asked");
    assert!(every([1, 2, 3], |n| Ok(n > 0)).expect("nothing raises"));
    assert!(every::<u8>([], |_| Ok(false)).expect("an empty list holds"));
}

/// Every sequence of `len` values drawn from `0..3`, as members tagged with
/// their position.
fn sequences(len: u32) -> Vec<Vec<(usize, u8)>> {
    (0..3_u32.pow(len))
        .map(|mut code| {
            (0..len as usize)
                .map(|at| {
                    let value = u8::try_from(code % 3).expect("below three");
                    code /= 3;
                    (at, value)
                })
                .collect()
        })
        .collect()
}

/// The matching the search must be: each member takes the first untaken equal
/// member of the other list, scanning from the front, and the comparisons it
/// asks are recorded as the positions they pair.
fn from_the_front(items: &[(usize, u8)], others: &[(usize, u8)]) -> (bool, Vec<(usize, usize)>) {
    let mut asked = Vec::new();
    let mut taken = vec![false; others.len()];
    for item in items {
        let found = others.iter().position(|other| {
            if taken[other.0] {
                return false;
            }
            asked.push((item.0, other.0));
            item.1 == other.1
        });
        match found {
            Some(at) => taken[at] = true,
            None => return (false, asked),
        }
    }
    (true, asked)
}

#[test]
fn the_search_asks_what_a_search_from_the_front_asks() {
    for len in 0..=4 {
        for items in sequences(len) {
            for others in sequences(len) {
                let mut asked = Vec::new();
                let matched = same_multiset(&items, &others, |item, other| {
                    asked.push((item.0, other.0));
                    Ok(item.1 == other.1)
                })
                .expect("nothing raises");
                assert_eq!(
                    (matched, asked),
                    from_the_front(&items, &others),
                    "{items:?} against {others:?}"
                );
            }
        }
    }
}

/// Lists that match in place are searched in one pass: each member looks at
/// its match and nothing before it, where a search from the front looks past
/// every member already taken.
#[test]
fn lists_that_match_in_place_are_one_pass() {
    let items: Vec<u32> = (0..50).collect();
    SCANNED.with(|count| count.set(0));
    assert!(same_multiset(&items, &items, |a, b| Ok(a == b)).expect("nothing raises"));
    assert_eq!(SCANNED.with(Cell::get), items.len());
}
