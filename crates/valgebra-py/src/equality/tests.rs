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

/// [`same_multiset`] under `==`, which cannot raise over integers, with one
/// bucket and with a bucket per value.
fn matched(items: &[u8], others: &[u8]) -> bool {
    let value = |n: &u8| Ok(u64::from(*n));
    let searched = same_multiset(items, others, |a, b| Ok(a == b));
    let bucketed = same_multiset_by(items, others, |a, b| Ok(a == b), value, value);
    let answer = searched.expect("an integer comparison does not raise");
    assert_eq!(bucketed.expect("nor does reading its value"), answer);
    answer
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
    let matched = same_multiset(&items, &items, |a, b| Ok(a == b));
    assert!(matched.expect("nothing raises"));
    assert_eq!(SCANNED.with(Cell::get), items.len());
}

/// A digest that separates the members makes two orders of one set one pass
/// too: each member looks at its own bucket alone. A search from the front
/// over the reversed list looks at the whole untaken rest for every member,
/// which is quadratic in the width.
#[test]
fn a_permutation_is_one_pass_under_a_digest_that_separates_the_members() {
    let items: Vec<u32> = (0..500).collect();
    let reversed: Vec<u32> = items.iter().rev().copied().collect();
    let value = |n: &u32| Ok(u64::from(*n));
    SCANNED.with(|count| count.set(0));
    let matched = same_multiset_by(&items, &reversed, |a, b| Ok(a == b), value, value);
    assert!(matched.expect("nothing raises"));
    assert_eq!(SCANNED.with(Cell::get), items.len());
}

/// A comparison of two tables with no pool, reading constants by slot.
fn comparison<'a>(left: &'a Side<'a>, right: &'a Side<'a>) -> Comparison<'a, 'static> {
    Comparison {
        constants: Constants::BySlot,
        left,
        right,
        renaming: RefCell::new(Renaming::between(left, right)),
        depth: Cell::new(0),
        steps: Cell::new(0),
        searching: true,
    }
}

/// A comparison opens levels up to its depth bound and no further, closes each
/// one it opened, and refuses a level once it has opened its step bound in all.
#[test]
fn a_comparison_holds_its_depth_and_its_steps() {
    let bare = Side {
        definitions: &[],
        pool: &[],
    };
    let cx = comparison(&bare, &bare);
    let levels: Vec<Level<'_>> = (0..MAX_COMPARISON_DEPTH)
        .map(|_| cx.descend().expect("a level under the depth bound opens"))
        .collect();
    assert!(
        cx.descend().is_none(),
        "the level past the depth bound is refused"
    );
    drop(levels);
    assert_eq!(cx.depth.get(), 0, "every level opened is closed");
    cx.steps.set(MAX_COMPARISON_STEPS - 1);
    assert!(cx.descend().is_some(), "the last step opens");
    assert!(cx.descend().is_none(), "and the one past it does not");
}

/// The identity pairs two tables only where they are as long, each definition
/// with the one at its own index and with no other.
#[test]
fn the_identity_pairs_tables_of_one_length_index_for_index() {
    let two = [Schema::Int, Schema::Str];
    let one = [Schema::Int];
    let side = |definitions| Side {
        definitions,
        pool: &[],
    };
    assert!(Renaming::identity(&side(&two), &side(&one)).is_none());
    let mut identity = Renaming::identity(&side(&two), &side(&two)).expect("one length");
    assert!(matches!(identity.pair(1, 1), Paired::Before));
    assert!(matches!(identity.pair(0, 1), Paired::Apart));
    assert!(matches!(identity.pair(1, 0), Paired::Apart));
}
