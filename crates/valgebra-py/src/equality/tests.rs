use super::*;

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
