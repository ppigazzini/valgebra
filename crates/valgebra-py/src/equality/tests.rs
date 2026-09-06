use super::*;

#[test]
fn a_multiset_matches_a_permutation_and_refuses_a_repeat() {
    assert!(same_multiset(&[1, 2, 3], &[3, 1, 2], |a, b| a == b));
    assert!(!same_multiset(&[1, 1, 2], &[1, 2, 2], |a, b| a == b));
    assert!(!same_multiset(&[1, 2], &[1, 2, 3], |a, b| a == b));
    assert!(same_multiset::<u8>(&[], &[], |a, b| a == b));
}
