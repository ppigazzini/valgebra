use super::{MAX_LINES, SetLattice};
use crate::descr::integers::IntSet;
use proptest::prelude::*;

/// The member sets a law is checked over.
///
/// Small sets of small integers, plus the empty one -- which carries more
/// weight here than anywhere else, being the value every powerset holds and
/// the reason `set[nothing]` is not `nothing`.
const SETS: [&[i64]; 8] = [&[], &[0], &[1], &[2], &[0, 1], &[1, 2], &[0, 1, 2], &[7]];

fn same(a: &SetLattice<IntSet>, b: &SetLattice<IntSet>) -> bool {
    SETS.iter()
        .all(|members| a.holds(members) == b.holds(members))
}

/// Lattices over the integer sets whose own laws are already held.
fn lattice() -> impl Strategy<Value = SetLattice<IntSet>> {
    let leaf = prop_oneof![
        Just(SetLattice::empty()),
        Just(SetLattice::all()),
        Just(SetLattice::of(IntSet::empty())),
        (-2i64..=2).prop_map(|n| SetLattice::of(IntSet::just(n))),
        (-2i64..=2).prop_map(|lo| SetLattice::of(IntSet::between(Some(lo), None))),
        (1i64..=3)
            .prop_map(|step| { SetLattice::of(IntSet::multiple_of(step).expect("a small step")) }),
    ];
    leaf.prop_recursive(3, 12, 2, |inner| {
        prop_oneof![
            (inner.clone(), inner.clone())
                .prop_map(|(a, b)| a.union(&b).unwrap_or_else(SetLattice::all)),
            (inner.clone(), inner.clone())
                .prop_map(|(a, b)| a.intersect(&b).unwrap_or_else(SetLattice::empty)),
            inner.prop_map(|a| a.complement()),
        ]
    })
}

proptest! {
    // A bounded shrink, so a broken invariant cannot turn a caught mutation
    // into a run that outlasts a sweep.
    #![proptest_config(ProptestConfig {
        max_shrink_time: 2_000,
        ..ProptestConfig::default()
    })]

    /// The Boolean algebra, checked against the sets rather than by equality
    /// of the forms, which a union of lines does not make canonical.
    #[test]
    fn the_lattice_laws_hold_of_the_sets(a in lattice(), b in lattice(), c in lattice()) {
        let (join, meet) = (
            |x: &SetLattice<IntSet>, y: &SetLattice<IntSet>| x.union(y),
            |x: &SetLattice<IntSet>, y: &SetLattice<IntSet>| x.intersect(y),
        );
        if let (Some(ab), Some(ba)) = (join(&a, &b), join(&b, &a)) {
            prop_assert!(same(&ab, &ba), "join commutes");
        }
        if let (Some(ab), Some(ba)) = (meet(&a, &b), meet(&b, &a)) {
            prop_assert!(same(&ab, &ba), "meet commutes");
        }
        if let (Some(bc), Some(ab)) = (join(&b, &c), join(&a, &b))
            && let (Some(left), Some(right)) = (join(&a, &bc), join(&ab, &c))
        {
            prop_assert!(same(&left, &right), "join associates");
        }
        if let (Some(bc), Some(ac)) = (join(&b, &c), meet(&a, &c))
            && let (Some(ab), Some(left)) = (meet(&a, &b), meet(&a, &bc))
            && let Some(right) = ab.union(&ac)
        {
            prop_assert!(same(&left, &right), "meet distributes over join");
        }
    }

    /// The complement laws, and De Morgan both ways.
    #[test]
    fn the_complement_laws_hold_of_the_sets(a in lattice(), b in lattice()) {
        let not_a = a.complement();
        {
            let not_a = &not_a;
            if let Some(met) = a.intersect(not_a) {
                prop_assert!(met.is_empty(), "a set is in one of the two");
            }
            if let Some(joined) = a.union(not_a) {
                prop_assert!(same(&joined, &SetLattice::all()), "and in one of them");
            }
            prop_assert!(same(&not_a.complement(), &a), "twice is nothing");
        }
        {
            let (not_a, not_b) = (&not_a, &b.complement());
            if let (Some(joined), Some(met)) = (a.union(&b), not_a.intersect(not_b)) {
                prop_assert!(same(&joined.complement(), &met), "de Morgan one way");
            }
            if let (Some(met), Some(joined)) = (a.intersect(&b), not_a.union(not_b)) {
                prop_assert!(same(&met.complement(), &joined), "and the other");
            }
        }
    }

    /// Emptiness is a decision about the sets, not about the form.
    #[test]
    fn emptiness_agrees_with_the_sets(a in lattice()) {
        if a.is_empty() {
            prop_assert!(
                SETS.iter().all(|members| !a.holds(members)),
                "an empty lattice holds no set"
            );
        }
    }

    /// A meet of two powersets is the powerset of the meet, which is the one
    /// operation the kind is closed under.
    #[test]
    fn a_meet_of_powersets_is_the_powerset_of_the_meet(a in -2i64..=2, b in -2i64..=2) {
        let (x, y) = (IntSet::between(Some(a), None), IntSet::between(None, Some(b)));
        let met = SetLattice::of(x.clone())
            .intersect(&SetLattice::of(y.clone()))
            .expect("two small powersets");
        let meet = x.intersect(&y).expect("two bounds share a period of one");
        prop_assert!(same(&met, &SetLattice::of(meet)));
    }
}

/// Two powersets over disjoint elements meet in the empty set alone.
///
/// The distinction the kind exists to make: `set[int] & set[str]` is not
/// empty and is not `set[int]` either -- it is `set[nothing]`, whose one
/// value is the empty set.
#[test]
fn disjoint_powersets_meet_in_the_empty_set_alone() {
    let evens = IntSet::multiple_of(2).expect("a small step");
    let odds = evens.complement();
    let met = SetLattice::of(evens)
        .intersect(&SetLattice::of(odds))
        .expect("two small powersets");

    assert!(!met.is_empty(), "the empty set is a member of both");
    assert!(met.holds(&[]));
    assert!(!met.holds(&[2]) && !met.holds(&[3]));
    assert!(same(&met, &SetLattice::of(IntSet::empty())));
}

/// The powerset of nothing is not nothing: it holds the empty set.
#[test]
fn the_powerset_of_nothing_holds_the_empty_set() {
    let none = SetLattice::of(IntSet::empty());

    assert!(!none.is_empty());
    assert!(none.holds(&[]));
    assert!(!none.holds(&[0]));
    assert!(SetLattice::<IntSet>::empty().is_empty());
    assert!(!SetLattice::<IntSet>::empty().holds(&[]));
}

/// A line is empty exactly when a subtracted powerset covers it, which is
/// the rule the whole kind rests on.
#[test]
fn a_line_is_empty_when_a_subtraction_covers_it() {
    let small = SetLattice::of(IntSet::just(1));
    let wide = SetLattice::of(IntSet::between(Some(0), Some(9)));

    // `P({1}) ∧ ¬P(0..=9)` is empty, because every subset of `{1}` is a
    // subset of `0..=9`.
    let covered = small
        .intersect(&wide.complement())
        .expect("two small lattices");
    assert!(covered.is_empty());
    // And it is the empty lattice as a value, not a lattice of one line
    // that happens to hold nothing: the line is dropped where the union is
    // put in order, so equality reads one form for one set of sets.
    assert_eq!(covered, SetLattice::empty());

    // The other way round it is not: `{0}` is a subset of `0..=9` and not
    // of `{1}`.
    let escaping = wide
        .intersect(&small.complement())
        .expect("two small lattices");
    assert!(!escaping.is_empty());
    assert!(escaping.holds(&[0]));
    assert!(
        !escaping.holds(&[]),
        "the empty set is a subset of every set"
    );
}

/// A union past the bound refuses rather than holding a form it cannot.
#[test]
fn a_union_past_the_bound_refuses() {
    let mut wide = SetLattice::of(IntSet::just(0));
    for n in 1..i64::try_from(MAX_LINES).unwrap_or(i64::MAX) {
        wide = wide
            .union(&SetLattice::of(IntSet::just(n)))
            .expect("inside the bound");
    }
    assert!(wide.union(&SetLattice::of(IntSet::just(-1))).is_none());
}
