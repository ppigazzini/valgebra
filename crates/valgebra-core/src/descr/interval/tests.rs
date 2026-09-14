use super::IntervalSet;
use proptest::prelude::*;

/// The integers a law is checked over.
///
/// A window rather than a proof: these sets are infinite, so a law is held
/// by asking every set in the window and by generating only endpoints
/// inside it. An endpoint outside the window would let two sets differ where
/// nothing looks, which is why the generator below is bounded to it.
const WINDOW: core::ops::RangeInclusive<i64> = -12..=12;

/// Whether two sets hold the same integers across the window.
fn same(a: &IntervalSet, b: &IntervalSet) -> bool {
    WINDOW.into_iter().all(|n| a.holds(n) == b.holds(n))
}

/// Sets built from bounded endpoints, so agreement on the window is
/// agreement everywhere: no generated set has a feature outside it.
fn interval_set() -> impl Strategy<Value = IntervalSet> {
    let leaf = prop_oneof![
        Just(IntervalSet::empty()),
        Just(IntervalSet::all()),
        (-8i64..=8).prop_map(IntervalSet::just),
        (-8i64..=8).prop_map(|lo| IntervalSet::between(Some(lo), None)),
        (-8i64..=8).prop_map(|hi| IntervalSet::between(None, Some(hi))),
        (-8i64..=8, -8i64..=8)
            .prop_map(|(a, b)| IntervalSet::between(Some(a.min(b)), Some(a.max(b)))),
    ];
    leaf.prop_recursive(4, 24, 2, |inner| {
        prop_oneof![
            (inner.clone(), inner.clone()).prop_map(|(a, b)| a.union(&b)),
            (inner.clone(), inner.clone()).prop_map(|(a, b)| a.intersect(&b)),
            inner.prop_map(|a| a.complement()),
        ]
    })
}

proptest! {
    // A bounded shrink, so a broken invariant cannot turn a caught mutation
    // into a run that outlasts a sweep: every draw is larger under one, and
    // shrinking a counterexample redraws it thousands of times.
    #![proptest_config(ProptestConfig {
        max_shrink_time: 2_000,
        ..ProptestConfig::default()
    })]

    /// The Boolean algebra, checked against the integers rather than against
    /// the two representations agreeing.
    #[test]
    fn the_lattice_laws_hold_of_the_integers(
        a in interval_set(),
        b in interval_set(),
        c in interval_set(),
    ) {
        prop_assert!(same(&a.union(&b), &b.union(&a)));
        prop_assert!(same(&a.intersect(&b), &b.intersect(&a)));
        prop_assert!(same(&a.union(&b).union(&c), &a.union(&b.union(&c))));
        prop_assert!(same(
            &a.intersect(&b).intersect(&c),
            &a.intersect(&b.intersect(&c))
        ));
        prop_assert!(same(&a.union(&a.intersect(&b)), &a));
        prop_assert!(same(&a.intersect(&a.union(&b)), &a));
        prop_assert!(same(
            &a.intersect(&b.union(&c)),
            &a.intersect(&b).union(&a.intersect(&c))
        ));
    }

    /// The complement laws, and De Morgan both ways.
    #[test]
    fn the_complement_laws_hold_of_the_integers(a in interval_set(), b in interval_set()) {
        prop_assert!(a.intersect(&a.complement()).is_empty());
        prop_assert!(same(&a.union(&a.complement()), &IntervalSet::all()));
        prop_assert!(same(&a.complement().complement(), &a));
        prop_assert!(same(
            &a.union(&b).complement(),
            &a.complement().intersect(&b.complement())
        ));
        prop_assert!(same(
            &a.intersect(&b).complement(),
            &a.complement().union(&b.complement())
        ));
    }

    /// Holding the same integers *is* being equal, which is what the merging
    /// and sorting are for. Without it two spellings of one set would be two
    /// sets, and the descriptor built on this could not decide equality by
    /// comparing representations.
    #[test]
    fn holding_the_same_integers_is_being_equal(a in interval_set(), b in interval_set()) {
        prop_assert_eq!(same(&a, &b), a == b);
    }

    /// Emptiness is a decision: a set is empty exactly when it holds no
    /// integer.
    #[test]
    fn emptiness_agrees_with_the_integers(a in interval_set()) {
        prop_assert_eq!(a.is_empty(), WINDOW.into_iter().all(|n| !a.holds(n)));
    }

    /// The change of variable is exactly that: `k` is in the preimage when
    /// the integer it names is in the set.
    #[test]
    fn a_preimage_holds_the_indices_of_the_integers_it_names(
        a in interval_set(),
        offset in -4i64..=4,
        stride in 1i64..=4,
    ) {
        let indices = a.preimage(offset, stride);
        for k in -6i64..=6 {
            let named = offset + stride * k;
            prop_assert_eq!(indices.holds(k), a.holds(named), "k={} names {}", k, named);
        }
    }
}

/// Adjacent spans are one span, which is what makes the form canonical
/// rather than merely sorted.
#[test]
fn touching_spans_become_one() {
    let low = IntervalSet::between(Some(0), Some(3));
    let high = IntervalSet::between(Some(4), Some(7));
    assert_eq!(low.union(&high), IntervalSet::between(Some(0), Some(7)));
    // A gap of one integer is a gap, and the two stay apart.
    let apart = IntervalSet::between(Some(5), Some(7));
    assert_ne!(low.union(&apart), IntervalSet::between(Some(0), Some(7)));
    assert!(!low.union(&apart).holds(4));
}

/// The ends are where a bound can wrap, so they are driven directly.
#[test]
fn the_ends_of_the_range_neither_wrap_nor_vanish() {
    let top = IntervalSet::just(i64::MAX);
    assert!(top.holds(i64::MAX));
    assert!(!top.complement().holds(i64::MAX));
    assert!(top.complement().holds(i64::MIN));
    let bottom = IntervalSet::just(i64::MIN);
    assert!(bottom.holds(i64::MIN));
    assert!(!bottom.complement().holds(i64::MIN));
    assert!(bottom.complement().holds(i64::MAX));
    // The whole range, reached from both ends, is every integer.
    assert_eq!(
        IntervalSet::between(None, Some(0)).union(&IntervalSet::between(Some(1), None)),
        IntervalSet::all()
    );
    assert!(IntervalSet::all().complement().is_empty());
    assert_eq!(IntervalSet::empty().complement(), IntervalSet::all());
}

/// A reversed pair of bounds is the empty set, not a set read backwards.
#[test]
fn a_lower_bound_above_its_upper_bound_holds_nothing() {
    let reversed = IntervalSet::between(Some(5), Some(1));
    assert!(reversed.is_empty());
    assert_eq!(reversed.complement(), IntervalSet::all());
    assert_eq!(
        IntervalSet::between(Some(5), None).intersect(&IntervalSet::between(None, Some(1))),
        IntervalSet::empty()
    );
}
