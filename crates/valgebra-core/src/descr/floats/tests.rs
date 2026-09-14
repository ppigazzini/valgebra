use super::FloatSet;
use proptest::prelude::*;

/// The floats a law is checked over.
///
/// Both zeros, both infinities, `nan`, and a point strictly inside every
/// gap the generator's endpoints leave -- including the two unbounded ones,
/// where a set like `(-inf, -2.0)` is inhabited by values no endpoint names.
/// The three that make floats their own case are all here: a law that held
/// over ordinary finite values alone would miss every one of them.
fn universe() -> Vec<f64> {
    vec![
        f64::NEG_INFINITY,
        -3.0,
        -2.0,
        -1.5,
        -1.0,
        -0.5,
        -0.0,
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        f64::INFINITY,
        f64::NAN,
    ]
}

/// Whether two sets hold the same floats. `nan` compares by membership, not
/// by equality, which is the whole reason it needs a bit of its own.
fn same(a: &FloatSet, b: &FloatSet) -> bool {
    universe().into_iter().all(|f| a.holds(f) == b.holds(f))
}

/// Sets built from endpoints inside the universe, so agreement on it is
/// agreement everywhere.
fn float_set() -> impl Strategy<Value = FloatSet> {
    let point = prop_oneof![
        Just(-2.0f64),
        Just(-1.0),
        Just(-0.0),
        Just(0.0),
        Just(1.0),
        Just(2.0),
        Just(f64::INFINITY),
        Just(f64::NEG_INFINITY),
    ];
    let leaf = prop_oneof![
        Just(FloatSet::empty()),
        Just(FloatSet::all()),
        Just(FloatSet::nan()),
        point.clone().prop_map(FloatSet::just),
        point.clone().prop_map(FloatSet::at_least),
        point.clone().prop_map(FloatSet::above),
        point.clone().prop_map(FloatSet::at_most),
        point.prop_map(FloatSet::below),
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

    /// The order is total and agrees with equality, which is all a sort of
    /// guards asks of it. Written by hand here because an endpoint is a
    /// float, so it is the one component whose order needs a law.
    #[test]
    fn the_order_is_total_and_agrees_with_equality(a in float_set(), b in float_set()) {
        prop_assert_eq!(a.partial_cmp(&b), Some(a.cmp(&b)));
        // The spans carry the order the set's is built from, and a partial
        // order that declined a pair would leave the set's undefined there.
        prop_assert_eq!(a.spans.partial_cmp(&b.spans), Some(a.spans.cmp(&b.spans)));
        prop_assert_eq!(a.cmp(&b) == core::cmp::Ordering::Equal, a == b);
        prop_assert_eq!(a.cmp(&b), b.cmp(&a).reverse());
    }

    /// The Boolean algebra, checked against the floats.
    #[test]
    fn the_lattice_laws_hold_of_the_floats(
        a in float_set(),
        b in float_set(),
        c in float_set(),
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

    /// The complement laws, and De Morgan both ways. The `nan` bit rides
    /// along: it is an ordinary two-element algebra beside the intervals.
    #[test]
    fn the_complement_laws_hold_of_the_floats(a in float_set(), b in float_set()) {
        prop_assert!(a.intersect(&a.complement()).is_empty());
        prop_assert!(same(&a.union(&a.complement()), &FloatSet::all()));
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

    /// Holding the same floats is being equal, which is what the merging and
    /// the endpoint normalisation are for.
    #[test]
    fn holding_the_same_floats_is_being_equal(a in float_set(), b in float_set()) {
        prop_assert_eq!(same(&a, &b), a == b);
    }

    /// Emptiness is a decision.
    #[test]
    fn emptiness_agrees_with_the_floats(a in float_set()) {
        prop_assert_eq!(a.is_empty(), universe().into_iter().all(|f| !a.holds(f)));
    }
}

/// The two halves of the ordered line do not cover `float`, and the reason
/// is the one value that sits in neither.
///
/// This is what the `nan` bit exists to say. Without it the two halves would
/// be the whole kind, and `float` would be decided equal to their union --
/// a claim no float supports, because `nan` is a float and is in neither.
#[test]
fn the_ordered_halves_leave_nan_outside() {
    let halves = FloatSet::at_least(0.0).union(&FloatSet::below(0.0));
    assert!(!halves.holds(f64::NAN));
    assert!(halves.holds(0.0));
    assert!(halves.holds(f64::INFINITY));
    assert!(halves.holds(f64::NEG_INFINITY));
    assert_ne!(halves, FloatSet::all());
    // What is missing is exactly `nan`, and adding it closes the gap.
    assert_eq!(halves.union(&FloatSet::nan()), FloatSet::all());
    assert_eq!(halves.complement(), FloatSet::nan());
}

/// A literal `nan` admits nothing, because `nan` is equal to no value --
/// itself included.
#[test]
fn a_literal_nan_is_the_empty_set() {
    assert!(FloatSet::just(f64::NAN).is_empty());
    assert!(!FloatSet::just(f64::NAN).holds(f64::NAN));
    // The set that *does* hold it is a different thing, and it is not a
    // literal: `float` holds `nan` because `nan` is a float.
    assert!(FloatSet::nan().holds(f64::NAN));
    assert!(FloatSet::all().holds(f64::NAN));
    // A bound of `nan` admits nothing either: every comparison with it is
    // false, so no float is at least it.
    assert!(FloatSet::at_least(f64::NAN).is_empty());
    assert!(FloatSet::below(f64::NAN).is_empty());
}

/// The two zeros are one value, so no set can hold one without the other.
#[test]
fn the_two_zeros_are_one_value() {
    assert_eq!(FloatSet::just(-0.0), FloatSet::just(0.0));
    assert!(FloatSet::just(0.0).holds(-0.0));
    assert!(FloatSet::just(-0.0).holds(0.0));
    // A strict bound at either zero excludes both, since they are the same
    // point on the line.
    assert!(!FloatSet::above(-0.0).holds(0.0));
    assert!(!FloatSet::above(0.0).holds(-0.0));
    assert_eq!(FloatSet::above(-0.0), FloatSet::above(0.0));
    assert_eq!(FloatSet::at_least(-0.0), FloatSet::at_least(0.0));

    // The assertions above pass whatever the endpoints carry, because
    // `==` on a pair of `f64` already equates the two zeros. What the
    // representation needs is that the *order* equates them too: it is
    // `total_cmp` that decides whether a span is empty, and IEEE 754 puts
    // `-0` below `+0` there. A negative zero reaching an endpoint would
    // make this span crossed and empty while it holds zero.
    assert_eq!(
        FloatSet::just(-0.0).cmp(&FloatSet::just(0.0)),
        core::cmp::Ordering::Equal
    );
    let straddling = FloatSet::at_most(-0.0).union(&FloatSet::above(0.0));
    assert_eq!(straddling.spans.len(), 1, "the two zeros are one point");
    assert!(straddling.holds(0.0) && straddling.holds(-0.0));
}

/// An open and a closed end differ by one value, which is the reason the
/// inclusivity is carried rather than rounded to the nearest float.
#[test]
fn an_open_end_excludes_exactly_its_own_point() {
    assert!(FloatSet::at_least(1.0).holds(1.0));
    assert!(!FloatSet::above(1.0).holds(1.0));
    assert!(FloatSet::above(1.0).holds(1.5));
    // The two differ by the point alone.
    assert_eq!(
        FloatSet::at_least(1.0).intersect(&FloatSet::above(1.0).complement()),
        FloatSet::just(1.0)
    );
    // Meeting at a point that either side holds is one interval; meeting at
    // one that neither holds leaves a hole.
    assert_eq!(
        FloatSet::at_most(1.0).union(&FloatSet::above(1.0)),
        FloatSet::all().intersect(&FloatSet::nan().complement())
    );
    let holed = FloatSet::below(1.0).union(&FloatSet::above(1.0));
    assert!(!holed.holds(1.0));
    assert!(holed.holds(0.5) && holed.holds(1.5));
}

/// The infinities are floats, not open ends: a set can hold them, exclude
/// them, and be bounded by them.
#[test]
fn the_infinities_are_values_of_the_set() {
    assert!(FloatSet::at_least(f64::NEG_INFINITY).holds(f64::NEG_INFINITY));
    assert!(FloatSet::at_least(f64::NEG_INFINITY).holds(f64::INFINITY));
    assert!(!FloatSet::above(f64::NEG_INFINITY).holds(f64::NEG_INFINITY));
    assert!(!FloatSet::below(f64::INFINITY).holds(f64::INFINITY));
    assert!(
        !FloatSet::just(f64::INFINITY)
            .complement()
            .holds(f64::INFINITY)
    );
    // The whole ordered line, reached from both ends, is everything but
    // `nan`.
    let line = FloatSet::at_least(f64::NEG_INFINITY);
    assert_eq!(
        line,
        FloatSet::all().intersect(&FloatSet::nan().complement())
    );
    assert_eq!(line.complement(), FloatSet::nan());
}
