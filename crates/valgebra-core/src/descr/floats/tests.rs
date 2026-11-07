use super::{
    FloatSet, MAX_PERIOD, Part, STRETCHES, Span, ceiling, fitted, gcd, inverse, is_either,
    normalize, stepless_at_or_above, whole_at_or_above,
};
use crate::descr::lattice_tests::Algebra;
use proptest::prelude::*;

/// The endpoints the generator writes, the float after `1.0` among them, so an
/// open end at one meets a closed end at its neighbor.
fn points() -> [f64; 9] {
    [
        f64::NEG_INFINITY,
        -2.0,
        -1.0,
        -0.0,
        0.0,
        1.0,
        1.0f64.next_up(),
        2.0,
        f64::INFINITY,
    ]
}

/// The floats a law is checked over.
///
/// Every endpoint a built set can have is a point or a neighbor of one: an
/// open end is held as its neighbor, and a complement ends a gap one float
/// short of the interval beside it. Each of those is here with its own
/// neighbors, so every run of floats between two endpoints has a member here,
/// the unbounded runs included. Both zeros, both infinities, and `nan` are here
/// too: a law that held over ordinary finite values alone would miss every one
/// of the three ways a float differs from an integer.
fn universe() -> Vec<f64> {
    let mut floats: Vec<f64> = points()
        .into_iter()
        .flat_map(|p| {
            let (up, down) = (p.next_up(), p.next_down());
            [p, up, down, up.next_up(), down.next_down()]
        })
        .collect();
    floats.push(f64::NAN);
    floats
}

/// Whether two sets hold the same floats. `nan` compares by membership, not
/// by equality, which is the whole reason it needs a bit of its own.
fn same(a: &FloatSet, b: &FloatSet) -> bool {
    universe().into_iter().all(|f| a.holds(f) == b.holds(f))
}

/// The float sets' operations, as the shared laws ask them. None of them
/// refuses.
const FLOATS: Algebra<FloatSet> = Algebra {
    join: FloatSet::union,
    meet: FloatSet::intersect,
    complement: FloatSet::complement,
    same,
    holds_nothing: FloatSet::is_empty,
    holds_everything: |a| same(a, &FloatSet::all()),
};

/// Sets built from endpoints inside the universe, so agreement on it is
/// agreement everywhere.
fn float_set() -> impl Strategy<Value = FloatSet> {
    let point = proptest::sample::select(points().to_vec());
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
            (inner.clone(), inner.clone()).prop_map(|(a, b)| a.union(&b).unwrap()),
            (inner.clone(), inner.clone()).prop_map(|(a, b)| a.intersect(&b).unwrap()),
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

    /// A set stepped at a period holds its classes as fitting them gives them,
    /// which is what lets it be read at that period as it stands.
    #[test]
    fn a_set_at_its_own_period_is_what_fitting_it_gives(a in stepped_set()) {
        let Some(stepped) = &a.stepped else {
            return Ok(());
        };
        let modulus = stepped.modulus;
        for (class, residue) in stepped.classes.iter().zip(0..) {
            prop_assert_eq!(
                &fitted(Part::Residue { modulus, residue }, class.clone()),
                class,
            );
        }
        prop_assert_eq!(&fitted(Part::Stepless, a.spans.clone()), &a.spans);
    }

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

    // THEORY: each-kind-is-closed
    /// The Boolean algebra, checked against the floats.
    #[test]
    fn the_lattice_laws_hold_of_the_floats(
        a in float_set(),
        b in float_set(),
        c in float_set(),
    ) {
        FLOATS.lattice_laws(&a, &b, &c)?;
    }

    /// The complement laws, and De Morgan both ways. The `nan` bit rides
    /// along: it is an ordinary two-element algebra beside the intervals.
    #[test]
    fn the_complement_laws_hold_of_the_floats(a in float_set(), b in float_set()) {
        FLOATS.complement_laws(&a, &b)?;
    }

    /// Holding the same floats is being equal, which is what the merging and
    /// the endpoint normalization are for.
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
    let halves = FloatSet::at_least(0.0)
        .union(&FloatSet::below(0.0))
        .unwrap();
    assert!(!halves.holds(f64::NAN));
    assert!(halves.holds(0.0));
    assert!(halves.holds(f64::INFINITY));
    assert!(halves.holds(f64::NEG_INFINITY));
    assert_ne!(halves, FloatSet::all());
    // What is missing is exactly `nan`, and adding it closes the gap.
    assert_eq!(halves.union(&FloatSet::nan()).unwrap(), FloatSet::all());
    assert_eq!(halves.complement(), FloatSet::nan());
}

// THEORY: a-constant-is-the-values-equal-to-it
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
    let straddling = FloatSet::at_most(-0.0)
        .union(&FloatSet::above(0.0))
        .unwrap();
    assert_eq!(straddling.spans.len(), 1, "the two zeros are one point");
    assert!(straddling.holds(0.0) && straddling.holds(-0.0));
}

/// An open and a closed end differ by one value, the end itself.
#[test]
fn an_open_end_excludes_exactly_its_own_point() {
    assert!(FloatSet::at_least(1.0).holds(1.0));
    assert!(!FloatSet::above(1.0).holds(1.0));
    assert!(FloatSet::above(1.0).holds(1.5));
    // The two differ by the point alone.
    assert_eq!(
        FloatSet::at_least(1.0)
            .intersect(&FloatSet::above(1.0).complement())
            .unwrap(),
        FloatSet::just(1.0)
    );
    // Meeting at a point that either side holds is one interval; meeting at
    // one that neither holds leaves a hole.
    assert_eq!(
        FloatSet::at_most(1.0).union(&FloatSet::above(1.0)).unwrap(),
        FloatSet::all()
            .intersect(&FloatSet::nan().complement())
            .unwrap()
    );
    let holed = FloatSet::below(1.0).union(&FloatSet::above(1.0)).unwrap();
    assert!(!holed.holds(1.0));
    assert!(holed.holds(0.5) && holed.holds(1.5));
}

/// No float lies between a float and the next one, so an open end is the
/// closed end at its neighbor, and two intervals no float separates are one.
#[test]
fn no_float_lies_between_a_float_and_the_next() {
    let next = 1.0f64.next_up();
    assert!(
        FloatSet::above(1.0)
            .intersect(&FloatSet::below(next))
            .unwrap()
            .is_empty()
    );
    assert_eq!(FloatSet::above(1.0), FloatSet::at_least(next));
    assert_eq!(FloatSet::below(next), FloatSet::at_most(1.0));
    assert_eq!(
        FloatSet::at_most(1.0)
            .union(&FloatSet::at_least(next))
            .unwrap(),
        FloatSet::all()
            .intersect(&FloatSet::nan().complement())
            .unwrap()
    );
    assert_eq!(
        FloatSet::just(1.0).union(&FloatSet::just(next)).unwrap(),
        FloatSet::at_least(1.0)
            .intersect(&FloatSet::at_most(next))
            .unwrap()
    );
    // Across zero, where the neighbor of the least positive float is the
    // zero both signs spell.
    let least = f64::from_bits(1);
    assert_eq!(FloatSet::above(0.0), FloatSet::at_least(least));
    assert_eq!(FloatSet::below(least), FloatSet::at_most(0.0));
    assert_eq!(FloatSet::above(-least), FloatSet::at_least(-0.0));
    // At the ends of the line: nothing lies past an infinity, and the largest
    // finite float is next to it.
    assert!(FloatSet::above(f64::INFINITY).is_empty());
    assert!(FloatSet::below(f64::NEG_INFINITY).is_empty());
    assert_eq!(FloatSet::above(f64::MAX), FloatSet::just(f64::INFINITY));
    assert_eq!(
        FloatSet::just(f64::INFINITY).complement(),
        FloatSet::at_most(f64::MAX).union(&FloatSet::nan()).unwrap()
    );
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
        FloatSet::all()
            .intersect(&FloatSet::nan().complement())
            .unwrap()
    );
    assert_eq!(line.complement(), FloatSet::nan());
}

/// The floats a law over steps is read at: whole floats of every residue the
/// drawn steps tell apart, on both sides of each edge where the spacing changes,
/// and the floats no step holds.
fn stepped_universe() -> Vec<f64> {
    let edges = [
        -f64::MAX,
        -(2.0f64.powi(63)),
        -(2.0f64.powi(53)),
        -(2.0f64.powi(52)),
        2.0f64.powi(52),
        2.0f64.powi(53),
        2.0f64.powi(63),
        2.0f64.powi(1000),
        f64::MAX,
    ];
    let mut floats: Vec<f64> = (-13..=13).map(f64::from).collect();
    floats.extend([
        0.5,
        -0.5,
        1.5,
        -1.5,
        -2.5,
        5e-324,
        -5e-324,
        1.0f64.next_up(),
        -0.0,
    ]);
    for edge in edges {
        let mut below = edge;
        let mut above = edge;
        floats.push(edge);
        for _ in 0..6 {
            below = below.next_down();
            above = above.next_up();
            floats.extend([below, above]);
        }
    }
    floats.extend([f64::INFINITY, f64::NEG_INFINITY, f64::NAN]);
    floats
}

/// The steps the laws draw, whose periods meet inside the bound.
const STEPS: [i64; 6] = [1, 2, 3, 4, 6, -3];

/// Whether `value % step == 0`, which is the remainder Python reads: exact, of a
/// finite float, and `nan` for an infinity.
fn a_multiple(value: f64, step: i64) -> bool {
    #[expect(clippy::cast_precision_loss, reason = "the steps here are small")]
    let step = step as f64;
    value.is_finite() && value % step == 0.0
}

/// Whether two sets hold the same floats of [`stepped_universe`].
fn same_over_steps(a: &FloatSet, b: &FloatSet) -> bool {
    stepped_universe()
        .into_iter()
        .all(|f| a.holds(f) == b.holds(f))
}

/// The float sets' operations over steps.
const STEPPED: Algebra<FloatSet> = Algebra {
    join: FloatSet::union,
    meet: FloatSet::intersect,
    complement: FloatSet::complement,
    same: same_over_steps,
    holds_nothing: FloatSet::is_empty,
    holds_everything: |a| same_over_steps(a, &FloatSet::all()),
};

/// Sets built from steps and from bounds at the universe's edges.
fn stepped_set() -> impl Strategy<Value = FloatSet> {
    let point = proptest::sample::select(vec![
        -(2.0f64.powi(53)),
        -3.0,
        -0.5,
        0.0,
        0.5,
        4.0,
        2.0f64.powi(52),
        2.0f64.powi(53),
        2.0f64.powi(63),
    ]);
    let leaf = prop_oneof![
        Just(FloatSet::all()),
        Just(FloatSet::empty()),
        Just(FloatSet::nan()),
        proptest::sample::select(STEPS.to_vec())
            .prop_map(|step| FloatSet::multiples_of(step).unwrap()),
        point.clone().prop_map(FloatSet::just),
        point.clone().prop_map(FloatSet::at_least),
        point.clone().prop_map(FloatSet::above),
        point.clone().prop_map(FloatSet::at_most),
        point.prop_map(FloatSet::below),
    ];
    leaf.prop_recursive(4, 24, 2, |inner| {
        prop_oneof![
            (inner.clone(), inner.clone()).prop_map(|(a, b)| a.union(&b).unwrap()),
            (inner.clone(), inner.clone()).prop_map(|(a, b)| a.intersect(&b).unwrap()),
            inner.prop_map(|a| a.complement()),
        ]
    })
}

proptest! {
    #![proptest_config(ProptestConfig {
        max_shrink_time: 2_000,
        ..ProptestConfig::default()
    })]

    /// The laws of the algebra over sets a step has split.
    #[test]
    fn the_laws_hold_over_steps(a in stepped_set(), b in stepped_set(), c in stepped_set()) {
        STEPPED.lattice_laws(&a, &b, &c)?;
        STEPPED.complement_laws(&a, &b)?;
    }

    /// Equality reads the floats and not the spelling, so a form a law builds at
    /// another period is the set it says it is.
    #[test]
    fn equality_reads_the_floats_over_steps(a in stepped_set(), b in stepped_set()) {
        prop_assert_eq!(a == b, same_over_steps(&a, &b));
        let lifted = a.intersect(&FloatSet::all()).unwrap();
        prop_assert_eq!(&lifted, &a);
        let spelled_at_a_period = a
            .intersect(&FloatSet::multiples_of(4).unwrap())
            .unwrap()
            .union(&a.intersect(&FloatSet::multiples_of(4).unwrap().complement()).unwrap())
            .unwrap();
        prop_assert_eq!(&spelled_at_a_period, &a);
    }

    /// A set the universe finds a float of is not read empty.
    #[test]
    fn a_set_holding_a_drawn_float_is_not_empty(a in stepped_set()) {
        let held = stepped_universe().into_iter().any(|f| a.holds(f));
        prop_assert!(!held || !a.is_empty());
    }
}

#[test]
fn a_step_holds_what_the_remainder_holds() {
    for step in [1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 100, 4096, -1, -3] {
        let set = FloatSet::multiples_of(step).unwrap();
        for value in stepped_universe() {
            assert_eq!(
                set.holds(value),
                a_multiple(value, step),
                "{value} against a step of {step}"
            );
        }
    }
}

/// The floats of a narrow window, by walking them, against what the set reads:
/// `is_empty` is exact, since a relation's refutation stands on it, so it is held
/// against the floats themselves where the spacing changes.
#[test]
fn emptiness_over_a_window_is_the_floats_the_window_holds() {
    let starts = [
        -f64::MAX,
        -(2.0f64.powi(1000)),
        -(2.0f64.powi(63)),
        -(2.0f64.powi(53)) - 8.0,
        -(2.0f64.powi(52)) - 4.0,
        -3.5,
        0.0,
        2.0f64.powi(52) - 4.0,
        2.0f64.powi(53) - 8.0,
        2.0f64.powi(63) - 4096.0,
        2.0f64.powi(1000),
        f64::MAX.next_down().next_down().next_down(),
    ];
    for start in starts {
        let mut floats = vec![start];
        for _ in 0..12 {
            let last = *floats.last().unwrap();
            if last.total_cmp(&f64::MAX).is_eq() {
                break;
            }
            floats.push(last.next_up());
        }
        for step in [1, 2, 3, 5, 8, 7] {
            let multiples = FloatSet::multiples_of(step).unwrap();
            let others = multiples.complement();
            for (first, lo) in floats.iter().enumerate() {
                for hi in floats.iter().skip(first) {
                    let window = FloatSet::at_least(*lo)
                        .intersect(&FloatSet::at_most(*hi))
                        .unwrap();
                    let inside = |f: &&f64| **f >= *lo && **f <= *hi;
                    let on = window.intersect(&multiples).unwrap();
                    assert_eq!(
                        on.is_empty(),
                        !floats.iter().filter(inside).any(|f| a_multiple(*f, step)),
                        "the multiples of {step} in [{lo}, {hi}]"
                    );
                    let off = window.intersect(&others).unwrap();
                    assert_eq!(
                        off.is_empty(),
                        floats.iter().filter(inside).all(|f| a_multiple(*f, step)),
                        "the floats that are no multiple of {step} in [{lo}, {hi}]"
                    );
                }
            }
        }
    }
}

/// A class is found where it first has a float, however far above the bound: past
/// `2^53` the whole floats are multiples of a power of two, which an odd residue
/// of an even period never is.
#[test]
fn a_class_is_found_where_it_first_has_a_float() {
    let at = |from, modulus, residue| whole_at_or_above(from, modulus, residue);
    let top = 2.0f64.powi(53);
    assert_eq!(at(0.5, 2, 1), Some(1.0));
    assert_eq!(at(-0.5, 2, 0), Some(0.0));
    assert_eq!(at(-1.5, 3, 2), Some(-1.0));
    assert_eq!(at(top, 2, 1), None, "past 2^53 every float is even");
    assert_eq!(at(top - 1.0, 2, 1), Some(top - 1.0));
    // A multiple of three past a spacing of two: 2^53 + 1 is the first, and no float.
    assert_eq!(at(top, 3, 0), Some(top + 4.0));
    assert_eq!(at(f64::MAX, 2, 0), Some(f64::MAX));
    assert_eq!(at(f64::MAX, 2, 1), None);
    assert_eq!(at(f64::INFINITY, 1, 0), None);
    assert_eq!(at(f64::NEG_INFINITY, 1, 0), Some(-f64::MAX));
    // The odd residues run out below `-2^53` as above it.
    assert_eq!(at(-top * 4.0, 2, 1), Some(-top + 1.0));
    // An odd period has every residue in every stretch.
    assert!(at(2.0f64.powi(1000), 3, 1).is_some());
    assert!(at(2.0f64.powi(1000), 3, 2).is_some());
}

/// The search for a class's next float agrees with walking the floats, from a few
/// floats short of each place the spacing changes, on both sides of zero: a stretch
/// read too long names a float that is not there, and one read too short misses one.
#[test]
fn a_class_search_is_the_floats_a_walk_finds() {
    // 971 binades past `2^53` on each side of zero, and the integers between them.
    let binades = usize::try_from(f64::MAX_EXP).unwrap() - 53;
    assert!(
        STRETCHES > 2 * binades + 1,
        "the search ends at its ceiling"
    );
    let mut starts = vec![-3.5, -0.5, 0.25, 7.0];
    for exponent in [52, 53, 54, 55, 63, 200, 1000] {
        let edge = 2.0f64.powi(exponent);
        let mut below = edge;
        let mut above = -edge;
        // From each float short of the edge, so a class whose next float lies just
        // past it is asked from the stretch before it.
        for _ in 0..6 {
            below = below.next_down();
            above = above.next_down();
            starts.extend([below, above]);
        }
    }
    // From each of the largest floats, where the last stretch ends and no float follows.
    let mut last = f64::MAX;
    starts.extend([last, -f64::MAX]);
    for _ in 0..6 {
        last = last.next_down();
        starts.push(last);
    }
    let wide = |value: i64| {
        #[expect(clippy::cast_precision_loss, reason = "the moduli here are small")]
        let value = value as f64;
        value
    };
    for start in starts {
        for modulus in [1, 2, 3, 4, 5, 6, 7, 8, 12] {
            for residue in 0..modulus {
                // A zero of either sign reads as whole, and as the residue zero.
                let in_class = |float: f64| {
                    float.fract().to_bits() << 1 == 0
                        && (float.rem_euclid(wide(modulus)) + 0.0).to_bits()
                            == wide(residue).to_bits()
                };
                let mut float = start;
                let mut walked = None;
                for _ in 0..24 {
                    if float.is_infinite() {
                        break;
                    }
                    if in_class(float) {
                        walked = Some(float);
                        break;
                    }
                    float = float.next_up();
                }
                let found = whole_at_or_above(start, modulus, residue);
                match walked {
                    Some(first) => assert_eq!(
                        found.map(f64::to_bits),
                        Some(normalize(first).to_bits()),
                        "from {start}, residue {residue} of {modulus}"
                    ),
                    // The walk stopped short of the search: whatever the search
                    // found lies past it and is a float of the class.
                    None => assert!(
                        found.is_none_or(|float| float >= start && in_class(float)),
                        "from {start}, residue {residue} of {modulus}: {found:?}"
                    ),
                }
            }
        }
    }
}

/// An inverse is read off the two sequences Euclid's algorithm keeps, each step of which
/// matters past the first few quotients.
#[test]
fn an_inverse_is_the_inverse() {
    for period in 2..=64 {
        for a in 1..period {
            if gcd(a, period) == 1 {
                assert_eq!(a * inverse(a, period) % period, 1, "{a} modulo {period}");
            }
        }
    }
    assert_eq!(
        inverse(0, 1),
        0,
        "every value is the inverse of every other modulo one"
    );
    assert_eq!(gcd(0, 12), 12);
    assert_eq!(gcd(8, 12), 4);
}

/// A residue holding fewer twos than the modulus has no float past the spacing its
/// twos reach, and one holding as many is read to the largest float.
#[test]
fn a_class_reaches_as_far_as_its_twos() {
    assert_eq!(ceiling(1, 0).to_bits(), f64::MAX.to_bits());
    assert_eq!(ceiling(3, 1).to_bits(), f64::MAX.to_bits());
    assert_eq!(ceiling(4, 0).to_bits(), f64::MAX.to_bits());
    assert_eq!(ceiling(4, 8).to_bits(), f64::MAX.to_bits());
    let top = 2.0f64.powi(53);
    assert_eq!(ceiling(4, 1).to_bits(), top.next_down().to_bits());
    assert_eq!(ceiling(4, 2).to_bits(), (top * 2.0).next_down().to_bits());
    // The last float of a class under its ceiling, and the first past it.
    assert_eq!(whole_at_or_above(top - 1.0, 4, 3), Some(top - 1.0));
    assert_eq!(whole_at_or_above(top, 4, 3), None);
    assert_eq!(
        whole_at_or_above(top * 2.0 - 2.0, 4, 2),
        Some(top * 2.0 - 2.0)
    );
    assert_eq!(whole_at_or_above(top * 2.0, 4, 2), None);
    assert_eq!(whole_at_or_above(-top * 8.0, 4, 3), Some(-top + 3.0));
}

/// A mirror's residue is read modulo the period, so the two ends of a span read one
/// class whichever side of zero the span lies on.
#[test]
fn the_upper_end_of_a_class_is_the_mirror_of_its_lower_end() {
    let part = |residue| Part::Residue {
        modulus: 5,
        residue,
    };
    assert_eq!(part(0).at_or_below(7.0), Some(5.0));
    assert_eq!(part(1).at_or_below(5.0), Some(1.0));
    assert_eq!(part(3).at_or_below(-1.0), Some(-2.0));
    assert_eq!(part(0).at_or_below(-1.0), Some(-5.0));
    assert_eq!(
        part(0).at_or_below(0.5).map(f64::to_bits),
        Some(0.0f64.to_bits())
    );
    assert_eq!(Part::Stepless.at_or_below(1.0), Some(1.0f64.next_down()));
    assert_eq!(
        Part::Stepless.at_or_below(f64::NEG_INFINITY),
        Some(f64::NEG_INFINITY)
    );
}

/// The widest period reads as the narrowest does, on the one stretch that tells
/// them apart: `2^52` is the first multiple of 4096 there, and past `2^53` the
/// spacing is a factor of the period.
#[test]
fn the_widest_period_finds_its_floats() {
    let multiples = FloatSet::multiples_of(MAX_PERIOD).unwrap();
    let top = 2.0f64.powi(52);
    let at = |lo: f64, hi: f64| {
        FloatSet::at_least(lo)
            .intersect(&FloatSet::at_most(hi))
            .unwrap()
            .intersect(&multiples)
            .unwrap()
    };
    assert!(!at(top, top).is_empty());
    assert!(at(top + 1.0, top + 4095.0).is_empty());
    assert!(!at(top + 1.0, top + 4096.0).is_empty());
    assert!(!at(2.0f64.powi(900), 2.0f64.powi(900)).is_empty());
}

#[test]
fn a_float_holding_no_step_is_found_past_the_whole_floats() {
    let top = 2.0f64.powi(52);
    assert_eq!(stepless_at_or_above(1.0), 1.0f64.next_up());
    assert_eq!(stepless_at_or_above(0.5), 0.5);
    assert_eq!(stepless_at_or_above(top), f64::INFINITY);
    assert_eq!(stepless_at_or_above(top - 1.0), top - 0.5);
    assert_eq!(stepless_at_or_above(-top), (-top).next_up());
    assert_eq!(stepless_at_or_above(-top * 4.0), (-top).next_up());
    assert_eq!(stepless_at_or_above(f64::NEG_INFINITY), f64::NEG_INFINITY);
    assert_eq!(stepless_at_or_above(f64::INFINITY), f64::INFINITY);
}

#[test]
fn a_set_a_step_split_lists_the_floats_it_holds() {
    let between = FloatSet::at_least(0.0)
        .intersect(&FloatSet::at_most(6.5))
        .unwrap();
    let even = between
        .intersect(&FloatSet::multiples_of(2).unwrap())
        .unwrap();
    assert_eq!(even.values_up_to(10), Some(vec![0.0, 2.0, 4.0, 6.0]));
    assert_eq!(even.values_up_to(3), None, "past the cap");
    let wholes = between
        .intersect(&FloatSet::multiples_of(1).unwrap())
        .unwrap();
    assert_eq!(wholes.values_up_to(7).map(|values| values.len()), Some(7));
    assert_eq!(wholes.complement().values_up_to(4), None, "it holds `nan`");
}

#[test]
fn a_step_past_the_period_bound_is_refused_by_the_floats() {
    assert!(FloatSet::multiples_of(MAX_PERIOD).is_some());
    assert!(FloatSet::multiples_of(MAX_PERIOD + 1).is_none());
    assert!(FloatSet::multiples_of(i64::MIN).is_none());
    assert!(
        FloatSet::multiples_of(0).is_none(),
        "no float is a multiple of zero"
    );
    let (a, b) = (
        FloatSet::multiples_of(64).unwrap(),
        FloatSet::multiples_of(81).unwrap(),
    );
    assert!(a.intersect(&b).is_none(), "they meet at 5184");
    assert!(a.union(&b).is_none());
    assert_ne!(a, b, "two periods past the bound read as two sets");
}

#[test]
fn a_step_splits_the_line_and_every_other_set_reads_across_it() {
    let whole = FloatSet::multiples_of(1).unwrap();
    let rest = whole.complement();
    assert!(rest.holds(0.5) && rest.holds(f64::INFINITY) && rest.holds(f64::NAN));
    assert!(!rest.holds(1.0) && !rest.holds(-0.0));
    assert_eq!(whole.union(&rest).unwrap(), FloatSet::all());
    assert!(whole.intersect(&rest).unwrap().is_empty());
    // Between two whole floats no whole float lies, though two fractional ones do.
    let open = FloatSet::above(30.0)
        .intersect(&FloatSet::below(31.0))
        .unwrap();
    assert!(whole.intersect(&open).unwrap().is_empty());
    assert!(!rest.intersect(&open).unwrap().is_empty());
    // And a literal against a step asks the one float.
    assert!(!FloatSet::just(4.0).intersect(&whole).unwrap().is_empty());
    assert!(FloatSet::just(4.5).intersect(&whole).unwrap().is_empty());
}

/// A table holds its residues and the spans in each, as the integers' does.
#[test]
fn a_table_holds_its_residues_and_their_spans() {
    let four = FloatSet::multiples_of(4).expect("a step inside the bound");
    let table = four.stepped.as_deref().expect("a step's table");
    assert_eq!(table.entries(), 4 + 1);
    let cut = four
        .intersect(
            &FloatSet::at_most(100.0)
                .union(&FloatSet::at_least(200.0))
                .unwrap(),
        )
        .expect("a small set");
    let table = cut.stepped.as_deref().expect("a step's table");
    assert_eq!(table.entries(), 4 + 2);
}

/// An operation over a step's table charges the residues and spans it holds, as
/// an operation over the integers' does.
#[test]
fn an_operation_over_a_table_charges_what_the_table_holds() {
    use crate::descr::budget::{TABLE_ENTRIES, under};
    let wide = FloatSet::multiples_of(MAX_PERIOD).expect("a step inside the bound");
    let table = wide.stepped.as_deref().expect("a step's table");
    let units = u64::try_from(table.entries() / TABLE_ENTRIES).expect("a table's size");
    assert!(units > 0, "the widest table is past a unit");
    assert!(under(units, || wide.union(&wide)).is_some());
    assert!(under(units - 1, || wide.union(&wide)).is_none());
    assert!(
        under(units, || {
            let other = wide.complement();
            wide.union(&other)
        })
        .is_none()
    );
}

/// A result is taken as fitted where it is either list it was made from, and no
/// other: the reading a fit is spared on, whose cost no answer shows.
#[test]
fn a_result_is_either_list_it_was_made_from() {
    let span = |lo: f64, hi: f64| Span::new(lo, true, hi, true);
    let (mine, theirs, other) = (
        vec![span(0.0, 1.0)],
        vec![span(2.0, 3.0)],
        vec![span(4.0, 5.0)],
    );
    assert!(is_either(&mine, &mine, &theirs));
    assert!(is_either(&theirs, &mine, &theirs));
    assert!(!is_either(&other, &mine, &theirs));
}
