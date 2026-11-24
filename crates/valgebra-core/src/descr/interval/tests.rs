use super::{
    IntervalSet, Span, Spans, narrow_complement, narrow_preimage, wide_canonical, wide_complement,
    wide_preimage,
};
use crate::descr::lattice_tests::Algebra;
use crate::descr::wide::WideInt;
use num_bigint::BigInt;
use proptest::prelude::*;
use std::sync::Arc;

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

/// The interval sets' operations, as the shared laws ask them. None of them
/// refuses.
const INTERVALS: Algebra<IntervalSet> = Algebra {
    join: |a, b| Some(a.union(b)),
    meet: |a, b| Some(a.intersect(b)),
    complement: IntervalSet::complement,
    same,
    holds_nothing: IntervalSet::is_empty,
    holds_everything: |a| same(a, &IntervalSet::all()),
};

/// Sets built from bounded endpoints, so agreement on the window is
/// agreement everywhere: no generated set has a feature outside it.
fn interval_set() -> impl Strategy<Value = IntervalSet> {
    let leaf = prop_oneof![
        Just(IntervalSet::empty()),
        Just(IntervalSet::all()),
        (-8i64..=8).prop_map(IntervalSet::just),
        (-8i128..=8).prop_map(|lo| IntervalSet::between(Some(WideInt::Small(lo)), None)),
        (-8i128..=8).prop_map(|hi| IntervalSet::between(None, Some(WideInt::Small(hi)))),
        (-8i128..=8, -8i128..=8).prop_map(|(a, b)| IntervalSet::between(
            Some(WideInt::Small(a.min(b))),
            Some(WideInt::Small(a.max(b)))
        )),
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

    // THEORY: each-kind-is-closed
    /// The Boolean algebra, checked against the integers rather than against
    /// the two representations agreeing.
    #[test]
    fn the_lattice_laws_hold_of_the_integers(
        a in interval_set(),
        b in interval_set(),
        c in interval_set(),
    ) {
        INTERVALS.lattice_laws(&a, &b, &c)?;
    }

    /// The complement laws, and De Morgan both ways.
    #[test]
    fn the_complement_laws_hold_of_the_integers(a in interval_set(), b in interval_set()) {
        INTERVALS.complement_laws(&a, &b)?;
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

    // THEORY: the-carriers-are-i64-and-f64
    /// The operations hold of the integers past any width: a set moved by an
    /// offset past `i128`'s range, or one that carries its ends across the
    /// range's edge, is met, joined, complemented and divided as the set it was
    /// moved from, and lands where that one's answer lands. The ends then take
    /// both spellings of `WideInt`, and the integer just past a span's end,
    /// which a complement names, is exact on either side of the edge.
    #[test]
    fn the_operations_hold_past_any_width(
        a in interval_set(),
        b in interval_set(),
        far in prop_oneof![
            Just(BigInt::from(1_i64) << 200_u32),
            Just(-(BigInt::from(1_i64) << 200_u32)),
            Just(BigInt::from(i128::MAX) - 4),
            Just(BigInt::from(i128::MIN) + 4),
        ],
        stride in 1i64..=4,
    ) {
        let by = |set: &IntervalSet| moved(set, &far);
        // Moved and moved back, a set is the set it was, in the form it was.
        prop_assert_eq!(&moved(&by(&a), &-&far), &a);
        prop_assert_eq!(by(&a.union(&b)), by(&a).union(&by(&b)));
        prop_assert_eq!(by(&a.intersect(&b)), by(&a).intersect(&by(&b)));
        prop_assert_eq!(by(&a.complement()), by(&a).complement());
        prop_assert_eq!(by(&a).is_empty(), a.is_empty());
        // Moved by a multiple of the stride, a set's preimage moves by the
        // multiple's quotient.
        let step = &far * stride;
        prop_assert_eq!(
            moved(&moved(&a, &step).preimage(0, stride), &-&far),
            a.preimage(0, stride)
        );
    }

    /// A set whose ends an `i128` holds is held in the narrow form, and each
    /// operation's narrow reading answers as its wide one: the two forms are
    /// one set, ordered alike.
    #[test]
    fn the_narrow_form_answers_as_the_wide_one(
        a in interval_set(),
        b in interval_set(),
        offset in -4i64..=4,
        stride in 1i64..=4,
    ) {
        let wide = |set: &IntervalSet| set.wide_spans().collect::<Vec<Span>>();
        prop_assert!(matches!(a.spans, Spans::Narrow(_)));
        prop_assert_eq!(
            a.union(&b),
            wide_canonical(wide(&a).into_iter().chain(wide(&b)).collect())
        );
        let met = wide(&a)
            .iter()
            .flat_map(|mine| wide(&b).into_iter().map(move |theirs| mine.intersect(&theirs)))
            .collect();
        prop_assert_eq!(a.intersect(&b), wide_canonical(met));
        prop_assert_eq!(a.complement(), wide_complement(&wide(&a)));
        prop_assert_eq!(a.preimage(offset, stride), wide_preimage(&wide(&a), offset, stride));
        prop_assert_eq!(a.cmp(&b), wide(&a).cmp(&wide(&b)));
        prop_assert_eq!(a.partial_cmp(&b), Some(wide(&a).cmp(&wide(&b))));
        // The window's ends are far inside the width, so the narrow readings
        // answer themselves, and answer as the wide ones do.
        let Spans::Narrow(spans) = &a.spans else {
            unreachable!("a set of the window is narrow")
        };
        prop_assert_eq!(narrow_complement(spans), Some(wide_complement(&wide(&a))));
        prop_assert_eq!(
            narrow_preimage(spans, offset, stride),
            Some(wide_preimage(&wide(&a), offset, stride))
        );
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
    let low = IntervalSet::between(Some(WideInt::Small(0)), Some(WideInt::Small(3)));
    let high = IntervalSet::between(Some(WideInt::Small(4)), Some(WideInt::Small(7)));
    assert_eq!(
        low.union(&high),
        IntervalSet::between(Some(WideInt::Small(0)), Some(WideInt::Small(7)))
    );
    // A gap of one integer is a gap, and the two stay apart.
    let apart = IntervalSet::between(Some(WideInt::Small(5)), Some(WideInt::Small(7)));
    assert_ne!(
        low.union(&apart),
        IntervalSet::between(Some(WideInt::Small(0)), Some(WideInt::Small(7)))
    );
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
        IntervalSet::between(None, Some(WideInt::Small(0)))
            .union(&IntervalSet::between(Some(WideInt::Small(1)), None)),
        IntervalSet::all()
    );
    assert!(IntervalSet::all().complement().is_empty());
    assert_eq!(IntervalSet::empty().complement(), IntervalSet::all());
}

/// A reversed pair of bounds is the empty set, not a set read backwards.
#[test]
fn a_lower_bound_above_its_upper_bound_holds_nothing() {
    let reversed = IntervalSet::between(Some(WideInt::Small(5)), Some(WideInt::Small(1)));
    assert!(reversed.is_empty());
    assert_eq!(reversed.complement(), IntervalSet::all());
    assert_eq!(
        IntervalSet::between(Some(WideInt::Small(5)), None)
            .intersect(&IntervalSet::between(None, Some(WideInt::Small(1)))),
        IntervalSet::empty()
    );
}

/// A residue class holds the values it holds at the bottom of the range.
///
/// `preimage` is the change of variable a modulus needs: it holds exactly the
/// `k` whose image `offset + stride * k` lies in the span. At the bottom of the
/// range that subtraction leaves `i64`, so the quotient is taken wide and
/// clamped back -- and the clamp has a side. Clamping the low end to the *top*
/// of the range gives a class holding nothing below it, which is a set neither
/// the schema nor any other class denotes.
#[test]
fn a_residue_class_at_the_bottom_of_the_range_keeps_its_members() {
    const WIDTH: i128 = 8;
    let low = i128::from(i64::MIN);
    let span = IntervalSet::between(
        Some(WideInt::Small(i128::from(i64::MIN))),
        Some(WideInt::Small(i128::from(i64::MIN) + 8)),
    );

    for stride in [1i64, 2, 3, 5] {
        for offset in 0..stride {
            let lifted = span.preimage(offset, stride);
            let center = low / i128::from(stride);
            for wide in (center - 4)..=(center + 4) {
                let Ok(k) = i64::try_from(wide) else {
                    continue;
                };
                let image = i128::from(offset) + i128::from(stride) * wide;
                let inside = image >= low && image <= low + WIDTH;
                assert_eq!(
                    lifted.holds(k),
                    inside,
                    "k = {k} maps to {image} under stride {stride} offset {offset}"
                );
            }
        }
    }
}

/// A half-line beginning at the smallest bound a schema can name has a
/// complement, and it is not the empty set.
///
/// The bounds a caller writes are `i64`, and the integers are not: `Ge(-2**63)`
/// refuses `-2**63 - 1`, which is a Python integer. Naming a span's ends in the
/// bounds' own width would leave that integer nowhere to go, so the complement
/// of the half-line would come out empty -- which reads as "every integer is
/// at least `-2**63`" and proves an inclusion a value refutes. The ends are
/// wider than the bounds for this, and one step of headroom is all the
/// operations ask for.
#[test]
fn a_half_line_at_the_end_of_the_bounds_keeps_its_complement() {
    let from_the_bottom = IntervalSet::between(Some(WideInt::Small(i128::from(i64::MIN))), None);
    let below = from_the_bottom.complement();
    assert!(!below.is_empty(), "the integers under the smallest bound");
    assert!(
        !below.holds(i64::MIN),
        "and none of the bound's own half-line"
    );
    assert!(!below.holds(0));

    let to_the_top = IntervalSet::between(None, Some(WideInt::Small(i128::from(i64::MAX))));
    let above = to_the_top.complement();
    assert!(!above.is_empty(), "the integers over the largest bound");
    assert!(!above.holds(i64::MAX));

    // The two halves still cover every integer between them, and neither is
    // the whole line: a complement that vanished would make each of them one.
    assert_eq!(
        from_the_bottom.union(&below),
        IntervalSet::all(),
        "a set and its complement are the universe"
    );
    assert!(from_the_bottom.intersect(&below).is_empty());
    assert_ne!(from_the_bottom, IntervalSet::all());
    assert_ne!(to_the_top, IntervalSet::all());

    // Complementing twice is the identity here as everywhere else, which is
    // what says the headroom is enough for the operation to be closed.
    assert_eq!(below.complement(), from_the_bottom);
    assert_eq!(above.complement(), to_the_top);
}

/// A bounded span at either end is unaffected: it names a set the bounds spell.
#[test]
fn a_bounded_span_at_the_end_of_the_range_is_the_set_it_names() {
    let least = IntervalSet::between(
        Some(WideInt::Small(i128::from(i64::MIN))),
        Some(WideInt::Small(i128::from(i64::MIN))),
    );
    assert_eq!(least, IntervalSet::just(i64::MIN));
    assert!(least.holds(i64::MIN));
    assert!(!least.complement().holds(i64::MIN));
    // And the residue reading of it, which is what a step meets it through.
    assert!(least.preimage(0, 2).holds(i64::MIN / 2));
}

/// A span ending at an end of `i128` has integers past that end, so its
/// complement leaves the narrow form, and a second complement comes back to it;
/// a preimage whose shift leaves the width is taken wide too.
#[test]
fn an_answer_past_the_narrow_width_is_taken_wide() {
    let from_the_bottom = IntervalSet::between(Some(WideInt::Small(i128::MIN)), None);
    let to_the_top = IntervalSet::between(None, Some(WideInt::Small(i128::MAX)));
    for set in [from_the_bottom, to_the_top] {
        assert!(matches!(set.spans, Spans::Narrow(_)));
        let outside = set.complement();
        assert!(matches!(outside.spans, Spans::Wide(_)), "{outside:?}");
        assert!(!outside.is_empty());
        assert_eq!(outside.complement(), set);
    }
    let low = IntervalSet::between(
        Some(WideInt::Small(i128::MIN)),
        Some(WideInt::Small(i128::MIN + 8)),
    );
    let lifted = low.preimage(3, 2);
    assert_eq!(
        lifted,
        wide_preimage(&low.wide_spans().collect::<Vec<_>>(), 3, 2)
    );
    assert!(!lifted.is_empty());
}

/// `set` moved by `by`: each end shifted, at whatever width it lands.
fn moved(set: &IntervalSet, by: &BigInt) -> IntervalSet {
    let shift =
        |end: &Option<WideInt>| end.as_ref().map(|end| WideInt::from_big(end.to_big() + by));
    IntervalSet::of_wide(
        set.wide_spans()
            .map(|span| Span {
                lo: shift(&span.lo),
                hi: shift(&span.hi),
            })
            .collect(),
    )
}

/// The wide slot fits beside the narrow vector's capacity, so a set is the
/// three words one vector of spans is, as it was before the wide form: a set
/// copied, allocated in a table or dropped moves no more than it did.
#[test]
fn a_set_is_the_size_of_one_vector_of_spans() {
    assert_eq!(size_of::<IntervalSet>(), size_of::<Vec<super::Narrow>>());
}

/// A wide set holds the machine integers between its ends: a set reaching past
/// `i128` at one end and ending at a small integer at the other holds that
/// integer and none past it.
#[test]
fn a_wide_set_holds_the_machine_integers_between_its_ends() {
    let far = BigInt::from(1_i64) << 200_u32;
    let below = IntervalSet::between(
        Some(WideInt::from_big(-far.clone())),
        Some(WideInt::Small(5)),
    );
    assert!(matches!(below.spans, Spans::Wide(_)));
    assert!(below.holds(5) && below.holds(i64::MIN) && !below.holds(6));
    let above = IntervalSet::between(Some(WideInt::Small(3)), Some(WideInt::from_big(far)));
    assert!(above.holds(3) && above.holds(i64::MAX) && !above.holds(2));
}

/// A wide set releases the integers its ends share when it is dropped: the
/// slot that holds them is released by hand, and a set that kept them would
/// leak a `BigInt` a drop.
#[test]
fn a_wide_set_releases_its_ends_when_dropped() {
    let end = WideInt::from_big(BigInt::from(1_i64) << 200_u32);
    let WideInt::Big(shared) = &end else {
        unreachable!("an integer past i128 is held wide")
    };
    let before = Arc::strong_count(shared);
    let set = IntervalSet::between(None, Some(end.clone()));
    assert!(Arc::strong_count(shared) > before);
    drop(set);
    assert_eq!(Arc::strong_count(shared), before);
}
