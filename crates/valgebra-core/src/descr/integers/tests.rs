use super::{IntSet, MAX_PERIOD, gcd, lcm};
use proptest::prelude::*;

/// Union and meet inside the test corpus, where the bound is out of reach.
///
/// Every leaf's period is one or a step of at most five, so every period
/// these compose divides sixty. A refusal here would be a broken generator
/// rather than a law that failed, which is why it is spelled as one.
fn or(a: &IntSet, b: &IntSet) -> IntSet {
    a.union(b).expect("a period inside the bound")
}

fn and(a: &IntSet, b: &IntSet) -> IntSet {
    a.intersect(b).expect("a period inside the bound")
}

/// The outermost endpoint the generator writes. Past it every set it builds
/// is purely periodic, because union, intersection and complement move no
/// endpoint and add none.
const REACH: i64 = 9;

/// The integers that decide whether two sets are equal.
///
/// Past [`REACH`] both sets repeat with the period they carry, so agreeing
/// over one whole common period beyond it is agreeing everywhere. The
/// window has to be derived rather than fixed: a fixed one narrower than
/// the period reads a set whose only members lie past it as empty, which is
/// agreement on an accident.
fn window(a: &IntSet, b: &IntSet) -> core::ops::RangeInclusive<i64> {
    let period = lcm(a.modulus, b.modulus);
    // A period past the bound cannot be materialised, so it cannot be
    // compared either. Clamping keeps the window finite where the assertion
    // is compiled out; reaching it at all is the bug the assertion names.
    debug_assert!(period <= MAX_PERIOD, "a period of {period} past the bound");
    let edge = REACH.saturating_add(period.min(MAX_PERIOD));
    -edge..=edge
}

fn same(a: &IntSet, b: &IntSet) -> bool {
    window(a, b).all(|n| a.holds(n) == b.holds(n))
}

/// Sets built from bounded endpoints and small steps, so agreement on the
/// window is agreement everywhere.
fn int_set() -> impl Strategy<Value = IntSet> {
    let leaf = prop_oneof![
        Just(IntSet::empty()),
        Just(IntSet::all()),
        (-9i64..=9).prop_map(IntSet::just),
        (-9i64..=9).prop_map(|lo| IntSet::between(Some(lo), None)),
        (-9i64..=9).prop_map(|hi| IntSet::between(None, Some(hi))),
        (-9i64..=9, -9i64..=9).prop_map(|(a, b)| IntSet::between(Some(a.min(b)), Some(a.max(b)))),
        (1i64..=5).prop_map(|step| IntSet::multiple_of(step).expect("a small step")),
    ];
    leaf.prop_recursive(4, 24, 2, |inner| {
        prop_oneof![
            (inner.clone(), inner.clone()).prop_map(|(a, b)| or(&a, &b)),
            (inner.clone(), inner.clone()).prop_map(|(a, b)| and(&a, &b)),
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
    /// The Boolean algebra, checked against the integers.
    #[test]
    fn the_lattice_laws_hold_of_the_integers(
        a in int_set(),
        b in int_set(),
        c in int_set(),
    ) {
        prop_assert!(same(&or(&a, &b), &or(&b, &a)));
        prop_assert!(same(&and(&a, &b), &and(&b, &a)));
        prop_assert!(same(&or(&or(&a, &b), &c), &or(&a, &or(&b, &c))));
        prop_assert!(same(&and(&and(&a, &b), &c), &and(&a, &and(&b, &c))));
        prop_assert!(same(&or(&a, &and(&a, &b)), &a));
        prop_assert!(same(&and(&a, &or(&a, &b)), &a));
        prop_assert!(same(
            &and(&a, &or(&b, &c)),
            &or(&and(&a, &b), &and(&a, &c))
        ));
    }

    /// The complement laws, and De Morgan both ways.
    #[test]
    fn the_complement_laws_hold_of_the_integers(a in int_set(), b in int_set()) {
        prop_assert!(and(&a, &a.complement()).is_empty());
        prop_assert!(same(&or(&a, &a.complement()), &IntSet::all()));
        prop_assert!(same(&a.complement().complement(), &a));
        prop_assert!(same(
            &or(&a, &b).complement(),
            &and(&a.complement(), &b.complement())
        ));
        prop_assert!(same(
            &and(&a, &b).complement(),
            &or(&a.complement(), &b.complement())
        ));
    }

    /// Holding the same integers is being equal, across periods.
    ///
    /// The property the lifting exists for: two sets built with different
    /// steps are compared where their classes line up, so the multiples of
    /// two and the same set written with a period of four are one set.
    #[test]
    fn holding_the_same_integers_is_being_equal(a in int_set(), b in int_set()) {
        prop_assert_eq!(same(&a, &b), a == b);
    }

    /// Emptiness is a decision, which is what a bound conjunction needs: the
    /// structural procedure compares two bounds and declines what it cannot
    /// pair, while this answers from the set.
    #[test]
    fn emptiness_agrees_with_the_integers(a in int_set()) {
        prop_assert_eq!(a.is_empty(), window(&a, &a).all(|n| !a.holds(n)));
    }

    /// The order is total, and one position holds one set.
    ///
    /// Transitivity is the property no *pair* can see: an order read off a
    /// pair is antisymmetric while putting three sets in a cycle, because
    /// each pair is read at its own period. Three draws are what lets the
    /// law fail, and [`no_three_sets_are_ordered_in_a_cycle`] pins the
    /// witness so the seed does not decide.
    ///
    /// Only one direction of agreement with equality is a law here, and it
    /// is the one the order owes: two sets in one position are one set,
    /// because one table read at one period is one set. The converse is
    /// what the order gives up to be total, and
    /// [`two_spellings_of_one_set_are_one_set_in_two_places`] is where that
    /// is said.
    #[test]
    fn the_order_is_total(a in int_set(), b in int_set(), c in int_set()) {
        prop_assert_eq!(a.partial_cmp(&b), Some(a.cmp(&b)));
        prop_assert_eq!(a.cmp(&b), b.cmp(&a).reverse());
        if a.cmp(&b) == core::cmp::Ordering::Equal {
            prop_assert_eq!(&a, &b, "one position holds one set");
        }
        if a < b && b < c {
            prop_assert!(a < c, "the order takes a step it cannot take twice");
        }
    }

    /// Reading a set's table at a multiple of its period changes no
    /// integer.
    ///
    /// What every operation between two periods rests on: the two tables
    /// are combined residue by residue, so a lift that moved an integer
    /// would move it in the answer, and what equality rests on, which is
    /// why the set built back from the table is equal to the one it came
    /// from.
    #[test]
    fn a_table_at_a_multiple_period_holds_the_same_integers(
        a in int_set(),
        factor in 1i64..=6,
    ) {
        let modulus = a.modulus.saturating_mul(factor);
        // A period of at most sixty times six is inside the bound, so the
        // table is available for every draw; a refusal here would be the
        // generator, not the lift.
        let classes = a.table_at(modulus).expect("a period inside the bound");
        let lifted = IntSet { modulus, classes: classes.into_owned() };
        prop_assert!(same(&a, &lifted));
        prop_assert_eq!(&a, &lifted);
    }
}

/// A set whose members all lie past a period is not the empty set.
///
/// The window has to reach past the period for the laws above to mean what
/// they say. The positive multiples of sixty are the smallest thing the
/// generator builds that a narrower window reads as empty: three, four and
/// five meet at sixty, and the first member is the period itself.
#[test]
fn a_set_whose_members_start_past_a_period_is_seen() {
    let sixties = and(
        &and(&and(&step(3), &step(4)), &step(5)),
        &IntSet::between(Some(1), None),
    );

    assert!(
        sixties.holds(60) && !sixties.holds(0),
        "the positive multiples"
    );
    assert!(!sixties.is_empty());
    assert_ne!(sixties, IntSet::empty());
    assert!(!same(&sixties, &IntSet::empty()));
}

/// The multiples of a step inside the bound.
fn step(n: i64) -> IntSet {
    IntSet::multiple_of(n).expect("a small step is inside the bound")
}

/// A step is the thing intervals cannot express, and the reason the period
/// is carried at all.
#[test]
fn a_step_holds_its_multiples_and_nothing_between() {
    let evens = IntSet::multiple_of(2).expect("two is inside the bound");
    for n in -6i64..=6 {
        assert_eq!(evens.holds(n), n % 2 == 0, "{n}");
    }
    // The complement of a step is the other residues, which is again a step
    // set rather than a union of intervals.
    let odds = evens.complement();
    for n in -6i64..=6 {
        assert_eq!(odds.holds(n), n % 2 != 0, "{n}");
    }
    assert!(and(&evens, &odds).is_empty());
    assert_eq!(or(&evens, &odds), IntSet::all());

    // Two steps meet at their least common multiple, which is what a naive
    // pairwise rule over bounds cannot see.
    let sixes = and(&step(2), &step(3));
    for n in -12i64..=12 {
        assert_eq!(sixes.holds(n), n % 6 == 0, "{n}");
    }
    // And two steps that share no multiple but zero still meet there.
    assert!(!and(&step(2), &step(3)).is_empty());
}

/// A bound conjunction the structural procedure declines, decided here by
/// the set: no integer is both at least five and below one.
#[test]
fn a_bound_conjunction_that_cannot_hold_is_empty() {
    let low = IntSet::between(Some(5), None);
    let high = IntSet::between(None, Some(1));
    assert!(and(&low, &high).is_empty());
    // Adjacent bounds leave exactly the integers between them, and none is
    // the empty set rather than a negative-width range.
    assert!(
        and(
            &IntSet::between(Some(2), None),
            &IntSet::between(None, Some(1))
        )
        .is_empty()
    );
    assert_eq!(
        and(
            &IntSet::between(Some(1), None),
            &IntSet::between(None, Some(1))
        ),
        IntSet::just(1)
    );
    // An even integer strictly between two consecutive even numbers: the
    // step and the bounds together empty a set neither empties alone.
    let between = and(&step(2), &IntSet::between(Some(3), Some(3)));
    assert!(between.is_empty());
}

/// A step of zero divides nothing, and a negative step names the multiples
/// of its magnitude.
#[test]
fn a_degenerate_step_is_read_as_the_set_it_names() {
    assert_eq!(IntSet::multiple_of(0), Some(IntSet::just(0)));
    assert_eq!(IntSet::multiple_of(-3), IntSet::multiple_of(3));
    assert_eq!(IntSet::multiple_of(1), Some(IntSet::all()));
    // The magnitude of the smallest integer is not representable, and it is
    // read as the step that divides nothing rather than wrapping to itself.
    assert_eq!(IntSet::multiple_of(i64::MIN), Some(IntSet::just(0)));
}

// THEORY: the-carriers-are-i64-and-f64
/// A step past the bound has no representation, and the refusal says so
/// rather than substituting a set that is wrong in one direction and,
/// complemented, wrong in the other.
#[test]
fn a_step_past_the_period_bound_is_refused() {
    assert!(IntSet::multiple_of(MAX_PERIOD).is_some());
    assert!(IntSet::multiple_of(-MAX_PERIOD).is_some());
    assert!(IntSet::multiple_of(MAX_PERIOD + 1).is_none());
    assert!(IntSet::multiple_of(-(MAX_PERIOD + 1)).is_none());
    assert!(IntSet::multiple_of(i64::MAX).is_none());
}

/// Two steps inside the bound can meet past it, and the meet is refused.
///
/// The half of the bound that is easy to lose: `multiple_of` refuses a step
/// too large, but a caller writing two steps that are each far inside it --
/// 64 and 81 -- asks for the period they share, 5,184, which is not. There
/// is no set to substitute. A table built at the largest period this holds
/// describes the multiples of something else, and handing that back as the
/// answer is the wrong verdict this representation exists to avoid; so the
/// operation refuses, exactly as the automaton components do, and the
/// relation above stays undecided.
#[test]
fn two_steps_meeting_past_the_period_bound_are_refused() {
    for (a, b) in [(64, 81), (4093, 4096), (3, MAX_PERIOD), (63, 65)] {
        let (left, right) = (step(a), step(b));
        let shared = lcm(a, b);
        let past = shared > MAX_PERIOD;
        assert_eq!(
            left.intersect(&right).is_none(),
            past,
            "the meet of {a} and {b}, which share {shared}"
        );
        assert_eq!(
            left.union(&right).is_none(),
            past,
            "the join of {a} and {b}, which share {shared}"
        );
    }
}

/// A meet inside the bound still answers, which is what the refusal above
/// must not cost: 63 and 64 share 4,032, and their multiples are decided.
#[test]
fn two_steps_meeting_inside_the_period_bound_still_answer() {
    let met = and(&step(63), &step(64));
    assert!(met.holds(0) && met.holds(4032) && met.holds(-4032));
    assert!(!met.holds(63) && !met.holds(64) && !met.holds(4031));
    assert!(!met.is_empty());
}

/// A step that cancels leaves no step behind.
///
/// `a | !a` is the integers however `a` was written, and carrying `a`'s
/// period into the answer would leave two spellings of one set. Since two
/// spellings are two values, every step that cancels would put another
/// copy of the integers in a table of guards, and two such sets built from
/// *different* steps would never meet. Dropping a step nothing uses keeps
/// them at a period of one, which is one value.
#[test]
fn a_step_that_cancels_is_not_carried() {
    for n in [2, 64, 81, MAX_PERIOD] {
        let set = step(n);
        let whole = or(&set, &set.complement());
        assert_eq!(whole, IntSet::all(), "the join over {n}");
        assert_eq!(whole.modulus, 1, "the join over {n} keeps a period");
        let nothing = and(&set, &set.complement());
        assert_eq!(nothing, IntSet::empty(), "the meet over {n}");
        assert_eq!(nothing.modulus, 1, "the meet over {n} keeps a period");
    }
    // And so two of them, from steps whose periods meet past the bound,
    // are still one set rather than a pair equality cannot align.
    let from_wide = or(&step(64), &step(64).complement());
    let from_tall = or(&step(81), &step(81).complement());
    assert_eq!(from_wide, from_tall);
}

/// The two number-theoretic helpers, driven directly: the periods meet at
/// their least common multiple, and every other answer would either miss
/// integers or carry classes that cannot occur.
#[test]
fn the_periods_meet_at_their_least_common_multiple() {
    assert_eq!(gcd(12, 18), 6);
    assert_eq!(gcd(7, 1), 1);
    assert_eq!(gcd(5, 5), 5);
    assert_eq!(lcm(4, 6), 12);
    assert_eq!(lcm(3, 3), 3);
    assert_eq!(lcm(1, 7), 7);
    // Coprime periods multiply, which is the case that grows fastest.
    assert_eq!(lcm(3, 5), 15);
}

/// Two spellings of one set are one set in two places.
///
/// Where equality and the order part, said once so a reader who expects
/// them to agree finds the reason rather than a surprise: the multiples of
/// two and the same set written with a period of four are equal and take
/// two positions. The order pays for being total, and it pays in the one
/// place nothing is charged -- a table of guards holds a second row, and no
/// question about the integers is answered differently.
///
/// A pair whose periods meet past the bound is the other side of the same
/// split: the order still answers, because it reads no period, while
/// equality declines to call them one set.
#[test]
fn two_spellings_of_one_set_are_one_set_in_two_places() {
    let evens = IntSet::multiple_of(2).expect("a small step");
    let at_four = IntSet {
        modulus: 4,
        classes: evens
            .table_at(4)
            .expect("a period inside the bound")
            .into_owned(),
    };
    assert!(same(&evens, &at_four), "the two hold the same integers");
    assert_eq!(evens, at_four);
    assert_ne!(evens.cmp(&at_four), core::cmp::Ordering::Equal);

    let coarse = IntSet::multiple_of(MAX_PERIOD - 3).expect("a step within the bound");
    let fine = IntSet::multiple_of(MAX_PERIOD - 5).expect("a step within the bound");
    assert!(
        coarse.union(&fine).is_none(),
        "the periods meet past the bound"
    );
    assert_ne!(coarse, fine);
    assert_ne!(coarse.cmp(&fine), core::cmp::Ordering::Equal);
    assert_eq!(coarse, coarse.clone());
    assert_eq!(fine, fine.clone());
}

/// Equality is not transitive past the bound, and this is the triple.
///
/// Two sets are equal when their tables agree at the period they share, and
/// a pair whose periods meet past [`MAX_PERIOD`] has no such period and is
/// read as two sets. Chain two such readings and the relation is not an
/// equivalence: the evens spelled at 64 and at 162 both equal the evens
/// spelled at 2, and meet each other at 5,184.
///
/// Nothing here decides on it. Every use of equality between guards is a
/// fold -- a merge, a scan for a duplicate, a state signature -- so a pair
/// it misses costs a row or a coarser partition, never a different answer.
/// What it costs is the `Eq` contract, and a limit nothing states is a
/// limit the next reader has to rediscover.
///
/// Settling it needs a canonical spelling, and the module header says why
/// there is none to have in this representation. One exists for the
/// arithmetic sets in general -- the minimal automaton over digits -- and
/// adopting it is a decision about what the representation is, not a
/// repair to equality.
#[test]
fn equality_is_not_transitive_where_two_periods_cannot_meet() {
    let evens_at = |period: i64| {
        let evens = step(2);
        IntSet {
            modulus: period,
            classes: evens
                .table_at(period)
                .expect("a period inside the bound")
                .into_owned(),
        }
    };
    // 64 and 162 are each a multiple of two, and meet at 5,184.
    let (coarse, plain, fine) = (evens_at(64), step(2), evens_at(162));
    assert!(lcm(64, 162) > MAX_PERIOD, "the pair has no shared period");

    assert_eq!(coarse, plain);
    assert_eq!(plain, fine);
    assert_ne!(coarse, fine);

    // All three hold the same integers, which is what makes the gap a
    // limit of the reading rather than a disagreement about the sets.
    for n in -20i64..=20 {
        let held = coarse.holds(n);
        assert_eq!(held, plain.holds(n), "{n} in the coarse spelling");
        assert_eq!(held, fine.holds(n), "{n} in the fine spelling");
    }
}

/// No three sets are ordered in a cycle.
///
/// The witness a sort of record atoms found, pinned so no seed is asked to
/// find it twice: read at the period each pair shares, the integers come
/// before `{-2}`, `{-2}` comes before the multiples of three, and the
/// integers come *after* them. Nothing is wrong with any one of the three
/// readings; what is wrong is reading a pair, and an order on the tables
/// has no pair to read.
#[test]
fn no_three_sets_are_ordered_in_a_cycle() {
    let corpus = [
        IntSet::all(),
        IntSet::empty(),
        IntSet::just(-2),
        IntSet::between(Some(0), None),
        step(2),
        step(3),
        step(4),
        step(3).complement(),
    ];
    for x in &corpus {
        for y in &corpus {
            for z in &corpus {
                assert!(
                    !(x < y && y < z) || x < z,
                    "the order puts {x:?}, {y:?} and {z:?} in a cycle"
                );
            }
        }
    }
}
