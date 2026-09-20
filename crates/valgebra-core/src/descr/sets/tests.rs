use super::{MAX_LINES, SetLattice};
use crate::descr::Descr;
use crate::descr::budget;
use crate::descr::integers::IntSet;
use crate::descr::values::Values;
use crate::kind::Kind;
use crate::verdict::Verdict;
use proptest::prelude::*;
use std::sync::Arc;

/// A meet past the build's allowance refuses, and the same meet succeeds
/// under one that covers it.
///
/// The fourth of the four polarity lattices to be held to this, and the one
/// that was not charging: the bound above says how wide a result may be, and
/// the allowance says what reaching one may cost. A product is where a build
/// multiplies, so a lattice whose product does not charge is one a caller can
/// spend unbounded time in after every other lattice has refused.
#[test]
fn a_meet_past_the_allowance_refuses() {
    let x = SetLattice::of(IntSet::just(1));
    let y = SetLattice::of(IntSet::just(2));

    assert!(budget::under(0, || x.intersect(&y)).is_none());
    assert!(budget::under(64, || x.intersect(&y)).is_some());
}

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
    // into a run that outlasts a sweep, and an allowance per case so a
    // mutation that removes a pruning shortcut cannot either: see
    // `budget::law`.
    #![proptest_config(ProptestConfig {
        max_shrink_time: 2_000,
        ..ProptestConfig::default()
    })]

    // THEORY: the-second-decider
    /// A starved verdict is the third answer or the decided one, over drawn
    /// lattices.
    ///
    /// The unit beside this reaches the negated form by hand. This complements
    /// whatever the lattice draws without an allowance, and asks the emptiness
    /// of the result starved and decided: starved it may decline, and where it
    /// answers it answers as the decided build does. The boolean reading is
    /// the safe direction, so a starved proof of emptiness is a decided one.
    #[test]
    fn a_starved_verdict_is_the_third_answer_or_the_decided_one(a in lattice()) {
        let negated = budget::under(0, || a.complement());
        let decided = budget::under(4096, || negated.emptiness());
        let starved = budget::under(0, || negated.emptiness());
        prop_assert!(
            starved == Verdict::Unknown || starved == decided,
            "starved {:?} against decided {:?}",
            starved,
            decided
        );
        if budget::under(0, || negated.is_empty()) {
            prop_assert_eq!(decided, Verdict::Empty);
        }
    }

    // THEORY: lattice-theory, property-testing, each-kind-is-closed
    /// The Boolean algebra, checked against the sets rather than by equality
    /// of the forms, which a union of lines does not make canonical.
    #[test]
    fn the_lattice_laws_hold_of_the_sets(a in lattice(), b in lattice(), c in lattice()) {
        let _allowance = budget::law();
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
        let _allowance = budget::law();
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
        let _allowance = budget::law();
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
        let _allowance = budget::law();
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

// THEORY: the-second-decider, the-decision-has-three-answers
/// A negated form the allowance cannot expand is *unknown*, never inhabited.
///
/// The three-valued verdict exists for exactly this: a lattice in negated form
/// has to be turned positive before its emptiness can be read, and past the
/// allowance there is no union to read. Answering `Inhabited` there would be
/// the claim that some set satisfies it, standing on no witness -- and `Empty`
/// would be worse, since a proof of emptiness is what a caller is allowed to
/// act on. The arm returns the third answer, and nothing had asked it to.
///
/// The negated form itself has to be *reached*: complementing normalises back
/// to a positive union wherever the product fits, so a complement taken with
/// an allowance is not negated at all. One taken without an allowance is, and
/// it is the shape a caller holds after a build that ran out.
#[test]
fn a_negated_set_the_allowance_cannot_expand_declines() {
    let negated = budget::under(0, || SetLattice::of(IntSet::just(1)).complement());

    // Expanded under an allowance that covers it, the verdict is decided.
    let decided = budget::under(64, || negated.emptiness());
    assert_ne!(decided, Verdict::Unknown, "the row needs a decidable pair");

    // Without one, the same lattice declines rather than guessing either way.
    let starved = budget::under(0, || negated.emptiness());
    assert_eq!(starved, Verdict::Unknown, "starved gave {starved:?}");

    // And the boolean reading is the safe direction: not *proved* empty.
    assert!(!budget::under(0, || negated.is_empty()));
}

// THEORY: the-second-decider
/// The value guard beside the elements declines where its own meet does.
///
/// A line's exclusions are answered by asking whether one covers the other,
/// and that question is a difference whose emptiness may be unproved. Read as
/// "does not cover", an unproved difference keeps the line -- and a line kept
/// is a set reported inhabited on no evidence. The answer is `None`, which the
/// caller turns into the third verdict rather than into a decision.
///
/// The guard here is the descriptor, which is the one the tree carries: a
/// guard whose own meet cannot decline, as an interval cannot, never reaches
/// the arm however the allowance is set.
#[test]
fn a_covering_question_the_allowance_cannot_settle_answers_neither_way() {
    let ints = || Arc::new(Descr::of_kind(Kind::Int));
    let words = || Arc::new(Descr::of_kind(Kind::Str));
    let some = Values::Only(ints());
    let other = Values::Only(words());

    // The universe covers everything without asking anything, so it answers
    // under any allowance at all: the row beside the one that declines.
    assert_eq!(budget::under(0, || Values::Every.covers(&some)), Some(true));

    // A guard against a guard is the difference, and starved it is unread.
    let starved = budget::under(0, || other.covers(&some));
    assert_eq!(starved, None, "starved gave {starved:?}");

    // Given the allowance the same question is settled, which is what makes
    // the decline a decline rather than the only answer this pair has.
    assert_eq!(budget::under(4096, || other.covers(&some)), Some(false));
    assert_eq!(budget::under(4096, || some.covers(&some)), Some(true));
}

// THEORY: the-descriptor
/// A complement is expanded where the expansion is one product, and carried
/// under the flag past that.
///
/// Three widths, three claims. No lines complements into every set and every
/// set back into none, which is what keeps the cheap forms canonical: two
/// descriptors holding the same sets compare equal rather than differing by the
/// route each took. Two lines is a product of two complements, a width the meet
/// it is headed for would have pruned, so the lines are carried as they are and
/// the polarity says what they mean.
#[test]
fn a_complement_is_expanded_only_where_it_is_one_product() {
    let none: SetLattice<IntSet> = SetLattice::empty();
    let every: SetLattice<IntSet> = SetLattice::all();
    assert_eq!(
        none.complement(),
        every,
        "no lines are every set complemented"
    );
    assert_eq!(
        every.complement(),
        none,
        "and every set is none complemented"
    );

    let two = SetLattice::of(IntSet::just(0))
        .union(&SetLattice::of(IntSet::just(1)))
        .expect("two lines");
    let carried = two.complement();
    assert!(
        carried.negated,
        "two lines are carried rather than expanded"
    );
    assert_eq!(carried.lines, two.lines, "with the lines as they were");
    assert!(
        same(&carried.complement(), &two),
        "and the flag complements back into the union it carries"
    );
}

/// The bound reads the width of the union, never the number of pairs.
///
/// Twenty lines met with twenty is four hundred pairs, and the pairs are where
/// the count grows: it passes [`MAX_LINES`] long before the meet is done. The
/// union they collapse to is narrower than that, because the meets of unlike
/// lines are one line and a union holds it once. Refusing on the raw count
/// would make the bound a question about the order the factors were multiplied
/// in, which is a property of how a difference was written rather than of the
/// sets it names.
#[test]
fn a_meet_is_bounded_by_the_width_of_its_union_and_not_by_its_pairs() {
    let wide = (1..20i64)
        .try_fold(SetLattice::of(IntSet::just(0)), |left, n| {
            left.union(&SetLattice::of(IntSet::just(n)))
        })
        .expect("twenty lines");
    let met = wide
        .intersect(&wide)
        .expect("four hundred pairs, one union");
    assert!(
        wide.lines.len() * wide.lines.len() > MAX_LINES,
        "the row needs a pair count the bound would refuse"
    );
    assert!(
        met.lines.len() <= wide.lines.len() + 1,
        "and an answer the width of either side, plus the line the unlike meets share"
    );
    assert!(same(&met, &wide), "a meet with itself holds what it held");
}
