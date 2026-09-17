use super::{Lines, MAX_LINES};
use crate::descr::classes::Class;
use crate::descr::integers::IntSet;
use crate::descr::records::RecordLattice;
use crate::descr::{Component, Op, Whole};
use crate::kind::Kind;
use crate::verdict::Verdict;
use proptest::prelude::*;

/// The kind the laws are checked in.
///
/// One kind, because the laws are about the union of lines rather than about
/// which structure sits on one: an operation takes the `Whole` it is a part of
/// and every line it builds is of that kind. `Int` is the kind whose component
/// is an exact set of values, so membership is decidable and a law can be asked
/// of the values rather than of the forms -- which is the whole point, since two
/// spellings of one set are the shape the bounds produce.
const WHOLE: Whole = Whole::Kind(Kind::Int);

/// A class laying down its own layout, and one deriving from it.
///
/// Three classes and a derivation, because the object half is a lattice over
/// them: `dog` is an `animal` and `mineral` conflicts with both, so a line
/// carrying one of them is not a line carrying another.
fn animal() -> Class {
    Class::laid_out(1, 1)
}

fn dog() -> Class {
    Class::new(2, 1, std::slice::from_ref(&animal()))
}

fn mineral() -> Class {
    Class::laid_out(3, 3)
}

/// The values a law is checked over: an integer and what class carries it.
///
/// Both halves, because a line is a structure met with the objects it admits
/// and a law that moved a value from one line to another would hold on either
/// half read alone. The `None` column is the object with no class at all, which
/// is what every plain integer is.
fn universe() -> Vec<(i64, Option<Class>)> {
    let mut rows = Vec::new();
    for n in -2i64..=2 {
        rows.push((n, None));
        for class in [animal(), dog(), mineral()] {
            rows.push((n, Some(class)));
        }
    }
    rows
}

/// Whether the integer `n`, carried by an object of `class`, is in `lines`.
///
/// The two halves are asked of the same line, which is what
/// [`Lines::admits`](super::Lines::admits) is for; this is that call with the
/// `Int` kind's reading of a structure filled in.
fn holds(lines: &Lines, n: i64, class: Option<&Class>) -> bool {
    lines.admits(
        &|structure| match structure {
            Component::Integers(set) => set.holds(n),
            Component::Coarse(present) => *present,
            // Every structure this file builds is the `Int` kind's, and the
            // operations keep it there: a meet combines two of one kind and a
            // complement is a flip. Anything else is a line of another kind on
            // this kind's list, which is the bug this arm would hide.
            other => unreachable!("a line of another kind: {other:?}"),
        },
        &|objects| objects.holds(class, &[]),
    )
}

/// Whether two unions admit the same values, which is the equality a law is
/// asked in.
///
/// Not `==`: the bounds mean one set has several spellings -- a negated form
/// carries the lines it could not expand -- and comparing forms would fail a law
/// the representation keeps.
fn same(a: &Lines, b: &Lines) -> bool {
    universe()
        .iter()
        .all(|(n, class)| holds(a, *n, class.as_ref()) == holds(b, *n, class.as_ref()))
}

/// Unions of lines of the `Int` kind, built the way the descriptor builds them.
///
/// The leaves are the three constructors a caller reaches for -- the bottom, a
/// structure with no object constraint, an object constraint over the whole kind
/// -- and the branches are the operations, so a drawn value is a term the tree
/// can actually produce. A refusal past the bound draws the bottom rather than
/// unwinding: the laws below are asked of whatever was built, and a strategy
/// that panicked on a refusal would test the bound instead of the laws.
fn lines() -> impl Strategy<Value = Lines> {
    let structures = prop_oneof![
        Just(IntSet::all()),
        Just(IntSet::empty()),
        (-2i64..=2).prop_map(IntSet::just),
        (-2i64..=2).prop_map(|lo| IntSet::between(Some(lo), None)),
    ];
    let objects = prop_oneof![
        Just(RecordLattice::all()),
        Just(RecordLattice::instance_of(animal())),
        Just(RecordLattice::instance_of(dog())),
        Just(RecordLattice::instance_of(mineral())),
    ];
    let leaf = prop_oneof![
        Just(Lines::bottom()),
        structures.prop_map(|set| Lines::everything(Component::Integers(set))),
        objects.prop_map(|constraint| Lines::objects(&WHOLE.component(), constraint)),
    ];
    leaf.prop_recursive(3, 12, 2, |inner| {
        prop_oneof![
            (inner.clone(), inner.clone()).prop_map(|(a, b)| a
                .combine(&b, Op::Union, WHOLE)
                .unwrap_or_else(Lines::bottom)),
            (inner.clone(), inner.clone()).prop_map(|(a, b)| a
                .combine(&b, Op::Intersect, WHOLE)
                .unwrap_or_else(Lines::bottom)),
            inner.prop_map(|a| a.complement(WHOLE)),
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

    /// The Boolean algebra, checked against the values a union admits.
    ///
    /// The component the descriptor holds per kind is a set, and these are the
    /// laws that say so. Held against membership rather than by equality of the
    /// forms, because `tidy` merges two lines agreeing on their objects and the
    /// polarity carries what the product could not expand: the same set, spelled
    /// two ways, and a law over the spellings would refuse both.
    #[test]
    fn the_lattice_laws_hold_of_the_lines(
        a in lines(), b in lines(), c in lines()
    ) {
        if let (Some(ab), Some(ba)) =
            (a.combine(&b, Op::Union, WHOLE), b.combine(&a, Op::Union, WHOLE))
        {
            prop_assert!(same(&ab, &ba), "join commutes");
        }
        if let (Some(ab), Some(ba)) =
            (a.combine(&b, Op::Intersect, WHOLE), b.combine(&a, Op::Intersect, WHOLE))
        {
            prop_assert!(same(&ab, &ba), "meet commutes");
        }
        if let (Some(bc), Some(ab)) =
            (b.combine(&c, Op::Union, WHOLE), a.combine(&b, Op::Union, WHOLE))
            && let (Some(left), Some(right)) =
                (a.combine(&bc, Op::Union, WHOLE), ab.combine(&c, Op::Union, WHOLE))
        {
            prop_assert!(same(&left, &right), "join associates");
        }
        if let (Some(bc), Some(ac)) =
            (b.combine(&c, Op::Union, WHOLE), a.combine(&c, Op::Intersect, WHOLE))
            && let (Some(ab), Some(left)) =
                (a.combine(&b, Op::Intersect, WHOLE), a.combine(&bc, Op::Intersect, WHOLE))
            && let Some(right) = ab.combine(&ac, Op::Union, WHOLE)
        {
            prop_assert!(same(&left, &right), "meet distributes over join");
        }
    }

    /// The complement laws, and De Morgan both ways.
    ///
    /// The complement is total -- the [`Guard`](crate::descr::Guard) contract
    /// asks for one -- so there is no arm here that skips on a refusal. What a
    /// refusal changes is the spelling: the flag carries the set the product
    /// could not lay out, and these rows are what say the flag means the same
    /// thing the lines would have.
    #[test]
    fn the_complement_laws_hold_of_the_lines(a in lines(), b in lines()) {
        let not_a = a.complement(WHOLE);
        if let Some(met) = a.combine(&not_a, Op::Intersect, WHOLE) {
            prop_assert!(
                universe().iter().all(|(n, class)| !holds(&met, *n, class.as_ref())),
                "a value is in one of the two"
            );
        }
        if let Some(joined) = a.combine(&not_a, Op::Union, WHOLE) {
            prop_assert!(
                universe().iter().all(|(n, class)| holds(&joined, *n, class.as_ref())),
                "and in one of them"
            );
        }
        prop_assert!(same(&not_a.complement(WHOLE), &a), "twice is nothing");

        let de_morgan = b.complement(WHOLE);
        if let (Some(met), Some(joined)) = (
            a.combine(&b, Op::Intersect, WHOLE),
            not_a.combine(&de_morgan, Op::Union, WHOLE),
        ) {
            prop_assert!(same(&met.complement(WHOLE), &joined), "de Morgan, one way");
        }
        if let (Some(joined), Some(met)) = (
            a.combine(&b, Op::Union, WHOLE),
            not_a.combine(&de_morgan, Op::Intersect, WHOLE),
        ) {
            prop_assert!(same(&joined.complement(WHOLE), &met), "and the other");
        }
    }
}

/// A class the order cannot close leaves the kind's emptiness unknown.
///
/// The third verdict is what keeps the component sound where the open world
/// begins: an integer that is an instance of a class laying down another layout
/// exists only if some class derives from both, and which classes exist is not
/// something a snapshot of the order can say. Neither answer is proved, and
/// reading the line as empty would drop it -- a value reported absent on no
/// evidence, which is the one direction that is wrong rather than coarse.
#[test]
fn a_class_the_order_cannot_close_leaves_the_kind_unknown() {
    let classed = Lines::objects(&WHOLE.component(), RecordLattice::instance_of(animal()));
    assert_eq!(classed.emptiness(WHOLE), Verdict::Unknown);

    // The two proved answers, so the unknown above is a third rather than the
    // only one this component ever gives.
    assert_eq!(Lines::bottom().emptiness(WHOLE), Verdict::Empty);
    assert_eq!(
        Lines::everything(WHOLE.component()).emptiness(WHOLE),
        Verdict::Inhabited
    );
    assert_eq!(
        Lines::everything(Component::Integers(IntSet::empty())).emptiness(WHOLE),
        Verdict::Empty,
        "a structure holding nothing is a line the union drops"
    );
}

/// A union of lines never carries a line proved empty, whatever built it.
///
/// `Lines::of` states it and `tidy` keeps it, and the reason is equality: a
/// union carrying a line that contributes no value would stop equalling the same
/// union without it, so `a ∪ a` and `a` would be two sets. Held over the drawn
/// terms rather than at the constructors, because the operations are where a
/// line becomes empty.
#[test]
fn no_operation_leaves_a_line_that_holds_nothing() {
    let empty = Lines::everything(Component::Integers(IntSet::empty()));
    let some = Lines::everything(Component::Integers(IntSet::just(1)));
    let joined = some
        .combine(&empty, Op::Union, WHOLE)
        .expect("two lines join");
    assert!(same(&joined, &some), "the empty line adds nothing");
    assert_eq!(joined, some, "and is not carried");

    let met = some
        .combine(&some.complement(WHOLE), Op::Intersect, WHOLE)
        .expect("a line meets its complement");
    assert_eq!(
        met,
        Lines::bottom(),
        "a meet that holds nothing is the bottom"
    );
}

/// Past the bound the flag carries the set, and it carries the same one.
///
/// A complement is two lines per line and De Morgan multiplies them, so a union
/// whose lines constrain both halves doubles on every step and passes
/// [`MAX_LINES`]. There is then no union to return, and the contract says the
/// answer is total: the lines stay as they are under the flipped flag. That is
/// the second spelling the laws above are checked across, and this is where it
/// is shown to be reachable rather than assumed.
#[test]
fn a_complement_past_the_bound_keeps_its_values_under_the_flag() {
    let mut wide = Lines::bottom();
    for n in 0..9i64 {
        let line = Lines::objects(
            &Component::Integers(IntSet::just(n)),
            RecordLattice::instance_of(Class::laid_out(10 + u32::try_from(n).unwrap_or(0), 1)),
        );
        wide = wide
            .combine(&line, Op::Union, WHOLE)
            .expect("nine lines fit");
    }
    let flagged = wide.complement(WHOLE);
    assert!(
        flagged.negated,
        "nine two-sided lines complement past {MAX_LINES} lines"
    );
    for (n, class) in universe() {
        assert_eq!(
            holds(&flagged, n, class.as_ref()),
            !holds(&wide, n, class.as_ref()),
            "the flag admits what the union does not, at {n}"
        );
    }
    assert!(
        same(&flagged.complement(WHOLE), &wide),
        "and complementing it back is the union again"
    );
}
