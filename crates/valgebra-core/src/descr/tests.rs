use super::{BoolSet, Class, Component, Descr, Label, Lines, Op, Value, Verdict};
use crate::decision::Kind;
use crate::descr::budget;
use crate::descr::lines;
use crate::descr::records::RecordLattice;
use crate::descr::symbolic::{Edge, Guard};
use core::mem::size_of;
use proptest::prelude::*;
use std::sync::{Arc, LazyLock};

/// A guard is held by handle, so an edge pays a pointer for it rather than a
/// descriptor.
///
/// This is the property the nesting rests on. A `Descr` is one component per
/// kind stored inline, so it is the same size whatever it describes; an edge
/// holding one *by value* is that size again, and an edge's guard has its own
/// automaton with its own edges. Held that way the size multiplied through
/// the levels rather than adding, and a clone deep-copied every level --
/// which is what a product of two automata does to their guards, once per
/// pair of states.
///
/// Pinned by size because size is what regressed: a guard stored by value
/// again would pass every behavioural law in this file and bring the growth
/// back with it.
/// A guard met or joined with itself is that guard, and the handle settles
/// it without a walk or a third allocation.
///
/// The automaton's product asks both laws of every pair of states that reach
/// on the same letter, and after a clone the two sides are the same handle --
/// which is the case this exists for. Two handles onto equal sets are not the
/// same case: comparing them is the work being avoided, so the shortcut is
/// about identity and says nothing about equality.
#[test]
fn a_guard_met_with_itself_is_shared_rather_than_rebuilt() {
    let guard = Arc::new(Descr::of_kind(Kind::Int));
    let same = Arc::clone(&guard);
    let met = Guard::meet(&guard, &same).expect("a meet with itself");
    assert!(Arc::ptr_eq(&met, &guard));
    let joined = Guard::join(&guard, &same).expect("a join with itself");
    assert!(Arc::ptr_eq(&joined, &guard));

    // A separate handle onto the same set takes the long way, and arrives at
    // the same set.
    let twin = Arc::new(Descr::of_kind(Kind::Int));
    let met = Guard::meet(&guard, &twin).expect("a meet with its twin");
    assert!(!Arc::ptr_eq(&met, &guard));
    assert_eq!(*met, *guard);
}

#[test]
fn an_edge_holds_its_guard_by_handle() {
    assert_eq!(size_of::<Arc<Descr>>(), size_of::<usize>());
    assert!(
        size_of::<Edge<Arc<Descr>>>() * 16 < size_of::<Descr>(),
        "an edge costs {} bytes against a descriptor's {}, which is not a handle",
        size_of::<Edge<Arc<Descr>>>(),
        size_of::<Descr>()
    );
}

/// A meet past the build's allowance refuses, and the same meet succeeds
/// under one that covers it.
#[test]
fn a_meet_past_the_allowance_refuses() {
    let words = Descr::of_kind(Kind::Str);

    assert!(budget::under(0, || words.intersect(&words)).is_none());
    assert!(budget::under(64, || words.intersect(&words)).is_some());
}

/// The product of two kinds' lines is one of the places the allowance is
/// charged, and it is asked here rather than through a whole descriptor
/// meet.
///
/// A meet charges in several places, so any one of them refusing gives the
/// same answer and none of them is pinned by the descriptor-level test
/// above. This one calls the line union's own meet, where the product is
/// the only charge there is.
#[test]
fn a_line_product_past_the_allowance_refuses() {
    let whole = Component::top(Kind::Str);
    let lines = Lines::everything(whole.clone());
    let meet = || lines.combine(&lines, Op::Intersect, &whole);

    // One unit, not none: meeting two lines meets the objects under them
    // too, and that charges. An empty allowance would be refused by either
    // charge and so would not say which one is here.
    assert!(budget::under(1, meet).is_none());
    assert!(budget::under(8, meet).is_some());
}

/// A kind's union past the line bound refuses rather than holding a form it
/// cannot complement.
///
/// Asked of the lines directly, because no schema builds this: a kind
/// reaches the bound through lines that differ in their *objects*, and the
/// bound is what keeps the complement -- which doubles the count -- from
/// returning a set narrower than the one it was asked for.
#[test]
fn a_kind_past_its_line_bound_refuses() {
    let whole = Component::top(Kind::Dict);
    let line = |n: i64| {
        Lines::objects(
            &whole,
            RecordLattice::attribute("x", Arc::new(Descr::integer(n)), false),
        )
    };
    let mut wide = line(0);
    for n in 1..i64::try_from(lines::MAX_LINES).unwrap_or(i64::MAX) {
        wide = wide
            .combine(&line(n), Op::Union, &whole)
            .expect("inside the bound");
    }
    assert!(wide.combine(&line(-1), Op::Union, &whole).is_none());
}

/// Every value the descriptor can currently tell apart.
///
/// One per coarse kind, both booleans, and one of no listed kind. A law is
/// checked by asking every descriptor about every one of these, which is
/// what makes "these two sets are equal" a statement about *values* rather
/// than about the two representations agreeing with each other.
fn universe() -> Vec<Value> {
    let mut values: Vec<Value> = Kind::ALL
        .iter()
        .filter(|kind| {
            !matches!(
                kind,
                Kind::Bool
                    | Kind::Int
                    | Kind::Float
                    | Kind::Str
                    | Kind::Bytes
                    | Kind::List
                    | Kind::Tuple
                    | Kind::Set
                    | Kind::FrozenSet
                    | Kind::Dict
            )
        })
        .map(|kind| Value::of_kind(*kind))
        .collect();
    values.push(Value::boolean(true));
    values.push(Value::boolean(false));
    // Enough integers to separate every step and bound the generator uses:
    // a window narrower than the periods would agree by accident.
    values.extend((-14i64..=14).map(Value::integer));
    // The three floats that make the kind its own case, and a point strictly
    // inside every gap the generator's endpoints leave -- the unbounded ones
    // included, where a set like `(-inf, -1.0)` is inhabited by values no
    // endpoint names.
    values.extend(
        [
            f64::NEG_INFINITY,
            -2.0,
            -1.0,
            -0.5,
            -0.0,
            0.0,
            0.5,
            1.0,
            2.0,
            f64::INFINITY,
            f64::NAN,
        ]
        .map(Value::float),
    );
    // Words over the alphabet the generated patterns are written in, plus
    // one non-ASCII, for both word kinds.
    for kind in [Kind::Str, Kind::Bytes] {
        for word in [
            b"".as_slice(),
            b"a",
            b"b",
            b"c",
            b"ab",
            b"ba",
            b"aba",
            "\u{e9}".as_bytes(),
        ] {
            values.push(Value::word(word, kind));
        }
    }
    // Sequences over the letters the generated guards separate, for both
    // sequence kinds. The empty one and the two lengths are what tell a
    // chain from a loop: `tuple[int]` and `list[int]` agree on every
    // one-element sequence and part on the others.
    for kind in [Kind::List, Kind::Tuple] {
        values.extend(SEQUENCES.map(|elements| Value::sequence(elements, kind)));
    }
    // The same member lists read as sets, for both set kinds. The empty one
    // carries the weight here: it inhabits every powerset, so it is what
    // separates `set[nothing]` from `nothing`.
    for kind in [Kind::Set, Kind::FrozenSet] {
        values.extend(SEQUENCES.map(|members| Value::sequence(members, kind)));
    }
    // Dicts, as the entries they carry. The empty one separates a map that
    // constrains a key from one that forbids it; the two values under one
    // key separate a label's type from its neighbour's; and the integer key
    // is what tells one part of the key partition from another.
    values.extend(DICTS.map(Value::dict));
    // Objects of no listed kind, described by the attributes they carry.
    // The one with an attribute nobody names is what holds the record open.
    values.extend(OBJECTS.map(Value::object));
    // The same objects under each class of the little order below, plus the
    // one whose class nobody told us.
    for class in [&ANIMAL, &DOG, &MINERAL] {
        values.extend(OBJECTS.map(|attributes| Value::instance(class, attributes)));
    }
    // Values that have a kind *and* a class, which is what a line holds and
    // what the kindless slot could not describe. One per kind whose values
    // the components tell apart, so a law that reads only the structure and
    // a law that reads only the class both meet a value the other decides.
    for class in [&ANIMAL, &DOG] {
        for base in KINDED {
            values.push(base.of_class(class, &[]));
            values.push(base.of_class(class, CARRYING_A));
        }
    }
    values
}

/// A class order small enough to enumerate and wide enough to separate the
/// three answers: deriving, unrelated, and laid out apart.
static ANIMAL: LazyLock<Class> = LazyLock::new(|| Class::laid_out(1, 1));
static DOG: LazyLock<Class> = LazyLock::new(|| Class::new(2, 1, std::slice::from_ref(&ANIMAL)));
static MINERAL: LazyLock<Class> = LazyLock::new(|| Class::laid_out(3, 3));
/// Laid out like an animal and deriving from nothing: the pair whose meet
/// only a class outside the order could inhabit.
static UNRELATED: LazyLock<Class> = LazyLock::new(|| Class::new(4, 1, &[]));
/// A class whose instances are strings, which is what `Class::of_kind` says.
static SUBSTR: LazyLock<Class> = LazyLock::new(|| Class::new(5, 5, &[]).of_kind(Kind::Str));

/// The attribute lists the universe's objects carry.
const OBJECTS: [&[(&str, Value)]; 7] = [
    &[],
    &[("x", Value::integer(0))],
    &[("x", Value::integer(1))],
    &[("y", Value::integer(0))],
    &[("x", Value::integer(0)), ("y", Value::integer(1))],
    &[("x", Value::word(b"a", Kind::Str))],
    &[("z", Value::integer(0))],
];

/// The distinction the coarse component could not make.
///
/// `list[int]` and `list[str]` are one component while a kind is coarse, so
/// their meet is the whole kind rather than the one sequence they share.
/// With the automaton they are two languages over different letters, and
/// what they share is the empty list -- which both hold, and which is the
/// answer a coarse component cannot give.
#[test]
fn two_lists_of_different_elements_share_only_the_empty_one() {
    const NOTHING: &[Value] = &[];
    const INTS: &[Value] = &[Value::integer(1)];
    const WORDS: &[Value] = &[Value::word(b"a", Kind::Str)];

    let ints = Descr::sequence(&[], Some(&Descr::of_kind(Kind::Int)), Kind::List)
        .expect("list is a sequence kind");
    let words = Descr::sequence(&[], Some(&Descr::of_kind(Kind::Str)), Kind::List)
        .expect("list is a sequence kind");

    assert!(!ints.is_empty() && !words.is_empty());
    let shared = ints.intersect(&words).expect("two small automata");
    assert!(shared.admits(Value::sequence(NOTHING, Kind::List)));
    assert!(!shared.admits(Value::sequence(INTS, Kind::List)));
    assert!(!shared.admits(Value::sequence(WORDS, Kind::List)));
    assert!(ints.admits(Value::sequence(INTS, Kind::List)));
    assert!(words.admits(Value::sequence(WORDS, Kind::List)));
    assert!(!ints.admits(Value::sequence(WORDS, Kind::List)));
}

/// A chain is not a loop: `tuple[int]` holds one element and `list[int]`
/// holds any number, which is the length the prefix pins and the tail does
/// not.
#[test]
fn a_prefix_pins_the_length_and_a_tail_does_not() {
    const ONE: &[Value] = &[Value::integer(1)];
    const TWO: &[Value] = &[Value::integer(1), Value::integer(1)];

    let int = Descr::of_kind(Kind::Int);
    let pair = Descr::sequence(&[int.clone(), int.clone()], None, Kind::Tuple)
        .expect("tuple is a sequence kind");
    let many = Descr::sequence(&[], Some(&int), Kind::Tuple).expect("tuple is a sequence kind");

    assert!(!pair.admits(Value::sequence(ONE, Kind::Tuple)));
    assert!(pair.admits(Value::sequence(TWO, Kind::Tuple)));
    assert!(many.admits(Value::sequence(ONE, Kind::Tuple)));
    assert!(many.admits(Value::sequence(TWO, Kind::Tuple)));
}

/// The kind is what separates a list from a tuple, not the language: the
/// same shape under two kinds is two components, and they do not meet.
#[test]
fn the_same_shape_under_two_kinds_does_not_meet() {
    const ELEMENTS: &[Value] = &[Value::integer(1)];

    let int = Descr::of_kind(Kind::Int);
    let one = std::slice::from_ref(&int);
    let list = Descr::sequence(one, None, Kind::List).expect("a sequence kind");
    let tuple = Descr::sequence(one, None, Kind::Tuple).expect("a sequence kind");

    assert!(list.admits(Value::sequence(ELEMENTS, Kind::List)));
    assert!(!list.admits(Value::sequence(ELEMENTS, Kind::Tuple)));
    assert!(
        list.intersect(&tuple)
            .expect("two small automata")
            .is_empty()
    );
}

/// A letter is a descriptor, so a sequence of sequences is a sequence: the
/// recursion the component carries is the one the values have.
#[test]
fn a_sequence_of_sequences_reads_its_elements() {
    const INNER: &[Value] = &[Value::integer(1)];
    const OUTER: &[Value] = &[Value::sequence(INNER, Kind::List)];
    const FLAT: &[Value] = &[Value::integer(1)];

    let inner = Descr::sequence(&[], Some(&Descr::of_kind(Kind::Int)), Kind::List)
        .expect("a sequence kind");
    let outer = Descr::sequence(&[], Some(&inner), Kind::List).expect("a sequence kind");

    assert!(outer.admits(Value::sequence(OUTER, Kind::List)));
    assert!(!outer.admits(Value::sequence(FLAT, Kind::List)));
}

/// A word kind refuses the sequence constructor rather than building a set
/// over letters it has none of.
#[test]
fn a_kind_whose_values_are_not_sequences_refuses() {
    assert!(Descr::sequence(&[], None, Kind::Str).is_none());
    assert!(Descr::sequence(&[], None, Kind::Set).is_none());
}

proptest! {
    // The same bounds, for the same reasons.
    #![proptest_config(ProptestConfig {
        cases: 64,
        max_shrink_time: 2_000,
        ..ProptestConfig::default()
    })]

    /// The lattice laws, over descriptors that hold sets.
    #[test]
    fn the_lattice_laws_hold_of_the_sets(
        a in descr_with_sets(),
        b in descr_with_sets(),
        c in descr_with_sets(),
    ) {
        if let (Some(ab), Some(ba)) = (a.union(&b), b.union(&a)) {
            prop_assert!(agree_on_values(&ab, &ba), "join commutes");
        }
        if let (Some(ab), Some(ba)) = (a.intersect(&b), b.intersect(&a)) {
            prop_assert!(agree_on_values(&ab, &ba), "meet commutes");
        }
        if let (Some(bc), Some(ab)) = (b.union(&c), a.union(&b))
            && let (Some(left), Some(right)) = (a.union(&bc), ab.union(&c))
        {
            prop_assert!(agree_on_values(&left, &right), "join associates");
        }
        if let (Some(bc), Some(ac)) = (b.union(&c), a.intersect(&c))
            && let (Some(ab), Some(left)) = (a.intersect(&b), a.intersect(&bc))
            && let Some(right) = ab.union(&ac)
        {
            prop_assert!(agree_on_values(&left, &right), "meet distributes over join");
        }
    }

    /// The complement laws, over descriptors that hold sets.
    #[test]
    fn the_complement_laws_hold_of_the_sets(
        a in descr_with_sets(),
        b in descr_with_sets(),
    ) {
        let not_a = a.complement();
        if let Some(met) = a.intersect(&not_a) {
            prop_assert!(met.is_empty(), "a value is in one of the two");
        }
        if let Some(joined) = a.union(&not_a) {
            prop_assert!(
                agree_on_values(&joined, &Descr::anything()),
                "and in one of them"
            );
        }
        prop_assert!(
            agree_on_values(&not_a.complement(), &a),
            "twice is nothing"
        );
        let not_b = b.complement();
        if let (Some(joined), Some(met)) = (a.union(&b), not_a.intersect(&not_b)) {
            prop_assert!(
                agree_on_values(&joined.complement(), &met),
                "de Morgan one way"
            );
        }
        if let (Some(met), Some(joined)) = (a.intersect(&b), not_a.union(&not_b)) {
            prop_assert!(
                agree_on_values(&met.complement(), &joined),
                "and the other"
            );
        }
    }

    /// A verdict is a claim about the values, and each of the three says
    /// something different.
    ///
    /// `Empty` is a proof that no value is admitted, so the universe must
    /// hold none. `Inhabited` is a proof that one is, and while the universe
    /// cannot always exhibit it -- a set may live entirely outside a finite
    /// list -- a value it *does* admit forbids `Empty`. `Unknown` claims
    /// nothing and so cannot be contradicted.
    #[test]
    fn a_verdict_is_a_claim_about_the_values(a in descr_with_sets()) {
        let admitted = universe().into_iter().filter(|v| a.admits(*v)).count();
        match a.emptiness() {
            Verdict::Empty => prop_assert_eq!(admitted, 0, "empty admits nothing"),
            Verdict::Inhabited | Verdict::Unknown => {}
        }
        if admitted > 0 {
            prop_assert_ne!(a.emptiness(), Verdict::Empty, "a value forbids empty");
        }
    }

    /// Subtyping reduces to emptiness, and the reduction is sound.
    ///
    /// The property the whole representation is for: proving `a ∧ ¬b` empty
    /// is proving that every value of `a` is a value of `b`. The converse is
    /// not asked -- failing to prove it is not a proof that a value escapes.
    #[test]
    fn an_empty_difference_is_containment(a in descr_with_sets(), b in descr_with_sets()) {
        let Some(difference) = a.intersect(&b.complement()) else {
            return Ok(());
        };
        if difference.emptiness() == Verdict::Empty {
            for value in universe() {
                prop_assert!(
                    !a.admits(value) || b.admits(value),
                    "an empty difference leaves no value of a outside b"
                );
            }
        }
    }

    /// A value the descriptor admits is a value its complement does not,
    /// with the sets in.
    #[test]
    fn a_complement_holds_every_set_the_descriptor_does_not(a in descr_with_sets()) {
        let complement = a.complement();
        for value in universe() {
            prop_assert_ne!(a.admits(value), complement.admits(value));
        }
    }
}

/// The distinction row 9 of the report asks for.
///
/// `set[int] & set[str]` is not empty and is not either operand: a meet of
/// two powersets is the powerset of the meet, so it holds exactly the sets
/// drawn from `int ∧ str` -- which is the empty set and nothing else.
#[test]
fn two_sets_of_disjoint_elements_meet_in_the_empty_set() {
    const NOTHING: &[Value] = &[];
    const ONE: &[Value] = &[Value::integer(1)];

    let ints = Descr::set(&Descr::of_kind(Kind::Int), Kind::Set).expect("a set kind");
    let words = Descr::set(&Descr::of_kind(Kind::Str), Kind::Set).expect("a set kind");
    let none = Descr::set(&Descr::nothing(), Kind::Set).expect("a set kind");

    let shared = ints.intersect(&words).expect("two small powersets");
    assert!(!shared.is_empty(), "the empty set is drawn from both");
    assert_eq!(shared, none, "and it is the only one");
    assert!(shared.admits(Value::sequence(NOTHING, Kind::Set)));
    assert!(!shared.admits(Value::sequence(ONE, Kind::Set)));
}

/// The distinction row 22 asks for: a set of an unhashable element holds
/// only the empty set, because no value of that kind can be a member.
#[test]
fn a_set_of_an_unhashable_element_is_the_set_of_nothing() {
    const NOTHING: &[Value] = &[];
    const A_LIST: &[Value] = &[Value::sequence(&[], Kind::List)];

    let lists = Descr::sequence(&[], Some(&Descr::of_kind(Kind::Int)), Kind::List)
        .expect("a sequence kind");
    let of_lists = Descr::set(&lists, Kind::Set).expect("a set kind");
    let none = Descr::set(&Descr::nothing(), Kind::Set).expect("a set kind");

    assert!(!of_lists.is_empty(), "the empty set is still a set");
    assert_eq!(of_lists, none);
    assert!(of_lists.admits(Value::sequence(NOTHING, Kind::Set)));
    assert!(!of_lists.admits(Value::sequence(A_LIST, Kind::Set)));
}

/// A frozenset is hashable and a set is not, which is the one place the two
/// set kinds differ.
#[test]
fn a_frozenset_is_a_member_and_a_set_is_not() {
    let frozen = Descr::set(&Descr::of_kind(Kind::Int), Kind::FrozenSet).expect("a set kind");
    let mutable = Descr::set(&Descr::of_kind(Kind::Int), Kind::Set).expect("a set kind");
    let none = Descr::set(&Descr::nothing(), Kind::Set).expect("a set kind");

    let of_frozen = Descr::set(&frozen, Kind::Set).expect("a set kind");
    assert!(
        !of_frozen.is_empty(),
        "a set of frozensets holds more than the empty set"
    );
    assert_ne!(of_frozen, none);
    assert_eq!(Descr::set(&mutable, Kind::Set).expect("a set kind"), none);
}

/// The two set kinds are separate components, so a set is never a frozenset.
#[test]
fn the_two_set_kinds_do_not_meet() {
    let mutable = Descr::set(&Descr::of_kind(Kind::Int), Kind::Set).expect("a set kind");
    let frozen = Descr::set(&Descr::of_kind(Kind::Int), Kind::FrozenSet).expect("a set kind");

    assert!(
        mutable
            .intersect(&frozen)
            .expect("two small powersets")
            .is_empty()
    );
}

/// A kind whose values are not sets refuses the constructor.
#[test]
fn a_kind_whose_values_are_not_sets_refuses() {
    assert!(Descr::set(&Descr::nothing(), Kind::List).is_none());
    assert!(Descr::set(&Descr::nothing(), Kind::Str).is_none());
}

/// The third answer, and where it comes from.
///
/// Two classes a value must both be an instance of, neither deriving from
/// the other and neither laid out apart, are satisfied only by a class
/// deriving from both. Whether one exists is not something a snapshot of the
/// order can say, so the verdict says so rather than guessing.
#[test]
fn two_unrelated_classes_meet_in_an_unknown() {
    let met = Descr::instance_of(ANIMAL.clone())
        .intersect(&Descr::instance_of(UNRELATED.clone()))
        .expect("two small atoms");

    assert_eq!(met.emptiness(), Verdict::Unknown);
    assert!(!met.is_empty(), "unknown is not a proof of emptiness");
}

/// The other two answers are proofs, and a proof beats the open world.
#[test]
fn deriving_and_layout_are_proofs_either_way() {
    let dogs = Descr::instance_of(DOG.clone());

    // Laid out apart: proved empty, whatever classes exist.
    assert_eq!(
        dogs.intersect(&Descr::instance_of(MINERAL.clone()))
            .expect("two small atoms")
            .emptiness(),
        Verdict::Empty
    );
    // Deriving: proved inhabited, because the class itself is a witness.
    assert_eq!(
        dogs.intersect(&Descr::instance_of(ANIMAL.clone()))
            .expect("two small atoms")
            .emptiness(),
        Verdict::Inhabited
    );
}

/// An unknown in one kind does not decide the descriptor: a component proved
/// inhabited settles the union whatever the others cannot say.
#[test]
fn a_proof_in_one_kind_outranks_an_unknown_in_another() {
    let unknown = Descr::instance_of(ANIMAL.clone())
        .intersect(&Descr::instance_of(UNRELATED.clone()))
        .expect("two small atoms");
    assert_eq!(unknown.emptiness(), Verdict::Unknown);

    let joined = unknown
        .union(&Descr::of_kind(Kind::Int))
        .expect("a small union");
    assert_eq!(joined.emptiness(), Verdict::Inhabited);
}

/// A class is a set, and so is its complement -- which is what the tree
/// could not say about an instance schema.
#[test]
fn a_class_and_its_complement_are_both_descriptors() {
    let dogs = Descr::instance_of(DOG.clone());
    let animals = Descr::instance_of(ANIMAL.clone());

    assert!(dogs.admits(Value::instance(&DOG, &[])));
    assert!(!dogs.admits(Value::instance(&ANIMAL, &[])));
    assert!(!dogs.admits(Value::other()), "no class is not this class");

    // A dog is an animal, so meeting the two changes nothing and meeting the
    // complement leaves nothing.
    assert!(
        dogs.intersect(&animals)
            .expect("two small atoms")
            .admits(Value::instance(&DOG, &[]))
    );
    assert!(
        dogs.intersect(&animals.complement())
            .expect("two small atoms")
            .is_empty()
    );
}

/// The world stays open: excluding a class nothing derives from leaves the
/// set inhabited, because the class list is never complete.
#[test]
fn excluding_an_unrelated_class_decides_nothing() {
    let animals = Descr::instance_of(ANIMAL.clone());
    let not_mineral = Descr::instance_of(MINERAL.clone()).complement();

    let met = animals.intersect(&not_mineral).expect("two small atoms");
    assert!(!met.is_empty());
    assert!(met.admits(Value::instance(&ANIMAL, &[])));
}

/// A class and an attribute constrain one value, which is why they share an
/// atom rather than sitting in two slots a complement would have to split.
#[test]
fn a_class_meets_a_builtin_kind() {
    let dog = Descr::instance_of(DOG.clone());
    let ints = Descr::of_kind(Kind::Int);
    let dog_int = Value::integer(1).of_class(&DOG, &[]);

    // A class constrains a value within its kind, so the two meet in the
    // values that are both -- which is a set the kindless slot could not
    // name, and which every question about a dataclass deriving from `int`
    // is asked of.
    let both = dog.intersect(&ints).expect("an int that is a Dog");
    assert!(both.admits(dog_int));
    assert_eq!(both.emptiness(), Verdict::Inhabited);

    // And it is *both*, not either: an integer nobody gave a class to is
    // outside it, and so is a Dog of no listed kind.
    assert!(!both.admits(Value::integer(1)));
    assert!(!both.admits(Value::instance(&DOG, &[])));

    // Each half on its own still admits what it always did.
    assert!(dog.admits(dog_int) && dog.admits(Value::instance(&DOG, &[])));
    assert!(ints.admits(dog_int) && ints.admits(Value::integer(1)));

    // The order the classes stand in is read through the kind, not around
    // it: a Dog is an Animal whatever kind the value also has.
    let animal_int = Descr::instance_of(ANIMAL.clone())
        .intersect(&ints)
        .expect("an int that is an Animal");
    assert!(animal_int.admits(dog_int));
    assert!(!animal_int.admits(Value::integer(1).of_class(&MINERAL, &[])));
}

/// A class that confines its instances to a kind stands on that kind's line
/// alone; one that confines none stands on every line.
///
/// `Class::of_kind` is the only thing that separates the two, and the pair
/// below is what it buys: `MyStr <= str` is decided here, and it is decided
/// because the class said which kind its instances have. Without that the
/// class would keep the open world's answer -- a subclass may lay down any
/// layout, so it might be an integer -- and no meet with a kind would ever be
/// a proof.
#[test]
fn a_class_confined_to_a_kind_is_on_that_kind_alone() {
    let words = Descr::instance_of(SUBSTR.clone());
    let ints = Descr::of_kind(Kind::Int);

    // A value of the class that is a string is one of its instances.
    assert!(words.admits(Value::word(b"x", Kind::Str).of_class(&SUBSTR, &[])));
    // The same class carried by a value of another kind is not, because the
    // class named the kind its instances have.
    assert!(!words.admits(Value::integer(1).of_class(&SUBSTR, &[])));

    // So the meet with another kind is *proved* empty rather than left open.
    assert_eq!(
        words.intersect(&ints).expect("two small atoms").emptiness(),
        Verdict::Empty
    );
    // And the class that confines nothing keeps the open world's answer, which
    // is the contrast the confinement is for.
    assert!(
        !Descr::instance_of(ANIMAL.clone())
            .intersect(&ints)
            .expect("two small atoms")
            .is_empty(),
        "an Animal may yet be laid out as an int"
    );
}

/// An attribute constrains a value of a listed kind, for the same reason a
/// class does: it is a narrowing *within* the kind.
#[test]
fn an_attribute_meets_a_builtin_kind() {
    let carrying = Descr::attribute("a", &Descr::of_kind(Kind::Int), false);
    let ints = Descr::of_kind(Kind::Int);
    let both = ints.intersect(&carrying).expect("an int carrying `a`");
    assert!(both.admits(Value::integer(1).of_class(&DOG, CARRYING_A)));
    assert!(!both.admits(Value::integer(1)));
    // The complement is taken inside the kind too: an integer carrying no
    // `a` is outside the meet and inside its complement.
    assert!(both.complement().admits(Value::integer(1)));
}

#[test]
fn a_class_meets_an_attribute_in_one_object() {
    const NAMED: &[(&str, Value)] = &[("x", Value::integer(0))];

    let met = Descr::instance_of(DOG.clone())
        .intersect(&Descr::attribute("x", &Descr::of_kind(Kind::Int), false))
        .expect("two small atoms");

    assert!(met.admits(Value::instance(&DOG, NAMED)));
    assert!(!met.admits(Value::instance(&DOG, &[])));
    assert!(
        !met.admits(Value::object(NAMED)),
        "the class is required too"
    );
}

/// The record the report calls carrier-free: a set of values fixed by the
/// attributes alone, with no class in it.
///
/// A dataclass `D(x: int)` is the meet of a class and this record -- which is
/// what the IR now spells, `Instance(D) ∧ AttrRecord` -- and the record on
/// its own is the half that makes the complement representable, because the
/// complement of an attribute constraint is another one.
#[test]
fn an_attribute_record_is_a_set_without_a_class_in_it() {
    const HAS_INT: &[(&str, Value)] = &[("x", Value::integer(1))];
    const HAS_WORD: &[(&str, Value)] = &[("x", Value::word(b"a", Kind::Str))];
    const HAS_NEITHER: &[(&str, Value)] = &[("y", Value::integer(1))];

    let with_int = Descr::attribute("x", &Descr::of_kind(Kind::Int), false);

    assert!(!with_int.is_empty());
    assert!(with_int.admits(Value::object(HAS_INT)));
    assert!(!with_int.admits(Value::object(HAS_WORD)));
    assert!(!with_int.admits(Value::object(HAS_NEITHER)));

    // The complement is a set of the same kind rather than an absence of
    // one, which is what the tree could not represent.
    let without_int = with_int.complement();
    assert!(without_int.admits(Value::object(HAS_WORD)));
    assert!(without_int.admits(Value::object(HAS_NEITHER)));
    assert!(!without_int.admits(Value::object(HAS_INT)));
}

/// The record is **open**: it constrains what it names and nothing else.
///
/// The only sound reading for a Python object, which may carry attributes no
/// schema mentions -- and the reason a complement stays finite, since an
/// attribute nobody named cannot make an object fail.
#[test]
fn a_record_constrains_the_attributes_it_names_and_no_others() {
    const NAMED: &[(&str, Value)] = &[("x", Value::integer(1))];
    const AND_MORE: &[(&str, Value)] = &[("x", Value::integer(1)), ("z", Value::integer(9))];

    let with_int = Descr::attribute("x", &Descr::of_kind(Kind::Int), false);

    assert!(with_int.admits(Value::object(NAMED)));
    assert!(with_int.admits(Value::object(AND_MORE)));
}

/// Meeting two records over different attributes is the object carrying
/// both, which is formula (12) pointwise.
#[test]
fn two_attributes_meet_rather_than_conflict() {
    const BOTH: &[(&str, Value)] = &[("x", Value::integer(1)), ("y", Value::integer(0))];
    const ONE: &[(&str, Value)] = &[("x", Value::integer(1))];

    let met = Descr::attribute("x", &Descr::of_kind(Kind::Int), false)
        .intersect(&Descr::attribute("y", &Descr::of_kind(Kind::Int), false))
        .expect("two small records");

    assert!(!met.is_empty());
    assert!(met.admits(Value::object(BOTH)));
    assert!(!met.admits(Value::object(ONE)));
}

/// An attribute required to hold nothing admits no object at all, which is
/// emptiness (11); an optional one still admits the object without it.
#[test]
fn a_required_attribute_of_no_values_is_empty_and_an_optional_one_is_not() {
    const NOTHING: &[(&str, Value)] = &[];

    let required = Descr::attribute("x", &Descr::nothing(), false);
    let optional = Descr::attribute("x", &Descr::nothing(), true);

    assert!(required.is_empty());
    assert!(!optional.is_empty());
    assert!(optional.admits(Value::object(NOTHING)));
    assert_eq!(optional, Descr::without_attribute("x"));
}

/// An object is of no listed kind, so a record never admits one that is.
#[test]
fn a_record_admits_no_value_of_a_listed_kind() {
    let with_int = Descr::attribute("x", &Descr::of_kind(Kind::Int), false);

    assert!(!with_int.admits(Value::integer(1)));
    assert!(!with_int.admits(Value::sequence(&[], Kind::List)));
}

/// A loop guarded by every value *is* the set of every sequence, and has to
/// be the same descriptor.
///
/// Two spellings of one table, and the reason the row carries an else edge
/// rather than a guard for the rest: a guard that leaves nothing leaves the
/// else edge dead, and a dead else edge is not written down. Without that,
/// two equal languages compare unequal and the lattice laws fail on a
/// difference that is not one.
#[test]
fn a_loop_on_every_value_is_the_set_of_every_sequence() {
    let anything = Descr::anything();
    for kind in [Kind::List, Kind::Tuple] {
        let looped = Descr::sequence(&[], Some(&anything), kind).expect("a sequence kind");
        assert_eq!(looped, Descr::of_kind(kind), "{kind:?}");
    }
}

/// One integer, as the elements of a sequence or the members of a set.
const ONE_INT: &[Value] = &[Value::integer(1)];

/// One value of each kind whose values the components tell apart, for the
/// universe to put a class and an attribute beside.
const KINDED: [Value; 6] = [
    Value::boolean(true),
    Value::integer(1),
    Value::float(1.0),
    Value::word(b"a", Kind::Str),
    Value::sequence(ONE_INT, Kind::List),
    Value::sequence(ONE_INT, Kind::Set),
];

/// One attribute, for a value that carries one beside its kind.
const CARRYING_A: &[(&str, Value)] = &[("a", Value::integer(1))];

/// Dicts the laws are asked about, as their entries.
const DICTS: [&[(Value, Value)]; 8] = [
    &[],
    &[(Value::word(b"a", Kind::Str), Value::integer(1))],
    &[(Value::word(b"a", Kind::Str), Value::word(b"x", Kind::Str))],
    &[(Value::word(b"b", Kind::Str), Value::integer(1))],
    &[
        (Value::word(b"a", Kind::Str), Value::integer(1)),
        (Value::word(b"b", Kind::Str), Value::integer(1)),
    ],
    &[(Value::integer(1), Value::integer(1))],
    &[(Value::integer(1), Value::word(b"x", Kind::Str))],
    // A key of no listed kind: an object whose class defines `__hash__`. It
    // is the part of the key partition that has no kind, and a default that
    // did not cover it would leave the dict governed by nothing.
    &[(Value::other(), Value::integer(1))],
];

/// The element sequences the universe is built from.
const SEQUENCES: [&[Value]; 6] = [
    &[],
    &[Value::integer(0)],
    &[Value::integer(1)],
    &[Value::word(b"a", Kind::Str)],
    &[Value::integer(0), Value::integer(1)],
    &[Value::integer(0), Value::integer(0), Value::integer(0)],
];

/// Whether two descriptors agree about every value in the universe.
///
/// The direction enumeration can support. Equality is stronger: two regular
/// languages agree exactly when they agree on every word shorter than the
/// product of their state counts, which no universe can list -- so the full
/// canonicity claim is held in each component's own module, where the word
/// component checks it against the emptiness decision instead.
fn agree_on_values(a: &Descr, b: &Descr) -> bool {
    universe().into_iter().all(|v| a.admits(v) == b.admits(v))
}

/// The word descriptors the generator draws from, built once.
///
/// A pattern's automaton is determinised and minimised, which is far more
/// work than drawing a number: built per draw it dominates the suite, and
/// proptest's shrinking -- thousands of draws over one failure -- turns a
/// caught mutation into a run that does not finish.
static WORD_SETS: LazyLock<Vec<Descr>> = LazyLock::new(|| {
    ["a", "b", "ab?", "[ab]+"]
        .iter()
        .filter_map(|pattern| Descr::pattern(pattern, Kind::Str))
        .collect()
});

/// Descriptors whose every component is canonical.
///
/// No sequence, set or object: those three are held as a union or as a
/// table a guard's own bound can leave coarse, so two of them can admit the
/// same values and compare unequal. Keeping them out is what lets the laws
/// below be checked *by equality*, which is stronger than any universe can
/// be -- two regular languages agree exactly when they agree on every word
/// shorter than the product of their state counts, and no list of values
/// says that. [`descr_with_sets`] puts them back and checks the same laws
/// against the values.
fn descr() -> impl Strategy<Value = Descr> {
    let leaf = prop_oneof![
        Just(Descr::nothing()),
        Just(Descr::anything()),
        (0..Kind::ALL.len())
            .prop_map(|i| Descr::of_kind(Kind::ALL.get(i).copied().unwrap_or(Kind::Int))),
        proptest::bool::ANY.prop_map(Descr::boolean),
        (-9i64..=9).prop_map(Descr::integer),
        (1i64..=5).prop_map(|step| Descr::multiple_of(step).expect("a small step")),
        prop_oneof![Just(-1.0f64), Just(0.0), Just(1.0), Just(f64::NAN)].prop_map(Descr::float),
        (0..WORD_SETS.len())
            .prop_map(|i| { WORD_SETS.get(i).cloned().unwrap_or_else(Descr::nothing) }),
    ];
    leaf.prop_recursive(3, 16, 2, |inner| {
        prop_oneof![
            (inner.clone(), inner.clone())
                .prop_map(|(a, b)| a.union(&b).unwrap_or_else(Descr::anything)),
            (inner.clone(), inner.clone())
                .prop_map(|(a, b)| a.intersect(&b).unwrap_or_else(Descr::nothing)),
            inner.prop_map(|a| a.complement()),
        ]
    })
}

/// The same descriptors, plus the sequences, sets, maps and objects the four
/// non-canonical components hold.
///
/// A second generator rather than a branch of the first, because those three
/// components are the ones that are **not canonical**: a union of powerset lines
/// can hold the same sets two ways. Every law below is therefore checked
/// against the values rather than by equality of the forms, which would fail
/// on a difference that is not a difference. The generator above stays free
/// of sets so the laws that *can* be checked at full strength still are.
fn descr_with_sets() -> impl Strategy<Value = Descr> {
    let leaf = prop_oneof![
        4 => descr(),
        2 => (descr(), prop_oneof![Just(Kind::Set), Just(Kind::FrozenSet)])
            .prop_map(|(elements, kind)| {
                Descr::set(&elements, kind).unwrap_or_else(Descr::nothing)
            }),
        2 => (
            proptest::collection::vec(descr(), 0..=2),
            proptest::option::of(descr()),
            prop_oneof![Just(Kind::List), Just(Kind::Tuple)],
        )
            .prop_map(|(prefix, tail, kind)| {
                Descr::sequence(&prefix, tail.as_ref(), kind).unwrap_or_else(Descr::nothing)
            }),
        2 => (
            prop_oneof![Just("x"), Just("y")],
            descr(),
            proptest::bool::ANY,
        )
            .prop_map(|(label, ty, optional)| Descr::attribute(label, &ty, optional)),
        1 => prop_oneof![Just("x"), Just("y")].prop_map(Descr::without_attribute),
        2 => (
            prop_oneof![Just("a"), Just("b")],
            descr(),
            proptest::bool::ANY,
        )
            .prop_map(|(label, ty, optional)| Descr::label(Label::str(label), &ty, optional)),
        2 => (
            prop_oneof![Just(Kind::Str), Just(Kind::Int)],
            descr(),
        )
            .prop_map(|(kind, ty)| Descr::mapping(kind, &ty)),
        1 => prop_oneof![Just(Kind::Str), Just(Kind::Int)]
            .prop_map(|kind| Descr::keys_among(&[Some(kind)])),
        // A record: named keys, and a *closed* rest. The shape the two above
        // cannot reach between them -- a label leaves every other key free
        // and `keys_among` names no key -- and the one whose complement is
        // hardest, because closing is what puts a negative constraint on
        // every part at once. A record complemented twice came back
        // forbidding the key it is about, and no law here saw it in twenty
        // thousand draws until this leaf existed.
        3 => (
            proptest::collection::vec(
                (prop_oneof![Just("a"), Just("b")], descr(), proptest::bool::ANY),
                0..=2,
            ),
            proptest::option::of((prop_oneof![Just(Kind::Str), Just(Kind::Int)], descr())),
        )
            .prop_map(|(labels, opened)| {
                Descr::keyed_map(
                    labels
                        .into_iter()
                        .map(|(name, ty, optional)| (Label::str(name), ty, optional)),
                    opened.map(|(kind, ty)| (Some(kind), ty)),
                )
                .unwrap_or_else(Descr::nothing)
            }),
        1 => prop_oneof![
            Just(ANIMAL.clone()),
            Just(DOG.clone()),
            Just(MINERAL.clone()),
        ]
        .prop_map(Descr::instance_of),
    ];
    leaf.prop_recursive(2, 8, 2, |inner| {
        prop_oneof![
            (inner.clone(), inner.clone())
                .prop_map(|(a, b)| a.union(&b).unwrap_or_else(Descr::anything)),
            (inner.clone(), inner.clone())
                .prop_map(|(a, b)| a.intersect(&b).unwrap_or_else(Descr::nothing)),
            inner.prop_map(|a| a.complement()),
        ]
    })
}

proptest! {
    // Fewer cases than the default, and a bounded shrink, because a word or
    // sequence component's operations are automaton products. The case count
    // is what keeps a pass cheap: the default spends most of the suite's
    // time here. The shrink bound is what keeps a *failure* cheap, and it is
    // the one that matters to the mutation sweep -- a broken invariant makes
    // every draw larger, so shrinking one counterexample takes longer than
    // the sweep waits, and a caught mutation reads as a run that hangs.
    #![proptest_config(ProptestConfig {
        cases: 64,
        max_shrink_time: 2_000,
        ..ProptestConfig::default()
    })]

    /// The Boolean algebra, checked by equality of the canonical forms.
    ///
    /// One component per kind, each canonical for its representation, so
    /// equality is equality of the sets and a law is checked at full
    /// strength rather than over whatever values a universe can list.
    #[test]
    fn the_lattice_laws_hold_of_the_descriptors(a in descr(), b in descr(), c in descr()) {
        prop_assert_eq!(a.union(&b), b.union(&a));
        prop_assert_eq!(a.intersect(&b), b.intersect(&a));
        prop_assert_eq!(
            a.union(&b).and_then(|ab| ab.union(&c)),
            b.union(&c).and_then(|bc| a.union(&bc))
        );
        prop_assert_eq!(
            a.intersect(&b).and_then(|ab| ab.intersect(&c)),
            b.intersect(&c).and_then(|bc| a.intersect(&bc))
        );
        prop_assert_eq!(a.union(&a), Some(a.clone()));
        prop_assert_eq!(a.intersect(&a), Some(a.clone()));
        // Absorption and distributivity, which the structural simplifier
        // cannot state because it does not apply them.
        if let Some(met) = a.intersect(&b) {
            prop_assert_eq!(a.union(&met), Some(a.clone()));
        }
        if let Some(joined) = a.union(&b) {
            prop_assert_eq!(a.intersect(&joined), Some(a.clone()));
        }
        if let (Some(left), Some(right)) = (
            b.union(&c).and_then(|bc| a.intersect(&bc)),
            a.intersect(&b).and_then(|ab| {
                a.intersect(&c).and_then(|ac| ab.union(&ac))
            }),
        ) {
            prop_assert_eq!(left, right);
        }
    }

    /// The complement laws, and the two the structural procedure declines.
    #[test]
    fn the_complement_laws_hold_of_the_descriptors(a in descr(), b in descr()) {
        prop_assert!(
            a.intersect(&a.complement())
                .is_some_and(|met| met.is_empty())
        );
        prop_assert_eq!(a.union(&a.complement()), Some(Descr::anything()));
        prop_assert_eq!(&a.complement().complement(), &a);
        // De Morgan, both ways.
        prop_assert_eq!(
            a.union(&b).map(|u| u.complement()),
            a.complement().intersect(&b.complement())
        );
        prop_assert_eq!(
            a.intersect(&b).map(|m| m.complement()),
            a.complement().union(&b.complement())
        );
    }

    /// Two equal descriptors agree about every value.
    #[test]
    fn equal_descriptors_agree_about_every_value(a in descr(), b in descr()) {
        if a == b {
            prop_assert!(agree_on_values(&a, &b));
        }
    }

    /// An empty descriptor admits no value, and one that admits a value of
    /// the universe is not empty.
    #[test]
    fn emptiness_agrees_with_the_values(a in descr()) {
        if a.is_empty() {
            prop_assert!(universe().into_iter().all(|v| !a.admits(v)));
        } else if universe().into_iter().any(|v| a.admits(v)) {
            prop_assert!(!a.is_empty());
        }
    }

    /// A complement saturates the kinds it does not mention.
    #[test]
    fn a_complement_holds_every_value_the_set_does_not(a in descr()) {
        for value in universe() {
            prop_assert_eq!(a.admits(value), !a.complement().admits(value));
        }
    }
}

/// The kind list is the partition, so it must hold every kind exactly once.
///
/// A `match` over `Kind` is exhaustive and this array is not, so the list is
/// counted against the variants rather than trusted: a kind added without a
/// component here would be a kind the descriptor silently cannot represent.
#[test]
fn every_kind_has_exactly_one_component() {
    let mut seen = Kind::ALL.to_vec();
    seen.sort_by_key(|kind| format!("{kind:?}"));
    seen.dedup();
    assert_eq!(seen.len(), Kind::ALL.len(), "a kind is listed twice");

    // Exhaustive by construction: the match forces a new variant to be
    // added here, and the count then forces it into the list.
    let counted = Kind::ALL
        .iter()
        .filter(|kind| {
            matches!(
                kind,
                Kind::NoneType
                    | Kind::Bool
                    | Kind::Int
                    | Kind::Float
                    | Kind::Str
                    | Kind::Bytes
                    | Kind::List
                    | Kind::Tuple
                    | Kind::Set
                    | Kind::FrozenSet
                    | Kind::Dict
            )
        })
        .count();
    assert_eq!(counted, Kind::ALL.len());
}

/// `bool` is its two values, which is what makes the union of the two
/// singletons the whole kind rather than a shape a rule must recognise.
#[test]
fn the_two_booleans_are_the_bool_kind() {
    let both = Descr::boolean(true).union(&Descr::boolean(false));
    assert_eq!(both, Some(Descr::of_kind(Kind::Bool)));
    assert!(Descr::boolean(true).admits(Value::boolean(true)));
    assert!(!Descr::boolean(true).admits(Value::boolean(false)));
    // And the complement of one singleton, inside the kind, is the other.
    let not_true = Descr::boolean(true).complement();
    assert!(not_true.admits(Value::boolean(false)));
    assert!(!not_true.admits(Value::boolean(true)));
    // ... while still holding every value of every other kind.
    assert!(not_true.admits(Value::integer(0)));
    assert!(not_true.admits(Value::word(b"a", Kind::Str)));
    assert!(not_true.admits(Value::other()));
}

/// A coarse component combines as the boolean it is.
///
/// Read here rather than through a descriptor, because a descriptor cannot
/// reach the interesting pair: a line whose structure is proved empty is
/// dropped, so a `Coarse(false)` never meets a `Coarse(true)` in a union. The
/// arm is still what a union and a meet of two kinds mean, and this is where
/// it says so.
#[test]
fn a_coarse_component_combines_as_a_boolean() {
    let yes = Component::Coarse(true);
    let no = Component::Coarse(false);
    assert_eq!(yes.combine(&no, Op::Union), Some(Component::Coarse(true)));
    assert_eq!(no.combine(&yes, Op::Union), Some(Component::Coarse(true)));
    assert_eq!(
        yes.combine(&no, Op::Intersect),
        Some(Component::Coarse(false))
    );
    assert_eq!(
        yes.combine(&yes, Op::Intersect),
        Some(Component::Coarse(true))
    );
}

/// The map rows the report calls undecided, decided.
///
/// `{"a": int}` and `dict[str, str]` share no dict, because the key `a` must
/// map into both and nothing does. The old component was coarse -- every dict
/// or none -- so it could not tell one map from another at all, and the two
/// met in "some dict".
#[test]
fn two_maps_that_disagree_about_a_key_share_no_dict() {
    const A_STR: &[(Value, Value)] =
        &[(Value::word(b"a", Kind::Str), Value::word(b"x", Kind::Str))];
    let ints = Descr::of_kind(Kind::Int);
    let strs = Descr::of_kind(Kind::Str);
    let a_is_int = Descr::label(Label::str("a"), &ints, false);
    let strs_to_strs = Descr::mapping(Kind::Str, &strs);
    let met = a_is_int
        .intersect(&strs_to_strs)
        .expect("the two maps meet");
    assert!(met.is_empty());

    // And two that agree do share one, so the emptiness above is the types
    // disagreeing rather than the meet collapsing.
    let a_is_str = Descr::label(Label::str("a"), &strs, false);
    let agreeing = a_is_str
        .intersect(&strs_to_strs)
        .expect("the two maps meet");
    assert!(!agreeing.is_empty());
    assert!(agreeing.admits(Value::dict(A_STR)));
}

/// A label's type distributes over the union: a map whose `a` is an `int` or
/// a `str` is one of the two maps that say which.
///
/// The row the report lists as `{"a": int|str} ≤ {"a": int} ∨ {"a": str}`.
/// It holds because the atom is a *function* on the labels -- the union
/// splits at the one label rather than having to be searched for.
#[test]
fn a_label_of_a_union_is_the_union_of_the_labels() {
    let ints = Descr::of_kind(Kind::Int);
    let strs = Descr::of_kind(Kind::Str);
    let either = ints.union(&strs).expect("int or str");
    let wide = Descr::label(Label::str("a"), &either, false);
    let split = Descr::label(Label::str("a"), &ints, false)
        .union(&Descr::label(Label::str("a"), &strs, false))
        .expect("the two maps join");
    let outside = wide
        .intersect(&split.complement())
        .expect("the difference is a map");
    assert!(outside.is_empty(), "{outside:?}");
}

/// The key partition is read per part: a map that says what its `str` keys
/// map to says nothing about its `int` keys.
///
/// The default is a *function* on the parts, which is what Castagna's §4.4
/// requires of a key type and what the kind partition supplies. Reading one
/// part for another would make `dict[str, int]` a claim about every key.
#[test]
fn a_map_constrains_one_part_of_the_key_partition() {
    const INT_KEY: &[(Value, Value)] = &[(Value::integer(1), Value::word(b"x", Kind::Str))];
    const STR_KEY: &[(Value, Value)] = &[(Value::word(b"a", Kind::Str), Value::integer(1))];
    let ints = Descr::of_kind(Kind::Int);
    let str_keys = Descr::mapping(Kind::Str, &ints);
    let int_keys = Descr::mapping(Kind::Int, &ints);

    // `dict[str, int]` says nothing about an integer key, and the other way
    // round.
    assert!(str_keys.admits(Value::dict(INT_KEY)));
    assert!(!int_keys.admits(Value::dict(INT_KEY)));
    assert!(int_keys.admits(Value::dict(STR_KEY)));
    assert!(str_keys.admits(Value::dict(STR_KEY)));

    // And closing the map is a claim about the other parts, not this one.
    let only_strs = Descr::keys_among(&[Some(Kind::Str)]);
    assert!(only_strs.admits(Value::dict(STR_KEY)));
    assert!(!only_strs.admits(Value::dict(INT_KEY)));
}

/// A constraint made before a label was known keeps its meaning after.
///
/// The paper fixes one label set for every atom of a normal form, so its `S`
/// is always about a key outside *that* set. Here an atom carries its own
/// labels and a meet unions them, so a constraint read against "the atom's
/// labels" would quietly strengthen as the atom learned names, and the meet
/// would hold fewer dicts than its operands share.
#[test]
fn a_constraint_keeps_its_meaning_when_a_label_arrives() {
    const A_IS_INT: &[(Value, Value)] = &[(Value::word(b"a", Kind::Str), Value::integer(1))];
    const B_IS_STR: &[(Value, Value)] =
        &[(Value::word(b"b", Kind::Str), Value::word(b"x", Kind::Str))];
    let ints = Descr::of_kind(Kind::Int);
    let strs = Descr::of_kind(Kind::Str);
    // "not every str key maps into int" -- which carries a constraint
    // wanting some str key outside no labels at all.
    let some_key_is_not_an_int = Descr::mapping(Kind::Str, &ints).complement();
    let b_is_str = Descr::label(Label::str("b"), &strs, false);
    let met = some_key_is_not_an_int
        .intersect(&b_is_str)
        .expect("the two maps meet");
    // `{"b": "x"}` is one: `b` is the key that is not an integer, and it is a
    // label of the meet rather than a key outside it.
    assert!(met.admits(Value::dict(B_IS_STR)));
    assert!(!met.is_empty());
    // And a union of two maps is a map, rather than a refusal the caller has
    // to widen away.
    let ints = Descr::of_kind(Kind::Int);
    let joined = Descr::label(Label::str("a"), &ints, false)
        .union(&Descr::label(Label::str("b"), &ints, false))
        .expect("the two maps join");
    assert!(!joined.is_empty());
    assert!(joined.admits(Value::dict(A_IS_INT)));
}

/// A map's complement is a map, which is what the negative set is for.
///
/// A dict fails `{"a": int}` by carrying no `a` at all, or by carrying one
/// that is not an integer -- and both are dicts the complement holds.
#[test]
fn a_map_fails_by_a_missing_key_or_a_wrong_one() {
    const A_INT: &[(Value, Value)] = &[(Value::word(b"a", Kind::Str), Value::integer(1))];
    const A_STR: &[(Value, Value)] =
        &[(Value::word(b"a", Kind::Str), Value::word(b"x", Kind::Str))];
    let ints = Descr::of_kind(Kind::Int);
    let a_is_int = Descr::label(Label::str("a"), &ints, false);
    let outside = a_is_int.complement();
    assert!(a_is_int.admits(Value::dict(A_INT)));
    assert!(!a_is_int.admits(Value::dict(&[])));
    assert!(!a_is_int.admits(Value::dict(A_STR)));
    assert!(!outside.admits(Value::dict(A_INT)));
    assert!(outside.admits(Value::dict(&[])));
    assert!(outside.admits(Value::dict(A_STR)));
}

/// A coarse component is all-or-nothing, and the tests must not read that as
/// a distinction the descriptor makes.
#[test]
fn a_coarse_kind_admits_all_of_its_values_or_none() {
    let dicts = Descr::of_kind(Kind::Dict);
    assert!(dicts.admits(Value::dict(&[])));
    assert!(!dicts.admits(Value::of_kind(Kind::NoneType)));
    assert!(!dicts.admits(Value::other()));
    assert!(!dicts.is_empty());
    assert!(
        dicts
            .intersect(&Descr::of_kind(Kind::NoneType))
            .is_some_and(|met| met.is_empty())
    );
    assert!(
        dicts
            .union(&Descr::of_kind(Kind::NoneType))
            .is_some_and(|joined| !joined.is_empty())
    );
}

/// The two word kinds are exact and separate: a pattern over one says
/// nothing about the other, which is what keeps `str` and `bytes` disjoint
/// while sharing a representation.
#[test]
fn the_word_kinds_are_languages_and_stay_apart() {
    let text = Descr::pattern("ab?", Kind::Str).expect("a small pattern");
    assert!(text.admits(Value::word(b"a", Kind::Str)));
    assert!(text.admits(Value::word(b"ab", Kind::Str)));
    assert!(!text.admits(Value::word(b"b", Kind::Str)));
    // The same word as `bytes` is a different value, in a component this
    // descriptor leaves empty.
    assert!(!text.admits(Value::word(b"a", Kind::Bytes)));
    assert!(
        text.intersect(&Descr::of_kind(Kind::Bytes))
            .is_some_and(|met| met.is_empty())
    );

    // One pattern inside another, which the structural procedure declines:
    // it relates two patterns only when they are written identically.
    let narrow = Descr::pattern("a", Kind::Str).expect("a small pattern");
    assert!(
        narrow
            .intersect(&text.complement())
            .is_some_and(|met| met.is_empty())
    );
    // And a pattern whose language is one word is that word.
    assert_eq!(narrow, Descr::word(b"a", Kind::Str).expect("a word kind"));
    // The bytes kind is a word kind too, and its patterns land in its own
    // component: the same pattern over the two kinds is two disjoint sets.
    let raw = Descr::pattern("a", Kind::Bytes).expect("a small pattern");
    assert!(raw.admits(Value::word(b"a", Kind::Bytes)));
    assert!(!raw.admits(Value::word(b"a", Kind::Str)));
    assert!(raw.intersect(&narrow).is_some_and(|met| met.is_empty()));
    // And the two kinds read a pattern over different alphabets: a byte
    // class outside UTF-8 is a language for `bytes` and no language at all
    // for `str`.
    assert!(Descr::pattern(r"(?-u:\xFF)", Kind::Bytes).is_some());
    assert!(Descr::pattern(r"(?-u:\xFF)", Kind::Str).is_none());

    // A pattern over a kind with no words is a caller error, not a set.
    assert!(Descr::pattern("a", Kind::Int).is_none());
    assert!(Descr::word(b"a", Kind::List).is_none());
}

/// The integers are exact, so a bound conjunction that cannot hold is
/// decided rather than declined -- and a step is a set the coarse
/// representation had no way to express at all.
#[test]
fn the_integers_are_a_set_rather_than_a_kind() {
    let ints = Descr::of_kind(Kind::Int);
    assert!(ints.admits(Value::integer(7)));
    assert!(!ints.admits(Value::of_kind(Kind::Str)));
    // A boolean is its own kind, so `int` does not admit one.
    assert!(!ints.admits(Value::boolean(true)));

    let evens = Descr::multiple_of(2).expect("two is inside the bound");
    assert!(evens.admits(Value::integer(4)));
    assert!(!evens.admits(Value::integer(3)));
    // The evens and the odds together are the kind, which no union of
    // intervals could say.
    let odds = evens.complement().intersect(&ints).expect("a small meet");
    assert_eq!(evens.union(&odds), Some(ints.clone()));

    // Two singletons meet in nothing, and each is inside the kind.
    assert!(
        Descr::integer(1)
            .intersect(&Descr::integer(2))
            .is_some_and(|met| met.is_empty())
    );
    assert!(
        Descr::integer(1)
            .intersect(&ints)
            .is_some_and(|met| !met.is_empty())
    );
}

/// The set operations on the two booleans, driven directly: the descriptor
/// laws above exercise them through a component, and these pin the set.
#[test]
fn the_boolean_set_is_a_two_element_boolean_algebra() {
    assert!(BoolSet::EMPTY.is_empty());
    assert!(!BoolSet::BOTH.is_empty());
    assert!(BoolSet::BOTH.holds(true) && BoolSet::BOTH.holds(false));
    assert!(BoolSet::just(true).holds(true) && !BoolSet::just(true).holds(false));
    assert_eq!(BoolSet::just(true).complement(), BoolSet::just(false));
    assert_eq!(
        BoolSet::just(true).union(BoolSet::just(false)),
        BoolSet::BOTH
    );
    assert_eq!(
        BoolSet::just(true).intersect(BoolSet::just(false)),
        BoolSet::EMPTY
    );
    assert_eq!(BoolSet::BOTH.complement(), BoolSet::EMPTY);
}

/// The bottom and top of one kind, which every constructor is written from.
///
/// A kind's bottom is *no lines* and its top is one line over the whole of
/// the kind with no object constraint on it, and each complements into the
/// other. The empty union is what makes the bottom free: a descriptor
/// holding one kind carries eleven empty vectors.
#[test]
fn a_kind_is_bottom_or_top_of_its_own_lines() {
    for kind in Kind::ALL {
        let whole = Component::top(kind);
        let bottom = Lines::bottom();
        let top = Lines::everything(whole.clone());
        assert_eq!(bottom.emptiness(&whole), Verdict::Empty, "{kind:?} bottom");
        assert_eq!(top.emptiness(&whole), Verdict::Inhabited, "{kind:?} top");
        assert_eq!(bottom.complement(&whole), top, "{kind:?} bottom");
        assert_eq!(top.complement(&whole), bottom, "{kind:?} top");
    }
}

/// A length bound over a sequence kind is the sequences of that length: "at
/// least n" admits n and more, "at most n" admits n and fewer, and the bound
/// past the state budget refuses rather than building one state per element.
#[test]
fn a_length_bound_over_a_sequence_kind_counts_its_elements() {
    const NONE: &[Value] = &[];
    const ONE: &[Value] = &[Value::integer(1)];
    const TWO: &[Value] = &[Value::integer(1), Value::integer(1)];

    let at_least_one = Descr::sequences_at_least(1, Kind::List).expect("a small bound");
    assert!(!at_least_one.admits(Value::sequence(NONE, Kind::List)));
    assert!(at_least_one.admits(Value::sequence(ONE, Kind::List)));
    assert!(at_least_one.admits(Value::sequence(TWO, Kind::List)));

    let at_most_one = Descr::sequences_at_most(1, Kind::List).expect("a small bound");
    assert!(at_most_one.admits(Value::sequence(NONE, Kind::List)));
    assert!(at_most_one.admits(Value::sequence(ONE, Kind::List)));
    assert!(!at_most_one.admits(Value::sequence(TWO, Kind::List)));
    // Inside the kind: a tuple of one is not a list of one.
    assert!(!at_most_one.admits(Value::sequence(ONE, Kind::Tuple)));

    // The bound is inclusive and refuses one past it -- checked with a bound
    // small enough to build, since the real one is an automaton of 4,096 states.
    assert!(Descr::sequences_at_least_within(3, 3, Kind::List).is_some());
    assert!(Descr::sequences_at_least_within(4, 3, Kind::List).is_none());
    assert!(Descr::sequences_at_least_within(0, 0, Kind::List).is_some());
}
