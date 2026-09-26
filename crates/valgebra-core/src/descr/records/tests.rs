use super::{MAX_ATOMS, RecordLattice};
use crate::descr::budget;
use crate::descr::classes::Class;
use crate::descr::integers::IntSet;
use crate::verdict::Verdict;
use proptest::prelude::*;

/// A meet past the build's allowance refuses, and the same meet succeeds
/// under one that covers it.
///
/// The atom product is a loop over every pair, so it is one of the places a
/// build multiplies and one of the places the allowance is charged. Both
/// directions, because an allowance that only ever refuses would pass half
/// of this and so would one that never does.
#[test]
fn a_meet_past_the_allowance_refuses() {
    let x = RecordLattice::attribute("x", IntSet::just(1), false);
    let y = RecordLattice::attribute("y", IntSet::just(2), false);

    assert!(budget::under(0, || x.intersect(&y)).is_none());
    assert!(budget::under(64, || x.intersect(&y)).is_some());
}

/// An atom that holds nothing is dropped from a union, and one already there
/// is not added twice.
///
/// Both keep the union a *name* for the objects it holds: an atom holding
/// nothing contributes none, and a repeat contributes none it did not
/// already. Left in, two unions holding the same objects would compare
/// unequal, which is the equality the lattice laws are asked in.
#[test]
fn a_union_drops_the_atoms_that_hold_nothing_and_the_ones_it_has() {
    // An attribute that must be present and holds nothing: no object at all.
    let barren = RecordLattice::attribute("x", IntSet::empty(), false);
    assert!(barren.is_empty());
    let live = RecordLattice::attribute("y", IntSet::just(1), false);

    let joined = barren.union(&live).expect("a union of two records");
    assert_eq!(joined, live, "the empty atom is dropped");
    assert_eq!(
        live.union(&live).expect("a union"),
        live,
        "and so is a repeat"
    );
}

/// A class and one it derives from are one object, and the meet says so
/// rather than declining.
///
/// The open world makes two *unrelated* classes an `Unknown` -- only a class
/// deriving from both satisfies them, and a snapshot of the order cannot say
/// whether one exists. A pair where one derives from the other is not that
/// case, and reading it as one would decline every dataclass beside its base.
#[test]
fn a_class_and_its_base_are_one_object() {
    let animal = Class::laid_out(1, 1);
    let dog = Class::new(2, Some(1), std::slice::from_ref(&animal));
    let mineral = Class::laid_out(3, 3);
    let of = |class: &Class| RecordLattice::<IntSet>::instance_of(class.clone());

    let both = of(&dog)
        .intersect(&of(&animal))
        .expect("a Dog and an Animal");
    assert_eq!(both.emptiness(), Verdict::Inhabited, "a Dog is an Animal");

    // Laid out apart, so no class derives from both: proved empty.
    assert_eq!(
        of(&dog)
            .intersect(&of(&mineral))
            .expect("a Dog and a Mineral")
            .emptiness(),
        Verdict::Empty
    );
}

/// The objects a law is checked over: the attributes the generator names,
/// carried or not, holding one of a few integers.
fn objects() -> Vec<Vec<(&'static str, i64)>> {
    let mut objects = vec![Vec::new()];
    for label in ["x", "y"] {
        for value in [-1i64, 0, 1, 4] {
            objects.push(vec![(label, value)]);
        }
    }
    objects.push(vec![("x", 0), ("y", 1)]);
    objects.push(vec![("x", 1), ("y", 0)]);
    objects.push(vec![("x", 4), ("y", 4)]);
    // One attribute no atom names, which the open reading must ignore.
    objects.push(vec![("z", 0)]);
    objects.push(vec![("x", 0), ("z", 0)]);
    objects
}

fn holds(lattice: &RecordLattice<IntSet>, object: &[(&'static str, i64)]) -> bool {
    let attributes: Vec<(&str, i64)> = object.to_vec();
    lattice.holds(None, &attributes)
}

fn same(a: &RecordLattice<IntSet>, b: &RecordLattice<IntSet>) -> bool {
    objects()
        .iter()
        .all(|object| holds(a, object) == holds(b, object))
}

fn lattice() -> impl Strategy<Value = RecordLattice<IntSet>> {
    let leaf = prop_oneof![
        Just(RecordLattice::empty()),
        Just(RecordLattice::all()),
        (prop_oneof![Just("x"), Just("y")]).prop_map(RecordLattice::without),
        (
            prop_oneof![Just("x"), Just("y")],
            -1i64..=1,
            proptest::bool::ANY,
        )
            .prop_map(|(label, n, optional)| {
                RecordLattice::attribute(label, IntSet::just(n), optional)
            }),
        (prop_oneof![Just("x"), Just("y")], proptest::bool::ANY).prop_map(|(label, optional)| {
            let evens = IntSet::multiple_of(2).expect("a small step");
            RecordLattice::attribute(label, evens, optional)
        }),
    ];
    leaf.prop_recursive(3, 12, 2, |inner| {
        prop_oneof![
            (inner.clone(), inner.clone())
                .prop_map(|(a, b)| a.union(&b).unwrap_or_else(RecordLattice::all)),
            (inner.clone(), inner.clone())
                .prop_map(|(a, b)| a.intersect(&b).unwrap_or_else(RecordLattice::empty)),
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

    // THEORY: each-kind-is-closed
    /// The Boolean algebra, checked against the objects rather than by
    /// equality of the forms, which a union of atoms does not make
    /// canonical.
    #[test]
    fn the_lattice_laws_hold_of_the_objects(
        a in lattice(),
        b in lattice(),
        c in lattice(),
    ) {
        let _allowance = budget::law();
        if let (Some(ab), Some(ba)) = (a.union(&b), b.union(&a)) {
            prop_assert!(same(&ab, &ba), "join commutes");
        }
        if let (Some(ab), Some(ba)) = (a.intersect(&b), b.intersect(&a)) {
            prop_assert!(same(&ab, &ba), "meet commutes");
        }
        if let (Some(bc), Some(ab)) = (b.union(&c), a.union(&b))
            && let (Some(left), Some(right)) = (a.union(&bc), ab.union(&c))
        {
            prop_assert!(same(&left, &right), "join associates");
        }
        if let (Some(bc), Some(ac)) = (b.union(&c), a.intersect(&c))
            && let (Some(ab), Some(left)) = (a.intersect(&b), a.intersect(&bc))
            && let Some(right) = ab.union(&ac)
        {
            prop_assert!(same(&left, &right), "meet distributes over join");
        }
    }

    /// The complement laws, and De Morgan both ways.
    #[test]
    fn the_complement_laws_hold_of_the_objects(a in lattice(), b in lattice()) {
        let _allowance = budget::law();
        let not_a = a.complement();
        if let Some(met) = a.intersect(&not_a) {
            prop_assert!(met.is_empty(), "an object is in one of the two");
        }
        if let Some(joined) = a.union(&not_a) {
            prop_assert!(same(&joined, &RecordLattice::all()), "and in one of them");
        }
        prop_assert!(same(&not_a.complement(), &a), "twice is nothing");
        let not_b = b.complement();
        if let (Some(joined), Some(met)) = (a.union(&b), not_a.intersect(&not_b)) {
            prop_assert!(same(&joined.complement(), &met), "de Morgan one way");
        }
        if let (Some(met), Some(joined)) = (a.intersect(&b), not_a.union(&not_b)) {
            prop_assert!(same(&met.complement(), &joined), "and the other");
        }
    }

    /// Emptiness is a decision about the objects, not about the form.
    #[test]
    fn emptiness_agrees_with_the_objects(a in lattice()) {
        let _allowance = budget::law();
        if a.is_empty() {
            prop_assert!(
                objects().iter().all(|object| !holds(&a, object)),
                "an empty lattice holds no object"
            );
        }
    }
}

/// A class constrains a value the way an attribute does, and its complement
/// is a constraint again rather than an absence.
#[test]
fn a_class_and_its_complement_are_both_sets() {
    let animal = Class::laid_out(1, 1);
    let dog = Class::new(2, Some(1), std::slice::from_ref(&animal));
    let dogs = RecordLattice::<IntSet>::instance_of(dog.clone());

    assert!(dogs.holds(Some(&dog), &[]));
    assert!(!dogs.holds(Some(&animal), &[]), "a base is not an instance");
    assert!(!dogs.holds(None, &[]), "a value with no class is not one");

    let others = dogs.complement();
    assert!(!others.holds(Some(&dog), &[]));
    assert!(others.holds(Some(&animal), &[]) && others.holds(None, &[]));
}

/// Being an instance of a class is being one of its bases, so a meet with a
/// base says nothing new -- and a meet with what a base excludes is empty.
#[test]
fn deriving_decides_the_meet_and_the_emptiness() {
    let animal = Class::laid_out(1, 1);
    let dog = Class::new(2, Some(1), std::slice::from_ref(&animal));
    let dogs = RecordLattice::<IntSet>::instance_of(dog.clone());
    let animals = RecordLattice::instance_of(animal.clone());

    let both = dogs.intersect(&animals).expect("two small atoms");
    assert!(same(&both, &dogs), "a dog is already an animal");

    let neither = dogs
        .intersect(&animals.complement())
        .expect("two small atoms");
    assert!(neither.is_empty(), "no dog is not an animal");
}

/// The world is open: a class nothing derives from leaves the atom
/// inhabited, because a class outside the list may yet describe a value.
#[test]
fn excluding_an_unrelated_class_leaves_the_atom_inhabited() {
    let animal = Class::laid_out(1, 1);
    let mineral = Class::laid_out(2, 2);
    let animals = RecordLattice::<IntSet>::instance_of(animal.clone());

    let not_mineral = animals
        .intersect(&RecordLattice::instance_of(mineral).complement())
        .expect("two small atoms");
    assert!(!not_mineral.is_empty());
    assert!(not_mineral.holds(Some(&animal), &[]));
}

/// Two classes that cannot both describe a value make the atom empty, which
/// the derivation order alone does not show.
#[test]
fn two_classes_of_conflicting_layouts_meet_in_nothing() {
    let ints = Class::laid_out(1, 1);
    let words = Class::laid_out(2, 2);
    let left = Class::new(3, Some(1), std::slice::from_ref(&ints));
    let right = Class::new(4, Some(1), std::slice::from_ref(&ints));

    let met = RecordLattice::<IntSet>::instance_of(ints)
        .intersect(&RecordLattice::instance_of(words))
        .expect("two small atoms");
    assert!(met.is_empty(), "no value is laid out both ways");

    // One layout, inherited by both, and no derivation between them: a class
    // deriving from both may exist, so this is *not* empty.
    let open = RecordLattice::<IntSet>::instance_of(left)
        .intersect(&RecordLattice::instance_of(right))
        .expect("two small atoms");
    assert!(!open.is_empty());
}

/// A class and an attribute constrain one value together, which is what
/// putting them in one atom is for.
#[test]
fn a_class_and_an_attribute_constrain_one_value() {
    let dog = Class::laid_out(1, 1);
    let named = RecordLattice::instance_of(dog.clone())
        .intersect(&RecordLattice::attribute("x", IntSet::just(1), false))
        .expect("two small atoms");

    assert!(named.holds(Some(&dog), &[("x", 1)]));
    assert!(!named.holds(Some(&dog), &[("x", 2)]));
    assert!(!named.holds(None, &[("x", 1)]));
}

/// An atom constrains the attributes it names and no others, which is what
/// makes the record *open*.
#[test]
fn an_atom_ignores_the_attributes_it_does_not_name() {
    let with_x = RecordLattice::attribute("x", IntSet::all(), false);

    assert!(with_x.holds(None, &[("x", 0)]));
    assert!(with_x.holds(None, &[("x", 0), ("z", 9)]), "z is not named");
    assert!(!with_x.holds(None, &[("z", 9)]), "x is missing");
    assert!(!with_x.holds(None, &[]));
}

/// An optional attribute is the type with `⊥` in it: carried and of the
/// right type, or not carried at all.
#[test]
fn an_optional_attribute_admits_the_object_without_it() {
    let evens = IntSet::multiple_of(2).expect("a small step");
    let optional = RecordLattice::attribute("x", evens.clone(), true);
    let required = RecordLattice::attribute("x", evens, false);

    assert!(optional.holds(None, &[]) && !required.holds(None, &[]));
    assert!(optional.holds(None, &[("x", 2)]) && required.holds(None, &[("x", 2)]));
    assert!(
        !optional.holds(None, &[("x", 1)]),
        "carried, so the type decides"
    );
}

/// A field that must be missing is the empty type with `⊥` in it, which is
/// what the complement of an always-present field gives.
#[test]
fn an_attribute_that_must_be_missing_is_the_bottom_with_undefined() {
    let without = RecordLattice::<IntSet>::without("x");
    let with_any = RecordLattice::attribute("x", IntSet::all(), false);

    assert!(without.holds(None, &[]) && !without.holds(None, &[("x", 0)]));
    assert!(same(&with_any.complement(), &without));
}

/// The complement of an atom is one atom per label, which is what the open
/// default buys: an unnamed attribute cannot make an object fail.
#[test]
fn a_complement_splits_over_the_labels() {
    let both = RecordLattice::attribute("x", IntSet::just(0), false)
        .intersect(&RecordLattice::attribute("y", IntSet::just(0), false))
        .expect("two small atoms");
    let failing = both.complement();

    assert!(both.holds(None, &[("x", 0), ("y", 0)]));
    assert!(!failing.holds(None, &[("x", 0), ("y", 0)]));
    // Failing at either label is enough, and so is missing either one.
    assert!(failing.holds(None, &[("x", 1), ("y", 0)]));
    assert!(failing.holds(None, &[("x", 0), ("y", 1)]));
    assert!(failing.holds(None, &[("x", 0)]));
    assert!(failing.holds(None, &[]));
}

/// Two atoms naming different attributes meet in the object carrying both.
#[test]
fn atoms_over_different_labels_meet_rather_than_conflict() {
    let met = RecordLattice::attribute("x", IntSet::just(0), false)
        .intersect(&RecordLattice::attribute("y", IntSet::just(1), false))
        .expect("two small atoms");

    assert!(!met.is_empty());
    assert!(met.holds(None, &[("x", 0), ("y", 1)]));
    assert!(!met.holds(None, &[("x", 0)]) && !met.holds(None, &[("y", 1)]));
}

/// A required attribute whose type is empty admits nothing, which is
/// emptiness (11) for the always-open record.
#[test]
fn a_required_attribute_of_no_values_is_empty() {
    let impossible = RecordLattice::attribute("x", IntSet::empty(), false);
    assert!(impossible.is_empty());

    // Two required types that do not meet are the same thing found by (12).
    let conflict = RecordLattice::attribute("x", IntSet::just(0), false)
        .intersect(&RecordLattice::attribute("x", IntSet::just(1), false))
        .expect("two small atoms");
    assert!(conflict.is_empty());
}

/// A union past the bound refuses rather than holding a form it cannot.
#[test]
fn a_union_past_the_bound_refuses() {
    let mut wide = RecordLattice::attribute("x", IntSet::just(0), false);
    for n in 1..i64::try_from(MAX_ATOMS).unwrap_or(i64::MAX) {
        wide = wide
            .union(&RecordLattice::attribute("x", IntSet::just(n), false))
            .expect("inside the bound");
    }
    assert!(
        wide.union(&RecordLattice::attribute("x", IntSet::just(-1), false))
            .is_none()
    );
}

// THEORY: the-descriptor
/// A complement is expanded where the expansion is one product, and carried
/// under the flag past that.
///
/// Three widths, three claims. No atoms complements into the whole and the
/// whole back into none, which is what keeps the cheap forms canonical: two
/// descriptors holding the same objects compare equal rather than differing by
/// the route each took. Two atoms is a product of two complements, a width the
/// meet it is headed for would have pruned, so the atoms are carried as they
/// are and the polarity says what they mean.
#[test]
fn a_complement_is_expanded_only_where_it_is_one_product() {
    let none: RecordLattice<IntSet> = RecordLattice::empty();
    let every: RecordLattice<IntSet> = RecordLattice::all();
    assert_eq!(
        none.complement(),
        every,
        "no atoms are the whole complemented"
    );
    assert_eq!(
        every.complement(),
        none,
        "and the whole is none complemented"
    );

    let two = RecordLattice::attribute("x", IntSet::just(0), false)
        .union(&RecordLattice::attribute("y", IntSet::just(1), false))
        .expect("two atoms");
    let carried = two.complement();
    assert!(
        carried.negated,
        "two atoms are carried rather than expanded"
    );
    assert_eq!(carried.atoms, two.atoms, "with the atoms as they were");
    assert!(
        same(&carried.complement(), &two),
        "and the flag complements back into the union it carries"
    );
}

/// The bound reads the width of the union, never the number of pairs.
///
/// Twenty atoms met with twenty is four hundred pairs, and the pairs are where
/// the count grows: it passes [`MAX_ATOMS`] long before the meet is done. The
/// union they collapse to is twenty atoms wide, because an atom wanting two
/// values of one attribute holds no object and a union carries no such atom.
/// Refusing on the raw count would make the bound a question about the order
/// the factors were multiplied in, which is a property of how a difference was
/// written rather than of the objects it names.
#[test]
fn a_meet_is_bounded_by_the_width_of_its_union_and_not_by_its_pairs() {
    let wide = (1..20i64)
        .try_fold(
            RecordLattice::attribute("x", IntSet::just(0), false),
            |left, n| left.union(&RecordLattice::attribute("x", IntSet::just(n), false)),
        )
        .expect("twenty atoms");
    let met = wide
        .intersect(&wide)
        .expect("four hundred pairs, one union");
    assert!(
        wide.atoms.len() * wide.atoms.len() > MAX_ATOMS,
        "the row needs a pair count the bound would refuse"
    );
    assert_eq!(
        met.atoms.len(),
        wide.atoms.len(),
        "and an answer no wider than either side"
    );
    assert!(same(&met, &wide), "a meet with itself holds what it held");
}
