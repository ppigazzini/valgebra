//! The class order, held at the constructors: derivation, layout, identity.

use super::Class;

/// The order is the one the bases give, closed over.
#[test]
fn a_class_derives_from_its_bases_and_from_theirs() {
    let animal = Class::laid_out(1, 1);
    let dog = Class::new(2, Some(1), std::slice::from_ref(&animal));
    let puppy = Class::new(3, Some(1), std::slice::from_ref(&dog));

    assert!(dog.derives_from(&animal) && puppy.derives_from(&animal));
    assert!(puppy.derives_from(&dog));
    assert!(!animal.derives_from(&dog));
    assert!(animal.derives_from(&animal), "and from itself");
}

/// Two bases meet in a class deriving from both, so neither deriving from
/// the other is not disjointness.
#[test]
fn unrelated_classes_of_one_layout_are_not_disjoint() {
    let left = Class::laid_out(1, 1);
    let right = Class::new(2, Some(1), &[]);

    assert!(!left.derives_from(&right) && !right.derives_from(&left));
    assert!(!left.disjoint_from(&right), "a common subclass may exist");
}

/// A layout conflict is disjointness, because no class can derive from both.
#[test]
fn classes_of_conflicting_layouts_are_disjoint() {
    let ints = Class::laid_out(1, 1);
    let words = Class::laid_out(2, 2);
    let counter = Class::new(3, Some(1), std::slice::from_ref(&ints));

    assert!(ints.disjoint_from(&words) && words.disjoint_from(&ints));
    assert!(!ints.disjoint_from(&counter), "one derives from the other");
    assert!(counter.disjoint_from(&words));
}

/// The plain layout is not a layout: every other extends it, so a class
/// carrying it can still meet any other in a common subclass.
///
/// Asked of [`Class::plain`] rather than of the layout constant, because the
/// constructor is the thing a caller reaches for and the contract is its
/// own: a class built this way is disjoint from nothing, and the pair of
/// constructors is a choice between saying that and saying the opposite.
#[test]
fn the_plain_layout_conflicts_with_nothing() {
    let plain = Class::plain(1);
    let words = Class::laid_out(2, 2);

    assert!(
        !plain.disjoint_from(&words),
        "`class Both(plain, str)` builds"
    );
    assert!(!words.disjoint_from(&plain));
    assert!(!plain.disjoint_from(&Class::plain(3)));
    // The other constructor says the opposite of the same two ids, which is
    // the whole reason there are two of them.
    assert!(Class::laid_out(1, 1).disjoint_from(&Class::laid_out(3, 3)));
}

/// Identity is the id: the order and the layout are what a class *knows*,
/// not what it *is*.
#[test]
fn a_class_is_its_id() {
    let one = Class::laid_out(1, 1);
    let same = Class::new(1, Some(9), &[Class::laid_out(7, 7)]);

    assert_eq!(one, same);
    assert_eq!(one.cmp(&same), core::cmp::Ordering::Equal);
    // Total, and by id: the sets below hold classes in a `BTreeSet`, and a
    // pair the order cannot compare is a pair that set would hold twice.
    assert!(Class::laid_out(1, 1) < Class::laid_out(2, 2));
}

/// A layout that extends another is no conflict: the class that laid down
/// the narrower one derives from the class that laid down the wider one, so
/// a class deriving from both exists.
///
/// The pair is a plain subclass of a slotted class beside a slotted
/// subclass of the same: `class X(C): pass` carries `C`'s layout and
/// `class D(C): __slots__ = ("d",)` lays down its own, and Python builds
/// `class E(X, D)` because `D`'s layout extends `C`'s. Neither derives from
/// the other, and their layouts differ, so a rule comparing layouts for
/// equality would call them disjoint and refuse a value that exists.
#[test]
fn a_layout_extending_another_is_not_a_conflict() {
    let slotted = Class::laid_out(1, 1);
    let plain_below = Class::new(2, Some(1), std::slice::from_ref(&slotted));
    let slotted_below = Class::new(3, Some(3), std::slice::from_ref(&slotted));
    let elsewhere = Class::laid_out(4, 4);

    assert!(!plain_below.derives_from(&slotted_below));
    assert!(!slotted_below.derives_from(&plain_below));
    assert!(!plain_below.disjoint_from(&slotted_below), "E(X, D) builds");
    assert!(
        !slotted_below.disjoint_from(&plain_below),
        "and in either order"
    );
    assert!(
        !slotted.disjoint_from(&slotted_below),
        "a base and what extends it"
    );
    assert!(
        slotted_below.disjoint_from(&elsewhere),
        "two layouts neither extends"
    );
    assert!(plain_below.disjoint_from(&elsewhere));
}

/// A class carrying no layout is disjoint from nothing, whatever the other
/// carries.
#[test]
fn a_class_carrying_no_layout_is_disjoint_from_nothing() {
    let plain = Class::plain(1);
    let slotted = Class::laid_out(2, 2);
    assert!(!plain.disjoint_from(&slotted) && !slotted.disjoint_from(&plain));
    assert!(!plain.disjoint_from(&Class::plain(3)));
    assert!(!plain.lays_down_a_layout() && slotted.lays_down_a_layout());
}

/// One layout is no conflict with itself, even one named by a class the order
/// does not hold: a class is not disjoint from itself, and two classes carrying
/// that layout are not disjoint from each other.
#[test]
fn one_layout_is_no_conflict_with_itself() {
    let apart = Class::laid_out(2, 9);
    let beside = Class::new(3, Some(9), &[]);
    let elsewhere = Class::laid_out(4, 4);

    assert!(
        !apart.disjoint_from(&apart),
        "a class shares its instances with itself"
    );
    assert!(!apart.disjoint_from(&beside) && !beside.disjoint_from(&apart));
    assert!(apart.disjoint_from(&elsewhere) && elsewhere.disjoint_from(&apart));
}
