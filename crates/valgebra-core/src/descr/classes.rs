//! Classes, as the order their instances inherit.
//!
//! The core cannot call `issubclass`, and it should not want to: the relation is
//! re-decided on every call, and `ABC.register` can change it after a schema is
//! built, so it is not even monotone in time. A relation that moves is not a
//! lattice to reason in. What the core carries instead is a **snapshot**: a class
//! is its identity together with the identities it derives from, taken once where
//! the schema is built, and every question below is asked of that.
//!
//! Only **pure** classes reach here -- those whose metaclass leaves
//! `isinstance` and `issubclass` alone and which register no subclasses after
//! the fact. A class with a hook answers arbitrary code, so it is not a set this
//! algebra can hold; it stays opaque, and staying opaque is what keeps `C ∧ ¬C`
//! from being decided empty while a hook admits a value to both.

use std::collections::BTreeSet;

/// One class, with the order it stands in.
///
/// Identity is the `id` alone: two values with one id are one class, whatever
/// else they carry, so the sets below compare and sort by it.
#[derive(Debug, Clone)]
pub struct Class {
    id: u32,
    /// This class and every class it derives from, transitively.
    ancestors: BTreeSet<u32>,
    /// The instance layout this class lays down.
    ///
    /// Two classes whose layouts conflict cannot both describe one value --
    /// Python refuses to build a class deriving from both -- which is a
    /// disjointness the derivation order alone does not show.
    layout: u32,
}

impl Class {
    /// A class deriving from `bases`, laid out as `layout`.
    ///
    /// The ancestors are closed here rather than walked later: `bases` carries
    /// each base's own ancestors, so one union is the whole transitive order.
    #[must_use]
    pub fn new(id: u32, layout: u32, bases: &[Class]) -> Class {
        let mut ancestors = BTreeSet::from([id]);
        for base in bases {
            ancestors.extend(base.ancestors.iter().copied());
        }
        Class {
            id,
            ancestors,
            layout,
        }
    }

    /// A class deriving from nothing, laid out on its own.
    #[must_use]
    pub fn root(id: u32) -> Class {
        Class::new(id, id, &[])
    }

    /// Whether every instance of this class is an instance of `other`.
    #[must_use]
    pub fn derives_from(&self, other: &Class) -> bool {
        self.ancestors.contains(&other.id)
    }

    /// Whether no value is an instance of both.
    ///
    /// Sound rather than complete: two classes neither of which derives from the
    /// other *may* still share an instance through a class deriving from both,
    /// unless their layouts conflict -- and a conflicting pair cannot have one,
    /// because no class can derive from both.
    #[must_use]
    pub fn disjoint_from(&self, other: &Class) -> bool {
        self.layout != other.layout && !self.derives_from(other) && !other.derives_from(self)
    }
}

/// One class, so identity is the id and nothing else.
impl PartialEq for Class {
    fn eq(&self, other: &Class) -> bool {
        self.id == other.id
    }
}

impl Eq for Class {}

impl Ord for Class {
    fn cmp(&self, other: &Class) -> core::cmp::Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for Class {
    fn partial_cmp(&self, other: &Class) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::Class;

    /// The order is the one the bases give, closed over.
    #[test]
    fn a_class_derives_from_its_bases_and_from_theirs() {
        let animal = Class::root(1);
        let dog = Class::new(2, 1, std::slice::from_ref(&animal));
        let puppy = Class::new(3, 1, std::slice::from_ref(&dog));

        assert!(dog.derives_from(&animal) && puppy.derives_from(&animal));
        assert!(puppy.derives_from(&dog));
        assert!(!animal.derives_from(&dog));
        assert!(animal.derives_from(&animal), "and from itself");
    }

    /// Two bases meet in a class deriving from both, so neither deriving from
    /// the other is not disjointness.
    #[test]
    fn unrelated_classes_of_one_layout_are_not_disjoint() {
        let left = Class::root(1);
        let right = Class::new(2, 1, &[]);

        assert!(!left.derives_from(&right) && !right.derives_from(&left));
        assert!(!left.disjoint_from(&right), "a common subclass may exist");
    }

    /// A layout conflict is disjointness, because no class can derive from both.
    #[test]
    fn classes_of_conflicting_layouts_are_disjoint() {
        let ints = Class::root(1);
        let words = Class::root(2);
        let counter = Class::new(3, 1, std::slice::from_ref(&ints));

        assert!(ints.disjoint_from(&words) && words.disjoint_from(&ints));
        assert!(!ints.disjoint_from(&counter), "one derives from the other");
        assert!(counter.disjoint_from(&words));
    }

    /// Identity is the id: the order and the layout are what a class *knows*,
    /// not what it *is*.
    #[test]
    fn a_class_is_its_id() {
        let root = Class::root(1);
        let same = Class::new(1, 9, &[Class::root(7)]);

        assert_eq!(root, same);
        assert_eq!(root.cmp(&same), core::cmp::Ordering::Equal);
    }
}
