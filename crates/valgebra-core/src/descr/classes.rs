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

use crate::kind::Kind;

/// One class, with the order it stands in.
///
/// Identity is the `id` alone: two values with one id are one class, whatever
/// else they carry, so the sets below compare and sort by it.
#[derive(Debug, Clone)]
pub struct Class {
    id: u32,
    /// This class and every class it derives from, transitively.
    ancestors: BTreeSet<u32>,
    /// The class whose instance layout this one carries -- itself, where it lays
    /// one down, or the nearest ancestor that does -- or `None` for a class
    /// carrying none: a plain class, whose instances have the shape of `object`.
    ///
    /// Python builds a class deriving from two others only where the layout one
    /// carries extends the other's, so two classes whose layouts neither extend
    /// the other share no value: a disjointness the derivation order alone does
    /// not show. A layout is named by the class that laid it down, which is
    /// always among the ancestors of a class carrying it, so "extends" is the
    /// order this snapshot already holds.
    layout: Option<u32>,
    /// The kind every instance of this class has, or `None` for a class whose
    /// instances may be of any kind.
    ///
    /// A class laying down a builtin layout confines its instances to that
    /// builtin's kind, and every subclass keeps the layout -- so a `str`
    /// subclass constrains a value *within* the `Str` kind rather than instead
    /// of it, and belongs on that kind's line alone. A class laying down no
    /// layout of its own confines nothing: `class Both(Plain, MyStr)` builds and
    /// its instances are strings, so placing such a class narrowly would claim a
    /// value does not exist. `None` is that case, and it is the default.
    kind: Option<Kind>,
}

impl Class {
    /// A class deriving from `bases`, carrying the layout of the class `layout`
    /// names -- its own id, or an ancestor's -- or none.
    ///
    /// The ancestors are closed here rather than walked later: `bases` carries
    /// each base's own ancestors, so one union is the whole transitive order.
    #[must_use]
    pub fn new(id: u32, layout: Option<u32>, bases: &[Class]) -> Class {
        let mut ancestors = BTreeSet::from([id]);
        for base in bases {
            ancestors.extend(base.ancestors.iter().copied());
        }
        Class {
            id,
            ancestors,
            layout,
            kind: None,
        }
    }

    /// The same class, confined to the kind its instances have.
    ///
    /// Only a caller that can see the class object knows this, so it is set
    /// beside the constructor rather than derived from the layout tag, which is
    /// a number the caller chose.
    #[must_use]
    pub fn of_kind(self, kind: Kind) -> Class {
        Class {
            kind: Some(kind),
            ..self
        }
    }

    /// The kind every instance of this class has, where the class confines it.
    #[must_use]
    pub fn kind(&self) -> Option<Kind> {
        self.kind
    }

    /// A class deriving from nothing and laying down no layout of its own.
    ///
    /// The one to reach for when nothing is known but the identity: it conflicts
    /// with no layout, so it is disjoint from nothing and a class deriving from
    /// it and from anything else may exist. That is what an ordinary Python
    /// class is.
    #[must_use]
    pub fn plain(id: u32) -> Class {
        Class::new(id, None, &[])
    }

    /// A class deriving from nothing and carrying the layout `layout` names.
    ///
    /// Named for the half that decides disjointness. This was `root`, which read
    /// as a statement about *derivation* -- and the derivation is the half that
    /// decides nothing here, since a class deriving from nothing is disjoint
    /// from nothing on that ground alone. What made two such classes disjoint
    /// was the layout the old constructor quietly set to the id, so two calls
    /// with different ids meant "these can share no value" without ever saying
    /// so.
    #[must_use]
    pub fn laid_out(id: u32, layout: u32) -> Class {
        Class::new(id, Some(layout), &[])
    }

    /// Whether this class carries a layout, its own or an ancestor's.
    ///
    /// A class that does confines every subclass to it: no class deriving from
    /// this one can take on another layout that does not extend it. One that
    /// carries none confines nothing, and a subclass of it may be anything.
    #[must_use]
    pub fn lays_down_a_layout(&self) -> bool {
        self.layout.is_some()
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
    /// because no class can derive from both. Two layouts conflict when neither
    /// extends the other, and a layout extends another when the class that laid
    /// it down derives from the class that laid down the other: `other`'s layout
    /// extends this one's exactly when this one's is among `other`'s ancestors.
    /// A class deriving from the other passes that test on its own, since the
    /// layout it carries is an ancestor's. One layout is no conflict with
    /// itself, and that is said outright rather than left to the ancestors,
    /// because a layout named by a class outside the order -- which a caller
    /// with nothing but a number may write -- is among nobody's. A class
    /// carrying no layout is disjoint from nothing here.
    #[must_use]
    pub fn disjoint_from(&self, other: &Class) -> bool {
        match (self.layout, other.layout) {
            (Some(mine), Some(theirs)) => {
                mine != theirs
                    && !self.ancestors.contains(&theirs)
                    && !other.ancestors.contains(&mine)
            }
            _ => false,
        }
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
mod tests;
