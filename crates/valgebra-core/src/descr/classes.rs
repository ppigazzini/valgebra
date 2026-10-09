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
//!
//! **A class also says what its direct instances can carry.** Every reading
//! that proves a class met with a record inhabited stands on one value: a
//! *direct* instance of the class carrying what the record's fields admit. That
//! value exists only where the class leaves room for it -- an attribute the
//! class body defines cannot be missing, a property answers with its getter,
//! and an instance laid out without a dictionary holds no attribute its slots do
//! not name. [`Attributes`] is that part of the snapshot, and [`Reach`] is what
//! it answers for one name.

use std::collections::BTreeSet;
use std::sync::{Arc, OnceLock};

use crate::kind::Kind;

/// What one attribute of a *direct* instance of a class can be.
///
/// A direct instance is the value the class assumption licenses: an object
/// whose type is the class itself, which a refutation about the class may stand
/// on. A subclass may carry anything, and nothing here says whether one exists,
/// so this answers for the direct instance alone and claims nothing about the
/// class's other instances.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reach {
    /// Missing, or holding any value: the class defines nothing by the name and
    /// the instance has a dictionary or a `__getattr__` hook to hold one, or the
    /// name is a slot, which is missing until it is assigned.
    Anything,
    /// Holding any value, through an entry in the instance's own dictionary,
    /// which a name the class defines without a data descriptor gives way to.
    /// Whether it can be missing is not read: a class attribute is found
    /// whenever the instance holds no entry, and a non-data descriptor's getter
    /// is code.
    AnyValue,
    /// Never there: the class defines no such name, its instances carry no
    /// dictionary, and no hook serves a name.
    Missing,
    /// What it holds is the answer of code -- a data descriptor such as a
    /// property, or a `__getattribute__` hook -- or a value fixed on a class
    /// whose instances carry no dictionary to replace it. No direct instance is
    /// assumed to carry any particular thing here.
    Unread,
}

/// What a class's namespace holds under one name, as an attribute lookup on an
/// instance finds it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Member {
    /// A slot its class's own `__slots__` lays down: missing until assigned,
    /// then whatever was assigned.
    Slot,
    /// A value or a non-data descriptor, such as a function: an entry in the
    /// instance's dictionary takes its place.
    Plain,
    /// A data descriptor other than a slot, such as a property: it answers
    /// every read, whatever the instance's dictionary holds.
    Descriptor,
}

/// Which attribute hook a class defines, if any.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hook {
    /// Neither: a lookup is the namespace and the instance's dictionary.
    Neither,
    /// `__getattr__`, which answers for a name a lookup does not find.
    Getattr,
    /// A `__getattribute__` other than the generic lookup, which answers every
    /// lookup.
    Getattribute,
}

/// The names one class's own namespace defines, with what each holds there.
///
/// One per class on a `__mro__`, and shared by every [`Attributes`] whose
/// `__mro__` the class stands on rather than copied into each: a namespace no
/// assignment reaches, such as `object`'s, is read once and enters every class
/// deriving from it as a reference.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Namespace {
    /// Sorted by name, with one entry per name.
    names: Box<[(Box<str>, Member)]>,
}

impl Namespace {
    /// The namespace defining each of `names` as what it holds. A name given
    /// twice keeps its first entry.
    #[must_use]
    pub fn new(mut names: Vec<(Box<str>, Member)>) -> Namespace {
        names.sort_by(|(left, _), (right, _)| left.cmp(right));
        names.dedup_by(|(later, _), (earlier, _)| later == earlier);
        Namespace {
            names: names.into_boxed_slice(),
        }
    }

    /// What the namespace holds under `name`, where it defines it.
    fn get(&self, name: &str) -> Option<Member> {
        let at = self
            .names
            .binary_search_by(|(defined, _)| (**defined).cmp(name))
            .ok()?;
        self.names.get(at).map(|(_, member)| *member)
    }
}

/// What a class's direct instances can carry, read off the class once.
///
/// Three facts, each fixed where the class is made: the names its `__mro__`
/// defines and what each is, whether its instances carry a dictionary, and
/// which hook it defines. Assigning to the class after the snapshot moves them,
/// as `ABC.register` moves the order; the snapshot is what a decision reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Attributes {
    /// The namespace of each class in the `__mro__`, in its order: a lookup
    /// finds a name in the first one to define it.
    namespaces: Vec<Arc<Namespace>>,
    /// Whether a direct instance has a `__dict__` to hold any attribute.
    carries_dict: bool,
    hook: Hook,
}

impl Attributes {
    /// A class defining no name a record asks for, whose instances carry a
    /// dictionary and which defines no hook: every attribute is
    /// [`Reach::Anything`].
    ///
    /// What a caller holding nothing but an identity gets. It is the class
    /// `class C: pass` is, for every name neither its namespace nor `object`'s
    /// defines.
    #[must_use]
    pub fn plain() -> Attributes {
        Attributes::new(true, Hook::Neither)
    }

    /// A class whose instances do or do not carry a dictionary, defining
    /// `hook` and, until [`Self::inherit`] says otherwise, no name.
    #[must_use]
    pub fn new(carries_dict: bool, hook: Hook) -> Attributes {
        Attributes {
            namespaces: Vec::new(),
            carries_dict,
            hook,
        }
    }

    /// Record that the next class in the `__mro__` defines `namespace`.
    ///
    /// Called in `__mro__` order, so the first namespace to define a name is
    /// the one read: it is the one an attribute lookup finds, and a later
    /// class's definition of the same name is shadowed by it.
    pub fn inherit(&mut self, namespace: Arc<Namespace>) {
        self.namespaces.push(namespace);
    }

    /// Record that the next class in the `__mro__` defines `name` as `member`
    /// and nothing else: [`Self::inherit`] of a namespace of one name.
    pub fn define(&mut self, name: &str, member: Member) {
        self.inherit(Arc::new(Namespace::new(vec![(name.into(), member)])));
    }

    /// What attribute `name` of a direct instance can be.
    #[must_use]
    pub fn reach(&self, name: &str) -> Reach {
        if self.hook == Hook::Getattribute {
            return Reach::Unread;
        }
        let defined = self
            .namespaces
            .iter()
            .find_map(|namespace| namespace.get(name));
        match defined {
            Some(Member::Slot) => Reach::Anything,
            Some(Member::Plain) if self.carries_dict => Reach::AnyValue,
            Some(Member::Plain | Member::Descriptor) => Reach::Unread,
            None if self.carries_dict || self.hook == Hook::Getattr => Reach::Anything,
            None => Reach::Missing,
        }
    }
}

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
    /// What a direct instance can carry, shared by every copy of the snapshot.
    attributes: Arc<Attributes>,
}

/// The facts of a class nothing has read: [`Attributes::plain`], built once
/// rather than once per snapshot.
fn plain_attributes() -> Arc<Attributes> {
    static PLAIN: OnceLock<Arc<Attributes>> = OnceLock::new();
    Arc::clone(PLAIN.get_or_init(|| Arc::new(Attributes::plain())))
}

/// What a builtin's direct instance carries: no dictionary, and every name
/// answered by the builtin's own descriptors, which the core does not read.
/// That is the reading a `__getattribute__` hook gets, [`Reach::Unread`] for
/// every name, so a record met with the class is never proved inhabited.
fn builtin_attributes() -> Arc<Attributes> {
    static BUILTIN: OnceLock<Arc<Attributes>> = OnceLock::new();
    Arc::clone(BUILTIN.get_or_init(|| Arc::new(Attributes::new(false, Hook::Getattribute))))
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
            attributes: plain_attributes(),
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

    /// The same class, with what its direct instances can carry read off it.
    ///
    /// Set beside the constructor, as the kind is, because only a caller that
    /// can see the class object can read its namespace. A class built without
    /// it carries [`Attributes::plain`]. Shared, because a caller reads a class
    /// once and hands the reading to every snapshot of it.
    #[must_use]
    pub fn carrying(self, attributes: Arc<Attributes>) -> Class {
        Class { attributes, ..self }
    }

    /// What attribute `name` of a direct instance of this class can be.
    #[must_use]
    pub fn reach(&self, name: &str) -> Reach {
        self.attributes.reach(name)
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

    /// The values whose type is `kind`'s builtin itself: an `int` and not an
    /// instance of an `int` subclass.
    ///
    /// A literal denotes its constant at the constant's exact type (`ir.rs`),
    /// and a kind holds the builtin's subclasses as well, so `Literal[5]` is
    /// the integer 5 met with this class. Nothing derives from it and it lays
    /// down a layout of its own, so it is disjoint from every class carrying
    /// another -- every subclass of the builtin among them, since each carries
    /// the builtin's layout or one extending it. A class carrying no layout
    /// stays undecided against it, as it does against any class.
    ///
    /// The ids are the top of the range, one per kind, below the `u32::MAX` a
    /// caller saturates at; the bindings number a query's classes from zero.
    #[must_use]
    pub fn exact(kind: Kind) -> Class {
        // The kind's place in `Kind::ALL`: one of eleven, so the subtraction
        // cannot wrap and no two kinds share an id.
        let at = kind as u32;
        let id = u32::MAX - 1 - at;
        Class::laid_out(id, id)
            .of_kind(kind)
            .carrying(builtin_attributes())
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
