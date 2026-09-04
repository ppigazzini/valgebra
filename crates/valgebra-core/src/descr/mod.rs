//! The set-theoretic descriptor: a set of values held one component per kind.
//!
//! The structural decision procedure reads a schema's *syntax* and applies
//! inclusion rules to it. That is why it is sound but incomplete: a relation it
//! has no rule for is declined, and the shape a caller happened to write decides
//! which rules fire. The descriptor is the
//! other approach, the one Frisch, Castagna and Benzaken take (JACM 55(4), 2008)
//! and Castagna and Duboc implement (§7): give the *set* a representation closed
//! under union, intersection and complement, and decide every relation by
//! emptiness of one combination.
//!
//! The value universe is partitioned by [`Kind`], so a set of values is a set
//! per kind and nothing more. Union, intersection and complement are then
//! componentwise, which is the whole reason to partition first: no rule relates
//! a list to an int, because they live in components that never meet.
//!
//! **This is built beside the structural procedure, not in place of it.** It
//! decides nothing a caller can reach yet. Each component starts *coarse* --
//! every value of the kind, or none -- and each later commit replaces one kind's
//! component with a representation that distinguishes its values. The type says
//! which is which, so what the descriptor can and cannot see is read off it
//! rather than inferred.

pub mod classes;
pub mod floats;
pub mod integers;
pub mod interval;
pub mod records;
pub mod regular;
pub mod sets;
pub mod symbolic;
pub mod values;

use crate::decision::Kind;
use classes::Class;
use floats::FloatSet;
use integers::IntSet;
use records::RecordLattice;
use regular::{Alphabet, RegularSet};
use sets::SetLattice;
use symbolic::{Guard, SymbolicDfa};

/// The two booleans, as a subset.
///
/// `bool` is a kind with exactly two values, so a *finite set* over it is exact:
/// `Literal[True]` is `{True}`, and `Literal[True] | Literal[False]` is the
/// whole kind rather than a union the procedure must recognise. A two-bit set is
/// the smallest thing closed under the three operations here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct BoolSet(u8);

impl BoolSet {
    /// Neither boolean.
    pub const EMPTY: BoolSet = BoolSet(0);
    /// Both booleans: the whole `bool` kind.
    pub const BOTH: BoolSet = BoolSet(0b11);

    /// The singleton holding just this boolean.
    ///
    /// The bit *is* the boolean: `false` takes bit zero and `true` bit one, so
    /// the two are related in one expression rather than by a named constant
    /// each. Two constants would be two more things to keep in step, and their
    /// disjointness -- which is what makes the union of the singletons the whole
    /// kind -- would be a fact about two literals rather than about the shift.
    #[must_use]
    pub fn just(value: bool) -> BoolSet {
        BoolSet(1 << u8::from(value))
    }

    /// Whether this set holds `value`.
    #[must_use]
    pub fn holds(self, value: bool) -> bool {
        self.0 & BoolSet::just(value).0 != 0
    }

    const fn union(self, other: BoolSet) -> BoolSet {
        BoolSet(self.0 | other.0)
    }

    const fn intersect(self, other: BoolSet) -> BoolSet {
        BoolSet(self.0 & other.0)
    }

    const fn complement(self) -> BoolSet {
        BoolSet(BoolSet::BOTH.0 & !self.0)
    }

    const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

/// The values of one kind that a [`Descr`] admits.
///
/// One variant per representation, not per kind: a kind whose values are not yet
/// distinguished carries [`Coarse`](Component::Coarse), and moving a kind to an
/// exact representation is adding a variant and the arms that go with it.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Component {
    /// Every value of the kind, or none.
    ///
    /// Exact for `None`, which has one value, and coarse for every other kind
    /// that still carries it: `set[int]` and `set[str]` are the same component,
    /// so the descriptor cannot yet tell them apart. Coarse is *sound* rather
    /// than wrong -- it is the honest representation of a distinction not yet
    /// made, and emptiness over it decides the kind partition and nothing finer.
    Coarse(bool),
    /// The booleans this descriptor admits.
    Booleans(BoolSet),
    /// The integers this descriptor admits.
    ///
    /// `bool` is a separate kind, so this is the integers that are not booleans
    /// -- which is what makes the two components independent. A schema that
    /// admits both spells that as a descriptor holding a component in each.
    Integers(IntSet),
    /// The floats this descriptor admits, `nan` included.
    Floats(FloatSet),
    /// The words this descriptor admits, as a regular language.
    ///
    /// Serves `str` and `bytes` both: a word is a byte string either way, and
    /// which alphabet a *pattern* is read over is settled where the language is
    /// built rather than carried here.
    Words(RegularSet),
    /// The sequences this descriptor admits, as an automaton over value sets.
    ///
    /// Serves `list` and `tuple` both, and the difference between them is the
    /// kind rather than the language: `list[int]` is a loop and `tuple[int, str]`
    /// is a chain, which one constructor spells. The letters are descriptors, so
    /// the component is recursive -- through the automaton's *states*, where the
    /// cycle is an edge and every guard stays a finite descriptor.
    Sequences(SymbolicDfa<Descr>),
    /// The sets this descriptor admits, as a union of powerset lines.
    ///
    /// Serves `set` and `frozenset` both. A set is its *members* and nothing
    /// else -- there is no order for an automaton to walk -- so the component is
    /// the powerset of a descriptor rather than a language over it.
    Sets(SetLattice<Descr>),
    /// The objects this descriptor admits, as a union of open record atoms.
    ///
    /// What describes a value of no listed kind: not its class, which only the
    /// bindings can compare, but the attributes it carries. Always open, because
    /// an object may carry attributes no schema mentions.
    Records(RecordLattice<Descr>),
}

impl Component {
    /// Every value of the kind.
    fn top(kind: Kind) -> Component {
        match kind {
            Kind::Bool => Component::Booleans(BoolSet::BOTH),
            Kind::Int => Component::Integers(IntSet::all()),
            Kind::Float => Component::Floats(FloatSet::all()),
            Kind::Str | Kind::Bytes => Component::Words(RegularSet::all()),
            Kind::List | Kind::Tuple => Component::Sequences(SymbolicDfa::all()),
            Kind::Set | Kind::FrozenSet => Component::Sets(SetLattice::all()),
            _ => Component::Coarse(true),
        }
    }

    /// No value of the kind.
    fn bottom(kind: Kind) -> Component {
        match kind {
            Kind::Bool => Component::Booleans(BoolSet::EMPTY),
            Kind::Int => Component::Integers(IntSet::empty()),
            Kind::Float => Component::Floats(FloatSet::empty()),
            Kind::Str | Kind::Bytes => Component::Words(RegularSet::empty()),
            Kind::List | Kind::Tuple => Component::Sequences(SymbolicDfa::empty()),
            Kind::Set | Kind::FrozenSet => Component::Sets(SetLattice::empty()),
            _ => Component::Coarse(false),
        }
    }

    /// Whether this component admits no value at all.
    fn is_empty(&self) -> bool {
        match self {
            Component::Coarse(present) => !present,
            Component::Booleans(set) => set.is_empty(),
            Component::Integers(set) => set.is_empty(),
            Component::Floats(set) => set.is_empty(),
            Component::Words(set) => set.is_empty(),
            Component::Sequences(set) => set.is_empty(),
            Component::Sets(set) => set.is_empty(),
            Component::Records(set) => set.is_empty(),
        }
    }

    /// The three operations, each on two components of the *same* kind.
    ///
    /// Mixing representations is a bug in the caller, not a case to handle: a
    /// descriptor holds one component per kind and combines them positionally,
    /// so both sides of every call are that kind's representation. The mismatch
    /// arm keeps the crate free of a panic across the boundary and is asserted
    /// unreachable in debug.
    fn combine(&self, other: &Component, op: Op) -> Option<Component> {
        Some(match (self, other) {
            (Component::Coarse(a), Component::Coarse(b)) => Component::Coarse(match op {
                Op::Union => *a || *b,
                Op::Intersect => *a && *b,
            }),
            (Component::Booleans(a), Component::Booleans(b)) => Component::Booleans(match op {
                Op::Union => a.union(*b),
                Op::Intersect => a.intersect(*b),
            }),
            (Component::Integers(a), Component::Integers(b)) => Component::Integers(match op {
                Op::Union => a.union(b),
                Op::Intersect => a.intersect(b),
            }),
            (Component::Floats(a), Component::Floats(b)) => Component::Floats(match op {
                Op::Union => a.union(b),
                Op::Intersect => a.intersect(b),
            }),
            (Component::Words(a), Component::Words(b)) => {
                // A language operation can pass the automaton bound, and there
                // is no sound set to substitute -- one too wide is complemented
                // into one too narrow. The whole descriptor becomes unbuildable
                // rather than quietly wrong, which is what the `Option` on the
                // three operations carries up.
                let combined = match op {
                    Op::Union => a.union(b),
                    Op::Intersect => a.intersect(b),
                };
                Component::Words(combined?)
            }
            (Component::Sequences(a), Component::Sequences(b)) => {
                // Refused for the same reason, one alphabet up: a product of two
                // automata can pass the bound, and a language too wide is
                // complemented into one too narrow.
                let combined = match op {
                    Op::Union => a.union(b),
                    Op::Intersect => a.intersect(b),
                };
                Component::Sequences(combined?)
            }
            (Component::Sets(a), Component::Sets(b)) => {
                // Refused for the same reason again: a union of powerset lines
                // multiplies under a meet, and past the bound there is no sound
                // union to substitute.
                let combined = match op {
                    Op::Union => a.union(b),
                    Op::Intersect => a.intersect(b),
                };
                Component::Sets(combined?)
            }
            (Component::Records(a), Component::Records(b)) => {
                // A union of record atoms multiplies under a meet in the same
                // way, and refuses in the same way past the bound.
                let combined = match op {
                    Op::Union => a.union(b),
                    Op::Intersect => a.intersect(b),
                };
                Component::Records(combined?)
            }
            (mine, theirs) => {
                debug_assert!(false, "combining {mine:?} with {theirs:?} of another kind");
                mine.clone()
            }
        })
    }

    /// Every value of the kind this component does not admit.
    fn complement(&self) -> Component {
        match self {
            Component::Coarse(present) => Component::Coarse(!present),
            Component::Booleans(set) => Component::Booleans(set.complement()),
            Component::Integers(set) => Component::Integers(set.complement()),
            Component::Floats(set) => Component::Floats(set.complement()),
            Component::Words(set) => Component::Words(set.complement()),
            Component::Sequences(set) => Component::Sequences(set.complement()),
            Component::Sets(set) => Component::Sets(set.complement()),
            Component::Records(set) => Component::Records(set.complement()),
        }
    }
}

/// Which alphabet a word kind's patterns are read over, or `None` for a kind
/// that has no words.
///
/// The one place the two word kinds differ: `str` patterns are Unicode, so `.`
/// is a code point and a length bound counts code points; `bytes` patterns are
/// not, so both count bytes.
fn alphabet_of(kind: Kind) -> Option<Alphabet> {
    match kind {
        Kind::Str => Some(Alphabet::Text),
        Kind::Bytes => Some(Alphabet::Bytes),
        _ => None,
    }
}

/// The values a set may hold: everything but the three kinds Python cannot hash.
///
/// A `list`, a `set` and a `dict` are mutable and unhashable, so no set holds
/// one. Written as the complement of those three rather than as a list of the
/// rest, so a kind added later is hashable until someone says otherwise -- the
/// direction that leaves a set *larger*, which declines rather than admits.
fn hashable() -> Descr {
    let mut unhashable = Descr::nothing();
    for kind in [Kind::List, Kind::Set, Kind::Dict] {
        unhashable.put(kind, Component::top(kind));
    }
    unhashable.complement()
}

/// Whether a kind's values are sets of values, which is what the powerset
/// component reads.
fn is_set(kind: Kind) -> bool {
    matches!(kind, Kind::Set | Kind::FrozenSet)
}

/// Whether a kind's values are sequences of values, which is what the automaton
/// component reads.
///
/// A dict is a container too, but its elements are pairs, so it wants its own
/// rule. A set has no order for an automaton to walk and gets [`is_set`].
fn is_sequence(kind: Kind) -> bool {
    matches!(kind, Kind::List | Kind::Tuple)
}

/// Which way two components combine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Op {
    Union,
    Intersect,
}

/// A set of values, held as one component per [`Kind`] plus everything else.
///
/// **Canonical by construction, with one exception.** Each component is
/// canonical for its representation, and there is exactly one component per
/// kind, so two descriptors that differ in a canonical component admit
/// different values. Nothing needs normalising afterwards, which is what makes
/// the operations' laws structural rather than up-to-equivalence.
///
/// The exceptions are three. [`sets`] and [`records`] are held as a *union*, and
/// a union can hold the same values two ways -- `P(A ∪ B)` is also the union of
/// `P(A)`, `P(B)` and the line subtracting both -- which costs a search for
/// coverings neither runs. [`symbolic`] is the third once its letters are
/// descriptors: minimisation asks the guards to join and to say what they leave,
/// and a descriptor cannot always answer, because answering rebuilds the very
/// automata being minimised. Each such refusal leaves a coarser table for the
/// same language.
///
/// So equality is *finer* than agreeing on values once any of the three is
/// involved: equal descriptors still admit the same values, but two that admit
/// the same values may compare unequal. Every law over a descriptor that can
/// hold one is therefore checked against the values, and the laws checked by
/// equality are the ones over the components that are canonical.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Descr {
    /// One component per kind, indexed by that kind's position in [`Kind::ALL`].
    kinds: [Component; Kind::ALL.len()],
    /// The values of no listed kind -- a class instance, a callable, a generator.
    ///
    /// Described by the attributes they carry, which is what the core can see of
    /// them. What it cannot see is their *class*: only the bindings can compare
    /// two of those, so the class half of a schema like `Attrs` stays outside
    /// this and meets it from there. The slot exists so a complement means what
    /// it says: the complement of `int` holds every non-int value, not merely
    /// the ones this partition names.
    other: Component,
}

impl Descr {
    /// The empty set.
    #[must_use]
    pub fn nothing() -> Descr {
        Descr {
            kinds: Kind::ALL.map(Component::bottom),
            other: Component::Records(RecordLattice::empty()),
        }
    }

    /// Every value.
    #[must_use]
    pub fn anything() -> Descr {
        Descr {
            kinds: Kind::ALL.map(Component::top),
            other: Component::Records(RecordLattice::all()),
        }
    }

    /// Every value of `kind` and nothing else.
    #[must_use]
    pub fn of_kind(kind: Kind) -> Descr {
        let mut descr = Descr::nothing();
        descr.put(kind, Component::top(kind));
        descr
    }

    /// The singleton holding one boolean.
    #[must_use]
    pub fn boolean(value: bool) -> Descr {
        let mut descr = Descr::nothing();
        descr.put(Kind::Bool, Component::Booleans(BoolSet::just(value)));
        descr
    }

    /// The singleton holding one integer.
    #[must_use]
    pub fn integer(value: i64) -> Descr {
        let mut descr = Descr::nothing();
        descr.put(Kind::Int, Component::Integers(IntSet::just(value)));
        descr
    }

    /// The singleton holding one float, which is empty for `nan`.
    #[must_use]
    pub fn float(value: f64) -> Descr {
        let mut descr = Descr::nothing();
        descr.put(Kind::Float, Component::Floats(FloatSet::just(value)));
        descr
    }

    /// The words one pattern matches whole, for a word kind.
    ///
    /// `None` where the pattern does not build, where its automaton is past the
    /// bound, or where the kind is not a word kind: a pattern over a kind that
    /// has no words is a caller error rather than a set.
    #[must_use]
    pub fn pattern(pattern: &str, kind: Kind) -> Option<Descr> {
        let language = RegularSet::pattern(pattern, alphabet_of(kind)?)?;
        let mut descr = Descr::nothing();
        descr.put(kind, Component::Words(language));
        Some(descr)
    }

    /// The one-word set, for a word kind. A `str` is its UTF-8 bytes.
    #[must_use]
    pub fn word(word: &[u8], kind: Kind) -> Option<Descr> {
        alphabet_of(kind)?;
        let mut descr = Descr::nothing();
        descr.put(kind, Component::Words(RegularSet::word(word)));
        Some(descr)
    }

    /// The sequences a shape spells, for a sequence kind.
    ///
    /// One constructor for the three spellings, because they are one shape with
    /// different parts filled in. `tuple[A, B]` is a prefix and no tail;
    /// `list[T]` is no prefix and a tail; `tuple[A, *tuple[B, ...], C]` is a
    /// prefix, a tail and a prefix, which is the same chain with a loop in it.
    ///
    /// `None` where the kind's values are not sequences, which is a caller error
    /// rather than a set.
    #[must_use]
    pub fn sequence(prefix: &[Descr], tail: Option<&Descr>, kind: Kind) -> Option<Descr> {
        if !is_sequence(kind) {
            return None;
        }
        let mut descr = Descr::nothing();
        descr.put(kind, Component::Sequences(SymbolicDfa::shape(prefix, tail)));
        Some(descr)
    }

    /// The sets whose members all lie in `elements`, for a set kind.
    ///
    /// The members are first cut down to what a set can *hold*. A set's members
    /// are hashed, and a list, a set and a dict are not hashable, so `elements`
    /// meets the hashable values before the powerset is taken. That is what makes
    /// `set[list[int]]` the same set as `set[nothing]`: neither holds a list, so
    /// both hold exactly one value, the empty set.
    ///
    /// The cut is coarser than Python's rule in one place, and coarser in the
    /// direction that declines rather than admits: a tuple is hashable only when
    /// its elements are, and this keeps every tuple. So `set[tuple[list[int]]]`
    /// reads as inhabited by more than the empty set when it is not, which
    /// leaves a question undecided rather than answering it wrongly.
    ///
    /// `None` where the kind's values are not sets, or where a component cannot
    /// hold the meet.
    #[must_use]
    pub fn set(elements: &Descr, kind: Kind) -> Option<Descr> {
        if !is_set(kind) {
            return None;
        }
        let members = elements.intersect(&hashable())?;
        let mut descr = Descr::nothing();
        descr.put(kind, Component::Sets(SetLattice::of(members)));
        Some(descr)
    }

    /// The objects carrying `label`, whose value is in `ty`.
    ///
    /// A value of no listed kind is described by the attributes it carries, and
    /// the record is always **open**: an object may carry attributes no schema
    /// mentions, so naming one constrains that attribute and no other. That is
    /// what makes a complement finite -- an object fails by holding a *named*
    /// attribute outside its type, so the complement splits over the labels.
    ///
    /// `optional` admits the objects that do not carry the attribute at all,
    /// which is membership of the undefined value rather than a rule beside the
    /// type.
    #[must_use]
    pub fn attribute(label: &str, ty: &Descr, optional: bool) -> Descr {
        let mut descr = Descr::nothing();
        descr.other = Component::Records(RecordLattice::attribute(label, ty.clone(), optional));
        descr
    }

    /// The objects that are instances of `class`.
    ///
    /// Only a **pure** class belongs here -- one whose metaclass leaves
    /// `isinstance` and `issubclass` alone. A class with a hook answers
    /// arbitrary code and is not a set this algebra holds.
    #[must_use]
    pub fn instance_of(class: Class) -> Descr {
        let mut descr = Descr::nothing();
        descr.other = Component::Records(RecordLattice::instance_of(class));
        descr
    }

    /// The objects that do not carry `label` at all.
    #[must_use]
    pub fn without_attribute(label: &str) -> Descr {
        let mut descr = Descr::nothing();
        descr.other = Component::Records(RecordLattice::without(label));
        descr
    }

    /// The integers that are multiples of `step`, or `None` where the integer
    /// set cannot hold that step.
    ///
    /// The refusal is carried up rather than absorbed: a descriptor that
    /// silently widened here would be complemented into one that is wrong the
    /// other way, and the caller is the one that can decide to keep the step
    /// opaque instead.
    #[must_use]
    pub fn multiple_of(step: i64) -> Option<Descr> {
        let mut descr = Descr::nothing();
        descr.put(Kind::Int, Component::Integers(IntSet::multiple_of(step)?));
        Some(descr)
    }

    fn position(kind: Kind) -> usize {
        Kind::ALL
            .iter()
            .position(|listed| *listed == kind)
            .unwrap_or(0)
    }

    fn component(&self, kind: Kind) -> &Component {
        self.kinds
            .get(Descr::position(kind))
            .unwrap_or(&Component::Coarse(false))
    }

    fn put(&mut self, kind: Kind, component: Component) {
        if let Some(slot) = self.kinds.get_mut(Descr::position(kind)) {
            *slot = component;
        }
    }

    /// Every value in either set, or `None` where a component cannot hold the
    /// result.
    ///
    /// Fallible because one component is: a regular language has an automaton
    /// bound, and past it there is no sound set to substitute -- one too wide is
    /// complemented into one too narrow. The refusal reaches the caller rather
    /// than being absorbed into a descriptor that is quietly wrong.
    #[must_use]
    pub fn union(&self, other: &Descr) -> Option<Descr> {
        self.zip(other, Op::Union)
    }

    /// Every value in both sets, or `None` where a component cannot hold the
    /// result.
    #[must_use]
    pub fn intersect(&self, other: &Descr) -> Option<Descr> {
        self.zip(other, Op::Intersect)
    }

    /// Every value this set does not hold.
    ///
    /// Componentwise, which is what makes it *saturate*: complementing a
    /// descriptor that holds only ints gives one holding every value of every
    /// other kind, because each of those components was empty and is now full.
    /// A representation that carried only the kinds it mentions would have to
    /// name the rest here, and would name the wrong set the day a kind is added.
    #[must_use]
    pub fn complement(&self) -> Descr {
        Descr {
            kinds: self.kinds.each_ref().map(Component::complement),
            other: self.other.complement(),
        }
    }

    fn zip(&self, other: &Descr, op: Op) -> Option<Descr> {
        let mut kinds = self.kinds.clone();
        for (slot, theirs) in kinds.iter_mut().zip(&other.kinds) {
            *slot = slot.combine(theirs, op)?;
        }
        Some(Descr {
            kinds,
            other: self.other.combine(&other.other, op)?,
        })
    }

    /// Whether this set admits no value.
    ///
    /// Every component empty, which is the whole emptiness decision at this
    /// resolution: the kinds partition the universe, so a value the descriptor
    /// admits is a value some component admits.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.kinds.iter().all(Component::is_empty) && self.other.is_empty()
    }

    /// Whether this set admits `value`.
    #[must_use]
    pub fn admits(&self, value: Value) -> bool {
        let component = match value.kind {
            Some(kind) => self.component(kind),
            None => &self.other,
        };
        match component {
            Component::Coarse(present) => *present,
            Component::Booleans(set) => value.boolean.is_some_and(|b| set.holds(b)),
            Component::Integers(set) => value.integer.is_some_and(|i| set.holds(i)),
            Component::Floats(set) => value.float.is_some_and(|f| set.holds(f)),
            Component::Words(set) => value.word.is_some_and(|w| set.holds(w)),
            Component::Sequences(set) => value.elements.is_some_and(|e| set.holds(e)),
            Component::Sets(set) => value.elements.is_some_and(|e| set.holds(e)),
            Component::Records(set) => value.attributes.is_some_and(|a| set.holds(value.class, a)),
        }
    }
}

/// A descriptor is a letter of the sequence automaton, which is what makes the
/// component recursive.
///
/// The three operations and emptiness are the ones above; the trait is the
/// interface the automaton asks a letter for, and nothing here is new work. It
/// is the *fallibility* that shows through: a guard that cannot join leaves the
/// table coarser rather than wrong, which is why the automaton's minimisation
/// asks for a join and accepts a refusal.
impl Guard for Descr {
    type Value = Value;

    fn none() -> Descr {
        Descr::nothing()
    }

    fn meet(&self, other: &Descr) -> Option<Descr> {
        self.intersect(other)
    }

    fn join(&self, other: &Descr) -> Option<Descr> {
        self.union(other)
    }

    fn complement(&self) -> Descr {
        Descr::complement(self)
    }

    fn is_empty(&self) -> bool {
        Descr::is_empty(self)
    }

    fn holds(&self, value: &Value) -> bool {
        self.admits(*value)
    }
}

/// A value, at the resolution the descriptor distinguishes.
///
/// Not a Python object: the core cannot see one. It is the *questions* a
/// descriptor can currently answer about a value -- which kind it belongs to,
/// and, where the kind's component distinguishes its values, which value it is.
/// Each commit that makes a component exact widens this alongside it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Value {
    /// The kind, or `None` for a value of no listed kind.
    pub kind: Option<Kind>,
    /// Which boolean, where the kind is [`Kind::Bool`].
    pub boolean: Option<bool>,
    /// Which integer, where the kind is [`Kind::Int`].
    pub integer: Option<i64>,
    /// Which float, where the kind is [`Kind::Float`].
    pub float: Option<f64>,
    /// Which word, where the kind is [`Kind::Str`] or [`Kind::Bytes`]. A `str`
    /// is its UTF-8 bytes, which is the same alphabet the language is over.
    pub word: Option<&'static [u8]>,
    /// The elements, where the kind is a sequence or a set kind.
    ///
    /// A value again, which is what makes the question recursive: whether a
    /// sequence is admitted is asked of its elements, one letter at a time.
    /// For a set kind these are its *members*, in no particular order -- a set
    /// is its members and the powerset rule reads them as a whole rather than
    /// as a word.
    /// The class, where the value is of no listed kind.
    ///
    /// What the value *is*, as far as the snapshot of the class order can say.
    /// A value with none is one whose class the core was never told, which every
    /// class constraint declines rather than admits.
    pub class: Option<&'static Class>,
    /// The attributes, where the value is of no listed kind.
    ///
    /// What an object *is*, at this resolution: the names it carries and what
    /// each one holds. An attribute absent from the list is one the object does
    /// not carry, which an open record reads as the undefined value.
    pub attributes: Option<&'static [(&'static str, Value)]>,
    pub elements: Option<&'static [Value]>,
}

impl Value {
    /// A representative value of `kind`, for a kind whose component is coarse.
    #[must_use]
    pub const fn of_kind(kind: Kind) -> Value {
        Value {
            kind: Some(kind),
            boolean: None,
            integer: None,
            float: None,
            word: None,
            elements: None,
            attributes: None,
            class: None,
        }
    }

    /// One of the two booleans.
    #[must_use]
    pub const fn boolean(value: bool) -> Value {
        Value {
            kind: Some(Kind::Bool),
            boolean: Some(value),
            integer: None,
            float: None,
            word: None,
            elements: None,
            attributes: None,
            class: None,
        }
    }

    /// One integer. `bool` is a kind of its own, so this is never a boolean.
    #[must_use]
    pub const fn integer(value: i64) -> Value {
        Value {
            kind: Some(Kind::Int),
            boolean: None,
            integer: Some(value),
            float: None,
            word: None,
            elements: None,
            attributes: None,
            class: None,
        }
    }

    /// One word, for a word kind. A `str` is its UTF-8 bytes, which is the
    /// alphabet the language is over.
    #[must_use]
    pub const fn word(word: &'static [u8], kind: Kind) -> Value {
        Value {
            kind: Some(kind),
            boolean: None,
            integer: None,
            float: None,
            word: Some(word),
            elements: None,
            attributes: None,
            class: None,
        }
    }

    /// One float, `nan` included.
    #[must_use]
    pub const fn float(value: f64) -> Value {
        Value {
            kind: Some(Kind::Float),
            boolean: None,
            integer: None,
            float: Some(value),
            word: None,
            elements: None,
            attributes: None,
            class: None,
        }
    }

    /// A value of no listed kind carrying no attribute anyone named.
    #[must_use]
    pub const fn other() -> Value {
        Value::object(&[])
    }

    /// An instance of `class`, described by that and the attributes it carries.
    #[must_use]
    pub const fn instance(
        class: &'static Class,
        attributes: &'static [(&'static str, Value)],
    ) -> Value {
        Value {
            class: Some(class),
            ..Value::object(attributes)
        }
    }

    /// A value of no listed kind -- a class instance, a callable -- described by
    /// the attributes it carries.
    #[must_use]
    pub const fn object(attributes: &'static [(&'static str, Value)]) -> Value {
        Value {
            kind: None,
            boolean: None,
            integer: None,
            float: None,
            word: None,
            elements: None,
            attributes: Some(attributes),
            class: None,
        }
    }
    /// One sequence, for a sequence kind. The elements are values again, which
    /// is the letters the automaton reads.
    #[must_use]
    pub const fn sequence(elements: &'static [Value], kind: Kind) -> Value {
        Value {
            kind: Some(kind),
            boolean: None,
            integer: None,
            float: None,
            word: None,
            elements: Some(elements),
            attributes: None,
            class: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{BoolSet, Class, Component, Descr, Value};
    use crate::decision::Kind;
    use proptest::prelude::*;
    use std::sync::LazyLock;

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
        // Objects of no listed kind, described by the attributes they carry.
        // The one with an attribute nobody names is what holds the record open.
        values.extend(OBJECTS.map(Value::object));
        // The same objects under each class of the little order below, plus the
        // one whose class nobody told us.
        for class in [&ANIMAL, &DOG, &MINERAL] {
            values.extend(OBJECTS.map(|attributes| Value::instance(class, attributes)));
        }
        values
    }

    /// A class order small enough to enumerate and wide enough to separate the
    /// three answers: deriving, unrelated, and laid out apart.
    static ANIMAL: LazyLock<Class> = LazyLock::new(|| Class::root(1));
    static DOG: LazyLock<Class> = LazyLock::new(|| Class::new(2, 1, std::slice::from_ref(&ANIMAL)));
    static MINERAL: LazyLock<Class> = LazyLock::new(|| Class::root(3));

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

    /// The record the report calls carrier-free: a set of objects fixed by the
    /// attributes alone, with no class in it.
    ///
    /// This is what `Attrs` could not say. A dataclass `D(x: int)` is the meet
    /// of a class and this record; the record on its own is the half that lives
    /// in the algebra, and it is the half that makes `¬Attrs` representable --
    /// the complement of an attribute constraint is another one.
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

    /// The same descriptors, plus the sequences, sets and objects the three
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

    /// A coarse component is all-or-nothing, and the tests must not read that as
    /// a distinction the descriptor makes.
    #[test]
    fn a_coarse_kind_admits_all_of_its_values_or_none() {
        let dicts = Descr::of_kind(Kind::Dict);
        assert!(dicts.admits(Value::of_kind(Kind::Dict)));
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
    #[test]
    fn a_component_is_bottom_or_top_of_its_own_kind() {
        for kind in Kind::ALL {
            assert!(Component::bottom(kind).is_empty(), "{kind:?} bottom");
            assert!(!Component::top(kind).is_empty(), "{kind:?} top");
            assert_eq!(Component::bottom(kind).complement(), Component::top(kind));
            assert_eq!(Component::top(kind).complement(), Component::bottom(kind));
        }
    }
}
