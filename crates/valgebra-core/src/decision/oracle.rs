//! The oracle: what the core cannot decide alone, asked of whoever holds the
//! objects.
//!
//! A class hierarchy and a concrete value are facts of the host, and the core
//! cannot see it. Every relation that turns on one -- whether a literal is a
//! member of a class, whether two classes share an instance, what kind a
//! constant is of -- is asked through this trait, and the answer is a proof
//! or `None`: an implementor that cannot say leaves the relation undecided,
//! and the procedure stays sound by treating undecided as "not proved". The
//! default oracle decides nothing, which is the core's own reading of every
//! such question.
//!
//! Each question is a method rather than a callback so the shipped page can
//! list what the core asks and the mutation sweep can name the arms an answer
//! is allowed to take.

use crate::descr::lower::Constants;
use crate::ir::{ClassIx, ConstIx, OperandIx, Schema};
use crate::kind::Kind;

/// Resolves the leaf relations the structural subtyping decision cannot.
///
/// Those are the ones that depend on the Python class hierarchy (an `Instance`)
/// or on a concrete value (a `Literal`). The bindings implement it with
/// `issubclass` and membership; the core defaults to [`NoLeafRelations`].
///
/// It carries [`Constants`] because both are the same question asked twice: the
/// object table lives in the bindings, and an implementor that can answer what a
/// `Literal` is a subtype of can also say what it *names*. Requiring it here is
/// what lets a relation lower its two schemas -- a descriptor holding a constant
/// as a value decides `MultipleOf(4) ≤ MultipleOf(2)`, which no rule about
/// constraints reaches -- rather than declining every schema that names one.
pub trait LeafRelations: Constants {
    /// Whether leaf schema `sub` is a subtype of `sup`, or `None` to leave the
    /// relation conservatively undecided.
    fn leaf_subtype(&self, sub: &Schema, sup: &Schema) -> Option<bool>;

    /// Order the two pool values behind refinement bounds at indices `left` and
    /// `right`, or `None` when the core cannot or the values are not comparable.
    /// The default decides nothing, so bound satisfiability stays conservative.
    fn compare(&self, _left: OperandIx, _right: OperandIx) -> Option<core::cmp::Ordering> {
        None
    }

    /// Whether two *sets* of literal constants share no value, or `None` to
    /// leave it to the pairwise question.
    ///
    /// The same relation [`literals_disjoint`](Self::literals_disjoint) answers
    /// for one pair, asked of every pair at once. The core walks members against
    /// members, which is quadratic in the oracle for two wide literal unions --
    /// a shape a contract really writes, since an enumeration of codes is one.
    /// An implementor that can hash its constants answers in one pass; the
    /// default declines, and the walk stands.
    fn literal_sets_disjoint(&self, _left: &[ConstIx], _right: &[ConstIx]) -> Option<bool> {
        None
    }

    /// Whether no integer lies between the pool values at `lo` and `hi`, under the
    /// strictness of each bound (`lo_strict` excludes `lo`, `hi_strict` excludes
    /// `hi`). The core asks this only for an integer-discrete refinement base, so a
    /// `Some(true)` proves the interval admits no integer and the refinement is
    /// empty. `None` leaves the discreteness rule conservative — the default, so a
    /// core with no value oracle never decides on integer adjacency.
    fn no_int_between(
        &self,
        _lo: OperandIx,
        _lo_strict: bool,
        _hi: OperandIx,
        _hi_strict: bool,
    ) -> Option<bool> {
        None
    }

    /// Whether an atom the core cannot read denotes a *set* -- the same values
    /// however often it is asked -- or `None` when the bindings cannot say.
    ///
    /// Asked of a class: `isinstance` against a metaclass that overrides
    /// `__instancecheck__` runs user code, so two occurrences of one class can
    /// disagree and `A ∩ ¬A` is not empty. Telling a pure class from a hooked one
    /// needs the class object, which only the bindings hold. The default decides
    /// nothing, so a core with no oracle treats every such atom as one it cannot
    /// reason about -- the conservative direction, which declines a law rather
    /// than applying it where it does not hold.
    fn atom_denotes_a_set(&self, _atom: &Schema) -> Option<bool> {
        None
    }

    /// The kind of the pooled constant behind a [`Schema::Literal`], or `None`
    /// when the bindings decline to kind it.
    ///
    /// A literal denotes `{x | type(x) is type(c) and x == c}`, so its kind is
    /// the kind of `c`'s type and the core cannot read it. Answering places the
    /// literal in the partition, which is what decides it against another kind.
    /// The default declines, so a core with no value oracle stays conservative.
    fn literal_kind(&self, _constant: ConstIx) -> Option<Kind> {
        None
    }

    /// Whether a value of `kind` can be an instance of the class behind a
    /// [`Schema::Instance`], or `None` when the bindings decline to say.
    ///
    /// A class is the one atom the core cannot read at all, and disjointness is
    /// the question it most often needs answered about one: a value of a kind
    /// the class cannot hold is a value the class's set does not contain. The
    /// question is asked this way round, rather than as "what kind is this
    /// class", because a class need not have one -- a class deriving from no
    /// builtin lays down no kind, and *that* is the answer that decides, since
    /// its instances are none of the kinds the partition names.
    ///
    /// `Some(false)` is the only answer that refutes, so an implementor may
    /// answer `Some(true)` for every kind its class could conceivably hold. It
    /// must decline for a class whose `isinstance` runs user code, where
    /// membership is not a property of the value's type at all. The default
    /// declines, so a core with no oracle keeps every class conservative.
    fn class_admits_kind(&self, _class: ClassIx, _kind: Kind) -> Option<bool> {
        None
    }

    /// Whether a value whose *type is* the pooled class has `kind`.
    ///
    /// The narrower question beside [`class_admits_kind`](Self::class_admits_kind),
    /// and the one a refutation can stand on. That one reads the whole subtree
    /// -- a subclass may derive from a builtin, so a plain class *admits* every
    /// kind -- and declines for a plain class because a claim there would be
    /// unsound. This one asks about `type(v) is C` alone, where no subclass can
    /// interfere: a direct instance of a class laying down no builtin layout is
    /// a plain object and has none of the kinds the partition names.
    ///
    /// Both answers are used, so an implementor must not widen either. It must
    /// decline for a class whose `isinstance` runs user code, and for a class
    /// it cannot read.
    fn direct_instance_of_kind(&self, _class: ClassIx, _kind: Kind) -> Option<bool> {
        None
    }

    /// Whether every value of `kind` is an instance of the pooled class.
    ///
    /// The dual of [`direct_instance_of_kind`](Self::direct_instance_of_kind),
    /// asked of the kind's own builtin: a value of `kind` built as that builtin
    /// has `type(v)` equal to it, so the order between the builtin and the
    /// class settles `isinstance`. `Some(false)` refutes -- the kind holds a
    /// value the class does not -- and it is the only answer that does.
    ///
    /// It must decline for a class whose `isinstance` runs user code. The
    /// default declines, so a core with no oracle keeps every class
    /// conservative.
    fn kind_derives_from(&self, _kind: Kind, _class: ClassIx) -> Option<bool> {
        None
    }

    /// Whether the two pooled constants behind a pair of [`Schema::Literal`]s
    /// denote disjoint singletons, or `None` when it cannot be settled soundly.
    ///
    /// Two literals share no value when their constants have different types --
    /// a literal pins `type(x)` exactly, so `Literal[1]` and `Literal[True]` are
    /// disjoint although `1 == True` -- or when the types are the same and the
    /// values differ under an equality the bindings trust. They must decline for
    /// a type carrying user-defined equality, where two distinct constants may
    /// still admit one value. The default declines.
    fn literals_disjoint(&self, _left: ConstIx, _right: ConstIx) -> Option<bool> {
        None
    }
}

/// The trivial [`LeafRelations`] that decides nothing — the core default, under
/// which `Instance` and `Literal` relations stay conservative.
pub struct NoLeafRelations;

impl LeafRelations for NoLeafRelations {
    fn leaf_subtype(&self, _sub: &Schema, _sup: &Schema) -> Option<bool> {
        None
    }
}

/// An oracle that decides nothing reads no pool either, so a schema naming a
/// constant refuses to lower here and is decided by the rules alone.
impl Constants for NoLeafRelations {}
