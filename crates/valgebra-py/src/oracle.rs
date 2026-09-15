//! The binding's half of the `LeafRelations` contract: the questions
//! `valgebra-core` cannot decide alone.
//!
//! A class hierarchy and a concrete value are Python facts, and the core cannot
//! see Python. It asks through a trait, and this is the implementation the
//! bindings give it: whether a literal belongs to a set, whether two sets of
//! constants share a value, how two refinement bounds order, whether a class is
//! an enumeration and which values it lists. The section "What the core cannot
//! decide alone" in `docs/dev/02-decision.md` is this module, and
//! `decision/oracle.rs` is the core's side of the same seam.
//!
//! It lives beside the validator rather than inside it because the theory's
//! fourth question is about placement: a citation can be load-bearing and still
//! sit where nothing it governs lives. The `Validator` holds the pool these
//! answers read; it is not the thing the contract is about.

use std::cell::RefCell;

use pyo3::PyTypeInfo;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{
    PyBool, PyBytes, PyDict, PyFloat, PyFrozenSet, PyInt, PyList, PySet, PyString, PyTuple, PyType,
};
use rustc_hash::FxHashMap;
use valgebra_core::descr::classes::Class;
use valgebra_core::descr::lower::{Constants, Operand};
use valgebra_core::{ClassIx, ConstIx, Kind, LeafRelations, OperandIx, Schema};

use crate::check::{Ctx, Frame, ValidatorIndex, WalkMode, WalkState, member};
use crate::input::Value;

/// A [`LeafRelations`] oracle backed by a validator's constant pool. It decides
/// a `Literal` subtyping by running membership of the literal's value against
/// the candidate supertype, and an `Instance`-versus-`Instance` subtyping by
/// `issubclass` on the pooled classes.
pub(crate) struct PoolRelations<'py, 'pool> {
    py: Python<'py>,
    literals: &'pool [Py<PyAny>],
    definitions: &'pool [Schema],
    /// The dense id each class has been given, keyed by the type object.
    ///
    /// [`Class`] identifies a class by a small integer, and the only stable name
    /// a Python class has is its address -- which does not fit one. So an id is
    /// handed out on first sight and remembered for the rest of the query, which
    /// is as long as any descriptor built from it lives. The pool holds every
    /// class it names alive, so an address cannot be reused while an id for it
    /// is outstanding.
    classes: RefCell<FxHashMap<usize, u32>>,
}

impl<'py, 'pool> PoolRelations<'py, 'pool> {
    /// The oracle for one query over one validator's pool.
    ///
    /// The class-id map starts empty and lives as long as the query, which is
    /// what makes an address a safe key: the pool holds every class it names
    /// alive, so none can be freed and its address reused while this answers.
    /// A caller passes what it has and does not have to know that.
    pub(crate) fn new(
        py: Python<'py>,
        literals: &'pool [Py<PyAny>],
        definitions: &'pool [Schema],
    ) -> Self {
        Self {
            py,
            literals,
            definitions,
            classes: RefCell::default(),
        }
    }
}

/// The deepest structural nesting a constructed schema may reach. A real schema
/// is nowhere near this deep and the annotation frontend caps its own nesting
/// lower, so a validator this deep is one built in an unbounded loop. Every
/// recursive walk over the tree — clone, drop, the decision procedure — descends
/// one native stack frame per level, so building past this bound returns an
/// error rather than overflowing the stack. Structural recursion in a schema is
/// written with `recursive`, whose back edge is a `Ref` leaf and does not count
/// toward this depth.
/// The `math.floor` and `math.ceil` callables, imported once per interpreter for
/// the integer-interval emptiness rule rather than re-imported on every decision.
/// `PyOnceLock` keeps the one-time initialization sound under free-threading.
static MATH_FLOOR_CEIL: PyOnceLock<(Py<PyAny>, Py<PyAny>)> = PyOnceLock::new();

fn math_floor_ceil(py: Python<'_>) -> PyResult<&'static (Py<PyAny>, Py<PyAny>)> {
    MATH_FLOOR_CEIL.get_or_try_init(py, || {
        let math = py.import("math")?;
        Ok((
            math.getattr("floor")?.unbind(),
            math.getattr("ceil")?.unbind(),
        ))
    })
}

/// `enum.EnumMeta`, imported once per interpreter rather than per question.
///
/// The subtype rule asks whether a class is an enumeration for *every* class it
/// cannot place in a union, so importing a module there would put an import on
/// a decision path.
static ENUM_TYPES: PyOnceLock<(Py<PyAny>, Py<PyAny>)> = PyOnceLock::new();

/// `(enum.EnumMeta, enum.Flag)`, imported once per interpreter.
///
/// Both are needed at the same place and neither on any other path, so they
/// share one lock. `Flag` is here because it is the one enumeration whose
/// instances are *not* the members it lists: `P.A | P.B` is an instance of `P`
/// that `list(P)` never yields.
fn enum_types(py: Python<'_>) -> Option<&(Py<PyAny>, Py<PyAny>)> {
    ENUM_TYPES
        .get_or_try_init(py, || {
            let module = py.import("enum")?;
            let meta = module.getattr(intern!(py, "EnumMeta"))?.unbind();
            let flag = module.getattr(intern!(py, "Flag"))?.unbind();
            Ok::<_, PyErr>((meta, flag))
        })
        .ok()
}

/// The most members an enumeration may have before it is read as an atom.
///
/// Reading one asks a membership question per member, so a class with thousands
/// would turn one relation into thousands of walks. Far past any enumeration
/// anybody writes -- the ones in a contract are error codes and states -- and
/// past it the class stays what it was: an `isinstance` atom, decided as one.
const MAX_ENUM_MEMBERS: usize = 512;

/// Whether values of this type are equal only when they are the same object.
///
/// `object.__eq__` is identity, so a type that neither defines `__eq__` nor
/// inherits one from a type that does compares by identity -- and then two
/// distinct constants of it are two values, which is what a literal needs to be
/// decided against another literal. An `IntEnum` inherits `int.__eq__` and is
/// refused here, which is right: its members are equal to the integers they
/// carry.
fn compares_by_identity(ty: &Bound<'_, PyType>) -> bool {
    let py = ty.py();
    let Ok(theirs) = ty.getattr(intern!(py, "__eq__")) else {
        return false;
    };
    PyAny::type_object(py)
        .getattr(intern!(py, "__eq__"))
        .is_ok_and(|inherited| theirs.is(&inherited))
}

impl PoolRelations<'_, '_> {
    /// The members of an enumeration that *is* the union of them, if this class
    /// is one.
    ///
    /// Reading a class as the union of `list(cls)` is sound only when every
    /// instance of the class is one of the values listed, and that takes four
    /// things:
    ///
    /// * it is an enumeration, so the members are fixed when the class is
    ///   created;
    /// * it is **not a `Flag`**. A flag's `|` builds instances the class never
    ///   listed: `P.A | P.B` is an instance of `P`, and `list(P)` is
    ///   `[P.A, P.B]`. Reading `P` as that union made `P <= Literal[P.A, P.B]`
    ///   true with `P.A | P.B` standing against it, and `P & ~Literal[P.A,
    ///   P.B]` was not decided empty although it admits that value.
    /// * it **has at least one member**. An enumeration with none can still be
    ///   subclassed -- that is how an enum base class is written -- so its
    ///   instances are its subclasses' members, and reading it as the empty
    ///   union made it a subtype of `nothing`.
    /// * two members are two values, which is the identity check. An `IntEnum`
    ///   fails it because its members equal the integers behind them.
    ///
    /// A class failing any of them stays the `isinstance` atom it was, which is
    /// sound for every enumeration and merely less complete.
    fn enum_members<'py>(class: &Bound<'py, PyAny>) -> Option<Vec<Bound<'py, PyAny>>> {
        let py = class.py();
        let class = class.cast::<PyType>().ok()?;
        let (meta, flag) = enum_types(py)?;
        if !class.is_instance(meta.bind(py)).ok()? {
            return None;
        }
        if class.is_subclass(flag.bind(py)).ok()? {
            return None;
        }
        if !compares_by_identity(class) {
            return None;
        }
        let members: Vec<Bound<'py, PyAny>> = class
            .try_iter()
            .ok()?
            .take(MAX_ENUM_MEMBERS + 1)
            .collect::<Result<_, _>>()
            .ok()?;
        (!members.is_empty() && members.len() <= MAX_ENUM_MEMBERS).then_some(members)
    }

    fn is_member(&self, schema: &Schema, value: &Bound<'_, PyAny>) -> bool {
        // These leaf-subtype probes run on transient schemas during compilation,
        // not on a finished validator, so they carry no precomputed index; the
        // walk falls back to its general path for any record or union here. A
        // fatal signal in a probe folds to non-membership here (the decision
        // procedure is not the interruptible hot path); the state is local.
        let state = WalkState::new();
        let index = ValidatorIndex::default();
        let ctx = Ctx {
            pool: self.literals,
            defs: self.definitions,
            records: &index.records,
            attrs: &index.attrs,
            unions: &index.unions,
            regexes: &index.regexes,
            guard: &state.guard,
            depth: &state.depth,
            fatal: &state.fatal,
            fatal_seen: &state.fatal_seen,
            mode: WalkMode::Fast,
        };
        member(
            schema,
            &Value::Py(value),
            &mut Frame::new(&mut Vec::new(), &mut Vec::new(), ctx),
        )
    }
}

/// The builtin type whose direct values have `kind`.
///
/// The inverse of the table [`layout_of`] scans, and the two are the same fact
/// read in the two directions a class question needs. `NoneType` has no builtin
/// to name here -- its one value is a singleton rather than a constructor's --
/// so it declines.
fn builtin_of(py: Python<'_>, kind: Kind) -> Option<Bound<'_, PyType>> {
    Some(match kind {
        Kind::Bool => PyBool::type_object(py),
        Kind::Int => PyInt::type_object(py),
        Kind::Str => PyString::type_object(py),
        Kind::Bytes => PyBytes::type_object(py),
        Kind::Float => PyFloat::type_object(py),
        Kind::Tuple => PyTuple::type_object(py),
        Kind::FrozenSet => PyFrozenSet::type_object(py),
        Kind::List => PyList::type_object(py),
        Kind::Set => PySet::type_object(py),
        Kind::Dict => PyDict::type_object(py),
        Kind::NoneType => return None,
    })
}

/// The builtin base a class is built on: the layout tag [`Class`] reads, and the
/// kind that layout confines an instance to.
///
/// Python refuses `class C(int, str)` -- "multiple bases have instance lay-out
/// conflict" -- so a class built on one of these derives from no other, and two
/// classes built on different ones share no instance. A class built on none of
/// them lays down no layout of its own and takes [`Class::PLAIN`] and no kind,
/// which conflicts with nothing and confines nothing: `class Both(Plain, MyStr)`
/// builds, so a plain class and a `str` subclass do share instances, and those
/// instances are strings.
fn layout_of(ty: &Bound<'_, PyType>) -> (u32, Option<Kind>) {
    let py = ty.py();
    [
        (PyInt::type_object(py), Kind::Int),
        (PyString::type_object(py), Kind::Str),
        (PyBytes::type_object(py), Kind::Bytes),
        (PyFloat::type_object(py), Kind::Float),
        (PyTuple::type_object(py), Kind::Tuple),
        (PyFrozenSet::type_object(py), Kind::FrozenSet),
        (PyList::type_object(py), Kind::List),
        (PySet::type_object(py), Kind::Set),
        (PyDict::type_object(py), Kind::Dict),
    ]
    .into_iter()
    .enumerate()
    // `bool` derives from `int` and shares its layout, so a `bool` and an
    // `int` subclass land in one part rather than two -- which is right: the
    // pair is disjoint for a reason this tag does not carry. It is also why the
    // kind beside the tag is `Int` and not `Bool`: `bool` is final, so an `int`
    // subclass is never a boolean.
    .find(|(_, (builtin, _))| ty.is_subclass(builtin).unwrap_or(false))
    .and_then(|(at, (_, kind))| u32::try_from(at).ok().map(|at| (at + 1, Some(kind))))
    .unwrap_or((Class::PLAIN, None))
}

impl PoolRelations<'_, '_> {
    /// The dense id this class carries for the rest of the query.
    fn class_id(&self, ty: &Bound<'_, PyType>) -> u32 {
        let mut seen = self.classes.borrow_mut();
        let next = u32::try_from(seen.len()).unwrap_or(u32::MAX);
        *seen.entry(ty.as_ptr() as usize).or_insert(next)
    }

    /// The order snapshot for a class: its own id, its layout, and the id of
    /// every class it derives from.
    ///
    /// Read off `__mro__` once, rather than by asking `issubclass` again later,
    /// because the relation can move -- `ABC.register` rewrites it after a schema
    /// is built -- and a relation that moves is not an order to reason in. The
    /// snapshot is what the descriptor carries instead.
    ///
    /// Declines for a class that does not denote a set, on the same test
    /// [`Self::atom_denotes_a_set`] applies, so an impure class refuses the
    /// lowering rather than entering it as a set it is not.
    fn snapshot(&self, ty: &Bound<'_, PyType>) -> Option<Class> {
        if !self.denotes_a_set(ty)? {
            return None;
        }
        let mut bases = Vec::new();
        for base in ty.getattr("__mro__").ok()?.try_iter().ok()? {
            let base = base.ok()?;
            let base = base.cast_into::<PyType>().ok()?;
            // A base need not denote a set for the order to hold: what `is_a`
            // reads is which classes an instance is one of, and `__mro__` answers
            // that whatever the metaclass does at a check.
            //
            // Each base enters laying down no layout of its own, because
            // `__mro__` is already closed under derivation -- every ancestor is
            // on this list -- so the union over the list is the whole order, and
            // no base's own layout is read. Reading a base as *plain* rather
            // than as laid out on its own is what says that: a layout here would
            // be a claim about disjointness that this loop is not making.
            if !base.is(ty) {
                bases.push(Class::plain(self.class_id(&base)));
            }
        }
        let (layout, kind) = layout_of(ty);
        let class = Class::new(self.class_id(ty), layout, &bases);
        Some(match kind {
            Some(kind) => class.of_kind(kind),
            None => class,
        })
    }

    /// A pooled object as the descriptor reads one, or `None` for a value whose
    /// equality this cannot answer for.
    fn pooled(&self, slot: usize) -> Option<Operand> {
        let value = self.literals.get(slot)?.bind(self.py);
        if value.is_none() {
            return Some(Operand::NoneType);
        }
        if let Ok(ty) = value.cast::<PyType>() {
            return self.snapshot(ty).map(Operand::Instance);
        }
        // Exact types only, as `literal_kind` reads them: a subclass carries its
        // own `__eq__` and its own `__hash__`, and the sets the descriptor holds
        // are equality on the builtin scalars alone.
        let ty = value.get_type();
        if ty.is(PyBool::type_object(self.py)) {
            return value.extract::<bool>().ok().map(Operand::Boolean);
        }
        if ty.is(PyInt::type_object(self.py)) {
            return value.extract::<i64>().ok().map(Operand::Integer);
        }
        if ty.is(PyFloat::type_object(self.py)) {
            return value.extract::<f64>().ok().map(Operand::Float);
        }
        if ty.is(PyString::type_object(self.py)) {
            let text = value.extract::<String>().ok()?;
            return Some(Operand::Word(text.into_bytes(), Kind::Str));
        }
        if ty.is(PyBytes::type_object(self.py)) {
            let raw = value.extract::<Vec<u8>>().ok()?;
            return Some(Operand::Word(raw, Kind::Bytes));
        }
        None
    }

    /// Whether a class denotes a set.
    ///
    /// A class is **pure** when its metaclass leaves both `isinstance` and
    /// `issubclass` alone. Override either and the answer is user code: two
    /// occurrences of one class can disagree, so `A ∩ ¬A` is not empty and the
    /// law that says it is must not fire. `type` itself is the one metaclass
    /// known to answer from the class hierarchy and nothing else.
    ///
    /// An `abc.ABC` is excluded by the same test rather than by a second one:
    /// `ABCMeta` overrides both hooks, which is how `register` can change the
    /// relation after a schema is built.
    fn denotes_a_set(&self, class: &Bound<'_, PyAny>) -> Option<bool> {
        let metaclass = class.get_type();
        let plain = self.py.get_type::<PyType>();
        let untouched = |hook: &Bound<'_, PyString>| -> Option<bool> {
            Some(metaclass.getattr(hook).ok()?.is(&plain.getattr(hook).ok()?))
        };
        Some(
            untouched(intern!(self.py, "__instancecheck__"))?
                && untouched(intern!(self.py, "__subclasscheck__"))?,
        )
    }
}

/// The descriptor reads the object pool through the bindings, the one place a
/// Python object can be read at all.
impl Constants for PoolRelations<'_, '_> {
    fn operand(&self, index: OperandIx) -> Option<Operand> {
        self.pooled(index.get())
    }

    fn constant(&self, index: ConstIx) -> Option<Operand> {
        self.pooled(index.get())
    }

    fn class(&self, index: ClassIx) -> Option<Class> {
        let value = self.literals.get(index.get())?.bind(self.py);
        self.snapshot(value.cast::<PyType>().ok()?)
    }
}

impl LeafRelations for PoolRelations<'_, '_> {
    /// Whether a value of `kind` can be an instance of the pooled class.
    ///
    /// Read from the layout the class lays down. A class deriving from a
    /// builtin holds values of that builtin's kind and of no other, and a
    /// *subclass* of it cannot escape that: Python refuses a class body that
    /// would lay down a second layout, so the kind an instance has is fixed by
    /// the base and inherited by everything below it.
    ///
    /// A class laying down no layout is declined rather than answered. Its own
    /// instances are plain objects, but `isinstance` reads the whole subtree
    /// beneath it, and a subclass may derive from a builtin as well -- a class
    /// deriving from a plain class and from `str` is a `str` and an instance of
    /// the plain one. That is the direction a claim would be unsound in, and it
    /// is the case `tests/test_classes.py` was written for.
    ///
    /// A class whose `isinstance` runs user code is declined here as it is
    /// everywhere else: what such a class holds is not a property of a value's
    /// type at all.
    ///
    /// `true` is the conservative answer and is given where the layouts agree,
    /// without asking whether this particular class holds the value: a list is
    /// not an instance of every list subclass, and this question does not need
    /// it to be.
    fn class_admits_kind(&self, class: ClassIx, kind: Kind) -> Option<bool> {
        let value = self.literals.get(class.get())?.bind(self.py);
        let class = value.cast::<PyType>().ok()?;
        if !self.denotes_a_set(class)? {
            return None;
        }
        let (_, own) = layout_of(class);
        // `bool` lays down `int`'s layout, so a class laid out as an int may be
        // `bool` itself and hold booleans. Sound and coarse: the pair is never
        // refuted here.
        own.map(|own| own == kind || matches!((own, kind), (Kind::Int, Kind::Bool)))
    }
    /// Whether a value whose *type is* the pooled class has `kind`.
    ///
    /// The narrower question beside `class_admits_kind`, and the one that can
    /// refute. That one reads the whole subtree and so declines a class laying
    /// down no layout, because a subclass of it may derive from a builtin. This
    /// one asks about `type(v) is C`, where no subclass interferes: a direct
    /// instance of such a class is a plain object and has none of the kinds the
    /// partition names, which is `Some(false)` for every kind rather than a
    /// decline.
    ///
    /// A class laid out as a builtin answers by comparing the two, with `bool`
    /// under `int` for the reason the layout tag gives: a class laid out as an
    /// int may be `bool` itself.
    fn direct_instance_of_kind(&self, class: ClassIx, kind: Kind) -> Option<bool> {
        let value = self.literals.get(class.get())?.bind(self.py);
        let class = value.cast::<PyType>().ok()?;
        if !self.denotes_a_set(class)? {
            return None;
        }
        let (_, own) = layout_of(class);
        Some(match own {
            Some(own) => own == kind || matches!((own, kind), (Kind::Int, Kind::Bool)),
            // A direct instance of a class that lays down no builtin layout is
            // a plain object: no kind the partition names, and no subclass in
            // the question to widen it.
            None => false,
        })
    }

    /// Whether every value of `kind` is an instance of the pooled class.
    ///
    /// Asked of the kind's own builtin, so the answer is one `issubclass` over
    /// the order. A value of the kind built as that builtin has `type(v)` equal
    /// to it, which is what makes a `false` here a value rather than a guess
    /// about which classes exist.
    fn kind_derives_from(&self, kind: Kind, class: ClassIx) -> Option<bool> {
        let value = self.literals.get(class.get())?.bind(self.py);
        let class = value.cast::<PyType>().ok()?;
        if !self.denotes_a_set(class)? {
            return None;
        }
        let builtin = builtin_of(self.py, kind)?;
        builtin.is_subclass(class).ok()
    }

    /// Whether the class behind an `Instance` atom denotes a set, on the test
    /// [`PoolRelations::denotes_a_set`] states.
    fn atom_denotes_a_set(&self, atom: &Schema) -> Option<bool> {
        let Schema::Instance(index) = atom else {
            return None;
        };
        self.denotes_a_set(self.literals.get(index.get())?.bind(self.py))
    }

    fn leaf_subtype(&self, sub: &Schema, sup: &Schema) -> Option<bool> {
        match sub {
            // A literal denotes a singleton: `{v}` is a subtype of `sup` exactly
            // when `v` is a member of `sup`.
            Schema::Literal(index) => {
                let value = self.literals.get(index.get())?.bind(self.py);
                Some(self.is_member(sup, value))
            }
            // The `isinstance(., C)` values are a subset of the `isinstance(., D)`
            // values exactly when `C` is a subclass of `D`.
            Schema::Instance(index) => {
                let class = self.literals.get(index.get())?.bind(self.py);
                if let Schema::Instance(superindex) = sup {
                    let superclass = self.literals.get(superindex.get())?.bind(self.py);
                    let decided = class
                        .cast::<PyType>()
                        .ok()
                        .and_then(|class| class.is_subclass(superclass).ok())
                        .unwrap_or(false);
                    return Some(decided);
                }
                // An enumeration whose members compare by identity is the union
                // of them: the members are fixed when the class is defined, a
                // class with any cannot be subclassed, and no other value is an
                // instance. So the inclusion is asked of each member, which is
                // what makes `Color` and `Literal[Color.RED, Color.GREEN]` one
                // set rather than two the procedure cannot relate.
                let members = Self::enum_members(class)?;
                Some(members.iter().all(|member| self.is_member(sup, member)))
            }
            _ => None,
        }
    }

    fn literal_kind(&self, constant: ConstIx) -> Option<Kind> {
        // A literal pins `type(x)` exactly, so its kind is its constant's type.
        // Exact types only: a subclass of `int` is not `Kind::Int`'s extension,
        // and any other type is a kind the partition does not name. Both decline,
        // which leaves disjointness conservative rather than wrong.
        let value = self.literals.get(constant.get())?.bind(self.py);
        if value.is_none() {
            return Some(Kind::NoneType);
        }
        let ty = value.get_type();
        [
            (Kind::Bool, PyBool::type_object(self.py)),
            (Kind::Int, PyInt::type_object(self.py)),
            (Kind::Float, PyFloat::type_object(self.py)),
            (Kind::Str, PyString::type_object(self.py)),
            (Kind::Bytes, PyBytes::type_object(self.py)),
        ]
        .into_iter()
        .find_map(|(tag, exact)| ty.is(&exact).then_some(tag))
    }

    fn literals_disjoint(&self, left: ConstIx, right: ConstIx) -> Option<bool> {
        let left_value = self.literals.get(left.get())?.bind(self.py);
        let right_value = self.literals.get(right.get())?.bind(self.py);
        // A literal admits a value only when `type(x)` matches exactly, so two
        // constants of different types share no value however they compare --
        // `Literal[1]` and `Literal[True]` are disjoint although `1 == True`.
        if !left_value.get_type().is(right_value.get_type()) {
            return Some(true);
        }
        // Same type, so the singletons are disjoint exactly when the constants
        // differ. `==` is the value's own, and a type carrying user-defined
        // equality can admit one value for two distinct constants, so this is
        // asked only where the type's equality is one this oracle can trust: a
        // builtin scalar, whose equality is Python's, or a type that compares by
        // identity, where two distinct objects are two values by definition.
        if self.literal_kind(left).is_none() && !compares_by_identity(&left_value.get_type()) {
            return None;
        }
        left_value.eq(right_value).ok().map(|equal| !equal)
    }

    fn literal_sets_disjoint(&self, left: &[ConstIx], right: &[ConstIx]) -> Option<bool> {
        // Hash the smaller side and probe with the larger, which turns the
        // core's member-by-member walk into one pass. Keyed by `(type, value)`
        // because a literal pins `type(x)` exactly: `Literal[1]` and
        // `Literal[True]` are disjoint although `1 == True`, and a set keyed by
        // the value alone would call them equal.
        let (probe, held) = if left.len() <= right.len() {
            (right, left)
        } else {
            (left, right)
        };
        let seen = PySet::empty(self.py).ok()?;
        for index in held {
            let value = self.literals.get(index.get())?.bind(self.py);
            // Only where the type's equality is one this oracle can trust,
            // which is the condition `literals_disjoint` applies per pair: a
            // builtin scalar, whose equality is Python's, or a type comparing by
            // identity, where two distinct objects are two values.
            if self.literal_kind(*index).is_none() && !compares_by_identity(&value.get_type()) {
                return None;
            }
            seen.add((value.get_type(), value)).ok()?;
        }
        for index in probe {
            let value = self.literals.get(index.get())?.bind(self.py);
            if self.literal_kind(*index).is_none() && !compares_by_identity(&value.get_type()) {
                return None;
            }
            if seen.contains((value.get_type(), value)).ok()? {
                return Some(false);
            }
        }
        Some(true)
    }

    fn compare(&self, left: OperandIx, right: OperandIx) -> Option<core::cmp::Ordering> {
        // Order two refinement-bound values by Python's own comparison, so the
        // core can decide an unsatisfiable bound conjunction. An incomparable
        // pair (a TypeError) leaves the bound undecided.
        let left = self.literals.get(left.get())?.bind(self.py);
        let right = self.literals.get(right.get())?.bind(self.py);
        left.compare(right).ok()
    }

    fn no_int_between(
        &self,
        lo: OperandIx,
        lo_strict: bool,
        hi: OperandIx,
        hi_strict: bool,
    ) -> Option<bool> {
        // Decide whether the open/half-open interval bounded by the pooled values
        // admits no integer. The least admissible integer is `floor(lo) + 1` when
        // `lo` is excluded and `ceil(lo)` when it is included; the greatest is
        // `ceil(hi) - 1` when `hi` is excluded and `floor(hi)` when included. No
        // integer fits exactly when the least exceeds the greatest. The bounds are
        // compared as Python integers, so arbitrary-precision values stay exact.
        let (floor, ceil) = math_floor_ceil(self.py).ok()?;
        let floor = floor.bind(self.py);
        let ceil = ceil.bind(self.py);
        let lo = self.literals.get(lo.get())?.bind(self.py);
        let hi = self.literals.get(hi.get())?.bind(self.py);
        // A non-real bound (`math.floor` raises a `TypeError`) or a non-finite one
        // (an `OverflowError`) leaves the rule undecided rather than guessing.
        let one = 1i64;
        let least = if lo_strict {
            floor
                .call1((&lo,))
                .ok()?
                .call_method1("__add__", (one,))
                .ok()?
        } else {
            ceil.call1((&lo,)).ok()?
        };
        let greatest = if hi_strict {
            ceil.call1((&hi,))
                .ok()?
                .call_method1("__sub__", (one,))
                .ok()?
        } else {
            floor.call1((&hi,)).ok()?
        };
        // `least > greatest` means the interval skips every integer.
        Some(matches!(
            least.compare(&greatest).ok()?,
            core::cmp::Ordering::Greater
        ))
    }
}

/// The oracle's own corpus, under the embedded interpreter.
///
/// Every answer here is a Python fact, so a corpus that could run without an
/// interpreter would be testing something else. The `interpreter-tests` feature
/// links one, which is what `check/walk.rs` and `build.rs` do for the same
/// reason.
#[cfg(all(test, feature = "interpreter-tests"))]
mod interpreter;
