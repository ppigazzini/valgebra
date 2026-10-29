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
use std::sync::Arc;

use pyo3::PyTypeInfo;
use pyo3::exceptions::PyBaseException;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{
    PyBool, PyByteArray, PyBytes, PyComplex, PyDict, PyFloat, PyFrozenSet, PyInt, PyList, PyRange,
    PySet, PyString, PyTuple, PyType,
};
use rustc_hash::FxHashMap;
use valgebra_core::descr::classes::{Attributes, Class, Hook, Member, Namespace, Reach};
use valgebra_core::descr::lower::{Constants, Operand};
use valgebra_core::{ClassIx, ConstIx, Kind, LeafRelations, OperandIx, Schema};

use crate::check::{
    Ctx, Unwritten, ValidatorIndex, WalkMode, WalkState, build_index, forget_unindexed, member,
};
use crate::errors::is_fatal;
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
    /// What each class's direct instances can carry, read once per class for
    /// the query and keyed by the type object's address on the same argument
    /// as [`Self::classes`]. Both deciders ask it -- the descriptor through the
    /// snapshot, the rules field by field -- and one reading is one answer.
    attributes: RefCell<FxHashMap<usize, Arc<Attributes>>>,
    /// The first fatal signal a probe raised, for the query to re-raise.
    ///
    /// A probe runs user code -- a predicate, an `__eq__`, a rich comparison, a
    /// metaclass hook -- and one interrupted there has answered nothing. It
    /// reads as a decline to the core, like any probe that raised, and the
    /// signal itself is carried to [`Self::answer`], which hands it back in
    /// place of whatever the core decided without it.
    fatal: RefCell<Option<PyErr>>,
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
            attributes: RefCell::default(),
            fatal: RefCell::default(),
        }
    }

    /// The query's answer, or the fatal signal a probe raised while it was
    /// decided. The one way a relation leaves the bindings.
    pub(crate) fn answer<T>(self, decided: T) -> PyResult<T> {
        self.fatal.take().map_or(Ok(decided), Err)
    }

    /// A probe's result, or `None` where it raised: a check that raised has
    /// answered nothing, and reading it as a refutation is the unsound
    /// direction. A fatal signal is kept for [`Self::answer`].
    fn heard<T>(&self, result: PyResult<T>) -> Option<T> {
        result.map_err(|err| self.keep_if_fatal(err)).ok()
    }

    /// Keep `err` if it is the query's first fatal signal.
    fn keep_if_fatal(&self, err: PyErr) {
        if is_fatal(&err, self.py) {
            self.fatal.borrow_mut().get_or_insert(err);
        }
    }
}

impl Drop for PoolRelations<'_, '_> {
    /// The patterns this query's walks compiled outside an index are its own,
    /// and go with it.
    fn drop(&mut self) {
        forget_unindexed();
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
static ENUM_TYPES: PyOnceLock<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> = PyOnceLock::new();

/// `(enum.EnumMeta, enum.Flag, enum.Enum)`, imported once per interpreter.
///
/// All three are needed at the same place and none on any other path, so they
/// share one lock. `Flag` is here because it is the one enumeration whose
/// instances are *not* the members it lists: `P.A | P.B` is an instance of `P`
/// that `list(P)` never yields. `Enum` holds the `_missing_` that makes no
/// instance.
fn enum_types(py: Python<'_>) -> Option<&(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    ENUM_TYPES
        .get_or_try_init(py, || {
            let module = py.import("enum")?;
            let meta = module.getattr(intern!(py, "EnumMeta"))?.unbind();
            let flag = module.getattr(intern!(py, "Flag"))?.unbind();
            let base = module.getattr(intern!(py, "Enum"))?.unbind();
            Ok::<_, PyErr>((meta, flag, base))
        })
        .ok()
}

/// Whether every instance an enumeration's class makes is a member it lists:
/// the class lists its members through `EnumMeta`'s `__iter__`, makes an
/// instance only through `EnumMeta`'s `__call__`, which looks a member up, and
/// answers a value it holds no member for with `Enum`'s `_missing_`, which
/// makes none.
///
/// A `_missing_` of its own is the hook a class makes a new instance through
/// -- `E(5)`, a pseudo-member `list(E)` never yields -- and a metaclass's own
/// `__iter__` can leave a member out of the list, or its `__call__` build an
/// instance past the lookup.
fn lists_every_instance(
    class: &Bound<'_, PyType>,
    meta: &Bound<'_, PyAny>,
    base: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    let py = class.py();
    let metaclass = class.get_type();
    for name in [intern!(py, "__iter__"), intern!(py, "__call__")] {
        if !metaclass.getattr(name)?.is(&meta.getattr(name)?) {
            return Ok(false);
        }
    }
    let (missing, function) = (intern!(py, "_missing_"), intern!(py, "__func__"));
    Ok(class
        .getattr(missing)?
        .getattr(function)?
        .is(&base.getattr(missing)?.getattr(function)?))
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
///
/// One type defines an `__eq__` and compares by identity all the same: the
/// interpreter's `sentinel` (PEP 661, a builtin from 3.15), whose `__eq__`
/// answers `True` for the sentinel itself and declines every other value, so
/// two sentinels of one name are two values, as two `object()`s are.
///
/// The lookup runs a metaclass's own attribute hook, so it can raise, and the
/// caller hears what it raised: a check that raised has answered nothing, and
/// a signal among it is the query's to raise.
fn compares_by_identity(ty: &Bound<'_, PyType>) -> PyResult<bool> {
    let py = ty.py();
    let theirs = ty.getattr(intern!(py, "__eq__"))?;
    let inherited = PyAny::type_object(py).getattr(intern!(py, "__eq__"))?;
    Ok(theirs.is(&inherited) || is_the_sentinel_type(ty)?)
}

/// Whether `ty` is the interpreter's `sentinel`: a static type, which no class
/// statement makes, named `sentinel` in `builtins`, which its qualified name
/// spells bare. A literal of one is the model's singleton [FCB08, §4.2], as
/// one of an `object()` is. The flag is read first, since every type has one
/// and a class made where no `__name__` is bound has no `__module__`, so its
/// qualified name raises.
fn is_the_sentinel_type(ty: &Bound<'_, PyType>) -> PyResult<bool> {
    Ok(!is_heap_type(ty)? && ty.fully_qualified_name()? == "sentinel")
}

/// Whether `ty` was made at run time, as a class statement and a C type built
/// from a spec make one, rather than defined static in C, as the builtins are.
///
/// The flags are the ones Python set, read through `type`'s own descriptor: a
/// metaclass answering `__flags__` makes no class static.
fn is_heap_type(ty: &Bound<'_, PyType>) -> PyResult<bool> {
    let py = ty.py();
    let flags: std::os::raw::c_ulong = PyType::type_object(py)
        .getattr(intern!(py, "__dict__"))?
        .get_item(intern!(py, "__flags__"))?
        .call_method1(intern!(py, "__get__"), (ty,))?
        .extract()?;
    Ok(flags & pyo3::ffi::Py_TPFLAGS_HEAPTYPE != 0)
}

/// The value types the standard library defines whose order, `%` and length
/// are their equality's, by the module that defines each: two values equal by
/// one's `==` answer a bound, a step and a length alike.
const STDLIB_VALUE_TYPES: &[(&str, &[&str])] = &[
    ("decimal", &["Decimal"]),
    ("fractions", &["Fraction"]),
    ("datetime", &["date", "datetime", "time", "timedelta"]),
    ("uuid", &["UUID"]),
    (
        "ipaddress",
        &[
            "IPv4Address",
            "IPv6Address",
            "IPv4Network",
            "IPv6Network",
            "IPv4Interface",
            "IPv6Interface",
        ],
    ),
    (
        "pathlib",
        &[
            "PurePath",
            "PurePosixPath",
            "PureWindowsPath",
            "Path",
            "PosixPath",
            "WindowsPath",
        ],
    ),
];

/// `sys.modules`, the modules imported so far, read once.
static SYS_MODULES: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

/// Whether `ty` is exactly a value type whose order, `%` and length are its
/// equality's: `complex`, `bytearray`, `range`, a builtin container, or one of
/// [`STDLIB_VALUE_TYPES`].
///
/// Asked by identity with the class the module that defines it binds, never by
/// the `__module__` or the name a class states: a class body can state any,
/// and a module of the program's own can take a standard-library name. A
/// module not imported defines no class a value has, so it is looked up among
/// the imported ones and never imported here. A subclass is none of these.
fn is_a_value_type(ty: &Bound<'_, PyType>) -> PyResult<bool> {
    let py = ty.py();
    let builtins = [
        PyComplex::type_object(py),
        PyByteArray::type_object(py),
        PyRange::type_object(py),
        PyTuple::type_object(py),
        PyList::type_object(py),
        PyDict::type_object(py),
        PySet::type_object(py),
        PyFrozenSet::type_object(py),
    ];
    if builtins.iter().any(|builtin| ty.is(builtin)) {
        return Ok(true);
    }
    let modules = SYS_MODULES.import(py, "sys", "modules")?;
    for (module, names) in STDLIB_VALUE_TYPES {
        let Some(module) = modules.get_item(*module).ok() else {
            continue;
        };
        for name in *names {
            if module.getattr_opt(*name)?.is_some_and(|class| class.is(ty)) {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// Whether the values `==` joins to `value` read alike to a pattern, a length,
/// a bound and a step: [`LeafRelations::equal_values_read_alike`] of its class.
///
/// Those atoms read an `int`, a `float`, a `str` or `bytes` through the
/// builtin's storage, so a subclass of one reads alike where it keeps the
/// builtin's `__eq__` -- an `IntEnum` does -- and not where its `__eq__` is its
/// own. Any other class reads alike where it compares by identity, the equal
/// values being one, or where it is exactly one of the value types whose
/// order, `%` and length are their equality's ([`is_a_value_type`]):
/// `Decimal`, `Fraction`, the date types, `complex`. A user class with an
/// `__eq__` of its own is read as unknown, whatever module it names.
///
/// Asked of the class Python reports, through `issubclass` and attribute
/// lookups a metaclass can answer, so it can raise, and the caller hears what
/// it raised.
fn equal_values_read_alike(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    if kind_of(value).is_some() {
        return Ok(true);
    }
    let py = value.py();
    let ty = value.get_type();
    if let Some(builtin) = builtin_base(&ty)? {
        return keeps(&ty, &builtin, &[intern!(py, "__eq__")]);
    }
    if compares_by_identity(&ty)? {
        return Ok(true);
    }
    is_a_value_type(&ty)
}

/// Whether a bound's operand orders as the walk reads it: an exact builtin, a
/// builtin subclass keeping every comparison and rounding method of its
/// builtin, or exactly one of the value types whose order is their equality's
/// ([`is_a_value_type`]).
///
/// The walk asks `x >= b` of a value `x`, which reads a builtin subclass's
/// operand through the builtin's storage unless the subclass overrides the
/// reflected method, while the rules order two operands through `==` first
/// and round one through `__floor__` and `__ceil__`. A subclass with its own
/// `__eq__` is one bound to the rules and another to the walk -- `Mod7(1) == 8`,
/// and `1 >= Mod7(1)` reads `1` -- and one with its own order reads the other
/// way round. Any other class orders as unknown: Python implies no
/// relationship among a class's comparison operators (the data model, "Basic
/// customization"), and the walk reads `>=` alone.
fn orders_as_the_walk_reads(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    if kind_of(value).is_some() {
        return Ok(true);
    }
    let py = value.py();
    let ty = value.get_type();
    let Some(builtin) = builtin_base(&ty)? else {
        return is_a_value_type(&ty);
    };
    keeps(
        &ty,
        &builtin,
        &[
            intern!(py, "__eq__"),
            intern!(py, "__lt__"),
            intern!(py, "__le__"),
            intern!(py, "__gt__"),
            intern!(py, "__ge__"),
            intern!(py, "__floor__"),
            intern!(py, "__ceil__"),
        ],
    )
}

/// The builtin whose storage the atoms read a value of `ty` through -- `int`,
/// `float`, `str` or `bytes` -- where `ty` derives from one.
fn builtin_base<'py>(ty: &Bound<'py, PyType>) -> PyResult<Option<Bound<'py, PyType>>> {
    let py = ty.py();
    for builtin in [
        PyInt::type_object(py),
        PyFloat::type_object(py),
        PyString::type_object(py),
        PyBytes::type_object(py),
    ] {
        if ty.is_subclass(&builtin)? {
            return Ok(Some(builtin));
        }
    }
    Ok(None)
}

/// Whether `ty` resolves each of `names` its builtin has to what the builtin
/// resolves it to. A subclass inherits every name the builtin has, so one the
/// builtin lacks is no method of its to keep.
fn keeps(
    ty: &Bound<'_, PyType>,
    builtin: &Bound<'_, PyType>,
    names: &[&Bound<'_, PyString>],
) -> PyResult<bool> {
    for name in names {
        if let Some(theirs) = builtin.getattr_opt(*name)?
            && !ty.getattr(*name)?.is(&theirs)
        {
            return Ok(false);
        }
    }
    Ok(true)
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
    ///   `[P.A, P.B]`. Reading `P` as that union would make
    ///   `P <= Literal[P.A, P.B]` true with `P.A | P.B` standing against it,
    ///   and would decide `P & ~Literal[P.A, P.B]` empty although it admits
    ///   that value.
    /// * it **has at least one member**. An enumeration with none can still be
    ///   subclassed -- that is how an enum base class is written -- so its
    ///   instances are its subclasses' members, and reading it as the empty
    ///   union would make it a subtype of `nothing`.
    /// * two members are two values, which is the identity check. An `IntEnum`
    ///   fails it because its members equal the integers behind them.
    /// * the class makes no instance but by looking a member up
    ///   ([`lists_every_instance`]). A `_missing_` of its own makes one --
    ///   `E(5)` -- and a metaclass's own `__iter__` or `__call__` can hide a
    ///   member from the list or build an instance past the lookup.
    ///
    /// A class failing any of them stays the `isinstance` atom it was, which is
    /// sound for every enumeration and merely less complete.
    fn enum_members<'py>(&self, class: &Bound<'py, PyAny>) -> Option<Vec<Bound<'py, PyAny>>> {
        let py = class.py();
        let class = class.cast::<PyType>().ok()?;
        let (meta, flag, base) = enum_types(py)?;
        if !self.heard(class.is_instance(meta.bind(py)))? {
            return None;
        }
        if self.heard(class.is_subclass(flag.bind(py)))? {
            return None;
        }
        if !self.heard(compares_by_identity(class))? {
            return None;
        }
        if !self.heard(lists_every_instance(class, meta.bind(py), base.bind(py)))? {
            return None;
        }
        let members: Vec<Bound<'py, PyAny>> = self.heard(
            class
                .try_iter()
                .and_then(|members| members.take(MAX_ENUM_MEMBERS + 1).collect()),
        )?;
        (!members.is_empty() && members.len() <= MAX_ENUM_MEMBERS).then_some(members)
    }

    fn is_member(&self, schema: &Schema, value: &Bound<'_, PyAny>) -> bool {
        // These leaf-subtype probes run on transient schemas during compilation,
        // not on a finished validator, so they carry no precomputed index; the
        // walk falls back to its general path for any record or union here. A
        // fatal signal the walk records is kept for the query, and a query
        // already interrupted runs no more of them.
        self.is_member_under(schema, value, &ValidatorIndex::default())
    }

    /// [`Self::is_member`] under an `index` built for `schema`, for a probe
    /// that walks several values through it.
    fn is_member_under(
        &self,
        schema: &Schema,
        value: &Bound<'_, PyAny>,
        index: &ValidatorIndex,
    ) -> bool {
        if self.fatal.borrow().is_some() {
            return false;
        }
        let state = WalkState::new();
        let ctx = Ctx {
            pool: self.literals,
            defs: self.definitions,
            index,
            state: &state,
            mode: WalkMode::Fast,
        };
        let admitted = member(
            schema,
            &Value::Py(value),
            &mut Unwritten::default().frame(ctx),
        );
        if let Some(err) = state.take_fatal() {
            self.keep_if_fatal(err);
        }
        admitted
    }
}

/// The kind a constant belongs to, or `None` where the partition does not name
/// one.
///
/// A literal pins `type(x)` exactly, so its kind is its constant's type. Exact
/// types only: a subclass of `int` is not `Kind::Int`'s extension, and any
/// other type is a kind the partition does not name. Both decline, which leaves
/// every reading of this conservative rather than wrong.
///
/// Two callers, and they ask the same question for different reasons. The
/// oracle asks it to tell two literals apart; the frontend asks it to know
/// whether a constraint can be put to a literal's values at all, since
/// `Annotated[Literal[1], MinLen(1)]` is `Annotated[int, MinLen(1)]` one value
/// narrower and has to be refused for the same reason. One reading, so the two
/// cannot come to different answers about what a constant is.
pub(crate) fn kind_of(value: &Bound<'_, PyAny>) -> Option<Kind> {
    if value.is_none() {
        return Some(Kind::NoneType);
    }
    let py = value.py();
    let ty = value.get_type();
    [
        (Kind::Bool, PyBool::type_object(py)),
        (Kind::Int, PyInt::type_object(py)),
        (Kind::Float, PyFloat::type_object(py)),
        (Kind::Str, PyString::type_object(py)),
        (Kind::Bytes, PyBytes::type_object(py)),
    ]
    .into_iter()
    .find_map(|(tag, exact)| ty.is(&exact).then_some(tag))
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

/// The class that lays down `ty`'s instance layout, and the kind that layout
/// confines an instance to.
///
/// Python refuses a class deriving from two others unless the layout one lays
/// down extends the other's: `class C(int, str)` is "multiple bases have
/// instance lay-out conflict", and so is `class C(A, B)` for two classes whose
/// `__slots__` each add a slot over `object`. The class laying down a layout is
/// the first on the `__mro__` that does -- one of the nine builtins below, or a
/// class that lays down one of its own ([`lays_down_its_own`]) -- and a class
/// built on none of them carries no layout, which conflicts with nothing and
/// confines nothing: `class Both(Plain, MyStr)` builds, so a plain class and a
/// `str` subclass do share instances, and those instances are strings.
///
/// The kind is the builtin's alone. A class whose `__slots__` add to a `str`
/// lays down a layout of its own and its instances are still strings; one whose
/// `__slots__` add to `object` lays down a layout and its instances are of no
/// listed kind, which is the reading [`Class::lays_down_a_layout`] carries.
///
/// Reading a class's own `__slots__` runs its code, and an error it raises is
/// the caller's to hear, as every other probe's is.
fn layout_of<'py>(ty: &Bound<'py, PyType>) -> PyResult<(Option<Bound<'py, PyType>>, Option<Kind>)> {
    let py = ty.py();
    let builtins = [
        (PyInt::type_object(py), Kind::Int),
        (PyString::type_object(py), Kind::Str),
        (PyBytes::type_object(py), Kind::Bytes),
        (PyFloat::type_object(py), Kind::Float),
        (PyTuple::type_object(py), Kind::Tuple),
        (PyFrozenSet::type_object(py), Kind::FrozenSet),
        (PyList::type_object(py), Kind::List),
        (PySet::type_object(py), Kind::Set),
        (PyDict::type_object(py), Kind::Dict),
    ];
    // `bool` derives from `int`, so its kind is `Int` and not `Bool`, and a
    // `bool` and an `int` subclass land in one part rather than two -- which is
    // right: `bool` is final, so an `int` subclass is never a boolean, and the
    // pair is disjoint for a reason the layout does not carry.
    //
    // Walked along `__base__`, the chain Python lays an instance out on, and
    // never along `__mro__` or `issubclass`, which read the order a
    // metaclass's `mro()` writes: one that leaves `int` out of it makes no
    // instance of the class less of an `int`.
    let (mut solid, mut kind) = (None, None);
    let mut at = Some(ty.clone());
    while let Some(class) = at {
        if let Some((_, of)) = builtins.iter().find(|(builtin, _)| class.is(builtin)) {
            kind = Some(*of);
            solid.get_or_insert(class);
            break;
        }
        if solid.is_none() && lays_down_its_own(&class)? {
            solid = Some(class.clone());
        }
        at = base_of(&class)?;
    }
    Ok((solid, kind))
}

/// `type.__base__`, the descriptor read once.
static TYPE_BASE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

/// The class Python lays `ty`'s instances out on, or `None` for `object`: its
/// `__base__`, read through `type`'s own descriptor, so a metaclass defining
/// an attribute of that name does not answer for it.
fn base_of<'py>(ty: &Bound<'py, PyType>) -> PyResult<Option<Bound<'py, PyType>>> {
    let py = ty.py();
    let descriptor = TYPE_BASE.get_or_try_init(py, || {
        Ok::<_, PyErr>(
            PyType::type_object(py)
                .getattr(intern!(py, "__dict__"))?
                .get_item(intern!(py, "__base__"))?
                .unbind(),
        )
    })?;
    let base = descriptor
        .bind(py)
        .call_method1(intern!(py, "__get__"), (ty,))?;
    Ok(base.cast_into::<PyType>().ok())
}

/// `types.MemberDescriptorType`, read once.
static MEMBER_DESCRIPTOR: PyOnceLock<Py<PyType>> = PyOnceLock::new();

/// Whether `value`, found in `ty`'s own namespace, is a slot Python laid down
/// for `ty`: a member descriptor whose `__objclass__` is `ty`. Python makes one
/// for each name a class's `__slots__` lists when it makes the class, and no
/// assignment makes another -- naming `__slots__` afterwards lays nothing down,
/// and one class's descriptor copied into another names its own class.
fn is_own_slot(value: &Bound<'_, PyAny>, ty: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = value.py();
    let member = MEMBER_DESCRIPTOR.import(py, "types", "MemberDescriptorType")?;
    Ok(value.get_type().is(member) && value.getattr(intern!(py, "__objclass__"))?.is(ty))
}

/// Whether a class lays down an instance layout of its own over the one its
/// base carries.
///
/// A class defined in Python does where its own `__slots__` add a slot
/// ([`adds_slots`]). A type defined in C as a static type does where its
/// instance is larger than its base's, by `__basicsize__` or by
/// `__itemsize__`: that is the test `CPython` puts to a static type in
/// `solid_base`, in `Objects/typeobject.c`, on every release this crate builds
/// for, and the layout Python refuses to combine with one that neither extends
/// nor is extended by it. `complex` lays down one, so no class derives
/// from it and from `int`; `Exception` lays down none, its instance being the
/// one `BaseException` lays down, so `class Both(ValueError, TypeError)` builds
/// and the two share instances.
///
/// A type defined in C as a heap type -- `collections.deque`, and
/// `decimal.Decimal` from 3.13 -- is read as a class defined in Python is, by
/// its own `__slots__`, and lays down nothing here: the size of a heap type
/// counts the `__dict__` and `__weakref__` pointers before 3.12, which lay down
/// nothing.
///
/// The flags, the base and the two sizes are the ones Python laid down, read
/// through `type`'s own descriptors as `solid_base` reads the type object: a
/// metaclass answering any of them lays no layout down.
#[cfg(not(PyPy))]
fn lays_down_its_own<'py>(ty: &Bound<'py, PyType>) -> PyResult<bool> {
    let py = ty.py();
    let readers = PyType::type_object(py).getattr(intern!(py, "__dict__"))?;
    let read = |name: &Bound<'py, PyString>, of: &Bound<'py, PyType>| {
        readers
            .get_item(name)?
            .call_method1(intern!(py, "__get__"), (of,))
    };
    if is_heap_type(ty)? {
        return adds_slots(ty);
    }
    let Ok(base) = read(intern!(py, "__base__"), ty)?.cast_into::<PyType>() else {
        return Ok(false);
    };
    let shape = |of: &Bound<'py, PyType>| -> PyResult<(isize, isize)> {
        Ok((
            read(intern!(py, "__basicsize__"), of)?.extract()?,
            read(intern!(py, "__itemsize__"), of)?.extract()?,
        ))
    };
    Ok(shape(ty)? != shape(&base)?)
}

/// Whether a class lays down an instance layout of its own: on PyPy, where its
/// own `__slots__` add a slot. PyPy refuses the classes `CPython` refuses but
/// names no size, so a type defined in C is read as laying down nothing, and
/// what `CPython` decides of it PyPy declines.
#[cfg(PyPy)]
fn lays_down_its_own(ty: &Bound<'_, PyType>) -> PyResult<bool> {
    adds_slots(ty)
}

/// Whether a class's own `__slots__` add an instance slot Python laid down.
///
/// Read from the class's own namespace rather than inherited, because a slot is
/// laid down once and a subclass without `__slots__` of its own adds none, and
/// read off the member descriptor each slot is ([`is_own_slot`]) rather than
/// off the names: `__slots__` assigned after the class is made lays nothing
/// down. `__dict__` and `__weakref__` name no slot: they ask for the two
/// pointers a plain instance has anyway, and Python lays down no member for
/// them -- a class listing only those meets any other. A name is one slot, and
/// a sequence of names is each of them; an empty sequence is no slot at all.
///
/// A class without `__slots__` of its own adds none; any other error reading
/// them -- the namespace through a metaclass's hook, the names through an
/// iterable of the class's -- is the caller's.
fn adds_slots(ty: &Bound<'_, PyType>) -> PyResult<bool> {
    let py = ty.py();
    let namespace = ty.getattr(intern!(py, "__dict__"))?;
    // Asked before it is read: most classes declare no slots, and a `KeyError`
    // built for each of them and read for its type costs a relation between
    // two dataclasses more than the rest of its class reading.
    if !namespace.contains(intern!(py, "__slots__"))? {
        return Ok(false);
    }
    let slots = namespace.get_item(intern!(py, "__slots__"))?;
    for name in slot_names(&slots)? {
        let value = namespace.call_method1(intern!(py, "get"), (slot_key(ty, &name)?,))?;
        if is_own_slot(&value, ty)? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// The key a slot named `name` is stored under in `ty`'s namespace: the name,
/// or for a private one -- two leading underscores and not two trailing --
/// the name mangled with the class's, as Python stores it.
fn slot_key(ty: &Bound<'_, PyType>, name: &str) -> PyResult<String> {
    if !name.starts_with("__") || name.ends_with("__") {
        return Ok(name.to_owned());
    }
    let owner: String = ty.getattr(intern!(ty.py(), "__name__"))?.extract()?;
    let owner = owner.trim_start_matches('_');
    Ok(if owner.is_empty() {
        name.to_owned()
    } else {
        format!("_{owner}{name}")
    })
}

/// The names a class's own `__slots__` -- one name, or a sequence of them --
/// lists, read once. Iterating them runs the class's code, and what it raises
/// is the caller's; an entry that is not a string names no slot.
fn slot_names(slots: &Bound<'_, PyAny>) -> PyResult<Vec<Box<str>>> {
    let name = |slot: &Bound<'_, PyAny>| {
        slot.cast::<PyString>()
            .ok()
            .and_then(|slot| slot.to_str().ok().map(Box::from))
    };
    if slots.is_instance_of::<PyString>() {
        return Ok(name(slots).into_iter().collect());
    }
    let mut names = Vec::new();
    for slot in slots.try_iter()? {
        names.extend(name(&slot?));
    }
    Ok(names)
}

/// The classes whose own `__getattribute__` is the generic lookup rather than a
/// hook: `object`, which defines it, and the builtins a value's kind is read
/// from, with `BaseException`.
///
/// `CPython` 3.10 to 3.13 put a slot wrapper for the generic lookup in each of
/// those namespaces, and 3.14 puts it in `object`'s alone; read as a hook, it
/// would make every `int` or exception subclass's attributes code's answer on
/// the older releases and not on the newer. The classes whose lookup is their
/// own -- `type`, `super`, a module, `functools.partial` -- are not on the list,
/// and read as hooks on every release: the conservative direction.
const GENERIC_LOOKUP: [fn(Python<'_>) -> Bound<'_, PyType>; 11] = [
    PyAny::type_object,
    PyInt::type_object,
    PyFloat::type_object,
    PyString::type_object,
    PyBytes::type_object,
    PyTuple::type_object,
    PyList::type_object,
    PyDict::type_object,
    PySet::type_object,
    PyFrozenSet::type_object,
    PyBaseException::type_object,
];

/// Where `class` stands in [`GENERIC_LOOKUP`], if it is one of those classes.
fn generic_lookup_position(class: &Bound<'_, PyAny>) -> Option<usize> {
    GENERIC_LOOKUP
        .iter()
        .position(|generic| class.is(generic(class.py())))
}

/// The namespaces of [`GENERIC_LOOKUP`], each read once per process.
///
/// Every one of them is a static type, whose namespace no assignment reaches --
/// `object.x = 1` raises on every interpreter this builds for -- so the reading
/// taken on first sight is the one any later query would take. `object`'s alone
/// holds most of the names a dataclass's `__mro__` defines, which a relation
/// between two classes would otherwise read again for each of them.
static GENERIC_NAMESPACES: [PyOnceLock<NamespaceReading>; GENERIC_LOOKUP.len()] =
    [const { PyOnceLock::new() }; GENERIC_LOOKUP.len()];

/// Whether `ty` has the attribute `name`, as Python's `hasattr` answers.
///
/// From 3.13 that is `PyO3`'s `hasattr`, which calls the interpreter's own.
#[cfg(Py_3_13)]
fn type_has(ty: &Bound<'_, PyType>, name: &Bound<'_, PyString>) -> PyResult<bool> {
    ty.hasattr(name)
}

/// Whether `ty` has the attribute `name`, as Python's `hasattr` answers.
///
/// Before 3.13 `PyO3`'s `hasattr` is a `getattr` whose `AttributeError` is
/// cleared, and on a type object the error's message is formatted before it
/// is. The builtin `hasattr` asks a type through the lookup 3.12 gives it,
/// which answers a miss without building the error; 3.10 and 3.11 build it
/// either way. On 3.12 that error is most of what reading a dataclass's
/// namespace costs, since every value type that is no data descriptor misses
/// twice.
#[cfg(not(Py_3_13))]
fn type_has(ty: &Bound<'_, PyType>, name: &Bound<'_, PyString>) -> PyResult<bool> {
    static HASATTR: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    let py = ty.py();
    HASATTR
        .get_or_try_init(py, || {
            Ok::<_, PyErr>(py.import("builtins")?.getattr("hasattr")?.unbind())
        })?
        .bind(py)
        .call1((ty, name))?
        .is_truthy()
}

/// Whether each type a class's namespace values have is a data descriptor's,
/// for the one reading of the class's `__mro__`: its namespaces hold a few
/// dozen values of a dozen types, and each type is two `hasattr` questions.
/// Each type is held, so no other takes its address while the reading lasts.
type Descriptors<'py> = Vec<(Bound<'py, PyType>, bool)>;

/// What one class's own namespace contributes to the [`Attributes`] of each
/// class whose `__mro__` it stands on.
#[derive(Clone)]
struct NamespaceReading {
    /// Each name it holds, with what it holds there.
    names: Arc<Namespace>,
    /// Whether it defines `__dict__`, the descriptor serving an instance's
    /// dictionary.
    carries_dict: bool,
    /// Whether it defines `__getattr__`.
    getattr: bool,
    /// Whether it defines a `__getattribute__` other than the generic lookup.
    getattribute: bool,
}

/// Read `base`'s own namespace. A `__getattribute__` it defines is a hook
/// unless `generic` says it is the generic lookup.
///
/// A name is a [`Member::Slot`] where the namespace's own `__slots__` names
/// it and holds the member descriptor Python laid down for it
/// ([`is_own_slot`]), a [`Member::Descriptor`] where its value's type defines
/// `__set__` or `__delete__`, and a [`Member::Plain`] otherwise. A slot whose
/// descriptor an assignment replaced is read as what replaced it: every
/// instance then reads the class's value. A mangled slot name, which
/// `__slots__` spells unmangled, reads as a descriptor: the conservative
/// direction. `__slots__` is found among the items rather than asked for,
/// because asking a namespace without it raises a `KeyError` for every
/// class that has none.
fn read_namespace<'py>(
    base: &Bound<'py, PyAny>,
    generic: bool,
    descriptors: &mut Descriptors<'py>,
) -> PyResult<NamespaceReading> {
    let py = base.py();
    let mut entries = Vec::new();
    let mut slots = None;
    for item in base
        .getattr(intern!(py, "__dict__"))?
        .call_method0(intern!(py, "items"))?
        .try_iter()?
    {
        let (name, value) = item?.extract::<(Bound<'py, PyAny>, Bound<'py, PyAny>)>()?;
        let Ok(name) = name.cast_into::<PyString>() else {
            continue;
        };
        let name: Box<str> = name.to_str()?.into();
        if &*name == "__slots__" {
            slots = Some(value.clone());
        }
        entries.push((name, value));
    }
    // The slots are read before the names they are checked against, so the
    // loop over the namespace runs none of the class's code but its
    // descriptors' types.
    let slots = match &slots {
        Some(slots) => slot_names(slots)?,
        None => Vec::new(),
    };
    let (mut carries_dict, mut getattr, mut getattribute) = (false, false, false);
    let mut names = Vec::new();
    for (name, value) in entries {
        match &*name {
            "__dict__" => carries_dict = true,
            "__getattr__" => getattr = true,
            "__getattribute__" => getattribute = !generic,
            _ => {}
        }
        let member = if slots.contains(&name) && is_own_slot(&value, base)? {
            Member::Slot
        } else if is_data_descriptor(&value, descriptors)? {
            Member::Descriptor
        } else {
            Member::Plain
        };
        names.push((name, member));
    }
    Ok(NamespaceReading {
        names: Arc::new(Namespace::new(names)),
        carries_dict,
        getattr,
        getattribute,
    })
}

/// Whether `value` is a data descriptor: its type defines `__set__` or
/// `__delete__`, so it answers a read whatever an instance's dictionary holds.
/// Asked of each type once per class read, through `descriptors`.
fn is_data_descriptor<'py>(
    value: &Bound<'py, PyAny>,
    descriptors: &mut Descriptors<'py>,
) -> PyResult<bool> {
    let ty = value.get_type();
    if let Some((_, answer)) = descriptors.iter().find(|(seen, _)| seen.is(&ty)) {
        return Ok(*answer);
    }
    let py = value.py();
    let answer =
        type_has(&ty, intern!(py, "__set__"))? || type_has(&ty, intern!(py, "__delete__"))?;
    descriptors.push((ty, answer));
    Ok(answer)
}

impl<'py> PoolRelations<'py, '_> {
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
    fn snapshot(&self, ty: &Bound<'py, PyType>) -> Option<Class> {
        if !self.denotes_a_set(ty)? {
            return None;
        }
        let mut bases = Vec::new();
        for base in self.heard(
            ty.getattr(intern!(ty.py(), "__mro__"))
                .and_then(|mro| mro.try_iter()),
        )? {
            let base = self.heard(base)?;
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
        let (solid, kind) = self.heard(layout_of(ty))?;
        let layout = solid.map(|base| self.class_id(&base));
        let class = Class::new(self.class_id(ty), layout, &bases).carrying(self.attributes_of(ty)?);
        Some(match kind {
            Some(kind) => class.of_kind(kind),
            None => class,
        })
    }

    /// What a direct instance of `ty` can carry, read once for the query.
    fn attributes_of(&self, ty: &Bound<'py, PyType>) -> Option<Arc<Attributes>> {
        let key = ty.as_ptr() as usize;
        if let Some(read) = self.attributes.borrow().get(&key) {
            return Some(Arc::clone(read));
        }
        let read = Arc::new(self.read_attributes(ty)?);
        self.attributes.borrow_mut().insert(key, Arc::clone(&read));
        Some(read)
    }

    /// Read the three facts [`Attributes`] holds off the namespaces of
    /// `ty.__mro__`, in order, as an attribute lookup on an instance reads them.
    ///
    /// The instances carry a dictionary where some class defines `__dict__`,
    /// which is where Python puts the descriptor that serves it. A
    /// `__getattribute__` answers every lookup, and a `__getattr__` every lookup
    /// that finds nothing -- except the generic lookup itself, which
    /// [`GENERIC_LOOKUP`] lists the classes of. A name keeps what the
    /// first namespace to define it holds, which is what a lookup finds.
    ///
    /// Any error reading a namespace declines the whole reading: a class whose
    /// namespace cannot be read is not one whose instances this can describe.
    fn read_attributes(&self, ty: &Bound<'py, PyType>) -> Option<Attributes> {
        let mut namespaces = Vec::new();
        let mut descriptors = Vec::new();
        for base in self.heard(
            ty.getattr(intern!(self.py, "__mro__"))
                .and_then(|mro| mro.try_iter()),
        )? {
            let base = self.heard(base)?;
            namespaces.push(self.heard(self.namespace_of(&base, &mut descriptors))?);
        }
        let any = |flag: fn(&NamespaceReading) -> bool| namespaces.iter().any(flag);
        let hook = if any(|namespace| namespace.getattribute) {
            Hook::Getattribute
        } else if any(|namespace| namespace.getattr) {
            Hook::Getattr
        } else {
            Hook::Neither
        };
        let mut attributes = Attributes::new(any(|namespace| namespace.carries_dict), hook);
        for namespace in namespaces {
            attributes.inherit(namespace.names);
        }
        Some(attributes)
    }

    /// The reading of `base`'s own namespace: the one [`GENERIC_NAMESPACES`]
    /// holds where `base` is one of [`GENERIC_LOOKUP`], and one taken
    /// here otherwise.
    fn namespace_of(
        &self,
        base: &Bound<'py, PyAny>,
        descriptors: &mut Descriptors<'py>,
    ) -> PyResult<NamespaceReading> {
        match generic_lookup_position(base).and_then(|at| GENERIC_NAMESPACES.get(at)) {
            Some(shared) => shared
                .get_or_try_init(self.py, || read_namespace(base, true, descriptors))
                .cloned(),
            None => read_namespace(base, false, descriptors),
        }
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
        // A class `type` made reads its hooks off `type` itself, whose
        // namespace no assignment reaches: they are untouched without asking.
        // It is nearly every class, and asking is four attribute lookups.
        if metaclass.is(&plain) {
            return Some(true);
        }
        let untouched = |hook: &Bound<'_, PyString>| -> Option<bool> {
            Some(
                self.heard(metaclass.getattr(hook))?
                    .is(&plain.getattr(hook).ok()?),
            )
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
    /// A class laying down a layout that is no builtin's -- its own `__slots__`,
    /// over `object` -- holds values of no listed kind, and a subclass cannot
    /// escape that either: Python refuses `class Both(Slotted, str)` as it
    /// refuses `class C(int, str)`. That is `Some(false)` for every kind.
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
        let (solid, own) = self.heard(layout_of(class))?;
        // `bool` derives from `int`, so a class laid out as an int may be
        // `bool` itself and hold booleans. Sound and coarse: the pair is never
        // refuted here.
        match own {
            Some(own) => Some(own == kind || matches!((own, kind), (Kind::Int, Kind::Bool))),
            None => solid.map(|_| false),
        }
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
    /// under `int` for the reason the kind gives: a class laid out as an
    /// int may be `bool` itself.
    fn direct_instance_of_kind(&self, class: ClassIx, kind: Kind) -> Option<bool> {
        let value = self.literals.get(class.get())?.bind(self.py);
        let class = value.cast::<PyType>().ok()?;
        if !self.denotes_a_set(class)? {
            return None;
        }
        let (_, own) = self.heard(layout_of(class))?;
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
        self.heard(builtin.is_subclass(class))
    }

    /// What attribute `name` of a direct instance of the pooled class can be,
    /// from the same reading the class's snapshot carries.
    ///
    /// The default reads it through [`Constants::class`], which builds the
    /// whole order snapshot to read one name; this reads the cached attribute
    /// facts alone, behind the same purity test the snapshot applies.
    fn attribute_reach(&self, class: ClassIx, name: &str) -> Option<Reach> {
        let value = self.literals.get(class.get())?.bind(self.py);
        let class = value.cast::<PyType>().ok()?;
        if !self.denotes_a_set(class)? {
            return None;
        }
        Some(self.attributes_of(class)?.reach(name))
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
            // The constant's membership, which is the literal's where the
            // constant is its one value or nothing in `sup` tells equal values
            // apart; the core asks which, and keeps this proof only there.
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
                    // Both sides must denote a set before the class order can
                    // answer for them, the way every other class question here
                    // asks first. A metaclass that computes `__subclasscheck__`
                    // makes `issubclass` say something `isinstance` does not --
                    // CPython's instance check reads the real order and never
                    // calls the hook -- and one that computes `__instancecheck__`
                    // gives its class a membership no snapshot of the order
                    // predicts. Neither is a set this algebra holds.
                    if !self.denotes_a_set(class)? || !self.denotes_a_set(superclass)? {
                        return None;
                    }
                    // A check that raises has answered nothing. Reading the
                    // error as `false` turns a question the bindings could not
                    // put into a refutation the rules then believe.
                    return class
                        .cast::<PyType>()
                        .ok()
                        .and_then(|class| self.heard(class.is_subclass(superclass)));
                }
                // An enumeration whose members compare by identity is the union
                // of them: the members are fixed when the class is defined, a
                // class with any cannot be subclassed, and no other value is an
                // instance. So the inclusion is asked of each member, which is
                // what makes `Color` and `Literal[Color.RED, Color.GREEN]` one
                // set rather than two the procedure cannot relate.
                let members = self.enum_members(class)?;
                Some(members.iter().all(|member| self.is_member(sup, member)))
            }
            // `None`'s kind is the literal of its one value: `NoneType` admits
            // no subclass, so the walk of `None` is the kind's. `bool`'s two
            // values are each a pooled literal, which the parts reading asks
            // one at a time, and no kind with more is a set of values to walk.
            Schema::NoneType => Some(self.is_member(sup, &self.py.None().into_bound(self.py))),
            _ => None,
        }
    }

    /// Each text as a fresh `str`, walked through one index built for
    /// `schema`, so a pattern compiles once for the question rather than once
    /// a text. A probe a fatal signal stopped has answered nothing.
    fn strings_admitted(&self, schema: &Schema, texts: &[&str]) -> Option<Vec<bool>> {
        let index = build_index(self.py, schema, self.definitions, self.literals);
        let admitted = texts
            .iter()
            .map(|text| self.is_member_under(schema, PyString::new(self.py, text).as_any(), &index))
            .collect();
        self.fatal.borrow().is_none().then_some(admitted)
    }

    fn literal_kind(&self, constant: ConstIx) -> Option<Kind> {
        kind_of(self.literals.get(constant.get())?.bind(self.py))
    }

    /// Whether the two constants' literals share no value: constants of two
    /// exact types, or of one type and unequal under an equality this oracle
    /// trusts. A literal is its constant's class under `==` within the type,
    /// the model's singleton [FCB08, §4.2] where `==` joins nothing else to
    /// the constant, so a constant asked against itself is how the core asks
    /// whether its literal holds a value, and `nan`, equal to nothing, holds
    /// none.
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
        if self.literal_kind(left).is_none()
            && !self.heard(compares_by_identity(&left_value.get_type()))?
        {
            return None;
        }
        self.heard(left_value.eq(right_value)).map(|equal| !equal)
    }

    /// [`literals_disjoint`](LeafRelations::literals_disjoint) over two sets of
    /// constants in one pass: two literals share a value where their constants
    /// are one exact type and equal [FCB08, §4.2], under the equality the pair
    /// form trusts. A constant unequal to itself denotes no value, as the pair
    /// form reads it, so it shares none even with itself.
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
            if self.literal_kind(*index).is_none()
                && !self.heard(compares_by_identity(&value.get_type()))?
            {
                return None;
            }
            self.heard(seen.add((value.get_type(), value)))?;
        }
        for index in probe {
            let value = self.literals.get(index.get())?.bind(self.py);
            if self.literal_kind(*index).is_none()
                && !self.heard(compares_by_identity(&value.get_type()))?
            {
                return None;
            }
            // A set finds a member by identity before it asks `==`, so the
            // one `nan` held on both sides is found; it is a value of neither
            // literal, and only a constant equal to itself is one shared.
            if self.heard(seen.contains((value.get_type(), value)))?
                && self.heard(value.eq(value))?
            {
                return Some(false);
            }
        }
        Some(true)
    }

    /// Whether the constant is the only value of its exact type that `==` it,
    /// which makes its literal the model's singleton [FCB08, §4.2].
    ///
    /// An exact `int`, `bool`, `str` or `bytes` and `None` are: their `==` is
    /// equality of value, and two equal values are one. A float is unless it
    /// is a zero, since `-0.0 == 0.0` is the one pair of distinct floats `==`
    /// joins and `nan` joins nothing; a complex number likewise, unless a part
    /// is. An instance of a type that compares by identity is, an `Enum`'s
    /// members among them. Anything else is not: `Decimal("1.0")` equals
    /// `Decimal("1.00")`, and a type whose `__eq__` is its own may equal values
    /// that differ.
    fn constant_is_alone(&self, constant: ConstIx) -> bool {
        let Some(value) = self.literals.get(constant.get()) else {
            return false;
        };
        let value = value.bind(self.py);
        let ty = value.get_type();
        if value.is_none()
            || ty.is(PyBool::type_object(self.py))
            || ty.is(PyInt::type_object(self.py))
            || ty.is(PyString::type_object(self.py))
            || ty.is(PyBytes::type_object(self.py))
        {
            return true;
        }
        if ty.is(PyFloat::type_object(self.py)) {
            return value.extract::<f64>().is_ok_and(|float| float != 0.0);
        }
        if let Ok(complex) = value.cast_exact::<PyComplex>() {
            return complex.real() != 0.0 && complex.imag() != 0.0;
        }
        self.heard(compares_by_identity(&ty)).unwrap_or(false)
    }

    /// Whether the values the constant's literal holds -- the model's
    /// singleton [FCB08, §4.2] widened to the values `==` joins to the
    /// constant -- read alike to a pattern, a length, a bound and a step, as
    /// [`equal_values_read_alike`] reads the constant's class. A check that
    /// raised has answered nothing.
    fn equal_values_read_alike(&self, constant: ConstIx) -> bool {
        self.literals.get(constant.get()).is_some_and(|value| {
            self.heard(equal_values_read_alike(value.bind(self.py)))
                .unwrap_or(false)
        })
    }

    fn compare(&self, left: OperandIx, right: OperandIx) -> Option<core::cmp::Ordering> {
        // Order two refinement-bound values by Python's own comparison, so the
        // core can decide an unsatisfiable bound conjunction. An incomparable
        // pair (a TypeError) leaves the bound undecided, and so does an operand
        // the walk reads through another order than this one.
        let left = self.literals.get(left.get())?.bind(self.py);
        let right = self.literals.get(right.get())?.bind(self.py);
        if !(self.heard(orders_as_the_walk_reads(left))?
            && self.heard(orders_as_the_walk_reads(right))?)
        {
            return None;
        }
        // A list or a tuple orders item by item, stopping at the first pair
        // that differs, so a value orders against one bound and raises against
        // another: `[2 + 0j] >= [2]` holds, and `[2 + 0j] >= [1]` orders a
        // complex number against an integer. No order between the bounds is an
        // order of the values.
        if [left, right].iter().any(|operand| {
            operand.is_instance_of::<PyList>() || operand.is_instance_of::<PyTuple>()
        }) {
            return None;
        }
        self.heard(left.compare(right))
    }

    fn divides(&self, step: OperandIx, multiple: OperandIx) -> Option<bool> {
        // `m % s == 0`, asked of the two steps by the operator the walk asks
        // of a value, reflected operand included: `4 % 2.0` is answered by
        // `float.__rmod__`. A step is whatever a caller wrote, so the operator
        // decides what dividing means: `int`, `float`, `Decimal` and
        // `Fraction` each answer by their own rules, and a pair that cannot be
        // divided at all raises a `TypeError` and leaves the question
        // undecided.
        //
        // The remainder is compared with `==` against the integer `0`, as the
        // walk compares it: a remainder is of the operands' type, and
        // `Decimal("0") == 0` is true while `timedelta(0) == 0` is not. The
        // constructor refuses a step whose remainder equals no zero, which is
        // what makes this comparison the one the build already made.
        let step = self.literals.get(step.get())?.bind(self.py);
        let multiple = self.literals.get(multiple.get())?.bind(self.py);
        self.heard(multiple.rem(step).and_then(|rest| rest.eq(0i64)))
    }

    fn operand_kind(&self, operand: OperandIx) -> Option<Kind> {
        kind_of(self.literals.get(operand.get())?.bind(self.py))
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
        // An operand the walk orders otherwise than `floor` and `ceil` round it
        // names no interval here.
        if !(self.heard(orders_as_the_walk_reads(lo))?
            && self.heard(orders_as_the_walk_reads(hi))?)
        {
            return None;
        }
        // A non-real bound (`math.floor` raises a `TypeError`) or a non-finite one
        // (an `OverflowError`) leaves the rule undecided rather than guessing.
        let one = 1i64;
        let least = self.heard(if lo_strict {
            floor
                .call1((&lo,))
                .and_then(|low| low.call_method1(intern!(self.py, "__add__"), (one,)))
        } else {
            ceil.call1((&lo,))
        })?;
        let greatest = self.heard(if hi_strict {
            ceil.call1((&hi,))
                .and_then(|high| high.call_method1(intern!(self.py, "__sub__"), (one,)))
        } else {
            floor.call1((&hi,))
        })?;
        // `least > greatest` means the interval skips every integer.
        Some(matches!(
            self.heard(least.compare(&greatest))?,
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
