//! The schema frontend: build the IR from Python types, typing annotations,
//! native container forms, and already-compiled validators.

use std::cell::{Cell, RefCell};

use pyo3::PyTypeInfo;
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{
    PyBool, PyBytes, PyDict, PyFloat, PyInt, PyList, PyModule, PySet, PyString, PyTuple, PyType,
};
use valgebra_core::{
    ClassIx, ConstIx, DefIx, DefShift, Guarded, OperandIx, PredIx, Schema, fresh_self_token,
};

use crate::errors::summarize;
use crate::validator::Validator;

thread_local! {
    /// Depth of the current `build_schema` recursion, bounding it so a
    /// self-referential class (whose field type names the class) fails cleanly
    /// instead of recursing until the native stack overflows.
    static BUILD_DEPTH: Cell<usize> = const { Cell::new(0) };

    /// The PEP 695 aliases whose bodies are being built on this thread.
    ///
    /// One entry per alias, holding the object's address, the self-reference
    /// token standing for it while its body is read, and whether that token was
    /// handed out. An alias reached again while its own body is being built is
    /// the fixpoint's back edge, and this is what tells the two apart -- the
    /// address, because an alias is one object and `__value__` yields the same
    /// one however many times it is read.
    static OPEN_ALIASES: RefCell<Vec<(usize, u64, Cell<bool>)>> =
        const { RefCell::new(Vec::new()) };
}

/// Build the body of a PEP 695 alias, tying the knot where it names itself.
///
/// `type Json = int | str | list[Json] | dict[str, Json]` is the standard
/// spelling of a recursive type, and the annotation is the whole definition: the
/// alias object is reached again while its own body is being read, and there is
/// no lambda to carry the fixpoint. So the alias *is* the binder. A token stands
/// for it while the body is built, every occurrence of the token becomes a
/// reference to the definition the body turns into, and an alias that never
/// names itself builds exactly what it did before -- one schema, no definition,
/// nothing to resolve.
///
/// The contractivity check is the same one `recursive` runs, for the same
/// reason: `type X = int | X` names a set no value settles, and a walk over it
/// would not terminate.
fn build_alias(
    obj: &Bound<'_, PyAny>,
    lits: &mut Pool,
    defs: &mut Vec<Schema>,
) -> PyResult<Schema> {
    let address = obj.as_ptr() as usize;
    if let Some(token) = OPEN_ALIASES.with_borrow(|open| {
        open.iter()
            .find(|(at, _, _)| *at == address)
            .map(|(_, token, used)| {
                used.set(true);
                *token
            })
    }) {
        return Ok(Schema::SelfRef(token));
    }

    let token = fresh_self_token();
    OPEN_ALIASES.with_borrow_mut(|open| open.push((address, token, Cell::new(false))));
    let body = build_schema(&obj.getattr("__value__")?, lits, defs);
    let recursive = OPEN_ALIASES
        .with_borrow_mut(Vec::pop)
        .is_some_and(|(_, _, used)| used.get());
    let body = body?;
    if !recursive {
        return Ok(body);
    }

    // The body becomes a definition and every occurrence of the token becomes a
    // reference to it -- in the body, and in any definition the body's own build
    // appended, since an inner fixpoint may name this one.
    let ref_id = DefIx::new(defs.len());
    for definition in defs.iter_mut() {
        *definition = definition.resolve_self(token, ref_id);
    }
    let resolved = body.resolve_self(token, ref_id);
    if resolved.occurs_unguarded_under(ref_id, Guarded::No, defs) {
        return Err(PyValueError::new_err(
            "recursive type alias is not contractive: the alias names itself \
             outside any structural constructor, so it denotes no set. Put the \
             self-reference under a list, tuple, set, dict, record, or object",
        ));
    }
    defs.push(resolved);
    Ok(Schema::Ref(ref_id))
}

/// The most levels of schema nesting the frontend descends while compiling.
///
/// One level above the published construction depth, so the frontend never
/// rejects a schema the construction bounds would accept: below this,
/// `Validator::checked` owns the verdict and reports which bound tripped. A
/// self-referential class is the shape that reaches this one, because its field
/// type names the class and the descent does not terminate.
///
/// The margin is what makes the ownership hold rather than happen to hold. Each
/// level the frontend descends builds at least one schema node, so the deepest
/// schema `checked` accepts is reached in at most that many descents; one more
/// lets the schema *past* the bound be built and refused by name. The two
/// constants were equal while a sequence counted two levels of depth per level of
/// descent, which hid the question: no list chain could reach this bound before
/// `checked` had already refused it.
const MAX_BUILD_DEPTH: usize = crate::validator::MAX_SCHEMA_DEPTH + 1;

/// RAII guard that bounds `build_schema` recursion. Entering past the bound is an
/// error; leaving (including on an early `?`) restores the depth.
struct BuildGuard;

impl BuildGuard {
    fn enter() -> PyResult<Self> {
        let depth = BUILD_DEPTH.with(|cell| {
            let depth = cell.get() + 1;
            cell.set(depth);
            depth
        });
        if depth > MAX_BUILD_DEPTH {
            BUILD_DEPTH.with(|cell| cell.set(cell.get() - 1));
            return Err(not_implemented(
                format!(
                    "schema nesting is too deep to compile: the frontend descended \
                 {MAX_BUILD_DEPTH} levels without reaching a leaf. A class whose \
                 own type appears in its fields is recursive and never bottoms \
                 out; write it with recursive(...), which ties the fixpoint \
                 explicitly."
                )
                .as_str(),
            ));
        }
        Ok(BuildGuard)
    }
}

impl Drop for BuildGuard {
    fn drop(&mut self) {
        BUILD_DEPTH.with(|cell| cell.set(cell.get() - 1));
    }
}

/// Per-interpreter cache of the typing and builtins special-form objects the
/// frontend compares schema descriptions against. Resolved once on first
/// compile rather than re-fetched on every node, so a deep schema does not
/// re-import a module and re-`getattr` the same singleton dozens of times. The
/// handles are immutable interpreter singletons; the optional ones are absent on
/// older Pythons (`Never`/`TypeAliasType`).
struct Forms {
    any: Py<PyAny>,
    never: Option<Py<PyAny>>,
    /// The sentinel a `TypedDict` carries in `__extra_items__` when its author
    /// gave no `extra_items` at all -- absent on a runtime without PEP 728.
    no_extra_items: Option<Py<PyAny>>,
    noreturn: Option<Py<PyAny>>,
    type_alias_type: Option<Py<PyAny>>,
    union: Py<PyAny>,
    optional: Py<PyAny>,
    union_type: Py<PyAny>,
    literal: Py<PyAny>,
    /// `typing.get_origin` and `typing.get_args`, the spec's own introspection,
    /// held as the callables they are.
    ///
    /// Every node asks both, and asking them through the module means importing
    /// `typing` and resolving the name per node: the import machinery alone was
    /// 7.7% of building a fifty-field validator, for a module `sys.modules` has
    /// held since the first one.
    get_origin: Py<PyAny>,
    get_args: Py<PyAny>,
    /// `typing.ForwardRef`, and the four classes a type variable or special
    /// form is an instance of, in the order they are asked.
    forward_ref: Option<Py<PyAny>>,
    type_variables: Vec<Py<PyAny>>,
    /// The `TypedDict` field qualifiers: `Required`, `NotRequired`,
    /// `ReadOnly`. Absent where the runtime is older than the one that spells
    /// them.
    required: Option<Py<PyAny>>,
    not_required: Option<Py<PyAny>>,
    read_only: Option<Py<PyAny>>,
    /// `typing.Unpack`, the origin of an unpacked tuple.
    unpack: Option<Py<PyAny>>,
    /// `typing.get_type_hints`, which resolves a class's annotations.
    get_type_hints: Py<PyAny>,
    object: Py<PyAny>,
    enum_class: Py<PyAny>,
    callable: Py<PyAny>,
    ellipsis: Py<PyAny>,
}

static FORMS: PyOnceLock<Forms> = PyOnceLock::new();

/// The special-form cache for this interpreter, built once. `PyOnceLock` makes
/// the one-time initialization safe under free-threading.
fn forms(py: Python<'_>) -> PyResult<&'static Forms> {
    FORMS.get_or_try_init(py, || {
        let typing = py.import("typing")?;
        let builtins = py.import("builtins")?;
        let optional_form = |module: &Bound<'_, PyModule>, name: &str| -> Option<Py<PyAny>> {
            module.getattr(name).ok().map(Bound::unbind)
        };
        Ok(Forms {
            any: typing.getattr("Any")?.unbind(),
            never: optional_form(&typing, "Never"),
            no_extra_items: optional_form(&typing, "NoExtraItems"),
            noreturn: optional_form(&typing, "NoReturn"),
            type_alias_type: optional_form(&typing, "TypeAliasType"),
            union: typing.getattr("Union")?.unbind(),
            optional: typing.getattr("Optional")?.unbind(),
            union_type: py.import("types")?.getattr("UnionType")?.unbind(),
            literal: typing.getattr("Literal")?.unbind(),
            get_origin: typing.getattr("get_origin")?.unbind(),
            get_args: typing.getattr("get_args")?.unbind(),
            forward_ref: optional_form(&typing, "ForwardRef"),
            type_variables: ["TypeVar", "ParamSpec", "TypeVarTuple", "_SpecialForm"]
                .iter()
                .filter_map(|name| optional_form(&typing, name))
                .collect(),
            required: optional_form(&typing, "Required"),
            not_required: optional_form(&typing, "NotRequired"),
            read_only: optional_form(&typing, "ReadOnly"),
            unpack: optional_form(&typing, "Unpack"),
            get_type_hints: typing.getattr("get_type_hints")?.unbind(),
            object: builtins.getattr("object")?.unbind(),
            enum_class: py.import("enum")?.getattr("Enum")?.unbind(),
            callable: py.import("collections.abc")?.getattr("Callable")?.unbind(),
            ellipsis: builtins.getattr("Ellipsis")?.unbind(),
        })
    })
}

/// Build the IR from a native Python schema description.
///
/// Recognized forms: the scalar types and `None`/`type(None)`; `object` as the
/// top schema and `Never`/`NoReturn` as the bottom; the list literal `[T]`
/// (homogeneous), `[A, B]` (fixed-length), and `[A, B, ...]` (prefix-plus-tail);
/// a single `{KeyType: ValueType}` entry as a mapping; an all-string-key dict as
/// a closed record (a trailing `"?"` marks an optional key); any other value as
/// an exact-value literal. Set literals (`{T}`) and tuple literals (`(A, B)`)
/// are rejected: they duplicate `set[T]`/`tuple[...]`, which typing spells.
pub(crate) fn build_schema(
    obj: &Bound<'_, PyAny>,
    lits: &mut Pool,
    defs: &mut Vec<Schema>,
) -> PyResult<Schema> {
    let _guard = BuildGuard::enter()?;
    let py = obj.py();
    if obj.is_none() {
        return Ok(Schema::NoneType);
    }

    let forms = forms(py)?;

    // `typing.Any` is a singleton special form: the gradual dynamic type. It is
    // checked before the type-object dispatch below because on 3.11+ `Any` is
    // itself a class and would otherwise be taken for an ordinary type.
    if obj.is(forms.any.bind(py)) {
        return Ok(Schema::ANY);
    }

    // `typing.Never`/`NoReturn` are the empty type: the lattice bottom, the
    // typing-native spelling of `nothing`. (`object` maps to the top below.)
    // `Never` is 3.11+, so it is absent (skipped) on older Pythons.
    for form in [&forms.never, &forms.noreturn].into_iter().flatten() {
        if obj.is(form.bind(py)) {
            return Ok(Schema::Nothing);
        }
    }

    // A plain type or class (a scalar, `object`, TypedDict, dataclass, enum,
    // protocol, ...) is dispatched here, before the typing introspection below.
    // A type never has a typing origin, so taking this path first skips a
    // `get_origin` call per scalar and class node on the common compile path.
    if let Ok(ty) = obj.cast::<PyType>() {
        return build_type_object(ty, lits, defs);
    }

    // Annotated[T, m1, ...]: the base type T with refinement metadata.
    if obj.hasattr("__metadata__")? {
        let base = obj.getattr("__origin__")?;
        let metadata = obj.getattr("__metadata__")?;
        return build_refine(&base, metadata.cast::<PyTuple>()?, lits, defs);
    }

    // Typing constructs (list[int], dict[K, V], tuple[...], X | Y, Literal,
    // ...) are read through the typing spec's own introspection, so the builtin
    // and legacy aliases share one path. A non-typing object has origin None and
    // falls through to the native handling below.
    let origin = forms.get_origin.bind(py).call1((obj,))?;
    if !origin.is_none() {
        let args = forms.get_args.bind(py).call1((obj,))?;
        let args = args.cast::<PyTuple>()?;
        // A *bare* legacy alias -- `typing.List`, `typing.Tuple` -- is the class
        // it aliases, so it is compiled as that class rather than as a
        // parametrization with no arguments.
        //
        // The two are told apart by whether the form carries a type argument
        // list at all, rather than by whether that list is empty. A bare alias
        // carries none; a parametrisation carries one even where it is empty,
        // and `typing.Tuple[()]` is exactly that -- the empty tuple, a different
        // type from every tuple. `get_args` answers `()` for both, so reading it
        // made `typing.Tuple[()]` admit `(1,)`.
        if !obj.hasattr(intern!(py, "__args__"))?
            && let Ok(class) = origin.cast::<PyType>()
        {
            return build_type_object(class, lits, defs);
        }
        return build_parametrized(&origin, args, lits, defs);
    }

    // PEP 695 `type X = ...` alias (3.12+): validate the aliased type, tying
    // the fixpoint where the alias names itself.
    if let Some(alias_type) = &forms.type_alias_type
        && obj.is_instance(alias_type.bind(py))?
    {
        return build_alias(obj, lits, defs);
    }

    // NewType: validate the supertype it wraps.
    if obj.hasattr("__supertype__")? {
        return build_schema(&obj.getattr("__supertype__")?, lits, defs);
    }

    if let Ok(list) = obj.cast::<PyList>() {
        return build_sequence(list, lits, defs);
    }
    // A tuple literal duplicates `tuple[...]`, which typing spells, so it is not
    // a native form. (The list literal exists only because typing cannot spell a
    // fixed-length list: `[A, B]` is the fixed list, `tuple[A, B]` the tuple.)
    if obj.is_instance_of::<PyTuple>() {
        return Err(not_implemented(
            "a tuple literal is not a schema; write a fixed-length tuple as \
             tuple[A, B] (the list literal [A, B] is the fixed-length list)",
        ));
    }
    // A set literal duplicates `set[T]`, which typing spells, so it is not a
    // native form either.
    if obj.is_instance_of::<PySet>() {
        return Err(not_implemented(
            "a set literal is not a schema; write a set as set[T]",
        ));
    }
    if let Ok(dict) = obj.cast::<PyDict>() {
        return build_dict(dict, lits, defs);
    }

    // An already-compiled validator composes in: intern its pooled constants
    // (so a constant shared by identity with one already present collapses to a
    // single index, which keeps structurally-equal schemas equal across a
    // merge), append its definitions, and remap its schema's indices.
    if let Ok(compiled) = obj.cast::<Validator>() {
        let inner = compiled.get();
        let lit_map: Vec<usize> = inner
            .literals
            .iter()
            .map(|o| lits.intern(o.bind(py)))
            .collect();
        let offset = DefShift::new(place_definitions(&inner.definitions, &lit_map, defs));
        return Ok(inner.schema.reindexed(&lit_map, offset));
    }

    // A type variable or typing special form reaches the constant fallthrough
    // only because it has no typing origin and is not a value. Reject it with a
    // clear message rather than interning it as a literal that would match
    // almost nothing (a free `T` accepts only objects equal to the TypeVar).
    // A forward reference names a type rather than being one, and the constant
    // fallthrough would intern it as a literal of the `ForwardRef` object --
    // matching nothing a caller has. The same refusal a type *argument* already
    // gives, reached at the top level too.
    if is_forward_reference(obj)? {
        return Err(not_implemented(&format!(
            "{} is a forward reference, and a schema is built from the types \
             themselves: resolve the annotation first with typing.get_type_hints(\
             ..., include_extras=True), or write the type rather than its name",
            summarize(obj)
        )));
    }

    if is_typing_construct(obj)? {
        return Err(not_implemented(&format!(
            "{} is a typing construct, not a value: a type variable, ParamSpec, \
             TypeVarTuple, or special form (such as Final or ClassVar) cannot be a \
             schema; use a concrete type",
            summarize(obj)
        )));
    }

    Ok(Schema::Literal(lits.intern_const(obj)))
}

/// True if `obj` is a `typing.ForwardRef`, on a runtime that has one.
fn is_forward_reference(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = obj.py();
    match &forms(py)?.forward_ref {
        Some(class) => obj.is_instance(class.bind(py)),
        None => Ok(false),
    }
}

/// True if `obj` is a type variable or a typing special form (`Final`,
/// `ClassVar`, a bare `Optional`/`Union`/`Literal`, ...): a type-system
/// construct carrying no runtime value, so it cannot denote a set of values.
fn is_typing_construct(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = obj.py();
    for class in &forms(py)?.type_variables {
        if obj.is_instance(class.bind(py))? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Compile arguments into one shared pool and combine them with `make`.
pub(crate) fn combine(
    args: &Bound<'_, PyTuple>,
    make: impl FnOnce(Vec<Schema>, &[Schema]) -> Schema,
) -> PyResult<Validator> {
    let mut literals = Pool::default();
    let mut definitions = Vec::new();
    let mut members = Vec::with_capacity(args.len());
    for arg in args.iter() {
        members.push(build_schema(&arg, &mut literals, &mut definitions)?);
    }
    // The definitions go to the fold as well as to the validator: a member that
    // is a reference is not a set on its own evidence, and the complement law
    // needs to read what it names.
    let schema = make(members, &definitions);
    Validator::checked(schema, literals.into_items(), definitions)
}

/// Where a merged validator's definitions live in the target list.
///
/// Appended, unless the same definitions are already there -- in which case the
/// offset they are already at is returned and nothing is added. That is what
/// makes a recursive schema equal to itself once combined: without it,
/// `intersection(json, complement(json))` copies one definition twice, the two
/// occurrences become `Ref(0)` and `Ref(1)`, and the fold that cancels a schema
/// against its own complement compares terms and sees two. Every law about a
/// recursive schema failed on that, including the one a reader tries first.
///
/// The whole block is matched rather than one definition at a time, because a
/// definition's body names its siblings by index: a block is self-consistent
/// only at the offset it was built for, so it is shifted to each candidate
/// offset and compared there. Quadratic in the definition count, which
/// `MAX_DEFINITIONS` holds at 128, and paid once per merged validator.
fn place_definitions(inner: &[Schema], lit_map: &[usize], defs: &mut Vec<Schema>) -> usize {
    if inner.is_empty() {
        return defs.len();
    }
    let at_offset = |start: usize| -> Vec<Schema> {
        inner
            .iter()
            .map(|d| d.reindexed(lit_map, DefShift::new(start)))
            .collect()
    };
    for start in 0..=defs.len().saturating_sub(inner.len()) {
        if defs.get(start..start + inner.len()) == Some(at_offset(start).as_slice()) {
            return start;
        }
    }
    let start = defs.len();
    defs.extend(at_offset(start));
    start
}

/// What two pooled objects have to agree on to be one constant.
///
/// A builtin scalar is keyed by its exact type and its value, which is the rule
/// a literal is read by everywhere else: `Literal[1]` and `Literal[True]` name
/// different singletons although `1 == True`, and two equal strings name one
/// however they were built. Anything else is keyed by address, because `==` on
/// it is the object's own and an object that answers it inconsistently would
/// merge two constants that are not one.
#[derive(PartialEq, Eq, Hash)]
enum Constant {
    /// An object this cannot read by value: a class, a callable, a container, a
    /// scalar too wide for the key below.
    Address(usize),
    NoneType,
    Bool(bool),
    Int(i64),
    /// A float by its bits, with the one value whose bits are not its identity
    /// folded: `0.0 == -0.0`, so the two are one constant. A `nan` never reaches
    /// here -- it equals nothing, itself included, so it is keyed by address.
    Float(u64),
    Str(String),
    Bytes(Vec<u8>),
}

/// The key an object interns under.
fn constant_of(obj: &Bound<'_, PyAny>) -> Constant {
    let py = obj.py();
    let address = Constant::Address(obj.as_ptr() as usize);
    if obj.is_none() {
        return Constant::NoneType;
    }
    // Exact types only. A subclass carries its own `__eq__`, so two of its
    // instances comparing equal is not the literal rule holding of them.
    let ty = obj.get_type();
    let is = |exact: Bound<'_, PyType>| ty.is(&exact);
    if is(PyBool::type_object(py)) {
        return obj.extract::<bool>().map_or(address, Constant::Bool);
    }
    if is(PyInt::type_object(py)) {
        // An `int` wider than the key falls back to its address, which pools it
        // once per object rather than once per value -- correct, just coarser.
        return obj.extract::<i64>().map_or(address, Constant::Int);
    }
    if is(PyFloat::type_object(py)) {
        return match obj.extract::<f64>() {
            Ok(value) if !value.is_nan() => Constant::Float((value + 0.0).to_bits()),
            _ => address,
        };
    }
    if is(PyString::type_object(py)) {
        return obj.extract::<String>().map_or(address, Constant::Str);
    }
    if is(PyBytes::type_object(py)) {
        return obj.extract::<Vec<u8>>().map_or(address, Constant::Bytes);
    }
    address
}

/// The constants pool a compile builds, plus an index into it. Pooling
/// deduplicates by [`Constant`]; the index makes each `intern` a hash lookup
/// rather than a linear scan, so compiling a wide `Literal[...]` or merging many
/// validators stays linear in the number of constants instead of quadratic.
///
/// The address key is stable for the pooled objects: every interned value is
/// kept alive by `items`, so no live pool entry is freed and reallocated during a
/// compile.
#[derive(Default)]
pub(crate) struct Pool {
    items: Vec<Py<PyAny>>,
    index: rustc_hash::FxHashMap<Constant, usize>,
}

impl Pool {
    /// Seed a pool with an existing validator's constants, rebuilding the index,
    /// so a second schema interns into the same pool when validators merge.
    ///
    /// A seeded pool can hold two slots under one key -- the constants were
    /// pooled before this rule, or by another pool -- and the later slot wins the
    /// key. Both stay in `items`, because a compiled schema already names the
    /// earlier one; what the key decides is only which slot a *new* occurrence
    /// joins.
    pub(crate) fn seeded(py: Python<'_>, items: Vec<Py<PyAny>>) -> Self {
        let index = items
            .iter()
            .enumerate()
            .map(|(at, obj)| (constant_of(obj.bind(py)), at))
            .collect();
        Pool { items, index }
    }

    /// Pool `obj` and return its slot, deduplicating by [`Constant`].
    ///
    /// Private, and reached only through the four typed forms below. One pool
    /// serves four index spaces, so the slot acquires its meaning here, at the
    /// line that decides what the object is being pooled *as*. Two occurrences
    /// of one constant land in one slot whichever space they arrive through,
    /// which is what makes two spellings of a literal one schema node.
    fn intern(&mut self, obj: &Bound<'_, PyAny>) -> usize {
        let key = constant_of(obj);
        if let Some(&index) = self.index.get(&key) {
            return index;
        }
        let index = self.items.len();
        self.items.push(obj.clone().unbind());
        self.index.insert(key, index);
        index
    }

    /// Pool `obj` as the constant of a typed singleton.
    fn intern_const(&mut self, obj: &Bound<'_, PyAny>) -> ConstIx {
        ConstIx::new(self.intern(obj))
    }

    /// Pool `obj` as the class of an `isinstance` atom or an attribute record.
    fn intern_class(&mut self, obj: &Bound<'_, PyAny>) -> ClassIx {
        ClassIx::new(self.intern(obj))
    }

    /// Pool `obj` as a comparison or multiple-of operand.
    fn intern_operand(&mut self, obj: &Bound<'_, PyAny>) -> OperandIx {
        OperandIx::new(self.intern(obj))
    }

    /// Pool `obj` as a user predicate's callable.
    fn intern_predicate(&mut self, obj: &Bound<'_, PyAny>) -> PredIx {
        PredIx::new(self.intern(obj))
    }

    /// The pooled constants, for reading against a compiled schema.
    pub(crate) fn items(&self) -> &[Py<PyAny>] {
        &self.items
    }

    /// Consume the pool, yielding the constants a validator stores.
    pub(crate) fn into_items(self) -> Vec<Py<PyAny>> {
        self.items
    }
}

/// Refuse a map key schema narrowed by a constraint or a predicate.
///
/// A clause's key says which keys it governs, and a map reads that as whole
/// *kinds* of key -- `str`, `int`, and the rest -- or as the constants a
/// `Literal` names. A key narrowed by a bound, a pattern or a callback is
/// neither: it names part of a kind, and two such clauses can overlap without
/// either containing the other. Overlapping key domains are a different theory
/// from the one this library's maps are built on -- with them "there would not
/// be any difference between record types and an intersection of function types"
/// (Castagna, ICFP 2023, §4.5) -- and reading them as this library does, one
/// clause at a time, answers a question the two spellings do not agree on.
///
/// So it is refused where it is written, which is the rule the zero divisor and
/// the invalid pattern already follow.
fn checked_key(schema: &Schema, spelling: &Bound<'_, PyAny>) -> PyResult<()> {
    if !narrows_its_keys(schema) {
        return Ok(());
    }
    Err(not_implemented(&format!(
        "{} narrows the keys it governs, and a map key must be a key type or a \
         Literal: write dict[str, V] to key every string, or dict[Literal[\"a\"], V] \
         (or {{\"a\": V}}) to key one. To constrain the keys themselves, check them \
         beside the mapping rather than inside it",
        summarize(spelling)
    )))
}

/// Whether a key schema narrows its keys by a constraint or a predicate, at any
/// depth a union or a complement can hide one.
fn narrows_its_keys(schema: &Schema) -> bool {
    match schema {
        Schema::Refine { constraints, .. } => !constraints.is_empty(),
        Schema::Union(members) | Schema::Intersection(members) => {
            members.iter().any(narrows_its_keys)
        }
        Schema::Complement(inner) => narrows_its_keys(inner),
        _ => false,
    }
}

pub(crate) fn not_implemented(message: &str) -> PyErr {
    PyNotImplementedError::new_err(message.to_owned())
}

// Needs a live interpreter; compiled and run only under the `interpreter-tests`
// feature, which links an embedded Python.
/// The frontend, driven against real annotations through a real interpreter.
///
/// Why this module exists rather than leaving the frontend to pytest: pytest
/// exercises this file thoroughly and a mutation harness cannot observe pytest,
/// so every mutant of this file read as a survivor and the file was excluded
/// from the sweep by name. An exclusion is a hole in the coverage claim, and
/// this one covered the widest file in the binding -- the one place a wrong arm
/// turns a written annotation into a schema that denotes something else,
/// silently, with every downstream test still passing.
///
/// The corpus is a table rather than a test per case, because the frontend is a
/// dispatch and the thing worth asserting is that each spelling lands on its own
/// arm. Each row is an annotation as a caller writes it and the schema it must
/// build, spelled as the render of that schema -- which is the same string
/// `repr(Validator(...))` prints, so a row can be read against the docs.
///
/// The refinement markers are built here rather than imported. An embedded
/// interpreter starts on the base prefix and does not see a virtual
/// environment's packages, so importing `annotated_types` would make this corpus
/// depend on how the harness was launched. It costs nothing to do without: the
/// frontend reads a marker by *attribute* -- `ge`, `min_length`, `pattern` --
/// which is the contract the vocabulary's classes meet, so an object carrying
/// the attribute exercises the same arm. The one arm that reads a marker's
/// identity is the refusal for a vocabulary member this frontend does not check,
/// and it reads `type(marker).__module__`, which a class here can set.
mod classes;
mod dialect;
mod generics;
mod refine;

use classes::build_type_object;
use generics::{build_dict, build_parametrized, build_sequence};
use refine::build_refine;

#[cfg(all(test, feature = "interpreter-tests"))]
mod interpreter;

#[cfg(all(test, feature = "interpreter-tests"))]
mod tests;
