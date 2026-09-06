//! The schema frontend: build the IR from Python types, typing annotations,
//! native container forms, and already-compiled validators.

use std::cell::{Cell, RefCell};

use pyo3::PyTypeInfo;
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{
    PyBool, PyBytes, PyDict, PyFloat, PyFrozenSet, PyInt, PyList, PyModule, PySet, PyString,
    PyTuple, PyType,
};
use valgebra_core::{
    ClassIx, ConstIx, Constraint, DefIx, DefShift, Field, Guarded, MapClause, OperandIx, PredIx,
    Schema, SeqShape, fresh_self_token,
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
    let typing = py.import("typing")?;

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
    let origin = typing.call_method1("get_origin", (obj,))?;
    if !origin.is_none() {
        let args = typing.call_method1("get_args", (obj,))?;
        return build_parametrized(&origin, args.cast::<PyTuple>()?, lits, defs);
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

/// True if `obj` is a type variable or a typing special form (`Final`,
/// `ClassVar`, a bare `Optional`/`Union`/`Literal`, ...): a type-system
/// construct carrying no runtime value, so it cannot denote a set of values.
fn is_typing_construct(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    let typing = obj.py().import("typing")?;
    for name in ["TypeVar", "ParamSpec", "TypeVarTuple", "_SpecialForm"] {
        if let Ok(class) = typing.getattr(name)
            && obj.is_instance(&class)?
        {
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

/// Build the schema for a Python type object (a builtin, `TypedDict`, `Enum`,
/// dataclass, `NamedTuple`, runtime-checkable `Protocol`, or `object`).
fn build_type_object(
    ty: &Bound<'_, PyType>,
    lits: &mut Pool,
    defs: &mut Vec<Schema>,
) -> PyResult<Schema> {
    let py = ty.py();
    if ty.is(py.get_type::<PyBool>()) {
        return Ok(Schema::Bool);
    }
    if ty.is(py.get_type::<PyInt>()) {
        return Ok(Schema::Int);
    }
    if ty.is(py.get_type::<PyFloat>()) {
        return Ok(Schema::Float);
    }
    if ty.is(py.get_type::<PyString>()) {
        return Ok(Schema::Str);
    }
    if ty.is(py.get_type::<PyBytes>()) {
        return Ok(Schema::Bytes);
    }
    if ty.is(py.None().bind(py).get_type()) {
        return Ok(Schema::NoneType);
    }
    // A bare container class is its kind: `list` admits every list, which is the
    // set `list[object]` names and the set the typing spec assigns an
    // unparameterised generic. Read as an `isinstance` atom it was a different
    // sort of thing from the sequence node beside it -- in no kind at all -- and
    // neither spelling was decided below the other. See "What a bare builtin
    // class denotes" in `docs/dev/01-schema-ir.md`.
    if ty.is(py.get_type::<PyList>()) {
        return Ok(Schema::list(SeqShape::homogeneous(Schema::ANYTHING)));
    }
    if ty.is(py.get_type::<PyTuple>()) {
        return Ok(Schema::tuple(SeqShape::homogeneous(Schema::ANYTHING)));
    }
    if ty.is(py.get_type::<PySet>()) {
        return Ok(Schema::set(Schema::ANYTHING));
    }
    if ty.is(py.get_type::<PyFrozenSet>()) {
        return Ok(Schema::frozen_set(Schema::ANYTHING));
    }
    if ty.is(py.get_type::<PyDict>()) {
        return Ok(Schema::KeyedMap {
            fields: Vec::new(),
            defaults: vec![MapClause::top()],
        });
    }
    let forms = forms(py)?;
    if ty.is(forms.object.bind(py)) {
        return Ok(Schema::ANYTHING);
    }
    // A bare typing special form is a class on some Pythons (notably `Union`).
    // It is not a value, so reject it rather than building an instance check that
    // accepts nothing; the special forms that are not types are rejected on the
    // value fallthrough in build_schema.
    for form in [&forms.union, &forms.optional] {
        if ty.is(form.bind(py)) {
            return Err(not_implemented(&format!(
                "{} is a typing special form, not a value; write a concrete type \
                 (for a union, X | Y or Union[X, Y])",
                summarize(ty.as_any())
            )));
        }
    }
    // TypedDict: a closed record whose required keys come from the class.
    if ty.hasattr("__required_keys__")? {
        return build_typed_dict(ty, lits, defs);
    }
    // Enum: an instance of the enumeration class (any of its members).
    if ty.is_subclass(forms.enum_class.bind(py))? {
        return Ok(Schema::Instance(lits.intern_class(ty.as_any())));
    }
    // dataclass / NamedTuple: isinstance plus a deep check of each field.
    let is_dataclass = py
        .import("dataclasses")?
        .call_method1("is_dataclass", (ty,))?
        .is_truthy()?;
    if is_dataclass || (ty.is_subclass_of::<PyTuple>()? && ty.hasattr("_fields")?) {
        return build_object(ty, lits, defs);
    }
    // Protocol: a runtime-checkable protocol validates by isinstance.
    if is_truthy_attr(ty, "_is_protocol") {
        if is_truthy_attr(ty, "_is_runtime_protocol") {
            return Ok(Schema::Instance(lits.intern_class(ty.as_any())));
        }
        return Err(not_implemented(
            "a Protocol must be @runtime_checkable to be used as a schema",
        ));
    }
    // Any other class names its instances: a bare class is an isinstance check.
    // This covers the remaining builtins (complex, bytearray, memoryview, range,
    // the `collections.abc` ABCs including Callable, ...) and arbitrary user
    // classes uniformly.
    Ok(Schema::Instance(lits.intern_class(ty.as_any())))
}

/// True if `obj.<name>` exists and is truthy; false on absence or error.
fn is_truthy_attr(obj: &Bound<'_, PyAny>, name: &str) -> bool {
    obj.getattr(name)
        .ok()
        .and_then(|value| value.is_truthy().ok())
        .unwrap_or(false)
}

/// Resolve a class's type hints with `Annotated` metadata preserved.
///
/// `include_extras=True` keeps `Annotated[...]` field types intact so a field's
/// refinement markers reach [`build_refine`]; without it `get_type_hints` strips
/// the metadata and the field's constraints are silently lost.
fn resolve_type_hints<'py>(ty: &Bound<'py, PyType>) -> PyResult<Bound<'py, PyAny>> {
    let py = ty.py();
    let kwargs = PyDict::new(py);
    kwargs.set_item("include_extras", true)?;
    py.import("typing")?
        .call_method("get_type_hints", (ty,), Some(&kwargs))
}

/// Read a record field name as Rust text, refusing a key that is not valid
/// Unicode (one carrying a lone surrogate). A field name is stored as UTF-8 and
/// matched against dict keys as UTF-8, so a surrogate key cannot round-trip;
/// refusing it at build time turns silent corruption — a lossy replacement that
/// makes the field unmatchable — into an explicit error.
fn field_name(name: &Bound<'_, PyString>) -> PyResult<String> {
    name.to_str().map(str::to_owned).map_err(|_| {
        PyValueError::new_err(
            "a record key must be valid Unicode; a field name cannot contain a lone surrogate",
        )
    })
}

/// Build the record a `TypedDict` denotes: its keys, and what it says about the
/// ones it does not name.
///
/// **Open unless the class says otherwise**, which is the set the typing spec
/// assigns it -- "By default, `TypedDict`s are open". `Validator(TD)` reads an
/// annotation whose meaning is fixed elsewhere, and reading it as a narrower set
/// is a deviation a caller has no way to see, because the class carries no mark
/// of it. The dict-literal form `{"a": int}` stays closed: it is this library's
/// own spelling, and a schema written as a *shape* means that shape
/// (`docs/dev/01-schema-ir.md`, "What a `TypedDict` denotes").
///
/// `ReadOnly` needs no arm here. It constrains writers and a value has none, and
/// `get_type_hints` has already resolved it away.
fn build_typed_dict(
    ty: &Bound<'_, PyType>,
    lits: &mut Pool,
    defs: &mut Vec<Schema>,
) -> PyResult<Schema> {
    let hints = resolve_type_hints(ty)?;
    let hints = hints.cast::<PyDict>()?;
    let required = ty.getattr("__required_keys__")?;
    let mut fields = Vec::with_capacity(hints.len());
    for (name, hint) in hints.iter() {
        fields.push(Field {
            name: field_name(&name.str()?)?,
            schema: build_schema(&hint, lits, defs)?,
            required: required.contains(&name)?,
        });
    }
    Ok(Schema::keyed_map(fields, unnamed_keys(ty, lits, defs)?))
}

/// Whether `extra_items` carries the sentinel for "the author gave none".
///
/// A runtime with PEP 728 fills `__extra_items__` in either way: with the type
/// its author wrote, or with `NoExtraItems` to say there was none. The sentinel
/// is not a type and reading it as one makes the record admit exactly the
/// sentinel -- which is a closed record wearing an open one's spelling, and is
/// what 3.15 turned this into before the check was here.
fn gave_no_extra_items(extra: &Bound<'_, PyAny>) -> PyResult<bool> {
    let Some(sentinel) = forms(extra.py())?.no_extra_items.as_ref() else {
        return Ok(false);
    };
    Ok(extra.is(sentinel.bind(extra.py())))
}

/// What a `TypedDict` says about the keys it does not name.
///
/// `closed=True` shuts them and `extra_items=T` gives them a type -- PEP 728,
/// which the typing spec carries. Neither marker is in `typing` yet, so both are
/// read where a runtime that has them puts them and are simply absent otherwise:
/// a `TypedDict` written for a runtime without PEP 728 cannot have said either,
/// and the spec's default is what is left.
fn unnamed_keys(
    ty: &Bound<'_, PyType>,
    lits: &mut Pool,
    defs: &mut Vec<Schema>,
) -> PyResult<Vec<MapClause>> {
    if let Ok(flag) = ty.getattr("__closed__")
        && flag.is_truthy()?
    {
        return Ok(Vec::new());
    }
    if let Ok(extra) = ty.getattr("__extra_items__")
        && !extra.is_none()
        && !gave_no_extra_items(&extra)?
    {
        // A `TypedDict`'s keys are strings, so the type it gives the extra ones
        // governs the string keys and leaves no other kind admitted.
        return Ok(vec![MapClause {
            key: Schema::Str,
            value: build_schema(&extra, lits, defs)?,
        }]);
    }
    // A `TypedDict`'s keys are strings -- the spec relates one to
    // `Mapping[str, object]` and to nothing wider -- so being open is being open
    // to further *string* keys, not to keys of every kind.
    Ok(vec![MapClause {
        key: Schema::Str,
        value: Schema::ANYTHING,
    }])
}

/// The attribute names a class declares, in declaration order.
///
/// Not every annotation on a dataclass names an attribute of its instances:
/// `ClassVar` annotates the class, and `InitVar` names a constructor parameter
/// the instance does not keep. Reading the hints alone asks an instance for
/// attributes it cannot have, which is a schema no instance of the class
/// satisfies — the empty set wearing the class's name. Each kind of class keeps
/// its own list of what it declares, so that list is what is read: a dataclass's
/// `fields()` and a named tuple's `_fields`. A field declared `init=False` is on
/// the instance and stays.
fn declared_fields<'py>(ty: &Bound<'py, PyType>) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let py = ty.py();
    let dataclasses = py.import("dataclasses")?;
    if dataclasses
        .call_method1("is_dataclass", (ty,))?
        .is_truthy()?
    {
        return dataclasses
            .call_method1("fields", (ty,))?
            .try_iter()?
            .map(|field| field?.getattr("name"))
            .collect();
    }
    ty.getattr("_fields")?.try_iter()?.collect()
}

/// Build the schema of a class with declared attributes: the meet of its
/// `isinstance` atom and a record of its fields, whose types come from the
/// resolved hints. Every attribute an instance declares is required, because an
/// instance carries it.
///
/// A class with no annotated field is the atom alone. The meet is what the form
/// means, and spelling it out is what lets each half relate on its own; the
/// surface does not change, because `render` reads the pair back as the class
/// name and the walk stops at the failing conjunct, which is what
/// `docs/dev/01-schema-ir.md` records under "What a class with attributes is, on
/// the surface".
fn build_object(
    ty: &Bound<'_, PyType>,
    lits: &mut Pool,
    defs: &mut Vec<Schema>,
) -> PyResult<Schema> {
    let hints = resolve_type_hints(ty)?;
    let hints = hints.cast::<PyDict>()?;
    let class_index = lits.intern_class(ty.as_any());
    let declared = declared_fields(ty)?;
    let mut fields = Vec::with_capacity(declared.len());
    for name in declared {
        // A name the hints do not carry is unannotated -- a `collections`
        // namedtuple's fields are the case -- so there is no type to check and
        // the class's own isinstance test is the whole of it.
        let Some(hint) = hints.get_item(&name)? else {
            continue;
        };
        fields.push(Field {
            name: field_name(&name.str()?)?,
            schema: build_schema(&hint, lits, defs)?,
            required: true,
        });
    }
    let instance = Schema::Instance(class_index);
    if fields.is_empty() {
        return Ok(instance);
    }
    Ok(Schema::meet([instance, Schema::attr_record(fields)]))
}

/// Build the IR for a parametrized typing generic given its origin and args.
fn build_parametrized(
    origin: &Bound<'_, PyAny>,
    args: &Bound<'_, PyTuple>,
    lits: &mut Pool,
    defs: &mut Vec<Schema>,
) -> PyResult<Schema> {
    let py = origin.py();
    if origin.is(py.get_type::<PyList>()) {
        return Ok(Schema::list(SeqShape::homogeneous(build_type_argument(
            &single_arg(args)?,
            lits,
            defs,
        )?)));
    }
    if origin.is(py.get_type::<PySet>()) {
        return Ok(Schema::set(build_type_argument(
            &single_arg(args)?,
            lits,
            defs,
        )?));
    }
    if origin.is(py.get_type::<PyFrozenSet>()) {
        return Ok(Schema::frozen_set(build_type_argument(
            &single_arg(args)?,
            lits,
            defs,
        )?));
    }
    if origin.is(py.get_type::<PyDict>()) {
        if args.len() != 2 {
            return Err(not_implemented(
                "dict[...] needs a key type and a value type",
            ));
        }
        // Named rather than positional: `dict[K, V]` compiled with the two
        // transposed is `dict[V, K]`, which typechecks and validates real values.
        let key_argument = args.get_item(0)?;
        let key = build_type_argument(&key_argument, lits, defs)?;
        checked_key(&key, &key_argument)?;
        return Ok(Schema::mapping(MapClause {
            key,
            value: build_type_argument(&args.get_item(1)?, lits, defs)?,
        }));
    }
    if origin.is(py.get_type::<PyTuple>()) {
        return build_tuple(args, lits, defs);
    }
    if is_field_qualifier(origin)? {
        // A field qualifier survives hint resolution because field metadata is
        // kept (include_extras), so the frontend unwraps it and compiles the type
        // it qualifies.
        return build_type_argument(&single_arg(args)?, lits, defs);
    }
    if is_union_origin(origin)? {
        let mut members = Vec::with_capacity(args.len());
        for arg in args.iter() {
            members.push(build_type_argument(&arg, lits, defs)?);
        }
        return Ok(Schema::union(members));
    }
    if is_literal_origin(origin)? {
        // Literal args are constant values; each becomes a literal, unioned when
        // there is more than one.
        let mut members = Vec::with_capacity(args.len());
        for arg in args.iter() {
            members.push(build_schema(&arg, lits, defs)?);
        }
        return Ok(Schema::union(members));
    }
    if origin.is(forms(py)?.callable.bind(py)) {
        // Callable[...] checks only callability at runtime; the argument and
        // return types cannot be inspected, so the parameters are ignored and
        // the schema is the opaque `isinstance(x, Callable)` test.
        return Ok(Schema::Instance(lits.intern_class(origin)));
    }
    Err(not_implemented(&format!(
        "unsupported typing form with origin {}; supported: list, set, dict, \
         tuple, Union, Optional, Literal, Callable",
        summarize(origin)
    )))
}

/// True if `origin` is `typing.Union` (from Union/Optional) or
/// `types.UnionType` (from `X | Y`).
fn is_union_origin(origin: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = origin.py();
    let forms = forms(py)?;
    Ok(origin.is(forms.union.bind(py)) || origin.is(forms.union_type.bind(py)))
}

/// True if `origin` is `typing.Literal`.
fn is_literal_origin(origin: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = origin.py();
    Ok(origin.is(forms(py)?.literal.bind(py)))
}

/// True if `origin` is one of the `TypedDict` field qualifiers, which
/// `include_extras` keeps in the resolved hints.
///
/// `Required`/`NotRequired` say whether the key must be present, which is read
/// from the class's own `__required_keys__` rather than from the annotation, and
/// `ReadOnly` says whether a consumer may write the key back — a statement about
/// use, not about which values belong. None of the three narrows the field's set,
/// so each is unwrapped to the type it qualifies.
fn is_field_qualifier(origin: &Bound<'_, PyAny>) -> PyResult<bool> {
    let typing = origin.py().import("typing")?;
    for name in ["Required", "NotRequired", "ReadOnly"] {
        if let Ok(marker) = typing.getattr(name)
            && origin.is(&marker)
        {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Compile a *type argument* of a typing form.
///
/// A string here is a forward reference, which the typing spec asks a consumer to
/// resolve against the namespace the annotation was written in. valgebra has no
/// such namespace in hand, and the constant fallthrough would read `list["int"]`
/// as a list of the string `"int"` — a schema that refuses what the annotation
/// admits and admits what it refuses. So the position is refused, and the caller
/// is pointed at the resolution the spec provides.
///
/// `Literal["a"]`'s arguments are not this position: they are values, which is
/// why the literal arm compiles them straight through. Nor is the native list or
/// dict literal, where a bare value is a literal by design.
fn build_type_argument(
    arg: &Bound<'_, PyAny>,
    lits: &mut Pool,
    defs: &mut Vec<Schema>,
) -> PyResult<Schema> {
    let py = arg.py();
    let forward_ref = match py.import("typing")?.getattr("ForwardRef") {
        Ok(class) => arg.is_instance(&class)?,
        Err(_) => false,
    };
    if arg.is_instance_of::<PyString>() || forward_ref {
        return Err(not_implemented(&format!(
            "{} is a forward reference, and a schema is built from the types \
             themselves: resolve the annotation first with typing.get_type_hints(\
             ..., include_extras=True), or write the type rather than its name",
            summarize(arg)
        )));
    }
    build_schema(arg, lits, defs)
}

/// What an unpacked tuple argument contributes to the tuple that carries it.
enum Unpacked<'py> {
    /// `*tuple[A, B]`: its elements splice in where it stands.
    Fixed(Vec<Bound<'py, PyAny>>),
    /// `*tuple[B, ...]`: the tuple repeats `B` from here on.
    Tail(Bound<'py, PyAny>),
}

/// Read `arg` as an unpacked tuple, or `None` when it is an ordinary element.
///
/// Two spellings say the same thing: `*tuple[B, ...]` is the tuple alias itself
/// carrying a flag, and `Unpack[tuple[B, ...]]` wraps it. Both have to be read,
/// because a reader who writes one and a reader who writes the other mean one
/// annotation, and reading only the second leaves the first looking like a
/// nested tuple — a schema that admits a different set than the annotation says.
///
/// Only a tuple can be unpacked into a tuple. `*Ts` over a `TypeVarTuple` binds
/// no element types at runtime, so it is refused here rather than read as an
/// empty splice.
fn unpacked_tuple<'py>(arg: &Bound<'py, PyAny>) -> PyResult<Option<Unpacked<'py>>> {
    let py = arg.py();
    let typing = py.import("typing")?;
    let inner = if is_truthy_attr(arg, "__unpacked__") {
        arg.clone()
    } else {
        let Ok(unpack) = typing.getattr("Unpack") else {
            return Ok(None);
        };
        if !typing.call_method1("get_origin", (arg,))?.is(&unpack) {
            return Ok(None);
        }
        let wrapped = typing.call_method1("get_args", (arg,))?;
        single_arg(wrapped.cast::<PyTuple>()?)?
    };
    if !typing
        .call_method1("get_origin", (&inner,))?
        .is(py.get_type::<PyTuple>())
    {
        return Err(not_implemented(&format!(
            "only a tuple can be unpacked into a tuple schema; {} binds no \
             element types at runtime",
            summarize(&inner)
        )));
    }
    let args = typing.call_method1("get_args", (&inner,))?;
    let args = args.cast::<PyTuple>()?;
    let len = args.len();
    if len == 2 && is_ellipsis(&args.get_item(1)?) {
        return Ok(Some(Unpacked::Tail(args.get_item(0)?)));
    }
    Ok(Some(Unpacked::Fixed(args.iter().collect())))
}

/// `tuple[...]`, in every shape typing spells it.
///
/// A trailing `...` repeats the element before it after a fixed prefix, and an
/// unpacked variadic tuple — `tuple[A, *tuple[B, ...]]` — is that same shape said
/// another way, so both compile to the prefix-and-tail form. An unpacked *fixed*
/// tuple splices its elements in where it stands.
///
/// What a sequence carries is a fixed prefix and then a repeating tail, with
/// nothing after it. An element following the tail is therefore refused: the set
/// it names is one this algebra cannot spell, and reading it as anything else
/// would admit a different one.
fn build_tuple(
    args: &Bound<'_, PyTuple>,
    lits: &mut Pool,
    defs: &mut Vec<Schema>,
) -> PyResult<Schema> {
    let mut prefix: Vec<Bound<'_, PyAny>> = Vec::with_capacity(args.len());
    let mut tail: Option<Bound<'_, PyAny>> = None;
    for arg in args.iter() {
        if tail.is_some() {
            return Err(not_implemented(
                "a tuple schema carries a fixed prefix and then a repeating \
                 tail, so nothing may follow the tail: write the repeating part \
                 last",
            ));
        }
        match unpacked_tuple(&arg)? {
            Some(Unpacked::Fixed(items)) => prefix.extend(items),
            Some(Unpacked::Tail(item)) => tail = Some(item),
            None if is_ellipsis(&arg) => {
                let Some(repeated) = prefix.pop() else {
                    return Err(not_implemented(
                        "`...` repeats the element before it, so a tuple schema \
                         cannot begin with one",
                    ));
                };
                tail = Some(repeated);
            }
            None => prefix.push(arg),
        }
    }
    let mut elements = Vec::with_capacity(prefix.len());
    for element in &prefix {
        elements.push(build_type_argument(element, lits, defs)?);
    }
    let regex = match tail {
        None => SeqShape::fixed(elements),
        Some(tail) => {
            let tail = build_type_argument(&tail, lits, defs)?;
            if elements.is_empty() {
                SeqShape::homogeneous(tail)
            } else {
                SeqShape::prefix_tail(elements, tail)
            }
        }
    };
    Ok(Schema::tuple(regex))
}

fn single_arg<'py>(args: &Bound<'py, PyTuple>) -> PyResult<Bound<'py, PyAny>> {
    if args.len() == 1 {
        args.get_item(0)
    } else {
        Err(not_implemented("expected exactly one type argument"))
    }
}

fn build_sequence(
    list: &Bound<'_, PyList>,
    lits: &mut Pool,
    defs: &mut Vec<Schema>,
) -> PyResult<Schema> {
    let len = list.len();
    // [T]: a homogeneous list of T (the single-element idiom).
    if len == 1 && !is_ellipsis(&list.get_item(0)?) {
        let element = build_schema(&list.get_item(0)?, lits, defs)?;
        return Ok(Schema::list(SeqShape::homogeneous(element)));
    }
    // [p0, ..., tail, ...]: a trailing `...` repeats the element before it, after
    // a fixed prefix of the earlier elements. [T, ...] is the prefix-free case,
    // and [T, T, ...] is the non-empty list.
    if len >= 2 && is_ellipsis(&list.get_item(len - 1)?) {
        let mut elements = Vec::with_capacity(len - 1);
        for index in 0..len - 1 {
            let item = list.get_item(index)?;
            if is_ellipsis(&item) {
                return Err(not_implemented(
                    "`...` may appear only as the last element of a list schema",
                ));
            }
            elements.push(build_schema(&item, lits, defs)?);
        }
        let tail = elements.pop().expect("at least one element precedes `...`");
        let regex = if elements.is_empty() {
            SeqShape::homogeneous(tail)
        } else {
            SeqShape::prefix_tail(elements, tail)
        };
        return Ok(Schema::list(regex));
    }
    // [A, B]: a fixed-length list matched positionally (and `[]` the empty list).
    // typing cannot spell a fixed-length list (`list[A, B]` is illegal), so the
    // list literal carries this shape; `tuple[A, B]` is the tuple counterpart.
    let mut elements = Vec::with_capacity(len);
    for index in 0..len {
        let item = list.get_item(index)?;
        if is_ellipsis(&item) {
            return Err(not_implemented(
                "`...` may appear only as the last element of a list schema",
            ));
        }
        elements.push(build_schema(&item, lits, defs)?);
    }
    Ok(Schema::list(SeqShape::fixed(elements)))
}

fn build_dict(
    dict: &Bound<'_, PyDict>,
    lits: &mut Pool,
    defs: &mut Vec<Schema>,
) -> PyResult<Schema> {
    // A string key is a named field (with the `"key?"` optional convention); any
    // other key is a schema keying a default clause for the rest. All string keys
    // give a record; a single schema key with no fields gives `dict[K, V]`;
    // several schema keys a heterogeneous mapping; a mix a record with a typed
    // catch-all; the empty dict the empty closed record.
    let mut fields = Vec::new();
    let mut defaults = Vec::new();
    for (key, value) in dict.iter() {
        if let Ok(name) = key.cast::<PyString>() {
            let raw = field_name(name)?;
            let (name, required) = match raw.strip_suffix('?') {
                Some(stripped) => (stripped.to_owned(), false),
                None => (raw, true),
            };
            fields.push(Field {
                name,
                schema: build_schema(&value, lits, defs)?,
                required,
            });
        } else {
            let key_schema = build_schema(&key, lits, defs)?;
            checked_key(&key_schema, &key)?;
            defaults.push(MapClause {
                key: key_schema,
                value: build_schema(&value, lits, defs)?,
            });
        }
    }
    Ok(Schema::keyed_map(fields, defaults))
}

/// Build a Refine node from an `Annotated` base and its metadata markers.
///
/// Markers are read structurally (annotated-types style): an object exposing
/// `ge`/`gt`/`le`/`lt` contributes a comparison bound, `min_length`/
/// `max_length` contribute length bounds, and `func` (or a bare callable)
/// contributes a predicate. Unrecognized metadata is ignored, per the typing
/// spec. With no recognized constraint the base schema is returned as-is.
fn build_refine(
    base: &Bound<'_, PyAny>,
    metadata: &Bound<'_, PyTuple>,
    lits: &mut Pool,
    defs: &mut Vec<Schema>,
) -> PyResult<Schema> {
    let base_schema = build_schema(base, lits, defs)?;
    let mut constraints = Vec::new();
    for marker in metadata.iter() {
        parse_constraint(&marker, &mut constraints, lits)?;
    }
    for constraint in &constraints {
        check_constraint_fits(&base_schema, constraint, lits)?;
    }
    Ok(Schema::refine(base_schema, constraints))
}

/// Whether the values a base admits can answer a constraint.
///
/// Three answers, because a refusal needs certainty. A constraint is refused only
/// where *no* value of the base can answer it, since that is the case where the
/// refinement denotes the empty set and the marker was written to narrow rather
/// than to empty. Where the base is opaque — a class, the gradual atom, a
/// literal's pooled constant, a recursive reference — the check stands aside.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Carries {
    /// Every value of the base can answer the constraint.
    Yes,
    /// No value of the base can, so the refinement admits nothing.
    No,
    /// The base does not say.
    Maybe,
}

impl Carries {
    /// The answer for a base that is a union of the two.
    ///
    /// A member that can answer makes the constraint a narrowing of the union
    /// rather than an emptying of it: `Annotated[int | str, MinLen(1)]` is the
    /// non-empty strings, which is a set a reader can mean.
    fn or(self, other: Carries) -> Carries {
        match (self, other) {
            (Carries::Yes, _) | (_, Carries::Yes) => Carries::Yes,
            (Carries::Maybe, _) | (_, Carries::Maybe) => Carries::Maybe,
            (Carries::No, Carries::No) => Carries::No,
        }
    }
}

/// Fold `answer` over the members of a union, and stand aside anywhere else that
/// is not a plain base: an intersection or a complement narrows a set this check
/// does not compute, and a refinement's answer is its own base's.
fn carries_through(base: &Schema, answer: &impl Fn(&Schema) -> Carries) -> Option<Carries> {
    match base {
        Schema::Union(members) => Some(members.iter().map(answer).fold(Carries::No, Carries::or)),
        Schema::Refine { base, .. } => Some(answer(base)),
        Schema::Intersection(_) | Schema::Complement(_) | Schema::Ref(_) | Schema::SelfRef(_) => {
            Some(Carries::Maybe)
        }
        _ => None,
    }
}

/// Whether the base's values have a length.
fn carries_length(base: &Schema) -> Carries {
    if let Some(answer) = carries_through(base, &carries_length) {
        return answer;
    }
    match base {
        Schema::Str
        | Schema::Bytes
        | Schema::Seq { .. }
        | Schema::Coll { .. }
        | Schema::KeyedMap { .. } => Carries::Yes,
        Schema::NoneType | Schema::Bool | Schema::Int | Schema::Float => Carries::No,
        _ => Carries::Maybe,
    }
}

/// Whether the base's values are text a pattern can be matched against.
fn carries_pattern(base: &Schema) -> Carries {
    if let Some(answer) = carries_through(base, &carries_pattern) {
        return answer;
    }
    match base {
        Schema::Str => Carries::Yes,
        Schema::NoneType
        | Schema::Bool
        | Schema::Int
        | Schema::Float
        | Schema::Bytes
        | Schema::Seq { .. }
        | Schema::Coll { .. }
        | Schema::KeyedMap { .. } => Carries::No,
        _ => Carries::Maybe,
    }
}

/// Whether the base's values are numbers, which is what a divisor needs.
fn carries_division(base: &Schema) -> Carries {
    if let Some(answer) = carries_through(base, &carries_division) {
        return answer;
    }
    match base {
        Schema::Bool | Schema::Int | Schema::Float => Carries::Yes,
        Schema::NoneType
        | Schema::Str
        | Schema::Bytes
        | Schema::Seq { .. }
        | Schema::Coll { .. }
        | Schema::KeyedMap { .. } => Carries::No,
        _ => Carries::Maybe,
    }
}

/// Whether the base's values are ordered against `operand`.
///
/// Python orders numbers with numbers, text with text and bytes with bytes, and
/// raises across those groups. A bound whose operand is in another group than the
/// base compares nothing, so the refinement it builds admits nothing.
fn carries_order(base: &Schema, operand: &Bound<'_, PyAny>) -> Carries {
    let by_group = |base: &Schema| carries_order(base, operand);
    if let Some(answer) = carries_through(base, &by_group) {
        return answer;
    }
    let number = operand
        .py()
        .import("numbers")
        .and_then(|numbers| numbers.getattr("Number"))
        .is_ok_and(|class| operand.is_instance(&class).unwrap_or(false));
    let matches = match base {
        Schema::Bool | Schema::Int | Schema::Float => number,
        Schema::Str => operand.is_instance_of::<PyString>(),
        Schema::Bytes => operand.is_instance_of::<PyBytes>(),
        _ => return Carries::Maybe,
    };
    if matches { Carries::Yes } else { Carries::No }
}

/// Refuse a constraint no value of the base can answer.
///
/// A constraint that cannot be asked of a value is not a narrowing: reading a
/// length off an `int` raises, and the walk reads a raise as a non-member, so
/// `Annotated[int, MinLen(1)]` compiles to a schema that admits nothing at all
/// and says nothing about why. That is a schema nobody writes on purpose, so it
/// is refused where it is written rather than at the first value that meets it.
fn check_constraint_fits(base: &Schema, constraint: &Constraint, lits: &Pool) -> PyResult<()> {
    let operand = |index: OperandIx| lits.items().get(index.get());
    let (answer, what) = match constraint {
        Constraint::MinLen(_) | Constraint::MaxLen(_) => (carries_length(base), "length"),
        Constraint::Regex(_) => (carries_pattern(base), "text for a pattern to match"),
        Constraint::MultipleOf(index) => match operand(*index) {
            Some(_) => (carries_division(base), "number for a divisor"),
            None => (Carries::Maybe, ""),
        },
        Constraint::Ge(index)
        | Constraint::Gt(index)
        | Constraint::Le(index)
        | Constraint::Lt(index) => match operand(*index) {
            Some(bound) => (
                Python::attach(|py| carries_order(base, bound.bind(py))),
                "order against that bound",
            ),
            None => (Carries::Maybe, ""),
        },
        Constraint::Predicate(_) => (Carries::Maybe, ""),
    };
    if answer == Carries::No {
        return Err(not_implemented(&format!(
            "{} values have no {what}, so this constraint admits none of them; \
             constrain a base the constraint can be asked of",
            base.expected()
        )));
    }
    Ok(())
}

/// The compilation flags a `re.Pattern` carries, folded into the pattern itself.
///
/// A compiled pattern keeps its flags beside its source, and the source alone is
/// a different expression: `re.compile("abc", re.I)` matches `"ABC"` and `"abc"`
/// does not. Dropping them silently narrows the set the marker was written for,
/// so each flag is either written into the pattern -- the engine here reads the
/// same inline spellings -- or refused by name.
///
/// `re.UNICODE` is the default for a `str` pattern in both engines and says
/// nothing extra, so it is not named here at all. `re.ASCII` and `re.LOCALE` change what a
/// character class means in ways this engine spells differently, and `re.DEBUG`
/// asks the other engine to talk about itself, so all three are refused rather
/// than approximated.
fn with_inline_flags(marker: &Bound<'_, PyAny>, pattern: String) -> PyResult<String> {
    // `re`'s own bit values, which are part of its published interface.
    const IGNORECASE: u32 = 2;
    const LOCALE: u32 = 4;
    const MULTILINE: u32 = 8;
    const DOTALL: u32 = 16;
    const DEBUG: u32 = 128;
    const VERBOSE: u32 = 64;
    const ASCII: u32 = 256;

    let Ok(flags) = marker.getattr("flags") else {
        return Ok(pattern);
    };
    let Ok(flags) = flags.extract::<u32>() else {
        return Ok(pattern);
    };
    for (bit, name, why) in [
        (
            ASCII,
            "re.ASCII",
            "write the ASCII spellings ([0-9], [A-Za-z0-9_]) instead",
        ),
        (
            LOCALE,
            "re.LOCALE",
            "a pattern here does not depend on a locale",
        ),
        (
            DEBUG,
            "re.DEBUG",
            "it asks the other engine to report on itself",
        ),
    ] {
        if flags & bit != 0 {
            return Err(not_implemented(&format!(
                "{name} cannot be carried into this pattern: {why}"
            )));
        }
    }
    let mut inline = String::new();
    for (bit, letter) in [
        (IGNORECASE, 'i'),
        (MULTILINE, 'm'),
        (DOTALL, 's'),
        (VERBOSE, 'x'),
    ] {
        if flags & bit != 0 {
            inline.push(letter);
        }
    }
    if inline.is_empty() {
        Ok(pattern)
    } else {
        Ok(format!("(?{inline}){pattern}"))
    }
}

/// Whether `marker` comes from `annotated_types`, whose vocabulary a reader
/// expects this frontend to know.
///
/// The typing spec says to ignore metadata a consumer does not recognise, and
/// that is right for metadata written for someone else. A marker from the
/// constraint vocabulary is not that: it was written to narrow this schema, and
/// ignoring it leaves a validator that admits everything the marker excludes.
/// The one member carrying no constraint is the documentation marker, which says
/// nothing about which values belong.
fn is_unhandled_constraint(marker: &Bound<'_, PyAny>) -> bool {
    let ty = marker.get_type();
    let from_vocabulary = ty
        .getattr("__module__")
        .ok()
        .and_then(|module| module.extract::<String>().ok())
        .is_some_and(|module| module == "annotated_types");
    let documentation = ty
        .getattr("__name__")
        .ok()
        .and_then(|name| name.extract::<String>().ok())
        .is_some_and(|name| name == "DocInfo");
    from_vocabulary && !documentation
}

fn parse_constraint(
    marker: &Bound<'_, PyAny>,
    out: &mut Vec<Constraint>,
    lits: &mut Pool,
) -> PyResult<()> {
    // A class is metadata this frontend does not recognise, and the typing spec
    // says to ignore what a consumer does not recognise.
    //
    // It has to be refused before any attribute is read, not only before the
    // predicate arms. A marker *class* exposes descriptors where an instance
    // exposes values: `at.Ge(0)` carries `ge = 0`, while `at.Ge` carries the
    // slot descriptor that reads it, and taking that for a bound builds a
    // comparison no value is ordered against. Calling one is the same trap a
    // step later — `Kilograms(1.5)` constructs a unit marker rather than
    // answering whether 1.5 belongs.
    if marker.is_instance_of::<PyType>() {
        return Ok(());
    }
    let before = out.len();

    // A string-pattern marker: valgebra's `Regex(...)` or a compiled
    // `re.Pattern`, both carrying the source pattern as `.pattern`. The pattern
    // is validated (anchored) here so an invalid expression fails at compile
    // time, not at first validation; the compiled regex is cached per validator.
    if let Ok(attr) = marker.getattr("pattern") {
        let Ok(pattern) = attr.extract::<String>() else {
            // A `bytes` pattern: `re` compiles one against `bytes` values, and a
            // pattern constraint here matches the text of a `str`. Reading the
            // marker and dropping the pattern would leave a schema that admits
            // every value of its base, which is the opposite of what a pattern
            // is written for.
            return Err(not_implemented(
                "a bytes pattern cannot constrain a schema: a pattern is matched \
                 against text, so write the pattern as a str",
            ));
        };
        let pattern = with_inline_flags(marker, pattern)?;
        crate::check::compile_pattern(&pattern).map_err(|err| {
            PyValueError::new_err(format!("invalid regular expression {pattern:?}: {err}"))
        })?;
        out.push(Constraint::Regex(pattern));
        return Ok(());
    }
    // Comparison bounds. One marker may carry several (e.g. an interval).
    for (attr, make) in [
        ("ge", Constraint::Ge as fn(OperandIx) -> Constraint),
        ("gt", Constraint::Gt),
        ("le", Constraint::Le),
        ("lt", Constraint::Lt),
    ] {
        if let Ok(bound) = marker.getattr(attr)
            && !bound.is_none()
        {
            out.push(make(lits.intern_operand(&bound)));
        }
    }
    // Length bounds. A bound no length can be compared against -- negative, or
    // past what a container can hold -- is refused rather than dropped: dropping
    // it leaves a schema that admits every value of its base, and the marker was
    // written to admit fewer.
    for (attr, make) in [
        ("min_length", Constraint::MinLen as fn(usize) -> Constraint),
        ("max_length", Constraint::MaxLen),
    ] {
        if let Ok(bound) = marker.getattr(attr)
            && !bound.is_none()
        {
            let n = bound.extract::<usize>().map_err(|_| {
                PyValueError::new_err(format!(
                    "{attr} must be a length a value can have, and {} is not",
                    summarize(&bound)
                ))
            })?;
            out.push(make(n));
        }
    }
    // Numeric multiple-of bound. A zero divisor is rejected here: no value is a
    // multiple of zero, and checking one would divide by zero at validation time,
    // so the schema is unsatisfiable and the error belongs at construction.
    if let Ok(multiple) = marker.getattr("multiple_of")
        && !multiple.is_none()
    {
        if multiple.eq(0).unwrap_or(false) {
            return Err(PyValueError::new_err(
                "MultipleOf(0) is not a valid constraint: no value is a multiple of \
                 zero. Use a nonzero divisor.",
            ));
        }
        out.push(Constraint::MultipleOf(lits.intern_operand(&multiple)));
    }
    // Predicate escape hatch: a callable marker, or `annotated_types.Predicate`,
    // which carries its callable on `.func` and is not callable itself.
    //
    // Callability is how `annotated_types` tells its two marker shapes apart:
    // `Not` defines `__call__` so a consumer calls it, and calling is what
    // applies the negation, while `Predicate` deliberately does not. Reading
    // `.func` from whichever marker has one drops `Not`'s negation and strips a
    // `functools.partial` of its bound arguments — both carry a `.func` too.
    if marker.is_callable() {
        out.push(Constraint::Predicate(lits.intern_predicate(marker)));
    } else if let Ok(func) = marker.getattr("func")
        && func.is_callable()
    {
        out.push(Constraint::Predicate(lits.intern_predicate(&func)));
    } else if out.len() == before && is_unhandled_constraint(marker) {
        return Err(not_implemented(&format!(
            "{} is a constraint this frontend does not check; a schema carrying \
             it would admit the values it excludes, so it is refused rather than \
             ignored",
            summarize(marker)
        )));
    }
    Ok(())
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

fn is_ellipsis(obj: &Bound<'_, PyAny>) -> bool {
    let py = obj.py();
    forms(py).is_ok_and(|forms| obj.is(forms.ellipsis.bind(py)))
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
#[cfg(all(test, feature = "interpreter-tests"))]
mod interpreter {
    use super::*;
    use crate::render::render;
    use rustc_hash::FxHashMap;
    use std::cell::RefCell;
    use std::ffi::CString;

    /// The namespace a row's expression is evaluated in.
    ///
    /// `at` holds the marker doubles, named for the vocabulary they stand in
    /// for so a row reads as the line a caller would write.
    fn namespace(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
        let namespace = PyDict::new(py);
        for module in ["typing", "dataclasses", "enum", "re", "types"] {
            namespace.set_item(module, py.import(module)?)?;
        }
        py.run(
            &CString::new(
                "import types\n\
                 class at:\n\
                 \x20   Ge = staticmethod(lambda n: types.SimpleNamespace(ge=n))\n\
                 \x20   Le = staticmethod(lambda n: types.SimpleNamespace(le=n))\n\
                 \x20   MinLen = staticmethod(\n\
                 \x20       lambda n: types.SimpleNamespace(min_length=n)\n\
                 \x20   )\n\
                 class Timezone:\n\
                 \x20   pass\n\
                 Timezone.__module__ = 'annotated_types'\n",
            )?,
            Some(&namespace),
            None,
        )?;
        Ok(namespace)
    }

    /// Build a schema from an annotation expression and render it back.
    fn built(py: Python<'_>, expression: &str) -> PyResult<String> {
        let namespace = namespace(py)?;
        let annotation = py.eval(&CString::new(expression)?, Some(&namespace), None)?;
        let mut pool = Pool::default();
        let mut defs = Vec::new();
        let schema = build_schema(&annotation, &mut pool, &mut defs)?;
        let active = RefCell::new(FxHashMap::default());
        Ok(render(py, &schema, pool.items(), &defs, &active, 0))
    }

    /// Every spelling the frontend dispatches on, and the schema it must reach.
    ///
    /// One row per arm rather than per feature: a mutant that folds two arms
    /// together is killed by the row of either, and a mutant that drops an arm
    /// is killed by its own.
    #[test]
    fn each_spelling_builds_its_own_schema() {
        Python::attach(|py| {
            for (expression, wanted) in [
                // The scalars and the two bounds, which are the leaves every
                // other row is built out of.
                ("int", "int"),
                ("bool", "bool"),
                ("float", "float"),
                ("str", "str"),
                ("bytes", "bytes"),
                ("None", "None"),
                ("type(None)", "None"),
                ("object", "anything"),
                ("typing.Any", "Any"),
                ("typing.Never", "nothing"),
                // A bare container class is its kind, which is what the typing
                // spec assigns an unparameterised generic.
                ("list", "list[anything]"),
                ("set", "set[anything]"),
                ("frozenset", "frozenset[anything]"),
                // The parameterised forms, one per container arm.
                ("list[int]", "list[int]"),
                ("set[str]", "set[str]"),
                ("frozenset[bytes]", "frozenset[bytes]"),
                ("dict[str, int]", "dict[str, int]"),
                // A tuple is four arms: fixed, homogeneous, prefix-tail, and
                // the unpacked spellings of the last two.
                ("tuple[int, str]", "tuple[int, str]"),
                ("tuple[int, ...]", "tuple[int, ...]"),
                ("tuple[str, *tuple[int, ...]]", "tuple[str, int, ...]"),
                ("tuple[str, *tuple[int, bool]]", "tuple[str, int, bool]"),
                ("tuple[()]", "tuple[()]"),
                // The list-literal spellings, which are this library's own and
                // the only place a prefix and a repeated tail are written
                // without `Unpack`.
                ("[int]", "list[int]"),
                ("[int, str]", "[int, str]"),
                ("[int, ...]", "list[int]"),
                ("[str, int, ...]", "[str, int, ...]"),
                ("[]", "[]"),
                // The connectives, in both spellings where there are two.
                ("int | str", "int | str"),
                ("typing.Union[int, str]", "int | str"),
                ("typing.Optional[int]", "None | int"),
                // A union inside a generic argument, in both spellings: the
                // argument is read by its own path, and each origin is one of
                // the two a union can have.
                ("list[int | str]", "list[int | str]"),
                (
                    "dict[str, typing.Union[int, None]]",
                    "dict[str, None | int]",
                ),
                // A literal is a typed singleton; several are a union of them.
                ("typing.Literal[1]", "Literal[1]"),
                ("typing.Literal['a', 'b']", "Literal['a'] | Literal['b']"),
                // A callable is a class here: what a runtime check can ask of a
                // value is whether it is one, not what it accepts or returns.
                ("typing.Callable[[int], int]", "Callable"),
                // Metadata the frontend does not recognise is ignored, as the
                // typing spec says to -- unless it is a constraint from the
                // vocabulary, which is the refusal below.
                ("typing.Annotated[int, 'a note']", "int"),
                (
                    "typing.Annotated[int, types.SimpleNamespace(multiple_of=3)]",
                    "Annotated[int, MultipleOf(3)]",
                ),
                // A refinement carries its markers on the base it narrows, and
                // a nested one folds onto that base rather than nesting.
                ("typing.Annotated[int, at.Ge(0)]", "Annotated[int, Ge(0)]"),
                (
                    "typing.Annotated[str, at.MinLen(1)]",
                    "Annotated[str, MinLen(1)]",
                ),
                (
                    "typing.Annotated[typing.Annotated[int, at.Ge(0)], at.Le(9)]",
                    "Annotated[int, Ge(0), Le(9)]",
                ),
            ] {
                let got = built(py, expression).unwrap_or_else(|error| {
                    panic!("{expression} did not build: {error}");
                });
                assert_eq!(got, wanted, "{expression}");
            }
        });
    }

    /// The forms the frontend refuses, and the message each refusal carries.
    ///
    /// A refusal is a decision about the algebra -- a construct that names no
    /// set does not silently become one -- so the message is asserted, not only
    /// the failure: a mutant that swaps two refusals leaves both refusing.
    #[test]
    fn each_refusal_says_what_it_refuses() {
        Python::attach(|py| {
            for (expression, wanted) in [
                ("typing.TypeVar('T')", "TypeVar"),
                ("list['Account']", "get_type_hints"),
                ("typing.Annotated[int, at.MinLen(1)]", "length"),
                ("[..., int]", "only as the last element"),
                ("tuple[*list[int]]", "only a tuple can be unpacked"),
                ("typing.Annotated[int, Timezone()]", "does not check"),
            ] {
                let error = match built(py, expression) {
                    Err(error) => error.to_string(),
                    Ok(schema) => panic!("{expression} built {schema} instead of refusing"),
                };
                assert!(
                    error.contains(wanted),
                    "{expression} refused with {error}, which does not name {wanted}"
                );
            }
        });
    }

    /// An alias that names itself is the fixpoint it writes, and one that does
    /// not is the schema its value builds.
    ///
    /// The `type` statement is 3.12 syntax, so the source is run rather than
    /// written here, and the case is skipped where the interpreter this links
    /// cannot parse it -- which is a skip rather than a silent pass because the
    /// assertion below says which it was.
    #[test]
    fn a_self_naming_alias_ties_its_own_fixpoint() {
        Python::attach(|py| {
            if py.version_info() < (3, 12) {
                return;
            }
            let namespace = PyDict::new(py);
            py.run(
                &CString::new(
                    "type Json = int | list[Json]\n\
                     type Plain = int | str\n\
                     type Bad = int | Bad\n",
                )
                .expect("a source with no interior nul"),
                Some(&namespace),
                None,
            )
            .expect("the aliases define");
            let build = |name: &str| {
                let alias = namespace
                    .get_item(name)
                    .expect("the namespace answers")
                    .expect("the alias is in it");
                let mut pool = Pool::default();
                let mut defs = Vec::new();
                build_schema(&alias, &mut pool, &mut defs).map(|schema| (schema, defs))
            };
            // The knot is tied: the alias becomes a definition and the body
            // names it.
            let (schema, defs) = build("Json").expect("a recursive alias builds");
            assert!(
                matches!(schema, Schema::Ref(_)),
                "{schema:?} is no fixpoint"
            );
            assert_eq!(defs.len(), 1, "one alias, one definition");
            // No knot, no definition: an ordinary alias is what its value builds.
            let (schema, defs) = build("Plain").expect("a plain alias builds");
            assert!(matches!(schema, Schema::Union(_)), "{schema:?}");
            assert!(defs.is_empty(), "nothing to define");
            // A self-reference outside a constructor denotes no set, and is
            // refused where it is written.
            let refusal = match build("Bad") {
                Err(refusal) => refusal.to_string(),
                Ok((schema, _)) => panic!("`type Bad = int | Bad` built {schema:?}"),
            };
            assert!(
                refusal.contains("not contractive"),
                "the refusal does not say why: {refusal}"
            );
        });
    }

    /// What a base's values can be asked, per constraint and per base.
    ///
    /// The four `carries_*` predicates decide whether a marker narrows a base or
    /// empties it, and the difference is a refusal at construction rather than a
    /// schema that admits nothing. Asked directly: each is a table over the node
    /// set, and a row through an annotation would exercise one cell of it.
    #[test]
    fn each_base_answers_the_constraints_its_values_can() {
        Python::attach(|py| {
            let seq = Schema::list(SeqShape::homogeneous(Schema::Int));
            let map = Schema::keyed_map(Vec::new(), vec![MapClause::top()]);
            // length: the shaped kinds have one, the scalars do not, and a class
            // does not say.
            for (base, answer) in [
                (Schema::Str, Carries::Yes),
                (Schema::Bytes, Carries::Yes),
                (seq.clone(), Carries::Yes),
                (Schema::set(Schema::Int), Carries::Yes),
                (Schema::frozen_set(Schema::Int), Carries::Yes),
                (map.clone(), Carries::Yes),
                (Schema::Int, Carries::No),
                (Schema::Bool, Carries::No),
                (Schema::Float, Carries::No),
                (Schema::NoneType, Carries::No),
                (Schema::ANY, Carries::Maybe),
            ] {
                assert!(carries_length(&base) == answer, "length of {base:?}");
            }
            // text for a pattern: `str` alone, and `bytes` explicitly not --
            // a pattern here is matched against text.
            for (base, answer) in [
                (Schema::Str, Carries::Yes),
                (Schema::Bytes, Carries::No),
                (Schema::Int, Carries::No),
                (seq.clone(), Carries::No),
                (map.clone(), Carries::No),
                (Schema::ANY, Carries::Maybe),
            ] {
                assert!(carries_pattern(&base) == answer, "pattern on {base:?}");
            }
            // a divisor: the numbers, `bool` among them since it subclasses int.
            for (base, answer) in [
                (Schema::Int, Carries::Yes),
                (Schema::Bool, Carries::Yes),
                (Schema::Float, Carries::Yes),
                (Schema::Str, Carries::No),
                (Schema::Bytes, Carries::No),
                (seq.clone(), Carries::No),
                (map.clone(), Carries::No),
                (Schema::ANY, Carries::Maybe),
            ] {
                assert!(carries_division(&base) == answer, "divisor on {base:?}");
            }
            // order: the operand's group has to be the base's, because Python
            // raises across the groups rather than ordering them.
            let five = 5i64.into_pyobject(py).expect("an int");
            let word = PyString::new(py, "a");
            let raw = PyBytes::new(py, b"a");
            for (base, operand, answer) in [
                (Schema::Int, five.as_any(), Carries::Yes),
                (Schema::Float, five.as_any(), Carries::Yes),
                (Schema::Bool, five.as_any(), Carries::Yes),
                (Schema::Str, five.as_any(), Carries::No),
                (Schema::Bytes, five.as_any(), Carries::No),
                (Schema::Str, word.as_any(), Carries::Yes),
                (Schema::Int, word.as_any(), Carries::No),
                (Schema::Bytes, raw.as_any(), Carries::Yes),
                (Schema::Str, raw.as_any(), Carries::No),
                (seq.clone(), five.as_any(), Carries::Maybe),
            ] {
                assert!(
                    carries_order(&base, operand) == answer,
                    "order of {base:?} against {operand}"
                );
            }
        });
    }

    /// A compound base answers for its parts, and the fold is a union's.
    ///
    /// A member that can answer makes the constraint a narrowing of the union
    /// rather than an emptying of it, which is why the fold is `or` and not
    /// `and`. An intersection and a complement narrow a set this check does not
    /// compute, so they stand aside; a refinement answers with its own base.
    #[test]
    fn a_compound_base_answers_for_its_parts() {
        let refined = Schema::Refine {
            base: Box::new(Schema::Str),
            constraints: vec![Constraint::MinLen(1)],
        };
        for (base, answer) in [
            (Schema::union([Schema::Str, Schema::Int]), Carries::Yes),
            (Schema::union([Schema::Int, Schema::Float]), Carries::No),
            (refined.clone(), Carries::Yes),
            (Schema::meet([Schema::Str, Schema::Int]), Carries::Maybe),
            (Schema::Str.complement(), Carries::Maybe),
        ] {
            assert!(carries_length(&base) == answer, "length of {base:?}");
        }
        // A plain base is not compound, and the fold says so by declining --
        // which is what sends the caller to the table above.
        assert!(carries_through(&Schema::Str, &carries_length).is_none());
    }

    /// A compiled pattern's flags are part of the pattern, and each is written
    /// into it or refused by name.
    ///
    /// Dropping one silently widens the set the marker was written for, so the
    /// two directions are asserted together: the flags this engine spells arrive
    /// inline, and the three it does not are refusals rather than approximations.
    #[test]
    fn a_patterns_flags_are_written_into_it_or_refused() {
        Python::attach(|py| {
            let re = py.import("re").expect("re imports");
            let compiled = |flags: &str| {
                let namespace = PyDict::new(py);
                namespace.set_item("re", &re).expect("a namespace holds it");
                py.eval(
                    &CString::new(format!("re.compile('a', {flags})")).expect("no nul"),
                    Some(&namespace),
                    None,
                )
                .expect("the pattern compiles")
            };
            for (flags, inline) in [
                ("re.I", "(?i)"),
                ("re.M", "(?m)"),
                ("re.S", "(?s)"),
                ("re.X", "(?x)"),
                // Two flags are one group, in the order the engine spells them.
                ("re.I | re.M", "(?im)"),
            ] {
                let pattern = with_inline_flags(&compiled(flags), "a".to_owned())
                    .unwrap_or_else(|error| panic!("{flags} refused: {error}"));
                assert!(
                    pattern.contains(inline),
                    "{flags} gave {pattern}, which does not carry {inline}"
                );
            }
            // The three refused flags are asked of a marker carrying the bit
            // rather than of a compiled pattern: `re` will not compile `re.L`
            // against a `str` at all, and the frontend reads `.flags` off
            // whatever carries it.
            let namespace = PyDict::new(py);
            namespace
                .set_item("types", py.import("types").expect("types imports"))
                .expect("a namespace holds it");
            let marked = |bit: u32| {
                py.eval(
                    &CString::new(format!("types.SimpleNamespace(flags={bit})")).expect("no nul"),
                    Some(&namespace),
                    None,
                )
                .expect("the marker builds")
            };
            for (bit, name) in [(256u32, "re.ASCII"), (4, "re.LOCALE"), (128, "re.DEBUG")] {
                let refusal = match with_inline_flags(&marked(bit), "a".to_owned()) {
                    Err(refusal) => refusal.to_string(),
                    Ok(pattern) => panic!("{name} was written into {pattern}"),
                };
                assert!(
                    refusal.contains(name),
                    "{name} refused without naming itself: {refusal}"
                );
            }
            // A marker with no flags at all is the pattern it carries.
            let bare = PyString::new(py, "a");
            assert_eq!(
                with_inline_flags(&bare, "a".to_owned()).expect("no flags to read"),
                "a"
            );
        });
    }

    /// A constant pools by value where the type is exact, and by identity
    /// everywhere else.
    ///
    /// The two float cases are the ones a reader has to be told: a `nan` is
    /// equal to nothing, so pooling two by value would make one constant of two
    /// values that share no membership; and `-0.0` and `0.0` are equal, so their
    /// keys must agree or one literal would build two nodes.
    #[test]
    fn a_constant_pools_by_value_only_where_equality_is_pythons() {
        Python::attach(|py| {
            let mut pool = Pool::default();
            let eval = |source: &str| {
                py.eval(&CString::new(source).expect("no nul"), None, None)
                    .expect("the expression evaluates")
            };
            // Equal values, two objects: one slot.
            let first = pool.intern_const(&eval("1000000"));
            let again = pool.intern_const(&eval("10 ** 6"));
            assert_eq!(first, again, "two spellings of one int");
            // The signed zeros are one constant, because they are one value.
            let zero = pool.intern_const(&eval("0.0"));
            let minus = pool.intern_const(&eval("-0.0"));
            assert_eq!(zero, minus, "0.0 == -0.0");
            // Two nans are two slots: nothing is equal to a nan, so nothing is
            // pooled with one.
            let nan = pool.intern_const(&eval("float('nan')"));
            let other = pool.intern_const(&eval("float('nan')"));
            assert_ne!(nan, other, "a nan is not equal to a nan");
            // A `bool` is not the `int` it equals, because a literal is typed.
            let one = pool.intern_const(&eval("1"));
            let true_ = pool.intern_const(&eval("True"));
            assert_ne!(one, true_, "Literal[1] is not Literal[True]");
        });
    }

    /// A seeded pool carries the constants it was given, and yields them back.
    #[test]
    fn a_seeded_pool_holds_what_it_was_seeded_with() {
        Python::attach(|py| {
            let held = vec![PyString::new(py, "x").into_any().unbind()];
            let mut pool = Pool::seeded(py, held);
            assert_eq!(pool.items().len(), 1, "the seed is in the pool");
            // A constant equal to the seed joins its slot rather than taking a
            // new one, which is what seeding is for when two validators merge.
            assert_eq!(
                pool.intern_const(&PyString::new(py, "x").into_any()),
                ConstIx::new(0),
                "an equal constant joins the seeded slot"
            );
            assert_eq!(pool.into_items().len(), 1);
        });
    }

    /// A key schema that narrows its keys is refused, at any depth a connective
    /// can hide the narrowing.
    #[test]
    fn a_narrowing_key_is_found_under_every_connective() {
        let narrowing = Schema::Refine {
            base: Box::new(Schema::Str),
            constraints: vec![Constraint::MinLen(1)],
        };
        for schema in [
            narrowing.clone(),
            Schema::union([Schema::Int, narrowing.clone()]),
            Schema::meet([Schema::Str, narrowing.clone()]),
            narrowing.clone().complement(),
            Schema::union([Schema::Int, narrowing.clone().complement()]),
        ] {
            assert!(narrows_its_keys(&schema), "{schema:?} narrows its keys");
        }
        // A refinement carrying no constraint narrows nothing, and neither does
        // a plain key type: both are keys a map may be written with.
        for schema in [
            Schema::Str,
            Schema::union([Schema::Str, Schema::Int]),
            Schema::Refine {
                base: Box::new(Schema::Str),
                constraints: Vec::new(),
            },
        ] {
            assert!(!narrows_its_keys(&schema), "{schema:?} keys as it is");
        }
    }

    /// The frontend descends one level past the construction bound and no
    /// further, so the schema *at* the bound builds and the one past it is
    /// refused by name.
    #[test]
    fn the_build_descends_one_level_past_the_construction_bound() {
        Python::attach(|py| {
            let nested = |depth: usize| {
                let mut annotation = "int".to_owned();
                for _ in 0..depth {
                    annotation = format!("list[{annotation}]");
                }
                annotation
            };
            // The boundary itself, from both sides: a chain reaching the bound
            // builds, and one level more is refused by name. Asserting the pair
            // is what pins the *number* -- either side alone holds only that
            // the guard exists somewhere.
            // Written from the *published* bound rather than from the
            // frontend's own, so the two are held apart: a change to either
            // constant alone moves this boundary and fails here.
            let deepest = crate::validator::MAX_SCHEMA_DEPTH;
            built(py, &nested(deepest)).expect("a chain at the bound builds");
            let refusal = match built(py, &nested(deepest + 1)) {
                Err(refusal) => refusal.to_string(),
                Ok(schema) => panic!("one level past the bound built {schema}"),
            };
            assert!(
                refusal.contains("too deep"),
                "the refusal does not name the depth: {refusal}"
            );
            // And the counter comes back down: a refusal leaves the depth where
            // it found it, so the next build starts from nought rather than
            // from wherever the last one stopped.
            built(py, &nested(deepest)).expect("the guard unwound");
        });
    }

    /// A class whose fields are declared builds the record beside the class,
    /// and one whose are not is the class alone.
    ///
    /// Asserted on the schema rather than on its render, because the render
    /// prints a class by name either way: a dataclass read as a bare
    /// `isinstance` and one read as a record of fields print the same string
    /// and denote different sets.
    #[test]
    fn a_declared_field_becomes_an_attribute_beside_the_class() {
        Python::attach(|py| {
            let namespace = namespace(py).expect("the corpus namespace builds");
            py.run(
                &CString::new(
                    "import dataclasses, typing\n\
                     @dataclasses.dataclass\n\
                     class Point:\n\
                     \x20   x: int\n\
                     class Pair(typing.NamedTuple):\n\
                     \x20   left: int\n\
                     \x20   right: str\n\
                     class Bare(tuple):\n\
                     \x20   pass\n\
                     @typing.runtime_checkable\n\
                     class Sized(typing.Protocol):\n\
                     \x20   def __len__(self) -> int: ...\n\
                     class Quiet(typing.Protocol):\n\
                     \x20   def __len__(self) -> int: ...\n",
                )
                .expect("a source with no interior nul"),
                Some(&namespace),
                None,
            )
            .expect("the corpus classes define");
            let build = |name: &str| {
                let annotation = namespace
                    .get_item(name)
                    .expect("the namespace answers")
                    .expect("the class is in it");
                let mut pool = Pool::default();
                let mut defs = Vec::new();
                build_schema(&annotation, &mut pool, &mut defs)
            };
            let fields = |name: &str| match build(name) {
                Ok(Schema::Intersection(members)) => members
                    .iter()
                    .find_map(|member| match member {
                        Schema::AttrRecord { fields } => Some(
                            fields
                                .iter()
                                .map(|field| field.name.clone())
                                .collect::<Vec<_>>(),
                        ),
                        _ => None,
                    })
                    .unwrap_or_default(),
                _ => Vec::new(),
            };
            // A dataclass and a named tuple carry their fields; a tuple subclass
            // that names none carries none, and is the class alone.
            assert_eq!(fields("Point"), vec!["x".to_owned()]);
            assert_eq!(fields("Pair"), vec!["left".to_owned(), "right".to_owned()]);
            assert!(fields("Bare").is_empty(), "a tuple subclass declares none");
            assert!(
                matches!(build("Bare"), Ok(Schema::Instance(_))),
                "a class with no declared field is the isinstance atom"
            );
            // A protocol is an isinstance check, and only where the class says
            // the check is allowed.
            assert!(matches!(build("Sized"), Ok(Schema::Instance(_))));
            let refusal = match build("Quiet") {
                Err(refusal) => refusal.to_string(),
                Ok(schema) => panic!("a plain Protocol built {schema:?}"),
            };
            assert!(
                refusal.contains("runtime_checkable"),
                "the refusal does not say what is missing: {refusal}"
            );
        });
    }

    /// A `TypedDict` says which keys it admits beyond the ones it declares, and
    /// the two ways it says so are read.
    #[test]
    fn a_typed_dict_says_which_other_keys_it_admits() {
        Python::attach(|py| {
            let namespace = namespace(py).expect("the corpus namespace builds");
            py.run(
                &CString::new(
                    "import typing\n\
                     class Open(typing.TypedDict):\n\
                     \x20   name: str\n\
                     class Closed(typing.TypedDict):\n\
                     \x20   name: str\n\
                     Closed.__closed__ = True\n\
                     class Extra(typing.TypedDict):\n\
                     \x20   name: str\n\
                     Extra.__extra_items__ = int\n",
                )
                .expect("a source with no interior nul"),
                Some(&namespace),
                None,
            )
            .expect("the corpus classes define");
            for (name, wanted) in [
                // Open is the spec's default, and open over the string keys:
                // a `TypedDict` relates to `Mapping[str, object]` and nothing
                // wider.
                ("Open", "{'name': str, str: anything}"),
                // `closed=True` (PEP 728) leaves the declared keys alone.
                ("Closed", "{'name': str}"),
                // `extra_items` types the rest rather than admitting anything.
                ("Extra", "{'name': str, str: int}"),
            ] {
                let annotation = namespace
                    .get_item(name)
                    .expect("the namespace answers")
                    .expect("the class is in it");
                let mut pool = Pool::default();
                let mut defs = Vec::new();
                let schema = build_schema(&annotation, &mut pool, &mut defs)
                    .unwrap_or_else(|error| panic!("{name} did not build: {error}"));
                let active = RefCell::new(FxHashMap::default());
                assert_eq!(
                    render(py, &schema, pool.items(), &defs, &active, 0),
                    wanted,
                    "{name}"
                );
            }
        });
    }

    /// A class is read through its declared fields, and what it declares is not
    /// every annotation on it.
    #[test]
    fn a_class_is_read_through_what_it_declares() {
        Python::attach(|py| {
            let namespace = namespace(py).expect("the corpus namespace builds");
            py.run(
                &CString::new(
                    "import dataclasses, typing\n\
                     @dataclasses.dataclass\n\
                     class Point:\n\
                     \x20   x: int\n\
                     \x20   tag: typing.ClassVar[str] = 'p'\n\
                     class Row(typing.TypedDict):\n\
                     \x20   name: str\n\
                     \x20   note: typing.NotRequired[int]\n",
                )
                .expect("a source with no interior nul"),
                Some(&namespace),
                None,
            )
            .expect("the corpus classes define");
            for (name, wanted) in [
                // The `ClassVar` annotates the class rather than an instance, so
                // it is not a field of the record the instances hold.
                ("Point", "Point"),
                // A `TypedDict` is a record, open as the typing spec defines
                // one, and `NotRequired` marks the key rather than its type.
                ("Row", "{'name': str, 'note?': int, str: anything}"),
            ] {
                let annotation = namespace
                    .get_item(name)
                    .expect("the namespace answers")
                    .expect("the class is in it");
                let mut pool = Pool::default();
                let mut defs = Vec::new();
                let schema = build_schema(&annotation, &mut pool, &mut defs)
                    .unwrap_or_else(|error| panic!("{name} did not build: {error}"));
                let active = RefCell::new(FxHashMap::default());
                assert_eq!(
                    render(py, &schema, pool.items(), &defs, &active, 0),
                    wanted,
                    "{name}"
                );
            }
        });
    }
}

#[cfg(all(test, feature = "interpreter-tests"))]
mod tests {
    use super::*;

    #[test]
    fn intern_deduplicates_by_identity() {
        Python::attach(|py| {
            let mut pool = Pool::default();
            let a = PyString::new(py, "x").into_any();

            // The same object interns to one slot.
            let first = pool.intern(&a);
            let again = pool.intern(&a);
            assert_eq!(first, again);
            assert_eq!(pool.items().len(), 1);

            // A distinct object takes a new slot.
            let b = PyList::empty(py).into_any();
            let second = pool.intern(&b);
            assert_ne!(first, second);
            assert_eq!(pool.items().len(), 2);

            // Dedup is by identity, not value: a fresh equal-but-distinct object
            // gets its own slot rather than collapsing onto the first.
            let c = PyList::empty(py).into_any();
            let third = pool.intern(&c);
            assert_ne!(second, third);
            assert_eq!(pool.items().len(), 3);
        });
    }

    /// A definition block is placed once: a block already present at some offset
    /// is reused at that offset, shifted references included, and a new one is
    /// appended at the end.
    #[test]
    fn a_definition_block_is_placed_once() {
        let body = |index: usize| {
            Schema::union([
                Schema::Int,
                Schema::list(SeqShape::homogeneous(Schema::Ref(DefIx::new(index)))),
            ])
        };
        let mut defs = Vec::new();

        // Into an empty list: offset zero.
        assert_eq!(place_definitions(&[Schema::Str], &[], &mut defs), 0);
        assert_eq!(defs, vec![Schema::Str]);

        // A block whose body names itself is shifted to the offset it lands at.
        assert_eq!(place_definitions(&[body(0)], &[], &mut defs), 1);
        assert_eq!(defs, vec![Schema::Str, body(1)]);

        // The same block again is found where it already is, and nothing grows.
        assert_eq!(place_definitions(&[body(0)], &[], &mut defs), 1);
        assert_eq!(defs.len(), 2);

        // A block that matches nowhere is appended after the last one.
        assert_eq!(place_definitions(&[Schema::Bytes], &[], &mut defs), 2);
        assert_eq!(defs, vec![Schema::Str, body(1), Schema::Bytes]);
    }
}
