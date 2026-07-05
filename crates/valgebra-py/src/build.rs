//! The schema frontend: build the IR from Python types, typing annotations,
//! native container forms, and already-compiled validators.

use std::cell::Cell;

use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{
    PyBool, PyBytes, PyDict, PyFloat, PyFrozenSet, PyInt, PyList, PyModule, PySet, PyString,
    PyTuple, PyType,
};
use valgebra_core::{Constraint, Field, Schema, SeqRegex};

use crate::Validator;
use crate::errors::summarize;

thread_local! {
    /// Depth of the current `build_schema` recursion, bounding it so a
    /// self-referential class (whose field type names the class) fails cleanly
    /// instead of recursing until the native stack overflows.
    static BUILD_DEPTH: Cell<usize> = const { Cell::new(0) };
}

/// The most levels of schema nesting allowed while compiling. A real schema is
/// nowhere near this deep; the bound exists so a recursive class — which must be
/// expressed with `recursive` — is rejected with a message instead of a crash.
const MAX_BUILD_DEPTH: usize = 100;

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
                "schema nesting is too deep to compile; a class whose own type \
                 appears in its fields is recursive and must be written with \
                 recursive(...), which ties the fixpoint explicitly",
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
        return Ok(Schema::Dynamic);
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

    // PEP 695 `type X = ...` alias (3.12+): validate the aliased type.
    if let Some(alias_type) = &forms.type_alias_type
        && obj.is_instance(alias_type.bind(py))?
    {
        return build_schema(&obj.getattr("__value__")?, lits, defs);
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
        let def_offset = defs.len();
        let lit_map: Vec<usize> = inner
            .literals
            .iter()
            .map(|o| lits.intern(o.bind(py)))
            .collect();
        defs.extend(
            inner
                .definitions
                .iter()
                .map(|d| d.reindexed(&lit_map, def_offset)),
        );
        return Ok(inner.schema.reindexed(&lit_map, def_offset));
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

    Ok(Schema::Literal(lits.intern(obj)))
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
    make: impl FnOnce(Vec<Schema>) -> Schema,
) -> PyResult<Validator> {
    let mut literals = Pool::default();
    let mut definitions = Vec::new();
    let mut members = Vec::with_capacity(args.len());
    for arg in args.iter() {
        members.push(build_schema(&arg, &mut literals, &mut definitions)?);
    }
    Validator::composed(make(members), literals.into_items(), definitions)
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
    let forms = forms(py)?;
    if ty.is(forms.object.bind(py)) {
        return Ok(Schema::Anything);
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
        return Ok(Schema::Instance(lits.intern(ty.as_any())));
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
            return Ok(Schema::Instance(lits.intern(ty.as_any())));
        }
        return Err(not_implemented(
            "a Protocol must be @runtime_checkable to be used as a schema",
        ));
    }
    // Any other class names its instances: a bare class is an isinstance check.
    // This covers the remaining builtins (complex, bytearray, memoryview, range,
    // the `collections.abc` ABCs including Callable, ...) and arbitrary user
    // classes uniformly.
    Ok(Schema::Instance(lits.intern(ty.as_any())))
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

/// Build a closed record from a `TypedDict`, reading its required keys.
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
    Ok(Schema::record(fields, false))
}

/// Build an Object node (isinstance plus per-attribute checks) for a class
/// whose fields come from its resolved type hints; all fields are required.
fn build_object(
    ty: &Bound<'_, PyType>,
    lits: &mut Pool,
    defs: &mut Vec<Schema>,
) -> PyResult<Schema> {
    let hints = resolve_type_hints(ty)?;
    let hints = hints.cast::<PyDict>()?;
    let class_index = lits.intern(ty.as_any());
    let mut fields = Vec::with_capacity(hints.len());
    for (name, hint) in hints.iter() {
        fields.push(Field {
            name: field_name(&name.str()?)?,
            schema: build_schema(&hint, lits, defs)?,
            required: true,
        });
    }
    Ok(Schema::Attrs {
        class_index,
        fields,
    })
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
        return Ok(Schema::list(SeqRegex::homogeneous(build_schema(
            &single_arg(args)?,
            lits,
            defs,
        )?)));
    }
    if origin.is(py.get_type::<PySet>()) {
        return Ok(Schema::Set(Box::new(build_schema(
            &single_arg(args)?,
            lits,
            defs,
        )?)));
    }
    if origin.is(py.get_type::<PyFrozenSet>()) {
        return Ok(Schema::FrozenSet(Box::new(build_schema(
            &single_arg(args)?,
            lits,
            defs,
        )?)));
    }
    if origin.is(py.get_type::<PyDict>()) {
        if args.len() != 2 {
            return Err(not_implemented(
                "dict[...] needs a key type and a value type",
            ));
        }
        return Ok(Schema::mapping(
            build_schema(&args.get_item(0)?, lits, defs)?,
            build_schema(&args.get_item(1)?, lits, defs)?,
        ));
    }
    if origin.is(py.get_type::<PyTuple>()) {
        // tuple[A, ..., Z, ...]: a trailing `...` repeats the element before it
        // after a fixed prefix, mirroring the list form; tuple[T, ...] is the
        // prefix-free homogeneous case. Other shapes are a fixed-length tuple.
        let len = args.len();
        if len >= 2 && is_ellipsis(&args.get_item(len - 1)?) {
            let mut elements = Vec::with_capacity(len - 1);
            for index in 0..len - 1 {
                let item = args.get_item(index)?;
                if is_ellipsis(&item) {
                    return Err(not_implemented(
                        "`...` may appear only as the last element of a tuple schema",
                    ));
                }
                elements.push(build_schema(&item, lits, defs)?);
            }
            let tail = elements.pop().expect("at least one element precedes `...`");
            let regex = if elements.is_empty() {
                SeqRegex::homogeneous(tail)
            } else {
                SeqRegex::prefix_tail(elements, tail)
            };
            return Ok(Schema::tuple(regex));
        }
        let mut elements = Vec::with_capacity(len);
        for arg in args.iter() {
            if is_ellipsis(&arg) {
                return Err(not_implemented(
                    "`...` may appear only as the last element of a tuple schema",
                ));
            }
            elements.push(build_schema(&arg, lits, defs)?);
        }
        return Ok(Schema::tuple(SeqRegex::fixed(elements)));
    }
    if is_required_marker(origin)? {
        // Required[T]/NotRequired[T] only annotate a TypedDict field's
        // requiredness, which is already read from __required_keys__; validate
        // the wrapped type. These wrappers survive hint resolution because field
        // metadata is kept (include_extras), so the frontend must unwrap them.
        return build_schema(&single_arg(args)?, lits, defs);
    }
    if is_union_origin(origin)? {
        let mut members = Vec::with_capacity(args.len());
        for arg in args.iter() {
            members.push(build_schema(&arg, lits, defs)?);
        }
        return Ok(Schema::Union(members));
    }
    if is_literal_origin(origin)? {
        // Literal args are constant values; each becomes a literal, unioned when
        // there is more than one.
        let mut members = Vec::with_capacity(args.len());
        for arg in args.iter() {
            members.push(build_schema(&arg, lits, defs)?);
        }
        return Ok(if members.len() == 1 {
            members.into_iter().next().expect("one member")
        } else {
            Schema::Union(members)
        });
    }
    if origin.is(forms(py)?.callable.bind(py)) {
        // Callable[...] checks only callability at runtime; the argument and
        // return types cannot be inspected, so the parameters are ignored and
        // the schema is the opaque `isinstance(x, Callable)` test.
        return Ok(Schema::Instance(lits.intern(origin)));
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

/// True if `origin` is `typing.Required` or `typing.NotRequired`, the
/// `TypedDict` field-requiredness markers (kept in field hints by
/// `include_extras`).
fn is_required_marker(origin: &Bound<'_, PyAny>) -> PyResult<bool> {
    let typing = origin.py().import("typing")?;
    for name in ["Required", "NotRequired"] {
        if let Ok(marker) = typing.getattr(name)
            && origin.is(&marker)
        {
            return Ok(true);
        }
    }
    Ok(false)
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
        return Ok(Schema::list(SeqRegex::homogeneous(element)));
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
            SeqRegex::homogeneous(tail)
        } else {
            SeqRegex::prefix_tail(elements, tail)
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
    Ok(Schema::list(SeqRegex::fixed(elements)))
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
            defaults.push((
                build_schema(&key, lits, defs)?,
                build_schema(&value, lits, defs)?,
            ));
        }
    }
    Ok(Schema::KeyedMap { fields, defaults })
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
    if constraints.is_empty() {
        Ok(base_schema)
    } else {
        Ok(Schema::Refine {
            base: Box::new(base_schema),
            constraints,
        })
    }
}

fn parse_constraint(
    marker: &Bound<'_, PyAny>,
    out: &mut Vec<Constraint>,
    lits: &mut Pool,
) -> PyResult<()> {
    // A string-pattern marker: valgebra's `Regex(...)` or a compiled
    // `re.Pattern`, both carrying the source pattern as `.pattern`. The pattern
    // is validated (anchored) here so an invalid expression fails at compile
    // time, not at first validation; the compiled regex is cached per validator.
    if let Ok(attr) = marker.getattr("pattern")
        && let Ok(pattern) = attr.extract::<String>()
    {
        crate::check::compile_pattern(&pattern).map_err(|err| {
            PyValueError::new_err(format!("invalid regular expression {pattern:?}: {err}"))
        })?;
        out.push(Constraint::Regex(pattern));
        return Ok(());
    }
    // Comparison bounds. One marker may carry several (e.g. an interval).
    for (attr, make) in [
        ("ge", Constraint::Ge as fn(usize) -> Constraint),
        ("gt", Constraint::Gt),
        ("le", Constraint::Le),
        ("lt", Constraint::Lt),
    ] {
        if let Ok(bound) = marker.getattr(attr)
            && !bound.is_none()
        {
            out.push(make(lits.intern(&bound)));
        }
    }
    // Length bounds.
    if let Ok(min) = marker.getattr("min_length")
        && let Ok(n) = min.extract::<usize>()
    {
        out.push(Constraint::MinLen(n));
    }
    if let Ok(max) = marker.getattr("max_length")
        && !max.is_none()
        && let Ok(n) = max.extract::<usize>()
    {
        out.push(Constraint::MaxLen(n));
    }
    // Numeric multiple-of bound.
    if let Ok(multiple) = marker.getattr("multiple_of")
        && !multiple.is_none()
    {
        out.push(Constraint::MultipleOf(lits.intern(&multiple)));
    }
    // Predicate escape hatch: annotated_types.Predicate(.func) or a bare
    // callable used directly as metadata.
    if let Ok(func) = marker.getattr("func")
        && func.is_callable()
    {
        out.push(Constraint::Predicate(lits.intern(&func)));
    } else if marker.is_callable() {
        out.push(Constraint::Predicate(lits.intern(marker)));
    }
    Ok(())
}

/// The constants pool a compile builds, plus an identity index into it. Pooling
/// deduplicates by object identity; the index makes each `intern` a hash lookup
/// rather than a linear scan, so compiling a wide `Literal[...]` or merging many
/// validators stays linear in the number of constants instead of quadratic.
///
/// The address key is stable for the pooled objects: every interned value is
/// kept alive by `items`, so no live pool entry is freed and reallocated during a
/// compile.
#[derive(Default)]
pub(crate) struct Pool {
    items: Vec<Py<PyAny>>,
    index: rustc_hash::FxHashMap<usize, usize>,
}

impl Pool {
    /// Seed a pool with an existing validator's constants, rebuilding the identity
    /// index, so a second schema interns into the same pool when validators merge.
    pub(crate) fn seeded(items: Vec<Py<PyAny>>) -> Self {
        let index = items
            .iter()
            .enumerate()
            .map(|(i, obj)| (obj.as_ptr() as usize, i))
            .collect();
        Pool { items, index }
    }

    /// Pool `obj` and return its index, deduplicating by object identity.
    fn intern(&mut self, obj: &Bound<'_, PyAny>) -> usize {
        let ptr = obj.as_ptr() as usize;
        if let Some(&index) = self.index.get(&ptr) {
            return index;
        }
        let index = self.items.len();
        self.items.push(obj.clone().unbind());
        self.index.insert(ptr, index);
        index
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

pub(crate) fn not_implemented(message: &str) -> PyErr {
    PyNotImplementedError::new_err(message.to_owned())
}

// Needs a live interpreter; compiled and run only under the `interpreter-tests`
// feature, which links an embedded Python.
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
}
