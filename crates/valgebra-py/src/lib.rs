//! `PyO3` bindings for valgebra: compile a Python schema once into the core IR
//! and walk it entirely in Rust, crossing the boundary once per call.
//!
//! The frontend ([`build_schema`]) reads native Python forms into the IR; the
//! walk ([`check`]) tests membership of a concrete value, reporting the first
//! failure as a structured [`ValidationError`].

use std::cell::RefCell;
use std::collections::HashSet;

use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{
    PyBool, PyBytes, PyDict, PyFloat, PyFrozenSet, PyInt, PyList, PySet, PyString, PyTuple, PyType,
};
use valgebra_core::{Constraint, Field, PathSegment, Schema, Violation, fresh_self_token};

create_exception!(
    _valgebra,
    ValidationError,
    PyException,
    "Raised when a value is not a member of a schema's set."
);

/// An immutable compiled validator: a schema IR, the constants pool its
/// literals index into, and the recursive definitions its `Ref`s resolve
/// against. Building one compiles the schema; the validator itself never
/// mutates.
#[pyclass(frozen, module = "valgebra._valgebra")]
pub struct CompiledValidator {
    schema: Schema,
    literals: Vec<Py<PyAny>>,
    definitions: Vec<Schema>,
}

impl CompiledValidator {
    /// The read-only walk context: the pool, the definitions, and a fresh
    /// recursion guard.
    fn context<'a>(&'a self, guard: &'a RefCell<HashSet<(usize, usize)>>) -> Ctx<'a> {
        Ctx {
            pool: &self.literals,
            defs: &self.definitions,
            guard,
        }
    }
}

#[pymethods]
impl CompiledValidator {
    /// Raise [`ValidationError`] if `obj` is not a member of the schema's set;
    /// return `None` otherwise. Check-only: the object is not copied or coerced.
    fn validate(&self, obj: &Bound<'_, PyAny>) -> PyResult<()> {
        let guard = RefCell::new(HashSet::new());
        let mut path = Vec::new();
        match check(&self.schema, obj, &mut path, self.context(&guard)) {
            Some(violation) => Err(into_pyerr(obj.py(), &violation)),
            None => Ok(()),
        }
    }

    /// Whether `obj` belongs to the schema's set. Check-only, returns a bool.
    fn is_valid(&self, obj: &Bound<'_, PyAny>) -> bool {
        let guard = RefCell::new(HashSet::new());
        let mut path = Vec::new();
        check(&self.schema, obj, &mut path, self.context(&guard)).is_none()
    }

    /// Validate `obj` and return it unchanged. The explicit conversion mode:
    /// validation is a membership check, so the returned object is the input.
    fn cast<'py>(&self, obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.validate(obj)?;
        Ok(obj.clone())
    }

    /// Render the compiled schema back as the annotation expression that
    /// produces it.
    fn __repr__(&self, py: Python<'_>) -> String {
        let active = RefCell::new(HashSet::new());
        render(py, &self.schema, &self.literals, &self.definitions, &active)
    }
}

/// The read-only context threaded through a validation walk: the constants
/// pool, the recursion definitions, and the active recursion guard. The guard
/// records `(object id, definition index)` pairs currently on the path so a
/// value that contains itself fails with `recursion_loop` instead of looping.
#[derive(Clone, Copy)]
struct Ctx<'a> {
    pool: &'a [Py<PyAny>],
    defs: &'a [Schema],
    guard: &'a RefCell<HashSet<(usize, usize)>>,
}

/// Compile `schema` into an immutable [`CompiledValidator`].
#[pyfunction]
fn validator(schema: &Bound<'_, PyAny>) -> PyResult<CompiledValidator> {
    let mut literals = Vec::new();
    let mut definitions = Vec::new();
    let schema = build_schema(schema, &mut literals, &mut definitions)?;
    Ok(CompiledValidator {
        schema,
        literals,
        definitions,
    })
}

/// Build a recursive schema as a checked fixpoint.
///
/// `builder` receives a placeholder validator standing for the schema being
/// defined and returns its body. The placeholder's self-reference is resolved
/// to a back edge, and a non-contractive body — one whose recursive reference
/// is not under a structural constructor — is rejected.
#[pyfunction]
fn lazy(builder: &Bound<'_, PyAny>) -> PyResult<CompiledValidator> {
    let py = builder.py();
    let token = fresh_self_token();
    let placeholder = Py::new(
        py,
        CompiledValidator {
            schema: Schema::SelfRef(token),
            literals: Vec::new(),
            definitions: Vec::new(),
        },
    )?;
    let body_obj = builder.call1((placeholder,))?;
    let mut literals = Vec::new();
    let mut definitions = Vec::new();
    let body = build_schema(&body_obj, &mut literals, &mut definitions)?;
    // The body becomes a definition; the self-reference resolves to it.
    let ref_id = definitions.len();
    let resolved = body.resolve_self(token, ref_id);
    if resolved.occurs_unguarded(ref_id, false) {
        return Err(PyValueError::new_err(
            "lazy schema is not contractive: the recursive reference must occur \
             under a structural constructor (a list, tuple, set, dict, record, \
             or object)",
        ));
    }
    definitions.push(resolved);
    Ok(CompiledValidator {
        schema: Schema::Ref(ref_id),
        literals,
        definitions,
    })
}

/// The union of the given schemas: a value in at least one of their sets.
#[pyfunction]
#[pyo3(signature = (*schemas))]
fn union(schemas: &Bound<'_, PyTuple>) -> PyResult<CompiledValidator> {
    combine(schemas, Schema::Union)
}

/// The intersection of the given schemas: a value in every one of their sets.
#[pyfunction]
#[pyo3(signature = (*schemas))]
fn intersect(schemas: &Bound<'_, PyTuple>) -> PyResult<CompiledValidator> {
    combine(schemas, Schema::Intersection)
}

/// The complement of a schema: every value not in its set.
#[pyfunction]
fn complement(schema: &Bound<'_, PyAny>) -> PyResult<CompiledValidator> {
    let mut literals = Vec::new();
    let mut definitions = Vec::new();
    let inner = build_schema(schema, &mut literals, &mut definitions)?;
    Ok(CompiledValidator {
        schema: Schema::Complement(Box::new(inner)),
        literals,
        definitions,
    })
}

/// An equivalent validator reduced by the lattice laws: it admits exactly the
/// same values, in a simpler form. The pool and definitions are shared
/// unchanged, as simplification only rewrites the schema's structure.
#[pyfunction]
fn simplify(validator: &CompiledValidator, py: Python<'_>) -> CompiledValidator {
    CompiledValidator {
        schema: validator.schema.simplify(),
        literals: validator.literals.iter().map(|o| o.clone_ref(py)).collect(),
        definitions: validator.definitions.clone(),
    }
}

/// A pool-free validator wrapping a single atom (the `anything`/`nothing`
/// lattice bounds).
fn atom(py: Python<'_>, schema: Schema) -> PyResult<Py<CompiledValidator>> {
    Py::new(
        py,
        CompiledValidator {
            schema,
            literals: Vec::new(),
            definitions: Vec::new(),
        },
    )
}

/// The `valgebra._valgebra` extension module.
#[pymodule]
fn _valgebra(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    module.add("ValidationError", py.get_type::<ValidationError>())?;
    module.add_class::<CompiledValidator>()?;
    module.add_function(wrap_pyfunction!(validator, module)?)?;
    module.add_function(wrap_pyfunction!(union, module)?)?;
    module.add_function(wrap_pyfunction!(intersect, module)?)?;
    module.add_function(wrap_pyfunction!(complement, module)?)?;
    module.add_function(wrap_pyfunction!(simplify, module)?)?;
    module.add_function(wrap_pyfunction!(lazy, module)?)?;
    // The lattice bounds: top admits every value, bottom admits none.
    module.add("anything", atom(py, Schema::Anything)?)?;
    module.add("nothing", atom(py, Schema::Nothing)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Frontend: native Python forms into the IR.
// ---------------------------------------------------------------------------

/// Build the IR from a native Python schema description.
///
/// Recognized forms: the scalar types and `None`/`type(None)`; `object` as the
/// top schema; `[T]`/`[T, ...]` as a list; `(A, B, ...)` as a fixed tuple;
/// `{T}` as a set; a single `{KeyType: ValueType}` entry as a mapping; an
/// all-string-key dict as a closed record (a trailing `"?"` marks an optional
/// key); any other value as an exact-value literal.
fn build_schema(
    obj: &Bound<'_, PyAny>,
    lits: &mut Vec<Py<PyAny>>,
    defs: &mut Vec<Schema>,
) -> PyResult<Schema> {
    let py = obj.py();
    if obj.is_none() {
        return Ok(Schema::NoneType);
    }

    let typing = py.import("typing")?;

    // `typing.Any` is a singleton special form: the gradual dynamic type.
    if obj.is(&typing.getattr("Any")?) {
        return Ok(Schema::Any);
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
    if let Ok(alias_type) = typing.getattr("TypeAliasType")
        && obj.is_instance(&alias_type)?
    {
        return build_schema(&obj.getattr("__value__")?, lits, defs);
    }

    // NewType: validate the supertype it wraps.
    if obj.hasattr("__supertype__")? {
        return build_schema(&obj.getattr("__supertype__")?, lits, defs);
    }

    if let Ok(ty) = obj.cast::<PyType>() {
        return build_type_object(ty, lits, defs);
    }
    if let Ok(list) = obj.cast::<PyList>() {
        return build_sequence(list, lits, defs);
    }
    if let Ok(tuple) = obj.cast::<PyTuple>() {
        let mut elements = Vec::with_capacity(tuple.len());
        for item in tuple.iter() {
            elements.push(build_schema(&item, lits, defs)?);
        }
        return Ok(Schema::Tuple(elements));
    }
    if let Ok(set) = obj.cast::<PySet>() {
        return build_set(set, lits, defs);
    }
    if let Ok(dict) = obj.cast::<PyDict>() {
        return build_dict(dict, lits, defs);
    }

    // An already-compiled validator composes in: append its pool and
    // definitions and shift its schema's indices past what is already there.
    if let Ok(compiled) = obj.cast::<CompiledValidator>() {
        let inner = compiled.get();
        let pool_offset = lits.len();
        let def_offset = defs.len();
        lits.extend(inner.literals.iter().map(|o| o.clone_ref(py)));
        defs.extend(
            inner
                .definitions
                .iter()
                .map(|d| d.shifted(pool_offset, def_offset)),
        );
        return Ok(inner.schema.shifted(pool_offset, def_offset));
    }

    Ok(Schema::Literal(intern(lits, obj)))
}

/// Compile arguments into one shared pool and combine them with `make`.
fn combine(
    args: &Bound<'_, PyTuple>,
    make: impl FnOnce(Vec<Schema>) -> Schema,
) -> PyResult<CompiledValidator> {
    let mut literals = Vec::new();
    let mut definitions = Vec::new();
    let mut members = Vec::with_capacity(args.len());
    for arg in args.iter() {
        members.push(build_schema(&arg, &mut literals, &mut definitions)?);
    }
    Ok(CompiledValidator {
        schema: make(members),
        literals,
        definitions,
    })
}

/// Build the schema for a Python type object (a builtin, `TypedDict`, `Enum`,
/// dataclass, `NamedTuple`, runtime-checkable `Protocol`, or `object`).
fn build_type_object(
    ty: &Bound<'_, PyType>,
    lits: &mut Vec<Py<PyAny>>,
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
    if ty.is(&py.import("builtins")?.getattr("object")?) {
        return Ok(Schema::Anything);
    }
    // TypedDict: a closed record whose required keys come from the class.
    if ty.hasattr("__required_keys__")? {
        return build_typed_dict(ty, lits, defs);
    }
    // Enum: an instance of the enumeration class (any of its members).
    if ty.is_subclass(&py.import("enum")?.getattr("Enum")?)? {
        return Ok(Schema::Instance(intern(lits, ty.as_any())));
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
            return Ok(Schema::Instance(intern(lits, ty.as_any())));
        }
        return Err(not_implemented(
            "a Protocol must be @runtime_checkable to be used as a schema",
        ));
    }
    Err(not_implemented(&format!(
        "unsupported type schema: {}",
        summarize(ty.as_any())
    )))
}

/// True if `obj.<name>` exists and is truthy; false on absence or error.
fn is_truthy_attr(obj: &Bound<'_, PyAny>, name: &str) -> bool {
    obj.getattr(name)
        .ok()
        .and_then(|value| value.is_truthy().ok())
        .unwrap_or(false)
}

/// Build a closed record from a `TypedDict`, reading its required keys.
fn build_typed_dict(
    ty: &Bound<'_, PyType>,
    lits: &mut Vec<Py<PyAny>>,
    defs: &mut Vec<Schema>,
) -> PyResult<Schema> {
    let py = ty.py();
    let hints = py.import("typing")?.call_method1("get_type_hints", (ty,))?;
    let hints = hints.cast::<PyDict>()?;
    let required = ty.getattr("__required_keys__")?;
    let mut fields = Vec::with_capacity(hints.len());
    for (name, hint) in hints.iter() {
        fields.push(Field {
            name: name.str()?.to_string(),
            schema: build_schema(&hint, lits, defs)?,
            required: required.contains(&name)?,
        });
    }
    Ok(Schema::Record { fields })
}

/// Build an Object node (isinstance plus per-attribute checks) for a class
/// whose fields come from its resolved type hints; all fields are required.
fn build_object(
    ty: &Bound<'_, PyType>,
    lits: &mut Vec<Py<PyAny>>,
    defs: &mut Vec<Schema>,
) -> PyResult<Schema> {
    let py = ty.py();
    let hints = py.import("typing")?.call_method1("get_type_hints", (ty,))?;
    let hints = hints.cast::<PyDict>()?;
    let class_index = intern(lits, ty.as_any());
    let mut fields = Vec::with_capacity(hints.len());
    for (name, hint) in hints.iter() {
        fields.push(Field {
            name: name.str()?.to_string(),
            schema: build_schema(&hint, lits, defs)?,
            required: true,
        });
    }
    Ok(Schema::Object {
        class_index,
        fields,
    })
}

/// Build the IR for a parametrized typing generic given its origin and args.
fn build_parametrized(
    origin: &Bound<'_, PyAny>,
    args: &Bound<'_, PyTuple>,
    lits: &mut Vec<Py<PyAny>>,
    defs: &mut Vec<Schema>,
) -> PyResult<Schema> {
    let py = origin.py();
    if origin.is(py.get_type::<PyList>()) {
        return Ok(Schema::Sequence(Box::new(build_schema(
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
        return Ok(Schema::Mapping {
            key: Box::new(build_schema(&args.get_item(0)?, lits, defs)?),
            value: Box::new(build_schema(&args.get_item(1)?, lits, defs)?),
        });
    }
    if origin.is(py.get_type::<PyTuple>()) {
        // tuple[T, ...] is the homogeneous variadic form.
        if args.len() == 2 && is_ellipsis(&args.get_item(1)?) {
            return Ok(Schema::VariadicTuple(Box::new(build_schema(
                &args.get_item(0)?,
                lits,
                defs,
            )?)));
        }
        let mut elements = Vec::with_capacity(args.len());
        for arg in args.iter() {
            if is_ellipsis(&arg) {
                return Err(not_implemented(
                    "tuple[...] supports a fixed shape or the homogeneous \
                     tuple[T, ...]; other uses of ... are not supported",
                ));
            }
            elements.push(build_schema(&arg, lits, defs)?);
        }
        return Ok(Schema::Tuple(elements));
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
    Err(not_implemented(&format!(
        "unsupported typing form with origin {}; supported: list, set, dict, \
         tuple, Union, Optional, Literal",
        summarize(origin)
    )))
}

/// True if `origin` is `typing.Union` (from Union/Optional) or
/// `types.UnionType` (from `X | Y`).
fn is_union_origin(origin: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = origin.py();
    let typing_union = py.import("typing")?.getattr("Union")?;
    let pep604_union = py.import("types")?.getattr("UnionType")?;
    Ok(origin.is(&typing_union) || origin.is(&pep604_union))
}

/// True if `origin` is `typing.Literal`.
fn is_literal_origin(origin: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(origin.is(&origin.py().import("typing")?.getattr("Literal")?))
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
    lits: &mut Vec<Py<PyAny>>,
    defs: &mut Vec<Schema>,
) -> PyResult<Schema> {
    match list.len() {
        1 => Ok(Schema::Sequence(Box::new(build_schema(
            &list.get_item(0)?,
            lits,
            defs,
        )?))),
        2 if is_ellipsis(&list.get_item(1)?) => Ok(Schema::Sequence(Box::new(build_schema(
            &list.get_item(0)?,
            lits,
            defs,
        )?))),
        _ => Err(not_implemented(
            "a list schema must be [T] or [T, ...]; other list shapes are not supported",
        )),
    }
}

fn build_set(
    set: &Bound<'_, PySet>,
    lits: &mut Vec<Py<PyAny>>,
    defs: &mut Vec<Schema>,
) -> PyResult<Schema> {
    if set.len() == 1
        && let Some(element) = set.iter().next()
    {
        return Ok(Schema::Set(Box::new(build_schema(&element, lits, defs)?)));
    }
    Err(not_implemented(
        "a set schema must have exactly one element, as in {T}",
    ))
}

fn build_dict(
    dict: &Bound<'_, PyDict>,
    lits: &mut Vec<Py<PyAny>>,
    defs: &mut Vec<Schema>,
) -> PyResult<Schema> {
    // An empty dict is the empty closed record: it matches only {}.
    if dict.is_empty() {
        return Ok(Schema::Record { fields: Vec::new() });
    }
    // All-string keys: a record. A single type-keyed entry: a mapping.
    if dict.iter().all(|(key, _)| key.is_instance_of::<PyString>()) {
        return build_record(dict, lits, defs);
    }
    if dict.len() == 1
        && let Some((key, value)) = dict.iter().next()
        && key.cast::<PyType>().is_ok()
    {
        return Ok(Schema::Mapping {
            key: Box::new(build_schema(&key, lits, defs)?),
            value: Box::new(build_schema(&value, lits, defs)?),
        });
    }
    Err(not_implemented(
        "a dict schema must use all string keys (a record) or a single \
         {KeyType: ValueType} entry (a mapping)",
    ))
}

fn build_record(
    dict: &Bound<'_, PyDict>,
    lits: &mut Vec<Py<PyAny>>,
    defs: &mut Vec<Schema>,
) -> PyResult<Schema> {
    let mut fields = Vec::with_capacity(dict.len());
    for (key, value) in dict.iter() {
        let raw = key.str()?.to_string();
        let (name, required) = match raw.strip_suffix('?') {
            Some(stripped) => (stripped.to_owned(), false),
            None => (raw, true),
        };
        fields.push(Field {
            name,
            schema: build_schema(&value, lits, defs)?,
            required,
        });
    }
    Ok(Schema::Record { fields })
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
    lits: &mut Vec<Py<PyAny>>,
    defs: &mut Vec<Schema>,
) -> PyResult<Schema> {
    let base_schema = build_schema(base, lits, defs)?;
    let mut constraints = Vec::new();
    for marker in metadata.iter() {
        parse_constraint(&marker, &mut constraints, lits);
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
    lits: &mut Vec<Py<PyAny>>,
) {
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
            out.push(make(intern(lits, &bound)));
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
    // Predicate escape hatch: annotated_types.Predicate(.func) or a bare
    // callable used directly as metadata.
    if let Ok(func) = marker.getattr("func")
        && func.is_callable()
    {
        out.push(Constraint::Predicate(intern(lits, &func)));
    } else if marker.is_callable() {
        out.push(Constraint::Predicate(intern(lits, marker)));
    }
}

/// Pool `obj` and return its index, deduplicating by object identity.
fn intern(pool: &mut Vec<Py<PyAny>>, obj: &Bound<'_, PyAny>) -> usize {
    let ptr = obj.as_ptr();
    if let Some(index) = pool.iter().position(|existing| existing.as_ptr() == ptr) {
        return index;
    }
    let index = pool.len();
    pool.push(obj.clone().unbind());
    index
}

fn is_ellipsis(obj: &Bound<'_, PyAny>) -> bool {
    obj.py()
        .import("builtins")
        .and_then(|builtins| builtins.getattr("Ellipsis"))
        .is_ok_and(|ellipsis| obj.is(&ellipsis))
}

fn not_implemented(message: &str) -> PyErr {
    PyNotImplementedError::new_err(message.to_owned())
}

// ---------------------------------------------------------------------------
// Walk: membership testing of a Python value against the IR.
// ---------------------------------------------------------------------------

/// Walk the schema against `value` in Rust, returning the first [`Violation`]
/// or `None` if the value is a member. `path` accumulates the location of the
/// current value; `pool` holds the literal constants.
fn check(
    schema: &Schema,
    value: &Bound<'_, PyAny>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
) -> Option<Violation> {
    match schema {
        Schema::Anything | Schema::Any => None,
        Schema::Nothing => Some(mismatch(schema, value, path)),
        Schema::NoneType => admit(value.is_none(), schema, value, path),
        Schema::Bool => admit(value.is_instance_of::<PyBool>(), schema, value, path),
        // bool subclasses int, so True/False are ints: Bool is a subset of Int.
        Schema::Int => admit(value.is_instance_of::<PyInt>(), schema, value, path),
        Schema::Float => admit(value.is_instance_of::<PyFloat>(), schema, value, path),
        Schema::Str => admit(value.is_instance_of::<PyString>(), schema, value, path),
        Schema::Bytes => admit(value.is_instance_of::<PyBytes>(), schema, value, path),
        Schema::Literal(index) => check_literal(*index, value, path, ctx),
        Schema::Sequence(element) => check_sequence(element, value, path, ctx),
        Schema::Tuple(elements) => check_tuple(elements, value, path, ctx),
        Schema::VariadicTuple(element) => check_variadic_tuple(element, value, path, ctx),
        Schema::Set(element) => check_set(element, value, path, ctx),
        Schema::FrozenSet(element) => check_frozenset(element, value, path, ctx),
        Schema::Mapping { key, value: val } => check_mapping(key, val, value, path, ctx),
        Schema::Record { fields } => check_record(fields, value, path, ctx),
        Schema::Union(members) => check_union(members, value, path, ctx),
        Schema::Intersection(members) => check_intersection(members, value, path, ctx),
        Schema::Complement(inner) => check_complement(inner, value, path, ctx),
        Schema::Instance(index) => check_instance(*index, value, path, ctx),
        Schema::Object {
            class_index,
            fields,
        } => check_object(*class_index, fields, value, path, ctx),
        Schema::Refine { base, constraints } => check_refine(base, constraints, value, path, ctx),
        Schema::Ref(id) => check_ref(*id, value, path, ctx),
        // A SelfRef should have been resolved into a Ref at build time; reaching
        // one means an unresolved recursion marker leaked into a validator.
        Schema::SelfRef(_) => Some(Violation {
            code: "unresolved_recursion",
            path: path.clone(),
            expected: "a resolved recursive value".to_owned(),
            value_summary: summarize(value),
        }),
    }
}

/// The most levels of recursive descent allowed before a value is rejected. A
/// finite value never reaches this; the bound exists so a pathologically deep
/// value fails with `recursion_limit` instead of overflowing the native stack.
const MAX_RECURSION_DEPTH: usize = 128;

/// Validate `value` against the definition `defs[id]`, guarding against value
/// cycles and unbounded depth. The guard holds `(object id, definition index)`
/// for every reference currently on the path: revisiting the same pair means
/// the value contains itself, and the number of entries is the current depth.
fn check_ref(
    id: usize,
    value: &Bound<'_, PyAny>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
) -> Option<Violation> {
    let key = (value.as_ptr() as usize, id);
    let depth = {
        let mut guard = ctx.guard.borrow_mut();
        if !guard.insert(key) {
            return Some(Violation {
                code: "recursion_loop",
                path: path.clone(),
                expected: "a finite (non-cyclic) value".to_owned(),
                value_summary: summarize(value),
            });
        }
        guard.len()
    };
    if depth > MAX_RECURSION_DEPTH {
        ctx.guard.borrow_mut().remove(&key);
        return Some(Violation {
            code: "recursion_limit",
            path: path.clone(),
            expected: format!("at most {MAX_RECURSION_DEPTH} levels of recursion"),
            value_summary: summarize(value),
        });
    }
    let result = check(&ctx.defs[id], value, path, ctx);
    ctx.guard.borrow_mut().remove(&key);
    result
}

fn check_refine(
    base: &Schema,
    constraints: &[Constraint],
    value: &Bound<'_, PyAny>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
) -> Option<Violation> {
    // Constraints narrow the base set, so they are meaningful only on a base
    // member: if the base fails, report that and do not run the constraints.
    if let Some(violation) = check(base, value, path, ctx) {
        return Some(violation);
    }
    for constraint in constraints {
        if let Some(violation) = check_constraint(constraint, value, path, ctx) {
            return Some(violation);
        }
    }
    None
}

fn check_constraint(
    constraint: &Constraint,
    value: &Bound<'_, PyAny>,
    path: &[PathSegment],
    ctx: Ctx<'_>,
) -> Option<Violation> {
    let py = value.py();
    let (ok, code, expected) = match constraint {
        Constraint::Ge(i) => {
            let bound = ctx.pool[*i].bind(py);
            (
                cmp(value.ge(bound)),
                "greater_than_equal",
                format!(">= {}", summarize(bound)),
            )
        }
        Constraint::Gt(i) => {
            let bound = ctx.pool[*i].bind(py);
            (
                cmp(value.gt(bound)),
                "greater_than",
                format!("> {}", summarize(bound)),
            )
        }
        Constraint::Le(i) => {
            let bound = ctx.pool[*i].bind(py);
            (
                cmp(value.le(bound)),
                "less_than_equal",
                format!("<= {}", summarize(bound)),
            )
        }
        Constraint::Lt(i) => {
            let bound = ctx.pool[*i].bind(py);
            (
                cmp(value.lt(bound)),
                "less_than",
                format!("< {}", summarize(bound)),
            )
        }
        Constraint::MinLen(n) => (
            value.len().is_ok_and(|len| len >= *n),
            "too_short",
            format!("length >= {n}"),
        ),
        Constraint::MaxLen(n) => (
            value.len().is_ok_and(|len| len <= *n),
            "too_long",
            format!("length <= {n}"),
        ),
        Constraint::Predicate(i) => {
            // Slow path: the user's Python callable runs at the boundary. A
            // raising predicate is surfaced as a distinct `predicate_error`
            // rather than masked as an ordinary failed match.
            let predicate = ctx.pool[*i].bind(py);
            match predicate_passes(predicate, value) {
                Ok(passed) => (passed, "predicate_failed", "a passing predicate".to_owned()),
                Err(err) => (
                    false,
                    "predicate_error",
                    format!("a predicate that does not raise (raised {err})"),
                ),
            }
        }
    };
    if ok {
        None
    } else {
        Some(Violation {
            code,
            path: path.to_vec(),
            expected,
            value_summary: summarize(value),
        })
    }
}

/// Run a user predicate and report whether it returned a truthy result.
/// Returns `Err` if the predicate itself raised, so callers can distinguish a
/// false result from a broken predicate.
fn predicate_passes(predicate: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<bool> {
    predicate.call1((value,))?.is_truthy()
}

/// Interpret a rich-comparison result, treating an error as "did not hold".
fn cmp(result: PyResult<bool>) -> bool {
    result.unwrap_or(false)
}

fn check_instance(
    index: usize,
    value: &Bound<'_, PyAny>,
    path: &[PathSegment],
    ctx: Ctx<'_>,
) -> Option<Violation> {
    let class = ctx.pool[index].bind(value.py());
    if value.is_instance(class).unwrap_or(false) {
        None
    } else {
        Some(type_mismatch(
            "instance_type",
            &class_label(class),
            value,
            path,
        ))
    }
}

fn check_object(
    class_index: usize,
    fields: &[Field],
    value: &Bound<'_, PyAny>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
) -> Option<Violation> {
    let class = ctx.pool[class_index].bind(value.py());
    if !value.is_instance(class).unwrap_or(false) {
        // Not an instance: the attribute checks below cannot be trusted.
        return Some(type_mismatch(
            "instance_type",
            &class_label(class),
            value,
            path,
        ));
    }
    for field in fields {
        match value.getattr(field.name.as_str()) {
            Ok(attr) => {
                path.push(PathSegment::Key(field.name.clone()));
                let result = check(&field.schema, &attr, path, ctx);
                path.pop();
                if result.is_some() {
                    return result;
                }
            }
            Err(_) => {
                return Some(located(
                    path,
                    field.name.clone(),
                    "missing_attribute",
                    format!("attribute {:?}", field.name),
                    "missing".to_owned(),
                ));
            }
        }
    }
    None
}

/// Whether `value` is a member of `schema`: the walk produces no violation.
fn is_member(schema: &Schema, value: &Bound<'_, PyAny>, ctx: Ctx<'_>) -> bool {
    let mut path = Vec::new();
    check(schema, value, &mut path, ctx).is_none()
}

fn check_union(
    members: &[Schema],
    value: &Bound<'_, PyAny>,
    path: &[PathSegment],
    ctx: Ctx<'_>,
) -> Option<Violation> {
    // A value is a member iff it matches at least one branch.
    if members.iter().any(|member| is_member(member, value, ctx)) {
        return None;
    }
    let labels: Vec<&str> = members.iter().map(Schema::expected).collect();
    Some(Violation {
        code: "union_error",
        path: path.to_vec(),
        expected: format!("one of: {}", labels.join(", ")),
        value_summary: summarize(value),
    })
}

fn check_intersection(
    members: &[Schema],
    value: &Bound<'_, PyAny>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
) -> Option<Violation> {
    // Every member must hold; report the first that fails.
    for member in members {
        if let Some(violation) = check(member, value, path, ctx) {
            return Some(violation);
        }
    }
    None
}

fn check_complement(
    inner: &Schema,
    value: &Bound<'_, PyAny>,
    path: &[PathSegment],
    ctx: Ctx<'_>,
) -> Option<Violation> {
    // A value that matches the inner schema fails the complement.
    if is_member(inner, value, ctx) {
        Some(Violation {
            code: "unexpected_match",
            path: path.to_vec(),
            expected: format!("not {}", inner.expected()),
            value_summary: summarize(value),
        })
    } else {
        None
    }
}

fn admit(
    ok: bool,
    schema: &Schema,
    value: &Bound<'_, PyAny>,
    path: &[PathSegment],
) -> Option<Violation> {
    if ok {
        None
    } else {
        Some(mismatch(schema, value, path))
    }
}

fn mismatch(schema: &Schema, value: &Bound<'_, PyAny>, path: &[PathSegment]) -> Violation {
    Violation {
        code: schema.error_code(),
        path: path.to_vec(),
        expected: schema.expected().to_owned(),
        value_summary: summarize(value),
    }
}

fn type_mismatch(
    code: &'static str,
    expected: &str,
    value: &Bound<'_, PyAny>,
    path: &[PathSegment],
) -> Violation {
    Violation {
        code,
        path: path.to_vec(),
        expected: expected.to_owned(),
        value_summary: summarize(value),
    }
}

/// Whether `value` is the typed singleton denoted by `literal`: same type and
/// equal. The same-type guard rules out Python's cross-type equality
/// (`1 == True == 1.0`), so `Literal[1]` denotes `{1}`, not `{1, True, 1.0}`.
fn literal_matches(value: &Bound<'_, PyAny>, literal: &Bound<'_, PyAny>) -> bool {
    value.get_type().is(literal.get_type()) && value.eq(literal).unwrap_or(false)
}

fn check_literal(
    index: usize,
    value: &Bound<'_, PyAny>,
    path: &[PathSegment],
    ctx: Ctx<'_>,
) -> Option<Violation> {
    let literal = ctx.pool[index].bind(value.py());
    if literal_matches(value, literal) {
        None
    } else {
        Some(Violation {
            code: "literal_value",
            path: path.to_vec(),
            expected: format!("the literal {}", summarize(literal)),
            value_summary: summarize(value),
        })
    }
}

fn check_sequence(
    element: &Schema,
    value: &Bound<'_, PyAny>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
) -> Option<Violation> {
    let Ok(list) = value.cast::<PyList>() else {
        return Some(type_mismatch("list_type", "list", value, path));
    };
    for (index, item) in list.iter().enumerate() {
        path.push(PathSegment::Index(index));
        let result = check(element, &item, path, ctx);
        path.pop();
        if result.is_some() {
            return result;
        }
    }
    None
}

fn check_tuple(
    elements: &[Schema],
    value: &Bound<'_, PyAny>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
) -> Option<Violation> {
    let Ok(tuple) = value.cast::<PyTuple>() else {
        return Some(type_mismatch("tuple_type", "tuple", value, path));
    };
    if tuple.len() != elements.len() {
        return Some(Violation {
            code: "tuple_length",
            path: path.clone(),
            expected: format!("tuple of length {}", elements.len()),
            value_summary: summarize(value),
        });
    }
    for (index, (schema, item)) in elements.iter().zip(tuple.iter()).enumerate() {
        path.push(PathSegment::Index(index));
        let result = check(schema, &item, path, ctx);
        path.pop();
        if result.is_some() {
            return result;
        }
    }
    None
}

fn check_variadic_tuple(
    element: &Schema,
    value: &Bound<'_, PyAny>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
) -> Option<Violation> {
    let Ok(tuple) = value.cast::<PyTuple>() else {
        return Some(type_mismatch("tuple_type", "tuple", value, path));
    };
    for (index, item) in tuple.iter().enumerate() {
        path.push(PathSegment::Index(index));
        let result = check(element, &item, path, ctx);
        path.pop();
        if result.is_some() {
            return result;
        }
    }
    None
}

fn check_set(
    element: &Schema,
    value: &Bound<'_, PyAny>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
) -> Option<Violation> {
    let Ok(set) = value.cast::<PySet>() else {
        return Some(type_mismatch("set_type", "set", value, path));
    };
    // Set order is not meaningful, so element failures carry no index segment.
    for item in set.iter() {
        let result = check(element, &item, path, ctx);
        if result.is_some() {
            return result;
        }
    }
    None
}

fn check_frozenset(
    element: &Schema,
    value: &Bound<'_, PyAny>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
) -> Option<Violation> {
    let Ok(set) = value.cast::<PyFrozenSet>() else {
        return Some(type_mismatch("frozenset_type", "frozenset", value, path));
    };
    for item in set.iter() {
        let result = check(element, &item, path, ctx);
        if result.is_some() {
            return result;
        }
    }
    None
}

fn check_mapping(
    key_schema: &Schema,
    value_schema: &Schema,
    value: &Bound<'_, PyAny>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
) -> Option<Violation> {
    let Ok(dict) = value.cast::<PyDict>() else {
        return Some(type_mismatch("dict_type", "dict", value, path));
    };
    for (key, val) in dict.iter() {
        path.push(PathSegment::Key(key_label(&key)));
        let result =
            check(key_schema, &key, path, ctx).or_else(|| check(value_schema, &val, path, ctx));
        path.pop();
        if result.is_some() {
            return result;
        }
    }
    None
}

fn check_record(
    fields: &[Field],
    value: &Bound<'_, PyAny>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
) -> Option<Violation> {
    let Ok(dict) = value.cast::<PyDict>() else {
        return Some(type_mismatch("dict_type", "dict", value, path));
    };
    // Declared fields, in declared order: present values are checked, absent
    // required keys fail.
    for field in fields {
        match dict.get_item(field.name.as_str()) {
            Ok(Some(item)) => {
                path.push(PathSegment::Key(field.name.clone()));
                let result = check(&field.schema, &item, path, ctx);
                path.pop();
                if result.is_some() {
                    return result;
                }
            }
            Ok(None) if field.required => {
                return Some(located(
                    path,
                    field.name.clone(),
                    "missing_key",
                    format!("required key {:?}", field.name),
                    "missing".to_owned(),
                ));
            }
            Ok(None) => {}
            Err(_) => return Some(type_mismatch("dict_type", "dict", value, path)),
        }
    }
    // Closed record: an undeclared key is a failure.
    let declared: HashSet<&str> = fields.iter().map(|field| field.name.as_str()).collect();
    for (key, _) in dict.iter() {
        let key_text = key
            .str()
            .map_or_else(|_| String::new(), |text| text.to_string());
        if !declared.contains(key_text.as_str()) {
            return Some(located(
                path,
                key_text.clone(),
                "extra_key",
                "no unexpected key".to_owned(),
                format!("{key_text:?}"),
            ));
        }
    }
    None
}

/// Build a violation whose path is `path` extended by one key segment.
fn located(
    path: &[PathSegment],
    key: String,
    code: &'static str,
    expected: String,
    value_summary: String,
) -> Violation {
    let mut full = path.to_vec();
    full.push(PathSegment::Key(key));
    Violation {
        code,
        path: full,
        expected,
        value_summary,
    }
}

/// A short, printable label for a mapping key, used in error paths.
fn key_label(key: &Bound<'_, PyAny>) -> String {
    match key.str() {
        Ok(text) => truncate(&text.to_string(), 40),
        Err(_) => summarize(key),
    }
}

// ---------------------------------------------------------------------------
// Error construction.
// ---------------------------------------------------------------------------

/// The class name for an error label, falling back to its repr.
fn class_label(class: &Bound<'_, PyAny>) -> String {
    class
        .getattr("__name__")
        .ok()
        .and_then(|name| name.extract::<String>().ok())
        .unwrap_or_else(|| summarize(class))
}

/// A short repr-style summary of a value for error messages.
fn summarize(value: &Bound<'_, PyAny>) -> String {
    match value.repr() {
        Ok(repr) => truncate(&repr.to_string(), 80),
        Err(_) => "<unrepresentable>".to_owned(),
    }
}

fn truncate(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        text.to_owned()
    } else {
        let head: String = text.chars().take(max_chars).collect();
        format!("{head}...")
    }
}

/// Build the Python [`ValidationError`] for a violation, carrying its
/// machine-readable `code`, `path`, `expected` label, `value` summary, and the
/// rendered `message`.
fn into_pyerr(py: Python<'_>, violation: &Violation) -> PyErr {
    let err = ValidationError::new_err(violation.to_string());
    let instance = err.value(py);
    let path = build_path(py, &violation.path).unwrap_or_else(|_| PyTuple::empty(py));
    let _ = instance.setattr("code", violation.code);
    let _ = instance.setattr("expected", violation.expected.as_str());
    let _ = instance.setattr("value", violation.value_summary.as_str());
    let _ = instance.setattr("message", violation.to_string());
    let _ = instance.setattr("path", &path);
    err
}

// ---------------------------------------------------------------------------
// Render: a compiled schema back to a readable annotation expression.
// ---------------------------------------------------------------------------

/// Render a schema back to the annotation/combinator expression that produces
/// it.
fn render(
    py: Python<'_>,
    schema: &Schema,
    pool: &[Py<PyAny>],
    defs: &[Schema],
    active: &RefCell<HashSet<usize>>,
) -> String {
    let r = |s: &Schema| render(py, s, pool, defs, active);
    let kids = |members: &[Schema]| members.iter().map(&r).collect::<Vec<_>>().join(", ");
    match schema {
        Schema::Anything => "anything".to_owned(),
        Schema::Any => "Any".to_owned(),
        Schema::Nothing => "nothing".to_owned(),
        Schema::NoneType => "None".to_owned(),
        Schema::Bool => "bool".to_owned(),
        Schema::Int => "int".to_owned(),
        Schema::Float => "float".to_owned(),
        Schema::Str => "str".to_owned(),
        Schema::Bytes => "bytes".to_owned(),
        Schema::Literal(i) => format!("Literal[{}]", pool_repr(py, pool, *i)),
        Schema::Sequence(e) => format!("list[{}]", r(e)),
        Schema::Tuple(es) => format!("tuple[{}]", kids(es)),
        Schema::VariadicTuple(e) => format!("tuple[{}, ...]", r(e)),
        Schema::Set(e) => format!("set[{}]", r(e)),
        Schema::FrozenSet(e) => format!("frozenset[{}]", r(e)),
        Schema::Mapping { key, value } => format!("dict[{}, {}]", r(key), r(value)),
        Schema::Record { fields } => render_record(py, fields, pool, defs, active),
        Schema::Union(members) => members.iter().map(&r).collect::<Vec<_>>().join(" | "),
        Schema::Intersection(members) => format!("intersect({})", kids(members)),
        Schema::Complement(inner) => format!("complement({})", r(inner)),
        Schema::Instance(i) | Schema::Object { class_index: i, .. } => {
            pool_class_name(py, pool, *i)
        }
        Schema::Refine { base, constraints } => {
            let mut parts = vec![r(base)];
            parts.extend(constraints.iter().map(|c| render_constraint(py, c, pool)));
            format!("Annotated[{}]", parts.join(", "))
        }
        Schema::Ref(id) => {
            // Unfold the definition once; a back-edge to a reference already
            // being rendered shows as `...`, so the form stays finite.
            if !active.borrow_mut().insert(*id) {
                return "...".to_owned();
            }
            let body = defs.get(*id).map_or_else(|| "...".to_owned(), &r);
            active.borrow_mut().remove(id);
            body
        }
        Schema::SelfRef(_) => "...".to_owned(),
    }
}

fn render_record(
    py: Python<'_>,
    fields: &[Field],
    pool: &[Py<PyAny>],
    defs: &[Schema],
    active: &RefCell<HashSet<usize>>,
) -> String {
    let entries: Vec<String> = fields
        .iter()
        .map(|field| {
            let suffix = if field.required { "" } else { "?" };
            format!(
                "'{}{}': {}",
                field.name,
                suffix,
                render(py, &field.schema, pool, defs, active)
            )
        })
        .collect();
    format!("{{{}}}", entries.join(", "))
}

fn render_constraint(py: Python<'_>, constraint: &Constraint, pool: &[Py<PyAny>]) -> String {
    match constraint {
        Constraint::Ge(i) => format!("Ge({})", pool_repr(py, pool, *i)),
        Constraint::Gt(i) => format!("Gt({})", pool_repr(py, pool, *i)),
        Constraint::Le(i) => format!("Le({})", pool_repr(py, pool, *i)),
        Constraint::Lt(i) => format!("Lt({})", pool_repr(py, pool, *i)),
        Constraint::MinLen(n) => format!("MinLen({n})"),
        Constraint::MaxLen(n) => format!("MaxLen({n})"),
        Constraint::Predicate(_) => "Predicate(...)".to_owned(),
    }
}

fn pool_repr(py: Python<'_>, pool: &[Py<PyAny>], index: usize) -> String {
    summarize(pool[index].bind(py))
}

fn pool_class_name(py: Python<'_>, pool: &[Py<PyAny>], index: usize) -> String {
    class_label(pool[index].bind(py))
}

fn build_path<'py>(py: Python<'py>, path: &[PathSegment]) -> PyResult<Bound<'py, PyTuple>> {
    let mut items: Vec<Bound<'py, PyAny>> = Vec::with_capacity(path.len());
    for segment in path {
        let item = match segment {
            PathSegment::Key(key) => key.as_str().into_pyobject(py)?.into_any(),
            PathSegment::Index(index) => (*index).into_pyobject(py)?.into_any(),
        };
        items.push(item);
    }
    PyTuple::new(py, items)
}
