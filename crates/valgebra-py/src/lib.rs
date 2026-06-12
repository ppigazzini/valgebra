//! `PyO3` bindings for valgebra: compile a Python schema once into the core IR
//! and walk it entirely in Rust, crossing the boundary once per call.
//!
//! The frontend ([`build_schema`]) reads native Python forms into the IR; the
//! walk ([`check`]) tests membership of a concrete value, reporting the first
//! failure as a structured [`ValidationError`].

use std::collections::HashSet;

use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyNotImplementedError};
use pyo3::prelude::*;
use pyo3::types::{
    PyBool, PyBytes, PyDict, PyFloat, PyInt, PyList, PySet, PyString, PyTuple, PyType,
};
use valgebra_core::{Field, PathSegment, Schema, Violation};

create_exception!(
    _valgebra,
    ValidationError,
    PyException,
    "Raised when a value is not a member of a schema's set."
);

/// An immutable compiled validator: a schema IR plus the constants pool its
/// literals index into. Building one compiles the schema; the validator itself
/// never mutates.
#[pyclass(frozen, module = "valgebra._valgebra")]
pub struct CompiledValidator {
    schema: Schema,
    literals: Vec<Py<PyAny>>,
}

#[pymethods]
impl CompiledValidator {
    /// Raise [`ValidationError`] if `obj` is not a member of the schema's set;
    /// return `None` otherwise. Check-only: the object is not copied or coerced.
    fn validate(&self, obj: &Bound<'_, PyAny>) -> PyResult<()> {
        let mut path = Vec::new();
        match check(&self.schema, obj, &mut path, &self.literals) {
            Some(violation) => Err(into_pyerr(obj.py(), &violation)),
            None => Ok(()),
        }
    }

    /// Whether `obj` belongs to the schema's set. Check-only, returns a bool.
    fn is_valid(&self, obj: &Bound<'_, PyAny>) -> bool {
        let mut path = Vec::new();
        check(&self.schema, obj, &mut path, &self.literals).is_none()
    }

    /// Validate `obj` and return it unchanged. The explicit conversion mode:
    /// validation is a membership check, so the returned object is the input.
    fn cast<'py>(&self, obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.validate(obj)?;
        Ok(obj.clone())
    }
}

/// Compile `schema` into an immutable [`CompiledValidator`].
#[pyfunction]
fn validator(schema: &Bound<'_, PyAny>) -> PyResult<CompiledValidator> {
    let mut literals = Vec::new();
    let schema = build_schema(schema, &mut literals)?;
    Ok(CompiledValidator { schema, literals })
}

/// The `valgebra._valgebra` extension module.
#[pymodule]
fn _valgebra(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("ValidationError", module.py().get_type::<ValidationError>())?;
    module.add_class::<CompiledValidator>()?;
    module.add_function(wrap_pyfunction!(validator, module)?)?;
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
fn build_schema(obj: &Bound<'_, PyAny>, lits: &mut Vec<Py<PyAny>>) -> PyResult<Schema> {
    let py = obj.py();
    if obj.is_none() {
        return Ok(Schema::NoneType);
    }

    // Typing constructs (list[int], dict[K, V], tuple[...], ...) are read
    // through the typing spec's own introspection, so the builtin and legacy
    // aliases share one path. A non-typing object has origin None and falls
    // through to the native handling below.
    let typing = py.import("typing")?;
    let origin = typing.call_method1("get_origin", (obj,))?;
    if !origin.is_none() {
        let args = typing.call_method1("get_args", (obj,))?;
        return build_parametrized(&origin, args.cast::<PyTuple>()?, lits);
    }

    if let Ok(ty) = obj.cast::<PyType>() {
        return build_type_object(ty);
    }
    if let Ok(list) = obj.cast::<PyList>() {
        return build_sequence(list, lits);
    }
    if let Ok(tuple) = obj.cast::<PyTuple>() {
        let mut elements = Vec::with_capacity(tuple.len());
        for item in tuple.iter() {
            elements.push(build_schema(&item, lits)?);
        }
        return Ok(Schema::Tuple(elements));
    }
    if let Ok(set) = obj.cast::<PySet>() {
        return build_set(set, lits);
    }
    if let Ok(dict) = obj.cast::<PyDict>() {
        return build_dict(dict, lits);
    }
    Ok(Schema::Literal(intern(lits, obj)))
}

/// Build the schema for a builtin type object or `object`.
fn build_type_object(ty: &Bound<'_, PyType>) -> PyResult<Schema> {
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
    Err(not_implemented(&format!(
        "unsupported type schema: {}",
        summarize(ty.as_any())
    )))
}

/// Build the IR for a parametrized typing generic given its origin and args.
fn build_parametrized(
    origin: &Bound<'_, PyAny>,
    args: &Bound<'_, PyTuple>,
    lits: &mut Vec<Py<PyAny>>,
) -> PyResult<Schema> {
    let py = origin.py();
    if origin.is(py.get_type::<PyList>()) {
        return Ok(Schema::Sequence(Box::new(build_schema(
            &single_arg(args)?,
            lits,
        )?)));
    }
    if origin.is(py.get_type::<PySet>()) {
        return Ok(Schema::Set(Box::new(build_schema(&single_arg(args)?, lits)?)));
    }
    if origin.is(py.get_type::<PyDict>()) {
        if args.len() != 2 {
            return Err(not_implemented(
                "dict[...] needs a key type and a value type",
            ));
        }
        return Ok(Schema::Mapping {
            key: Box::new(build_schema(&args.get_item(0)?, lits)?),
            value: Box::new(build_schema(&args.get_item(1)?, lits)?),
        });
    }
    if origin.is(py.get_type::<PyTuple>()) {
        let mut elements = Vec::with_capacity(args.len());
        for arg in args.iter() {
            if is_ellipsis(&arg) {
                return Err(not_implemented(
                    "tuple[T, ...] is not supported yet; use a fixed tuple shape",
                ));
            }
            elements.push(build_schema(&arg, lits)?);
        }
        return Ok(Schema::Tuple(elements));
    }
    Err(not_implemented(&format!(
        "unsupported typing form with origin {}; supported: list, set, dict, tuple",
        summarize(origin)
    )))
}

fn single_arg<'py>(args: &Bound<'py, PyTuple>) -> PyResult<Bound<'py, PyAny>> {
    if args.len() == 1 {
        args.get_item(0)
    } else {
        Err(not_implemented("expected exactly one type argument"))
    }
}

fn build_sequence(list: &Bound<'_, PyList>, lits: &mut Vec<Py<PyAny>>) -> PyResult<Schema> {
    match list.len() {
        1 => Ok(Schema::Sequence(Box::new(build_schema(
            &list.get_item(0)?,
            lits,
        )?))),
        2 if is_ellipsis(&list.get_item(1)?) => Ok(Schema::Sequence(Box::new(build_schema(
            &list.get_item(0)?,
            lits,
        )?))),
        _ => Err(not_implemented(
            "a list schema must be [T] or [T, ...]; other list shapes are not supported",
        )),
    }
}

fn build_set(set: &Bound<'_, PySet>, lits: &mut Vec<Py<PyAny>>) -> PyResult<Schema> {
    if set.len() == 1
        && let Some(element) = set.iter().next()
    {
        return Ok(Schema::Set(Box::new(build_schema(&element, lits)?)));
    }
    Err(not_implemented(
        "a set schema must have exactly one element, as in {T}",
    ))
}

fn build_dict(dict: &Bound<'_, PyDict>, lits: &mut Vec<Py<PyAny>>) -> PyResult<Schema> {
    // An empty dict is the empty closed record: it matches only {}.
    if dict.is_empty() {
        return Ok(Schema::Record { fields: Vec::new() });
    }
    // All-string keys: a record. A single type-keyed entry: a mapping.
    if dict.iter().all(|(key, _)| key.is_instance_of::<PyString>()) {
        return build_record(dict, lits);
    }
    if dict.len() == 1
        && let Some((key, value)) = dict.iter().next()
        && key.cast::<PyType>().is_ok()
    {
        return Ok(Schema::Mapping {
            key: Box::new(build_schema(&key, lits)?),
            value: Box::new(build_schema(&value, lits)?),
        });
    }
    Err(not_implemented(
        "a dict schema must use all string keys (a record) or a single \
         {KeyType: ValueType} entry (a mapping)",
    ))
}

fn build_record(dict: &Bound<'_, PyDict>, lits: &mut Vec<Py<PyAny>>) -> PyResult<Schema> {
    let mut fields = Vec::with_capacity(dict.len());
    for (key, value) in dict.iter() {
        let raw = key.str()?.to_string();
        let (name, required) = match raw.strip_suffix('?') {
            Some(stripped) => (stripped.to_owned(), false),
            None => (raw, true),
        };
        fields.push(Field {
            name,
            schema: build_schema(&value, lits)?,
            required,
        });
    }
    Ok(Schema::Record { fields })
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
    pool: &[Py<PyAny>],
) -> Option<Violation> {
    match schema {
        Schema::Anything => None,
        Schema::Nothing => Some(mismatch(schema, value, path)),
        Schema::NoneType => admit(value.is_none(), schema, value, path),
        Schema::Bool => admit(value.is_instance_of::<PyBool>(), schema, value, path),
        // bool subclasses int, so True/False are ints: Bool is a subset of Int.
        Schema::Int => admit(value.is_instance_of::<PyInt>(), schema, value, path),
        Schema::Float => admit(value.is_instance_of::<PyFloat>(), schema, value, path),
        Schema::Str => admit(value.is_instance_of::<PyString>(), schema, value, path),
        Schema::Bytes => admit(value.is_instance_of::<PyBytes>(), schema, value, path),
        Schema::Literal(index) => check_literal(*index, value, path, pool),
        Schema::Sequence(element) => check_sequence(element, value, path, pool),
        Schema::Tuple(elements) => check_tuple(elements, value, path, pool),
        Schema::Set(element) => check_set(element, value, path, pool),
        Schema::Mapping { key, value: val } => check_mapping(key, val, value, path, pool),
        Schema::Record { fields } => check_record(fields, value, path, pool),
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
    pool: &[Py<PyAny>],
) -> Option<Violation> {
    let literal = pool[index].bind(value.py());
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
    pool: &[Py<PyAny>],
) -> Option<Violation> {
    let Ok(list) = value.cast::<PyList>() else {
        return Some(type_mismatch("list_type", "list", value, path));
    };
    for (index, item) in list.iter().enumerate() {
        path.push(PathSegment::Index(index));
        let result = check(element, &item, path, pool);
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
    pool: &[Py<PyAny>],
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
        let result = check(schema, &item, path, pool);
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
    pool: &[Py<PyAny>],
) -> Option<Violation> {
    let Ok(set) = value.cast::<PySet>() else {
        return Some(type_mismatch("set_type", "set", value, path));
    };
    // Set order is not meaningful, so element failures carry no index segment.
    for item in set.iter() {
        let result = check(element, &item, path, pool);
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
    pool: &[Py<PyAny>],
) -> Option<Violation> {
    let Ok(dict) = value.cast::<PyDict>() else {
        return Some(type_mismatch("dict_type", "dict", value, path));
    };
    for (key, val) in dict.iter() {
        path.push(PathSegment::Key(key_label(&key)));
        let result =
            check(key_schema, &key, path, pool).or_else(|| check(value_schema, &val, path, pool));
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
    pool: &[Py<PyAny>],
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
                let result = check(&field.schema, &item, path, pool);
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
