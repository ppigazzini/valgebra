//! The value abstraction the membership walk runs over.
//!
//! A value being validated comes from one of two sources: a Python object (the
//! object path) or a parsed JSON value (the JSON path, validated in place
//! without materializing Python objects). [`Value`] unifies them so the
//! membership walk is written once and dispatches per node, which keeps the two
//! paths membership-equivalent by construction.
//!
//! Scalar membership and structural traversal are answered directly on each
//! variant. Nodes that compare against a pooled Python object — literals,
//! refinements, instance and object checks, predicates — materialize the value
//! with [`Value::to_python`]; for a JSON value that builds the Python object the
//! object path would have received from `json.loads`, so the decision matches.

use jiter::JsonValue;
use jiter::PythonParse;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use crate::errors::json_invalid_error;
use pyo3::types::{PyBool, PyBytes, PyDict, PyFloat, PyInt, PyList, PyString};

/// A value to validate: a borrowed Python object, or a borrowed parsed JSON
/// value carrying the interpreter token it materializes against.
#[derive(Clone, Copy)]
pub(crate) enum Value<'a, 'py> {
    Py(&'a Bound<'py, PyAny>),
    Json(Python<'py>, &'a JsonValue<'a>),
}

impl<'py> Value<'_, 'py> {
    /// The interpreter token, which binds pooled objects and materializes.
    pub(crate) fn py(&self) -> Python<'py> {
        match self {
            Value::Py(v) => v.py(),
            Value::Json(py, _) => *py,
        }
    }

    /// Denotes `None`/JSON `null`.
    pub(crate) fn is_none(&self) -> bool {
        match self {
            Value::Py(v) => v.is_none(),
            Value::Json(_, v) => matches!(v, JsonValue::Null),
        }
    }

    /// Denotes the `bool` instances / JSON `true`/`false`.
    pub(crate) fn is_bool(&self) -> bool {
        match self {
            Value::Py(v) => v.is_instance_of::<PyBool>(),
            Value::Json(_, v) => matches!(v, JsonValue::Bool(_)),
        }
    }

    /// Denotes the `int` instances. `bool` is a subtype of `int`, and a JSON
    /// boolean parses to a Python `bool`, so it is admitted here too.
    pub(crate) fn is_int(&self) -> bool {
        match self {
            Value::Py(v) => v.is_instance_of::<PyInt>(),
            Value::Json(_, v) => matches!(
                v,
                JsonValue::Int(_) | JsonValue::BigInt(_) | JsonValue::Bool(_)
            ),
        }
    }

    /// Denotes the `float` instances. `int` is disjoint from `float`.
    pub(crate) fn is_float(&self) -> bool {
        match self {
            Value::Py(v) => v.is_instance_of::<PyFloat>(),
            Value::Json(_, v) => matches!(v, JsonValue::Float(_)),
        }
    }

    /// Denotes the `str` instances / JSON strings.
    pub(crate) fn is_str(&self) -> bool {
        match self {
            Value::Py(v) => v.is_instance_of::<PyString>(),
            Value::Json(_, v) => matches!(v, JsonValue::Str(_)),
        }
    }

    /// Denotes the `bytes` instances. JSON has no bytes, so a JSON value never
    /// belongs to the bytes set.
    pub(crate) fn is_bytes(&self) -> bool {
        match self {
            Value::Py(v) => v.is_instance_of::<PyBytes>(),
            Value::Json(..) => false,
        }
    }

    /// A process-unique identity for the recursion guard. A JSON value is a
    /// finite tree, so its node address is stable for the walk and never repeats
    /// on a path; the guard's length still bounds recursion depth.
    pub(crate) fn id(&self) -> usize {
        match self {
            Value::Py(v) => v.as_ptr() as usize,
            Value::Json(_, v) => std::ptr::from_ref::<JsonValue<'_>>(v) as usize,
        }
    }

    /// The value as a Python object: the input itself for the object path, or
    /// the object `json.loads` would have produced for the JSON path. Used by
    /// the nodes that compare against a pooled Python object.
    pub(crate) fn to_python(self) -> PyResult<Bound<'py, PyAny>> {
        match self {
            Value::Py(v) => Ok(v.clone()),
            Value::Json(py, v) => json_to_python(py, v),
        }
    }
}

/// Build the Python object that `json.loads` would produce for a parsed JSON
/// value: `null`/bool/int/float/str map to their builtins, arrays to lists, and
/// objects to dicts (last value wins on a duplicate key, as `json.loads` does).
fn json_to_python<'py>(py: Python<'py>, value: &JsonValue<'_>) -> PyResult<Bound<'py, PyAny>> {
    Ok(match value {
        JsonValue::Null => py.None().into_bound(py),
        JsonValue::Bool(b) => PyBool::new(py, *b).to_owned().into_any(),
        JsonValue::Int(i) => i.into_pyobject(py)?.into_any(),
        JsonValue::BigInt(b) => b.into_pyobject(py)?.into_any(),
        JsonValue::Float(f) => f.into_pyobject(py)?.into_any(),
        JsonValue::Str(s) => PyString::new(py, s).into_any(),
        JsonValue::Array(items) => {
            let list = PyList::empty(py);
            for item in items.iter() {
                list.append(json_to_python(py, item)?)?;
            }
            list.into_any()
        }
        JsonValue::Object(entries) => {
            let dict = PyDict::new(py);
            // Iterate in order so a later duplicate key overwrites an earlier one.
            for (key, val) in entries.iter() {
                dict.set_item(key.as_ref(), json_to_python(py, val)?)?;
            }
            dict.into_any()
        }
    })
}

/// The UTF-8 bytes a JSON `str`/`bytes` argument hands to the parser, or why they
/// are unavailable. A `str` carrying a lone surrogate cannot be encoded to UTF-8;
/// both JSON entry points must treat that the same malformed-input way rather than
/// let the raw `UnicodeEncodeError` leak — which would disagree with the check
/// path and break `validate_json`'s documented exception set.
pub(crate) enum JsonInput<'a> {
    /// Bytes ready for the parser.
    Bytes(&'a [u8]),
    /// A `str` the interpreter cannot encode to UTF-8 (a lone surrogate).
    Undecodable,
    /// Neither `str` nor `bytes`.
    NotStrOrBytes,
}

/// Decode a JSON argument to the bytes the parser reads, classifying the two ways
/// it can be unusable so both entry points agree on them.
pub(crate) fn decode_json_input<'a>(data: &'a Bound<'_, PyAny>) -> JsonInput<'a> {
    if let Ok(text) = data.cast::<PyString>() {
        match text.to_str() {
            Ok(text) => JsonInput::Bytes(text.as_bytes()),
            Err(_) => JsonInput::Undecodable,
        }
    } else if let Ok(raw) = data.cast::<PyBytes>() {
        JsonInput::Bytes(raw.as_bytes())
    } else {
        JsonInput::NotStrOrBytes
    }
}

/// Parse a JSON `str` or `bytes` into a Python value with jiter.
///
/// jiter's defaults match the standard JSON model: standard `float`s, no
/// `Infinity`/`NaN`, and complete (non-partial) input — so the parsed value is
/// what the object path would receive from `json.loads`. A parse failure, or a
/// `str` the interpreter cannot encode to UTF-8, is surfaced as a structured
/// `json_invalid` `ValidationError`; a non-string, non-bytes argument is a
/// `TypeError`.
pub(crate) fn parse_json<'py>(data: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let py = data.py();
    let parse = PythonParse::default();
    match decode_json_input(data) {
        JsonInput::Bytes(bytes) => parse
            .python_parse(py, bytes)
            .map_err(|err| json_invalid_error(py, &err.description(bytes))),
        // An undecodable string is malformed input, reported through the same
        // structured `json_invalid` model as an unparseable document — never as a
        // raw `UnicodeEncodeError`, which `validate_json` promises not to raise.
        JsonInput::Undecodable => Err(json_invalid_error(
            py,
            "input string is not valid UTF-8 (contains a lone surrogate)",
        )),
        JsonInput::NotStrOrBytes => Err(PyTypeError::new_err(
            "JSON input must be a str or bytes object",
        )),
    }
}
