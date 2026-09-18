//! Building the structured [`Violation`] values the explain walk reports.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyInt, PyString};
use valgebra_core::{PathSegment, Schema, Violation};

use crate::check::ctx::Ctx;
use crate::check::walk::{is_fatal, record_fatal};
use crate::codes::Code;
use crate::errors::{summarize, try_summarize};
use crate::input::Value;

/// A type/value mismatch for a leaf schema.
pub(crate) fn mismatch(
    schema: &Schema,
    value: &Value<'_, '_>,
    path: &[PathSegment],
    ctx: Ctx<'_>,
) -> Violation {
    Violation {
        code: Code::of_schema(schema).as_str(),
        path: path.to_vec(),
        expected: schema.expected().to_owned(),
        value_summary: summarize_value(value, ctx),
    }
}

/// Record a structural type mismatch and report non-membership.
pub(crate) fn type_fail(
    code: Code,
    expected: &str,
    value: &Value<'_, '_>,
    path: &[PathSegment],
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
    if ctx.mode.explains() {
        out.push(type_mismatch(code, expected, value, path, ctx));
    }
    false
}

pub(crate) fn type_mismatch(
    code: Code,
    expected: &str,
    value: &Value<'_, '_>,
    path: &[PathSegment],
    ctx: Ctx<'_>,
) -> Violation {
    Violation {
        code: code.as_str(),
        path: path.to_vec(),
        expected: expected.to_owned(),
        value_summary: summarize_value(value, ctx),
    }
}

/// Build a violation whose path is `path` extended by one field name.
pub(crate) fn located(
    path: &[PathSegment],
    key: Arc<str>,
    code: Code,
    expected: String,
    value_summary: String,
) -> Violation {
    at_key(path, PathSegment::Key(key), code, expected, value_summary)
}

/// Build a violation whose path is `path` extended by one segment.
///
/// The general form of [`located`], for a key that is not a field name: an
/// undeclared key is whatever the value carried, and [`key_segment`] is what
/// says how a path names one.
pub(crate) fn at_key(
    path: &[PathSegment],
    key: PathSegment,
    code: Code,
    expected: String,
    value_summary: String,
) -> Violation {
    let mut full = path.to_vec();
    full.push(key);
    Violation {
        code: code.as_str(),
        path: full,
        expected,
        value_summary,
    }
}

/// A short repr-style summary of a value, materializing a JSON value first.
pub(crate) fn summarize_value(value: &Value<'_, '_>, ctx: Ctx<'_>) -> String {
    let Ok(obj) = value.to_python() else {
        return "<unrepresentable>".to_owned();
    };
    match try_summarize(&obj) {
        Ok(text) => text,
        Err(err) => {
            // A `__repr__` that raises an ordinary exception is a value that
            // cannot render, and the message says so. One that raises a fatal
            // signal is the interpreter unwinding, and the walk carries it out
            // rather than folding it into a summary -- which is the rule at
            // every other site a value answers a question.
            if is_fatal(&err, obj.py()) {
                record_fatal(err, ctx);
            }
            "<unrepresentable>".to_owned()
        }
    }
}

/// The segment a mapping key carries in an error path.
///
/// A string key is itself, in full: the path is what a caller walks back down to
/// the value, and a truncated key indexes nothing. An **integer** key is itself
/// too, as an integer -- `d[2]` and `d["2"]` are different entries, and a path
/// that spelled the first as text pointed at the second. A key of any other type
/// has no spelling in a path made of strings and integers, so it appears as its
/// `repr` -- which names the key without pretending to be it, and is why the
/// error model says a path is walkable only when every key is one of the two.
///
/// A `bool` is an `int` in Python and not a key anybody indexes by number, so it
/// takes the repr path with the rest.
pub(crate) fn key_segment(key: &Bound<'_, PyAny>) -> PathSegment {
    if let Ok(text) = key.cast::<PyString>() {
        return PathSegment::Key(Arc::from(text.to_cow().unwrap_or_default().as_ref()));
    }
    if let Ok(number) = key.cast::<PyInt>() {
        // Every `int`, whatever its size, and `bool` with them: `d[True]` and
        // `d[1]` are one entry in Python, so the integer indexes back down to
        // the value while `'True'` indexed nothing at all.
        if let Ok(small) = number.extract::<i64>() {
            return PathSegment::IntKey(small);
        }
        if let Ok(digits) = number.str() {
            return PathSegment::BigIntKey(digits.to_string());
        }
    }
    PathSegment::Key(Arc::from(summarize(key).as_str()))
}
