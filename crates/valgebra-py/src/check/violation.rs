//! Building the structured [`Violation`] values the explain walk reports.

use pyo3::prelude::*;
use pyo3::types::PyString;
use valgebra_core::{PathSegment, Schema, Violation};

use crate::check::ctx::Ctx;
use crate::errors::summarize;
use crate::input::Value;

/// A type/value mismatch for a leaf schema.
pub(crate) fn mismatch(schema: &Schema, value: &Value<'_, '_>, path: &[PathSegment]) -> Violation {
    Violation {
        code: schema.error_code(),
        path: path.to_vec(),
        expected: schema.expected().to_owned(),
        value_summary: summarize_value(value),
    }
}

/// Record a structural type mismatch and report non-membership.
pub(crate) fn type_fail(
    code: &'static str,
    expected: &str,
    value: &Value<'_, '_>,
    path: &[PathSegment],
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
    if ctx.mode.explains() {
        out.push(type_mismatch(code, expected, value, path));
    }
    false
}

pub(crate) fn type_mismatch(
    code: &'static str,
    expected: &str,
    value: &Value<'_, '_>,
    path: &[PathSegment],
) -> Violation {
    Violation {
        code,
        path: path.to_vec(),
        expected: expected.to_owned(),
        value_summary: summarize_value(value),
    }
}

/// Build a violation whose path is `path` extended by one key segment.
pub(crate) fn located(
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

/// A short repr-style summary of a value, materializing a JSON value first.
pub(crate) fn summarize_value(value: &Value<'_, '_>) -> String {
    match value.to_python() {
        Ok(obj) => summarize(&obj),
        Err(_) => "<unrepresentable>".to_owned(),
    }
}

/// The label a mapping key carries in an error path.
///
/// A string key is itself, in full: the path is what a caller walks back down to
/// the value, and a truncated key indexes nothing. A key of any other type has no
/// spelling in a path made of strings and integers, so it appears as its `repr` —
/// which names the key without pretending to be it, and is why the error model
/// says a path is walkable only when every key is a string.
pub(crate) fn key_label(key: &Bound<'_, PyAny>) -> String {
    match key.cast::<PyString>() {
        Ok(text) => text.to_string(),
        Err(_) => summarize(key),
    }
}
