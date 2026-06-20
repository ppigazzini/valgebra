//! Building the structured [`Violation`] values the explain walk reports.

use pyo3::prelude::*;
use valgebra_core::{PathSegment, Schema, Violation};

use crate::check::Ctx;
use crate::errors::{summarize, truncate};
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
    if ctx.explain {
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

/// A short, printable label for a mapping key, used in error paths.
pub(crate) fn key_label(key: &Bound<'_, PyAny>) -> String {
    match key.str() {
        Ok(text) => truncate(&text.to_string(), 40),
        Err(_) => summarize(key),
    }
}
