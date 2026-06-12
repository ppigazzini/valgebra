//! Error construction: violation summaries, value labels, and the Python
//! `ValidationError` raised from a [`valgebra_core::Violation`].

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use valgebra_core::{PathSegment, Violation};

use crate::ValidationError;

/// The class name for an error label, falling back to its repr.
pub(crate) fn class_label(class: &Bound<'_, PyAny>) -> String {
    class
        .getattr("__name__")
        .ok()
        .and_then(|name| name.extract::<String>().ok())
        .unwrap_or_else(|| summarize(class))
}

/// A short repr-style summary of a value for error messages.
pub(crate) fn summarize(value: &Bound<'_, PyAny>) -> String {
    match value.repr() {
        Ok(repr) => truncate(&repr.to_string(), 80),
        Err(_) => "<unrepresentable>".to_owned(),
    }
}

pub(crate) fn truncate(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        text.to_owned()
    } else {
        let head: String = text.chars().take(max_chars).collect();
        format!("{head}...")
    }
}

/// Build a [`ValidationError`] for input that is not valid JSON.
///
/// Malformed JSON never reaches the validation walk, so it is reported through
/// the same structured model as a membership failure — a single `errors` item
/// coded `json_invalid` whose message is jiter's parse diagnostic — rather than
/// a bare `ValueError`. The path is the root and there is no value to summarize.
pub(crate) fn json_invalid_error(py: Python<'_>, description: &str) -> PyErr {
    let violation = Violation {
        code: "json_invalid",
        path: Vec::new(),
        expected: "valid JSON".to_owned(),
        value_summary: truncate(description, 80),
    };
    into_pyerr(py, &[violation])
}

/// Build the Python [`ValidationError`] for one or more violations.
///
/// The raised instance carries the structured, machine-readable error model:
/// `errors` is a tuple of per-failure items, each a JSON-serializable dict with
/// `code`/`path`/`message`/`expected`/`value`, so `json.dumps(err.errors)` is
/// the JSON output mode. The scalar `message`/`code`/`path`/`expected`/`value`
/// mirror the first item; `str(exc)` is a summary of every failure.
pub(crate) fn into_pyerr(py: Python<'_>, violations: &[Violation]) -> PyErr {
    debug_assert!(!violations.is_empty(), "into_pyerr needs a failure");
    let err = ValidationError::new_err(summary_message(violations));
    let instance = err.value(py);
    let first = &violations[0];
    let first_path = build_path(py, &first.path).unwrap_or_else(|_| PyTuple::empty(py));
    let _ = instance.setattr("code", first.code);
    let _ = instance.setattr("expected", first.expected.as_str());
    let _ = instance.setattr("value", first.value_summary.as_str());
    let _ = instance.setattr("message", first.to_string());
    let _ = instance.setattr("path", &first_path);
    let errors = error_items(py, violations).unwrap_or_else(|_| PyTuple::empty(py));
    let _ = instance.setattr("errors", errors);
    err
}

/// The exception's `str()`: the single message for one failure, or a counted,
/// newline-joined summary for several.
fn summary_message(violations: &[Violation]) -> String {
    if violations.len() == 1 {
        return violations[0].to_string();
    }
    let mut summary = format!("{} validation errors:", violations.len());
    for violation in violations {
        summary.push('\n');
        summary.push_str(&violation.to_string());
    }
    summary
}

/// Build the `errors` tuple: one JSON-serializable item per failure, in walk
/// order.
fn error_items<'py>(py: Python<'py>, violations: &[Violation]) -> PyResult<Bound<'py, PyTuple>> {
    let mut items = Vec::with_capacity(violations.len());
    for violation in violations {
        let item = PyDict::new(py);
        item.set_item("code", violation.code)?;
        item.set_item("path", build_path(py, &violation.path)?)?;
        item.set_item("message", violation.to_string())?;
        item.set_item("expected", violation.expected.as_str())?;
        item.set_item("value", violation.value_summary.as_str())?;
        items.push(item);
    }
    PyTuple::new(py, items)
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
