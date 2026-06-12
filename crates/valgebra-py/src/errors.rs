//! Error construction: violation summaries, value labels, and the Python
//! `ValidationError` raised from a [`valgebra_core::Violation`].

use pyo3::prelude::*;
use pyo3::types::PyTuple;
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

/// Build the Python [`ValidationError`] for a violation, carrying its
/// machine-readable `code`, `path`, `expected` label, `value` summary, and the
/// rendered `message`.
pub(crate) fn into_pyerr(py: Python<'_>, violation: &Violation) -> PyErr {
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
