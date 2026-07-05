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
    // The caller always reports at least one failure. A debug build asserts it; a
    // release build must not index into an empty slice below, so if the fast
    // matcher and the explain pass ever disagreed and left no violation, stand in
    // a generic one. The error stays well-formed instead of raising a panic from
    // an out-of-bounds access on the error-reporting path.
    let fallback;
    let violations = if violations.is_empty() {
        fallback = [Violation {
            code: "validation_error",
            path: Vec::new(),
            expected: "a member of the schema's set".to_owned(),
            value_summary: String::new(),
        }];
        &fallback[..]
    } else {
        violations
    };
    // Populating the structured attributes is pure interpreter bookkeeping
    // (attribute sets on a fresh exception, dict/tuple builds over owned data) and
    // does not fail in practice. If it ever does, surface that failure (the `Err`)
    // rather than shipping a `ValidationError` whose `.errors` is silently empty
    // while `str(exc)` still summarizes real failures.
    match build_validation_error(py, violations) {
        Ok(err) | Err(err) => err,
    }
}

fn build_validation_error(py: Python<'_>, violations: &[Violation]) -> PyResult<PyErr> {
    let err = ValidationError::new_err(summary_message(violations));
    let instance = err.value(py);
    let first = &violations[0];
    instance.setattr("code", first.code)?;
    instance.setattr("expected", first.expected.as_str())?;
    instance.setattr("value", first.value_summary.as_str())?;
    instance.setattr("message", first.to_string())?;
    instance.setattr("path", build_path(py, &first.path)?)?;
    instance.setattr("errors", error_items(py, violations)?)?;
    Ok(err)
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

// Needs a live interpreter; compiled and run only under the `interpreter-tests`
// feature, which links an embedded Python.
#[cfg(all(test, feature = "interpreter-tests"))]
mod tests {
    use super::*;

    fn violation(code: &'static str, path: Vec<PathSegment>) -> Violation {
        Violation {
            code,
            path,
            expected: "int".to_owned(),
            value_summary: "'x'".to_owned(),
        }
    }

    #[test]
    fn into_pyerr_maps_violations_to_the_structured_attributes() {
        Python::attach(|py| {
            let violations = vec![
                violation("int_type", vec![PathSegment::Key("a".to_owned())]),
                violation("missing", vec![PathSegment::Index(2)]),
            ];
            let err = into_pyerr(py, &violations);
            let value = err.value(py);

            // The scalar attributes mirror the first violation; the path is the
            // built tuple.
            assert_eq!(
                value.getattr("code").unwrap().extract::<String>().unwrap(),
                "int_type"
            );
            assert_eq!(
                value
                    .getattr("expected")
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "int"
            );
            let path: Vec<String> = value.getattr("path").unwrap().extract().unwrap();
            assert_eq!(path, vec!["a".to_owned()]);

            // `errors` carries one item per violation, in order, each with its code.
            let errors = value.getattr("errors").unwrap();
            assert_eq!(errors.len().unwrap(), 2);
            let second_code: String = errors
                .get_item(1)
                .unwrap()
                .get_item("code")
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(second_code, "missing");
        });
    }
}
