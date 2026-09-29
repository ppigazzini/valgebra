//! Building the structured [`Violation`] values the explain walk reports.

use std::sync::Arc;

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyInt, PyString};
use valgebra_core::{PathSegment, Schema, Violation};

use crate::check::ctx::Ctx;
use crate::check::walk::record_if_fatal;
use crate::codes::Code;
use crate::errors::{UNREPRESENTABLE, try_summarize};
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
    match value.to_python() {
        Ok(obj) => summarize_in(&obj, ctx),
        Err(err) => {
            record_if_fatal(err, value.py(), ctx);
            UNREPRESENTABLE.to_owned()
        }
    }
}

/// A short repr-style summary of an object a message the walk builds names:
/// the value, a constant, a bound, a key.
///
/// A `__repr__` that raises an ordinary exception is an object that cannot
/// render, and the message says so. One that raises a fatal signal is the
/// interpreter unwinding, and the walk carries it out rather than folding it
/// into a summary -- which is the rule at every other site an object answers a
/// question.
pub(crate) fn summarize_in(obj: &Bound<'_, PyAny>, ctx: Ctx<'_>) -> String {
    try_summarize(obj).unwrap_or_else(|err| {
        record_if_fatal(err, obj.py(), ctx);
        UNREPRESENTABLE.to_owned()
    })
}

/// A class's name for a message the walk builds, or its summary where the
/// name is not a string.
///
/// [`summarize_in`]'s rule for the name as well as the repr: a metaclass may
/// answer `__name__` by running code, and a fatal signal it raises is carried
/// out rather than read as a class with no name.
pub(crate) fn class_label_in(class: &Bound<'_, PyAny>, ctx: Ctx<'_>) -> String {
    match class.getattr(intern!(class.py(), "__name__")) {
        Ok(name) => match name.extract::<String>() {
            Ok(text) => text,
            Err(_) => summarize_in(class, ctx),
        },
        Err(err) => {
            record_if_fatal(err, class.py(), ctx);
            summarize_in(class, ctx)
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
/// A `bool` is an `int` in Python, and `d[True]` is the entry `d[1]` is, so a
/// boolean key takes the integer path and reads as `1` or `0`. A string holding
/// a lone surrogate has no UTF-8 text to spell, so it appears as its `repr` too:
/// read as the empty string, it named the entry `d[""]` is.
pub(crate) fn key_segment(key: &Bound<'_, PyAny>, ctx: Ctx<'_>) -> PathSegment {
    if let Ok(text) = key.cast::<PyString>() {
        if let Ok(text) = text.to_cow() {
            return PathSegment::Key(Arc::from(text.as_ref()));
        }
    } else if let Ok(number) = key.cast::<PyInt>() {
        // Every `int`, whatever its size, and `bool` with them: `d[True]` and
        // `d[1]` are one entry in Python, so the integer indexes back down to
        // the value while `'True'` indexed nothing at all.
        if let Ok(small) = number.extract::<i64>() {
            return PathSegment::IntKey(small);
        }
        match stored_digits(number) {
            Ok(digits) => return PathSegment::BigIntKey(digits),
            Err(err) => record_if_fatal(err, key.py(), ctx),
        }
    }
    PathSegment::Key(Arc::from(summarize_in(key, ctx).as_str()))
}

/// An integer's decimal digits, read from its storage: `int.__repr__`, the
/// base's own method, called on it. A subclass's `__str__` answers what it
/// likes, which is a spelling of some other key or a raise; the small-integer
/// path above reads the storage too, so both halves name the same entry.
fn stored_digits(number: &Bound<'_, PyInt>) -> PyResult<String> {
    let py = number.py();
    py.get_type::<PyInt>()
        .getattr(intern!(py, "__repr__"))?
        .call1((number,))?
        .extract()
}
