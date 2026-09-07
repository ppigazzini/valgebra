//! Error construction: violation summaries, value labels, and the Python
//! `ValidationError` raised from a [`valgebra_core::Violation`].

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyFrozenSet, PyList, PySet, PyString, PyTuple};
use valgebra_core::{PathSegment, Violation};

use crate::exception::ValidationError;

/// The class name for an error label, falling back to its repr.
pub(crate) fn class_label(class: &Bound<'_, PyAny>) -> String {
    class
        .getattr("__name__")
        .ok()
        .and_then(|name| name.extract::<String>().ok())
        .unwrap_or_else(|| summarize(class))
}

/// The characters of a value a summary keeps.
const SUMMARY_CHARS: usize = 80;

/// A bounded renderer for containers, built once per interpreter.
///
/// `reprlib.Repr` cuts a container off by depth and by width *while* rendering,
/// which is the whole point: `repr()` builds the string in full and only then is
/// it cut, so a value whose repr is enormous was paid for in full and thrown
/// away 80 characters later.
static BOUNDED_REPR: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

/// The limits are set well above anything a readable message shows, so a value
/// small enough to print renders exactly as `repr` would and only a value that
/// was going to be cut anyway takes a different path.
fn bounded_repr(py: Python<'_>) -> Option<&Py<PyAny>> {
    BOUNDED_REPR
        .get_or_try_init(py, || {
            let repr = py
                .import("reprlib")?
                .getattr(intern!(py, "Repr"))?
                .call0()?;
            for (limit, value) in [
                ("maxlevel", 12_usize),
                ("maxtuple", 32),
                ("maxlist", 32),
                ("maxarray", 32),
                ("maxdict", 32),
                ("maxset", 32),
                ("maxfrozenset", 32),
                ("maxdeque", 32),
                ("maxstring", SUMMARY_CHARS),
                ("maxlong", SUMMARY_CHARS),
                ("maxother", SUMMARY_CHARS),
            ] {
                repr.setattr(limit, value)?;
            }
            Ok::<_, PyErr>(repr.unbind())
        })
        .ok()
}

/// A short repr-style summary of a value for error messages.
///
/// A container is rendered under a bound rather than rendered and then cut. The
/// difference is not cosmetic: explaining a 20,000-deep list built its 40,000
/// character repr once per level of the walk -- 128 of them, since that is where
/// the recursion bound stops -- and kept 80 characters of each. That was twelve
/// seconds for one error, against `is_valid` at twenty microseconds for the same
/// value, and it grew with the size of the value rather than with the number of
/// mistakes in it.
///
/// A scalar keeps the direct path: its repr is its size, there is nothing to
/// bound, and it is the common case in an error message.
pub(crate) fn summarize(value: &Bound<'_, PyAny>) -> String {
    if !value.is_instance_of::<PyList>()
        && !value.is_instance_of::<PyTuple>()
        && !value.is_instance_of::<PyDict>()
        && !value.is_instance_of::<PySet>()
        && !value.is_instance_of::<PyFrozenSet>()
    {
        return match value.repr() {
            Ok(repr) => shorten(repr.to_string(), SUMMARY_CHARS),
            Err(_) => "<unrepresentable>".to_owned(),
        };
    }
    let rendered = bounded_repr(value.py())
        .and_then(|repr| {
            repr.bind(value.py())
                .call_method1(intern!(value.py(), "repr"), (value,))
                .ok()
        })
        .and_then(|text| text.extract::<String>().ok());
    match rendered {
        Some(text) => shorten(text, SUMMARY_CHARS),
        // `reprlib` is a standard-library module and the call is total, so this
        // is unreachable in practice; falling back to the plain repr keeps the
        // message right rather than trading correctness for the bound.
        None => match value.repr() {
            Ok(repr) => shorten(repr.to_string(), SUMMARY_CHARS),
            Err(_) => "<unrepresentable>".to_owned(),
        },
    }
}

/// Truncate a string this code already owns, keeping it where it is short.
///
/// The short case is every case in practice, and copying it was a second
/// allocation per violation on top of the one the repr already made: a value
/// summary is built for every failure a wide record reports.
pub(crate) fn shorten(text: String, max_chars: usize) -> String {
    // Counting stops at the limit rather than walking a long repr to the end.
    if text.chars().nth(max_chars).is_none() {
        return text;
    }
    let head: String = text.chars().take(max_chars).collect();
    format!("{head}...")
}

pub(crate) fn truncate(text: &str, max_chars: usize) -> String {
    shorten(text.to_owned(), max_chars)
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
    into_pyerr(py, vec![violation])
}

/// Build the Python [`ValidationError`] for one or more violations.
///
/// The raised instance carries the structured, machine-readable error model:
/// `errors` is a tuple of per-failure items, each a JSON-serializable dict with
/// `code`/`path`/`message`/`expected`/`value`, so `json.dumps(err.errors)` is
/// the JSON output mode. The scalar `message`/`code`/`path`/`expected`/`value`
/// mirror the first item; `str(exc)` is a summary of every failure.
/// The failures a raised [`ValidationError`] reports, carried until asked for.
///
/// Populating the six documented attributes cost about four microseconds per
/// raise -- an exception instantiated, six attribute writes, a path tuple and a
/// dict per violation -- and a caller that logs `str(error)` and moves on paid
/// every bit of it. `is_valid` on the same value is 214 nanoseconds.
///
/// So the walk's own `Violation`s are carried here, in Rust, and the attributes
/// are built on the first access that asks for one. `ValidationError` has a
/// `__getattr__` that reads this and caches what it built into the instance, so
/// the second access is an ordinary attribute lookup and nothing is computed
/// twice.
#[pyclass(module = "valgebra", frozen)]
pub(crate) struct Failures {
    violations: Vec<Violation>,
}

impl Failures {
    /// What the first violation says, which five of the six attributes mirror.
    fn first(&self) -> &Violation {
        // The producer never builds one of these without a failure, and every
        // caller degrades rather than panicking, so the boundary does too.
        self.violations.first().unwrap_or(&GENERIC)
    }
}

/// The stand-in for a walk that reported a non-member with no violation, which
/// is an invariant break rather than a state a caller can reach.
static GENERIC: std::sync::LazyLock<Violation> = std::sync::LazyLock::new(|| Violation {
    code: "validation_error",
    path: Vec::new(),
    expected: "a member of the schema's set".to_owned(),
    value_summary: String::new(),
});

/// The attribute names `__getattr__` answers, and nothing else.
const LAZY_ATTRIBUTES: [&str; 6] = ["code", "path", "message", "expected", "value", "errors"];

/// Build one of the six documented attributes from the failures behind it.
///
/// Assigned to the class rather than each instance, so a raised error carries
/// **one** attribute -- the failures -- and the rest are built by the first
/// access that wants them. Python calls this only when ordinary lookup fails,
/// so the value cached below is what every later access finds.
#[pyfunction]
#[pyo3(signature = (instance, name, /))]
pub(crate) fn validation_error_getattr<'py>(
    instance: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let py = instance.py();
    if !LAZY_ATTRIBUTES.contains(&name) {
        return Err(pyo3::exceptions::PyAttributeError::new_err(format!(
            "'{}' object has no attribute '{name}'",
            instance.get_type().name()?
        )));
    }
    // An error built by hand reports no failures, and the model describes
    // *failures*: the empty answer is the honest one rather than an
    // `AttributeError`, which is what the class defaults used to say.
    let Ok(carried) = instance.getattr(intern!(py, "_failures")) else {
        return empty_attribute(py, name);
    };
    let Ok(failures) = carried.cast::<Failures>() else {
        return empty_attribute(py, name);
    };
    let failures = failures.get();
    let first = failures.first();
    let keys = Keys::get(py);
    let built = match name {
        "code" => first.code.into_pyobject(py)?.into_any(),
        "expected" => first.expected.as_str().into_pyobject(py)?.into_any(),
        "value" => first.value_summary.as_str().into_pyobject(py)?.into_any(),
        "message" => first.to_string().into_pyobject(py)?.into_any(),
        "path" => build_path(py, &first.path)?.into_any(),
        _ => {
            let (first_violation, rest) =
                failures.violations.split_first().unwrap_or((&GENERIC, &[]));
            let message = first_violation.to_string();
            let path = build_path(py, &first_violation.path)?;
            error_items(py, &keys, &message, &path, first_violation, rest)?.into_any()
        }
    };
    // Cached on the instance, so this runs once per attribute per error and the
    // next lookup never reaches here.
    instance.setattr(name, &built)?;
    Ok(built)
}

/// What each attribute is for an error carrying no failures.
fn empty_attribute<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    Ok(match name {
        "path" | "errors" => PyTuple::empty(py).into_any(),
        _ => "".into_pyobject(py)?.into_any(),
    })
}

/// Put the two hooks on `ValidationError`, which is what makes the six lazy.
///
/// The attributes exist only where `__getattr__` is on the class, so every
/// entry point that can raise one of these errors installs them: the module
/// initialiser, and a test that reaches [`into_pyerr`] without importing the
/// module. One installer rather than two keeps the class one shape.
///
/// Through a Python `def` rather than the compiled functions directly: a
/// builtin function assigned to a class is not a descriptor, so it never binds
/// `self` and the hook is called one argument short. The lines below are the
/// smallest thing that binds; the work stays in Rust.
pub(crate) fn install_lazy_attributes(py: Python<'_>) -> PyResult<()> {
    let shim = PyModule::from_code(
        py,
        c"def make(read, reduce):
    def __getattr__(self, name):
        return read(self, name)

    def __reduce__(self):
        return reduce(self)

    return __getattr__, __reduce__
",
        c"valgebra/_error_hooks.py",
        c"valgebra._error_hooks",
    )?;
    let hooks = shim
        .getattr("make")?
        .call1((
            wrap_pyfunction!(validation_error_getattr, py)?,
            wrap_pyfunction!(validation_error_reduce, py)?,
        ))?
        .cast_into::<PyTuple>()?;
    let failure = py.get_type::<ValidationError>();
    failure.setattr("__getattr__", hooks.get_item(0)?)?;
    // Pickling and copying carry the instance dictionary, and the failures live
    // there as a Rust object that does not travel; this materialises the six
    // first, so what crosses a process boundary is the plain data the model
    // documents.
    failure.setattr("__reduce__", hooks.get_item(1)?)?;
    Ok(())
}

/// Materialise every attribute before the error is pickled or copied.
///
/// `BaseException.__reduce__` carries the instance dictionary, and the failures
/// live there as a Rust object that does not travel. Forcing the six turns the
/// error into the plain data the model documents -- which is what crosses a
/// process boundary, and what `docs/08-error-model.md` promises pickling gives.
#[pyfunction]
#[pyo3(signature = (instance, /))]
pub(crate) fn validation_error_reduce<'py>(
    instance: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    for name in LAZY_ATTRIBUTES {
        let _ = instance.getattr(name)?;
    }
    let _ = instance.delattr(intern!(instance.py(), "_failures"));
    instance
        .get_type()
        .getattr(intern!(instance.py(), "__mro__"))?;
    let base = instance.py().get_type::<pyo3::exceptions::PyException>();
    base.getattr(intern!(instance.py(), "__reduce__"))?
        .call1((instance,))
}

pub(crate) fn into_pyerr(py: Python<'_>, violations: Vec<Violation>) -> PyErr {
    debug_assert!(!violations.is_empty(), "into_pyerr needs a failure");
    // The caller always reports at least one failure, and the reader below takes
    // the first one. A walk arm that reports a non-member without a violation is
    // an internal invariant break, and every producer of that state degrades
    // rather than panicking, so the boundary does too.
    let (first, rest) = violations.split_first().unwrap_or((&GENERIC, &[]));
    // The message is built here because it is the exception's `args`, which is
    // what `str(error)` reads and what pickling carries. Everything else waits
    // for an access -- see `Failures`.
    let err = ValidationError::new_err(summary_message(&first.to_string(), first, rest));
    // Attaching the failures is one attribute write, and the violations are
    // **moved** into it: cloning them here copies a string per message and a
    // path per violation, which is the work this whole change exists not to do.
    // If the write ever fails, surface that rather than shipping an error whose
    // `.errors` is silently empty while `str(error)` still summarises real
    // failures.
    match attach_failures(py, &err, violations) {
        Ok(()) => err,
        Err(err) => err,
    }
}

fn attach_failures(py: Python<'_>, err: &PyErr, violations: Vec<Violation>) -> PyResult<()> {
    let carried = Bound::new(py, Failures { violations })?;
    err.value(py).setattr(intern!(py, "_failures"), carried)
}

/// The five keys of an error item, and the six attributes of the exception.
///
/// Interned rather than passed as `&str`. A `set_item("code", ..)` builds a
/// fresh Python string for the key on every call, and one failing `validate`
/// writes eleven of them -- five per item plus the six attributes -- which is
/// most of what raising used to cost. `intern!` caches one object per call site
/// for the interpreter's life, so the write is a hash of a string that already
/// knows its hash.
struct Keys<'py> {
    code: &'py Bound<'py, PyString>,
    path: &'py Bound<'py, PyString>,
    message: &'py Bound<'py, PyString>,
    expected: &'py Bound<'py, PyString>,
    value: &'py Bound<'py, PyString>,
}

impl<'py> Keys<'py> {
    fn get(py: Python<'py>) -> Keys<'py> {
        Keys {
            code: intern!(py, "code"),
            path: intern!(py, "path"),
            message: intern!(py, "message"),
            expected: intern!(py, "expected"),
            value: intern!(py, "value"),
        }
    }
}

/// The exception's `str()`: the single message for one failure, or a counted,
/// newline-joined summary for several.
fn summary_message(message: &str, first: &Violation, rest: &[Violation]) -> String {
    if rest.is_empty() {
        return message.to_owned();
    }
    let mut summary = format!("{} validation errors:", rest.len() + 1);
    for violation in core::iter::once(first).chain(rest) {
        summary.push('\n');
        summary.push_str(&violation.to_string());
    }
    summary
}

/// Build the `errors` tuple: one JSON-serializable item per failure, in walk
/// order.
fn error_items<'py>(
    py: Python<'py>,
    keys: &Keys<'py>,
    first_message: &str,
    first_path: &Bound<'py, PyTuple>,
    first: &Violation,
    rest: &[Violation],
) -> PyResult<Bound<'py, PyTuple>> {
    let mut items = Vec::with_capacity(rest.len() + 1);
    for (at, violation) in core::iter::once(first).chain(rest).enumerate() {
        let item = PyDict::new(py);
        item.set_item(keys.code, violation.code)?;
        if at == 0 {
            item.set_item(keys.path, first_path)?;
            item.set_item(keys.message, first_message)?;
        } else {
            item.set_item(keys.path, build_path(py, &violation.path)?)?;
            item.set_item(keys.message, violation.to_string())?;
        }
        item.set_item(keys.expected, violation.expected.as_str())?;
        item.set_item(keys.value, violation.value_summary.as_str())?;
        items.push(item);
    }
    PyTuple::new(py, items)
}

fn build_path<'py>(py: Python<'py>, path: &[PathSegment]) -> PyResult<Bound<'py, PyTuple>> {
    let mut items: Vec<Bound<'py, PyAny>> = Vec::with_capacity(path.len());
    for segment in path {
        let item = match segment {
            PathSegment::Key(key) => key.as_str().into_pyobject(py)?.into_any(),
            PathSegment::IntKey(key) => (*key).into_pyobject(py)?.into_any(),
            // Back to the `int` it came from: a caller indexes with the key, not
            // with its digits.
            PathSegment::BigIntKey(key) => py
                .get_type::<pyo3::types::PyInt>()
                .call1((key.as_str(),))?
                .into_any(),
            PathSegment::Index(index) => (*index).into_pyobject(py)?.into_any(),
        };
        items.push(item);
    }
    PyTuple::new(py, items)
}

// Needs a live interpreter; compiled and run only under the `interpreter-tests`
// feature, which links an embedded Python.
#[cfg(all(test, feature = "interpreter-tests"))]
mod tests;
