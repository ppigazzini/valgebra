//! The Python exception a failed validation raises.
//!
//! A leaf module: it names the exception and nothing else. It lives here rather
//! than in the crate root because the error layer needs it, and a type defined
//! in the aggregator and imported back by one of its members is what puts the
//! two in a cycle. `lib.rs` re-exports it, so no caller spells a new path.

use pyo3::create_exception;
use pyo3::exceptions::PyException;

create_exception!(
    _valgebra,
    ValidationError,
    PyException,
    "Raised when a value is not a member of a schema's set."
);
