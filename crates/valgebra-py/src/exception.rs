//! The Python exception a failed validation raises.
//!
//! A leaf module: it names the exception and nothing else. It lives here rather
//! than in the crate root because the error layer needs it, and a type defined
//! in the aggregator and imported back by one of its members is what puts the
//! two in a cycle. `lib.rs` re-exports it, so no caller spells a new path.

use pyo3::create_exception;
use pyo3::exceptions::PyException;

create_exception!(
    // The **public** package, not the extension underneath it.
    //
    // `create_exception!` stringifies this argument into the type's `__module__`
    // and into its qualified name, and `pickle` locates a class by exactly that
    // pair -- so the string is baked into every serialized error and has to name
    // something that keeps resolving. A bare `_valgebra` resolves nowhere at
    // all; `valgebra._valgebra` resolves today, but the API reference reserves
    // the right to rename that module in any release, which would strand data
    // already written. `valgebra` re-exports this name and is the path a user
    // imports it from, so it is the one that stays true.
    valgebra,
    ValidationError,
    PyException,
    "Raised when a value is not a member of a schema's set."
);
