//! `PyO3` bindings for valgebra: compile a Python schema once into the core IR
//! and walk it in Rust — tree walks, key lookups, and bound checks stay in the
//! validator tree; a comparison against a Python object (a literal, a refinement
//! predicate, an instance or attribute check) is the documented step back across
//! the boundary.
//!
//! The crate is split into the frontend (`build`) that reads Python forms into
//! the IR, the walk (`check`) with its explain path and membership fast path,
//! the `render` back to an annotation string, and `errors` that build the
//! Python `ValidationError`.
//!
//! The crate forbids `unsafe`, so the security policy's no-unsafe guarantee is
//! compiler-enforced across the binding boundary too, not merely asserted.
#![forbid(unsafe_code)]

mod build;
mod check;
mod equality;
mod errors;
mod exception;
mod input;
mod oracle;
mod render;
mod validator;
pub mod workload;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyModule, PyTuple};
use valgebra_core::{DefIx, Guarded, Schema, fresh_self_token};

use crate::errors::install_lazy_attributes;
pub use crate::exception::ValidationError;
pub use crate::validator::Validator;

use crate::validator::{MAX_DEFINITIONS, MAX_SCHEMA_DEPTH, MAX_SCHEMA_NODES, OpenDefinition};

use crate::build::{Pool, build_schema, combine};

/// Build a recursive schema as a checked fixpoint.
///
/// `builder` receives a placeholder validator standing for the schema being
/// defined and returns its body. The placeholder's self-reference is resolved
/// to a back edge, and a non-contractive body — one whose recursive reference
/// is not under a structural constructor — is rejected.
#[pyfunction]
#[pyo3(signature = (builder, /))]
fn recursive(builder: &Bound<'_, PyAny>) -> PyResult<Validator> {
    let py = builder.py();
    let token = fresh_self_token();
    let placeholder = Py::new(
        py,
        Validator::new(Schema::SelfRef(token), Vec::new(), Vec::new()),
    )?;
    // The placeholder is meaningful for the length of the builder call: the
    // schemas the caller composes inside it carry the marker, and construction
    // has to tell those from a placeholder that outlives the call.
    let body_obj = {
        let _open = OpenDefinition::open(token);
        builder.call1((placeholder,))?
    };
    let mut literals = Pool::default();
    let mut definitions = Vec::new();
    let body = build_schema(&body_obj, &mut literals, &mut definitions)?;
    // The body becomes a definition; the self-reference resolves to it.
    let ref_id = DefIx::new(definitions.len());
    // The marker is resolved wherever the build put it. A `recursive` inside the
    // body compiles to a definition of its own, and that definition may name
    // *this* fixpoint, so the body is not the only place the marker lands.
    for definition in &mut definitions {
        *definition = definition.resolve_self(token, ref_id);
    }
    let resolved = body.resolve_self(token, ref_id);
    // Contractivity is a property of the whole system of definitions: an inner
    // fixpoint that names this one puts the occurrence behind a `Ref`, which a
    // walk over the body alone reads as a leaf. The definitions the body's build
    // appended are the graph; this definition is not among them and needs not be,
    // since reaching it is the answer rather than a step.
    if resolved.occurs_unguarded_under(ref_id, Guarded::No, &definitions) {
        return Err(PyValueError::new_err(
            "recursive schema is not contractive: the recursive reference must \
             occur under a structural constructor (a list, tuple, set, dict, \
             record, or object)",
        ));
    }
    definitions.push(resolved);
    Validator::checked(Schema::Ref(ref_id), literals.into_items(), definitions)
}

/// The union of the given schemas: a value in at least one of their sets.
#[pyfunction]
#[pyo3(signature = (*schemas))]
fn union(schemas: &Bound<'_, PyTuple>) -> PyResult<Validator> {
    combine(schemas, Schema::union_within)
}

/// The intersection of the given schemas: a value in every one of their sets.
#[pyfunction]
#[pyo3(signature = (*schemas))]
fn intersection(schemas: &Bound<'_, PyTuple>) -> PyResult<Validator> {
    combine(schemas, Schema::meet_within)
}

/// The complement of a schema: every value not in its set.
///
/// Membership is decided by the inner schema: a value belongs to the complement
/// exactly when it is not a member of the inner. When deciding the inner raises
/// an ordinary Python exception — a value whose comparison or `__eq__` throws —
/// that value folds to a non-member of the inner, and therefore a **member** of
/// the complement. A filter of the form `complement(P)` over values whose own
/// methods can raise should not rely on the complement alone to exclude them;
/// intersect with a positive type that pins the shape instead.
#[pyfunction]
#[pyo3(signature = (schema, /))]
fn complement(schema: &Bound<'_, PyAny>) -> PyResult<Validator> {
    let mut literals = Pool::default();
    let mut definitions = Vec::new();
    let inner = build_schema(schema, &mut literals, &mut definitions)?;
    Validator::checked(inner.complement(), literals.into_items(), definitions)
}

/// A pool-free validator wrapping a single atom (the `anything`/`nothing`
/// lattice bounds).
fn atom(py: Python<'_>, schema: Schema) -> PyResult<Py<Validator>> {
    Py::new(py, Validator::new(schema, Vec::new(), Vec::new()))
}

/// The `valgebra._valgebra` extension module.
///
/// `gil_used = false` declares the module free-threading-ready, so a
/// free-threaded interpreter keeps the global interpreter lock disabled on
/// import instead of re-enabling it. This is sound because every shared surface
/// is immutable or internally synchronized: a `Validator` is `frozen`, its
/// schema, constants pool, and definitions never change after construction, and
/// its only lazy state is a `std::sync::OnceLock` holding precompute whose
/// Python objects are interned `str` keys -- immutable, shared, and reachable
/// through `__traverse__` -- whose initialization the standard library
/// serializes.
/// The validation walk keeps its recursion guard in a per-call local, so no two
/// threads share mutable walk state.
#[pymodule(gil_used = false)]
fn _valgebra(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    // The distribution version, from the crate the wheel is built from.
    //
    // `importlib.metadata.version()` answers the same question and costs 20 ms
    // of the 32 ms `import valgebra` took: it pulls `email`, `zipfile`,
    // `inspect` and the compression modules to read a file that says what
    // `Cargo.toml` already said. `maturin` builds the wheel from that manifest,
    // so the two cannot disagree -- and `tests/test_version.py` holds them to
    // each other rather than trusting that.
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    // Whether this extension was built with debug assertions on, which is what
    // separates a `maturin develop` build from a `--release` or PGO one.
    //
    // A timing harness has to know: a figure taken from a debug build is an
    // order of magnitude off and looks exactly like a regression, and guessing
    // the build from the file's size does not separate them -- the debug
    // extension is eight megabytes against the release three, close enough that
    // a threshold between the two is a coin toss on the next toolchain. The
    // build knows what it is, so it says so rather than being inferred.
    module.add("_debug_build", cfg!(debug_assertions))?;
    let failure = py.get_type::<ValidationError>();
    // The six documented attributes are built on the access that asks for one,
    // from the failures a raised error carries. `__getattr__` runs only when
    // ordinary lookup fails, and it caches what it built on the instance, so the
    // second access is a plain attribute read.
    //
    // The type carries no class defaults, and must not: a default makes ordinary
    // lookup *succeed*, so the hook would never run. The hook is what answers
    // for a hand-built error, which keeps the type one shape rather than two.
    install_lazy_attributes(py)?;
    module.add("ValidationError", failure)?;
    module.add_class::<Validator>()?;
    module.add_function(wrap_pyfunction!(union, module)?)?;
    module.add_function(wrap_pyfunction!(intersection, module)?)?;
    module.add_function(wrap_pyfunction!(complement, module)?)?;
    module.add_function(wrap_pyfunction!(recursive, module)?)?;
    // The lattice bounds: top admits every value, bottom admits none.
    module.add("anything", atom(py, Schema::ANYTHING)?)?;
    module.add("nothing", atom(py, Schema::Nothing)?)?;
    // The construction bounds, published so a caller can size its schemas and a
    // test can assert rejection at the exact edge rather than a hard-coded guess.
    module.add("MAX_SCHEMA_DEPTH", MAX_SCHEMA_DEPTH)?;
    module.add("MAX_DEFINITIONS", MAX_DEFINITIONS)?;
    module.add("MAX_SCHEMA_NODES", MAX_SCHEMA_NODES)?;
    Ok(())
}
