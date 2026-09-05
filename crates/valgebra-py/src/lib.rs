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
mod errors;
mod exception;
mod input;
mod render;
mod validator;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use valgebra_core::{DefIx, Guarded, Schema, SeqKind, SeqShape, fresh_self_token};

pub use crate::exception::ValidationError;
pub use crate::validator::Validator;

use crate::validator::{MAX_DEFINITIONS, MAX_SCHEMA_DEPTH, MAX_SCHEMA_NODES, OpenDefinition};

use crate::build::{Pool, build_schema, combine};
use crate::check::{WalkMode, WalkState, member};
use crate::input::Value;

/// A deterministic, binding-level instruction workload for the perf gate.
///
/// The shipped hot path is the membership walk over a live Python value, where
/// the core's deterministic core-only workload does not reach. This runs that walk
/// `iters` times over a fixed record value crossing the boundary at every node
/// kind on the hot path (an int, a str, and a homogeneous int list), returning a
/// checksum so the optimizer cannot discard the work.
///
/// Embedding `CPython` makes the absolute instruction count include a non-fixed
/// interpreter startup, so the gate ([`scripts/perf_gate.py`]) measures the
/// *difference* between two iteration counts: startup is identical in both runs
/// and cancels, leaving the deterministic per-iteration walk cost. This is the
/// budgeted signal, and it also covers the per-node `ctx.fatal.borrow()` tax.
#[doc(hidden)]
#[must_use]
pub fn binding_perf_workload(py: Python<'_>, iters: usize) -> u64 {
    // A homogeneous int list: the walk crosses the boundary at the container and
    // at each element (an `isinstance` check and the per-node `ctx.fatal.borrow()`
    // tax), the most common shape on the hot path.
    let schema = Schema::Seq {
        container: SeqKind::List,
        shape: SeqShape::homogeneous(Schema::Int),
    };
    let validator = Validator::new(schema, Vec::new(), Vec::new());

    // A fixed matching value, built once; the walk visits each list element.
    let items: Vec<i64> = (0..64).collect();
    let obj = PyList::new(py, items)
        .expect("a fresh list of i64 always builds")
        .into_any();

    let mut checksum: u64 = 0;
    for _ in 0..iters {
        let state = WalkState::new();
        // `black_box` the inputs so the optimizer cannot hoist the loop-invariant
        // walk out of the loop: the per-iteration walk is the signal being timed.
        let ok = member(
            std::hint::black_box(&validator.schema),
            &Value::Py(std::hint::black_box(&obj)),
            &mut Vec::new(),
            validator.context(py, &state, WalkMode::Fast),
            &mut Vec::new(),
        );
        checksum = checksum.wrapping_add(u64::from(ok));
    }
    checksum
}

/// Build a recursive schema as a checked fixpoint.
///
/// `builder` receives a placeholder validator standing for the schema being
/// defined and returns its body. The placeholder's self-reference is resolved
/// to a back edge, and a non-contractive body — one whose recursive reference
/// is not under a structural constructor — is rejected.
#[pyfunction]
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
    combine(schemas, Schema::union)
}

/// The intersection of the given schemas: a value in every one of their sets.
#[pyfunction]
#[pyo3(signature = (*schemas))]
fn intersection(schemas: &Bound<'_, PyTuple>) -> PyResult<Validator> {
    combine(schemas, Schema::meet)
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
/// its only lazy state is a `std::sync::OnceLock` holding pure-Rust precompute
/// (no Python objects), whose initialization the standard library serializes.
/// The validation walk keeps its recursion guard in a per-call local, so no two
/// threads share mutable walk state.
#[pymodule(gil_used = false)]
fn _valgebra(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    module.add("ValidationError", py.get_type::<ValidationError>())?;
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
