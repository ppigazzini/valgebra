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
mod render;
mod validator;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use valgebra_core::{
    DefIx, Field, Guarded, MapClause, Schema, SeqKind, SeqShape, fresh_self_token,
};

use crate::errors::install_lazy_attributes;
pub use crate::exception::ValidationError;
pub use crate::validator::Validator;

use crate::validator::{MAX_DEFINITIONS, MAX_SCHEMA_DEPTH, MAX_SCHEMA_NODES, OpenDefinition};

use crate::build::{Pool, build_schema, combine};
use crate::check::{Frame, WalkMode, WalkState, member};
use crate::input::Value;

/// Which deterministic workload to run.
///
/// The comparison gate -- `scripts/compare_gate.py` -- measures seven shapes
/// against pydantic-core on a wall clock; the instruction gate measured one of
/// them. The gap is how a shape regresses without a gate saying so: schema
/// construction grew twelve percent over one release cycle and nothing caught
/// it, because no deterministic workload built a schema.
///
/// Each variant below is the deterministic twin of a comparison shape, so a
/// wall-clock movement can be confirmed or refuted by an instruction count on
/// the same work. They are *not* the same code as the comparison shapes and are
/// not meant to be: the gate compares against another library and has to run
/// what that library can also run, while these run the thing being budgeted.
// A gate's own hook, like the workload it selects: not part of the extension's
// surface, and not in the crate's documentation.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BindingShape {
    /// The membership walk over a homogeneous list of sixty-four integers: the
    /// original workload, and the twin of `large_array`.
    Walk,
    /// The walk over a single integer: the floor every other shape stands on,
    /// and the nearest deterministic twin of `scalar`.
    ///
    /// Not the FFI crossing itself. This runs inside one interpreter attachment
    /// and calls the walk directly, so what it counts is the dispatch and the
    /// type check with no container around them -- the part of `scalar` this
    /// crate owns. The crossing is `PyO3`'s and is measured by the comparison
    /// gate's wall clock, where it belongs.
    Boundary,
    /// The accepting walk over a fifty-field record: the twin of `wide_record`,
    /// and the path that resolves each key through the plan built with the
    /// validator rather than through the walk.
    Record,
    /// Building a fifty-field record schema and its validator: the twin of
    /// `build`.
    ///
    /// The core's construction, not the frontend's: it assembles the fields
    /// directly rather than reading a Python annotation, so it counts the
    /// canonical form being imposed -- fields ordered, clauses deduplicated --
    /// and the validator's own index, without the annotation walk in front of
    /// them. That is the half of `build` this crate can change.
    Build,
    /// The explaining walk over a fifty-field record with one bad field, read
    /// to the end: the twin of `error_report`, and the only shape here that
    /// builds violations rather than answering a bool.
    Explain,
    /// The accepting walk over the same fifty fields declared by a record that
    /// is **open** the way a `TypedDict` is: its clause admits any further
    /// `str` key. The shape the record walk takes for the annotation users
    /// write most, and the one the closed twin above could not see -- an open
    /// record was scanned key by key where a closed one was read by its keys,
    /// and nothing counted the difference.
    Open,
}

impl BindingShape {
    /// The name the gate passes on the command line.
    #[must_use]
    pub fn named(name: &str) -> Option<BindingShape> {
        Some(match name {
            "walk" => BindingShape::Walk,
            "boundary" => BindingShape::Boundary,
            "record" => BindingShape::Record,
            "build" => BindingShape::Build,
            "explain" => BindingShape::Explain,
            "open" => BindingShape::Open,
            _ => return None,
        })
    }
}

/// The fifty-field record both record shapes use, as a schema and as a value.
///
/// Fifty fields is the comparison gate's width, kept identical so the two
/// measurements are of the same size of problem.
fn wide_record(py: Python<'_>) -> (Schema, Py<PyAny>) {
    let (fields, value) = wide_fields(py);
    (Schema::keyed_map(fields, Vec::new()), value)
}

/// The same fifty fields and value under the clause a `TypedDict` carries.
fn open_record(py: Python<'_>) -> (Schema, Py<PyAny>) {
    let (fields, value) = wide_fields(py);
    let any_str_key = MapClause {
        key: Schema::Str,
        value: Schema::ANYTHING,
    };
    (Schema::keyed_map(fields, vec![any_str_key]), value)
}

fn wide_fields(py: Python<'_>) -> (Vec<Field>, Py<PyAny>) {
    let fields: Vec<Field> = (0..50)
        .map(|i| Field {
            name: format!("f{i}").into(),
            schema: Schema::Int,
            required: true,
        })
        .collect();
    let value = PyDict::new(py);
    for i in 0..50 {
        value
            .set_item(format!("f{i}"), i)
            .expect("a fresh dict of small ints always builds");
    }
    (fields, value.into_any().unbind())
}

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
pub fn binding_perf_workload_shape(py: Python<'_>, shape: BindingShape, iters: usize) -> u64 {
    match shape {
        BindingShape::Walk => binding_perf_workload(py, iters),
        BindingShape::Boundary => {
            let validator = Validator::new(Schema::Int, Vec::new(), Vec::new());
            let obj = 42_i64
                .into_pyobject(py)
                .expect("an i64 always converts")
                .into_any();
            let mut checksum: u64 = 0;
            for _ in 0..iters {
                let state = WalkState::new();
                let ok = member(
                    std::hint::black_box(&validator.schema),
                    &Value::Py(std::hint::black_box(&obj)),
                    &mut Frame::new(
                        &mut Vec::new(),
                        &mut Vec::new(),
                        validator.context(py, &state, WalkMode::Fast),
                    ),
                );
                checksum = checksum.wrapping_add(u64::from(ok));
            }
            checksum
        }
        BindingShape::Record | BindingShape::Open => {
            let (schema, value) = if matches!(shape, BindingShape::Record) {
                wide_record(py)
            } else {
                open_record(py)
            };
            let validator = Validator::new(schema, Vec::new(), Vec::new());
            let obj = value.bind(py).clone();
            let mut checksum: u64 = 0;
            for _ in 0..iters {
                let state = WalkState::new();
                let ok = member(
                    std::hint::black_box(&validator.schema),
                    &Value::Py(std::hint::black_box(&obj)),
                    &mut Frame::new(
                        &mut Vec::new(),
                        &mut Vec::new(),
                        validator.context(py, &state, WalkMode::Fast),
                    ),
                );
                checksum = checksum.wrapping_add(u64::from(ok));
            }
            checksum
        }
        BindingShape::Build => {
            let mut checksum: u64 = 0;
            for _ in 0..iters {
                let (schema, _) = wide_record(py);
                let validator =
                    Validator::new(std::hint::black_box(schema), Vec::new(), Vec::new());
                checksum = checksum.wrapping_add(validator.schema.node_count() as u64);
            }
            checksum
        }
        BindingShape::Explain => {
            let (schema, value) = wide_record(py);
            let validator = Validator::new(schema, Vec::new(), Vec::new());
            let obj = value.bind(py).clone();
            obj.cast::<PyDict>()
                .expect("the record value is a dict")
                .set_item("f37", "not an int")
                .expect("replacing one key always succeeds");
            let mut checksum: u64 = 0;
            for _ in 0..iters {
                let state = WalkState::new();
                let mut out = Vec::new();
                let ok = member(
                    std::hint::black_box(&validator.schema),
                    &Value::Py(std::hint::black_box(&obj)),
                    &mut Frame::new(
                        &mut Vec::new(),
                        &mut out,
                        validator.context(py, &state, WalkMode::Explain),
                    ),
                );
                checksum = checksum
                    .wrapping_add(u64::from(ok))
                    .wrapping_add(out.len() as u64);
            }
            checksum
        }
    }
}

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
            &mut Frame::new(
                &mut Vec::new(),
                &mut Vec::new(),
                validator.context(py, &state, WalkMode::Fast),
            ),
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
/// its only lazy state is a `std::sync::OnceLock` holding pure-Rust precompute
/// (no Python objects), whose initialization the standard library serializes.
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
    // There are no class defaults any more, and there must not be: a default
    // makes ordinary lookup *succeed*, so the hook would never run. The empty
    // answers a hand-built error used to get from those defaults come from the
    // hook instead, which keeps the type one shape rather than two.
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
