//! `PyO3` bindings for valgebra: compile a Python schema once into the core IR
//! and walk it entirely in Rust, crossing the boundary once per call.
//!
//! The crate is split into the frontend ([`build`]) that reads Python forms
//! into the IR, the walk ([`check`]) with its explain path and membership fast
//! path, the [`render`] back to an annotation string, and [`errors`] that build
//! the Python [`ValidationError`].

mod build;
mod check;
mod errors;
mod render;

use std::cell::RefCell;
use std::collections::HashSet;

use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use valgebra_core::{Schema, fresh_self_token};

use crate::build::{build_schema, combine};
use crate::check::{Ctx, check, matches};
use crate::errors::into_pyerr;
use crate::render::render;

create_exception!(
    _valgebra,
    ValidationError,
    PyException,
    "Raised when a value is not a member of a schema's set."
);

/// An immutable compiled validator: a schema IR, the constants pool its
/// literals index into, and the recursive definitions its `Ref`s resolve
/// against. Building one compiles the schema; the validator itself never
/// mutates.
#[pyclass(frozen, module = "valgebra._valgebra")]
pub struct CompiledValidator {
    pub(crate) schema: Schema,
    pub(crate) literals: Vec<Py<PyAny>>,
    pub(crate) definitions: Vec<Schema>,
}

impl CompiledValidator {
    /// The read-only walk context: the pool, the definitions, and a fresh
    /// recursion guard.
    fn context<'a>(&'a self, guard: &'a RefCell<HashSet<(usize, usize)>>) -> Ctx<'a> {
        Ctx {
            pool: &self.literals,
            defs: &self.definitions,
            guard,
        }
    }
}

#[pymethods]
impl CompiledValidator {
    /// Raise [`ValidationError`] if `obj` is not a member of the schema's set;
    /// return `None` otherwise. Check-only: the object is not copied or coerced.
    fn validate(&self, obj: &Bound<'_, PyAny>) -> PyResult<()> {
        let guard = RefCell::new(HashSet::new());
        let mut path = Vec::new();
        match check(&self.schema, obj, &mut path, self.context(&guard)) {
            Some(violation) => Err(into_pyerr(obj.py(), &violation)),
            None => Ok(()),
        }
    }

    /// Whether `obj` belongs to the schema's set. Check-only, returns a bool via
    /// the membership fast path.
    fn is_valid(&self, obj: &Bound<'_, PyAny>) -> bool {
        let guard = RefCell::new(HashSet::new());
        matches(&self.schema, obj, self.context(&guard))
    }

    /// Validate `obj` and return it unchanged. The explicit conversion mode:
    /// validation is a membership check, so the returned object is the input.
    fn cast<'py>(&self, obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.validate(obj)?;
        Ok(obj.clone())
    }

    /// Render the compiled schema back as the annotation expression that
    /// produces it.
    fn __repr__(&self, py: Python<'_>) -> String {
        let active = RefCell::new(HashSet::new());
        render(py, &self.schema, &self.literals, &self.definitions, &active)
    }
}

/// Compile `schema` into an immutable [`CompiledValidator`].
#[pyfunction]
fn validator(schema: &Bound<'_, PyAny>) -> PyResult<CompiledValidator> {
    let mut literals = Vec::new();
    let mut definitions = Vec::new();
    let schema = build_schema(schema, &mut literals, &mut definitions)?;
    Ok(CompiledValidator {
        schema,
        literals,
        definitions,
    })
}

/// Build a recursive schema as a checked fixpoint.
///
/// `builder` receives a placeholder validator standing for the schema being
/// defined and returns its body. The placeholder's self-reference is resolved
/// to a back edge, and a non-contractive body — one whose recursive reference
/// is not under a structural constructor — is rejected.
#[pyfunction]
fn lazy(builder: &Bound<'_, PyAny>) -> PyResult<CompiledValidator> {
    let py = builder.py();
    let token = fresh_self_token();
    let placeholder = Py::new(
        py,
        CompiledValidator {
            schema: Schema::SelfRef(token),
            literals: Vec::new(),
            definitions: Vec::new(),
        },
    )?;
    let body_obj = builder.call1((placeholder,))?;
    let mut literals = Vec::new();
    let mut definitions = Vec::new();
    let body = build_schema(&body_obj, &mut literals, &mut definitions)?;
    // The body becomes a definition; the self-reference resolves to it.
    let ref_id = definitions.len();
    let resolved = body.resolve_self(token, ref_id);
    if resolved.occurs_unguarded(ref_id, false) {
        return Err(PyValueError::new_err(
            "lazy schema is not contractive: the recursive reference must occur \
             under a structural constructor (a list, tuple, set, dict, record, \
             or object)",
        ));
    }
    definitions.push(resolved);
    Ok(CompiledValidator {
        schema: Schema::Ref(ref_id),
        literals,
        definitions,
    })
}

/// The union of the given schemas: a value in at least one of their sets.
#[pyfunction]
#[pyo3(signature = (*schemas))]
fn union(schemas: &Bound<'_, PyTuple>) -> PyResult<CompiledValidator> {
    combine(schemas, Schema::Union)
}

/// The intersection of the given schemas: a value in every one of their sets.
#[pyfunction]
#[pyo3(signature = (*schemas))]
fn intersect(schemas: &Bound<'_, PyTuple>) -> PyResult<CompiledValidator> {
    combine(schemas, Schema::Intersection)
}

/// The complement of a schema: every value not in its set.
#[pyfunction]
fn complement(schema: &Bound<'_, PyAny>) -> PyResult<CompiledValidator> {
    let mut literals = Vec::new();
    let mut definitions = Vec::new();
    let inner = build_schema(schema, &mut literals, &mut definitions)?;
    Ok(CompiledValidator {
        schema: Schema::Complement(Box::new(inner)),
        literals,
        definitions,
    })
}

/// An equivalent validator reduced by the lattice laws: it admits exactly the
/// same values, in a simpler form. The pool and definitions are shared
/// unchanged, as simplification only rewrites the schema's structure.
#[pyfunction]
fn simplify(validator: &CompiledValidator, py: Python<'_>) -> CompiledValidator {
    CompiledValidator {
        schema: validator.schema.simplify(),
        literals: validator.literals.iter().map(|o| o.clone_ref(py)).collect(),
        definitions: validator.definitions.clone(),
    }
}

/// A pool-free validator wrapping a single atom (the `anything`/`nothing`
/// lattice bounds).
fn atom(py: Python<'_>, schema: Schema) -> PyResult<Py<CompiledValidator>> {
    Py::new(
        py,
        CompiledValidator {
            schema,
            literals: Vec::new(),
            definitions: Vec::new(),
        },
    )
}

/// The `valgebra._valgebra` extension module.
#[pymodule]
fn _valgebra(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    module.add("ValidationError", py.get_type::<ValidationError>())?;
    module.add_class::<CompiledValidator>()?;
    module.add_function(wrap_pyfunction!(validator, module)?)?;
    module.add_function(wrap_pyfunction!(union, module)?)?;
    module.add_function(wrap_pyfunction!(intersect, module)?)?;
    module.add_function(wrap_pyfunction!(complement, module)?)?;
    module.add_function(wrap_pyfunction!(simplify, module)?)?;
    module.add_function(wrap_pyfunction!(lazy, module)?)?;
    // The lattice bounds: top admits every value, bottom admits none.
    module.add("anything", atom(py, Schema::Anything)?)?;
    module.add("nothing", atom(py, Schema::Nothing)?)?;
    Ok(())
}
