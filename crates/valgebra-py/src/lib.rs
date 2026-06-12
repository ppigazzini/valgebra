//! `PyO3` bindings for valgebra: compile a Python schema once into the core IR
//! and walk it entirely in Rust, crossing the boundary once per call.

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyInt;
use valgebra_core::Schema;

/// An immutable compiled validator: a schema IR plus the membership walk over
/// it. Building one compiles the schema; the validator itself never mutates.
#[pyclass(frozen, module = "valgebra._valgebra")]
pub struct CompiledValidator {
    schema: Schema,
}

#[pymethods]
impl CompiledValidator {
    /// Whether `obj` belongs to the schema's set. Check-only: the object is not
    /// copied or coerced.
    fn is_valid(&self, obj: &Bound<'_, PyAny>) -> bool {
        matches_schema(&self.schema, obj)
    }
}

/// Whether `value` is a member of `schema`.
fn matches_schema(schema: &Schema, value: &Bound<'_, PyAny>) -> bool {
    match schema {
        // `isinstance(value, int)`, which holds for `bool` as well.
        Schema::Int => value.is_instance_of::<PyInt>(),
    }
}

/// Compile a Python schema description into the core IR.
fn compile(schema: &Bound<'_, PyAny>) -> PyResult<Schema> {
    let int_type = schema.py().get_type::<PyInt>();
    if schema.is(&int_type) {
        return Ok(Schema::Int);
    }
    Err(PyTypeError::new_err(format!(
        "unsupported schema: {schema:?}"
    )))
}

/// Compile `schema` into an immutable [`CompiledValidator`].
#[pyfunction]
fn validator(schema: &Bound<'_, PyAny>) -> PyResult<CompiledValidator> {
    Ok(CompiledValidator {
        schema: compile(schema)?,
    })
}

/// The `valgebra._valgebra` extension module.
#[pymodule]
fn _valgebra(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<CompiledValidator>()?;
    module.add_function(wrap_pyfunction!(validator, module)?)?;
    Ok(())
}
