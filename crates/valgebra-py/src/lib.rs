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
mod input;
mod render;

use std::cell::RefCell;
use std::collections::HashSet;

use jiter::{JsonValue, PythonParse};
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyString, PyTuple};
use valgebra_core::{Schema, SeqRegex, fresh_self_token};

use crate::build::{build_schema, combine};
use crate::check::{Ctx, member};
use crate::errors::{into_pyerr, json_invalid_error};
use crate::input::Value;
use crate::render::render;

create_exception!(
    _valgebra,
    ValidationError,
    PyException,
    "Raised when a value is not a member of a schema's set."
);

/// A compiled, immutable schema validator.
///
/// Build one with `validator`, or with a combinator such as `union`,
/// `intersect`, or `lazy`. Then check values with `validate`, `is_valid`, or
/// `cast`, and JSON documents with `validate_json` or `is_valid_json`.
///
/// Validation is a membership test against the set the schema denotes: the value
/// is never copied or coerced. A validator never changes after it is built and
/// is safe to share across threads. Its `repr` is the annotation that produces
/// it, and it can be copied with `copy.copy`/`copy.deepcopy`.
#[pyclass(frozen, module = "valgebra._valgebra")]
pub struct CompiledValidator {
    pub(crate) schema: Schema,
    pub(crate) literals: Vec<Py<PyAny>>,
    pub(crate) definitions: Vec<Schema>,
}

impl CompiledValidator {
    /// The read-only walk context: the pool, the definitions, a fresh recursion
    /// guard, the explain flag, and the fail-fast flag.
    fn context<'a>(
        &'a self,
        guard: &'a RefCell<HashSet<(usize, usize)>>,
        explain: bool,
        fail_fast: bool,
    ) -> Ctx<'a> {
        Ctx {
            pool: &self.literals,
            defs: &self.definitions,
            guard,
            explain,
            fail_fast,
        }
    }

    /// Whether the JSON in `bytes` parses and belongs to the schema's set,
    /// validated in place against the parsed JSON value with no intermediate
    /// Python objects. `bytes` outlives the parsed value and the walk.
    fn matches_json(&self, py: Python<'_>, bytes: &[u8]) -> bool {
        let Ok(json) = JsonValue::parse(bytes, false) else {
            return false;
        };
        let guard = RefCell::new(HashSet::new());
        member(
            &self.schema,
            &Value::Json(py, &json),
            &mut Vec::new(),
            self.context(&guard, false, true),
            &mut Vec::new(),
        )
    }
}

// These doc comments are the Python API reference (rendered by mkdocstrings),
// written in Google docstring style: the `Args:`/`Returns:`/`Raises:` sections
// must name parameters and exceptions as bare identifiers for the reference to
// parse them, which is exactly what clippy's doc_markdown wants backticked.
// Python documentation conventions win here over the Rust-doc lint.
#[allow(clippy::doc_markdown)]
#[pymethods]
impl CompiledValidator {
    /// Validate `obj`, raising `ValidationError` if it is not a member of the
    /// schema's set. Check-only: `obj` is never copied or coerced.
    ///
    /// Args:
    ///     obj: The object to check.
    ///     fail_fast: Stop at the first failure instead of aggregating every
    ///         independent failure into the error.
    ///
    /// Returns:
    ///     `None` if `obj` is a member of the schema's set.
    ///
    /// Raises:
    ///     ValidationError: If `obj` is not a member; its `errors` lists each
    ///         failure with a code and a path.
    #[pyo3(signature = (obj, *, fail_fast = false))]
    fn validate(&self, obj: &Bound<'_, PyAny>, fail_fast: bool) -> PyResult<()> {
        let guard = RefCell::new(HashSet::new());
        let mut path = Vec::new();
        let mut violations = Vec::new();
        let ok = member(
            &self.schema,
            &Value::Py(obj),
            &mut path,
            self.context(&guard, true, fail_fast),
            &mut violations,
        );
        if ok {
            Ok(())
        } else {
            Err(into_pyerr(obj.py(), &violations))
        }
    }

    /// Whether `obj` is a member of the schema's set.
    ///
    /// Check-only and never raises. This is the fast path: it returns as soon as
    /// membership is decided, without building an error.
    ///
    /// Args:
    ///     obj: The object to check.
    ///
    /// Returns:
    ///     `True` if `obj` is a member of the schema's set, else `False`.
    fn is_valid(&self, obj: &Bound<'_, PyAny>) -> bool {
        let guard = RefCell::new(HashSet::new());
        member(
            &self.schema,
            &Value::Py(obj),
            &mut Vec::new(),
            self.context(&guard, false, true),
            &mut Vec::new(),
        )
    }

    /// Validate `obj` and return it unchanged.
    ///
    /// The explicit conversion entry point. Because validation is a membership
    /// check rather than a coercion, the returned object is exactly the input;
    /// `cast` exists so converting code reads distinctly from checking code.
    ///
    /// Args:
    ///     obj: The object to check.
    ///
    /// Returns:
    ///     `obj` unchanged.
    ///
    /// Raises:
    ///     ValidationError: If `obj` is not a member of the schema's set.
    fn cast<'py>(&self, obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.validate(obj, false)?;
        Ok(obj.clone())
    }

    /// Validate a JSON document, parsing it on the Rust path.
    ///
    /// Parsing runs in Rust, faster than the standard library, and the parsed
    /// value runs the same validation walk as a native object, so this reaches
    /// the same decision and the same errors as `validate` on the parsed object.
    /// `fail_fast` behaves as it does for `validate`.
    ///
    /// Args:
    ///     data: The JSON document, as `str` or `bytes`.
    ///     fail_fast: Stop at the first failure instead of aggregating.
    ///
    /// Returns:
    ///     `None` if the document parses and is a member of the schema's set.
    ///
    /// Raises:
    ///     ValidationError: If the document is malformed JSON (code
    ///         `json_invalid`) or is not a member of the schema's set.
    ///     TypeError: If `data` is not `str` or `bytes`.
    #[pyo3(signature = (data, *, fail_fast = false))]
    fn validate_json(&self, data: &Bound<'_, PyAny>, fail_fast: bool) -> PyResult<()> {
        let parsed = parse_json(data)?;
        self.validate(&parsed, fail_fast)
    }

    /// Whether a JSON document parses and is a member of the schema's set.
    ///
    /// Check-only and never raises: malformed JSON, or input that is neither
    /// `str` nor `bytes`, is simply not a member and returns `False`. The
    /// document is validated in place against the parsed value, with no
    /// intermediate Python objects for the structure it walks.
    ///
    /// Args:
    ///     data: The JSON document, as `str` or `bytes`.
    ///
    /// Returns:
    ///     `True` if `data` parses and is a member of the schema's set, else
    ///     `False`.
    fn is_valid_json(&self, data: &Bound<'_, PyAny>) -> bool {
        let py = data.py();
        if let Ok(text) = data.cast::<PyString>() {
            text.to_str()
                .is_ok_and(|json| self.matches_json(py, json.as_bytes()))
        } else if let Ok(raw) = data.cast::<PyBytes>() {
            self.matches_json(py, raw.as_bytes())
        } else {
            false
        }
    }

    /// Whether the schema is unsatisfiable — provably empty, so `is_valid`
    /// returns `False` for every value.
    ///
    /// Decided soundly: `True` only when no value can belong to the schema — an
    /// unsatisfiable intersection, a fixed sequence with an impossible position,
    /// a record with an impossible required field. It never reports a
    /// satisfiable schema as empty; for forms it cannot decide it returns
    /// `False`.
    ///
    /// Returns:
    ///     `True` if the schema denotes the empty set, else `False`.
    fn is_empty(&self) -> bool {
        self.schema.is_empty()
    }

    /// Whether every value of this schema is also a value of `other` — set
    /// inclusion, the subtyping relation.
    ///
    /// `other` is any schema spec or compiled validator. The decision is sound:
    /// `True` only when the inclusion provably holds (`bool` is a subtype of
    /// `int`, `list[bool]` of `list[int]`); for forms it cannot decide — `Or`
    /// sequences, recursive references, class checks across schemas — it returns
    /// `False` rather than a relation it cannot justify.
    ///
    /// Args:
    ///     other: The candidate supertype, as a schema spec or validator.
    ///
    /// Returns:
    ///     `True` if this schema is a subtype of `other`, else `False`.
    fn is_subtype(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let mut literals: Vec<Py<PyAny>> = self.literals.iter().map(|o| o.clone_ref(py)).collect();
        let mut definitions = self.definitions.clone();
        let other = build_schema(other, &mut literals, &mut definitions)?;
        Ok(self.schema.is_subtype(&other))
    }

    /// Whether this schema and `other` denote the same set — mutual inclusion.
    ///
    /// `other` is any schema spec or compiled validator. Sound, like
    /// `is_subtype`: `True` only when the two are provably equivalent, whatever
    /// their syntax (`bool | int` is equivalent to `int`).
    ///
    /// Args:
    ///     other: The schema to compare, as a spec or validator.
    ///
    /// Returns:
    ///     `True` if the two schemas are equivalent, else `False`.
    fn equivalent(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let mut literals: Vec<Py<PyAny>> = self.literals.iter().map(|o| o.clone_ref(py)).collect();
        let mut definitions = self.definitions.clone();
        let other = build_schema(other, &mut literals, &mut definitions)?;
        Ok(self.schema.is_subtype(&other) && other.is_subtype(&self.schema))
    }

    /// Render the compiled schema back as the annotation expression that
    /// produces it.
    fn __repr__(&self, py: Python<'_>) -> String {
        let active = RefCell::new(HashSet::new());
        render(py, &self.schema, &self.literals, &self.definitions, &active)
    }

    /// Return an equivalent validator. The validator is immutable, so the copy
    /// shares the pooled constants, classes, and predicates rather than
    /// duplicating them.
    fn __copy__(&self, py: Python<'_>) -> CompiledValidator {
        CompiledValidator {
            schema: self.schema.clone(),
            literals: self.literals.iter().map(|o| o.clone_ref(py)).collect(),
            definitions: self.definitions.clone(),
        }
    }

    /// Deep-copy to an equivalent validator. Since the validator is immutable,
    /// this shares the pool like `__copy__`; the memo is unused.
    fn __deepcopy__(&self, py: Python<'_>, _memo: &Bound<'_, PyAny>) -> CompiledValidator {
        self.__copy__(py)
    }
}

/// Parse a JSON `str` or `bytes` into a Python value with jiter.
///
/// jiter's defaults match the standard JSON model: standard `float`s, no
/// `Infinity`/`NaN`, and complete (non-partial) input — so the parsed value is
/// what the object path would receive from `json.loads`. A parse failure is
/// surfaced as a structured `json_invalid` `ValidationError`; a non-string,
/// non-bytes argument is a `TypeError`.
fn parse_json<'py>(data: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let py = data.py();
    let parse = PythonParse::default();
    if let Ok(text) = data.cast::<PyString>() {
        let bytes = text.to_str()?;
        parse
            .python_parse(py, bytes.as_bytes())
            .map_err(|err| json_invalid_error(py, &err.description(bytes.as_bytes())))
    } else if let Ok(raw) = data.cast::<PyBytes>() {
        let bytes = raw.as_bytes();
        parse
            .python_parse(py, bytes)
            .map_err(|err| json_invalid_error(py, &err.description(bytes)))
    } else {
        Err(PyTypeError::new_err(
            "JSON input must be a str or bytes object",
        ))
    }
}

/// Compile a schema into a reusable validator.
///
/// The schema is any supported form: a type or typing annotation (`int`,
/// `list[str]`, `int | None`, `Literal[...]`, a `TypedDict`, a dataclass, an
/// `Annotated` refinement, ...), a native form (a `[T]` list, a `{T}` set, a
/// `{K: V}` mapping, an all-string-key dict record, or any constant as a
/// literal), or another compiled validator.
///
/// Args:
///     schema: The schema to compile.
///
/// Returns:
///     An immutable `CompiledValidator` for the schema.
///
/// Raises:
///     NotImplementedError: If the schema uses an unsupported form (for
///         example a recursive class, which must be written with `lazy`).
// Google-style docstring for the Python API reference; see the note on the
// CompiledValidator impl above.
#[allow(clippy::doc_markdown)]
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

/// A fixed-length list matched positionally: element `i` must satisfy the `i`th
/// member schema, and the list length must equal the number of members.
#[pyfunction]
#[pyo3(signature = (*members))]
fn fixed_sequence(members: &Bound<'_, PyTuple>) -> PyResult<CompiledValidator> {
    combine(members, |elements| Schema::list(SeqRegex::fixed(elements)))
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

/// Open every record in a schema: undeclared keys are admitted throughout.
///
/// Returns a new validator; the original is unchanged.
#[pyfunction]
fn lax(validator: &CompiledValidator, py: Python<'_>) -> CompiledValidator {
    with_records_open(validator, true, py)
}

/// Close every record in a schema: only declared keys are admitted throughout.
///
/// The inverse of `lax`. Returns a new validator; the original is unchanged.
#[pyfunction]
fn strict(validator: &CompiledValidator, py: Python<'_>) -> CompiledValidator {
    with_records_open(validator, false, py)
}

fn with_records_open(
    validator: &CompiledValidator,
    open: bool,
    py: Python<'_>,
) -> CompiledValidator {
    CompiledValidator {
        schema: validator.schema.with_records_open(open),
        literals: validator.literals.iter().map(|o| o.clone_ref(py)).collect(),
        definitions: validator.definitions.clone(),
    }
}

/// An equivalent validator reduced by the lattice laws.
///
/// The result admits exactly the same values in a simpler form (flattened and
/// deduplicated unions and intersections, identities applied, complements in
/// negation-normal form). Returns a new validator; the original is unchanged.
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
    module.add_function(wrap_pyfunction!(fixed_sequence, module)?)?;
    module.add_function(wrap_pyfunction!(simplify, module)?)?;
    module.add_function(wrap_pyfunction!(lax, module)?)?;
    module.add_function(wrap_pyfunction!(strict, module)?)?;
    module.add_function(wrap_pyfunction!(lazy, module)?)?;
    // The lattice bounds: top admits every value, bottom admits none.
    module.add("anything", atom(py, Schema::Anything)?)?;
    module.add("nothing", atom(py, Schema::Nothing)?)?;
    Ok(())
}
