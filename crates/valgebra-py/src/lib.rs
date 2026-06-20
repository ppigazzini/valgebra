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
use std::sync::OnceLock;

use jiter::{JsonValue, PythonParse};
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyString, PyTuple, PyType};
use rustc_hash::FxHashSet;
use valgebra_core::{LeafRelations, Schema, fresh_self_token};

use crate::build::{build_schema, combine};
use crate::check::{Ctx, ValidatorIndex, build_index, member};
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
/// Build one by calling `Validator(schema)`, or with a combinator such as
/// `union`, `intersection`, or `recursive`. Then check values with `validate`,
/// `is_valid`, or `ensure`, and JSON documents with `validate_json` or
/// `is_valid_json`.
///
/// Validation is a membership test against the set the schema denotes: the value
/// is never copied or coerced. A validator never changes after it is built and
/// is safe to share across threads. Its `repr` is the annotation that produces
/// it, and it can be copied with `copy.copy`/`copy.deepcopy`.
#[pyclass(frozen, module = "valgebra._valgebra")]
pub struct Validator {
    pub(crate) schema: Schema,
    pub(crate) literals: Vec<Py<PyAny>>,
    pub(crate) definitions: Vec<Schema>,
    /// Per-node precompute (record-field lookups and literal-union decision
    /// tables), built once on first use from this validator's own schema and
    /// reused across calls. Lazy so an unused validator never pays for it, and
    /// rebuilt per validator (a copy starts empty) so its buffer-address keys
    /// always refer to this schema's nodes.
    index: OnceLock<ValidatorIndex>,
}

impl Validator {
    /// Assemble a validator from its parts, deferring the precompute to first
    /// use. Every construction path goes through here so the index is never
    /// copied between validators.
    pub(crate) fn new(schema: Schema, literals: Vec<Py<PyAny>>, definitions: Vec<Schema>) -> Self {
        Validator {
            schema,
            literals,
            definitions,
            index: OnceLock::new(),
        }
    }

    /// The precompute, built once from this validator's schema, definitions, and
    /// constants pool.
    fn index(&self, py: Python<'_>) -> &ValidatorIndex {
        self.index
            .get_or_init(|| build_index(py, &self.schema, &self.definitions, &self.literals))
    }

    /// The read-only walk context: the pool, the definitions, the precomputed
    /// record and union indexes, a fresh recursion guard, the explain flag, and
    /// the fail-fast flag.
    fn context<'a>(
        &'a self,
        py: Python<'_>,
        guard: &'a RefCell<FxHashSet<(usize, usize)>>,
        fatal: &'a RefCell<Option<PyErr>>,
        explain: bool,
        fail_fast: bool,
    ) -> Ctx<'a> {
        let index = self.index(py);
        Ctx {
            pool: &self.literals,
            defs: &self.definitions,
            records: &index.records,
            unions: &index.unions,
            regexes: &index.regexes,
            guard,
            fatal,
            explain,
            fail_fast,
        }
    }

    /// Union this schema with `other` (a spec or validator), placing `other`
    /// first when it is the `|` right operand. Backs `__or__`/`__ror__`: the
    /// fresh pool seeds with this validator's constants so its schema indices
    /// stay valid, then `other` interns into it.
    fn union_with(&self, other: &Bound<'_, PyAny>, other_first: bool) -> PyResult<Validator> {
        let py = other.py();
        let mut literals: Vec<Py<PyAny>> = self.literals.iter().map(|o| o.clone_ref(py)).collect();
        let mut definitions = self.definitions.clone();
        let other_schema = build_schema(other, &mut literals, &mut definitions)?;
        let members = if other_first {
            vec![other_schema, self.schema.clone()]
        } else {
            vec![self.schema.clone(), other_schema]
        };
        Ok(Validator::new(
            Schema::Union(members),
            literals,
            definitions,
        ))
    }

    /// Whether the JSON in `bytes` parses and belongs to the schema's set,
    /// validated in place against the parsed JSON value with no intermediate
    /// Python objects. `bytes` outlives the parsed value and the walk.
    fn matches_json(&self, py: Python<'_>, bytes: &[u8]) -> PyResult<bool> {
        let Ok(json) = JsonValue::parse(bytes, false) else {
            return Ok(false);
        };
        let guard = RefCell::new(FxHashSet::default());
        let fatal = RefCell::new(None);
        let ok = member(
            &self.schema,
            &Value::Json(py, &json),
            &mut Vec::new(),
            self.context(py, &guard, &fatal, false, true),
            &mut Vec::new(),
        );
        reraise_fatal(fatal, ok)
    }
}

// These doc comments are the Python API reference (rendered by mkdocstrings),
// written in Google docstring style: the `Args:`/`Returns:`/`Raises:` sections
// must name parameters and exceptions as bare identifiers for the reference to
// parse them, which is exactly what clippy's doc_markdown wants backticked.
// Python documentation conventions win here over the Rust-doc lint.
#[allow(clippy::doc_markdown)]
#[pymethods]
impl Validator {
    /// Compile a schema into a reusable, immutable validator.
    ///
    /// The schema is any supported form: a type or typing annotation (`int`,
    /// `list[str]`, `int | None`, `Literal[...]`, a `TypedDict`, a dataclass, an
    /// `Annotated` refinement, ...), a native form (a `[T]` list, a `{T}` set, a
    /// `{K: V}` mapping, an all-string-key dict record, or any constant as a
    /// literal), or another `Validator`.
    ///
    /// Args:
    ///     schema: The schema to compile.
    ///
    /// Raises:
    ///     NotImplementedError: If the schema uses an unsupported form (for
    ///         example a recursive class, which must be written with `recursive`).
    #[new]
    fn py_new(schema: &Bound<'_, PyAny>) -> PyResult<Validator> {
        let mut literals = Vec::new();
        let mut definitions = Vec::new();
        let schema = build_schema(schema, &mut literals, &mut definitions)?;
        Ok(Validator::new(schema, literals, definitions))
    }

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
        let guard = RefCell::new(FxHashSet::default());
        let fatal = RefCell::new(None);
        let mut path = Vec::new();
        let mut violations = Vec::new();
        let ok = member(
            &self.schema,
            &Value::Py(obj),
            &mut path,
            self.context(obj.py(), &guard, &fatal, true, fail_fast),
            &mut violations,
        );
        if let Some(err) = fatal.into_inner() {
            return Err(err);
        }
        if ok {
            Ok(())
        } else {
            Err(into_pyerr(obj.py(), &violations))
        }
    }

    /// Whether `obj` is a member of the schema's set.
    ///
    /// Check-only: it does not build an error and returns as soon as membership
    /// is decided. It raises only if a comparison the membership test performs
    /// raises a fatal interpreter signal (for example a KeyboardInterrupt during
    /// a long check); an ordinary exception in a comparison is folded to a
    /// non-member, as the membership contract requires.
    ///
    /// Args:
    ///     obj: The object to check.
    ///
    /// Returns:
    ///     `True` if `obj` is a member of the schema's set, else `False`.
    ///
    /// Raises:
    ///     BaseException: If a membership comparison raises a fatal interpreter
    ///         signal, it propagates rather than being read as a non-member.
    fn is_valid(&self, obj: &Bound<'_, PyAny>) -> PyResult<bool> {
        let guard = RefCell::new(FxHashSet::default());
        let fatal = RefCell::new(None);
        let ok = member(
            &self.schema,
            &Value::Py(obj),
            &mut Vec::new(),
            self.context(obj.py(), &guard, &fatal, false, true),
            &mut Vec::new(),
        );
        reraise_fatal(fatal, ok)
    }

    /// Validate `obj` and return it unchanged.
    ///
    /// The value-returning check. Because validation is a membership test rather
    /// than a coercion, the returned object is exactly the input; `ensure` exists
    /// so code that wants the checked value back reads distinctly from the
    /// boolean `is_valid`.
    ///
    /// Args:
    ///     obj: The object to check.
    ///
    /// Returns:
    ///     `obj` unchanged.
    ///
    /// Raises:
    ///     ValidationError: If `obj` is not a member of the schema's set.
    fn ensure<'py>(&self, obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
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

    /// Validate a JSON document and return the parsed value.
    ///
    /// Like `validate_json`, but returns the parsed Python object instead of
    /// discarding it, so a caller that needs the data does not parse it again.
    /// Parsing runs in Rust (jiter), and the parsed value is validated by the
    /// same walk, reaching the same decision and errors as `validate`.
    ///
    /// Args:
    ///     data: The JSON document, as `str` or `bytes`.
    ///     fail_fast: Stop at the first failure instead of aggregating.
    ///
    /// Returns:
    ///     The parsed Python object, once it is confirmed a member of the set.
    ///
    /// Raises:
    ///     ValidationError: If the document is malformed JSON (code
    ///         `json_invalid`) or is not a member of the schema's set.
    ///     TypeError: If `data` is not `str` or `bytes`.
    #[pyo3(signature = (data, *, fail_fast = false))]
    fn load<'py>(&self, data: &Bound<'py, PyAny>, fail_fast: bool) -> PyResult<Bound<'py, PyAny>> {
        let parsed = parse_json(data)?;
        self.validate(&parsed, fail_fast)?;
        Ok(parsed)
    }

    /// Whether a JSON document parses and is a member of the schema's set.
    ///
    /// Check-only: malformed JSON, or input that is neither `str` nor `bytes`, is
    /// simply not a member and returns `False`. The document is validated in place
    /// against the parsed value, with no intermediate Python objects for the
    /// structure it walks. It raises only if a membership comparison raises a
    /// fatal interpreter signal.
    ///
    /// Args:
    ///     data: The JSON document, as `str` or `bytes`.
    ///
    /// Returns:
    ///     `True` if `data` parses and is a member of the schema's set, else
    ///     `False`.
    ///
    /// Raises:
    ///     BaseException: If a membership comparison raises a fatal interpreter
    ///         signal, it propagates rather than being read as a non-member.
    fn is_valid_json(&self, data: &Bound<'_, PyAny>) -> PyResult<bool> {
        let py = data.py();
        if let Ok(text) = data.cast::<PyString>() {
            match text.to_str() {
                Ok(json) => self.matches_json(py, json.as_bytes()),
                Err(_) => Ok(false),
            }
        } else if let Ok(raw) = data.cast::<PyBytes>() {
            self.matches_json(py, raw.as_bytes())
        } else {
            Ok(false)
        }
    }

    /// Whether the schema is unsatisfiable — provably empty, so `is_valid`
    /// returns `False` for every value.
    ///
    /// Decided soundly: `True` only when no value can belong to the schema — an
    /// unsatisfiable intersection, a fixed sequence with an impossible position,
    /// a record with an impossible required field, a refinement whose bounds
    /// cannot hold together (a lower bound above an upper bound, or a minimum
    /// length above a maximum), or a recursive schema with no base case (a
    /// mandatory self-reference that can never bottom out). It never reports a
    /// satisfiable schema as empty; for forms it cannot decide it returns `False`.
    ///
    /// Returns:
    ///     `True` if the schema denotes the empty set, else `False`.
    fn is_empty(&self, py: Python<'_>) -> bool {
        let oracle = PoolRelations {
            py,
            literals: &self.literals,
            definitions: &self.definitions,
        };
        self.schema.is_empty_with(&oracle, &self.definitions)
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
    fn is_subtype_of(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let mut literals: Vec<Py<PyAny>> = self.literals.iter().map(|o| o.clone_ref(py)).collect();
        let mut definitions = self.definitions.clone();
        let other = build_schema(other, &mut literals, &mut definitions)?;
        let oracle = PoolRelations {
            py,
            literals: &literals,
            definitions: &definitions,
        };
        Ok(self
            .schema
            .is_subtype_of_under(&other, &oracle, &definitions))
    }

    /// Whether this schema and `other` denote the same set — mutual inclusion.
    ///
    /// `other` is any schema spec or compiled validator. Sound, like
    /// `is_subtype_of`: `True` only when the two are provably equivalent,
    /// whatever their syntax (`bool | int` is equivalent to `int`).
    ///
    /// Args:
    ///     other: The schema to compare, as a spec or validator.
    ///
    /// Returns:
    ///     `True` if the two schemas are equivalent, else `False`.
    fn is_equivalent(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let mut literals: Vec<Py<PyAny>> = self.literals.iter().map(|o| o.clone_ref(py)).collect();
        let mut definitions = self.definitions.clone();
        let other = build_schema(other, &mut literals, &mut definitions)?;
        let oracle = PoolRelations {
            py,
            literals: &literals,
            definitions: &definitions,
        };
        Ok(self
            .schema
            .is_equivalent_under(&other, &oracle, &definitions))
    }

    /// Render the compiled schema back as the annotation expression that
    /// produces it.
    fn __repr__(&self, py: Python<'_>) -> String {
        let active = RefCell::new(FxHashSet::default());
        render(py, &self.schema, &self.literals, &self.definitions, &active)
    }

    /// Return an equivalent validator. The validator is immutable, so the copy
    /// shares the pooled constants, classes, and predicates rather than
    /// duplicating them.
    fn __copy__(&self, py: Python<'_>) -> Validator {
        Validator::new(
            self.schema.clone(),
            self.literals.iter().map(|o| o.clone_ref(py)).collect(),
            self.definitions.clone(),
        )
    }

    /// Deep-copy to an equivalent validator. Since the validator is immutable,
    /// this shares the pool like `__copy__`; the memo is unused.
    fn __deepcopy__(&self, py: Python<'_>, _memo: &Bound<'_, PyAny>) -> Validator {
        self.__copy__(py)
    }

    /// Open every record in the schema: undeclared keys are admitted throughout.
    ///
    /// Returns a new validator; this one is unchanged.
    ///
    /// Returns:
    ///     A validator whose every record admits keys beyond those declared.
    fn open(&self, py: Python<'_>) -> Validator {
        with_records_open(self, true, py)
    }

    /// Close every record in the schema: only declared keys are admitted
    /// throughout. The inverse of `open`.
    ///
    /// Returns a new validator; this one is unchanged.
    ///
    /// Returns:
    ///     A validator whose every record admits only its declared keys.
    fn close(&self, py: Python<'_>) -> Validator {
        with_records_open(self, false, py)
    }

    /// An equivalent validator reduced by the lattice laws.
    ///
    /// The result admits exactly the same values in a simpler form (flattened
    /// and deduplicated unions and intersections, identities applied,
    /// complements in negation-normal form). Returns a new validator; this one
    /// is unchanged.
    ///
    /// Returns:
    ///     A validator denoting the same set in negation-normal form.
    fn simplify(&self, py: Python<'_>) -> Validator {
        Validator::new(
            self.schema.simplify(),
            self.literals.iter().map(|o| o.clone_ref(py)).collect(),
            self.definitions.clone(),
        )
    }

    /// Whether `obj` is a member of the schema's set: the operator form of
    /// `is_valid`, so `obj in validator` reads as the set membership it is.
    fn __contains__(&self, obj: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.is_valid(obj)
    }

    /// The union of this schema and `other`, written `validator | other`. `|` is
    /// the one operator typing already uses for unions; intersection and
    /// complement stay spelled out as `intersection`/`complement`. `other` is any
    /// schema spec or validator.
    fn __or__(&self, other: &Bound<'_, PyAny>) -> PyResult<Validator> {
        self.union_with(other, false)
    }

    /// The union `other | validator`, used when the left operand does not handle
    /// `|` (for example `None | validator`).
    fn __ror__(&self, other: &Bound<'_, PyAny>) -> PyResult<Validator> {
        self.union_with(other, true)
    }

    /// Structural equality: two validators are equal when their schema trees,
    /// recursive definitions, and pooled constants all match. This is *syntactic*
    /// — `union(int, str)` and `union(str, int)` are not equal — whereas
    /// `is_equivalent` compares the sets two schemas denote regardless of shape.
    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        let Ok(bound) = other.cast::<Validator>() else {
            return false;
        };
        let py = bound.py();
        let other = bound.get();
        if self.schema != other.schema
            || self.definitions != other.definitions
            || self.literals.len() != other.literals.len()
        {
            return false;
        }
        // Compare pooled constants by value (identity first, so a validator
        // equals itself even when it pools a value that is not equal to itself,
        // such as NaN).
        self.literals.iter().zip(&other.literals).all(|(a, b)| {
            let (a, b) = (a.bind(py), b.bind(py));
            a.is(b) || a.eq(b).unwrap_or(false)
        })
    }

    /// A hash consistent with structural equality. It digests the schema shape
    /// and definitions only, never the pooled constant values, so it stays total
    /// (an unhashable pooled constant cannot break it) and equal validators hash
    /// alike.
    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.schema.hash(&mut hasher);
        self.definitions.hash(&mut hasher);
        hasher.finish()
    }
}

/// Parse a JSON `str` or `bytes` into a Python value with jiter.
///
/// jiter's defaults match the standard JSON model: standard `float`s, no
/// `Infinity`/`NaN`, and complete (non-partial) input — so the parsed value is
/// what the object path would receive from `json.loads`. A parse failure is
/// surfaced as a structured `json_invalid` `ValidationError`; a non-string,
/// non-bytes argument is a `TypeError`.
/// Turn a membership walk's outcome into a Python result: re-raise a fatal
/// interpreter signal the walk recorded, otherwise report the membership verdict.
fn reraise_fatal(fatal: RefCell<Option<PyErr>>, ok: bool) -> PyResult<bool> {
    match fatal.into_inner() {
        Some(err) => Err(err),
        None => Ok(ok),
    }
}

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
    let body_obj = builder.call1((placeholder,))?;
    let mut literals = Vec::new();
    let mut definitions = Vec::new();
    let body = build_schema(&body_obj, &mut literals, &mut definitions)?;
    // The body becomes a definition; the self-reference resolves to it.
    let ref_id = definitions.len();
    let resolved = body.resolve_self(token, ref_id);
    if resolved.occurs_unguarded(ref_id, false) {
        return Err(PyValueError::new_err(
            "recursive schema is not contractive: the recursive reference must \
             occur under a structural constructor (a list, tuple, set, dict, \
             record, or object)",
        ));
    }
    definitions.push(resolved);
    Ok(Validator::new(Schema::Ref(ref_id), literals, definitions))
}

/// The union of the given schemas: a value in at least one of their sets.
#[pyfunction]
#[pyo3(signature = (*schemas))]
fn union(schemas: &Bound<'_, PyTuple>) -> PyResult<Validator> {
    combine(schemas, Schema::Union)
}

/// The intersection of the given schemas: a value in every one of their sets.
#[pyfunction]
#[pyo3(signature = (*schemas))]
fn intersection(schemas: &Bound<'_, PyTuple>) -> PyResult<Validator> {
    combine(schemas, Schema::Intersection)
}

/// The complement of a schema: every value not in its set.
#[pyfunction]
fn complement(schema: &Bound<'_, PyAny>) -> PyResult<Validator> {
    let mut literals = Vec::new();
    let mut definitions = Vec::new();
    let inner = build_schema(schema, &mut literals, &mut definitions)?;
    Ok(Validator::new(
        Schema::Complement(Box::new(inner)),
        literals,
        definitions,
    ))
}

fn with_records_open(validator: &Validator, open: bool, py: Python<'_>) -> Validator {
    Validator::new(
        validator.schema.with_records_open(open),
        validator.literals.iter().map(|o| o.clone_ref(py)).collect(),
        validator.definitions.clone(),
    )
}

/// A pool-free validator wrapping a single atom (the `anything`/`nothing`
/// lattice bounds).
fn atom(py: Python<'_>, schema: Schema) -> PyResult<Py<Validator>> {
    Py::new(py, Validator::new(schema, Vec::new(), Vec::new()))
}

/// A [`LeafRelations`] oracle backed by a validator's constant pool. It decides
/// a `Literal` subtyping by running membership of the literal's value against
/// the candidate supertype, and an `Instance`-versus-`Instance` subtyping by
/// `issubclass` on the pooled classes.
struct PoolRelations<'py, 'pool> {
    py: Python<'py>,
    literals: &'pool [Py<PyAny>],
    definitions: &'pool [Schema],
}

impl PoolRelations<'_, '_> {
    fn is_member(&self, schema: &Schema, value: &Bound<'_, PyAny>) -> bool {
        let guard = RefCell::new(FxHashSet::default());
        // These leaf-subtype probes run on transient schemas during compilation,
        // not on a finished validator, so they carry no precomputed index; the
        // walk falls back to its general path for any record or union here. A
        // fatal signal in a probe folds to non-membership here (the decision
        // procedure is not the interruptible hot path); the cell is local.
        let fatal = RefCell::new(None);
        let index = ValidatorIndex::default();
        let ctx = Ctx {
            pool: self.literals,
            defs: self.definitions,
            records: &index.records,
            unions: &index.unions,
            regexes: &index.regexes,
            guard: &guard,
            fatal: &fatal,
            explain: false,
            fail_fast: false,
        };
        member(
            schema,
            &Value::Py(value),
            &mut Vec::new(),
            ctx,
            &mut Vec::new(),
        )
    }
}

impl LeafRelations for PoolRelations<'_, '_> {
    fn leaf_subtype(&self, sub: &Schema, sup: &Schema) -> Option<bool> {
        match sub {
            // A literal denotes a singleton: `{v}` is a subtype of `sup` exactly
            // when `v` is a member of `sup`.
            Schema::Literal(index) => {
                let value = self.literals.get(*index)?.bind(self.py);
                Some(self.is_member(sup, value))
            }
            // The `isinstance(., C)` values are a subset of the `isinstance(., D)`
            // values exactly when `C` is a subclass of `D`.
            Schema::Instance(index) => match sup {
                Schema::Instance(superindex) => {
                    let class = self.literals.get(*index)?.bind(self.py);
                    let superclass = self.literals.get(*superindex)?.bind(self.py);
                    let decided = class
                        .cast::<PyType>()
                        .ok()
                        .and_then(|class| class.is_subclass(superclass).ok())
                        .unwrap_or(false);
                    Some(decided)
                }
                _ => None,
            },
            _ => None,
        }
    }

    fn compare(&self, left: usize, right: usize) -> Option<core::cmp::Ordering> {
        // Order two refinement-bound values by Python's own comparison, so the
        // core can decide an unsatisfiable bound conjunction. An incomparable
        // pair (a TypeError) leaves the bound undecided.
        let left = self.literals.get(left)?.bind(self.py);
        let right = self.literals.get(right)?.bind(self.py);
        left.compare(right).ok()
    }
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
    module.add("anything", atom(py, Schema::Anything)?)?;
    module.add("nothing", atom(py, Schema::Nothing)?)?;
    Ok(())
}
