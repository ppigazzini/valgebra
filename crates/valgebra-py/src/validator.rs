//! The compiled validator: the object a user holds, and every method on it.
//!
//! A leaf module. It lives here rather than in the crate root because the schema
//! frontend needs the type -- a compiled validator is itself a schema
//! description -- and a type defined in the aggregator and imported back by one
//! of its members is what puts the two in a cycle. `lib.rs` re-exports it, so no
//! caller spells a new path and the Python surface is unchanged.

use std::cell::RefCell;
use std::sync::OnceLock;

use jiter::{JsonValue, PythonParse};
use pyo3::PyTypeInfo;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBool, PyBytes, PyFloat, PyInt, PyString, PyType};
use rustc_hash::FxHashSet;
use valgebra_core::{ConstIx, Kind, LeafRelations, Openness, OperandIx, Schema};

use crate::build::{Pool, build_schema};
use crate::check::{Ctx, ValidatorIndex, WalkMode, WalkState, build_index, member};
use crate::errors::{into_pyerr, json_invalid_error};
use crate::input::Value;
use crate::render::render;
/// The deepest structural nesting a constructed schema may reach. A real schema
/// is nowhere near this deep and the annotation frontend caps its own nesting
/// lower, so a validator this deep is one built in an unbounded loop. Every
/// recursive walk over the tree — clone, drop, the decision procedure — descends
/// one native stack frame per level, so building past this bound returns an
/// error rather than overflowing the stack. Structural recursion in a schema is
/// written with `recursive`, whose back edge is a `Ref` leaf and does not count
/// toward this depth.
/// The `math.floor` and `math.ceil` callables, imported once per interpreter for
/// the integer-interval emptiness rule rather than re-imported on every decision.
/// `PyOnceLock` keeps the one-time initialization sound under free-threading.
static MATH_FLOOR_CEIL: PyOnceLock<(Py<PyAny>, Py<PyAny>)> = PyOnceLock::new();

pub(crate) const MAX_SCHEMA_DEPTH: usize = 128;
/// The most recursive definitions a constructed schema may hold. A recursive
/// schema needs a mere handful; a validator with more is one whose definitions
/// were chained in an unbounded loop, and the render and decision walks descend
/// the chain one native stack frame per link (a chain of distinct definitions is
/// invisible to the per-tree depth measure, which counts a `Ref` as a leaf).
pub(crate) const MAX_DEFINITIONS: usize = 128;
/// The most schema nodes a constructed schema may hold, across its tree and every
/// definition. Bounds a schema that is shallow but exponentially wide — a
/// doubling union grows its node count, not its depth — where the depth bound
/// alone cannot. Set far above any real schema (a record with tens of thousands
/// of fields still fits) so only a runaway loop trips it; combining two operands
/// whose sizes already sum past it is rejected, so the doubling stops early
/// rather than exhausting memory.
pub(crate) const MAX_SCHEMA_NODES: usize = 100_000;
thread_local! {
    /// The self-reference tokens of the `recursive` definitions being built on
    /// this thread, innermost last.
    ///
    /// A placeholder validator says nothing about where it means something: it is
    /// an ordinary `Validator`, and a caller can keep it past the builder it was
    /// handed to. Its token sits here for exactly the span in which the fixpoint
    /// it stands for is being defined, which is what lets construction tell a
    /// marker that is about to become a back edge from one that outlived its
    /// call.
    static OPEN_DEFINITIONS: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) };
}

/// One `recursive` definition open on this thread, closed when the guard drops.
pub(crate) struct OpenDefinition;

impl OpenDefinition {
    /// Open `token` for as long as the returned guard lives.
    pub(crate) fn open(token: u64) -> Self {
        OPEN_DEFINITIONS.with_borrow_mut(|open| open.push(token));
        OpenDefinition
    }
}

impl Drop for OpenDefinition {
    fn drop(&mut self) {
        OPEN_DEFINITIONS.with_borrow_mut(|open| {
            open.pop();
        });
    }
}

/// Whether `token` names a definition being built on this thread.
fn definition_is_open(token: u64) -> bool {
    OPEN_DEFINITIONS.with_borrow(|open| open.contains(&token))
}

fn math_floor_ceil(py: Python<'_>) -> PyResult<&'static (Py<PyAny>, Py<PyAny>)> {
    MATH_FLOOR_CEIL.get_or_try_init(py, || {
        let math = py.import("math")?;
        Ok((
            math.getattr("floor")?.unbind(),
            math.getattr("ceil")?.unbind(),
        ))
    })
}
/// Turn a membership walk's outcome into a Python result: re-raise a fatal
/// interpreter signal the walk recorded, otherwise report the membership verdict.
fn reraise_fatal(state: WalkState, ok: bool) -> PyResult<bool> {
    match state.into_fatal() {
        Some(err) => Err(err),
        None => Ok(ok),
    }
}

/// The UTF-8 bytes a JSON `str`/`bytes` argument hands to the parser, or why they
/// are unavailable. A `str` carrying a lone surrogate cannot be encoded to UTF-8;
/// both JSON entry points must treat that the same malformed-input way rather than
/// let the raw `UnicodeEncodeError` leak — which would disagree with the check
/// path and break `validate_json`'s documented exception set.
enum JsonInput<'a> {
    /// Bytes ready for the parser.
    Bytes(&'a [u8]),
    /// A `str` the interpreter cannot encode to UTF-8 (a lone surrogate).
    Undecodable,
    /// Neither `str` nor `bytes`.
    NotStrOrBytes,
}

/// Decode a JSON argument to the bytes the parser reads, classifying the two ways
/// it can be unusable so both entry points agree on them.
fn decode_json_input<'a>(data: &'a Bound<'_, PyAny>) -> JsonInput<'a> {
    if let Ok(text) = data.cast::<PyString>() {
        match text.to_str() {
            Ok(text) => JsonInput::Bytes(text.as_bytes()),
            Err(_) => JsonInput::Undecodable,
        }
    } else if let Ok(raw) = data.cast::<PyBytes>() {
        JsonInput::Bytes(raw.as_bytes())
    } else {
        JsonInput::NotStrOrBytes
    }
}

/// Parse a JSON `str` or `bytes` into a Python value with jiter.
///
/// jiter's defaults match the standard JSON model: standard `float`s, no
/// `Infinity`/`NaN`, and complete (non-partial) input — so the parsed value is
/// what the object path would receive from `json.loads`. A parse failure, or a
/// `str` the interpreter cannot encode to UTF-8, is surfaced as a structured
/// `json_invalid` `ValidationError`; a non-string, non-bytes argument is a
/// `TypeError`.
fn parse_json<'py>(data: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let py = data.py();
    let parse = PythonParse::default();
    match decode_json_input(data) {
        JsonInput::Bytes(bytes) => parse
            .python_parse(py, bytes)
            .map_err(|err| json_invalid_error(py, &err.description(bytes))),
        // An undecodable string is malformed input, reported through the same
        // structured `json_invalid` model as an unparseable document — never as a
        // raw `UnicodeEncodeError`, which `validate_json` promises not to raise.
        JsonInput::Undecodable => Err(json_invalid_error(
            py,
            "input string is not valid UTF-8 (contains a lone surrogate)",
        )),
        JsonInput::NotStrOrBytes => Err(PyTypeError::new_err(
            "JSON input must be a str or bytes object",
        )),
    }
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
        // These leaf-subtype probes run on transient schemas during compilation,
        // not on a finished validator, so they carry no precomputed index; the
        // walk falls back to its general path for any record or union here. A
        // fatal signal in a probe folds to non-membership here (the decision
        // procedure is not the interruptible hot path); the state is local.
        let state = WalkState::new();
        let index = ValidatorIndex::default();
        let ctx = Ctx {
            pool: self.literals,
            defs: self.definitions,
            records: &index.records,
            unions: &index.unions,
            regexes: &index.regexes,
            guard: &state.guard,
            depth: &state.depth,
            fatal: &state.fatal,
            fatal_seen: &state.fatal_seen,
            mode: WalkMode::Fast,
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
                let value = self.literals.get(index.get())?.bind(self.py);
                Some(self.is_member(sup, value))
            }
            // The `isinstance(., C)` values are a subset of the `isinstance(., D)`
            // values exactly when `C` is a subclass of `D`.
            Schema::Instance(index) => match sup {
                Schema::Instance(superindex) => {
                    let class = self.literals.get(index.get())?.bind(self.py);
                    let superclass = self.literals.get(superindex.get())?.bind(self.py);
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

    fn literal_kind(&self, constant: ConstIx) -> Option<Kind> {
        // A literal pins `type(x)` exactly, so its kind is its constant's type.
        // Exact types only: a subclass of `int` is not `Kind::Int`'s extension,
        // and any other type is a kind the partition does not name. Both decline,
        // which leaves disjointness conservative rather than wrong.
        let value = self.literals.get(constant.get())?.bind(self.py);
        if value.is_none() {
            return Some(Kind::NoneType);
        }
        let ty = value.get_type();
        [
            (Kind::Bool, PyBool::type_object(self.py)),
            (Kind::Int, PyInt::type_object(self.py)),
            (Kind::Float, PyFloat::type_object(self.py)),
            (Kind::Str, PyString::type_object(self.py)),
            (Kind::Bytes, PyBytes::type_object(self.py)),
        ]
        .into_iter()
        .find_map(|(tag, exact)| ty.is(&exact).then_some(tag))
    }

    fn literals_disjoint(&self, left: ConstIx, right: ConstIx) -> Option<bool> {
        let left_value = self.literals.get(left.get())?.bind(self.py);
        let right_value = self.literals.get(right.get())?.bind(self.py);
        // A literal admits a value only when `type(x)` matches exactly, so two
        // constants of different types share no value however they compare --
        // `Literal[1]` and `Literal[True]` are disjoint although `1 == True`.
        if !left_value.get_type().is(right_value.get_type()) {
            return Some(true);
        }
        // Same type, so the singletons are disjoint exactly when the constants
        // differ. `==` is the value's own, and a type carrying user-defined
        // equality can admit one value for two distinct constants, so this is
        // asked only where the type is one this oracle kinds -- a builtin scalar
        // whose equality is Python's. An `Enum` member declines here.
        self.literal_kind(left)?;
        left_value.eq(right_value).ok().map(|equal| !equal)
    }

    fn compare(&self, left: OperandIx, right: OperandIx) -> Option<core::cmp::Ordering> {
        // Order two refinement-bound values by Python's own comparison, so the
        // core can decide an unsatisfiable bound conjunction. An incomparable
        // pair (a TypeError) leaves the bound undecided.
        let left = self.literals.get(left.get())?.bind(self.py);
        let right = self.literals.get(right.get())?.bind(self.py);
        left.compare(right).ok()
    }

    fn no_int_between(
        &self,
        lo: OperandIx,
        lo_strict: bool,
        hi: OperandIx,
        hi_strict: bool,
    ) -> Option<bool> {
        // Decide whether the open/half-open interval bounded by the pooled values
        // admits no integer. The least admissible integer is `floor(lo) + 1` when
        // `lo` is excluded and `ceil(lo)` when it is included; the greatest is
        // `ceil(hi) - 1` when `hi` is excluded and `floor(hi)` when included. No
        // integer fits exactly when the least exceeds the greatest. The bounds are
        // compared as Python integers, so arbitrary-precision values stay exact.
        let (floor, ceil) = math_floor_ceil(self.py).ok()?;
        let floor = floor.bind(self.py);
        let ceil = ceil.bind(self.py);
        let lo = self.literals.get(lo.get())?.bind(self.py);
        let hi = self.literals.get(hi.get())?.bind(self.py);
        // A non-real bound (`math.floor` raises a `TypeError`) or a non-finite one
        // (an `OverflowError`) leaves the rule undecided rather than guessing.
        let one = 1i64;
        let least = if lo_strict {
            floor
                .call1((&lo,))
                .ok()?
                .call_method1("__add__", (one,))
                .ok()?
        } else {
            ceil.call1((&lo,)).ok()?
        };
        let greatest = if hi_strict {
            ceil.call1((&hi,))
                .ok()?
                .call_method1("__sub__", (one,))
                .ok()?
        } else {
            floor.call1((&hi,)).ok()?
        };
        // `least > greatest` means the interval skips every integer.
        Some(matches!(
            least.compare(&greatest).ok()?,
            core::cmp::Ordering::Greater
        ))
    }
}

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

    /// Assemble a validator, rejecting one whose schema is too deep, holds too
    /// many recursive definitions, or spans too many nodes. Every growth path —
    /// the `Validator(...)` constructor, the `|`/`union`/`intersection`/
    /// `complement` combinators, `recursive`, and `simplify` — routes through
    /// here, so no public call can build a schema that overflows the stack or
    /// exhausts memory on a later walk. A schema reaching this point grew by one
    /// step from operands already within the bounds, so it is at most one step
    /// past them — shallow and small enough that measuring it and dropping it when
    /// a bound is exceeded are themselves safe.
    pub(crate) fn checked(
        schema: Schema,
        literals: Vec<Py<PyAny>>,
        definitions: Vec<Schema>,
    ) -> PyResult<Self> {
        let depth = definitions
            .iter()
            .map(Schema::depth)
            .max()
            .unwrap_or(0)
            .max(schema.depth());
        if depth > MAX_SCHEMA_DEPTH {
            return Err(PyValueError::new_err(format!(
                "schema nesting is too deep: this validator reaches {depth} levels of \
                 nesting, past the limit of {MAX_SCHEMA_DEPTH}. A validator this deep \
                 comes from building in an unbounded loop; checking it would risk a \
                 native stack overflow. Express structural recursion with recursive(...) \
                 instead of nesting schemas."
            )));
        }
        if definitions.len() > MAX_DEFINITIONS {
            return Err(PyValueError::new_err(format!(
                "schema holds too many recursive definitions: this validator has {} of \
                 them, past the limit of {MAX_DEFINITIONS}. A validator with this many \
                 comes from chaining recursive schemas in an unbounded loop; rendering \
                 or deciding it would risk a native stack overflow.",
                definitions.len()
            )));
        }
        let nodes = definitions
            .iter()
            .map(Schema::node_count)
            .sum::<usize>()
            .saturating_add(schema.node_count());
        if nodes > MAX_SCHEMA_NODES {
            return Err(PyValueError::new_err(format!(
                "schema is too large: this validator spans {nodes} nodes, past the limit \
                 of {MAX_SCHEMA_NODES}. A schema this large comes from combining a \
                 validator with itself in an unbounded loop, which doubles its size each \
                 step; building it would exhaust memory."
            )));
        }
        // The placeholder a `recursive` builder receives is an ordinary validator,
        // so a caller can keep it and hand it back after the call it stands for
        // has finished -- at which point it names a fixpoint nobody is defining.
        // A marker for a definition still open is the ordinary way a body is
        // written and passes; one for a closed definition denotes no set, so it
        // is refused here rather than validating as a value nothing matches.
        let is_open: &dyn Fn(u64) -> bool = &definition_is_open;
        if schema.has_escaped_self_ref(is_open)
            || definitions
                .iter()
                .any(|definition| definition.has_escaped_self_ref(is_open))
        {
            return Err(PyValueError::new_err(
                "schema holds an unresolved recursive placeholder: the validator a \
                 recursive(...) builder receives stands for the schema being defined \
                 and is only meaningful inside that call. Build the schema that uses \
                 it inside the builder, and use the validator recursive(...) returns \
                 outside.",
            ));
        }
        Ok(Validator::new(schema, literals, definitions))
    }

    /// Rebuild this validator with `f` applied to the root schema and to every
    /// recursive definition.
    ///
    /// Both fields hold the carrier. A recursive validator's root is a single
    /// [`Schema::Ref`] leaf and every record, union and refinement it declares
    /// lives in the definitions table, so a rewrite reaching only the root is a
    /// no-op on exactly the schemas with the most to rewrite.
    ///
    /// The result goes through [`checked`](Self::checked) because a rewrite is a
    /// growth path: opening a closed record adds a catch-all clause, and
    /// negation-normal form distributes a complement over a union.
    fn map_schemas(&self, py: Python<'_>, f: impl Fn(&Schema) -> Schema) -> PyResult<Validator> {
        Validator::checked(
            f(&self.schema),
            self.literals.iter().map(|o| o.clone_ref(py)).collect(),
            self.definitions.iter().map(&f).collect(),
        )
    }

    /// The precompute, built once from this validator's schema, definitions, and
    /// constants pool.
    fn index(&self, py: Python<'_>) -> &ValidatorIndex {
        self.index
            .get_or_init(|| build_index(py, &self.schema, &self.definitions, &self.literals))
    }

    /// The read-only walk context: the pool, the definitions, the precomputed
    /// record and union indexes, the call's own [`WalkState`], and the mode.
    pub(crate) fn context<'a>(
        &'a self,
        py: Python<'_>,
        state: &'a WalkState,
        mode: WalkMode,
    ) -> Ctx<'a> {
        let index = self.index(py);
        Ctx {
            pool: &self.literals,
            defs: &self.definitions,
            records: &index.records,
            unions: &index.unions,
            regexes: &index.regexes,
            guard: &state.guard,
            depth: &state.depth,
            fatal: &state.fatal,
            fatal_seen: &state.fatal_seen,
            mode,
        }
    }

    /// Union this schema with `other` (a spec or validator), placing `other`
    /// first when it is the `|` right operand. Backs `__or__`/`__ror__`: the
    /// fresh pool seeds with this validator's constants so its schema indices
    /// stay valid, then `other` interns into it.
    fn union_with(&self, other: &Bound<'_, PyAny>, other_first: bool) -> PyResult<Validator> {
        let py = other.py();
        let mut literals = Pool::seeded(self.literals.iter().map(|o| o.clone_ref(py)).collect());
        let mut definitions = self.definitions.clone();
        let other_schema = build_schema(other, &mut literals, &mut definitions)?;
        let members = if other_first {
            vec![other_schema, self.schema.clone()]
        } else {
            vec![self.schema.clone(), other_schema]
        };
        Validator::checked(Schema::union(members), literals.into_items(), definitions)
    }

    /// Whether the JSON in `bytes` parses and belongs to the schema's set,
    /// validated in place against the parsed JSON value with no intermediate
    /// Python objects. `bytes` outlives the parsed value and the walk.
    fn matches_json(&self, py: Python<'_>, bytes: &[u8]) -> PyResult<bool> {
        let Ok(json) = JsonValue::parse(bytes, false) else {
            return Ok(false);
        };
        let state = WalkState::new();
        let ok = member(
            &self.schema,
            &Value::Json(py, &json),
            &mut Vec::new(),
            self.context(py, &state, WalkMode::Fast),
            &mut Vec::new(),
        );
        reraise_fatal(state, ok)
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
        let mut literals = Pool::default();
        let mut definitions = Vec::new();
        let schema = build_schema(schema, &mut literals, &mut definitions)?;
        Validator::checked(schema, literals.into_items(), definitions)
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
        let state = WalkState::new();
        let mut path = Vec::new();
        let mut violations = Vec::new();
        let ok = member(
            &self.schema,
            &Value::Py(obj),
            &mut path,
            self.context(obj.py(), &state, WalkMode::explaining(fail_fast)),
            &mut violations,
        );
        if let Some(err) = state.into_fatal() {
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
        let state = WalkState::new();
        let ok = member(
            &self.schema,
            &Value::Py(obj),
            &mut Vec::new(),
            self.context(obj.py(), &state, WalkMode::Fast),
            &mut Vec::new(),
        );
        reraise_fatal(state, ok)
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
    ///     ValidationError: If the document is malformed or undecodable JSON
    ///         (code `json_invalid`) or is not a member of the schema's set.
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
    ///     ValidationError: If the document is malformed or undecodable JSON
    ///         (code `json_invalid`) or is not a member of the schema's set.
    ///     TypeError: If `data` is not `str` or `bytes`.
    #[pyo3(signature = (data, *, fail_fast = false))]
    fn load<'py>(&self, data: &Bound<'py, PyAny>, fail_fast: bool) -> PyResult<Bound<'py, PyAny>> {
        let parsed = parse_json(data)?;
        self.validate(&parsed, fail_fast)?;
        Ok(parsed)
    }

    /// Whether a JSON document parses and is a member of the schema's set.
    ///
    /// Check-only: malformed or undecodable JSON, or input that is neither `str`
    /// nor `bytes`, is simply not a member and returns `False`. The raising entries
    /// (`validate_json`/`load`) report the same undecodable input as a structured
    /// `json_invalid` error. The document is validated in place
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
        match decode_json_input(data) {
            JsonInput::Bytes(bytes) => self.matches_json(py, bytes),
            // An undecodable string and a non-str/bytes argument are both simply
            // not members — the same verdict the raising path reports structurally.
            JsonInput::Undecodable | JsonInput::NotStrOrBytes => Ok(false),
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
    /// The decision is also bounded by a fixed work budget, so on a deeply nested
    /// adversarial schema a `False` can mean "not proven empty within the bound"
    /// rather than "non-empty"; a real schema decides far inside the bound.
    ///
    /// Deciding a refinement's bounds orders the two operands, which runs their
    /// rich comparison, so this can call back into Python.
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
    /// `int`, `list[bool]` of `list[int]`, a recursive schema of a wider one, a
    /// class of a base class). For the forms it cannot decide — an alternation of
    /// sequence shapes, or a leaf relation the oracle declines — it returns
    /// `False` rather than a relation it cannot justify. The decision is bounded by
    /// a fixed work budget, so on a deeply nested adversarial schema a `False` can
    /// mean "not proven a subtype within the bound"; a real schema decides far
    /// inside the bound.
    ///
    /// Nothing is inferred from a predicate refinement, but one may be **called**:
    /// deciding whether a literal is a subtype of a refinement decides whether the
    /// literal's value belongs to it, and belonging runs the predicate. A slow or
    /// side-effecting predicate is one this query pays for.
    ///
    /// Args:
    ///     other: The candidate supertype, as a schema spec or validator.
    ///
    /// Returns:
    ///     `True` if this schema is a subtype of `other`, else `False`.
    fn is_subtype_of(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let mut literals = Pool::seeded(self.literals.iter().map(|o| o.clone_ref(py)).collect());
        let mut definitions = self.definitions.clone();
        let other = build_schema(other, &mut literals, &mut definitions)?;
        let oracle = PoolRelations {
            py,
            literals: literals.items(),
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
    /// whatever their syntax (`bool | int` is equivalent to `int`). Bounded by the
    /// same work budget as `is_subtype_of`, so on a deeply nested adversarial
    /// schema a `False` can mean "not proven equivalent within the bound".
    ///
    /// Mutual inclusion, so it calls back into Python wherever `is_subtype_of`
    /// does.
    ///
    /// Args:
    ///     other: The schema to compare, as a spec or validator.
    ///
    /// Returns:
    ///     `True` if the two schemas are equivalent, else `False`.
    fn is_equivalent(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let mut literals = Pool::seeded(self.literals.iter().map(|o| o.clone_ref(py)).collect());
        let mut definitions = self.definitions.clone();
        let other = build_schema(other, &mut literals, &mut definitions)?;
        let oracle = PoolRelations {
            py,
            literals: literals.items(),
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
        render(
            py,
            &self.schema,
            &self.literals,
            &self.definitions,
            &active,
            0,
        )
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

    /// Open every record in the schema: undeclared keys are admitted throughout,
    /// including inside recursive definitions.
    ///
    /// Returns a new validator; this one is unchanged. A record that already
    /// carries a typed catch-all clause has it widened to admit any key with any
    /// value, so `open` and `close` are idempotent projections rather than
    /// inverses.
    ///
    /// Returns:
    ///     A validator whose every record admits keys beyond those declared.
    ///
    /// Raises:
    ///     ValueError: If admitting undeclared keys expands the schema past the
    ///         size bound; opening a record adds a catch-all clause, so a
    ///         validator near the limit can cross it.
    fn open(&self, py: Python<'_>) -> PyResult<Validator> {
        self.map_schemas(py, |schema| schema.with_records_open(Openness::Open))
    }

    /// Close every record in the schema: only declared keys are admitted
    /// throughout, including inside recursive definitions.
    ///
    /// Returns a new validator; this one is unchanged. Closing drops a record's
    /// catch-all clause, typed or not, so it admits only its declared keys.
    ///
    /// Returns:
    ///     A validator whose every record admits only its declared keys.
    fn close(&self, py: Python<'_>) -> PyResult<Validator> {
        self.map_schemas(py, |schema| schema.with_records_open(Openness::Closed))
    }

    /// An equivalent validator reduced by the lattice laws.
    ///
    /// The result admits exactly the same values in a simpler form (flattened
    /// and deduplicated unions and intersections, identities applied,
    /// complements in negation-normal form), throughout the schema and every
    /// recursive definition. Returns a new validator; this one is unchanged.
    ///
    /// Returns:
    ///     A validator denoting the same set in negation-normal form.
    ///
    /// Raises:
    ///     ValueError: If negation-normal form expands the schema past the size
    ///         bound (distributing a complement over a wide union can grow the
    ///         node count); a schema built within the bounds does not hit this.
    fn simplify(&self, py: Python<'_>) -> PyResult<Validator> {
        self.map_schemas(py, Schema::simplify)
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
    ///
    /// A pooled constant is compared the way a literal reads one: same type and
    /// equal. Python's `==` runs across types, so equality alone makes
    /// `Literal[1]` and `Literal[True]` the same validator while `is_equivalent`
    /// reports them disjoint — two answers about one pair. The type test is what
    /// `Literal` already means, so equality says what the schema says.
    ///
    /// A non-validator gets `NotImplemented` rather than `False`: the data model
    /// asks the other operand next, and it may know something about validators
    /// that validators do not know about it.
    fn __eq__<'py>(&self, other: &Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        let py = other.py();
        let Ok(bound) = other.cast::<Validator>() else {
            return py.NotImplemented().into_bound(py);
        };
        let other = bound.get();
        let equal = self.schema == other.schema
            && self.definitions == other.definitions
            && self.literals.len() == other.literals.len()
            // Identity first, so a validator equals itself even when it pools a
            // value that is not equal to itself, such as NaN.
            && self
                .literals
                .iter()
                .zip(&other.literals)
                .all(|(a, b)| {
                    let (a, b) = (a.bind(py), b.bind(py));
                    a.is(b)
                        || (a.get_type().is(b.get_type()) && a.eq(b).unwrap_or(false))
                });
        PyBool::new(py, equal).to_owned().into_any()
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

// Needs a live interpreter; compiled and run only under the `interpreter-tests`
// feature, which links an embedded Python.
#[cfg(all(test, feature = "interpreter-tests"))]
mod tests {
    use super::*;
    use valgebra_core::{DefIx, Field, MapClause};

    /// A validator whose root is a bare back edge, so every schema it declares
    /// sits in the definitions table. This is the shape a rewrite reaching only
    /// the root leaves untouched, and it is what `recursive` builds.
    fn recursive_record() -> Validator {
        Validator::new(
            Schema::Ref(DefIx::new(0)),
            Vec::new(),
            vec![Schema::record(
                vec![Field {
                    name: "a".to_owned(),
                    schema: Schema::Int,
                    required: true,
                }],
                Openness::Closed,
            )],
        )
    }

    #[test]
    fn map_schemas_rewrites_the_definitions_table() {
        Python::attach(|py| {
            let mapped = recursive_record()
                .map_schemas(py, |schema| schema.with_records_open(Openness::Open))
                .expect("opening one record stays within the construction bounds");
            // The root is a leaf, so the rewrite has to land in the definitions
            // table for the record to have been opened at all.
            assert_eq!(mapped.schema, Schema::Ref(DefIx::new(0)));
            let Schema::KeyedMap { defaults, .. } = &mapped.definitions[0] else {
                panic!("the definition is a record");
            };
            assert_eq!(
                defaults.as_slice(),
                [MapClause::top()],
                "the record in the definitions table gained its catch-all clause"
            );
        });
    }

    #[test]
    fn map_schemas_measures_what_the_rewrite_produced() {
        Python::attach(|py| {
            // Routing through `checked` rather than `new` is what rejects a
            // rewrite that grows the schema past the node bound.
            let wide = Validator::new(
                Schema::Union(vec![Schema::Int; MAX_SCHEMA_NODES / 2]),
                Vec::new(),
                Vec::new(),
            );
            assert!(wide.map_schemas(py, Schema::clone).is_ok());
            let doubled = wide.map_schemas(py, |schema| {
                Schema::Union(vec![schema.clone(), schema.clone()])
            });
            assert!(
                doubled.is_err(),
                "a rewrite that doubles the node count is past the bound"
            );
        });
    }
}
