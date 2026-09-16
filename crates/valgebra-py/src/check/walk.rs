//! The validation walk: one membership test of a value against the IR.
//!
//! [`member`] is the single walk. It returns whether the value belongs to the
//! schema's set, and in an *explain* mode (`ctx.mode`) it also aggregates a
//! [`Violation`] for each independent failure into `out` (each record field,
//! each sequence element, each mapping entry), unless the fail-fast mode stops it
//! at the first. In *fast* mode it allocates nothing and short-circuits as soon
//! as membership is decided.
//!
//! ## Comparison-raises policy
//!
//! Membership reads a value through Python operations that can raise — `__eq__`
//! for a literal, a rich comparison for a bound, `isinstance` for a class,
//! `getattr` for an attribute, `__mod__` for a multiple-of, `__len__` for a
//! length. The single rule across every such site: **a value whose comparison,
//! instance check, or attribute access raises an ordinary exception is treated as
//! a non-member**. This matches pydantic-core: a value that cannot answer "are
//! you in this set?" is not in it. The one ordinary-exception case carved out is
//! a *user predicate*, whose raised error is surfaced as a distinct
//! `predicate_error` rather than folded, so a buggy predicate is visible.
//!
//! A *fatal* interpreter signal is the one error never folded — at every site,
//! the predicate and `getattr` included. [`is_fatal`] classifies it: a base
//! exception that is not an ordinary exception (`KeyboardInterrupt`,
//! `SystemExit`, `GeneratorExit`), and `MemoryError`/`RecursionError` (ordinary
//! exceptions whose meaning is "the interpreter cannot continue"). It is not an
//! answer to "are you in this set?": the interpreter is unwinding. The first such
//! signal is recorded in `ctx.fatal`; the walk then short-circuits (every later
//! [`member`] call returns at once) and the entry point re-raises it, so an
//! interrupted check stops instead of being silently reported as a non-member.

use pyo3::exceptions::{PyException, PyMemoryError, PyRecursionError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBytes, PyDict, PyFrozenSet, PyList, PySet, PyString, PyTuple, PyType};
use valgebra_core::{
    ClassIx, CollKind, ConstIx, DefIx, OperandIx, PathSegment, PredIx, Schema, Violation,
};

use crate::check::ctx::{Ctx, Entered, MAX_RECURSION_DEPTH, MAX_WALK_DEPTH, WalkMode};
use crate::check::violation::{summarize_value, type_mismatch};
use crate::errors::{class_label, summarize};
use crate::input::Value;

mod record;
mod scalar;
mod sequence;

use record::{Decided, check_attr_record, keyed_map_explain, keyed_map_matches};
use scalar::{admit, check_literal, check_refine, homogeneous_scalar, scalar_admits, scalar_of};
use sequence::{check_frozenset, check_seq, check_set};

/// Where a walk is, and what it has found there, beside the context it reads.
///
/// The three travel together through every arm of the walk, and passing them
/// one at a time put the same three names in fourteen signatures and on every
/// recursive call. They are one value here, and a walk that needs a different
/// one -- a union probing a branch into its own buffer, a complement deciding
/// its inner schema on the fast path -- builds one from the parts it keeps,
/// which is why the three are named rather than folded into methods: what a
/// sub-walk changes differs at every site.
pub(crate) struct Frame<'a, 'ctx> {
    /// Where the walk is in the value: the location a violation is reported at.
    /// Written only in the modes that explain, since a fast walk reports none.
    pub(crate) path: &'a mut Vec<PathSegment>,
    /// What the walk has found. A fast walk writes to a buffer nothing reads.
    pub(crate) out: &'a mut Vec<Violation>,
    /// What the walk may look up, and what it is being run for.
    pub(crate) ctx: Ctx<'ctx>,
}

impl<'a, 'ctx> Frame<'a, 'ctx> {
    /// A frame over a caller's buffers.
    pub(crate) fn new(
        path: &'a mut Vec<PathSegment>,
        out: &'a mut Vec<Violation>,
        ctx: Ctx<'ctx>,
    ) -> Self {
        Frame { path, out, ctx }
    }
}

/// Every base container's own `__len__`, and the two bases' own `__iter__`,
/// resolved once per process.
///
/// Two tables rather than one keyed by which slot is wanted: a caller asks for
/// a length or for the elements, never for "whichever of the two", and folding
/// them into one reading put a match on the per-element path of every walk that
/// meets a subclass.
static BASE_LENGTHS: PyOnceLock<Slots> = PyOnceLock::new();
static BASE_ITERS: PyOnceLock<Slots> = PyOnceLock::new();

/// Which builtin container a storage question is about.
///
/// The helpers below each ask the same question of one of these, and a `bool`
/// cannot name which: `held_len(value, true)` reads as a length that *is* held,
/// which is what the function does either way. Every member has a name, so it is
/// an enum and the call sites say the name -- the rule this library applies to a
/// Python value it is handed, turned on its own arguments.
///
/// Every container whose contents the walk reads is a member, because the
/// argument below is about all of them alike: a `set` subclass whose `__iter__`
/// yields values its storage does not hold was a member of `set[int]` while
/// holding a `str`, and a `str` subclass whose `__len__` answers nine over one
/// character satisfied `MinLen(3)` over text nothing counts that way.
#[derive(Clone, Copy)]
pub(super) enum Base {
    List,
    Tuple,
    Str,
    Bytes,
    Set,
    FrozenSet,
    Dict,
}

impl Base {
    /// The type this base names.
    fn type_object(self, py: Python<'_>) -> Bound<'_, PyType> {
        match self {
            Base::List => py.get_type::<PyList>(),
            Base::Tuple => py.get_type::<PyTuple>(),
            Base::Str => py.get_type::<PyString>(),
            Base::Bytes => py.get_type::<PyBytes>(),
            Base::Set => py.get_type::<PySet>(),
            Base::FrozenSet => py.get_type::<PyFrozenSet>(),
            Base::Dict => py.get_type::<PyDict>(),
        }
    }
}

/// One attribute of every base, held by name.
///
/// By name rather than by position, so reading one is a field select rather
/// than a bounds-checked load through a heap pointer: this sits on the path of
/// every walk that meets a container subclass, and the answer is a table entry
/// rather than a decision.
struct Slots {
    list: Py<PyAny>,
    tuple: Py<PyAny>,
    string: Py<PyAny>,
    bytes: Py<PyAny>,
    set: Py<PyAny>,
    frozen_set: Py<PyAny>,
    dict: Py<PyAny>,
}

impl Slots {
    /// Resolve `name` on every base, once per process.
    fn of(py: Python<'_>, name: &Bound<'_, PyString>) -> PyResult<Slots> {
        let on = |base: Base| base.type_object(py).getattr(name).map(Bound::unbind);
        Ok(Slots {
            list: on(Base::List)?,
            tuple: on(Base::Tuple)?,
            string: on(Base::Str)?,
            bytes: on(Base::Bytes)?,
            set: on(Base::Set)?,
            frozen_set: on(Base::FrozenSet)?,
            dict: on(Base::Dict)?,
        })
    }

    fn get(&self, base: Base) -> &Py<PyAny> {
        match base {
            Base::List => &self.list,
            Base::Tuple => &self.tuple,
            Base::Str => &self.string,
            Base::Bytes => &self.bytes,
            Base::Set => &self.set,
            Base::FrozenSet => &self.frozen_set,
            Base::Dict => &self.dict,
        }
    }
}

/// The base type's own `__len__`.
fn base_length(py: Python<'_>, base: Base) -> PyResult<&'static Py<PyAny>> {
    let name = intern!(py, "__len__");
    Ok(BASE_LENGTHS
        .get_or_try_init(py, || Slots::of(py, name))?
        .get(base))
}

/// The base type's own `__iter__`.
fn base_iter(py: Python<'_>, base: Base) -> PyResult<&'static Py<PyAny>> {
    let name = intern!(py, "__iter__");
    Ok(BASE_ITERS
        .get_or_try_init(py, || Slots::of(py, name))?
        .get(base))
}

/// How many items a container *subclass* holds.
///
/// The walk counts what the value holds, which is the storage rather than the
/// answer `__len__` gives -- a subclass may override it and say anything, and
/// [`scalar::stored_len`] has refused to believe it since a length marker and
/// the shape beside it described two different sets.
///
/// The C accessor is not the way to read the storage either. `PyTuple_Size`
/// reads it on `CPython`; `PyPy`'s `cpyext` implements it *through the object's
/// own* `__len__`, so it is the overridden answer again under another name, and
/// a walk that indexes against it runs off the end of the allocation: a
/// `tuple` subclass reporting ten over one element takes the process down.
/// Asking past the end does not report cleanly there either: the accessor
/// answers with nothing and sets no exception.
///
/// So the length comes from the base type's own slot, called on the value. It
/// is the definition the page already gives ("the items the value holds"), it
/// is what a reader checks by hand with `tuple.__len__(value)`, and it means
/// the same thing on every interpreter.
///
/// Only a subclass that **overrides** `__len__` pays for it. One that inherits
/// the base's slot is read where it lies: see [`reads_its_length`].
fn held_len(value: &Bound<'_, PyAny>, base: Base) -> PyResult<usize> {
    let py = value.py();
    base_length(py, base)?.bind(py).call1((value,))?.extract()
}

/// The items a container *subclass* holds, over the base type's own `__iter__`.
///
/// The set kinds are read through an iterator rather than by position, so this
/// is where their storage is reached: a subclass overriding `__iter__` yields
/// whatever it likes, and a walk believing it decides membership of a set that
/// is not the value. The iterator the base slot returns is the builtin one, so
/// mutation during the scan still raises where [`sequence::scan_set`] expects
/// it to.
fn held_iter<'py>(value: &Bound<'py, PyAny>, base: Base) -> PyResult<Bound<'py, PyAny>> {
    let py = value.py();
    base_iter(py, base)?.bind(py).call1((value,))
}

/// Whether this value's type reports the length of its own storage.
///
/// The accessor is untrustworthy for exactly one reason: `cpyext` implements
/// `PyTuple_Size` through the object's own `__len__`, so a subclass that
/// **overrides** it answers the C level with whatever it likes. A subclass that
/// *inherits* the slot does not -- the overridden answer and the base's answer
/// are the same function -- and the accessor reads the storage on every
/// interpreter, as it does for an exact tuple.
///
/// So the question is not "is this exactly a tuple" but "is this type's
/// `__len__` the base's own", asked by identity. A `NamedTuple` answers yes,
/// which is what makes the common tuple subclass cost what a tuple costs; the
/// subclass that returns ten over one element answers no, and is copied.
///
/// Asked of the type on every walk that reaches a subclass, and **not
/// remembered**: a map from type to answer was built and measured beside this,
/// and a dict probe costs what the type lookup costs -- 67.6 ns against 67.6 on
/// the fixed-arity `NamedTuple` row. A cache that buys nothing is a global
/// mutable object, a bound, and a free-threading argument for nothing.
///
/// A wrong answer here is safe in one direction only, and this errs that way: a
/// type that cannot be read at all is treated as a liar and read through the
/// base.
#[inline]
fn reads_its_length(value: &Bound<'_, PyAny>, base: Base) -> bool {
    let py = value.py();
    let name = intern!(py, "__len__");
    let Ok(slots) = BASE_LENGTHS.get_or_try_init(py, || Slots::of(py, name)) else {
        return false;
    };
    carries(value, name, slots.get(base))
}

/// Whether this value's type yields the elements of its own storage.
///
/// [`reads_its_length`]'s question, one slot over, and it is the one the set
/// kinds turn on: they are read through an iterator rather than by position.
fn reads_its_elements(value: &Bound<'_, PyAny>, base: Base) -> bool {
    let py = value.py();
    let name = intern!(py, "__iter__");
    let Ok(slots) = BASE_ITERS.get_or_try_init(py, || Slots::of(py, name)) else {
        return false;
    };
    carries(value, name, slots.get(base))
}

/// Whether the value's type carries `name` as the base's own slot.
fn carries(
    value: &Bound<'_, PyAny>,
    name: &Bound<'_, PyString>,
    of_base: &'static Py<PyAny>,
) -> bool {
    value
        .get_type()
        .getattr_opt(name)
        .ok()
        .flatten()
        .is_some_and(|found| found.is(of_base.bind(value.py())))
}

fn stop(ctx: Ctx<'_>) -> bool {
    ctx.mode.stops_at_first()
}

/// Whether a raised error is a *fatal* interpreter signal that must propagate
/// rather than fold to non-membership. Two disjoint cases: a base exception that
/// is not an ordinary exception (`KeyboardInterrupt`, `SystemExit`,
/// `GeneratorExit`), and `MemoryError`/`RecursionError` — which *are* ordinary
/// exceptions, so the `PyException` test alone misses them, yet they mean "the
/// interpreter cannot continue", not "this value is not a member". Any other
/// exception is an ordinary failed comparison and folds to a non-member.
pub(super) fn is_fatal(err: &PyErr, py: Python<'_>) -> bool {
    !err.is_instance_of::<PyException>(py)
        || err.is_instance_of::<PyMemoryError>(py)
        || err.is_instance_of::<PyRecursionError>(py)
}

/// Record the first fatal signal so the walk unwinds (every later `member` call
/// returns at once) and the entry point re-raises it.
pub(super) fn record_fatal(err: PyErr, ctx: Ctx<'_>) {
    let mut slot = ctx.fatal.borrow_mut();
    if slot.is_none() {
        *slot = Some(err);
    }
    // Mirror into the cheap flag the per-node short-circuit reads.
    ctx.fatal_seen.set(true);
}

/// Fold a membership probe's result into a boolean. An ordinary exception means
/// the value cannot answer "are you in this set?", so it is a non-member. A fatal
/// interpreter signal is recorded in `ctx.fatal` so the walk unwinds and the
/// entry point re-raises it, and reported locally as a non-member so the current
/// frame returns.
fn fold(result: PyResult<bool>, py: Python<'_>, ctx: Ctx<'_>) -> bool {
    match result {
        Ok(holds) => holds,
        Err(err) => {
            if is_fatal(&err, py) {
                record_fatal(err, ctx);
            }
            false
        }
    }
}

/// Bind a pooled object by slot, or `None` when the slot is out of range. Every
/// IR index is in range by construction (the builder fills the pool), so a miss is
/// an internal invariant break unreachable from user input; the walk degrades to a
/// non-member rather than panicking across the language boundary.
///
/// Private, and reached only through the four typed accessors below: this is the
/// one place an index space stops being tracked, so the pool's four uses each
/// name themselves at the call site.
fn pool_slot<'a, 'py>(ctx: Ctx<'a>, slot: usize, py: Python<'py>) -> Option<&'a Bound<'py, PyAny>> {
    let obj = ctx.pool.get(slot);
    debug_assert!(obj.is_some(), "pool index {slot} out of range");
    // Borrowed, not cloned: the pool outlives the walk, and a clone here is a
    // reference-count round trip per literal compared and per class checked.
    obj.map(|object| object.bind(py))
}

/// The constant behind a [`Schema::Literal`].
fn const_at<'a, 'py>(
    ctx: Ctx<'a>,
    index: ConstIx,
    py: Python<'py>,
) -> Option<&'a Bound<'py, PyAny>> {
    pool_slot(ctx, index.get(), py)
}

/// The class behind a [`Schema::Instance`].
fn class_at<'a, 'py>(
    ctx: Ctx<'a>,
    index: ClassIx,
    py: Python<'py>,
) -> Option<&'a Bound<'py, PyAny>> {
    pool_slot(ctx, index.get(), py)
}

/// The operand behind a comparison or multiple-of constraint.
fn operand_at<'a, 'py>(
    ctx: Ctx<'a>,
    index: OperandIx,
    py: Python<'py>,
) -> Option<&'a Bound<'py, PyAny>> {
    pool_slot(ctx, index.get(), py)
}

/// The callable behind a [`Constraint::Predicate`].
fn predicate_at<'a, 'py>(
    ctx: Ctx<'a>,
    index: PredIx,
    py: Python<'py>,
) -> Option<&'a Bound<'py, PyAny>> {
    pool_slot(ctx, index.get(), py)
}

/// Decide whether `value` is a member of `schema`'s set.
///
/// In explain mode a [`Violation`] is pushed into `out` for every independent
/// failure and `path` accumulates the location of the current value; in fast
/// mode nothing is allocated. The returned bool is authoritative: it is the same
/// answer `is_valid` and `validate` report.
pub(crate) fn member(schema: &Schema, value: &Value<'_, '_>, frame: &mut Frame<'_, '_>) -> bool {
    let ctx = frame.ctx;
    // A fatal interpreter signal recorded earlier in the walk unwinds the whole
    // traversal: every remaining node reports a non-member at once, so a large
    // value stops promptly instead of finishing the walk after a KeyboardInterrupt.
    if ctx.fatal_seen.get() {
        return false;
    }
    // One level of the walk is one native stack frame, so the walk counts its own
    // levels rather than trusting the value to be shallow. A recursive definition
    // unfolds once per level of the value and descends its whole body each time,
    // so the frames a value demands are the product of the two construction
    // bounds; the counter bounds that product, and a value that reaches it is
    // refused the way an over-deep one already is.
    let Some(_level) = ctx.descend() else {
        if ctx.mode.explains() {
            frame.out.push(Violation {
                code: "recursion_limit",
                path: frame.path.clone(),
                expected: format!("at most {MAX_WALK_DEPTH} levels of nesting"),
                value_summary: summarize_value(value, ctx),
            });
        }
        return false;
    };
    match schema {
        Schema::Anything(_) => true,
        // Bottom admits nothing; an unresolved self-reference is never a member.
        Schema::Nothing => admit(false, schema, value, frame),
        Schema::SelfRef(_) => {
            if ctx.mode.explains() {
                frame.out.push(Violation {
                    code: "unresolved_recursion",
                    path: frame.path.clone(),
                    expected: "a resolved recursive value".to_owned(),
                    value_summary: summarize_value(value, ctx),
                });
            }
            false
        }
        Schema::NoneType => admit(value.is_none(), schema, value, frame),
        Schema::Bool => admit(value.is_bool(), schema, value, frame),
        // bool subclasses int, so True/False are ints: Bool is a subset of Int.
        Schema::Int => admit(value.is_int(), schema, value, frame),
        Schema::Float => admit(value.is_float(), schema, value, frame),
        Schema::Str => admit(value.is_str(), schema, value, frame),
        Schema::Bytes => admit(value.is_bytes(), schema, value, frame),
        Schema::Literal(index) => check_literal(*index, value, frame),
        Schema::Seq { container, shape } => check_seq(*container, shape, value, frame),
        Schema::Coll { container, element } => match container {
            CollKind::Set => check_set(element, value, frame),
            CollKind::FrozenSet => check_frozenset(element, value, frame),
        },
        Schema::KeyedMap { fields, defaults } => {
            // Membership is the single-pass fast check; on failure the explain
            // pass re-walks in declared order to aggregate ordered violations,
            // resuming where the first pass stopped rather than re-reading the
            // fields it already found to match.
            let mut decided = Decided::default();
            let keep = ctx.mode.explains().then_some(&mut decided);
            let ok = keyed_map_matches(fields, defaults, value, ctx, keep);
            if !ok && ctx.mode.explains() {
                let before = frame.out.len();
                keyed_map_explain(fields, defaults, value, frame, &decided);
                if frame.out.len() == before {
                    // Two passes read the same dict and disagreed, so the dict
                    // did not stay still between them: report that rather than a
                    // failure with nothing behind it.
                    mutated(value, frame);
                }
            }
            ok
        }
        Schema::Union(members) => check_union(members, value, frame),
        Schema::Intersection(members) => check_intersection(members, value, frame),
        Schema::Complement(inner) => check_complement(inner, value, frame),
        Schema::Instance(index) => check_instance(*index, value, frame),
        Schema::AttrRecord { fields } => check_attr_record(fields, value, frame),
        Schema::Refine { base, constraints } => check_refine(base, constraints, value, frame),
        Schema::Ref(id) => check_ref(*id, value, frame),
    }
}

/// What a scan over a container the walk does not own produced.
///
/// A container can change while it is being read: membership runs arbitrary
/// Python at every entry — a predicate, an `__eq__`, an `isinstance` hook — and a
/// free-threaded interpreter lets another thread write to it meanwhile. The scan
/// therefore has a third outcome beside "walked it all" and "stopped early".
enum Scan {
    /// Every entry was visited.
    Complete,
    /// The visitor stopped the scan before the end.
    Stopped,
    /// The container could not be read to the end, so there is no reading of its
    /// contents to answer from. Membership reports a non-member and names the
    /// mutation rather than answering from the part it managed to see.
    Unreadable,
}

/// The code and message a value that changed under the walk reports.
///
/// valgebra-coined, because it describes a failure of the *check* rather than of
/// the value: nothing about the value's contents was decided. Two shapes reach
/// it — a container whose entries move while they are being read, and a value
/// whose two readings disagree because something it runs is not a function of
/// the value.
const MUTATED_CODE: &str = "mutated_during_validation";
const MUTATED_EXPECTED: &str = "a value that does not change while it is checked";

/// Record that a container changed under the walk, and report a non-member.
fn mutated(value: &Value<'_, '_>, frame: &mut Frame<'_, '_>) -> bool {
    let ctx = frame.ctx;
    if ctx.mode.explains() {
        frame.out.push(Violation {
            code: MUTATED_CODE,
            path: frame.path.clone(),
            expected: MUTATED_EXPECTED.to_owned(),
            value_summary: summarize_value(value, ctx),
        });
    }
    false
}

/// Cap on how many branches the closest-branch error probe re-walks. The
/// membership decision has already scanned every branch to confirm non-matching;
/// this bounds the *second*, explain-mode pass so building the error for a
/// pathologically wide union (a large `Literal[...]`, say) stays linear in the
/// cap rather than the branch count. Beyond the cap the report falls back to the
/// union summary. Error-path only — the membership result is never affected.
const CLOSEST_BRANCH_PROBE_LIMIT: usize = 64;

/// The most labels a union's `expected` names before it truncates.
///
/// A separate bound from the probe above, on a separate quantity. The probe
/// bounds how many branches are *walked*, which costs a descent each; this
/// bounds how many labels are *named*, which costs a string. They are not the
/// same count even for one union: the label pass flattens a nested union, and
/// `Literal[...]` is a union of its constants, so two branches can yield a
/// hundred labels.
const UNION_LABEL_LIMIT: usize = 64;

/// The branch labels of a union, collected until the limit and no further.
struct BranchLabels {
    rendered: Vec<String>,
    truncated: bool,
}

impl BranchLabels {
    fn new() -> Self {
        Self {
            rendered: Vec::new(),
            truncated: false,
        }
    }

    /// Take `label` unless the limit is reached, in which case record that the
    /// list is short rather than growing it.
    fn push(&mut self, label: String) {
        if self.rendered.len() < UNION_LABEL_LIMIT {
            self.rendered.push(label);
        } else {
            self.truncated = true;
        }
    }

    fn render(&self) -> String {
        let joined = self.rendered.join(", ");
        if self.truncated {
            format!("one of: {joined}, ...")
        } else {
            format!("one of: {joined}")
        }
    }
}

/// Name `schema` the way it names itself when it is the only thing that failed.
///
/// [`Schema::expected`] gives a node's *kind*, which is the least informative
/// thing about a literal or an instance: a union of permitted strings joined by
/// kind reads `one of: literal, literal`. The concrete name needs the pool, and
/// the pool lives in this crate, so the rendering does too and the core keeps
/// the kind as the fallback for every node with nothing better to say.
///
/// A nested union contributes its members rather than itself, because
/// `Literal[...]` builds one: without this, a single-constant `Literal` names
/// itself `union`. A reference contributes the definition's, for the same
/// reason: `recursive(...)` is a `Ref`, and its bare kind is the word `value`.
///
/// `unfolded` bounds the references followed, so a definition naming another
/// names a depth of them rather than running forever. The bound is small
/// because a label is prose: past it the node's own kind is the honest answer.
fn push_branch_label(schema: &Schema, ctx: Ctx<'_>, py: Python<'_>, out: &mut BranchLabels) {
    push_branch_label_within(schema, ctx, py, out, 0);
}

/// How many references a branch label follows before naming the node's kind.
const MAX_LABEL_UNFOLDS: u32 = 4;

fn push_branch_label_within(
    schema: &Schema,
    ctx: Ctx<'_>,
    py: Python<'_>,
    out: &mut BranchLabels,
    unfolded: u32,
) {
    match schema {
        Schema::Union(members) => {
            for member in members.iter() {
                push_branch_label_within(member, ctx, py, out, unfolded);
            }
        }
        // What a complement excludes, which is what it says when it is the only
        // thing that failed: `not str` rather than the word `complement`.
        Schema::Complement(inner) => out.push(format!("not {}", inner.expected())),
        // The definition's own branches. A reference is the node `recursive`
        // builds, and its kind alone tells a reader nothing about what it admits.
        Schema::Ref(id) => match ctx.defs.get(id.get()) {
            Some(body) if unfolded < MAX_LABEL_UNFOLDS => {
                push_branch_label_within(body, ctx, py, out, unfolded + 1);
            }
            _ => out.push(schema.expected().to_owned()),
        },
        Schema::Literal(index) => {
            let label = const_at(ctx, *index, py).map_or_else(
                || schema.expected().to_owned(),
                |c| format!("the literal {}", summarize(c)),
            );
            out.push(label);
        }
        Schema::Instance(index) => out.push(class_name(*index, schema, ctx, py)),
        // A class with declared attributes is a meet of an atom and a record, and
        // the branch names the class the user wrote rather than the algebra's
        // spelling of it.
        Schema::Intersection(_) => match schema.object_class() {
            Some(class) => out.push(class_name(class, schema, ctx, py)),
            None => out.push(schema.expected().to_owned()),
        },
        // A refinement's type is its base, matching `Schema::expected`; the
        // constraints report themselves when one of them is what failed.
        Schema::Refine { base, .. } => push_branch_label_within(base, ctx, py, out, unfolded),
        other => out.push(other.expected().to_owned()),
    }
}

/// The pooled class's own name, falling back to the node's kind when the pool
/// cannot be read.
fn class_name(index: ClassIx, schema: &Schema, ctx: Ctx<'_>, py: Python<'_>) -> String {
    class_at(ctx, index, py).map_or_else(|| schema.expected().to_owned(), |c| class_label(c))
}

/// Whether this code says the *walk* stopped rather than that a value is outside
/// a set.
///
/// A union summary stands in for branches that did not match, and each of these
/// is something else: the walk ran out of levels, the value contains itself, it
/// moved while it was read, or a predicate raised. Folding one into "this value
/// matched no branch" drops the only sentence that says what to do about it.
fn walk_declined(code: &str) -> bool {
    matches!(
        code,
        "recursion_limit" | "recursion_loop" | "mutated_during_validation" | "predicate_error"
    )
}

fn check_union(members: &[Schema], value: &Value<'_, '_>, frame: &mut Frame<'_, '_>) -> bool {
    let ctx = frame.ctx;
    // Fast path for an all-literal union: an exact int or str value is decided by
    // a single set lookup. Only the membership decision uses it; the explain walk
    // below, and every value type the plan does not cover, fall through to the
    // linear scan, which stays the one source of truth for behavior.
    if !ctx.mode.explains()
        && let Some(plan) = ctx.unions.get(&(members.as_ptr() as usize))
        && let Some(decided) = plan.decide(value)
    {
        return decided;
    }
    if ctx.mode.explains() {
        return explain_union(members, value, frame);
    }
    // A value is a member iff it matches at least one branch; decide that on the
    // fast path, where a discarded branch pays for no location or violation.
    let sub = fast(ctx);
    members.iter().any(|m| {
        member(
            m,
            value,
            &mut Frame::new(&mut Vec::new(), &mut Vec::new(), sub),
        )
    })
}

/// Decide a union **and** explain it in one walk of each branch.
///
/// The two questions were asked separately: a fast pass over every branch to
/// decide membership, then -- on failure -- a probe that walked every branch
/// again in explain mode to find the closest one. Each pass is linear in the
/// subtree, and the probe's walk reaches the next union one level down, which
/// did the same thing to the subtree below *it*. The result was quadratic in
/// the depth of the value: a 5,000-deep value took 0.8 s to explain, 10,000
/// took 3.3, and 20,000 took 13, against `is_valid` at 1.6 ms for the same
/// 20,000.
///
/// A branch walked in explain mode already answers both: it returns whether it
/// matched, and it reports what failed if it did not. Asking once makes the
/// recursion linear, and keeps the walk a second question would throw away.
fn explain_union(members: &[Schema], value: &Value<'_, '_>, frame: &mut Frame<'_, '_>) -> bool {
    let ctx = frame.ctx;
    // The *closest* branch -- the one that descended furthest into the value
    // before failing -- is reported, rather than every branch. "Furthest" is the
    // greatest path depth past the union's own location. Where no branch makes
    // progress (`int | str` against a float, say) a single union error stands
    // for all of them. Every branch is walked whole whatever the mode, since
    // the depth each reached is what chooses between them and a walk stopped
    // early has not measured it; the mode then decides how much of the chosen
    // branch is reported. This runs only where a value is being explained.
    let base_depth = frame.path.len();
    let probe = Ctx {
        mode: WalkMode::Explain,
        ..ctx
    };
    let mut best: Option<(usize, Vec<Violation>)> = None;
    // The first branch the walk could not answer for, kept aside. A branch that
    // ran out of levels, found the value inside itself, or raised inside a
    // predicate has not said "this value is not a member" -- it has said the
    // walk stopped -- and its failure sits at the union's own location, so the
    // progress rule below would fold it into a summary and drop the reason.
    let mut declined: Option<Vec<Violation>> = None;
    for (position, branch_schema) in members.iter().enumerate() {
        if position >= CLOSEST_BRANCH_PROBE_LIMIT {
            // Past the probe's width the branch is asked the cheap question
            // only: a union this wide reports the closest of the branches
            // already walked, and the rest merely decide membership.
            if member(
                branch_schema,
                value,
                &mut Frame::new(&mut Vec::new(), &mut Vec::new(), fast(ctx)),
            ) {
                return true;
            }
            continue;
        }
        let mut branch = Vec::new();
        let matched = {
            let mut probing = Frame::new(&mut *frame.path, &mut branch, probe);
            member(branch_schema, value, &mut probing)
        };
        if matched {
            return true;
        }
        let progress = branch
            .iter()
            .map(|v| v.path.len())
            .max()
            .unwrap_or(base_depth)
            .saturating_sub(base_depth);
        if declined.is_none() && branch.iter().any(|v| walk_declined(v.code)) {
            declined = Some(branch.clone());
        }
        // Strictly greater keeps the earliest branch on a tie.
        let replace = best
            .as_ref()
            .is_none_or(|(best_progress, _)| progress > *best_progress);
        if replace {
            best = Some((progress, branch));
        }
    }
    if let Some(branch) = declined.filter(|_| best.as_ref().is_none_or(|(p, _)| *p == 0)) {
        let reported = if ctx.mode.stops_at_first() {
            1
        } else {
            branch.len()
        };
        frame.out.extend(branch.into_iter().take(reported));
        return false;
    }
    match best {
        // The branch is walked whole whatever the mode, because the *closest*
        // one is chosen by how far each descended and a walk stopped early has
        // not measured that. What the mode decides is how much of the chosen
        // branch is reported: `fail_fast` promises one violation, and the one
        // it keeps is the one the aggregate would lead with.
        Some((progress, branch)) if progress > 0 => {
            let reported = if ctx.mode.stops_at_first() {
                1
            } else {
                branch.len()
            };
            frame.out.extend(branch.into_iter().take(reported));
        }
        _ => {
            let mut labels = BranchLabels::new();
            for member in members {
                push_branch_label(member, ctx, value.py(), &mut labels);
            }
            frame.out.push(Violation {
                code: "union_error",
                path: frame.path.clone(),
                expected: labels.render(),
                value_summary: summarize_value(value, ctx),
            });
        }
    }
    false
}

fn check_intersection(
    members: &[Schema],
    value: &Value<'_, '_>,
    frame: &mut Frame<'_, '_>,
) -> bool {
    let ctx = frame.ctx;
    // Every member must hold; in explain mode each member's failure is collected,
    // until one rejects the value itself.
    let mut ok = true;
    for member_schema in members {
        let before = frame.out.len();
        ok &= member(member_schema, value, frame);
        if !ok && (stop(ctx) || rejected_the_value(frame.out, before, frame.path.len())) {
            return false;
        }
    }
    ok
}

/// Whether the violations recorded since `before` include one about the value at
/// the current path, rather than about something inside it.
///
/// A member that rejects the value *itself* has settled the meet, and what the
/// remaining members would say describes a value that is already the wrong kind
/// of thing: an attribute record beside a class atom reports missing attributes
/// on an object that is not an instance of the class, which is not a second
/// problem with the value but the same one, said again about a value that never
/// had to have those attributes. A member that fails *inside* the value -- an
/// element, a field, an attribute -- leaves the others meaningful, and they are
/// still collected.
///
/// This is the rule [`scalar::check_refine`] already applies between a base and its
/// constraints, said once for the meet: `Annotated[int, Gt(0)]` does not report
/// a bound on a string.
fn rejected_the_value(out: &[Violation], before: usize, depth: usize) -> bool {
    // The walk only appends to `path` as it descends, so a violation whose path
    // is as long as the current one is at the current one.
    out.get(before..)
        .is_some_and(|since| since.iter().any(|v| v.path.len() == depth))
}

fn check_complement(inner: &Schema, value: &Value<'_, '_>, frame: &mut Frame<'_, '_>) -> bool {
    let ctx = frame.ctx;
    // A value matches the complement iff it does not match the inner schema; the
    // inner explanation is irrelevant, so decide it on the fast path.
    if member(
        inner,
        value,
        &mut Frame::new(&mut Vec::new(), &mut Vec::new(), fast(ctx)),
    ) {
        if ctx.mode.explains() {
            frame.out.push(Violation {
                code: "unexpected_match",
                path: frame.path.clone(),
                expected: format!("not {}", inner.expected()),
                value_summary: summarize_value(value, ctx),
            });
        }
        return false;
    }
    true
}

fn check_instance(index: ClassIx, value: &Value<'_, '_>, frame: &mut Frame<'_, '_>) -> bool {
    let ctx = frame.ctx;
    let Some(class) = class_at(ctx, index, value.py()) else {
        return false;
    };
    let ok = fold(
        value.to_python().and_then(|obj| obj.is_instance(class)),
        value.py(),
        ctx,
    );
    if !ok && ctx.mode.explains() {
        frame.out.push(type_mismatch(
            "instance_type",
            &class_label(class),
            value,
            frame.path,
            ctx,
        ));
    }
    ok
}

/// Whether this value is a member of the definition the reference names.
///
/// The reference is entered on the trail for the length of the definition's
/// walk, so a value reached from inside itself meets its own pair and is
/// refused as cyclic rather than walked forever. Each of the three refusals
/// below leaves the trail as it found it: two are refused before a level is
/// opened, and the third closes the level it opened.
fn check_ref(id: DefIx, value: &Value<'_, '_>, frame: &mut Frame<'_, '_>) -> bool {
    let ctx = frame.ctx;
    let key = (value.id(), id.get());
    match ctx.guard.borrow_mut().enter(key) {
        Entered::Open => {}
        Entered::Cycle => {
            if ctx.mode.explains() {
                frame.out.push(Violation {
                    code: "recursion_loop",
                    path: frame.path.clone(),
                    expected: "a finite (non-cyclic) value".to_owned(),
                    value_summary: summarize_value(value, ctx),
                });
            }
            return false;
        }
        Entered::Full => {
            if ctx.mode.explains() {
                frame.out.push(Violation {
                    code: "recursion_limit",
                    path: frame.path.clone(),
                    expected: format!("at most {MAX_RECURSION_DEPTH} levels of recursion"),
                    value_summary: summarize_value(value, ctx),
                });
            }
            return false;
        }
    }
    let Some(def) = ctx.defs.get(id.get()) else {
        // A reference past the definitions table is an internal invariant break,
        // not reachable from user input; release builds degrade to a non-member
        // rather than panicking across the language boundary.
        debug_assert!(false, "definition index {} out of range", id.get());
        ctx.guard.borrow_mut().leave();
        return false;
    };
    let result = member(def, value, frame);
    ctx.guard.borrow_mut().leave();
    result
}

/// A copy of `ctx` switched to the membership fast path (no explanation), for the
/// speculative sub-checks of union, complement, and the record fast walk.
fn fast(ctx: Ctx<'_>) -> Ctx<'_> {
    Ctx {
        mode: WalkMode::Fast,
        ..ctx
    }
}

// Read by the index's tests, which check that the literal table the index
// builds decides equality the way the walk does. Declared here, below every
// item of the walk itself, because a `cfg(test)` gate above them hides them
// from the tools that read this file for what it defines.
//
// Gated on the feature its one reader is gated on, not on `test` alone: the
// reader needs a live interpreter, so a default-feature test build compiles
// this re-export and nothing that uses it, and warns on every such build. The
// lint lane passes `--all-features` and sees a file that has one.
#[cfg(all(test, feature = "interpreter-tests"))]
pub(crate) use scalar::literal_matches;

// Needs a live interpreter; compiled and run only under the `interpreter-tests`
// feature, which links an embedded Python. This is the walk's own harness: it
// drives real Python values through `member` so the membership decision — where
// soundness is decided, and the one surface the Python suite covers from outside
// but no `cargo test` reaches — carries evidence a mutation sweep can observe.
#[cfg(all(test, feature = "interpreter-tests"))]
mod interpreter;

#[cfg(test)]
mod label_tests;
