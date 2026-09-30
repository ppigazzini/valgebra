//! The validation walk: one membership test of a value against the IR.
//!
//! [`member`] is the single walk. It returns whether the value belongs to the
//! schema's set, and in an *explain* mode (`ctx.mode`) it also aggregates a
//! [`Violation`] for each independent failure into `out` (each record field,
//! each sequence element, each mapping entry), unless the fail-fast mode stops it
//! at the first. In *fast* mode it builds no violation, path or value summary and
//! short-circuits as soon as membership is decided; storage a reading needs for
//! itself, such as a list's snapshot or a copy of a tuple subclass, is allocated
//! at the site that says why.
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

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBytes, PyDict, PyFrozenSet, PyList, PySet, PyString, PyTuple, PyType};
use valgebra_core::{
    ClassIx, CollKind, ConstIx, DefIx, OperandIx, PathSegment, PredIx, Schema, Violation,
};

use crate::check::ctx::{Ctx, Entered, MAX_RECURSION_DEPTH, MAX_WALK_DEPTH, WalkMode};
use crate::check::violation::{class_label_in, summarize_in, summarize_value, type_mismatch};
use crate::codes::{
    Code, INSTANCE_TYPE, MUTATED_DURING_VALIDATION, PREDICATE_ERROR, RECURSION_LIMIT,
    RECURSION_LOOP, UNEXPECTED_MATCH, UNION_ERROR, UNRESOLVED_RECURSION,
};
use crate::input::Value;

mod record;
mod scalar;
mod sequence;

use record::{Decided, check_attr_record, keyed_map_explain, keyed_map_matches};
use scalar::{
    admit, check_literal, check_refine, homogeneous_scalar, is_of_its_kind, scalar_admits,
    scalar_member, scalar_of,
};
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

/// `dict.__getitem__` and `dict.copy`, resolved once per process: the slot
/// `PyPy` reads a dict's values through, and the method that reads its storage.
#[cfg(PyPy)]
static DICT_GETITEM: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
#[cfg(PyPy)]
static DICT_COPY: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

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
/// [`stored_len`](scalar::stored_len) has refused to believe it since a length
/// marker and the shape beside it described two different sets.
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
/// subclass that returns ten over one element answers no, and is copied. A
/// length read through `PyObject_Size` asks it on every interpreter; the tuple
/// walk reads `PyTuple_GET_SIZE`, which is the storage on `CPython`, and asks it
/// on `PyPy` alone.
///
/// Asked of the type on every walk that reaches a subclass, and **not
/// remembered**: a map from type to answer was built and measured beside this,
/// and a dict probe costs what the type lookup costs -- 67.6 ns against 67.6 on
/// the fixed-arity `NamedTuple` row. A cache that buys nothing is a global
/// mutable object, a bound, and a free-threading argument for nothing.
///
/// A wrong answer here is safe in one direction only, and this errs that way: a
/// type that cannot be read at all is treated as a liar and read through the
/// base, and a fatal signal raised while reading it is recorded.
#[inline]
fn reads_its_length(value: &Bound<'_, PyAny>, base: Base, ctx: Ctx<'_>) -> bool {
    let py = value.py();
    let name = intern!(py, "__len__");
    match BASE_LENGTHS.get_or_try_init(py, || Slots::of(py, name)) {
        Ok(slots) => carries(value, name, slots.get(base), ctx),
        Err(err) => unread(err, py, ctx),
    }
}

/// Whether this value's type yields the elements of its own storage.
///
/// [`reads_its_length`]'s question, one slot over, and it is the one the set
/// kinds turn on: they are read through an iterator rather than by position.
fn reads_its_elements(value: &Bound<'_, PyAny>, base: Base, ctx: Ctx<'_>) -> bool {
    let py = value.py();
    let name = intern!(py, "__iter__");
    match BASE_ITERS.get_or_try_init(py, || Slots::of(py, name)) {
        Ok(slots) => carries(value, name, slots.get(base), ctx),
        Err(err) => unread(err, py, ctx),
    }
}

/// Whether this dict's type reads its values from its own storage.
///
/// [`reads_its_length`]'s question for the slot `PyPy`'s `cpyext` reads a
/// dict's values through: see `record::stored`.
#[cfg(PyPy)]
fn reads_its_values(value: &Bound<'_, PyAny>, ctx: Ctx<'_>) -> bool {
    let py = value.py();
    let name = intern!(py, "__getitem__");
    let own = DICT_GETITEM.get_or_try_init(py, || {
        py.get_type::<PyDict>().getattr(name).map(Bound::unbind)
    });
    match own {
        Ok(own) => carries(value, name, own, ctx),
        Err(err) => unread(err, py, ctx),
    }
}

/// What a dict holds, as an exact dict nothing else reaches: `dict.copy`, the
/// base's own method, called on the value.
#[cfg(PyPy)]
fn held_dict<'py>(value: &Bound<'py, PyDict>) -> PyResult<Bound<'py, PyDict>> {
    let py = value.py();
    let copy = DICT_COPY.get_or_try_init(py, || {
        py.get_type::<PyDict>()
            .getattr(intern!(py, "copy"))
            .map(Bound::unbind)
    })?;
    Ok(copy.bind(py).call1((value,))?.cast_into::<PyDict>()?)
}

/// Whether the value's type carries `name` as the base's own slot. The lookup
/// runs a metaclass's own attribute hook, so it can raise.
fn carries(
    value: &Bound<'_, PyAny>,
    name: &Bound<'_, PyString>,
    of_base: &'static Py<PyAny>,
    ctx: Ctx<'_>,
) -> bool {
    match value.get_type().getattr_opt(name) {
        Ok(found) => found.is_some_and(|found| found.is(of_base.bind(value.py()))),
        Err(err) => unread(err, value.py(), ctx),
    }
}

/// Answer a type whose slot could not be read as one that overrides it, after
/// recording a fatal signal the read raised.
#[cold]
fn unread(err: PyErr, py: Python<'_>, ctx: Ctx<'_>) -> bool {
    record_if_fatal(err, py, ctx);
    false
}

fn stop(ctx: Ctx<'_>) -> bool {
    ctx.mode.stops_at_first()
}

pub(crate) use crate::errors::is_fatal;

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

/// Record `err` if it is a fatal signal, for the walk to unwind and the entry
/// point to re-raise. An ordinary error needs nothing recorded: the site that
/// caught it answers for it.
#[cold]
pub(super) fn record_if_fatal(err: PyErr, py: Python<'_>, ctx: Ctx<'_>) {
    if is_fatal(&err, py) {
        record_fatal(err, ctx);
    }
}

/// Fold a membership probe's result into a boolean. An ordinary exception means
/// the value cannot answer "are you in this set?", so it is a non-member. A fatal
/// interpreter signal is recorded in `ctx.fatal` so the walk unwinds and the
/// entry point re-raises it, and reported locally as a non-member so the current
/// frame returns.
pub(super) fn fold(result: PyResult<bool>, py: Python<'_>, ctx: Ctx<'_>) -> bool {
    result.unwrap_or_else(|err| {
        record_if_fatal(err, py, ctx);
        false
    })
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

/// The callable behind a
/// [`Constraint::Predicate`](valgebra_core::Constraint::Predicate).
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
                code: RECURSION_LIMIT.as_str(),
                path: frame.path.clone(),
                expected: format!("a value at most {MAX_WALK_DEPTH} levels deep"),
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
                    code: UNRESOLVED_RECURSION.as_str(),
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
const MUTATED_CODE: Code = MUTATED_DURING_VALIDATION;
const MUTATED_EXPECTED: &str = "a value that does not change while it is checked";

/// Record that a container changed under the walk, and report a non-member.
fn mutated(value: &Value<'_, '_>, frame: &mut Frame<'_, '_>) -> bool {
    let ctx = frame.ctx;
    if ctx.mode.explains() {
        frame.out.push(Violation {
            code: MUTATED_CODE.as_str(),
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
                |c| format!("the literal {}", summarize_in(c, ctx)),
            );
            out.push(label);
        }
        Schema::Instance(index) => out.push(class_name(*index, schema, ctx, py)),
        // A class with declared attributes is a meet of an atom and a record, and
        // the branch names the class the user wrote rather than the algebra's
        // spelling of it. Any other meet names what each member admits.
        Schema::Intersection(members) => match schema.object_class() {
            Some(class) => out.push(class_name(class, schema, ctx, py)),
            None => out.push(meet_label(members, ctx, py, unfolded)),
        },
        // A refinement's type is its base, matching `Schema::expected`; the
        // constraints report themselves when one of them is what failed.
        Schema::Refine { base, .. } => push_branch_label_within(base, ctx, py, out, unfolded),
        other => out.push(other.expected().to_owned()),
    }
}

/// Name a meet by what each of its members admits, joined with `and`:
/// `int and not bool`, where the kind alone would say `intersection` and name no
/// set. A member that is itself a union names its branches joined with `or`, in
/// parentheses, so the meet stays one entry of the list it is a branch of.
fn meet_label(members: &[Schema], ctx: Ctx<'_>, py: Python<'_>, unfolded: u32) -> String {
    members
        .iter()
        .map(|member| {
            let mut labels = BranchLabels::new();
            push_branch_label_within(member, ctx, py, &mut labels, unfolded);
            let mut joined = labels.rendered.join(" or ");
            if labels.truncated {
                joined.push_str(" or ...");
            }
            if labels.rendered.len() > 1 {
                format!("({joined})")
            } else {
                joined
            }
        })
        .collect::<Vec<_>>()
        .join(" and ")
}

/// The pooled class's own name, falling back to the node's kind when the pool
/// cannot be read.
fn class_name(index: ClassIx, schema: &Schema, ctx: Ctx<'_>, py: Python<'_>) -> String {
    class_at(ctx, index, py)
        .map_or_else(|| schema.expected().to_owned(), |c| class_label_in(c, ctx))
}

/// Whether this code says the *walk* stopped rather than that a value is outside
/// a set.
///
/// A union summary stands in for branches that did not match, and each of these
/// is something else: the walk ran out of levels, the value contains itself, it
/// moved while it was read, or a predicate raised. Folding one into "this value
/// matched no branch" drops the only sentence that says what to do about it.
fn walk_declined(code: &str) -> bool {
    // A string rather than a [`Code`], because this reads one back *off* a
    // violation the core owns, where a code is the text it carries. The names
    // are the table's either way, which is what keeps the set here from
    // drifting from the set reported.
    [
        RECURSION_LIMIT,
        RECURSION_LOOP,
        MUTATED_DURING_VALIDATION,
        PREDICATE_ERROR,
    ]
    .iter()
    .any(|declined| declined.as_str() == code)
}

fn check_union(members: &[Schema], value: &Value<'_, '_>, frame: &mut Frame<'_, '_>) -> bool {
    let ctx = frame.ctx;
    if ctx.mode.explains() {
        return explain_union(members, value, frame);
    }
    // Fast path for an all-literal union: an exact int or str value is decided by
    // a single set lookup. Only the membership decision uses it; the explain walk
    // above, and every value type the plan does not cover, fall through to the
    // linear scan, which stays the one source of truth for behavior. A plan is
    // built only for a union whose every member is a literal, so a union whose
    // first member is not one has none, and is not looked up.
    if matches!(members.first(), Some(Schema::Literal(_)))
        && let Some(plan) = ctx.unions.get(&(members.as_ptr() as usize))
        && let Some(decided) = plan.decide(value)
    {
        return decided;
    }
    // A value is a member iff it matches at least one branch; decide that on the
    // fast path, where a discarded branch pays for no location or violation. A
    // scalar branch is its type test, asked without the walk around it.
    let sub = fast(ctx);
    let room = ctx.room_to_descend();
    members
        .iter()
        .any(|m| match scalar_member(m, value, ctx, room) {
            Some(admitted) => admitted,
            None => member(
                m,
                value,
                &mut Frame::new(&mut Vec::new(), &mut Vec::new(), sub),
            ),
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
    // path depth, past the union's own location, of the branch's first failure,
    // the one its walk records first. Where no branch makes progress (`int |
    // str` against a float, say) a single union error stands for all of them.
    //
    // Each branch is walked in the caller's mode. A fail-fast walk stops at the
    // first failure, which is all the choice reads, so a fail-fast report costs
    // what a fail-fast walk of each branch costs rather than the size of the
    // value; a full report walks each branch whole, and both modes choose the
    // same branch, so the one failure fail-fast keeps is the one the full report
    // leads with. This runs only where a value is being explained.
    let base_depth = frame.path.len();
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
        match decided_quietly(branch_schema, value, ctx) {
            Some(true) => return true,
            Some(false) => continue,
            None => {}
        }
        let mut branch = Vec::new();
        let matched = {
            let mut probing = Frame::new(&mut *frame.path, &mut branch, ctx);
            member(branch_schema, value, &mut probing)
        };
        if matched {
            return true;
        }
        let first = branch.first();
        let progress = first
            .map_or(base_depth, |v| v.path.len())
            .saturating_sub(base_depth);
        if declined.is_none() && first.is_some_and(|v| walk_declined(v.code)) {
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
        // A fail-fast walk recorded one violation, the one the full walk leads
        // with; a full walk recorded every one, and all are reported.
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
                code: UNION_ERROR.as_str(),
                path: frame.path.clone(),
                expected: labels.render(),
                value_summary: summarize_value(value, ctx),
            });
        }
    }
    false
}

/// A union branch whose failure, if it fails, is one no report reads: its
/// answer, decided without explaining it, or `None` for a branch to explain.
///
/// A scalar kind or a literal fails at the union's own location, with a
/// mismatch, and so does any branch whose kind refuses the value before its
/// constraints or contents are read. [`explain_union`] reports such a failure
/// only through the union's label, so explaining the branch built a violation
/// nothing kept -- and summarizing the value in it ran the value's `__repr__`.
/// A value a union *admits* had its repr run once for each branch before the
/// one that matched, and a repr that raised a fatal signal made `validate`
/// raise for a member.
///
/// Each level the branch's own walk would enter is held while it is decided,
/// as [`member`] holds it: at the walk's depth bound the branch records the
/// bound instead, which a report does read, so there it is explained.
fn decided_quietly(schema: &Schema, value: &Value<'_, '_>, ctx: Ctx<'_>) -> Option<bool> {
    if ctx.fatal_seen.get() {
        return Some(false);
    }
    let _level = ctx.descend()?;
    if let Some(kind) = scalar_of(schema) {
        return Some(scalar_admits(kind, value));
    }
    match schema {
        Schema::Literal(index) => Some(check_literal(
            *index,
            value,
            &mut Frame::new(&mut Vec::new(), &mut Vec::new(), fast(ctx)),
        )),
        Schema::Refine { base, .. } => decided_quietly(base, value, ctx).filter(|admits| !admits),
        Schema::Seq { .. } | Schema::Coll { .. } | Schema::KeyedMap { .. } => {
            (!is_of_its_kind(schema, value)).then_some(false)
        }
        _ => None,
    }
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
                code: UNEXPECTED_MATCH.as_str(),
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
    let ok = match value {
        Value::Py(obj) => is_exactly_a(obj, class) || fold(obj.is_instance(class), obj.py(), ctx),
        Value::Json(..) => fold(
            value.to_python().and_then(|obj| obj.is_instance(class)),
            value.py(),
            ctx,
        ),
    };
    if !ok && ctx.mode.explains() {
        frame.out.push(type_mismatch(
            INSTANCE_TYPE,
            &class_label_in(class, ctx),
            value,
            frame.path,
            ctx,
        ));
    }
    ok
}

/// Whether `obj`'s type is `class` itself, which `isinstance` answers yes to
/// without asking the class anything.
///
/// `CPython`'s `PyObject_IsInstance` makes this test first, before it looks up
/// an `__instancecheck__` -- "quick test for an exact match", from 3.10 to the
/// current branch -- so it is the same answer read off the type pointer, and a
/// value of the class a schema names pays neither the call nor the result it
/// folds. A list of `date`, of one enumeration, of one dataclass is that case
/// at every element. `PyPy` implements `isinstance` otherwise, so there the
/// call answers.
#[inline]
fn is_exactly_a(obj: &Bound<'_, PyAny>, class: &Bound<'_, PyAny>) -> bool {
    #[cfg(not(PyPy))]
    {
        std::ptr::eq(obj.get_type_ptr().cast(), class.as_ptr())
    }
    #[cfg(PyPy)]
    {
        let _ = (obj, class);
        false
    }
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
                    code: RECURSION_LOOP.as_str(),
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
                    code: RECURSION_LIMIT.as_str(),
                    path: frame.path.clone(),
                    expected: format!(
                        "a value at most {MAX_RECURSION_DEPTH} recursive levels deep"
                    ),
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
