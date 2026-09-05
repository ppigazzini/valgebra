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

use std::borrow::Cow;
use std::ops::ControlFlow;

use jiter::JsonValue;
use pyo3::exceptions::{PyException, PyMemoryError, PyRecursionError};
use pyo3::prelude::*;
use pyo3::sync::critical_section::with_critical_section;
use pyo3::types::{PyDict, PyFrozenSet, PyList, PySet, PyString, PyTuple};
use rustc_hash::{FxHashMap, FxHashSet};
use valgebra_core::{
    ClassIx, ConstIx, Constraint, DefIx, Field, MapClause, OperandIx, PathSegment, PredIx, Schema,
    SeqKind, SeqShape, Violation,
};

use crate::check::ctx::{Ctx, MAX_WALK_DEPTH, WalkMode};
use crate::check::index::compile_pattern;
use crate::check::violation::{
    key_label, located, mismatch, summarize_value, type_fail, type_mismatch,
};
use crate::errors::{class_label, summarize};
use crate::input::Value;

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
fn is_fatal(err: &PyErr, py: Python<'_>) -> bool {
    !err.is_instance_of::<PyException>(py)
        || err.is_instance_of::<PyMemoryError>(py)
        || err.is_instance_of::<PyRecursionError>(py)
}

/// Record the first fatal signal so the walk unwinds (every later `member` call
/// returns at once) and the entry point re-raises it.
fn record_fatal(err: PyErr, ctx: Ctx<'_>) {
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
pub(crate) fn member(
    schema: &Schema,
    value: &Value<'_, '_>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
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
            out.push(Violation {
                code: "recursion_limit",
                path: path.clone(),
                expected: format!("at most {MAX_WALK_DEPTH} levels of nesting"),
                value_summary: summarize_value(value),
            });
        }
        return false;
    };
    match schema {
        Schema::Anything(_) => true,
        // Bottom admits nothing; an unresolved self-reference is never a member.
        Schema::Nothing => admit(false, schema, value, path, ctx, out),
        Schema::SelfRef(_) => {
            if ctx.mode.explains() {
                out.push(Violation {
                    code: "unresolved_recursion",
                    path: path.clone(),
                    expected: "a resolved recursive value".to_owned(),
                    value_summary: summarize_value(value),
                });
            }
            false
        }
        Schema::NoneType => admit(value.is_none(), schema, value, path, ctx, out),
        Schema::Bool => admit(value.is_bool(), schema, value, path, ctx, out),
        // bool subclasses int, so True/False are ints: Bool is a subset of Int.
        Schema::Int => admit(value.is_int(), schema, value, path, ctx, out),
        Schema::Float => admit(value.is_float(), schema, value, path, ctx, out),
        Schema::Str => admit(value.is_str(), schema, value, path, ctx, out),
        Schema::Bytes => admit(value.is_bytes(), schema, value, path, ctx, out),
        Schema::Literal(index) => check_literal(*index, value, path, ctx, out),
        Schema::Seq { container, shape } => check_seq(*container, shape, value, path, ctx, out),
        Schema::Set(element) => check_set(element, value, path, ctx, out),
        Schema::FrozenSet(element) => check_frozenset(element, value, path, ctx, out),
        Schema::KeyedMap { fields, defaults } => {
            // Membership is the single-pass fast check; on failure the explain
            // pass re-walks in declared order to aggregate ordered violations.
            let ok = keyed_map_matches(fields, defaults, value, ctx);
            if !ok && ctx.mode.explains() {
                let before = out.len();
                keyed_map_explain(fields, defaults, value, path, ctx, out);
                if out.len() == before {
                    // Two passes read the same dict and disagreed, so the dict
                    // did not stay still between them: report that rather than a
                    // failure with nothing behind it.
                    mutated(value, path, ctx, out);
                }
            }
            ok
        }
        Schema::Union(members) => check_union(members, value, path, ctx, out),
        Schema::Intersection(members) => check_intersection(members, value, path, ctx, out),
        Schema::Complement(inner) => check_complement(inner, value, path, ctx, out),
        Schema::Instance(index) => check_instance(*index, value, path, ctx, out),
        Schema::AttrRecord { fields } => check_attr_record(fields, value, path, ctx, out),
        Schema::Refine { base, constraints } => {
            check_refine(base, constraints, value, path, ctx, out)
        }
        Schema::Ref(id) => check_ref(*id, value, path, ctx, out),
    }
}

/// A leaf decision: pass `ok` through, recording a type/value mismatch when it is
/// false in explain mode.
fn admit(
    ok: bool,
    schema: &Schema,
    value: &Value<'_, '_>,
    path: &[PathSegment],
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
    if !ok && ctx.mode.explains() {
        out.push(mismatch(schema, value, path));
    }
    ok
}

fn check_literal(
    index: ConstIx,
    value: &Value<'_, '_>,
    path: &[PathSegment],
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
    let Some(literal) = const_at(ctx, index, value.py()) else {
        return false;
    };
    let ok = fold(
        value
            .to_python()
            .and_then(|obj| literal_matches(&obj, literal)),
        value.py(),
        ctx,
    );
    if !ok && ctx.mode.explains() {
        out.push(Violation {
            code: "literal_error",
            path: path.to_vec(),
            expected: format!("the literal {}", summarize(literal)),
            value_summary: summarize_value(value),
        });
    }
    ok
}

/// Membership for a sequence node: the value is a list or tuple whose elements
/// take the schema's shape — a fixed positional prefix then an optional repeated
/// tail. The elements are walked lazily against the shape the node holds, with
/// no automaton and no collection, identical in cost to a direct positional or
/// homogeneous check. JSON arrays are lists.
fn check_seq(
    container: SeqKind,
    shape: &SeqShape,
    value: &Value<'_, '_>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
    let (kind_word, type_code, len_code) = match container {
        SeqKind::List => ("list", "list_type", "list_length"),
        SeqKind::Tuple => ("tuple", "tuple_type", "tuple_length"),
    };
    let (prefix, tail) = (shape.prefix.as_slice(), shape.tail.as_deref());
    match (container, value) {
        (SeqKind::List, Value::Py(v)) => {
            let Ok(list) = v.cast::<PyList>() else {
                return type_fail(type_code, kind_word, value, path, ctx, out);
            };
            if !SeqArity::of(prefix.len(), tail).admits(list.len()) {
                return seq_length_fail(len_code, kind_word, prefix, tail, value, path, ctx, out);
            }
            let mut ok = true;
            for (i, item) in list.iter().enumerate() {
                ok &= seq_element(prefix, tail, i, &Value::Py(&item), path, ctx, out);
                if !ok && stop(ctx) {
                    return false;
                }
            }
            ok
        }
        (SeqKind::List, Value::Json(py, JsonValue::Array(items))) => {
            if !SeqArity::of(prefix.len(), tail).admits(items.len()) {
                return seq_length_fail(len_code, kind_word, prefix, tail, value, path, ctx, out);
            }
            let mut ok = true;
            for (i, item) in items.iter().enumerate() {
                ok &= seq_element(prefix, tail, i, &Value::Json(*py, item), path, ctx, out);
                if !ok && stop(ctx) {
                    return false;
                }
            }
            ok
        }
        (SeqKind::Tuple, Value::Py(v)) => {
            let Ok(tuple) = v.cast::<PyTuple>() else {
                return type_fail(type_code, kind_word, value, path, ctx, out);
            };
            if !SeqArity::of(prefix.len(), tail).admits(tuple.len()) {
                return seq_length_fail(len_code, kind_word, prefix, tail, value, path, ctx, out);
            }
            let mut ok = true;
            for (i, item) in tuple.iter().enumerate() {
                ok &= seq_element(prefix, tail, i, &Value::Py(&item), path, ctx, out);
                if !ok && stop(ctx) {
                    return false;
                }
            }
            ok
        }
        // A tuple is never a JSON value; a list needs a JSON array.
        _ => type_fail(type_code, kind_word, value, path, ctx, out),
    }
}

/// The element counts a sequence shape admits: exactly the prefix length with no
/// tail, or at least it when a repeated tail follows.
///
/// One argument rather than a length and a flag beside the value's own length.
/// The two lengths were adjacent and the same type, and transposing them turns
/// "this list is too short" into "this schema wants fewer elements" without
/// failing.
#[derive(Clone, Copy)]
enum SeqArity {
    /// A fixed-length shape: the count must equal this.
    Exactly(usize),
    /// A prefix followed by a repeated tail: the count must be at least this.
    AtLeast(usize),
}

impl SeqArity {
    /// The arity of a prefix-and-optional-tail shape.
    fn of(prefix_len: usize, tail: Option<&Schema>) -> Self {
        if tail.is_some() {
            SeqArity::AtLeast(prefix_len)
        } else {
            SeqArity::Exactly(prefix_len)
        }
    }

    /// Whether a value of this element count fits.
    fn admits(self, len: usize) -> bool {
        match self {
            SeqArity::Exactly(n) => len == n,
            SeqArity::AtLeast(n) => len >= n,
        }
    }
}

/// Match one element at position `i`: the prefix schema at `i`, or the repeated
/// tail past the prefix. The index segment is pushed only in explain mode.
fn seq_element(
    prefix: &[Schema],
    tail: Option<&Schema>,
    i: usize,
    item: &Value<'_, '_>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
    let Some(schema) = prefix.get(i).or(tail) else {
        // Unreachable: the caller's length check guarantees `i` lands in the
        // prefix, or a repeated tail covers the overflow. Fold to non-member
        // rather than panic across the FFI boundary if that ever breaks.
        return false;
    };
    if ctx.mode.explains() {
        path.push(PathSegment::Index(i));
    }
    let ok = member(schema, item, path, ctx, out);
    if ctx.mode.explains() {
        path.pop();
    }
    ok
}

/// A sequence-length mismatch: terminal, since the positional match is then
/// meaningless. A tailless shape wants an exact length; a tailed one a minimum.
#[allow(clippy::too_many_arguments)]
fn seq_length_fail(
    len_code: &'static str,
    kind_word: &str,
    prefix: &[Schema],
    tail: Option<&Schema>,
    value: &Value<'_, '_>,
    path: &[PathSegment],
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
    if ctx.mode.explains() {
        let expected = if tail.is_some() {
            format!("{kind_word} of length at least {}", prefix.len())
        } else {
            format!("{kind_word} of length {}", prefix.len())
        };
        out.push(Violation {
            code: len_code,
            path: path.to_vec(),
            expected,
            value_summary: summarize_value(value),
        });
    }
    false
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
fn mutated(
    value: &Value<'_, '_>,
    path: &[PathSegment],
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
    if ctx.mode.explains() {
        out.push(Violation {
            code: MUTATED_CODE,
            path: path.to_vec(),
            expected: MUTATED_EXPECTED.to_owned(),
            value_summary: summarize_value(value),
        });
    }
    false
}

/// Visit a dict's entries, refusing rather than panicking when the dict changes
/// size underneath the scan.
///
/// The iterator `PyO3` hands out panics when the dict's length moves, and the walk
/// runs Python at every entry, so that panic is reachable from an ordinary
/// schema — and a panic is not one of the answers this library gives: it crosses
/// the boundary as a `BaseException` that no caller catches as a validation
/// failure. The scan asks the same question one step earlier, before each step
/// rather than inside it, and stops at the entry count it began with, so the
/// iterator is never advanced into either of the states it panics in. The
/// critical section keeps a second thread out of the dict for the parts of the
/// scan that do not call back into the interpreter.
fn scan_dict<'py>(
    dict: &Bound<'py, PyDict>,
    mut visit: impl FnMut(&Bound<'py, PyAny>, &Bound<'py, PyAny>) -> ControlFlow<()>,
) -> Scan {
    with_critical_section(dict.as_any(), || {
        let entries = dict.len();
        let mut iter = dict.iter();
        let mut seen = 0;
        while seen < entries {
            if dict.len() != entries {
                return Scan::Unreadable;
            }
            let Some((key, value)) = iter.next() else {
                break;
            };
            seen += 1;
            if visit(&key, &value).is_break() {
                return Scan::Stopped;
            }
        }
        if dict.len() == entries {
            Scan::Complete
        } else {
            Scan::Unreadable
        }
    })
}

/// Visit a set's or frozenset's elements, reporting rather than panicking when
/// the container changes underneath the scan.
///
/// A set is walked through its own Python iterator, which *raises* on mutation
/// where `PyO3`'s wrapper unwraps that error into a panic. Taking the iterator
/// directly keeps the raise, which is an outcome the walk already knows how to
/// carry: a fatal signal propagates, and anything else means the container did
/// not answer.
fn scan_set<'py>(
    set: &Bound<'py, PyAny>,
    ctx: Ctx<'_>,
    mut visit: impl FnMut(&Bound<'py, PyAny>) -> ControlFlow<()>,
) -> Scan {
    with_critical_section(set, || {
        let Ok(iter) = set.try_iter() else {
            return Scan::Unreadable;
        };
        for item in iter {
            match item {
                Ok(item) => {
                    if visit(&item).is_break() {
                        return Scan::Stopped;
                    }
                }
                Err(err) => {
                    if is_fatal(&err, set.py()) {
                        record_fatal(err, ctx);
                    }
                    return Scan::Unreadable;
                }
            }
        }
        Scan::Complete
    })
}

/// A set-like container the walk reads: what its type failure reports, and the
/// test that recognises it.
struct Collection {
    code: &'static str,
    word: &'static str,
    is_kind: fn(&Bound<'_, PyAny>) -> bool,
}

const SET: Collection = Collection {
    code: "set_type",
    word: "set",
    is_kind: |value| value.is_instance_of::<PySet>(),
};

const FROZEN_SET: Collection = Collection {
    code: "frozen_set_type",
    word: "frozenset",
    is_kind: |value| value.is_instance_of::<PyFrozenSet>(),
};

/// A set whose every element matches `element`. Set order is not meaningful, so
/// element failures carry no index segment. JSON has no sets.
fn check_set(
    element: &Schema,
    value: &Value<'_, '_>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
    check_elements(&SET, element, value, path, ctx, out)
}

/// A frozenset whose every element matches `element`. JSON has no frozensets.
fn check_frozenset(
    element: &Schema,
    value: &Value<'_, '_>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
    check_elements(&FROZEN_SET, element, value, path, ctx, out)
}

/// Membership for either set-like container: the value is of the container's
/// kind and every element belongs to `element`. One rule for both, because the
/// two differ only in the type they admit and the code they report.
fn check_elements(
    collection: &Collection,
    element: &Schema,
    value: &Value<'_, '_>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
    let Value::Py(container) = value else {
        return type_fail(collection.code, collection.word, value, path, ctx, out);
    };
    if !(collection.is_kind)(container) {
        return type_fail(collection.code, collection.word, value, path, ctx, out);
    }
    if ctx.mode.explains() {
        return explain_elements(element, container, value, path, ctx, out);
    }
    let mut ok = true;
    let scan = scan_set(container, ctx, |item| {
        ok &= member(element, &Value::Py(item), path, ctx, out);
        if !ok && stop(ctx) {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    });
    match scan {
        Scan::Complete => ok,
        Scan::Stopped => false,
        Scan::Unreadable => mutated(value, path, ctx, out),
    }
}

/// Report a set's failing elements in an order that is a property of the value
/// rather than of the run.
///
/// A set has no positions, so an element failure carries no index and the only
/// thing distinguishing two of them is what they say. Iteration order is the
/// interpreter's and moves with the hash seed, so following it means the same
/// schema and the same value report differently between runs — which the error
/// model promises they do not. Every element is walked and the failures are
/// ordered by what they report; fail-fast then keeps the first of *that* order,
/// which costs a full scan of a set that is already failing.
fn explain_elements(
    element: &Schema,
    container: &Bound<'_, PyAny>,
    value: &Value<'_, '_>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
    let mut failures: Vec<(String, Vec<Violation>)> = Vec::new();
    let scan = scan_set(container, ctx, |item| {
        let mut reported = Vec::new();
        if !member(element, &Value::Py(item), path, ctx, &mut reported) {
            let key = reported
                .first()
                .map(|violation| format!("{} {}", violation.value_summary, violation.code))
                .unwrap_or_default();
            failures.push((key, reported));
        }
        ControlFlow::Continue(())
    });
    if matches!(scan, Scan::Unreadable) {
        return mutated(value, path, ctx, out);
    }
    let ok = failures.is_empty();
    failures.sort_by(|left, right| left.0.cmp(&right.0));
    let reported = if ctx.mode.stops_at_first() {
        1
    } else {
        failures.len()
    };
    for (_, group) in failures.into_iter().take(reported) {
        out.extend(group);
    }
    ok
}

/// Membership for a keyed map: named fields, then a default clause for every
/// other key. The walk is inverted — it visits each entry once — and a JSON
/// object's keys are strings, a duplicate keeping its last value as
/// `json.loads` does.
fn keyed_map_matches(
    fields: &[Field],
    defaults: &[MapClause],
    value: &Value<'_, '_>,
    ctx: Ctx<'_>,
) -> bool {
    match value {
        Value::Py(v) => keyed_map_matches_py(fields, defaults, v, ctx),
        Value::Json(py, JsonValue::Object(entries)) => {
            keyed_map_matches_json(fields, defaults, *py, entries, ctx)
        }
        Value::Json(..) => false,
    }
}

/// Whether `(key, val)` is covered by some default clause: the key belongs to a
/// clause's key schema and the value to that clause's value schema. The clauses
/// denote a union of key×value rectangles.
fn covered(defaults: &[MapClause], key: &Value<'_, '_>, val: &Value<'_, '_>, ctx: Ctx<'_>) -> bool {
    let sub = fast(ctx);
    defaults.iter().any(|clause| {
        member(&clause.key, key, &mut Vec::new(), sub, &mut Vec::new())
            && member(&clause.value, val, &mut Vec::new(), sub, &mut Vec::new())
    })
}

/// The keyed-map fast path over a Python dict. A string key naming a declared
/// field is checked against it; any other key (non-string, or undeclared) must
/// be covered by a default clause. Closed records have no clauses, so an
/// undeclared key is rejected; an open record's `anything` clause covers it.
///
/// The declared-field lookup comes from the validator's precomputed
/// [`RecordIndex`] when present, so a wide record skips rebuilding its name map
/// on every call; a record not in the index (an empty one, or a node the
/// build-time traversal did not reach) falls back to building the map here.
fn keyed_map_matches_py(
    fields: &[Field],
    defaults: &[MapClause],
    dict: &Bound<'_, PyAny>,
    ctx: Ctx<'_>,
) -> bool {
    let Ok(dict) = dict.cast::<PyDict>() else {
        return false;
    };
    if let Some(plan) = ctx.records.get(&(fields.as_ptr() as usize)) {
        keyed_map_scan(fields, defaults, dict, ctx, plan.required, |name| {
            plan.by_name.get(name).copied()
        })
    } else {
        let declared: FxHashMap<&str, usize> = fields
            .iter()
            .enumerate()
            .map(|(i, f)| (f.name.as_str(), i))
            .collect();
        let required = fields.iter().filter(|f| f.required).count();
        keyed_map_scan(fields, defaults, dict, ctx, required, |name| {
            declared.get(name).copied()
        })
    }
}

/// Walk a dict once against a record's fields, resolving each string key to a
/// declared-field index through `lookup` (a precomputed plan or a freshly built
/// map). A key that resolves checks its value against that field; any other key
/// must be covered by a default clause. The record matches iff every entry
/// matches and every required field was seen.
fn keyed_map_scan(
    fields: &[Field],
    defaults: &[MapClause],
    dict: &Bound<'_, PyDict>,
    ctx: Ctx<'_>,
    mut required_remaining: usize,
    lookup: impl Fn(&str) -> Option<usize>,
) -> bool {
    let sub = fast(ctx);
    let scan = scan_dict(dict, |key, val| {
        // A non-string key, or a string carrying a lone surrogate (which cannot
        // equal a field name, since names are valid UTF-8 by build-time check),
        // resolves to no field and must instead be covered by a default clause.
        let index = key
            .cast::<PyString>()
            .ok()
            .and_then(|s| s.to_str().ok())
            .and_then(&lookup);
        match index.and_then(|i| fields.get(i)) {
            Some(field) => {
                if !member(
                    &field.schema,
                    &Value::Py(val),
                    &mut Vec::new(),
                    sub,
                    &mut Vec::new(),
                ) {
                    return ControlFlow::Break(());
                }
                if field.required {
                    // Saturating: the counter is the precomputed required-field
                    // count, so it cannot legitimately pass zero, but a malformed
                    // index must not wrap a release build into a false pass.
                    required_remaining = required_remaining.saturating_sub(1);
                }
            }
            None => {
                if !covered(defaults, &Value::Py(key), &Value::Py(val), ctx) {
                    return ControlFlow::Break(());
                }
            }
        }
        ControlFlow::Continue(())
    });
    matches!(scan, Scan::Complete) && required_remaining == 0
}

/// The keyed-map fast path over a JSON object. Keys are strings; a duplicate key
/// keeps its last value (a reverse find), as `json.loads` does. Records are
/// small, so a linear scan beats building a per-object map.
fn keyed_map_matches_json(
    fields: &[Field],
    defaults: &[MapClause],
    py: Python<'_>,
    entries: &[(Cow<'_, str>, JsonValue<'_>)],
    ctx: Ctx<'_>,
) -> bool {
    let sub = fast(ctx);
    for field in fields {
        match entries
            .iter()
            .rev()
            .find(|(key, _)| field.name == key.as_ref())
        {
            Some((_, val)) => {
                if !member(
                    &field.schema,
                    &Value::Json(py, val),
                    &mut Vec::new(),
                    sub,
                    &mut Vec::new(),
                ) {
                    return false;
                }
            }
            None if field.required => return false,
            None => {}
        }
    }
    // Every key that is not a declared field must be covered by a default clause,
    // testing each key's last value (json.loads semantics).
    //
    // Whether a key is a declared field is a question about the *schema*, so it
    // is answered from the record plan built once per validator rather than from
    // a name set rebuilt per object. A schema absent from the plan falls back to
    // scanning the field list, so correctness never depends on the plan being
    // complete.
    let plan = ctx.records.get(&(fields.as_ptr() as usize));
    let declares = |name: &str| match plan {
        Some(plan) => plan.by_name.contains_key(name),
        None => fields.iter().any(|f| f.name == name),
    };
    // A closed record has no clause to cover an undeclared key with, so the first
    // one decides and there is nothing to collapse.
    if defaults.is_empty() {
        return entries.iter().all(|(key, _)| declares(key.as_ref()));
    }
    // Collapse the entries to each non-field key's last value in one pass, so a
    // document with many keys (or many duplicates) is covered linearly rather
    // than by rescanning the tail per key.
    let mut last_value: FxHashMap<&str, &JsonValue<'_>> = FxHashMap::default();
    for (key, val) in entries {
        if declares(key.as_ref()) {
            continue;
        }
        last_value.insert(key.as_ref(), val);
    }
    for (key, val) in last_value {
        let key_value = JsonValue::Str(Cow::Borrowed(key));
        if !covered(
            defaults,
            &Value::Json(py, &key_value),
            &Value::Json(py, val),
            ctx,
        ) {
            return false;
        }
    }
    true
}

/// The explain pass over a keyed map, run only after [`keyed_map_matches`] has
/// reported the value is not a member. It walks in declared order — present
/// fields checked in order, then absent required keys — then reports each
/// undeclared key: an uncovered key with no clauses reads as an unexpected key,
/// and with clauses its key and value are checked against the first clause.
fn keyed_map_explain(
    fields: &[Field],
    defaults: &[MapClause],
    value: &Value<'_, '_>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) {
    let Value::Py(v) = value else {
        // The explain pass only ever sees a Python value; a JSON value here is
        // unreachable, but keep the false-implies-a-violation invariant.
        out.push(type_mismatch("dict_type", "dict", value, path));
        return;
    };
    let Ok(dict) = v.cast::<PyDict>() else {
        out.push(type_mismatch("dict_type", "dict", value, path));
        return;
    };
    let declared: FxHashSet<&str> = fields.iter().map(|field| field.name.as_str()).collect();
    for field in fields {
        match dict.get_item(field.name.as_str()) {
            Ok(Some(item)) => {
                path.push(PathSegment::Key(field.name.clone()));
                member(&field.schema, &Value::Py(&item), path, ctx, out);
                path.pop();
            }
            Ok(None) if field.required => out.push(located(
                path,
                field.name.clone(),
                "missing_key",
                format!("required key {:?}", field.name),
                "missing".to_owned(),
            )),
            Ok(None) => {}
            Err(_) => out.push(type_mismatch("dict_type", "dict", value, path)),
        }
        if ctx.mode.stops_at_first() && !out.is_empty() {
            return;
        }
    }
    let scan = scan_dict(dict, |key, val| {
        if let Some(name) = key.cast::<PyString>().ok().and_then(|s| s.to_str().ok())
            && declared.contains(name)
        {
            return ControlFlow::Continue(());
        }
        if covered(defaults, &Value::Py(key), &Value::Py(val), ctx) {
            return ControlFlow::Continue(());
        }
        if let Some(clause) = defaults.first() {
            // A clause exists but did not cover this key: surface the key and
            // value violations against it (the homogeneous-mapping error).
            path.push(PathSegment::Key(key_label(key)));
            member(&clause.key, &Value::Py(key), path, ctx, out);
            member(&clause.value, &Value::Py(val), path, ctx, out);
            path.pop();
        } else {
            // A closed record: the key is simply not allowed.
            let key_text = key
                .str()
                .map_or_else(|_| String::new(), |text| text.to_string());
            out.push(located(
                path,
                key_text.clone(),
                "extra_forbidden",
                "no unexpected key".to_owned(),
                format!("{key_text:?}"),
            ));
        }
        if ctx.mode.stops_at_first() && !out.is_empty() {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    });
    // The fast pass reported a non-member; if the dict moved underneath this one
    // there is nothing else to report, and the mutation is the finding.
    if matches!(scan, Scan::Unreadable) {
        mutated(value, path, ctx, out);
    }
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
/// itself `union`.
fn push_branch_label(schema: &Schema, ctx: Ctx<'_>, py: Python<'_>, out: &mut BranchLabels) {
    match schema {
        Schema::Union(members) => {
            for member in members {
                push_branch_label(member, ctx, py, out);
            }
        }
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
        Schema::Refine { base, .. } => push_branch_label(base, ctx, py, out),
        other => out.push(other.expected().to_owned()),
    }
}

/// The pooled class's own name, falling back to the node's kind when the pool
/// cannot be read.
fn class_name(index: ClassIx, schema: &Schema, ctx: Ctx<'_>, py: Python<'_>) -> String {
    class_at(ctx, index, py).map_or_else(|| schema.expected().to_owned(), |c| class_label(c))
}

fn check_union(
    members: &[Schema],
    value: &Value<'_, '_>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
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
    // A value is a member iff it matches at least one branch; decide that on the
    // fast path, where a discarded branch pays for no path or violation.
    let sub = fast(ctx);
    if members
        .iter()
        .any(|m| member(m, value, &mut Vec::new(), sub, &mut Vec::new()))
    {
        return true;
    }
    if !ctx.mode.explains() {
        return false;
    }
    // No branch matches. Explain the *closest* branch — the one that descended
    // furthest into the value before failing — rather than dumping every branch.
    // "Furthest" is the greatest path depth past the union's own location. When
    // no branch makes progress (every branch is a flat type mismatch, e.g.
    // `int | str` against a float), fall back to a single union error. The probe
    // aggregates regardless of fail_fast so the deepest progress is visible; this
    // runs only on the error path.
    let base_depth = path.len();
    let probe = Ctx {
        mode: WalkMode::Explain,
        ..ctx
    };
    let mut best: Option<(usize, Vec<Violation>)> = None;
    for branch_schema in members.iter().take(CLOSEST_BRANCH_PROBE_LIMIT) {
        let mut branch = Vec::new();
        member(branch_schema, value, path, probe, &mut branch);
        let progress = branch
            .iter()
            .map(|v| v.path.len())
            .max()
            .unwrap_or(base_depth)
            .saturating_sub(base_depth);
        // Strictly greater keeps the earliest branch on a tie.
        let replace = best
            .as_ref()
            .is_none_or(|(best_progress, _)| progress > *best_progress);
        if replace {
            best = Some((progress, branch));
        }
    }
    match best {
        Some((progress, branch)) if progress > 0 => out.extend(branch),
        _ => {
            let mut labels = BranchLabels::new();
            for member in members {
                push_branch_label(member, ctx, value.py(), &mut labels);
            }
            out.push(Violation {
                code: "union_error",
                path: path.clone(),
                expected: labels.render(),
                value_summary: summarize_value(value),
            });
        }
    }
    false
}

fn check_intersection(
    members: &[Schema],
    value: &Value<'_, '_>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
    // Every member must hold; in explain mode each member's failure is collected,
    // until one rejects the value itself.
    let mut ok = true;
    for member_schema in members {
        let before = out.len();
        ok &= member(member_schema, value, path, ctx, out);
        if !ok && (stop(ctx) || rejected_the_value(out, before, path.len())) {
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
/// This is the rule [`check_refine`] already applies between a base and its
/// constraints, said once for the meet: `Annotated[int, Gt(0)]` does not report
/// a bound on a string.
fn rejected_the_value(out: &[Violation], before: usize, depth: usize) -> bool {
    // The walk only appends to `path` as it descends, so a violation whose path
    // is as long as the current one is at the current one.
    out.get(before..)
        .is_some_and(|since| since.iter().any(|v| v.path.len() == depth))
}

fn check_complement(
    inner: &Schema,
    value: &Value<'_, '_>,
    path: &[PathSegment],
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
    // A value matches the complement iff it does not match the inner schema; the
    // inner explanation is irrelevant, so decide it on the fast path.
    if member(inner, value, &mut Vec::new(), fast(ctx), &mut Vec::new()) {
        if ctx.mode.explains() {
            out.push(Violation {
                code: "unexpected_match",
                path: path.to_vec(),
                expected: format!("not {}", inner.expected()),
                value_summary: summarize_value(value),
            });
        }
        return false;
    }
    true
}

fn check_instance(
    index: ClassIx,
    value: &Value<'_, '_>,
    path: &[PathSegment],
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
    let Some(class) = class_at(ctx, index, value.py()) else {
        return false;
    };
    let ok = fold(
        value.to_python().and_then(|obj| obj.is_instance(class)),
        value.py(),
        ctx,
    );
    if !ok && ctx.mode.explains() {
        out.push(type_mismatch(
            "instance_type",
            &class_label(class),
            value,
            path,
        ));
    }
    ok
}

fn check_attr_record(
    fields: &[Field],
    value: &Value<'_, '_>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
    // Attributes are read off a Python object, so a value that will not
    // materialize into one carries none and belongs to no record.
    let Ok(obj) = value.to_python() else {
        return false;
    };
    // The interned names, in field order. A schema absent from the index (an
    // incomplete build traversal) falls back to the field's own text, so
    // correctness never depends on the plan being complete.
    let interned = ctx.attrs.get(&(fields.as_ptr() as usize));
    let mut ok = true;
    for (position, field) in fields.iter().enumerate() {
        let name = interned.and_then(|plan| plan.names.get(position));
        let attribute = match name {
            Some(interned) => obj.getattr(interned.bind(value.py())),
            None => obj.getattr(field.name.as_str()),
        };
        match attribute {
            Ok(attr) => {
                if ctx.mode.explains() {
                    path.push(PathSegment::Key(field.name.clone()));
                }
                ok &= member(&field.schema, &Value::Py(&attr), path, ctx, out);
                if ctx.mode.explains() {
                    path.pop();
                }
            }
            // A fatal signal during attribute access is the interpreter
            // unwinding, not a missing attribute: record it and stop.
            Err(err) if is_fatal(&err, value.py()) => {
                record_fatal(err, ctx);
                return false;
            }
            // A field the schema does not require is satisfied by its absence.
            Err(_) if !field.required => {}
            Err(_) => {
                if ctx.mode.explains() {
                    out.push(located(
                        path,
                        field.name.clone(),
                        "missing_attribute",
                        format!("attribute {:?}", field.name),
                        "missing".to_owned(),
                    ));
                }
                ok = false;
            }
        }
        if !ok && stop(ctx) {
            return false;
        }
    }
    ok
}

fn check_refine(
    base: &Schema,
    constraints: &[Constraint],
    value: &Value<'_, '_>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
    // Constraints narrow the base set, so they are meaningful only on a base
    // member: if the base fails, report that and do not run the constraints.
    if !member(base, value, path, ctx, out) {
        return false;
    }
    let Ok(obj) = value.to_python() else {
        return false;
    };
    let mut ok = true;
    for constraint in constraints {
        ok &= check_constraint(constraint, &obj, path, ctx, out);
        if !ok && stop(ctx) {
            return false;
        }
    }
    ok
}

/// The length of a value, read the way the rest of the walk reads it.
///
/// A `list` and a `tuple` answer with the items they *hold*, because that is
/// what a sequence schema counts when it walks them. Everything else answers
/// `__len__`, which is what a `str`, `bytes`, `set` and `dict` are read through
/// anyway.
///
/// **One value has one length.** A `list` subclass may override `__len__` and
/// say anything; before this, `MinLen(5)` believed it and the sequence shape
/// beside it counted the storage, so the two constraints described different
/// sets and a value could satisfy each in a different sense. A length that two
/// parts of one schema disagree about is not a property of the value, and a set
/// defined by one is not a set.
fn stored_len(value: &Bound<'_, PyAny>) -> PyResult<usize> {
    if let Ok(list) = value.cast::<PyList>() {
        return Ok(list.len());
    }
    if let Ok(tuple) = value.cast::<PyTuple>() {
        return Ok(tuple.len());
    }
    value.len()
}

/// What a violation would say, carried unrendered until one is recorded.
///
/// Naming a bound takes the bound's `repr`, and a value that belongs produces no
/// violation to name it in. Rendering eagerly therefore makes accepting a value
/// cost the size of the schema's *operand* rather than the size of the value.
enum Expected<'py> {
    /// A comparison against a pool operand, as `symbol operand`.
    Order(&'static str, Bound<'py, PyAny>),
    /// A length bound, as `length symbol n`.
    Length(&'static str, usize),
    /// A divisibility operand from the pool.
    Multiple(Bound<'py, PyAny>),
    /// A pattern the whole string must match.
    Pattern(&'py str),
    /// A message with nothing to render into it.
    Fixed(&'static str),
    /// A predicate that raised, carrying the error it raised. Only ever built on
    /// the failing path, so it renders no sooner than the rest.
    Raised(String),
}

impl Expected<'_> {
    /// Render the message. Called once per recorded violation, never per check.
    fn render(&self) -> String {
        match self {
            Self::Order(symbol, operand) => format!("{symbol} {}", summarize(operand)),
            Self::Length(symbol, n) => format!("length {symbol} {n}"),
            Self::Multiple(operand) => format!("a multiple of {}", summarize(operand)),
            Self::Pattern(pattern) => format!("a string matching {pattern:?}"),
            Self::Fixed(text) => (*text).to_owned(),
            Self::Raised(error) => {
                format!("a predicate that does not raise (raised {error})")
            }
        }
    }
}

/// Check one order bound (`Ge`/`Gt`/`Le`/`Lt`) against `value`: resolve the pool
/// constant and run the rich comparison at the boundary, folding an ordinary
/// error to a non-match. Returns `None` when the pool constant is unavailable,
/// the signal the caller turns into a non-member.
fn order_bound<'py>(
    value: &Bound<'py, PyAny>,
    index: OperandIx,
    ctx: Ctx<'_>,
    py: Python<'py>,
    compare: impl Fn(&Bound<'py, PyAny>, &Bound<'py, PyAny>) -> PyResult<bool>,
    code: &'static str,
    symbol: &'static str,
) -> Option<(bool, &'static str, Expected<'py>)> {
    let bound = operand_at(ctx, index, py)?;
    let ok = fold(compare(value, bound), py, ctx);
    // Cloned only here, where the violation payload owns what it will summarize.
    // The lookup itself borrows, which is what keeps a literal comparison and an
    // isinstance check off the reference-count path.
    Some((ok, code, Expected::Order(symbol, bound.clone())))
}

/// Whether `value` (already a base member, materialized once) satisfies one
/// constraint, recording a violation on failure in explain mode.
fn check_constraint<'py>(
    constraint: &'py Constraint,
    value: &Bound<'py, PyAny>,
    path: &[PathSegment],
    ctx: Ctx<'py>,
    out: &mut Vec<Violation>,
) -> bool {
    let py = value.py();
    let (ok, code, expected): (bool, &'static str, Expected<'py>) = match constraint {
        Constraint::Ge(i) => {
            let Some(t) = order_bound(
                value,
                *i,
                ctx,
                py,
                |v, b| v.ge(b),
                "greater_than_equal",
                ">=",
            ) else {
                return false;
            };
            t
        }
        Constraint::Gt(i) => {
            let Some(t) = order_bound(value, *i, ctx, py, |v, b| v.gt(b), "greater_than", ">")
            else {
                return false;
            };
            t
        }
        Constraint::Le(i) => {
            let Some(t) = order_bound(value, *i, ctx, py, |v, b| v.le(b), "less_than_equal", "<=")
            else {
                return false;
            };
            t
        }
        Constraint::Lt(i) => {
            let Some(t) = order_bound(value, *i, ctx, py, |v, b| v.lt(b), "less_than", "<") else {
                return false;
            };
            t
        }
        Constraint::MinLen(n) => (
            fold(stored_len(value).map(|len| len >= *n), py, ctx),
            "too_short",
            Expected::Length(">=", *n),
        ),
        Constraint::MaxLen(n) => (
            fold(stored_len(value).map(|len| len <= *n), py, ctx),
            "too_long",
            Expected::Length("<=", *n),
        ),
        Constraint::MultipleOf(i) => {
            let Some(operand) = operand_at(ctx, *i, py) else {
                return false;
            };
            let ok = fold(is_multiple_of(value, operand), py, ctx);
            (ok, "multiple_of", Expected::Multiple(operand.clone()))
        }
        Constraint::Predicate(i) => {
            // Slow path: the user's Python callable runs at the boundary. A
            // raising predicate is surfaced as a distinct `predicate_error`
            // rather than masked as an ordinary failed match.
            let Some(predicate) = predicate_at(ctx, *i, py) else {
                return false;
            };
            match predicate_passes(value, predicate) {
                Ok(passed) => (
                    passed,
                    "predicate_failed",
                    Expected::Fixed("a passing predicate"),
                ),
                // A fatal signal raised inside the predicate is the interpreter
                // unwinding, not a predicate that merely errored: propagate it.
                Err(err) if is_fatal(&err, py) => {
                    record_fatal(err, ctx);
                    return false;
                }
                Err(err) => (false, "predicate_error", Expected::Raised(err.to_string())),
            }
        }
        Constraint::Regex(pattern) => {
            // Native fast path: the precompiled, anchored pattern matches the
            // borrowed string UTF-8 in Rust. A non-string never matches (the base
            // of a pattern refinement is a string, so this is reached only after
            // a string base check, but stays defensive). A pattern absent from
            // the per-validator cache (an incomplete build traversal) is compiled
            // on the spot rather than silently passing.
            let matched = value
                .cast::<PyString>()
                .ok()
                .and_then(|s| s.to_str().ok())
                .is_some_and(|text| match ctx.regexes.get(&(pattern.as_ptr() as usize)) {
                    Some(compiled) => compiled.is_match(text),
                    None => compile_pattern(pattern).is_ok_and(|re| re.is_match(text)),
                });
            (
                matched,
                "string_pattern_mismatch",
                Expected::Pattern(pattern),
            )
        }
    };
    // The message is rendered here and nowhere else: a check that passes never
    // reads the operand it would have named.
    if !ok && ctx.mode.explains() {
        out.push(Violation {
            code,
            path: path.to_vec(),
            expected: expected.render(),
            value_summary: summarize(value),
        });
    }
    ok
}

/// The most levels of recursive descent allowed before a value is rejected. A
/// finite value never reaches this; the bound exists so a pathologically deep
/// value fails with `recursion_limit` instead of overflowing the native stack.
const MAX_RECURSION_DEPTH: usize = 128;

fn check_ref(
    id: DefIx,
    value: &Value<'_, '_>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
    let key = (value.id(), id.get());
    let depth = {
        let mut guard = ctx.guard.borrow_mut();
        if !guard.insert(key) {
            if ctx.mode.explains() {
                out.push(Violation {
                    code: "recursion_loop",
                    path: path.clone(),
                    expected: "a finite (non-cyclic) value".to_owned(),
                    value_summary: summarize_value(value),
                });
            }
            return false;
        }
        guard.len()
    };
    if depth > MAX_RECURSION_DEPTH {
        ctx.guard.borrow_mut().remove(&key);
        if ctx.mode.explains() {
            out.push(Violation {
                code: "recursion_limit",
                path: path.clone(),
                expected: format!("at most {MAX_RECURSION_DEPTH} levels of recursion"),
                value_summary: summarize_value(value),
            });
        }
        return false;
    }
    let Some(def) = ctx.defs.get(id.get()) else {
        // A reference past the definitions table is an internal invariant break,
        // not reachable from user input; release builds degrade to a non-member
        // rather than panicking across the language boundary.
        debug_assert!(false, "definition index {} out of range", id.get());
        ctx.guard.borrow_mut().remove(&key);
        return false;
    };
    let result = member(def, value, path, ctx, out);
    ctx.guard.borrow_mut().remove(&key);
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

/// Whether `value` is the typed singleton denoted by `literal`: same type and
/// equal. The same-type guard rules out Python's cross-type equality
/// (`1 == True == 1.0`), so `Literal[1]` denotes `{1}`, not `{1, True, 1.0}`.
/// Returns the comparison result so a raising `__eq__` is folded by the caller.
/// The three probe helpers below take the **value first** and the pooled object
/// second. All three arguments are `&Bound<'_, PyAny>`, so every transposition
/// typechecks, and a reader who has just read two of them carries a prior into
/// the third -- which is the condition under which a transposition gets written.
pub(crate) fn literal_matches(
    value: &Bound<'_, PyAny>,
    literal: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    Ok(value.get_type().is(literal.get_type()) && value.eq(literal)?)
}

/// Whether `value % operand == 0`. The remainder is zero iff it is falsy. Returns
/// the result so a raising `%` is folded by the caller (a non-numeric value whose
/// modulo is not defined is then a non-multiple).
///
/// The operator, not `__mod__` by name: the dunder is only half of what `%`
/// means. A type that does not know the operand answers `NotImplemented` and the
/// operand's `__rmod__` is asked next, which is how a `Fraction` or a `Decimal`
/// divides an `int` — and `NotImplemented` is truthy, so reading the dunder's
/// result directly reports every such pair a non-multiple.
fn is_multiple_of(value: &Bound<'_, PyAny>, operand: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(!value.rem(operand)?.is_truthy()?)
}

/// Run a user predicate and report whether it returned a truthy result.
fn predicate_passes(value: &Bound<'_, PyAny>, predicate: &Bound<'_, PyAny>) -> PyResult<bool> {
    predicate.call1((value,))?.is_truthy()
}

// Needs a live interpreter; compiled and run only under the `interpreter-tests`
// feature, which links an embedded Python. This is the walk's own harness: it
// drives real Python values through `member` so the membership decision — where
// soundness is decided, and the one surface the Python suite covers from outside
// but no `cargo test` reaches — carries evidence a mutation sweep can observe.
#[cfg(all(test, feature = "interpreter-tests"))]
mod interpreter {
    use super::*;
    use crate::check::index::ValidatorIndex;
    use crate::check::{WalkMode, WalkState, build_index};
    use pyo3::types::{PyBool, PyBytes, PyDict, PyFloat, PyInt, PyModule};
    use valgebra_core::{Field, MapClause, Openness};

    /// Decide membership of a Python value against a schema, through the real
    /// walk, in the mode a validator's `is_valid` uses.
    fn holds(
        py: Python<'_>,
        schema: &Schema,
        value: &Bound<'_, PyAny>,
        pool: &[Py<PyAny>],
        defs: &[Schema],
    ) -> bool {
        let index = build_index(py, schema, defs, pool);
        let state = WalkState::new();
        let ctx = Ctx {
            pool,
            defs,
            records: &index.records,
            attrs: &index.attrs,
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

    /// The same decision in explain mode, returning the violations it aggregated
    /// alongside the verdict. The two modes must agree on the verdict — the "one
    /// walk" invariant — so every case below is driven through both.
    fn explain(
        py: Python<'_>,
        schema: &Schema,
        value: &Bound<'_, PyAny>,
        pool: &[Py<PyAny>],
        defs: &[Schema],
    ) -> (bool, Vec<Violation>) {
        let index = build_index(py, schema, defs, pool);
        let state = WalkState::new();
        let ctx = Ctx {
            pool,
            defs,
            records: &index.records,
            attrs: &index.attrs,
            unions: &index.unions,
            regexes: &index.regexes,
            guard: &state.guard,
            depth: &state.depth,
            fatal: &state.fatal,
            fatal_seen: &state.fatal_seen,
            mode: WalkMode::Explain,
        };
        let mut out = Vec::new();
        let ok = member(schema, &Value::Py(value), &mut Vec::new(), ctx, &mut out);
        (ok, out)
    }

    /// Drive one case through both modes and assert they agree, then return the
    /// verdict. A case that only ran fast would leave the explain arms — half of
    /// every composite in this file — unobserved.
    fn decide(
        py: Python<'_>,
        schema: &Schema,
        value: &Bound<'_, PyAny>,
        pool: &[Py<PyAny>],
        defs: &[Schema],
    ) -> bool {
        let fast = holds(py, schema, value, pool, defs);
        let (explained, violations) = explain(py, schema, value, pool, defs);
        assert_eq!(fast, explained, "fast and explain modes disagree");
        assert_eq!(
            violations.is_empty(),
            fast,
            "a rejected value must report at least one violation, an accepted one none"
        );
        fast
    }

    /// A dict scan stops at the entry count it began with, and reports rather
    /// than reads a dict whose size moved.
    ///
    /// The count is what keeps the iterator away from the state `PyO3` panics in,
    /// and a panic is not one of the answers this library gives: it crosses the
    /// FFI boundary as a `BaseException` no caller catches as a validation
    /// failure. Driven against the scan rather than through a schema, because
    /// the schema path needs a value that mutates itself mid-walk and the
    /// question here is what the scan does with the count.
    #[test]
    fn a_dict_scan_visits_each_entry_once_and_stops_where_it_is_told() {
        Python::attach(|py| {
            let dict = PyDict::new(py);
            for i in 0..5 {
                dict.set_item(i, i).expect("set_item");
            }

            // Every entry, exactly once: a count that advanced by more than one
            // per entry would visit fewer, and one that compared loosely would
            // step past the end.
            let mut seen = 0;
            let scan = scan_dict(&dict, |_, _| {
                seen += 1;
                ControlFlow::Continue(())
            });
            assert!(matches!(scan, Scan::Complete));
            assert_eq!(seen, 5, "each of the five entries is visited once");

            // A visitor that breaks stops the scan, and the answer says so: a
            // stop is not a complete reading, and the caller reports the miss
            // that caused it rather than the container.
            let mut before_break = 0;
            let scan = scan_dict(&dict, |_, _| {
                before_break += 1;
                if before_break == 2 {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            });
            assert!(matches!(scan, Scan::Stopped));
            assert_eq!(before_break, 2);

            // A dict that grows under the scan has no reading to answer from.
            let moving = PyDict::new(py);
            for i in 0..4 {
                moving.set_item(i, i).expect("set_item");
            }
            let scan = scan_dict(&moving, |key, _| {
                if key.extract::<i64>().unwrap_or(-1) == 0 {
                    moving.set_item("added", 1).expect("set_item");
                }
                ControlFlow::Continue(())
            });
            assert!(
                matches!(scan, Scan::Unreadable),
                "a dict whose size moved is not readable"
            );

            // And one that shrinks, which is the same fact reached at the other
            // end: the scan began expecting entries that are no longer there.
            let shrinking = PyDict::new(py);
            for i in 0..4 {
                shrinking.set_item(i, i).expect("set_item");
            }
            let scan = scan_dict(&shrinking, |key, _| {
                if key.extract::<i64>().unwrap_or(-1) == 0 {
                    shrinking.del_item(3).expect("del_item");
                }
                ControlFlow::Continue(())
            });
            assert!(matches!(scan, Scan::Unreadable));
        });
    }

    /// A value that changed under the walk is a non-member, and in explain mode
    /// it says which failure it was.
    ///
    /// The code is valgebra-coined because it reports a failure of the *check*:
    /// nothing about the value's contents was decided. A verdict of `true` here
    /// would admit a value no reading of it supports.
    #[test]
    fn a_changed_container_is_a_non_member_that_names_itself() {
        Python::attach(|py| {
            let value = PyDict::new(py);
            let state = WalkState::new();
            let index = build_index(py, &Schema::ANYTHING, &[], &[]);
            let ctx = |mode| Ctx {
                pool: &[],
                defs: &[],
                records: &index.records,
                attrs: &index.attrs,
                unions: &index.unions,
                regexes: &index.regexes,
                guard: &state.guard,
                depth: &state.depth,
                fatal: &state.fatal,
                fatal_seen: &state.fatal_seen,
                mode,
            };

            let mut out = Vec::new();
            let held = mutated(&Value::Py(&value), &[], ctx(WalkMode::Explain), &mut out);
            assert!(!held, "a value that changed under the walk is a non-member");
            assert_eq!(out.len(), 1);
            assert_eq!(out[0].code, MUTATED_CODE);
            assert_eq!(out[0].expected, MUTATED_EXPECTED);

            // Fast mode reports the same verdict and writes nothing: the
            // violations it would build go to a buffer nothing reads.
            let mut fast_out = Vec::new();
            let held = mutated(&Value::Py(&value), &[], ctx(WalkMode::Fast), &mut fast_out);
            assert!(!held);
            assert!(fast_out.is_empty());
        });
    }

    /// A union's `expected` names each branch the way that branch names itself,
    /// and a nested union contributes its members rather than itself.
    ///
    /// `Literal[...]` builds a union of its constants, so without the nesting
    /// rule a single-constant literal would name itself `union` and a table of
    /// permitted strings would read `one of: literal, literal`.
    #[test]
    fn a_union_names_its_branches_by_their_constants() {
        Python::attach(|py| {
            let pool: Vec<Py<PyAny>> = ["torch", "jax"]
                .iter()
                .map(|name| PyString::new(py, name).into_any().unbind())
                .collect();
            let table = Schema::Union(vec![
                Schema::Literal(ConstIx::new(0)),
                Schema::Literal(ConstIx::new(1)),
            ]);
            // Nested, which is the shape `Literal[...]` beside another branch
            // builds: the inner union's members are the branches, not the union.
            let schema = Schema::Union(vec![table, Schema::Int]);
            let index = build_index(py, &schema, &[], &pool);
            let state = WalkState::new();
            let ctx = Ctx {
                pool: &pool,
                defs: &[],
                records: &index.records,
                attrs: &index.attrs,
                unions: &index.unions,
                regexes: &index.regexes,
                guard: &state.guard,
                depth: &state.depth,
                fatal: &state.fatal,
                fatal_seen: &state.fatal_seen,
                mode: WalkMode::Explain,
            };
            let mut labels = BranchLabels::new();
            push_branch_label(&schema, ctx, py, &mut labels);
            assert_eq!(
                labels.render(),
                "one of: the literal 'torch', the literal 'jax', int",
                "each branch names itself, and the nested union names its members"
            );
        });
    }

    /// A meet collects every member's failure until one rejects the value
    /// *itself*, and then stops: what a later member would say describes a value
    /// already known to be the wrong kind of thing. A member that fails *inside*
    /// the value leaves the others meaningful, and they are still collected.
    ///
    /// This is what keeps a class with declared attributes -- the meet of an
    /// `isinstance` atom and an attribute record -- reporting one
    /// `instance_type` for a foreign object rather than that violation plus the
    /// attributes the object never had to carry.
    #[test]
    fn a_meet_stops_at_the_member_that_rejects_the_value() {
        Python::attach(|py| {
            let module = classes(py);
            let point = module.getattr("Point").expect("Point");
            let pool: Vec<Py<PyAny>> = vec![point.clone().unbind()];
            let object = Schema::meet([
                Schema::Instance(ClassIx::new(0)),
                Schema::AttrRecord {
                    fields: vec![field("x", Schema::Int, true)],
                },
            ]);

            // Not a Point: the class atom rejects the value at the meet's own
            // path, so the record is not asked about attributes it does not have.
            let foreign = PyInt::new(py, 1i64).into_any();
            let (ok, violations) = explain(py, &object, &foreign, &pool, &[]);
            assert!(!ok);
            assert_eq!(violations.len(), 1, "{violations:?}");
            assert_eq!(violations[0].code, "instance_type");
            assert!(violations[0].path.is_empty());

            // A Point whose attribute is wrong fails *inside* the value: the
            // class atom held, and the record reports the attribute.
            let bad = point
                .call1((PyString::new(py, "x"), PyInt::new(py, 2i64)))
                .expect("Point(str, int)");
            let (ok, violations) = explain(py, &object, &bad, &pool, &[]);
            assert!(!ok);
            assert_eq!(violations.len(), 1, "{violations:?}");
            assert_eq!(violations[0].location(), "x");

            // Two members that each fail inside the value both report: neither
            // rejected the value itself, so neither silences the other.
            let deep = |element| Schema::list(SeqShape::homogeneous(element));
            let both = Schema::Intersection(vec![deep(Schema::Int), deep(Schema::Bool)]);
            let list = PyList::new(py, [PyString::new(py, "a")]).expect("list");
            let (ok, violations) = explain(py, &both, &list.into_any(), &[], &[]);
            assert!(!ok);
            assert_eq!(violations.len(), 2, "{violations:?}");

            // And a member that rejects the value itself stops the rest even
            // when no class is involved.
            let scalars = Schema::Intersection(vec![Schema::Int, Schema::Str]);
            let number = PyFloat::new(py, 1.5).into_any();
            let (ok, violations) = explain(py, &scalars, &number, &[], &[]);
            assert!(!ok);
            assert_eq!(violations.len(), 1, "{violations:?}");
            assert_eq!(violations[0].code, "int_type");
        });
    }

    /// A class branch names its class, and a class with declared attributes --
    /// the meet of an atom and a record -- names that same class rather than the
    /// algebra's spelling of it. A meet that is not an object has no class to
    /// name and falls back to its kind.
    #[test]
    fn a_union_names_a_class_branch_by_its_class() {
        Python::attach(|py| {
            let module = classes(py);
            let point = module.getattr("Point").expect("Point");
            let other = module.getattr("Other").expect("Other");
            let pool: Vec<Py<PyAny>> = vec![point.unbind(), other.unbind()];
            let object = Schema::meet([
                Schema::Instance(ClassIx::new(0)),
                Schema::AttrRecord {
                    fields: vec![Field {
                        name: "x".to_owned(),
                        schema: Schema::Int,
                        required: true,
                    }],
                },
            ]);
            let schema = Schema::Union(vec![
                object,
                Schema::Instance(ClassIx::new(1)),
                Schema::meet([Schema::Int, Schema::Str]),
            ]);
            let index = build_index(py, &schema, &[], &pool);
            let state = WalkState::new();
            let ctx = Ctx {
                pool: &pool,
                defs: &[],
                records: &index.records,
                attrs: &index.attrs,
                unions: &index.unions,
                regexes: &index.regexes,
                guard: &state.guard,
                depth: &state.depth,
                fatal: &state.fatal,
                fatal_seen: &state.fatal_seen,
                mode: WalkMode::Explain,
            };
            let mut labels = BranchLabels::new();
            push_branch_label(&schema, ctx, py, &mut labels);
            assert_eq!(
                labels.render(),
                "one of: Point, Other, intersection",
                "an object meet names its class; any other meet names its kind"
            );
        });
    }

    /// `(schema, value, expected)` over an empty pool and no definitions.
    fn case(py: Python<'_>, schema: &Schema, value: &Bound<'_, PyAny>, expected: bool) {
        assert_eq!(
            decide(py, schema, value, &[], &[]),
            expected,
            "schema {schema:?} against {value}"
        );
    }

    fn list_of(py: Python<'_>, items: Vec<i64>) -> Bound<'_, PyAny> {
        PyList::new(py, items)
            .expect("a list of i64 builds")
            .into_any()
    }

    #[test]
    fn the_scalar_atoms_admit_their_own_kind_and_no_other() {
        Python::attach(|py| {
            let none = py.None().into_bound(py);
            let boolean = PyBool::new(py, true).to_owned().into_any();
            let integer = PyInt::new(py, 7i64).into_any();
            let float = PyFloat::new(py, 1.5).into_any();
            let text = PyString::new(py, "x").into_any();
            let raw = PyBytes::new(py, b"x").into_any();
            let values = [&none, &boolean, &integer, &float, &text, &raw];

            // Each atom admits exactly its own column, with one exception the
            // typing spec forces: `bool` is a subclass of `int`, so a boolean is
            // an integer and `Int` admits it.
            let rows: [(Schema, [bool; 6]); 6] = [
                (Schema::NoneType, [true, false, false, false, false, false]),
                (Schema::Bool, [false, true, false, false, false, false]),
                (Schema::Int, [false, true, true, false, false, false]),
                (Schema::Float, [false, false, false, true, false, false]),
                (Schema::Str, [false, false, false, false, true, false]),
                (Schema::Bytes, [false, false, false, false, false, true]),
            ];
            for (schema, expected) in &rows {
                for (value, want) in values.iter().zip(expected) {
                    case(py, schema, value, *want);
                }
            }

            // The lattice bounds, over the same column set. The top is checked
            // in both spellings: one node admits every value whichever way the
            // user wrote it.
            for value in values {
                case(py, &Schema::ANYTHING, value, true);
                case(py, &Schema::ANY, value, true);
                case(py, &Schema::Nothing, value, false);
                // A self-reference never survives compilation, and is never a
                // member if one is reached anyway.
                case(py, &Schema::SelfRef(0), value, false);
            }
        });
    }

    #[test]
    fn a_literal_admits_its_own_value_at_its_own_type() {
        Python::attach(|py| {
            let one = PyInt::new(py, 1i64).into_any();
            let pool = vec![one.clone().unbind()];
            let schema = Schema::Literal(ConstIx::new(0));

            assert!(decide(
                py,
                &schema,
                &PyInt::new(py, 1i64).into_any(),
                &pool,
                &[]
            ));
            assert!(!decide(
                py,
                &schema,
                &PyInt::new(py, 2i64).into_any(),
                &pool,
                &[]
            ));
            // Python's `==` conflates across types (`1 == True == 1.0`), so the
            // same-type test is what makes this a singleton rather than a class.
            let truth = PyBool::new(py, true).to_owned().into_any();
            assert!(!decide(py, &schema, &truth, &pool, &[]));
            let float_one = PyFloat::new(py, 1.0).into_any();
            assert!(!decide(py, &schema, &float_one, &pool, &[]));
        });
    }

    #[test]
    fn a_sequence_matches_its_regex_and_its_container_kind() {
        Python::attach(|py| {
            let homogeneous = Schema::list(SeqShape::homogeneous(Schema::Int));
            case(py, &homogeneous, &list_of(py, vec![]), true);
            case(py, &homogeneous, &list_of(py, vec![1, 2, 3]), true);
            let mixed = PyList::new(py, [1i64])
                .expect("a one-element list builds")
                .into_any();
            mixed
                .cast::<PyList>()
                .expect("a list")
                .append(PyString::new(py, "x"))
                .expect("append");
            case(py, &homogeneous, &mixed, false);

            // The container kind is part of the denotation: a tuple is not a list.
            let tuple = PyTuple::new(py, [1i64, 2])
                .expect("a tuple builds")
                .into_any();
            case(py, &homogeneous, &tuple, false);
            case(
                py,
                &Schema::tuple(SeqShape::homogeneous(Schema::Int)),
                &tuple,
                true,
            );

            // Fixed arity: exactly the prefix length, no more and no fewer.
            let fixed = Schema::list(SeqShape::fixed([Schema::Int, Schema::Str]));
            let ok = PyList::new(py, [1i64]).expect("builds").into_any();
            ok.cast::<PyList>()
                .expect("a list")
                .append(PyString::new(py, "x"))
                .expect("append");
            case(py, &fixed, &ok, true);
            case(py, &fixed, &list_of(py, vec![1]), false);
            case(py, &fixed, &list_of(py, vec![1, 2, 3]), false);

            // Prefix plus tail: at least the prefix length, and the tail repeats.
            let prefixed = Schema::list(SeqShape::prefix_tail([Schema::Int], Schema::Int));
            case(py, &prefixed, &list_of(py, vec![]), false);
            case(py, &prefixed, &list_of(py, vec![1]), true);
            case(py, &prefixed, &list_of(py, vec![1, 2, 3]), true);
        });
    }

    #[test]
    fn a_set_and_a_frozenset_are_distinct_containers() {
        Python::attach(|py| {
            let set_of_int = Schema::Set(Box::new(Schema::Int));
            let frozen_of_int = Schema::FrozenSet(Box::new(Schema::Int));
            let set = PySet::new(py, [1i64, 2]).expect("a set builds").into_any();
            let frozen = PyFrozenSet::new(py, [1i64, 2])
                .expect("a frozenset builds")
                .into_any();

            case(py, &set_of_int, &set, true);
            case(py, &set_of_int, &frozen, false);
            case(py, &frozen_of_int, &frozen, true);
            case(py, &frozen_of_int, &set, false);

            let mixed = PySet::new(py, [1i64]).expect("a set builds").into_any();
            mixed
                .cast::<PySet>()
                .expect("a set")
                .add(PyString::new(py, "x"))
                .expect("add");
            case(py, &set_of_int, &mixed, false);
        });
    }

    #[test]
    fn a_keyed_map_separates_fields_from_the_catch_all() {
        Python::attach(|py| {
            let field = |name: &str, schema, required| Field {
                name: name.to_owned(),
                schema,
                required,
            };
            let closed = Schema::record(
                vec![
                    field("x", Schema::Int, true),
                    field("y", Schema::Str, false),
                ],
                Openness::Closed,
            );
            let open = Schema::record(vec![field("x", Schema::Int, true)], Openness::Open);
            let mapping = Schema::mapping(MapClause {
                key: Schema::Str,
                value: Schema::Int,
            });

            let build = |pairs: &[(&str, Bound<'_, PyAny>)]| {
                let dict = PyDict::new(py);
                for (key, value) in pairs {
                    dict.set_item(key, value).expect("set_item");
                }
                dict.into_any()
            };
            let int = |n: i64| PyInt::new(py, n).into_any();
            let text = |s: &str| PyString::new(py, s).into_any();

            // The required field must be present and match; the optional one need
            // not be present, but must match when it is.
            case(py, &closed, &build(&[("x", int(1))]), true);
            case(
                py,
                &closed,
                &build(&[("x", int(1)), ("y", text("a"))]),
                true,
            );
            case(py, &closed, &build(&[("x", int(1)), ("y", int(2))]), false);
            case(py, &closed, &build(&[("y", text("a"))]), false);
            case(py, &closed, &build(&[("x", text("a"))]), false);
            // A closed record forbids an undeclared key; an open one admits it.
            case(py, &closed, &build(&[("x", int(1)), ("z", int(2))]), false);
            case(py, &open, &build(&[("x", int(1)), ("z", int(2))]), true);
            // A pure mapping judges every key and value by the clause.
            case(py, &mapping, &build(&[("k", int(1))]), true);
            case(py, &mapping, &build(&[("k", text("a"))]), false);
            case(py, &mapping, &build(&[]), true);
            // Not a dict at all.
            case(py, &closed, &list_of(py, vec![1]), false);
        });
    }

    /// A violation says what the value was measured against, for every kind of
    /// constraint. The message is built only on the failing path, so nothing
    /// else pins its text: a `render` returning a constant would satisfy every
    /// other test in this file.
    #[test]
    fn a_violation_names_the_constraint_the_value_failed() {
        Python::attach(|py| {
            let pool = vec![PyInt::new(py, 10).into_any().unbind()];
            let ten = OperandIx::new(0);
            let refine = |base: Schema, constraint: Constraint| Schema::Refine {
                base: Box::new(base),
                constraints: vec![constraint],
            };
            let int = |n: i64| PyInt::new(py, n).into_any();
            let text = |s: &str| PyString::new(py, s).into_any();

            // Each row is a constraint, a value that fails it, and the whole of
            // the message that failure must carry.
            for (schema, value, want) in [
                (refine(Schema::Int, Constraint::Ge(ten)), int(1), ">= 10"),
                (refine(Schema::Int, Constraint::Gt(ten)), int(1), "> 10"),
                (refine(Schema::Int, Constraint::Le(ten)), int(11), "<= 10"),
                (refine(Schema::Int, Constraint::Lt(ten)), int(11), "< 10"),
                (
                    refine(Schema::Int, Constraint::MultipleOf(ten)),
                    int(3),
                    "a multiple of 10",
                ),
                (
                    refine(Schema::Str, Constraint::MinLen(2)),
                    text("a"),
                    "length >= 2",
                ),
                (
                    refine(Schema::Str, Constraint::MaxLen(1)),
                    text("abc"),
                    "length <= 1",
                ),
                (
                    refine(Schema::Str, Constraint::Regex("[0-9]+".to_owned())),
                    text("x"),
                    "a string matching \"[0-9]+\"",
                ),
            ] {
                let (ok, violations) = explain(py, &schema, &value, &pool, &[]);
                assert!(!ok, "{want}: the value must fail for a message to exist");
                let [violation] = violations.as_slice() else {
                    panic!("{want}: expected exactly one violation, got {violations:?}")
                };
                assert_eq!(violation.expected, want);
            }
        });
    }

    #[test]
    fn the_boolean_combinators_compose_the_member_sets() {
        Python::attach(|py| {
            let int = PyInt::new(py, 1i64).into_any();
            let text = PyString::new(py, "x").into_any();
            let float = PyFloat::new(py, 1.5).into_any();

            let union = Schema::Union(vec![Schema::Int, Schema::Str]);
            case(py, &union, &int, true);
            case(py, &union, &text, true);
            case(py, &union, &float, false);

            let intersection = Schema::Intersection(vec![
                Schema::Int,
                Schema::Complement(Box::new(Schema::Bool)),
            ]);
            case(py, &intersection, &int, true);
            let truth = PyBool::new(py, true).to_owned().into_any();
            case(py, &intersection, &truth, false);

            let complement = Schema::Complement(Box::new(Schema::Int));
            case(py, &complement, &int, false);
            case(py, &complement, &text, true);
            // Double negation returns the original set.
            let doubled = Schema::Complement(Box::new(complement));
            case(py, &doubled, &int, true);
            case(py, &doubled, &text, false);
        });
    }

    #[test]
    fn a_refinement_narrows_its_base_by_every_constraint() {
        Python::attach(|py| {
            let five = PyInt::new(py, 5i64).into_any();
            let pool = vec![five.clone().unbind()];
            let refine = |constraints: Vec<Constraint>| Schema::Refine {
                base: Box::new(Schema::Int),
                constraints,
            };
            let int = |n: i64| PyInt::new(py, n).into_any();

            // Each comparison arm, at and around its bound.
            for (constraint, at, above, below) in [
                (Constraint::Ge(OperandIx::new(0)), true, true, false),
                (Constraint::Gt(OperandIx::new(0)), false, true, false),
                (Constraint::Le(OperandIx::new(0)), true, false, true),
                (Constraint::Lt(OperandIx::new(0)), false, false, true),
            ] {
                let schema = refine(vec![constraint]);
                assert_eq!(decide(py, &schema, &int(5), &pool, &[]), at);
                assert_eq!(decide(py, &schema, &int(6), &pool, &[]), above);
                assert_eq!(decide(py, &schema, &int(4), &pool, &[]), below);
            }

            // The base is checked first: a bound on a non-int rejects rather than
            // raising through the comparison.
            let ge = refine(vec![Constraint::Ge(OperandIx::new(0))]);
            let text = PyString::new(py, "x").into_any();
            assert!(!decide(py, &ge, &text, &pool, &[]));

            // A multiple-of divides; length bounds measure `len`.
            let multiple = refine(vec![Constraint::MultipleOf(OperandIx::new(0))]);
            assert!(decide(py, &multiple, &int(10), &pool, &[]));
            assert!(!decide(py, &multiple, &int(11), &pool, &[]));

            let sized = Schema::Refine {
                base: Box::new(Schema::Str),
                constraints: vec![Constraint::MinLen(2), Constraint::MaxLen(3)],
            };
            for (text, want) in [("a", false), ("ab", true), ("abc", true), ("abcd", false)] {
                let value = PyString::new(py, text).into_any();
                assert_eq!(decide(py, &sized, &value, &pool, &[]), want, "{text:?}");
            }

            // A pattern is anchored: `re.fullmatch` semantics, not a search.
            let pattern = Schema::Refine {
                base: Box::new(Schema::Str),
                constraints: vec![Constraint::Regex("a+".to_owned())],
            };
            for (text, want) in [("a", true), ("aaa", true), ("ab", false), ("ba", false)] {
                let value = PyString::new(py, text).into_any();
                assert_eq!(decide(py, &pattern, &value, &pool, &[]), want, "{text:?}");
            }

            // Every constraint must hold, not merely one.
            let both = refine(vec![
                Constraint::Ge(OperandIx::new(0)),
                Constraint::Le(OperandIx::new(0)),
            ]);
            assert!(decide(py, &both, &int(5), &pool, &[]));
            assert!(!decide(py, &both, &int(6), &pool, &[]));
        });
    }

    #[test]
    fn a_reference_unfolds_its_definition_and_a_cycle_is_refused() {
        Python::attach(|py| {
            // `T = None | {"next": T}`: a finite chain is a member.
            let defs = vec![Schema::Union(vec![
                Schema::NoneType,
                Schema::record(
                    vec![Field {
                        name: "next".to_owned(),
                        schema: Schema::Ref(DefIx::new(0)),
                        required: true,
                    }],
                    Openness::Closed,
                ),
            ])];
            let schema = Schema::Ref(DefIx::new(0));

            let none = py.None().into_bound(py);
            assert!(decide(py, &schema, &none, &[], &defs));

            let one = PyDict::new(py);
            one.set_item("next", py.None()).expect("set_item");
            assert!(decide(py, &schema, &one.clone().into_any(), &[], &defs));

            let two = PyDict::new(py);
            two.set_item("next", &one).expect("set_item");
            assert!(decide(py, &schema, &two.into_any(), &[], &defs));

            let wrong = PyDict::new(py);
            wrong.set_item("next", 1i64).expect("set_item");
            assert!(!decide(py, &schema, &wrong.into_any(), &[], &defs));

            // A value that contains itself is refused rather than looped on.
            let cyclic = PyDict::new(py);
            cyclic.set_item("next", &cyclic).expect("set_item");
            assert!(!decide(py, &schema, &cyclic.into_any(), &[], &defs));

            // A reference past the definitions table is an internal invariant
            // break, and degrades to a non-member rather than panicking. Checked
            // in release only: the walk `debug_assert`s it.
            #[cfg(not(debug_assertions))]
            assert!(!decide(py, &Schema::Ref(DefIx::new(9)), &none, &[], &defs));
        });
    }

    /// Define a small class hierarchy in the embedded interpreter, for the two
    /// class-based arms. `Point` carries `x: int` and `y: int`; `Sub` is a
    /// subclass of it; `Other` is unrelated.
    fn classes(py: Python<'_>) -> Bound<'_, PyAny> {
        let module = PyModule::from_code(
            py,
            std::ffi::CString::new(
                "class Point:\n\
                 \x20   def __init__(self, x, y):\n\
                 \x20       self.x = x\n\
                 \x20       self.y = y\n\
                 class Sub(Point):\n\
                 \x20   pass\n\
                 class Other:\n\
                 \x20   pass\n\
                 class NoAttrs:\n\
                 \x20   pass\n",
            )
            .expect("no interior nul")
            .as_c_str(),
            std::ffi::CString::new("classes.py")
                .expect("no interior nul")
                .as_c_str(),
            std::ffi::CString::new("classes")
                .expect("no interior nul")
                .as_c_str(),
        )
        .expect("the module compiles");
        module.into_any()
    }

    #[test]
    fn an_instance_atom_admits_the_class_and_its_subclasses() {
        Python::attach(|py| {
            let module = classes(py);
            let point_class = module.getattr("Point").expect("Point");
            let sub_class = module.getattr("Sub").expect("Sub");
            let other_class = module.getattr("Other").expect("Other");
            let pool = vec![point_class.clone().unbind()];
            let schema = Schema::Instance(ClassIx::new(0));

            let point = point_class.call1((1i64, 2i64)).expect("Point(1, 2)");
            let sub = sub_class.call1((1i64, 2i64)).expect("Sub(1, 2)");
            let other = other_class.call0().expect("Other()");

            // `isinstance`, so a subclass instance is a member and an unrelated
            // one is not. A non-object value is not a member either.
            assert!(decide(py, &schema, &point, &pool, &[]));
            assert!(decide(py, &schema, &sub, &pool, &[]));
            assert!(!decide(py, &schema, &other, &pool, &[]));
            assert!(!decide(
                py,
                &schema,
                &PyInt::new(py, 1i64).into_any(),
                &pool,
                &[]
            ));
            // The class itself is not one of its instances.
            assert!(!decide(py, &schema, &point_class, &pool, &[]));
        });
    }

    #[test]
    fn an_attribute_record_checks_the_class_then_every_attribute() {
        Python::attach(|py| {
            let module = classes(py);
            let point_class = module.getattr("Point").expect("Point");
            let other_class = module.getattr("Other").expect("Other");
            let bare_class = module.getattr("NoAttrs").expect("NoAttrs");
            let pool = vec![point_class.clone().unbind(), bare_class.clone().unbind()];
            let field = |name: &str, schema| Field {
                name: name.to_owned(),
                schema,
                required: true,
            };
            let object = |class, fields| {
                Schema::meet([
                    Schema::Instance(ClassIx::new(class)),
                    Schema::AttrRecord { fields },
                ])
            };
            let schema = object(0, vec![field("x", Schema::Int), field("y", Schema::Int)]);

            let good = point_class.call1((1i64, 2i64)).expect("Point(1, 2)");
            assert!(decide(py, &schema, &good, &pool, &[]));

            // Every attribute must match: one wrong value rejects the whole.
            let text = PyString::new(py, "x").into_any();
            let wrong = point_class.call1((1i64, text)).expect("Point(1, \"x\")");
            assert!(!decide(py, &schema, &wrong, &pool, &[]));

            // The isinstance check is not rescued by the attributes matching: an
            // unrelated object carrying x and y is still not a Point.
            let impostor = other_class.call0().expect("Other()");
            impostor.setattr("x", 1i64).expect("setattr x");
            impostor.setattr("y", 2i64).expect("setattr y");
            assert!(!decide(py, &schema, &impostor, &pool, &[]));

            // A missing attribute is a rejection, not a raise.
            let missing = object(1, vec![field("absent", Schema::Int)]);
            let bare = bare_class.call0().expect("NoAttrs()");
            assert!(!decide(py, &missing, &bare, &pool, &[]));

            // The class atom is what the frontend emits when nothing is declared.
            let nominal = Schema::Instance(ClassIx::new(0));
            assert!(decide(py, &nominal, &good, &pool, &[]));
            assert!(!decide(py, &nominal, &impostor, &pool, &[]));

            // An optional attribute is satisfied by its absence, and still
            // checked when the value carries it. No annotation builds one -- a
            // declared attribute is one an instance has -- so the record's own
            // denotation is what holds the walk to it.
            let optional = Schema::AttrRecord {
                fields: vec![Field {
                    name: "absent".to_owned(),
                    schema: Schema::Int,
                    required: false,
                }],
            };
            assert!(decide(py, &optional, &bare, &pool, &[]));
            bare.setattr("absent", "not an int")
                .expect("setattr absent");
            assert!(!decide(py, &optional, &bare, &pool, &[]));
        });
    }

    /// Decide membership and report whether a fatal interpreter signal was
    /// recorded on the way. The signal is what the entry point re-raises, so a
    /// corpus that only reads the verdict cannot tell a refused value from an
    /// interrupted walk.
    fn decide_with_fatal(
        py: Python<'_>,
        schema: &Schema,
        value: &Bound<'_, PyAny>,
        pool: &[Py<PyAny>],
    ) -> (bool, bool) {
        let index = build_index(py, schema, &[], pool);
        let state = WalkState::new();
        let ctx = Ctx {
            pool,
            defs: &[],
            records: &index.records,
            attrs: &index.attrs,
            unions: &index.unions,
            regexes: &index.regexes,
            guard: &state.guard,
            depth: &state.depth,
            fatal: &state.fatal,
            fatal_seen: &state.fatal_seen,
            mode: WalkMode::Fast,
        };
        let ok = member(
            schema,
            &Value::Py(value),
            &mut Vec::new(),
            ctx,
            &mut Vec::new(),
        );
        (ok, state.fatal.borrow().is_some())
    }

    /// Decide membership of a parsed JSON value, in the mode `is_valid_json` uses.
    fn holds_json(py: Python<'_>, schema: &Schema, json: &JsonValue<'_>) -> bool {
        let index = build_index(py, schema, &[], &[]);
        let state = WalkState::new();
        let ctx = Ctx {
            pool: &[],
            defs: &[],
            records: &index.records,
            attrs: &index.attrs,
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
            &Value::Json(py, json),
            &mut Vec::new(),
            ctx,
            &mut Vec::new(),
        )
    }

    fn json_object<'a>(pairs: Vec<(&'a str, JsonValue<'a>)>) -> JsonValue<'a> {
        JsonValue::Object(std::sync::Arc::new(
            pairs
                .into_iter()
                .map(|(k, v)| (Cow::Borrowed(k), v))
                .collect(),
        ))
    }

    fn field(name: &str, schema: Schema, required: bool) -> Field {
        Field {
            name: name.to_owned(),
            schema,
            required,
        }
    }

    #[test]
    fn a_json_keyed_map_decides_like_its_object_form() {
        Python::attach(|py| {
            // The JSON path has its own keyed-map walk -- it reads entries in
            // document order rather than a dict -- so every rule the object path
            // holds is asserted against it separately.
            let closed = Schema::record(
                vec![
                    field("x", Schema::Int, true),
                    field("y", Schema::Str, false),
                ],
                Openness::Closed,
            );
            let open = Schema::record(vec![field("x", Schema::Int, true)], Openness::Open);
            let mapping = Schema::mapping(MapClause {
                key: Schema::Str,
                value: Schema::Int,
            });

            for (schema, entries, want) in [
                // The required field must be present and match.
                (&closed, vec![("x", JsonValue::Int(1))], true),
                (&closed, vec![("y", JsonValue::Str("a".into()))], false),
                (&closed, vec![("x", JsonValue::Str("a".into()))], false),
                // The optional one may be absent, and must match when present.
                (
                    &closed,
                    vec![("x", JsonValue::Int(1)), ("y", JsonValue::Str("a".into()))],
                    true,
                ),
                (
                    &closed,
                    vec![("x", JsonValue::Int(1)), ("y", JsonValue::Int(2))],
                    false,
                ),
                // A closed record forbids an undeclared key; an open one admits it.
                (
                    &closed,
                    vec![("x", JsonValue::Int(1)), ("z", JsonValue::Int(2))],
                    false,
                ),
                (
                    &open,
                    vec![("x", JsonValue::Int(1)), ("z", JsonValue::Int(2))],
                    true,
                ),
                // A pure mapping judges every key and value by the clause.
                (&mapping, vec![("k", JsonValue::Int(1))], true),
                (&mapping, vec![("k", JsonValue::Str("a".into()))], false),
                (&mapping, vec![], true),
            ] {
                let json = json_object(entries.clone());
                assert_eq!(holds_json(py, schema, &json), want, "{entries:?}");
            }

            // A duplicate key takes its LAST value, which is what `json.loads`
            // would have produced, so the two input paths cannot disagree here.
            let last_wins = json_object(vec![
                ("x", JsonValue::Str("a".into())),
                ("x", JsonValue::Int(1)),
            ]);
            assert!(holds_json(py, &closed, &last_wins));
            let last_loses = json_object(vec![
                ("x", JsonValue::Int(1)),
                ("x", JsonValue::Str("a".into())),
            ]);
            assert!(!holds_json(py, &closed, &last_loses));
            // The same rule for a key the default clause covers.
            let default_last = json_object(vec![
                ("k", JsonValue::Int(1)),
                ("k", JsonValue::Str("a".into())),
            ]);
            assert!(!holds_json(py, &mapping, &default_last));
        });
    }

    #[test]
    fn a_composite_rejects_when_any_one_element_fails() {
        Python::attach(|py| {
            // Each container folds its element verdicts with a conjunction. A
            // disjunction there accepts a container whose first element happens
            // to match, which is the shape a single all-good case cannot see --
            // so every container is driven with a value that is part-good.
            let int_text = PyList::new(py, [1i64]).expect("builds");
            int_text.append(PyString::new(py, "x")).expect("append");

            let list_schema = Schema::list(SeqShape::homogeneous(Schema::Int));
            case(py, &list_schema, &int_text.clone().into_any(), false);

            let tuple_schema = Schema::tuple(SeqShape::homogeneous(Schema::Int));
            let tuple = PyTuple::new(py, [1i64, 2]).expect("builds").into_any();
            case(py, &tuple_schema, &tuple, true);
            let mixed_tuple = int_text.to_tuple().into_any();
            case(py, &tuple_schema, &mixed_tuple, false);

            // Both container kinds on the JSON path too, where the fold is a
            // separate arm.
            let good = JsonValue::Array(std::sync::Arc::new(vec![
                JsonValue::Int(1),
                JsonValue::Int(2),
            ]));
            let part_good = JsonValue::Array(std::sync::Arc::new(vec![
                JsonValue::Int(1),
                JsonValue::Str("x".into()),
            ]));
            assert!(holds_json(py, &list_schema, &good));
            assert!(!holds_json(py, &list_schema, &part_good));
            // JSON has no tuple: `json.loads` produces a list, so the JSON path
            // has no tuple arm at all and a tuple schema rejects an array
            // whatever its elements are. Pinned here because it is the one place
            // the two input paths deliberately decide differently.
            assert!(!holds_json(py, &tuple_schema, &good));
            assert!(!holds_json(py, &tuple_schema, &part_good));
            assert!(holds(py, &tuple_schema, &tuple, &[], &[]));

            // Sets and frozensets fold the same way.
            let mixed_set = PySet::new(py, [1i64]).expect("builds");
            mixed_set.add(PyString::new(py, "x")).expect("add");
            case(
                py,
                &Schema::Set(Box::new(Schema::Int)),
                &mixed_set.clone().into_any(),
                false,
            );
            let mixed_frozen = PyFrozenSet::new(py, mixed_set.iter())
                .expect("builds")
                .into_any();
            case(
                py,
                &Schema::FrozenSet(Box::new(Schema::Int)),
                &mixed_frozen,
                false,
            );
            let good_frozen = PyFrozenSet::new(py, [1i64, 2]).expect("builds").into_any();
            case(
                py,
                &Schema::FrozenSet(Box::new(Schema::Int)),
                &good_frozen,
                true,
            );
        });
    }

    #[test]
    fn a_union_explains_the_branch_that_descended_furthest() {
        Python::attach(|py| {
            // No branch matches, and the two fail at different depths: one is a
            // flat type mismatch, the other descends into a field. The report is
            // the deeper branch's, so a reader is shown the branch the value was
            // closest to rather than every branch's noise.
            let deep = Schema::record(vec![field("x", Schema::Int, true)], Openness::Closed);
            let schema = Schema::Union(vec![Schema::Int, deep]);
            let value = PyDict::new(py);
            value.set_item("x", PyString::new(py, "a")).expect("set");
            let (ok, violations) = explain(py, &schema, &value.into_any(), &[], &[]);
            assert!(!ok);
            assert_eq!(violations.len(), 1);
            assert_eq!(violations[0].location(), "x");
            assert_ne!(violations[0].code, "union_error");

            // No branch makes any progress: a single union error, not two flat
            // mismatches. This is the arm the depth comparison selects between.
            let flat = Schema::Union(vec![Schema::Int, Schema::Str]);
            let number = PyFloat::new(py, 1.5).into_any();
            let (ok, violations) = explain(py, &flat, &number, &[], &[]);
            assert!(!ok);
            assert_eq!(violations.len(), 1);
            assert_eq!(violations[0].code, "union_error");

            // The probe runs with its own aggregating mode, so it measures how
            // far each branch got even when the caller asked to stop at the
            // first violation. Inheriting the caller's mode truncates a branch's
            // report and can change which branch is judged closest.
            let index = build_index(py, &schema, &[], &[]);
            let deep_value = PyDict::new(py);
            deep_value
                .set_item("x", PyString::new(py, "a"))
                .expect("set");
            let deep_value = deep_value.into_any();
            let state = WalkState::new();
            let mut out = Vec::new();
            let ok = member(
                &schema,
                &Value::Py(&deep_value),
                &mut Vec::new(),
                Ctx {
                    pool: &[],
                    defs: &[],
                    records: &index.records,
                    attrs: &index.attrs,
                    unions: &index.unions,
                    regexes: &index.regexes,
                    guard: &state.guard,
                    depth: &state.depth,
                    fatal: &state.fatal,
                    fatal_seen: &state.fatal_seen,
                    mode: WalkMode::ExplainFailFast,
                },
                &mut out,
            );
            assert!(!ok);
            assert_eq!(out.len(), 1);
            assert_eq!(out[0].location(), "x");

            // The probe aggregates a branch's violations even when the caller
            // asked to stop at the first, so the whole of the closest branch is
            // reported. A branch with two failing fields is what distinguishes
            // that from a probe that inherited the caller's mode.
            let wide = Schema::record(
                vec![field("p", Schema::Int, true), field("q", Schema::Int, true)],
                Openness::Closed,
            );
            let union_wide = Schema::Union(vec![Schema::Int, wide]);
            let wide_value = PyDict::new(py);
            for key in ["p", "q"] {
                wide_value
                    .set_item(key, PyString::new(py, "s"))
                    .expect("set");
            }
            let wide_value = wide_value.into_any();
            assert_eq!(
                run_mode(py, &union_wide, &wide_value, WalkMode::ExplainFailFast),
                (false, 2)
            );

            // A tie keeps the earliest branch, so the choice is deterministic.
            let left = Schema::record(vec![field("a", Schema::Int, true)], Openness::Closed);
            let right = Schema::record(vec![field("b", Schema::Int, true)], Openness::Closed);
            let tied = Schema::Union(vec![left, right]);
            let value = PyDict::new(py);
            value.set_item("a", PyString::new(py, "s")).expect("set");
            value.set_item("b", PyString::new(py, "s")).expect("set");
            let (_, violations) = explain(py, &tied, &value.into_any(), &[], &[]);
            assert_eq!(violations[0].location(), "a");
        });
    }

    #[test]
    fn an_explaining_walk_aggregates_every_independent_failure() {
        Python::attach(|py| {
            // Three fields fail independently. Explain mode reports all three;
            // fail-fast reports the first; the fast path reports none and
            // allocates nothing. All three modes agree on the verdict.
            let schema = Schema::record(
                vec![
                    field("a", Schema::Int, true),
                    field("b", Schema::Int, true),
                    field("c", Schema::Int, true),
                ],
                Openness::Closed,
            );
            let value = PyDict::new(py);
            for key in ["a", "b", "c"] {
                value.set_item(key, PyString::new(py, "s")).expect("set");
            }
            let value = value.into_any();

            let index = build_index(py, &schema, &[], &[]);
            let run = |mode: WalkMode| {
                let state = WalkState::new();
                let ctx = Ctx {
                    pool: &[],
                    defs: &[],
                    records: &index.records,
                    attrs: &index.attrs,
                    unions: &index.unions,
                    regexes: &index.regexes,
                    guard: &state.guard,
                    depth: &state.depth,
                    fatal: &state.fatal,
                    fatal_seen: &state.fatal_seen,
                    mode,
                };
                let mut out = Vec::new();
                let ok = member(&schema, &Value::Py(&value), &mut Vec::new(), ctx, &mut out);
                (ok, out.len())
            };
            assert_eq!(run(WalkMode::Explain), (false, 3));
            assert_eq!(run(WalkMode::ExplainFailFast), (false, 1));
            assert_eq!(run(WalkMode::Fast), (false, 0));
        });
    }

    /// Run one membership walk in a given mode and report the verdict and how
    /// many violations it aggregated.
    fn run_mode(
        py: Python<'_>,
        schema: &Schema,
        value: &Bound<'_, PyAny>,
        mode: WalkMode,
    ) -> (bool, usize) {
        let index = build_index(py, schema, &[], &[]);
        let state = WalkState::new();
        let ctx = Ctx {
            pool: &[],
            defs: &[],
            records: &index.records,
            attrs: &index.attrs,
            unions: &index.unions,
            regexes: &index.regexes,
            guard: &state.guard,
            depth: &state.depth,
            fatal: &state.fatal,
            fatal_seen: &state.fatal_seen,
            mode,
        };
        let mut out = Vec::new();
        let ok = member(schema, &Value::Py(value), &mut Vec::new(), ctx, &mut out);
        (ok, out.len())
    }

    #[test]
    fn a_composite_stops_at_its_first_failing_child_only_when_asked_to() {
        Python::attach(|py| {
            // Two elements fail independently. Aggregating mode reports both;
            // fail-fast reports the first; the fast path reports none. Every
            // composite consults one predicate for this, so a sequence pins it
            // for the arms a record does not reach.
            let schema = Schema::list(SeqShape::homogeneous(Schema::Int));
            let value = PyList::new(py, [1i64]).expect("builds");
            value.append(PyString::new(py, "a")).expect("append");
            value.append(PyString::new(py, "b")).expect("append");
            let value = value.into_any();

            assert_eq!(run_mode(py, &schema, &value, WalkMode::Explain), (false, 2));
            assert_eq!(
                run_mode(py, &schema, &value, WalkMode::ExplainFailFast),
                (false, 1)
            );
            assert_eq!(run_mode(py, &schema, &value, WalkMode::Fast), (false, 0));

            // The same for a set, whose fold is a separate arm.
            let set_schema = Schema::Set(Box::new(Schema::Int));
            let set = PySet::new(py, [1i64]).expect("builds");
            set.add(PyString::new(py, "a")).expect("add");
            set.add(PyString::new(py, "b")).expect("add");
            let set = set.into_any();
            assert_eq!(
                run_mode(py, &set_schema, &set, WalkMode::Explain),
                (false, 2)
            );
            assert_eq!(
                run_mode(py, &set_schema, &set, WalkMode::ExplainFailFast),
                (false, 1)
            );
        });
    }

    #[test]
    fn a_raising_comparison_folds_and_a_fatal_signal_does_not() {
        Python::attach(|py| {
            // A value that cannot answer "are you in this set?" is not in it --
            // unless the interpreter is unwinding, which is not an answer at all.
            let module = PyModule::from_code(
                py,
                std::ffi::CString::new(
                    "class Rude:\n\
                     \x20   def __eq__(self, other):\n\
                     \x20       raise ValueError('no')\n\
                     class Stopping:\n\
                     \x20   def __eq__(self, other):\n\
                     \x20       raise KeyboardInterrupt\n\
                     class NoLen:\n\
                     \x20   def __len__(self):\n\
                     \x20       raise TypeError('no')\n\
                     class OutOfMemory:\n\
                     \x20   def __eq__(self, other):\n\
                     \x20       raise MemoryError\n\
                     class TooDeep:\n\
                     \x20   def __eq__(self, other):\n\
                     \x20       raise RecursionError\n",
                )
                .expect("no interior nul")
                .as_c_str(),
                std::ffi::CString::new("raising.py")
                    .expect("no interior nul")
                    .as_c_str(),
                std::ffi::CString::new("raising")
                    .expect("no interior nul")
                    .as_c_str(),
            )
            .expect("the module compiles");

            let literal = Schema::Literal(ConstIx::new(0));
            let instance = |name: &str| {
                module
                    .getattr(name)
                    .expect("the class")
                    .call0()
                    .expect("the instance")
            };

            // A literal's same-type test runs BEFORE `==`, so a value of another
            // type never reaches the comparison at all. Pinned first, because it
            // is why the two cases below have to pool an instance of the raising
            // class rather than an int.
            let one = PyInt::new(py, 1i64).into_any();
            let int_pool = vec![one.unbind()];
            let rude = instance("Rude");
            assert_eq!(
                decide_with_fatal(py, &literal, &rude, &int_pool),
                (false, false)
            );

            // An ordinary exception folds to a non-member, and records nothing.
            let rude_pool = vec![instance("Rude").unbind()];
            assert_eq!(
                decide_with_fatal(py, &literal, &rude, &rude_pool),
                (false, false)
            );

            // A fatal signal is recorded so the entry point re-raises it. The
            // local answer is still a non-member so the frame returns.
            let stopping = instance("Stopping");
            let stopping_pool = vec![instance("Stopping").unbind()];
            assert_eq!(
                decide_with_fatal(py, &literal, &stopping, &stopping_pool),
                (false, true)
            );

            // MemoryError and RecursionError ARE ordinary exceptions, so the
            // base-exception test alone misses them -- and they still mean the
            // interpreter cannot continue. Each is a separate disjunct of the
            // classifier, so each needs its own case.
            for name in ["OutOfMemory", "TooDeep"] {
                let value = instance(name);
                let pool = vec![instance(name).unbind()];
                assert_eq!(
                    decide_with_fatal(py, &literal, &value, &pool),
                    (false, true),
                    "{name}"
                );
            }

            // The same split at a length bound, which reaches the value through
            // `__len__` rather than `__eq__`.
            let sized = Schema::Refine {
                base: Box::new(Schema::ANYTHING),
                constraints: vec![Constraint::MinLen(1)],
            };
            let no_len = module.getattr("NoLen").expect("NoLen").call0().expect("()");
            let (ok, fatal) = decide_with_fatal(py, &sized, &no_len, &[]);
            assert!(
                !ok,
                "a value whose __len__ raises cannot satisfy a length bound"
            );
            assert!(!fatal, "a TypeError is an ordinary exception, not a signal");
        });
    }

    #[test]
    fn a_predicate_constraint_runs_the_pooled_callable() {
        Python::attach(|py| {
            let module = PyModule::from_code(
                py,
                std::ffi::CString::new("def is_even(x):\n\x20   return x % 2 == 0\n")
                    .expect("no interior nul")
                    .as_c_str(),
                std::ffi::CString::new("pred.py")
                    .expect("no interior nul")
                    .as_c_str(),
                std::ffi::CString::new("pred")
                    .expect("no interior nul")
                    .as_c_str(),
            )
            .expect("the module compiles");
            let is_even = module.getattr("is_even").expect("is_even");
            let pool = vec![is_even.unbind()];
            let schema = Schema::Refine {
                base: Box::new(Schema::Int),
                constraints: vec![Constraint::Predicate(PredIx::new(0))],
            };
            assert!(decide(
                py,
                &schema,
                &PyInt::new(py, 4i64).into_any(),
                &pool,
                &[]
            ));
            assert!(!decide(
                py,
                &schema,
                &PyInt::new(py, 3i64).into_any(),
                &pool,
                &[]
            ));
        });
    }

    #[test]
    fn a_literal_union_decides_alike_through_the_fast_plan_and_the_scan() {
        Python::attach(|py| {
            // A union whose members are all literals is decided by a precomputed
            // set lookup on the membership path, and by the linear scan
            // everywhere else. The two must agree, and the plan must not be
            // consulted while explaining -- an early return there would report a
            // rejection with no violation behind it.
            let pool: Vec<Py<PyAny>> = (1i64..=3)
                .map(|n| PyInt::new(py, n).into_any().unbind())
                .collect();
            let schema = Schema::Union((0..3).map(|i| Schema::Literal(ConstIx::new(i))).collect());
            for n in 1i64..=3 {
                assert!(decide(
                    py,
                    &schema,
                    &PyInt::new(py, n).into_any(),
                    &pool,
                    &[]
                ));
            }
            // Rejections, which are what an explain walk must produce a violation
            // for. `decide` runs both modes and holds them to agreeing.
            for n in [0i64, 4, 99] {
                assert!(!decide(
                    py,
                    &schema,
                    &PyInt::new(py, n).into_any(),
                    &pool,
                    &[]
                ));
            }
            // A value of a type the plan does not cover falls to the scan.
            let text = PyString::new(py, "1").into_any();
            assert!(!decide(py, &schema, &text, &pool, &[]));
        });
    }

    #[test]
    fn the_explain_pass_reports_only_the_fields_that_actually_failed() {
        Python::attach(|py| {
            // The explain pass re-walks a record that already failed. An absent
            // OPTIONAL field is not a failure, so it must not be reported -- and
            // the only way to see that is a record that fails for another reason
            // while an optional field is absent.
            let schema = Schema::record(
                vec![
                    field("x", Schema::Int, true),
                    field("y", Schema::Str, false),
                ],
                Openness::Closed,
            );
            let value = PyDict::new(py);
            value.set_item("x", PyString::new(py, "s")).expect("set");
            let (ok, violations) = explain(py, &schema, &value.into_any(), &[], &[]);
            assert!(!ok);
            assert_eq!(violations.len(), 1, "{violations:?}");
            assert_eq!(violations[0].location(), "x");

            // A required field that IS absent is reported, and only once.
            let empty = PyDict::new(py);
            let (ok, violations) = explain(py, &schema, &empty.into_any(), &[], &[]);
            assert!(!ok);
            assert_eq!(violations.len(), 1, "{violations:?}");
            assert_eq!(violations[0].code, "missing_key");
        });
    }

    #[test]
    fn a_closed_record_reports_every_extra_key_unless_fail_fast_stops_it() {
        Python::attach(|py| {
            // Two undeclared keys, so the loop that reports them is driven past
            // its first iteration: aggregating mode reports both, fail-fast the
            // first only.
            let schema = Schema::record(vec![field("x", Schema::Int, true)], Openness::Closed);
            let value = PyDict::new(py);
            value.set_item("x", 1i64).expect("set");
            value.set_item("extra1", 1i64).expect("set");
            value.set_item("extra2", 1i64).expect("set");
            let value = value.into_any();

            let index = build_index(py, &schema, &[], &[]);
            let run = |mode: WalkMode| {
                let state = WalkState::new();
                let ctx = Ctx {
                    pool: &[],
                    defs: &[],
                    records: &index.records,
                    attrs: &index.attrs,
                    unions: &index.unions,
                    regexes: &index.regexes,
                    guard: &state.guard,
                    depth: &state.depth,
                    fatal: &state.fatal,
                    fatal_seen: &state.fatal_seen,
                    mode,
                };
                let mut out = Vec::new();
                let ok = member(&schema, &Value::Py(&value), &mut Vec::new(), ctx, &mut out);
                (ok, out.len())
            };
            assert_eq!(run(WalkMode::Explain), (false, 2));
            assert_eq!(run(WalkMode::ExplainFailFast), (false, 1));
        });
    }

    #[test]
    fn a_fatal_signal_propagates_from_an_attribute_and_from_a_predicate() {
        Python::attach(|py| {
            // The two sites the literal and length cases do not reach: attribute
            // access on an object schema, and a user predicate. Both fold an
            // ordinary exception to a non-member and record a fatal signal.
            let module = PyModule::from_code(
                py,
                std::ffi::CString::new(
                    "class Base:\n\
                     \x20   pass\n\
                     class RudeAttr(Base):\n\
                     \x20   def __getattr__(self, name):\n\
                     \x20       raise ValueError('no')\n\
                     class StoppingAttr(Base):\n\
                     \x20   def __getattr__(self, name):\n\
                     \x20       raise KeyboardInterrupt\n\
                     def rude(x):\n\
                     \x20   raise ValueError('no')\n\
                     def stopping(x):\n\
                     \x20   raise KeyboardInterrupt\n",
                )
                .expect("no interior nul")
                .as_c_str(),
                std::ffi::CString::new("fatal.py")
                    .expect("no interior nul")
                    .as_c_str(),
                std::ffi::CString::new("fatal")
                    .expect("no interior nul")
                    .as_c_str(),
            )
            .expect("the module compiles");
            let base = module.getattr("Base").expect("Base");

            let attrs = Schema::meet([
                Schema::Instance(ClassIx::new(0)),
                Schema::AttrRecord {
                    fields: vec![field("missing", Schema::Int, true)],
                },
            ]);
            let pool = vec![base.clone().unbind()];
            for (name, want_fatal) in [("RudeAttr", false), ("StoppingAttr", true)] {
                let value = module.getattr(name).expect("class").call0().expect("()");
                assert_eq!(
                    decide_with_fatal(py, &attrs, &value, &pool),
                    (false, want_fatal),
                    "{name}"
                );
            }

            let one = PyInt::new(py, 1i64).into_any();
            for (name, want_fatal) in [("rude", false), ("stopping", true)] {
                let predicate = module.getattr(name).expect("callable");
                let pool = vec![predicate.unbind()];
                let schema = Schema::Refine {
                    base: Box::new(Schema::Int),
                    constraints: vec![Constraint::Predicate(PredIx::new(0))],
                };
                assert_eq!(
                    decide_with_fatal(py, &schema, &one, &pool),
                    (false, want_fatal),
                    "{name}"
                );
            }
        });
    }

    // SWEEP-SKIP: this case exists to prove a bound, so a mutation that removes
    // the bound makes it run without end. It stays in the test lane and leaves
    // the mutation sweep, where a run that returns no verdict is a rig fault.
    #[test]
    fn recursion_deeper_than_the_bound_is_refused() {
        Python::attach(|py| {
            // `T = None | {"next": T}`. A chain the walk can carry is a member; a
            // chain past the guard's depth bound is refused rather than recursed
            // into, because the walk descends one native frame per level.
            let defs = vec![Schema::Union(vec![
                Schema::NoneType,
                Schema::record(
                    vec![field("next", Schema::Ref(DefIx::new(0)), true)],
                    Openness::Closed,
                ),
            ])];
            let schema = Schema::Ref(DefIx::new(0));

            let chain = |depth: usize| {
                let mut node = py.None().into_bound(py);
                for _ in 0..depth {
                    let dict = PyDict::new(py);
                    dict.set_item("next", &node).expect("set_item");
                    node = dict.into_any();
                }
                node
            };
            assert!(decide(py, &schema, &chain(8), &[], &defs));
            assert!(decide(
                py,
                &schema,
                &chain(MAX_RECURSION_DEPTH - 1),
                &[],
                &defs
            ));
            assert!(!decide(
                py,
                &schema,
                &chain(MAX_RECURSION_DEPTH + 2),
                &[],
                &defs
            ));
        });
    }

    /// The JSON record path answers the same with the plan and without it.
    ///
    /// Whether a key is a declared field is read from the per-validator record
    /// plan, and a schema absent from that plan falls back to scanning the field
    /// list. The fallback is what keeps correctness from depending on the index
    /// being complete, so it has to answer the same -- and nothing exercises it
    /// through the ordinary entry points, because the index is always built.
    #[test]
    fn the_json_record_path_agrees_with_and_without_its_plan() {
        Python::attach(|py| {
            let schema = Schema::keyed_map(
                vec![Field {
                    name: "a".to_owned(),
                    schema: Schema::Int,
                    required: true,
                }],
                vec![MapClause {
                    key: Schema::Str,
                    value: Schema::Str,
                }],
            );
            let Schema::KeyedMap { fields, defaults } = &schema else {
                panic!("the schema is a keyed map")
            };

            // `a` is the declared field and takes an int; `b` is undeclared and
            // must go to the clause, which takes a string. A reading that
            // confused the two would accept the first and reject the second.
            let good = [
                ("a".into(), JsonValue::Int(1)),
                ("b".into(), JsonValue::Str("x".into())),
            ];
            let bad = [
                ("a".into(), JsonValue::Int(1)),
                ("b".into(), JsonValue::Int(2)),
            ];

            let built = build_index(py, &schema, &[], &[]);
            let empty = ValidatorIndex::default();
            for index in [&built, &empty] {
                let state = WalkState::new();
                let ctx = Ctx {
                    pool: &[],
                    defs: &[],
                    records: &index.records,
                    attrs: &index.attrs,
                    unions: &index.unions,
                    regexes: &index.regexes,
                    guard: &state.guard,
                    depth: &state.depth,
                    fatal: &state.fatal,
                    fatal_seen: &state.fatal_seen,
                    mode: WalkMode::Fast,
                };
                assert!(keyed_map_matches_json(fields, defaults, py, &good, ctx));
                assert!(!keyed_map_matches_json(fields, defaults, py, &bad, ctx));
            }
            // The plan really was absent for the second pass, so the two answers
            // came from the two readings rather than from one of them twice.
            assert!(built.records.contains_key(&(fields.as_ptr() as usize)));
            assert!(empty.records.is_empty());
        });
    }

    #[test]
    fn the_json_path_and_the_object_path_agree() {
        Python::attach(|py| {
            // The two input paths share one walk, so they must decide alike. This
            // drives the `Value::Json` arms the object corpus above never reaches.
            let schema = Schema::list(SeqShape::homogeneous(Schema::Int));
            let json = JsonValue::Array(std::sync::Arc::new(vec![
                JsonValue::Int(1),
                JsonValue::Int(2),
            ]));
            let index = build_index(py, &schema, &[], &[]);
            let state = WalkState::new();
            let ctx = Ctx {
                pool: &[],
                defs: &[],
                records: &index.records,
                attrs: &index.attrs,
                unions: &index.unions,
                regexes: &index.regexes,
                guard: &state.guard,
                depth: &state.depth,
                fatal: &state.fatal,
                fatal_seen: &state.fatal_seen,
                mode: WalkMode::Fast,
            };
            assert!(member(
                &schema,
                &Value::Json(py, &json),
                &mut Vec::new(),
                ctx,
                &mut Vec::new()
            ));
            assert!(holds(py, &schema, &list_of(py, vec![1, 2]), &[], &[]));

            let bad = JsonValue::Array(std::sync::Arc::new(vec![JsonValue::Str("x".into())]));
            assert!(!member(
                &schema,
                &Value::Json(py, &bad),
                &mut Vec::new(),
                ctx,
                &mut Vec::new()
            ));
        });
    }
}

#[cfg(test)]
mod label_tests {
    use super::{BranchLabels, UNION_LABEL_LIMIT};

    /// The list a union's `expected` reads is bounded, and says so when it is
    /// short.
    ///
    /// A wide union -- an error-code table, a currency list -- has more branches
    /// than a message can carry, so the list stops and ends in `...`. The two
    /// halves are the claim: a reader must not be shown a truncated list as if
    /// it were the whole set, and must not be shown `...` after a complete one.
    /// Neither needs an interpreter: the labels arrive as strings.
    #[test]
    fn the_branch_list_stops_at_its_limit_and_marks_that_it_did() {
        let mut labels = BranchLabels::new();
        labels.push("int".to_owned());
        labels.push("str".to_owned());
        assert_eq!(labels.render(), "one of: int, str");

        // Exactly the limit is a complete list: the boundary belongs to the
        // labels, not to the ellipsis.
        let mut full = BranchLabels::new();
        for i in 0..UNION_LABEL_LIMIT {
            full.push(format!("b{i}"));
        }
        let rendered = full.render();
        assert!(!rendered.ends_with(", ..."), "{rendered} is not truncated");
        assert_eq!(rendered.matches(", ").count(), UNION_LABEL_LIMIT - 1);

        // One past it is short, and the label is dropped rather than kept.
        full.push("dropped".to_owned());
        let rendered = full.render();
        assert!(rendered.ends_with(", ..."), "{rendered} is truncated");
        assert!(!rendered.contains("dropped"));
        assert_eq!(rendered.matches(", ").count(), UNION_LABEL_LIMIT);

        // An empty union renders its prefix and nothing else, which is what a
        // renderer returning a fixed string would also do for every other case.
        assert_eq!(BranchLabels::new().render(), "one of: ");
    }
}
