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

use std::sync::Arc;

use std::borrow::Cow;
use std::ops::ControlFlow;

use jiter::JsonValue;
use pyo3::exceptions::{PyException, PyMemoryError, PyRecursionError};
use pyo3::prelude::*;
use pyo3::sync::critical_section::with_critical_section;
use pyo3::types::{PyDict, PyFrozenSet, PyList, PySet, PyString, PyTuple};
use rustc_hash::{FxHashMap, FxHashSet};
use valgebra_core::{
    ClassIx, CollKind, ConstIx, Constraint, DefIx, Field, MapClause, OperandIx, PathSegment,
    PredIx, Schema, SeqKind, SeqShape, Violation,
};

use crate::check::ctx::{Ctx, MAX_WALK_DEPTH, WalkMode};
use crate::check::index::{RecordPlan, compile_pattern};
use crate::check::violation::{
    key_segment, located, mismatch, summarize_value, type_fail, type_mismatch,
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
        Schema::Coll { container, element } => match container {
            CollKind::Set => check_set(element, value, path, ctx, out),
            CollKind::FrozenSet => check_frozenset(element, value, path, ctx, out),
        },
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

/// A schema whose membership is decided by the value alone.
///
/// Named as a kind rather than answered as a test, so a caller that asks the
/// same schema about many values -- a homogeneous sequence -- reads the schema
/// once and tests many times.
///
/// This *is* a second statement of the scalar rules, beside the arms of
/// [`member`], and it is one deliberately: routing those arms through here cost
/// the call boundary 2.8% and the record walk 2.4%, because a match the
/// compiler can see through is worth more there than the sharing is. The two
/// are held together by a test rather than by construction --
/// `the_scalar_loop_and_the_walk_admit_the_same_values` asks both about every
/// scalar schema and every kind of value, so a rule changed in one place and
/// not the other fails rather than deciding one value two ways.
#[derive(Clone, Copy)]
enum Scalar {
    /// The top and the bottom, which admit everything and nothing.
    Everything,
    Nothing,
    NoneType,
    Bool,
    Int,
    Float,
    Str,
    Bytes,
}

/// The scalar kind of a schema, or `None` for one the walk must descend into.
#[inline]
fn scalar_of(schema: &Schema) -> Option<Scalar> {
    Some(match schema {
        Schema::Anything(_) => Scalar::Everything,
        Schema::Nothing => Scalar::Nothing,
        Schema::NoneType => Scalar::NoneType,
        Schema::Bool => Scalar::Bool,
        Schema::Int => Scalar::Int,
        Schema::Float => Scalar::Float,
        Schema::Str => Scalar::Str,
        Schema::Bytes => Scalar::Bytes,
        _ => return None,
    })
}

/// Whether a scalar kind admits a value.
#[inline]
fn scalar_admits(kind: Scalar, value: &Value<'_, '_>) -> bool {
    match kind {
        Scalar::Everything => true,
        Scalar::Nothing => false,
        Scalar::NoneType => value.is_none(),
        Scalar::Bool => value.is_bool(),
        // bool subclasses int, so True/False are ints: Bool is a subset of Int.
        Scalar::Int => value.is_int(),
        Scalar::Float => value.is_float(),
        Scalar::Str => value.is_str(),
        Scalar::Bytes => value.is_bytes(),
    }
}

/// The scalar kind every position of a sequence takes, where the walk of it
/// needs no path and reports no violation.
///
/// The shape a homogeneous list or tuple of a builtin type takes -- `list[int]`,
/// `tuple[str, ...]` -- and the one whose per-element cost is almost all
/// bookkeeping: a depth guard, a fatal-signal check and a dispatch around a
/// single type test. An explaining walk is not this shape, since it records the
/// position of each element it rejects.
#[inline]
fn homogeneous_scalar(prefix: &[Schema], tail: Option<&Schema>, ctx: Ctx<'_>) -> Option<Scalar> {
    if !prefix.is_empty() || ctx.mode.explains() {
        return None;
    }
    scalar_of(tail?)
}

/// A leaf decision: pass `ok` through, recording a type/value mismatch when it is
/// false in explain mode.
///
/// Inlined, and its recording half is not. Every scalar arm of the walk ends
/// here, so on a list of integers this is called once per element and does
/// nothing but return the bool it was handed: measured at twenty-two
/// instructions of call and return around a single test, which was a fifth of
/// the per-element cost. Inlining leaves the test where the answer already is.
/// The other half is a `Vec` push and a value summary, which only a failing
/// element in explain mode reaches -- it is marked cold so the branch predicts
/// the accepting path and the code sits away from it.
#[inline]
fn admit(
    ok: bool,
    schema: &Schema,
    value: &Value<'_, '_>,
    path: &[PathSegment],
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
    if !ok && ctx.mode.explains() {
        record_mismatch(schema, value, path, out);
    }
    ok
}

/// Record a leaf's type or value mismatch. The half of [`admit`] that allocates.
#[cold]
#[inline(never)]
fn record_mismatch(
    schema: &Schema,
    value: &Value<'_, '_>,
    path: &[PathSegment],
    out: &mut Vec<Violation>,
) {
    out.push(mismatch(schema, value, path));
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
    let (prefix, tail) = (&shape.prefix[..], shape.tail.as_deref());
    match (container, value) {
        (SeqKind::List, Value::Py(v)) => {
            let Ok(list) = v.cast::<PyList>() else {
                return type_fail(type_code, kind_word, value, path, ctx, out);
            };
            if !SeqArity::of(prefix.len(), tail).admits(list.len()) {
                return seq_length_fail(len_code, kind_word, prefix, tail, value, path, ctx, out);
            }
            // A list of one scalar kind -- `list[int]`, `list[str]` -- is the
            // shape whose per-element cost is almost all bookkeeping: the walk's
            // depth guard, its fatal-signal check and its dispatch, around a
            // single type test. None of the three is needed per element here: a
            // scalar cannot recurse, cannot run Python, and is the same schema at
            // every position, so they are paid once for the list.
            if let Some(kind) = homogeneous_scalar(prefix, tail, ctx) {
                let mut ok = true;
                let scan = scan_list(list, |_, item| {
                    ok &= scalar_admits(kind, &Value::Py(item));
                    if ok {
                        ControlFlow::Continue(())
                    } else {
                        ControlFlow::Break(())
                    }
                });
                return match scan {
                    Scan::Complete => ok,
                    Scan::Stopped => false,
                    Scan::Unreadable => mutated(value, path, ctx, out),
                };
            }
            let mut ok = true;
            let scan = scan_list(list, |i, item| {
                ok &= seq_element(prefix, tail, i, &Value::Py(item), path, ctx, out);
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
        (SeqKind::List, Value::Json(py, JsonValue::Array(items))) => {
            if !SeqArity::of(prefix.len(), tail).admits(items.len()) {
                return seq_length_fail(len_code, kind_word, prefix, tail, value, path, ctx, out);
            }
            // The same shape over a parsed JSON array.
            if let Some(kind) = homogeneous_scalar(prefix, tail, ctx) {
                return items
                    .iter()
                    .all(|item| scalar_admits(kind, &Value::Json(*py, item)));
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
            // The list arm's reasoning, for the immutable container: `tuple[int,
            // ...]` tests one scalar at every position, so the walk's
            // per-element bookkeeping is paid once for the tuple.
            //
            // Both arms borrow their elements rather than owning them. An owned
            // handle is a reference-count increment when it is made and a
            // decrement when it drops, and the walk keeps no element past the
            // test it runs on it: it reads the value and answers. A tuple is
            // frozen and is held for the whole walk by the caller's own handle,
            // so an element cannot be removed or freed underneath the borrow --
            // which is why `PyO3` offers this iterator for a tuple and for no
            // mutable container.
            if let Some(kind) = homogeneous_scalar(prefix, tail, ctx) {
                return tuple
                    .iter_borrowed()
                    .all(|item| scalar_admits(kind, &Value::Py(&item)));
            }
            let mut ok = true;
            for (i, item) in tuple.iter_borrowed().enumerate() {
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
///
/// Inlined into the one loop that calls it. It sits between that loop and
/// [`member`], so leaving it out of line put a second call frame on every
/// element of every sequence -- eleven instructions of entry and thirteen of
/// call, against the one index lookup and two mode tests it exists to do.
#[inline]
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

/// Visit a list's items by position, refusing when the list resizes underneath.
///
/// A sequence is walked by position against a length read once, so a list that
/// grows past that length hides its new items from the walk and one that shrinks
/// leaves the walk answering about items that are gone. Either way the reading
/// covers no state the list was ever in, and `is_valid` returned `True` for a
/// value that is not a member -- which the dict and set scans already refuse to
/// do, on the same argument, for the same reason.
///
/// The count is read once and re-read before each item and after the last, which
/// is [`scan_dict`]'s rule applied to positions instead of entries. A tuple needs
/// none of this: it cannot be resized, so its arm walks the iterator directly.
fn scan_list<'py>(
    list: &Bound<'py, PyList>,
    mut visit: impl FnMut(usize, &Bound<'py, PyAny>) -> ControlFlow<()>,
) -> Scan {
    with_critical_section(list.as_any(), || {
        let items = list.len();
        let mut iter = list.iter();
        let mut seen = 0;
        while seen < items {
            if list.len() != items {
                return Scan::Unreadable;
            }
            let Some(item) = iter.next() else {
                break;
            };
            let at = seen;
            seen += 1;
            if visit(at, &item).is_break() {
                return Scan::Stopped;
            }
        }
        if list.len() == items {
            Scan::Complete
        } else {
            Scan::Unreadable
        }
    })
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
    // A set of one scalar kind, as a sequence of one is: the element schema is
    // read once and each element tested against the kind, without the walk's
    // per-element depth guard, signal check and dispatch.
    let scalar = scalar_of(element);
    let mut ok = true;
    let scan = scan_set(container, ctx, |item| {
        let value = Value::Py(item);
        ok &= match scalar {
            Some(kind) => scalar_admits(kind, &value),
            None => member(element, &value, path, ctx, out),
        };
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
    // One pair of scratch buffers for every clause rather than a pair per call
    // into the walk. Neither is written on this path -- a fast walk reports
    // nothing and records no location -- but each is a value with a destructor,
    // and building and dropping four of them per key is work the answer does
    // not depend on.
    let (mut path, mut out) = (Vec::new(), Vec::new());
    defaults.iter().any(|clause| {
        member(&clause.key, key, &mut path, sub, &mut out)
            && member(&clause.value, val, &mut path, sub, &mut out)
    })
}

/// A closed record's membership, asked key by key rather than read entry by
/// entry, or `None` where that reading does not settle it.
///
/// A closed record declares every key the value may carry, so the value belongs
/// exactly when each declared key it holds matches and it holds nothing else --
/// and "nothing else" is a count, since a dict cannot repeat a key. Asking for
/// the declared keys costs one probe each with the key's own hash, where
/// scanning the value costs an iteration step, a decode of the key's bytes, a
/// second hash of those bytes and a comparison against the name they matched.
///
/// A key is resolved the way Python resolves one -- by the dict's own lookup --
/// rather than by decoding its bytes and matching those, so a key of a `str`
/// subclass with an `__eq__` of its own is found exactly where indexing the
/// dict would find it.
///
/// `None` means "ask the scan instead": the record is open, so an undeclared
/// key may still be covered by a clause; the plan has no interned key for a
/// field; or a probe raised, which is not an answer. A value that changes size
/// under the probes is not one of those: it is answered here, as the scan
/// answers it, because there is no reading of it left to fall back to.
fn keyed_map_asks_for_its_keys(
    fields: &[Field],
    defaults: &[MapClause],
    dict: &Bound<'_, PyDict>,
    ctx: Ctx<'_>,
    plan: &RecordPlan,
) -> Option<bool> {
    if !defaults.is_empty() || plan.keys.len() != fields.len() {
        return None;
    }
    let sub = fast(ctx);
    // One pair of scratch buffers for the record, as the scan takes: a fast
    // walk writes to neither.
    let (mut path, mut out) = (Vec::new(), Vec::new());
    with_critical_section(dict.as_any(), || {
        let entries = dict.len();
        let mut present = 0usize;
        for (position, field) in fields.iter().enumerate() {
            let key = plan.keys.get(position)?.bind(dict.py());
            match dict.get_item(key) {
                Ok(Some(value)) => {
                    present += 1;
                    if !member(&field.schema, &Value::Py(&value), &mut path, sub, &mut out) {
                        return Some(false);
                    }
                }
                // A key the value does not carry: the record still matches when
                // the field is optional.
                Ok(None) if !field.required => {}
                Ok(None) => return Some(false),
                // A failed probe is not an answer -- an unhashable key cannot be
                // in a dict, but a `__eq__` that raises can stop the lookup.
                Err(_) => return None,
            }
        }
        if dict.len() != entries {
            // The value changed while it was being read, so there is no reading
            // to answer from: not a member, exactly as the scan answers it, and
            // the explain pass names the mutation.
            return Some(false);
        }
        // Every key the value carries is one of the declared ones exactly when
        // the count of declared keys found equals the count it holds.
        Some(present == entries)
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
        if let Some(answered) = keyed_map_asks_for_its_keys(fields, defaults, dict, ctx, plan) {
            return answered;
        }
        keyed_map_scan(fields, defaults, dict, ctx, plan.required, |name| {
            plan.by_name.get(name).copied()
        })
    } else {
        let declared: FxHashMap<&str, usize> = fields
            .iter()
            .enumerate()
            .map(|(i, f)| (&*f.name, i))
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
    // Scratch buffers for the whole record, not one pair per field: a fast walk
    // writes to neither, and a fifty-field record was building and dropping a
    // hundred of them to answer one membership question.
    let (mut path, mut out) = (Vec::new(), Vec::new());
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
                if !member(&field.schema, &Value::Py(val), &mut path, sub, &mut out) {
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
    let (mut path, mut out) = (Vec::new(), Vec::new());
    // A closed record resolves the document's keys through the plan instead of
    // searching the document once per field. The search is quadratic in the
    // width -- a fifty-field record read a fifty-entry object fifty times -- and
    // the plan already holds the name-to-position map the resolution wants.
    if let Some(plan) = ctx
        .records
        .get(&(fields.as_ptr() as usize))
        .filter(|plan| defaults.is_empty() && plan.by_name.len() == fields.len())
    {
        // The document's value for each declared field, last occurrence winning
        // as `json.loads` does, gathered before any of them is checked: an
        // earlier duplicate that fails is not the entry the document means.
        let mut found: Vec<Option<&JsonValue<'_>>> = vec![None; fields.len()];
        for (key, value) in entries {
            match plan.by_name.get(key.as_ref()) {
                Some(&at) => match found.get_mut(at) {
                    Some(slot) => *slot = Some(value),
                    // The plan and the field list disagree about a position,
                    // which the filter above rules out; answer conservatively
                    // rather than indexing.
                    None => return false,
                },
                // A closed record has no clause to cover an undeclared key.
                None => return false,
            }
        }
        for (field, value) in fields.iter().zip(found) {
            match value {
                Some(value) => {
                    if !member(
                        &field.schema,
                        &Value::Json(py, value),
                        &mut path,
                        sub,
                        &mut out,
                    ) {
                        return false;
                    }
                }
                None if field.required => return false,
                None => {}
            }
        }
        return true;
    }
    for field in fields {
        match entries
            .iter()
            .rev()
            .find(|(key, _)| &*field.name == key.as_ref())
        {
            Some((_, val)) => {
                if !member(
                    &field.schema,
                    &Value::Json(py, val),
                    &mut path,
                    sub,
                    &mut out,
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
        None => fields.iter().any(|f| &*f.name == name),
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
    // The interned keys, in field order. Asking the dict by Rust text decodes a
    // fresh `PyString` and hashes it before the probe can start, once per field
    // per call; an interned key carries its hash. A schema absent from the index
    // falls back to its own text, so correctness never depends on the plan being
    // complete -- the two spellings name the same key.
    let interned = ctx.records.get(&(fields.as_ptr() as usize));
    let entries = dict.len();
    let mut present = 0usize;
    for (position, field) in fields.iter().enumerate() {
        let key = interned.and_then(|plan| plan.keys.get(position));
        let found = match key {
            Some(interned) => dict.get_item(interned.bind(dict.py())),
            None => dict.get_item(&*field.name),
        };
        match found {
            Ok(Some(item)) => {
                present += 1;
                path.push(PathSegment::Key(Arc::clone(&field.name)));
                member(&field.schema, &Value::Py(&item), path, ctx, out);
                path.pop();
            }
            Ok(None) if field.required => out.push(located(
                path,
                Arc::clone(&field.name),
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
    // A closed record that holds exactly the keys it declares has no undeclared
    // key to find, and the field loop above has already established it: a dict
    // cannot repeat a key, so finding as many declared keys as the value has
    // entries accounts for every one of them. The length is re-read because the
    // walk of a field's value runs Python, which can resize the dict -- and a
    // value that moved under the reading falls through to the scan, which is
    // where a mutation is reported.
    if defaults.is_empty() && present == entries && dict.len() == entries {
        return;
    }
    // Built here rather than above, because the scan is the only reader and a
    // record that answers by count never reaches it.
    let declared: FxHashSet<&str> = fields.iter().map(|field| &*field.name).collect();
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
            path.push(key_segment(key));
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
                Arc::from(key_text.as_str()),
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
            for member in members.iter() {
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
    if ctx.mode.explains() {
        return explain_union(members, value, path, ctx, out);
    }
    // A value is a member iff it matches at least one branch; decide that on the
    // fast path, where a discarded branch pays for no path or violation.
    let sub = fast(ctx);
    members
        .iter()
        .any(|m| member(m, value, &mut Vec::new(), sub, &mut Vec::new()))
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
/// recursion linear, and the walk that used to be thrown away is the one that
/// is kept.
fn explain_union(
    members: &[Schema],
    value: &Value<'_, '_>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
    // The *closest* branch -- the one that descended furthest into the value
    // before failing -- is reported, rather than every branch. "Furthest" is the
    // greatest path depth past the union's own location. Where no branch makes
    // progress (`int | str` against a float, say) a single union error stands
    // for all of them. Violations are aggregated regardless of fail_fast so the
    // deepest progress is visible; this runs only where a value is being
    // explained.
    let base_depth = path.len();
    let probe = Ctx {
        mode: WalkMode::Explain,
        ..ctx
    };
    let mut best: Option<(usize, Vec<Violation>)> = None;
    for (position, branch_schema) in members.iter().enumerate() {
        if position >= CLOSEST_BRANCH_PROBE_LIMIT {
            // Past the probe's width the branch is asked the cheap question
            // only: a union this wide reports the closest of the branches
            // already walked, and the rest merely decide membership.
            if member(
                branch_schema,
                value,
                &mut Vec::new(),
                fast(ctx),
                &mut Vec::new(),
            ) {
                return true;
            }
            continue;
        }
        let mut branch = Vec::new();
        if member(branch_schema, value, path, probe, &mut branch) {
            return true;
        }
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
            None => obj.getattr(&*field.name),
        };
        match attribute {
            Ok(attr) => {
                if ctx.mode.explains() {
                    path.push(PathSegment::Key(Arc::clone(&field.name)));
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
                        Arc::clone(&field.name),
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
mod interpreter;

#[cfg(test)]
mod label_tests;
