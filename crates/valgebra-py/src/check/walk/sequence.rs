//! The containers a value is read as a run of *elements*.
//!
//! A list, a tuple, a parsed array, a set and a frozenset differ in how their
//! elements are reached and agree on what each element must be. What they share
//! is the reading a record does not: an arity, a positional schema or a
//! repeated tail, a count taken once and compared again, and the snapshot a
//! list is read through where one test settles every element.

use std::ops::ControlFlow;

use jiter::JsonValue;
use pyo3::prelude::*;
use pyo3::sync::critical_section::with_critical_section;
use pyo3::types::{PyFrozenSet, PyIterator, PyList, PySet, PyTuple};
use valgebra_core::{ClassIx, PathSegment, Schema, SeqKind, SeqShape, Violation};

#[cfg(PyPy)]
use super::reads_its_length;
use super::scalar::{Scalar, admitted_quietly, homogeneous_scalar_union, scalar_union_admits};
use super::{
    Base, Frame, Scan, class_at, held_iter, homogeneous_scalar, is_exactly_a, is_fatal, member,
    mutated, reads_its_elements, record_fatal, record_if_fatal, scalar_admits, scalar_of, stop,
};
use crate::check::ctx::Ctx;
use crate::check::violation::{summarize_value, type_fail};
use crate::codes::{
    Code, FROZEN_SET_TYPE, LIST_LENGTH, LIST_TYPE, SET_TYPE, TUPLE_LENGTH, TUPLE_TYPE,
};
use crate::input::Value;

/// Membership for a sequence node: the value is a list or tuple whose elements
/// take the schema's shape — a fixed positional prefix then an optional repeated
/// tail. The elements are walked lazily against the shape the node holds, with
/// no automaton and no collection, identical in cost to a direct positional or
/// homogeneous check. JSON arrays are lists.
pub(super) fn check_seq(
    container: SeqKind,
    shape: &SeqShape,
    value: &Value<'_, '_>,
    frame: &mut Frame<'_, '_>,
) -> bool {
    let ctx = frame.ctx;
    let (kind_word, type_code, len_code) = match container {
        SeqKind::List => ("list", LIST_TYPE, LIST_LENGTH),
        SeqKind::Tuple => ("tuple", TUPLE_TYPE, TUPLE_LENGTH),
    };
    let (prefix, tail) = (&shape.prefix[..], shape.tail.as_deref());
    match (container, value) {
        (SeqKind::List, Value::Py(v)) => {
            let Ok(list) = v.cast::<PyList>() else {
                return type_fail(
                    type_code, kind_word, value, frame.path, frame.ctx, frame.out,
                );
            };
            if !SeqArity::of(prefix.len(), tail).admits(list.len()) {
                return seq_length_fail(len_code, kind_word, prefix, tail, value, frame);
            }
            if let Some((kind, schema)) = homogeneous_scalar(prefix, tail, ctx) {
                return scalar_list_matches(list, kind, schema, value, frame);
            }
            if let Some(element @ (Schema::Union(_) | Schema::Instance(_))) = tail
                && let Some(ok) = element_list_matches(list, prefix, element, value, frame)
            {
                return ok;
            }
            let mut ok = true;
            let scan = scan_list(list, |i, item| {
                ok &= seq_element(prefix, tail, i, &Value::Py(item), frame);
                if !ok && stop(ctx) {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            });
            match scan {
                Scan::Complete => ok,
                Scan::Stopped => false,
                Scan::Unreadable => mutated(value, frame),
            }
        }
        (SeqKind::List, Value::Json(py, JsonValue::Array(items))) => {
            if !SeqArity::of(prefix.len(), tail).admits(items.len()) {
                return seq_length_fail(len_code, kind_word, prefix, tail, value, frame);
            }
            json_array_matches(prefix, tail, *py, items, frame)
        }
        (SeqKind::Tuple, Value::Py(v)) => {
            let Ok(tuple) = v.cast::<PyTuple>() else {
                return type_fail(
                    type_code, kind_word, value, frame.path, frame.ctx, frame.out,
                );
            };
            tuple_matches(prefix, tail, tuple, value, len_code, kind_word, frame)
        }
        // A tuple is never a JSON value; a list needs a JSON array.
        _ => type_fail(
            type_code, kind_word, value, frame.path, frame.ctx, frame.out,
        ),
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

/// The element-by-element half of a parsed JSON array's walk, its arity already
/// admitted.
///
/// Its own function so the three containers `check_seq` reads share the arm
/// that dispatches and not the arm that walks: a list, a tuple and a document's
/// array differ in how their elements are *reached*, and agree on what each
/// element must be.
fn json_array_matches(
    prefix: &[Schema],
    tail: Option<&Schema>,
    py: Python<'_>,
    items: &[JsonValue<'_>],
    frame: &mut Frame<'_, '_>,
) -> bool {
    let ctx = frame.ctx;
    // The homogeneous shape over a parsed array: one scalar kind at every
    // position, and nothing per element but the test. A document is never
    // explained, so this loop records nothing.
    if let Some((kind, _)) = homogeneous_scalar(prefix, tail, ctx).filter(|_| !ctx.mode.explains())
    {
        return items
            .iter()
            .all(|item| scalar_admits(kind, &Value::Json(py, item)));
    }
    if let Some(Schema::Union(members)) = tail
        && let Some(members) =
            homogeneous_scalar_union(prefix, members, ctx).filter(|_| !ctx.mode.explains())
    {
        return items
            .iter()
            .all(|item| scalar_union_admits(members, &Value::Json(py, item)));
    }
    let mut ok = true;
    for (i, item) in items.iter().enumerate() {
        ok &= seq_element(prefix, tail, i, &Value::Json(py, item), frame);
        if !ok && stop(ctx) {
            return false;
        }
    }
    ok
}

/// Match one element at position `i`: the prefix schema at `i`, or the repeated
/// tail past the prefix. The index segment is pushed only in explain mode, and
/// only for an element [`admitted_quietly`] does not answer: an explaining walk
/// reaches here with the elements no sequence reading takes, and each was a
/// location pushed and popped and a dispatch around what can be one type test.
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
    frame: &mut Frame<'_, '_>,
) -> bool {
    let ctx = frame.ctx;
    let Some(schema) = prefix.get(i).or(tail) else {
        // Unreachable: the caller's length check guarantees `i` lands in the
        // prefix, or a repeated tail covers the overflow. Fold to non-member
        // rather than panic across the FFI boundary if that ever breaks.
        return false;
    };
    if ctx.mode.explains() {
        if admitted_quietly(schema, item, ctx) {
            return true;
        }
        frame.path.push(PathSegment::Index(i));
    }
    let ok = member(schema, item, frame);
    if ctx.mode.explains() {
        frame.path.pop();
    }
    ok
}

/// A sequence-length mismatch: terminal, since the positional match is then
/// meaningless. A tailless shape wants an exact length; a tailed one a minimum.
fn seq_length_fail(
    len_code: Code,
    kind_word: &str,
    prefix: &[Schema],
    tail: Option<&Schema>,
    value: &Value<'_, '_>,
    frame: &mut Frame<'_, '_>,
) -> bool {
    let ctx = frame.ctx;
    if ctx.mode.explains() {
        let expected = if tail.is_some() {
            format!("{kind_word} of length at least {}", prefix.len())
        } else {
            format!("{kind_word} of length {}", prefix.len())
        };
        frame.out.push(Violation {
            code: len_code.as_str(),
            path: frame.path.clone(),
            expected,
            value_summary: summarize_value(value, ctx),
        });
    }
    false
}

/// Membership for a list whose every element is one scalar kind, `schema`.
///
/// A list of one scalar kind -- `list[int]`, `list[str]` -- is the shape whose
/// per-element cost is almost all bookkeeping: the walk's depth guard, its
/// fatal-signal check and its dispatch, around a single type test. None of the
/// three is needed per element here: a scalar cannot recurse, cannot run
/// Python, and is the same schema at every position, so they are paid once for
/// the list. An explaining walk reads it the same way, and walks only the
/// elements that fail: see [`list_explained`].
///
/// Out of line, so the loop's code is its own: held inside [`check_seq`], its
/// register allocation follows every arm beside it, and `scripts/perf_gate.py
/// --binding` reads a change to the tuple arm as a change to `list[int]`. The
/// list pays one call.
///
/// **The kind is read once per list, not once per element.** Each builtin kind
/// has a loop of its own, its type test a constant inside it, so no element
/// pays the dispatch on `kind`. With the dispatch inside one shared loop, the
/// PGO wheel laid that loop out with one instruction more per element than
/// the loop read in place had, and `is_valid` on ten thousand integers took 6%
/// longer, while the instruction gate, which builds without a profile, read the
/// same change 24% cheaper.
#[expect(
    clippy::redundant_closure_for_method_calls,
    reason = "the loop takes a test over a `Value` of any lifetime, and the \
              method path names one lifetime; each closure is the method, \
              generic over the lifetime the loop hands it"
)]
#[inline(never)]
fn scalar_list_matches(
    list: &Bound<'_, PyList>,
    kind: Scalar,
    schema: &Schema,
    value: &Value<'_, '_>,
    frame: &mut Frame<'_, '_>,
) -> bool {
    if frame.ctx.mode.explains() {
        return list_explained(list, schema, |item| scalar_admits(kind, item), value, frame);
    }
    match kind {
        Scalar::Int => scalar_list_loop(list, |item| item.is_int(), value, frame),
        Scalar::Str => scalar_list_loop(list, |item| item.is_str(), value, frame),
        Scalar::Float => scalar_list_loop(list, |item| item.is_float(), value, frame),
        Scalar::Bool => scalar_list_loop(list, |item| item.is_bool(), value, frame),
        Scalar::Bytes => scalar_list_loop(list, |item| item.is_bytes(), value, frame),
        Scalar::NoneType => scalar_list_loop(list, |item| item.is_none(), value, frame),
        Scalar::Everything | Scalar::Nothing => {
            scalar_list_loop(list, |item| scalar_admits(kind, item), value, frame)
        }
    }
}

/// A list read in an explaining walk as its readers read it outside one: each
/// element is the test `admits` makes of it, and one that fails is walked
/// against `schema` at its own location, which records what the walk records of
/// it.
///
/// `validate` explains as it decides, so this is how it reads a `list[int]`
/// that belongs. The general scan asks the same of each element, through a
/// call and a read of the element's schema a position: a thousand integers
/// cost it 89 instructions each, against 22 for `is_valid`. An element that
/// passes records nothing -- a scalar records only a mismatch, and a union of
/// them returns at the first branch that matches -- and no element can raise
/// before the one that fails: a test runs no Python, and only a failing
/// element's summary can, which fails the list before a signal it raises is
/// read. The scan and its count are the general walk's, so a list that moves
/// reports the move as the general walk does.
///
/// **A list that belongs is read through the deciding walk's snapshot.** Where
/// a snapshot pays ([`snapshot_pays`]), an exact list is copied to a tuple and
/// its elements read borrowed, as [`scalar_list_loop`] reads them: an element
/// that passes records nothing, so a snapshot every element passes, over a
/// count that did not move, is the whole answer. Read in place, each element
/// was an owned handle, and on 3.12 `validate` took twice the time `is_valid`
/// took over a thousand integers, for fewer instructions. A list holding an
/// element that fails is read in place from its start instead: the failing
/// element's summary runs Python, which may move the list, and the in-place
/// scan reads it as the general walk does.
///
/// Out of line, beside the deciding loop rather than inside it: inlined into
/// [`scalar_list_matches`], it moved the PGO wheel's layout of that loop, and
/// `is_valid` on ten thousand integers took 5% longer.
#[inline(never)]
fn list_explained(
    list: &Bound<'_, PyList>,
    schema: &Schema,
    admits: impl Fn(&Value<'_, '_>) -> bool,
    value: &Value<'_, '_>,
    frame: &mut Frame<'_, '_>,
) -> bool {
    let ctx = frame.ctx;
    if let Some(answer) = admitted_through_snapshot(list, &admits, value, frame) {
        return answer;
    }
    let mut ok = true;
    let scan = scan_list(list, |at, item| {
        if admits(&Value::Py(item)) {
            return ControlFlow::Continue(());
        }
        ok &= element_explained(schema, at, &Value::Py(item), frame);
        if !ok && stop(ctx) {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    });
    match scan {
        Scan::Complete => ok,
        Scan::Stopped => false,
        Scan::Unreadable => mutated(value, frame),
    }
}

/// A list read through a snapshot where one pays, as [`scalar_list_loop`]
/// reads it: the answer where every element passes `admits` -- `true`, or the
/// move reported where the count did not hold -- and `false` where the copy
/// could not be made, a fatal signal recorded; `None` where no snapshot is
/// taken or an element does not pass, which the caller reads in place from the
/// start, as the general walk reads it.
///
/// Read so by the readers whose elements a test alone may not settle: a
/// failing element, or one only the walk can decide, may run Python that moves
/// the list, and the snapshot would answer about the list as it was.
fn admitted_through_snapshot(
    list: &Bound<'_, PyList>,
    admits: &impl Fn(&Value<'_, '_>) -> bool,
    value: &Value<'_, '_>,
    frame: &mut Frame<'_, '_>,
) -> Option<bool> {
    if !(snapshot_pays(list.len()) && list.is_exact_instance_of::<PyList>()) {
        return None;
    }
    match list.as_sequence().to_tuple() {
        Ok(snapshot) => {
            if !snapshot
                .iter_borrowed()
                .all(|item| admits(&Value::Py(&item)))
            {
                return None;
            }
            Some(if list.len() == snapshot.len() {
                true
            } else {
                mutated(value, frame)
            })
        }
        Err(err) => {
            record_fatal(err, frame.ctx);
            Some(false)
        }
    }
}

/// [`list_explained`] for a tuple, whose elements are read borrowed and cannot
/// move: `admits` is asked of each position, and one that fails is walked
/// against the schema [`seq_element`] would read there.
#[inline(never)]
fn tuple_explained(
    tuple: &Bound<'_, PyTuple>,
    prefix: &[Schema],
    tail: Option<&Schema>,
    admits: impl Fn(usize, &Value<'_, '_>) -> bool,
    frame: &mut Frame<'_, '_>,
) -> bool {
    let ctx = frame.ctx;
    let mut ok = true;
    for (at, item) in tuple.iter_borrowed().enumerate() {
        let item = Value::Py(&item);
        if admits(at, &item) {
            continue;
        }
        ok &= seq_element(prefix, tail, at, &item, frame);
        if !ok && stop(ctx) {
            return false;
        }
    }
    ok
}

/// Walk an element at its own location in an explaining walk: what
/// [`seq_element`] does with an element it cannot answer quietly.
fn element_explained(
    schema: &Schema,
    at: usize,
    item: &Value<'_, '_>,
    frame: &mut Frame<'_, '_>,
) -> bool {
    frame.path.push(PathSegment::Index(at));
    let ok = member(schema, item, frame);
    frame.path.pop();
    ok
}

/// Membership for a list whose every element is a union of scalars --
/// `list[int | None]` -- where [`homogeneous_scalar_union`] reads it so, and
/// `None` where it does not. An explaining walk reads it through
/// [`list_explained`], and walks `union` at an element that fails.
///
/// [`scalar_list_matches`]'s loop with a test per branch. The question is asked
/// here rather than in [`check_seq`], which pays only the test of the tail's
/// tag: asked there, it moved the register allocation of the whole arm and cost
/// a list nested twenty-five deep four percent.
#[inline(never)]
pub(super) fn scalar_union_list_matches(
    list: &Bound<'_, PyList>,
    prefix: &[Schema],
    union: &Schema,
    members: &[Schema],
    value: &Value<'_, '_>,
    frame: &mut Frame<'_, '_>,
) -> Option<bool> {
    let ctx = frame.ctx;
    let Some(members) = homogeneous_scalar_union(prefix, members, ctx) else {
        return literal_list_matches(list, prefix, union, members, value, frame);
    };
    let admits = |item: &Value<'_, '_>| scalar_union_admits(members, item);
    if ctx.mode.explains() {
        return Some(list_explained(list, union, admits, value, frame));
    }
    Some(scalar_list_loop(list, admits, value, frame))
}

/// Membership for a list whose element is a union or a class, read by the
/// reader for its kind, and `None` where that reader declines.
///
/// One call for both, behind the one test of the tail's tag [`check_seq`]
/// makes: a second test there moved the PGO wheel's layout of the general
/// scan beside it.
#[inline(never)]
pub(super) fn element_list_matches(
    list: &Bound<'_, PyList>,
    prefix: &[Schema],
    element: &Schema,
    value: &Value<'_, '_>,
    frame: &mut Frame<'_, '_>,
) -> Option<bool> {
    match element {
        Schema::Union(members) => {
            scalar_union_list_matches(list, prefix, element, members, value, frame)
        }
        Schema::Instance(index) => {
            instance_list_matches(list, prefix, element, *index, value, frame)
        }
        _ => None,
    }
}

/// Membership for a list whose every element is an instance of one class --
/// `list[datetime.date]`, a list of one enumeration -- where a level is free
/// under the list, and `None` elsewhere.
///
/// An element whose type is the class is an instance of it, read off the type
/// pointer as [`check_instance`](super::check_instance) reads it; any other is
/// walked, which asks `isinstance` and may run the class's
/// `__instancecheck__`. The general loop paid a call and a dispatch around the
/// pointer test, 130 instructions an element. A list is read in place unless a
/// snapshot whose every element is exactly the class settles it.
pub(super) fn instance_list_matches(
    list: &Bound<'_, PyList>,
    prefix: &[Schema],
    element: &Schema,
    index: ClassIx,
    value: &Value<'_, '_>,
    frame: &mut Frame<'_, '_>,
) -> Option<bool> {
    let ctx = frame.ctx;
    if !prefix.is_empty() || !ctx.room_to_descend() {
        return None;
    }
    let class = class_at(ctx, index, value.py())?;
    let exact = |item: &Value<'_, '_>| matches!(item, Value::Py(obj) if is_exactly_a(obj, class));
    if ctx.mode.explains() {
        return Some(list_explained(list, element, exact, value, frame));
    }
    if let Some(answer) = admitted_through_snapshot(list, &exact, value, frame) {
        return Some(answer);
    }
    let mut ok = true;
    let scan = scan_list(list, |_, item| {
        let item = Value::Py(item);
        ok &= exact(&item) || member(element, &item, frame);
        if !ok && stop(ctx) {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    });
    Some(match scan {
        Scan::Complete => ok,
        Scan::Stopped => false,
        Scan::Unreadable => mutated(value, frame),
    })
}

/// Membership for a list whose every element is a union of literals --
/// `list[Literal["a", "b", "c"]]` -- where the union has a table and a level is
/// free under the list, and `None` elsewhere.
///
/// Each element is the table's answer, found once for the list rather than
/// looked up by the union at every element: the general loop paid a call, a
/// dispatch and a lookup of the table around an answer the table gives alone,
/// 208 instructions an element against 30 for a `list[str]`. The table answers
/// exactly what the union's walk answers, wherever it answers -- an exact
/// `int` or `str` -- and needs the level the walk would take for the union;
/// an element it does not decide is walked, which may run Python, so a list
/// is read in place unless a snapshot the table admits entirely settles it.
pub(super) fn literal_list_matches(
    list: &Bound<'_, PyList>,
    prefix: &[Schema],
    union: &Schema,
    members: &[Schema],
    value: &Value<'_, '_>,
    frame: &mut Frame<'_, '_>,
) -> Option<bool> {
    let ctx = frame.ctx;
    if !prefix.is_empty() || !ctx.room_to_descend() {
        return None;
    }
    if !matches!(members.first(), Some(Schema::Literal(_))) {
        return None;
    }
    let plan = ctx.unions.get(&(members.as_ptr() as usize))?;
    let decided = |item: &Value<'_, '_>| plan.decide(item) == Some(true);
    if ctx.mode.explains() {
        return Some(list_explained(list, union, decided, value, frame));
    }
    if let Some(answer) = admitted_through_snapshot(list, &decided, value, frame) {
        return Some(answer);
    }
    let mut ok = true;
    let scan = scan_list(list, |_, item| {
        let item = Value::Py(item);
        ok &= match plan.decide(&item) {
            Some(answer) => answer,
            None => member(union, &item, frame),
        };
        if !ok && stop(ctx) {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    });
    Some(match scan {
        Scan::Complete => ok,
        Scan::Stopped => false,
        Scan::Unreadable => mutated(value, frame),
    })
}

/// The loop [`scalar_list_matches`] and [`scalar_union_list_matches`] share,
/// over the test each hands it.
#[inline]
fn scalar_list_loop(
    list: &Bound<'_, PyList>,
    admits: impl Fn(&Value<'_, '_>) -> bool,
    value: &Value<'_, '_>,
    frame: &mut Frame<'_, '_>,
) -> bool {
    let ctx = frame.ctx;
    // A snapshot of the list is a tuple, and a tuple's elements are
    // read borrowed. The reading it answers about is the list as it
    // was when the copy was taken, so the count is compared again
    // afterwards and a value that moved reports the move, exactly as
    // the in-place scan does. Only an exact list is copied: a
    // subclass's copy goes through its own `__iter__`, which need not
    // yield what it holds, and the walk reads what a list holds on
    // every interpreter. Copying an exact list runs no Python, so
    // the copy fails only where the tuple cannot be allocated, and
    // that `MemoryError` is a fatal signal: recorded, never an
    // answer.
    if snapshot_pays(list.len()) && list.is_exact_instance_of::<PyList>() {
        match list.as_sequence().to_tuple() {
            Ok(snapshot) => {
                let ok = snapshot
                    .iter_borrowed()
                    .all(|item| admits(&Value::Py(&item)));
                if !ok {
                    return false;
                }
                return if list.len() == snapshot.len() {
                    true
                } else {
                    mutated(value, frame)
                };
            }
            Err(err) => {
                record_fatal(err, ctx);
                return false;
            }
        }
    }
    let mut ok = true;
    let scan = scan_list(list, |_, item| {
        ok &= admits(&Value::Py(item));
        if ok {
            ControlFlow::Continue(())
        } else {
            ControlFlow::Break(())
        }
    });
    match scan {
        Scan::Complete => ok,
        Scan::Stopped => false,
        Scan::Unreadable => mutated(value, frame),
    }
}

/// Membership for a tuple, over the elements the value holds.
fn tuple_matches(
    prefix: &[Schema],
    tail: Option<&Schema>,
    tuple: &Bound<'_, PyTuple>,
    value: &Value<'_, '_>,
    len_code: Code,
    kind_word: &'static str,
    frame: &mut Frame<'_, '_>,
) -> bool {
    let ctx = frame.ctx;
    // A tuple the C accessors read the storage of is read where it lies; any
    // other is copied. `CPython`'s accessors read the storage whatever a
    // subclass overrides, so there every tuple is read where it lies and its
    // type is asked nothing. See `reads_where_it_lies` and `storage_of`.
    #[cfg(PyPy)]
    let copied;
    #[cfg(PyPy)]
    let tuple = if reads_where_it_lies(tuple, ctx) {
        tuple
    } else {
        let Some(storage) = storage_of(tuple, ctx) else {
            return mutated(value, frame);
        };
        copied = storage;
        &copied
    };
    if !SeqArity::of(prefix.len(), tail).admits(tuple.len()) {
        return seq_length_fail(len_code, kind_word, prefix, tail, value, frame);
    }
    // The list arm's reasoning, for the immutable container: `tuple[int, ...]`
    // tests one scalar at every position, so the walk's per-element bookkeeping
    // is paid once for the tuple.
    //
    // Both arms borrow their elements rather than owning them. An owned handle
    // is a reference-count increment when it is made and a decrement when it
    // drops, and the walk keeps no element past the test it runs on it: it
    // reads the value and answers. A tuple is frozen and is held for the whole
    // walk by the caller's own handle, so an element cannot be removed or freed
    // underneath the borrow -- which is why `PyO3` offers this iterator for a
    // tuple and for no mutable container.
    // An explaining walk reads a homogeneous tuple through the positions reader
    // below, which walks a position that fails.
    if let Some((kind, _)) = homogeneous_scalar(prefix, tail, ctx).filter(|_| !ctx.mode.explains())
    {
        return tuple
            .iter_borrowed()
            .all(|item| scalar_admits(kind, &Value::Py(&item)));
    }
    if let Some(union @ Schema::Union(members)) = tail
        && let Some(ok) = scalar_union_tuple_matches(tuple, prefix, union, members, frame)
    {
        return ok;
    }
    if let Some(ok) = scalar_positions_tuple_matches(tuple, prefix, tail, frame) {
        return ok;
    }
    let mut ok = true;
    for (i, item) in tuple.iter_borrowed().enumerate() {
        ok &= seq_element(prefix, tail, i, &Value::Py(&item), frame);
        if !ok && stop(ctx) {
            return false;
        }
    }
    ok
}

/// Membership for a tuple whose every element is a union of scalars --
/// `tuple[int | None, ...]` -- where [`homogeneous_scalar_union`] reads it so,
/// and `None` where it does not: [`scalar_union_list_matches`] for the frozen
/// container, and out of line for the same reason.
#[inline(never)]
pub(super) fn scalar_union_tuple_matches(
    tuple: &Bound<'_, PyTuple>,
    prefix: &[Schema],
    union: &Schema,
    members: &[Schema],
    frame: &mut Frame<'_, '_>,
) -> Option<bool> {
    let members = homogeneous_scalar_union(prefix, members, frame.ctx)?;
    if frame.ctx.mode.explains() {
        let admits = |_: usize, item: &Value<'_, '_>| scalar_union_admits(members, item);
        return Some(tuple_explained(tuple, &[], Some(union), admits, frame));
    }
    Some(
        tuple
            .iter_borrowed()
            .all(|item| scalar_union_admits(members, &Value::Py(&item))),
    )
}

/// Membership for a tuple whose every position is a scalar --
/// `tuple[int, str, float]`, a `NamedTuple` of builtin fields -- where the walk
/// of it needs no path, and `None` elsewhere.
///
/// The fixed-arity twin of the homogeneous reading: each position is one type
/// test, and the walk around it -- a level, the fatal-signal flag, the dispatch
/// -- is paid once for the tuple rather than once a position. The arity is
/// already admitted, so a position past the prefix is the tail's, and a tail
/// that is not a scalar declines, as a prefix position that is not one does.
/// The level is read as [`homogeneous_scalar`] reads it, and for the same
/// reason: the walk would refuse each element where none is free.
///
/// An explaining walk reads it through [`tuple_explained`], and walks a
/// position that fails.
///
/// Out of line, and asked of every tuple the two readings above decline: a
/// tuple holding a record pays a call and a type test of its first position.
#[inline(never)]
pub(super) fn scalar_positions_tuple_matches(
    tuple: &Bound<'_, PyTuple>,
    prefix: &[Schema],
    tail: Option<&Schema>,
    frame: &mut Frame<'_, '_>,
) -> Option<bool> {
    let ctx = frame.ctx;
    if !ctx.room_to_descend() {
        return None;
    }
    if !prefix.iter().all(|schema| scalar_of(schema).is_some()) {
        return None;
    }
    let repeated = match tail {
        Some(schema) => Some(scalar_of(schema)?),
        None => None,
    };
    let admits = |at: usize, item: &Value<'_, '_>| {
        prefix
            .get(at)
            .map_or(repeated, scalar_of)
            .is_some_and(|kind| scalar_admits(kind, item))
    };
    if ctx.mode.explains() {
        return Some(tuple_explained(tuple, prefix, tail, admits, frame));
    }
    Some(
        tuple
            .iter_borrowed()
            .enumerate()
            .all(|(at, item)| admits(at, &Value::Py(&item))),
    )
}

/// Whether `PyPy`'s C accessors read this tuple's storage, so the walk can read
/// it where it lies.
///
/// An exact tuple overrides nothing, and a subclass that inherits
/// `tuple.__len__` -- every `NamedTuple` does -- reports its storage's length.
/// It must inherit `tuple.__iter__` too: `cpyext` fills a subclass's C-level
/// items from its own `__iter__`, so one that overrides it hands the borrowed
/// walk items that are not its storage, fewer than the length beside them, and
/// the walk read past their end -- a subclass whose `__iter__` yields one item
/// over two took `validate` down on `PyPy` 3.11.
///
/// `PyPy` alone. `CPython`'s `PyTuple_GET_SIZE` and `PyTuple_GET_ITEM` read the
/// storage whatever the type overrides, so the question has one answer there,
/// and asking it cost a `NamedTuple` an attribute read of its type per value --
/// a quarter of walking a list of them.
#[cfg(PyPy)]
fn reads_where_it_lies(tuple: &Bound<'_, PyTuple>, ctx: Ctx<'_>) -> bool {
    tuple.is_exact_instance_of::<PyTuple>()
        || (reads_its_elements(tuple, Base::Tuple, ctx)
            && reads_its_length(tuple, Base::Tuple, ctx))
}

/// What a tuple *subclass* holds, as a tuple of its own.
///
/// A tuple's length and its elements have to come from the same place, and for
/// a subclass they can come from two. `PyTuple_Size` reads the storage on
/// `CPython`; on an interpreter that implements it through the object's own
/// `__len__` -- `PyPy`'s `cpyext` does -- a subclass that overrides `__len__`
/// answers whatever it likes, and the borrowed iterator reads that many slots
/// out of a storage holding fewer. That is a read past the end of an
/// allocation, and it takes the process with it: a `tuple` subclass returning
/// ten from `__len__` over one element segfaults `PyPy` 3.11 on
/// `tuple[int, ...]`.
///
/// So a subclass is copied through the base type's own iterator, [`held_iter`],
/// and the walk reads the copy. Not by position over the base's own length:
/// `PyTuple_GetItem` reads the same C-level items the borrowed walk does, which
/// `cpyext` filled from an overridden `__iter__`. The elements are the ones the
/// value holds, which is what the sequence walk promises and what `CPython`
/// reads in place. An exact tuple overrides nothing and is read where it lies:
/// this path costs the common case nothing.
///
/// `None` where the copy cannot be made, which the caller reports as a value it
/// could not read rather than as a membership answer. A fatal signal the copy
/// raises is recorded, and one the type probe recorded before it stops the copy,
/// so the entry point re-raises the signal in place of that report.
///
/// Cold and out of line: an exact tuple and a `NamedTuple` are read where they
/// lie, and only a subclass [`reads_where_it_lies`] refuses is copied.
#[cfg(PyPy)]
#[cold]
#[inline(never)]
fn storage_of<'py>(tuple: &Bound<'py, PyTuple>, ctx: Ctx<'_>) -> Option<Bound<'py, PyTuple>> {
    if ctx.fatal_seen.get() {
        return None;
    }
    let copy = || -> PyResult<Bound<'py, PyTuple>> {
        let items = held_iter(tuple, Base::Tuple)?
            .try_iter()?
            .collect::<PyResult<Vec<_>>>()?;
        PyTuple::new(tuple.py(), items)
    };
    copy()
        .map_err(|err| record_if_fatal(err, tuple.py(), ctx))
        .ok()
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
/// is [`scan_dict`](super::record::scan_dict)'s rule applied to positions
/// instead of entries. A tuple needs none of this: it cannot be resized, so its
/// arm walks the iterator directly over the storage [`tuple_matches`] hands it.
pub(super) fn scan_list<'py>(
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

/// Visit a set's or frozenset's elements, reporting rather than panicking when
/// the container changes underneath the scan.
///
/// A set is walked through its own Python iterator, which *raises* on mutation
/// where `PyO3`'s wrapper unwraps that error into a panic. Taking the iterator
/// directly keeps the raise, which is an outcome the walk already knows how to
/// carry: a fatal signal propagates, and anything else means the container did
/// not answer.
pub(super) fn scan_set<'py>(
    set: &Bound<'py, PyAny>,
    ctx: Ctx<'_>,
    mut visit: impl FnMut(&Bound<'py, PyAny>) -> ControlFlow<()>,
) -> Scan {
    with_critical_section(set, || {
        let iter = match storage_iter(set, ctx) {
            Ok(iter) => iter,
            Err(err) => {
                record_if_fatal(err, set.py(), ctx);
                return Scan::Unreadable;
            }
        };
        // Reading the type may have raised a fatal signal, which ends the walk.
        if ctx.fatal_seen.get() {
            return Scan::Unreadable;
        }
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

/// The members a set-like value *holds*, whatever its type says it yields.
///
/// A set is read through an iterator rather than by position, so this is where
/// its storage is reached. A subclass overriding `__iter__` yields whatever it
/// likes, and a walk believing it decides membership of a set that is not the
/// value: `set[int]` admitted a subclass holding a `str` because its iterator
/// answered with integers.
///
/// Exactness first, then the slot, which is the rule
/// [`stored_len`](super::scalar::stored_len) applies to a length: an exact set
/// overrides nothing, and a subclass that inherits the slot is read where it
/// lies. The iterator the base returns is the builtin one, so mutation during
/// the scan still raises where the caller below expects it to.
fn storage_iter<'py>(set: &Bound<'py, PyAny>, ctx: Ctx<'_>) -> PyResult<Bound<'py, PyIterator>> {
    for held in &HELD {
        if (held.is_exact)(set) {
            return set.try_iter();
        }
        if (held.is_kind)(set) {
            return if reads_its_elements(set, held.base, ctx) {
                set.try_iter()
            } else {
                held_iter(set, held.base)?.try_iter()
            };
        }
    }
    set.try_iter()
}

/// A set-like kind, its exact test and its base, in the order `storage_iter`
/// asks them. The shape [`stored_len`](super::scalar::stored_len) reads a
/// length through, one slot over: a frozenset is not a set, so each is asked
/// for itself.
struct Held {
    is_exact: fn(&Bound<'_, PyAny>) -> bool,
    is_kind: fn(&Bound<'_, PyAny>) -> bool,
    base: Base,
}

const HELD: [Held; 2] = [
    Held {
        is_exact: |value| value.is_exact_instance_of::<PySet>(),
        is_kind: |value| value.is_instance_of::<PySet>(),
        base: Base::Set,
    },
    Held {
        is_exact: |value| value.is_exact_instance_of::<PyFrozenSet>(),
        is_kind: |value| value.is_instance_of::<PyFrozenSet>(),
        base: Base::FrozenSet,
    },
];

/// A set-like container the walk reads: what its type failure reports, and the
/// test that recognises it.
struct Collection {
    code: Code,
    word: &'static str,
    is_kind: fn(&Bound<'_, PyAny>) -> bool,
}

const SET: Collection = Collection {
    code: SET_TYPE,
    word: "set",
    is_kind: |value| value.is_instance_of::<PySet>(),
};

const FROZEN_SET: Collection = Collection {
    code: FROZEN_SET_TYPE,
    word: "frozenset",
    is_kind: |value| value.is_instance_of::<PyFrozenSet>(),
};

/// A set whose every element matches `element`. Set order is not meaningful, so
/// element failures carry no index segment. JSON has no sets.
pub(super) fn check_set(
    element: &Schema,
    value: &Value<'_, '_>,
    frame: &mut Frame<'_, '_>,
) -> bool {
    check_elements(&SET, element, value, frame)
}

/// A frozenset whose every element matches `element`. JSON has no frozensets.
pub(super) fn check_frozenset(
    element: &Schema,
    value: &Value<'_, '_>,
    frame: &mut Frame<'_, '_>,
) -> bool {
    check_elements(&FROZEN_SET, element, value, frame)
}

/// Membership for either set-like container: the value is of the container's
/// kind and every element belongs to `element`. One rule for both, because the
/// two differ only in the type they admit and the code they report.
fn check_elements(
    collection: &Collection,
    element: &Schema,
    value: &Value<'_, '_>,
    frame: &mut Frame<'_, '_>,
) -> bool {
    let ctx = frame.ctx;
    let Value::Py(container) = value else {
        return type_fail(
            collection.code,
            collection.word,
            value,
            frame.path,
            frame.ctx,
            frame.out,
        );
    };
    if !(collection.is_kind)(container) {
        return type_fail(
            collection.code,
            collection.word,
            value,
            frame.path,
            frame.ctx,
            frame.out,
        );
    }
    if ctx.mode.explains() {
        return explain_elements(element, container, value, frame);
    }
    // A set of one scalar kind, as a sequence of one is: the element schema is
    // read once and each element tested against the kind, without the walk's
    // per-element signal check and dispatch. The level every element sits at is
    // taken once for the loop rather than skipped, so this answers what the
    // explaining walk beside it answers at the depth bound.
    let scalar = scalar_of(element).filter(|_| ctx.room_to_descend());
    let mut ok = true;
    let scan = scan_set(container, ctx, |item| {
        let value = Value::Py(item);
        ok &= match scalar {
            Some(kind) => scalar_admits(kind, &value),
            None => member(element, &value, frame),
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
        Scan::Unreadable => mutated(value, frame),
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
/// which costs a full scan of a set that is already failing. An element the
/// walk would admit reports nothing, so one [`admitted_quietly`] answers is
/// passed without a probe of its own: `validate` over a `set[str]` that belongs
/// read every element that way, at three quarters again what `is_valid` pays.
fn explain_elements(
    element: &Schema,
    container: &Bound<'_, PyAny>,
    value: &Value<'_, '_>,
    frame: &mut Frame<'_, '_>,
) -> bool {
    let ctx = frame.ctx;
    let where_it_is = &mut *frame.path;
    let mut failures: Vec<(String, Vec<Violation>)> = Vec::new();
    let scan = scan_set(container, ctx, |item| {
        if admitted_quietly(element, &Value::Py(item), ctx) {
            return ControlFlow::Continue(());
        }
        let mut reported = Vec::new();
        let held = {
            let mut probe = Frame::new(&mut *where_it_is, &mut reported, ctx);
            member(element, &Value::Py(item), &mut probe)
        };
        if !held {
            let key = reported
                .first()
                .map(|violation| format!("{} {}", violation.value_summary, violation.code))
                .unwrap_or_default();
            failures.push((key, reported));
        }
        ControlFlow::Continue(())
    });
    if matches!(scan, Scan::Unreadable) {
        return mutated(value, frame);
    }
    let ok = failures.is_empty();
    failures.sort_by(|left, right| left.0.cmp(&right.0));
    let reported = if ctx.mode.stops_at_first() {
        1
    } else {
        failures.len()
    };
    for (_, group) in failures.into_iter().take(reported) {
        frame.out.extend(group);
    }
    ok
}

/// The narrowest list a snapshot pays for, and the widest.
///
/// A list one test settles element by element is read through a snapshot of it
/// (see [`snapshot_pays`]), and the two ends of that band are where the
/// snapshot stops paying. Below the first, its fixed cost -- one call, one
/// allocation -- outweighs what it saves: sixteen elements is where the two
/// meet, and a four-element list reads thirteen percent dearer through a
/// snapshot on `CPython` 3.12. Above the second, the copy is large enough that
/// walking it costs more cache than the reference counts it avoids: measured on
/// one box, a snapshot reads a hundred thousand elements at 1.68 ns each
/// against 4.57 in place, two hundred thousand at 1.73 against 4.74, four
/// hundred thousand at 3.88 against 4.68, and six hundred thousand at 5.76
/// against 4.61 -- so the crossing is between four and six hundred thousand,
/// and the cap sits below it with margin, at two mebibytes of transient.
///
/// Neither end changes an answer: both sides of each read the same elements and
/// report the same membership. They are where one reading of a value stops
/// being cheaper than another, which is why the walk holds them rather than the
/// frontend refusing anything.
const SNAPSHOT_MIN_ELEMENTS: usize = 16;

/// The widest list a snapshot pays for; see [`SNAPSHOT_MIN_ELEMENTS`].
const SNAPSHOT_MAX_ELEMENTS: usize = 262_144;

/// Whether a list of `len` elements is cheaper read through a snapshot than in
/// place, on the interpreter this extension is built against.
///
/// Reading an element out of a list hands back an *owned* handle: a reference
/// count written when it is made and again when it drops, on an object the walk
/// only type-tests. Copying the list into a tuple pays the same two writes in
/// two tight loops inside the interpreter, and the tuple is frozen, so its
/// elements are read borrowed and the walk pays neither.
///
/// Which is cheaper is a property of the interpreter. `CPython` 3.14 makes the
/// pair cheap enough that the copy is pure cost -- a ten-thousand element list
/// reads 1.44 ns per element in place there and 1.65 through a snapshot -- so
/// it reads in place. Below 3.14 the pair dominates: 4.77 against 1.66. The
/// free-threaded build pays a lock per element on top of the pair, and the copy
/// takes one lock for the whole list: 8.90 against 2.22, and it pays at every
/// width, so the lower end of the band does not apply there.
const fn snapshot_pays(len: usize) -> bool {
    if cfg!(Py_GIL_DISABLED) {
        len <= SNAPSHOT_MAX_ELEMENTS
    } else if cfg!(Py_3_14) {
        false
    } else {
        len >= SNAPSHOT_MIN_ELEMENTS && len <= SNAPSHOT_MAX_ELEMENTS
    }
}
