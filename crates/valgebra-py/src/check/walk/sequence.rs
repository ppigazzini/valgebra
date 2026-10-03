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
use valgebra_core::{
    ClassIx, Constraint, Field, PathSegment, Schema, SeqKind, SeqShape, Violation,
};

#[cfg(PyPy)]
use super::reads_its_length;
use super::record::check_attr_record;
use super::scalar::{
    Scalar, admitted_quietly, check_refine, homogeneous_scalar_union, scalar_union_admits,
};
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
            // Every tag from a union to a refinement, a complement and an
            // attribute record among them, though the reader declines both:
            // asked as that run of tags, the test costs a list nested
            // twenty-five deep 0.03% more instructions, and asked as only the
            // tags the reader takes, 1.25%.
            if let Some(
                element @ (Schema::Union(_)
                | Schema::Intersection(_)
                | Schema::Complement(_)
                | Schema::Instance(_)
                | Schema::AttrRecord { .. }
                | Schema::Refine { .. }
                | Schema::Seq {
                    container: SeqKind::Tuple,
                    ..
                }),
            ) = tail
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
/// same change 24% cheaper. The explaining walk reads its list with a test per
/// kind for the same reason: a match on `kind` at every element is three
/// instructions of `validate`'s per element on 3.14, where the list is read in
/// place.
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
        return match kind {
            Scalar::Int => list_explained(list, schema, |item| item.is_int(), value, frame),
            Scalar::Str => list_explained(list, schema, |item| item.is_str(), value, frame),
            Scalar::Float => list_explained(list, schema, |item| item.is_float(), value, frame),
            Scalar::Bool => list_explained(list, schema, |item| item.is_bool(), value, frame),
            Scalar::Bytes => list_explained(list, schema, |item| item.is_bytes(), value, frame),
            Scalar::NoneType => list_explained(list, schema, |item| item.is_none(), value, frame),
            Scalar::Everything | Scalar::Nothing => {
                list_explained(list, schema, |item| scalar_admits(kind, item), value, frame)
            }
        };
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
    if let [Schema::NoneType, kind] = members
        && let Some(answer) = nullable_list_matches(list, union, kind, value, frame)
    {
        return Some(answer);
    }
    let admits = |item: &Value<'_, '_>| scalar_union_admits(members, item);
    if ctx.mode.explains() {
        return Some(list_explained(list, union, admits, value, frame));
    }
    Some(scalar_list_loop(list, admits, value, frame))
}

/// Membership for a list whose every element is one scalar kind or `None` --
/// `list[int | None]`, `list[Optional[str]]` -- read with a loop for that kind,
/// and `None` for a kind it has no loop for.
///
/// [`scalar_list_matches`] for the union a scalar and `None` make: each kind
/// has a loop of its own, both tests constants inside it, where the loop over
/// the union's branches matched each branch's kind at every element, read the
/// branch's schema again to do it, and asked the kind it had already ruled out
/// of every `None`. Reached behind [`homogeneous_scalar_union`], which holds
/// the two levels the walk would take, as the union's other readings are.
#[inline(never)]
pub(super) fn nullable_list_matches(
    list: &Bound<'_, PyList>,
    union: &Schema,
    kind: &Schema,
    value: &Value<'_, '_>,
    frame: &mut Frame<'_, '_>,
) -> Option<bool> {
    Some(match kind {
        Schema::Int => nullable_list_read(
            list,
            union,
            |item| item.is_none() || item.is_int(),
            value,
            frame,
        ),
        Schema::Str => nullable_list_read(
            list,
            union,
            |item| item.is_none() || item.is_str(),
            value,
            frame,
        ),
        Schema::Float => nullable_list_read(
            list,
            union,
            |item| item.is_none() || item.is_float(),
            value,
            frame,
        ),
        Schema::Bool => nullable_list_read(
            list,
            union,
            |item| item.is_none() || item.is_bool(),
            value,
            frame,
        ),
        Schema::Bytes => nullable_list_read(
            list,
            union,
            |item| item.is_none() || item.is_bytes(),
            value,
            frame,
        ),
        _ => return None,
    })
}

/// One kind's loop of [`nullable_list_matches`], in the walk's mode: the
/// deciding loop, or the explaining reader, which walks `union` at an element
/// `admits` refuses. One test for both, so the two modes cannot read a kind
/// differently.
#[inline]
fn nullable_list_read(
    list: &Bound<'_, PyList>,
    union: &Schema,
    admits: impl Fn(&Value<'_, '_>) -> bool,
    value: &Value<'_, '_>,
    frame: &mut Frame<'_, '_>,
) -> bool {
    if frame.ctx.mode.explains() {
        list_explained(list, union, admits, value, frame)
    } else {
        scalar_list_loop(list, admits, value, frame)
    }
}

/// Membership for a list whose element is a union, a meet, a class, a
/// refinement or a tuple, read by the reader for its kind, and `None` where
/// that reader declines.
///
/// One call for every kind, behind the one test of the tail's tag [`check_seq`]
/// makes: a second test there moved the PGO wheel's layout of the general
/// scan beside it.
///
/// **Not marked cold, though it is called once per list.** Without a profile,
/// the compiler tests first the tags a case covers most of, and it reads a
/// union and a meet side by side as a range of two, which it tests before a
/// nested list's tag: four instructions more for every list holding lists,
/// 1.86% of `--binding-deep`. Marked cold, the plain build tests the nested
/// list's tag first, and the PGO wheel the release ships lays the list arm out
/// worse: a list of lists costs it 4.4% more instructions, and lists of refined
/// values 1.6% to 3.5%. The wheel is what a caller runs, so the plain build
/// pays the 1.86%.
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
        Schema::Intersection(members) => match &members[..] {
            [Schema::Instance(index), Schema::AttrRecord { fields }] => {
                class_record_list_matches(list, prefix, element, *index, fields, value, frame)
            }
            _ => named_tuple_list_matches(list, prefix, element, members, value, frame),
        },
        Schema::Instance(index) => {
            instance_list_matches(list, prefix, element, *index, value, frame)
        }
        Schema::Refine { base, constraints } => {
            refined_list_matches(list, prefix, base, constraints, value, frame)
        }
        Schema::Seq {
            container: SeqKind::Tuple,
            shape,
        } => tuple_list_matches(list, prefix, element, shape, value, frame),
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

/// Membership for a list whose every element is a refinement --
/// `list[Annotated[int, Ge(0)]]`, `list[Annotated[str, MinLen(1)]]` -- where a
/// level is free under the list, and `None` elsewhere.
///
/// Each element is the refinement's own check, called directly, and the level
/// the walk opens for an element is opened once and held for the list: what
/// the walk spent around the check -- a call into [`member`], its depth guard
/// and its dispatch on the element's tag -- is spent once. The check makes the
/// rest of what [`member`] makes: a fatal signal recorded at one element fails
/// the next at its base, which [`member`] refuses at once. No test settles an
/// element apart from that check, as a type test settles one for the other
/// readers: a constraint may run Python, through an order bound whose operand
/// is a class of its own or through a predicate, so each element is checked
/// once, in the walk's own mode, and an explaining walk names it at its index
/// as the walk does.
#[inline(never)]
fn refined_list_matches(
    list: &Bound<'_, PyList>,
    prefix: &[Schema],
    base: &Schema,
    constraints: &[Constraint],
    value: &Value<'_, '_>,
    frame: &mut Frame<'_, '_>,
) -> Option<bool> {
    let ctx = frame.ctx;
    if !prefix.is_empty() {
        return None;
    }
    let _element = ctx.descend()?;
    let explains = ctx.mode.explains();
    let mut ok = true;
    let scan = scan_list(list, |at, item| {
        if explains {
            frame.path.push(PathSegment::Index(at));
        }
        ok &= check_refine(base, constraints, &Value::Py(item), frame);
        if explains {
            frame.path.pop();
        }
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

/// Membership for a list whose every element is a tuple of scalar positions --
/// `list[tuple[int, str]]` -- where the two levels below the list are free, and
/// `None` elsewhere.
///
/// An exact tuple of the arity whose every position passes its kind's test is
/// a member, as [`scalar_positions_tuple_matches`] reads it; any other element
/// is walked, which reads a tuple subclass through its storage and reports
/// what does not fit. The general loop paid a call into `member` and the
/// tuple's arm of [`check_seq`] around the same tests, and the positions reader
/// read each position's schema again for every tuple: 285 instructions an
/// element on 3.14, against 110 here. A list is read in place unless a snapshot
/// every element of which fits settles it, as [`instance_list_matches`] reads
/// its own, whose body this one repeats rather than shares: drawn into one
/// generic reader, the class list's elements cost the PGO wheel two
/// instructions more each, 8% of a thousand `date`s. Out of line for the same
/// reason: inlined into [`element_list_matches`] beside the class reader, it
/// cost that reader one instruction an element.
#[inline(never)]
pub(super) fn tuple_list_matches(
    list: &Bound<'_, PyList>,
    prefix: &[Schema],
    element: &Schema,
    shape: &SeqShape,
    value: &Value<'_, '_>,
    frame: &mut Frame<'_, '_>,
) -> Option<bool> {
    let ctx = frame.ctx;
    let positions = scalar_tuple_positions(prefix, shape, ctx)?;
    let fits = |item: &Value<'_, '_>| {
        matches!(item, Value::Py(obj) if obj.cast_exact::<PyTuple>().is_ok_and(|tuple| {
            tuple.len() == positions.len()
                && tuple.iter_borrowed().zip(positions).all(|(item, position)| {
                    scalar_of(position).is_some_and(|kind| scalar_admits(kind, &Value::Py(&item)))
                })
        }))
    };
    if ctx.mode.explains() {
        return Some(list_explained(list, element, fits, value, frame));
    }
    if let Some(answer) = admitted_through_snapshot(list, &fits, value, frame) {
        return Some(answer);
    }
    let mut ok = true;
    let scan = scan_list(list, |_, item| {
        let item = Value::Py(item);
        ok &= fits(&item) || member(element, &item, frame);
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

/// The positions of the fixed tuple of scalars every element of a list is, and
/// `None` for any other element.
///
/// A position sits a level below its tuple, and the tuple a level below the
/// list, so two levels must be free: the tuple's is held while the position's
/// is asked for, which is what the walk does with each element, and both are
/// refused together, as [`homogeneous_scalar_union`] refuses a union's.
fn scalar_tuple_positions<'s>(
    prefix: &[Schema],
    shape: &'s SeqShape,
    ctx: Ctx<'_>,
) -> Option<&'s [Schema]> {
    if !prefix.is_empty() || shape.tail.is_some() {
        return None;
    }
    let _tuple = ctx.descend()?;
    (ctx.room_to_descend()
        && shape
            .prefix
            .iter()
            .all(|position| scalar_of(position).is_some()))
    .then_some(&shape.prefix)
}

/// Membership for a list whose every element is a dataclass -- `list[Point]`,
/// each element the meet of the class and the record of its fields -- and
/// `None` behind a fixed prefix.
///
/// An element whose type is the class passes the meet's class conjunct as the
/// walk reads it, off the type pointer, so its membership is the record's, and
/// the record is walked directly, at the levels the walk would walk it at: the
/// meet's and the record's are held around it. Any other element -- a subclass
/// instance, another value, one where either level is not free or a fatal
/// signal is recorded -- is walked whole, which asks `isinstance` and refuses
/// where the walk refuses. The general loop paid a call into `member` for the
/// meet, one for the class and one for the record around the same walk of the
/// fields. The list is read in place: reading an attribute may run Python,
/// which may move the list.
#[inline(never)]
pub(super) fn class_record_list_matches(
    list: &Bound<'_, PyList>,
    prefix: &[Schema],
    element: &Schema,
    index: ClassIx,
    fields: &[Field],
    value: &Value<'_, '_>,
    frame: &mut Frame<'_, '_>,
) -> Option<bool> {
    let ctx = frame.ctx;
    if !prefix.is_empty() {
        return None;
    }
    let class = class_at(ctx, index, value.py())?;
    let mut ok = true;
    let scan = scan_list(list, |at, item| {
        if ctx.mode.explains() {
            frame.path.push(PathSegment::Index(at));
        }
        let held = !ctx.fatal_seen.get() && is_exactly_a(item, class);
        let item = Value::Py(item);
        ok &= if held
            && let Some(_meet) = ctx.descend()
            && let Some(_record) = ctx.descend()
        {
            check_attr_record(fields, &item, frame)
        } else {
            member(element, &item, frame)
        };
        if ctx.mode.explains() {
            frame.path.pop();
        }
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

/// Membership for a list whose every element is a named tuple of scalar
/// fields -- `list[Point]`, each element the meet of the class and the tuple
/// its fields lay out -- where the three levels below the list are free, and
/// `None` elsewhere.
///
/// An element whose type is the class, holding as many positions as the tuple
/// has and each passing its kind's test, belongs to both conjuncts as the walk
/// reads them: the class off the type pointer, as
/// [`check_instance`](super::check_instance) reads it, and the tuple through
/// its storage, which `CPython` reads whatever the class overrides. Any other
/// element is walked, which asks `isinstance` and may run the class's
/// `__instancecheck__`. The general loop paid a call into `member` for the meet
/// and one for each conjunct around the same tests, and read each position's
/// schema again for every tuple. A list is read in place unless a snapshot
/// every element of which fits settles it.
///
/// The meet's level is held while the tuple's and its positions' are asked
/// for, which is what the walk does with each element. `PyPy` reads no type
/// pointer, so there every element is walked.
#[inline(never)]
pub(super) fn named_tuple_list_matches(
    list: &Bound<'_, PyList>,
    prefix: &[Schema],
    element: &Schema,
    members: &[Schema],
    value: &Value<'_, '_>,
    frame: &mut Frame<'_, '_>,
) -> Option<bool> {
    let ctx = frame.ctx;
    let [
        Schema::Seq {
            container: SeqKind::Tuple,
            shape,
        },
        Schema::Instance(index),
    ] = members
    else {
        return None;
    };
    let positions = {
        let _meet = ctx.descend()?;
        scalar_tuple_positions(prefix, shape, ctx)?
    };
    let class = class_at(ctx, *index, value.py())?;
    let fits = |item: &Value<'_, '_>| {
        matches!(item, Value::Py(obj) if is_exactly_a(obj, class)
        && obj.cast::<PyTuple>().is_ok_and(|tuple| {
            tuple.len() == positions.len()
                && tuple.iter_borrowed().zip(positions).all(|(item, position)| {
                    scalar_of(position)
                        .is_some_and(|kind| scalar_admits(kind, &Value::Py(&item)))
                })
        }))
    };
    if ctx.mode.explains() {
        return Some(list_explained(list, element, fits, value, frame));
    }
    if let Some(answer) = admitted_through_snapshot(list, &fits, value, frame) {
        return Some(answer);
    }
    let mut ok = true;
    let scan = scan_list(list, |_, item| {
        let item = Value::Py(item);
        ok &= fits(&item) || member(element, &item, frame);
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
        if let (true, Some(kind)) = (prefix.is_empty(), repeated) {
            return Some(homogeneous_tuple_explained(tuple, tail, kind, frame));
        }
        return Some(tuple_explained(tuple, prefix, tail, admits, frame));
    }
    Some(
        tuple
            .iter_borrowed()
            .enumerate()
            .all(|(at, item)| admits(at, &Value::Py(&item))),
    )
}

/// [`tuple_explained`] for a tuple whose every position is one scalar kind --
/// `tuple[int, ...]` -- with that kind's test a constant inside the loop.
///
/// The positions reader asks each position which schema it holds and matches
/// that schema's kind at every element, which is the dispatch
/// [`scalar_list_matches`] makes once per list. Read through it, `validate` on
/// a thousand integers in a tuple costs two and a half times what `is_valid`
/// does, which reads the kind once.
fn homogeneous_tuple_explained(
    tuple: &Bound<'_, PyTuple>,
    tail: Option<&Schema>,
    kind: Scalar,
    frame: &mut Frame<'_, '_>,
) -> bool {
    match kind {
        Scalar::Int => tuple_explained(tuple, &[], tail, |_, item| item.is_int(), frame),
        Scalar::Str => tuple_explained(tuple, &[], tail, |_, item| item.is_str(), frame),
        Scalar::Float => tuple_explained(tuple, &[], tail, |_, item| item.is_float(), frame),
        Scalar::Bool => tuple_explained(tuple, &[], tail, |_, item| item.is_bool(), frame),
        Scalar::Bytes => tuple_explained(tuple, &[], tail, |_, item| item.is_bytes(), frame),
        Scalar::NoneType => tuple_explained(tuple, &[], tail, |_, item| item.is_none(), frame),
        Scalar::Everything | Scalar::Nothing => {
            tuple_explained(tuple, &[], tail, |_, item| scalar_admits(kind, item), frame)
        }
    }
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
///
/// The loop is spelled once for a build with a global lock and once for a
/// free-threaded one (`scan_held_list`), and the two read the same positions
/// against the same count. Here the count is compared just before each item is
/// asked for, and the compiler reads the iterator's own bound against that
/// comparison and drops it: `is_valid` on a thousand integers read in place on
/// 3.14 costs 20 instructions an element, against 32 through the free-threaded
/// spelling. The critical section is no lock on this build, and its closure
/// stays because the explaining reader inlines the loop differently without
/// it: `validate` on the same list read 9% more instructions.
#[cfg(not(Py_GIL_DISABLED))]
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

/// [`scan_list`] on a free-threaded build, where the list is held for the
/// whole scan and read by `scan_held_list`.
#[cfg(Py_GIL_DISABLED)]
pub(super) fn scan_list<'py>(
    list: &Bound<'py, PyList>,
    visit: impl FnMut(usize, &Bound<'py, PyAny>) -> ControlFlow<()>,
) -> Scan {
    with_critical_section(list.as_any(), || scan_held_list(list, visit))
}

/// Visit a list's items by position inside the critical section [`scan_list`]
/// holds on it, on a free-threaded build.
///
/// `PyO3`'s list iterator takes the list's section around every step it is
/// asked for, and inside the section the scan already holds each of those is
/// a re-entry: a call into the interpreter, a compare-and-swap that fails, a
/// second call that finds the section already held, and a third to end it, at
/// every element. Its `find_map` takes the section once and steps inside it,
/// so the scan is spelled through it and pays the re-entry once a list:
/// `is_valid` on a `list[tuple[int, str]]` of a thousand elements reads 12%
/// fewer instructions on 3.14t and 14% fewer on 3.15t, and the 3.14t PGO
/// wheel takes 19% less time over it.
///
/// The count is re-read after each item rather than before it. That is the
/// same reading: nothing runs between taking the count and asking for the
/// first item, and the re-read after the last item is the one the scan ends on.
#[cfg(Py_GIL_DISABLED)]
fn scan_held_list<'py>(
    list: &Bound<'py, PyList>,
    mut visit: impl FnMut(usize, &Bound<'py, PyAny>) -> ControlFlow<()>,
) -> Scan {
    let items = list.len();
    let mut at = 0;
    list.iter()
        .find_map(|item| {
            let flow = visit(at, &item);
            at += 1;
            if flow.is_break() {
                Some(Scan::Stopped)
            } else if list.len() != items {
                Some(Scan::Unreadable)
            } else {
                None
            }
        })
        .unwrap_or(Scan::Complete)
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
#[expect(
    clippy::redundant_closure_for_method_calls,
    reason = "the explaining reader takes a test over a `Value` of any \
              lifetime, and the method path names one lifetime; each closure \
              is the method, generic over the lifetime the reader hands it"
)]
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
        let (set, schema) = (container, element);
        return match scalar_of(element).filter(|_| ctx.room_to_descend()) {
            Some(Scalar::Int) => explain_elements(schema, set, |v| v.is_int(), value, frame),
            Some(Scalar::Str) => explain_elements(schema, set, |v| v.is_str(), value, frame),
            Some(Scalar::Float) => explain_elements(schema, set, |v| v.is_float(), value, frame),
            Some(Scalar::Bool) => explain_elements(schema, set, |v| v.is_bool(), value, frame),
            Some(Scalar::Bytes) => explain_elements(schema, set, |v| v.is_bytes(), value, frame),
            Some(Scalar::NoneType) => explain_elements(schema, set, |v| v.is_none(), value, frame),
            Some(Scalar::Everything | Scalar::Nothing) | None => {
                let admits = |v: &Value<'_, '_>| admitted_quietly(schema, v, ctx);
                explain_elements(schema, set, admits, value, frame)
            }
        };
    }
    // A set of one scalar kind, as a sequence of one is: the element schema is
    // read once and each element tested against the kind, a scan per kind with
    // its test a constant inside it, without the walk's per-element signal
    // check and dispatch. The level every element sits at is taken once for
    // the scan rather than skipped, so this answers what the explaining walk
    // beside it answers at the depth bound.
    let scan = match scalar_of(element).filter(|_| ctx.room_to_descend()) {
        Some(Scalar::Int) => elements_admitted(container, ctx, |v| v.is_int()),
        Some(Scalar::Str) => elements_admitted(container, ctx, |v| v.is_str()),
        Some(Scalar::Float) => elements_admitted(container, ctx, |v| v.is_float()),
        Some(Scalar::Bool) => elements_admitted(container, ctx, |v| v.is_bool()),
        Some(Scalar::Bytes) => elements_admitted(container, ctx, |v| v.is_bytes()),
        Some(Scalar::NoneType) => elements_admitted(container, ctx, |v| v.is_none()),
        Some(kind @ (Scalar::Everything | Scalar::Nothing)) => {
            elements_admitted(container, ctx, |v| scalar_admits(kind, v))
        }
        None => elements_admitted(container, ctx, |v| member(element, v, frame)),
    };
    match scan {
        Scan::Complete => true,
        Scan::Stopped => false,
        Scan::Unreadable => mutated(value, frame),
    }
}

/// Scan a set until an element does not pass `admits`: the deciding walk's
/// reading, which stops at the first element that refuses.
fn elements_admitted(
    set: &Bound<'_, PyAny>,
    ctx: Ctx<'_>,
    mut admits: impl FnMut(&Value<'_, '_>) -> bool,
) -> Scan {
    scan_set(set, ctx, |item| {
        if admits(&Value::Py(item)) {
            ControlFlow::Continue(())
        } else {
            ControlFlow::Break(())
        }
    })
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
/// walk would admit reports nothing, so one `admits` answers is passed without
/// a probe of its own: [`admitted_quietly`] for any element schema, and for a
/// scalar kind with a level free under the set the kind's own test, chosen
/// once for the set as the deciding loop chooses it. A test runs no Python, so
/// an element it passes is one the walk passes; asked through
/// [`admitted_quietly`] at every element, it cost `validate` over a `set[str]`
/// a fifth again what `is_valid` pays on 3.14.
fn explain_elements(
    element: &Schema,
    container: &Bound<'_, PyAny>,
    admits: impl Fn(&Value<'_, '_>) -> bool,
    value: &Value<'_, '_>,
    frame: &mut Frame<'_, '_>,
) -> bool {
    let ctx = frame.ctx;
    let where_it_is = &mut *frame.path;
    let mut failures: Vec<(String, Vec<Violation>)> = Vec::new();
    let scan = scan_set(container, ctx, |item| {
        if admits(&Value::Py(item)) {
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
    elements_reported(failures, frame)
}

/// Report the failures [`explain_elements`] gathered, ordered by what they say.
///
/// Apart from the scan, which takes a test of its own per kind, so the one
/// sort and report serve them all.
fn elements_reported(
    mut failures: Vec<(String, Vec<Violation>)>,
    frame: &mut Frame<'_, '_>,
) -> bool {
    let ok = failures.is_empty();
    failures.sort_by(|left, right| left.0.cmp(&right.0));
    let reported = if frame.ctx.mode.stops_at_first() {
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
/// free-threaded build pays each count of the pair through a call into the
/// interpreter, where the copy writes both inline: 6.63 against 2.05 on 3.14t,
/// with the list held once for the whole scan, and the copy pays at every
/// width -- three elements read 65 ns through it against 77 in place -- so the
/// lower end of the band does not apply there.
const fn snapshot_pays(len: usize) -> bool {
    if cfg!(Py_GIL_DISABLED) {
        len <= SNAPSHOT_MAX_ELEMENTS
    } else if cfg!(Py_3_14) {
        false
    } else {
        len >= SNAPSHOT_MIN_ELEMENTS && len <= SNAPSHOT_MAX_ELEMENTS
    }
}
