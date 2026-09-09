//! The containers a value is read as a run of *elements*.
//!
//! A list, a tuple, a parsed array, a set and a frozenset differ in how their
//! elements are reached and agree on what each element must be. What they share
//! is the reading a record does not: an arity, a positional schema or a
//! repeated tail, a count taken once and compared again, and the snapshot a
//! list of one scalar kind is read through.

use std::ops::ControlFlow;

use jiter::JsonValue;
use pyo3::prelude::*;
use pyo3::sync::critical_section::with_critical_section;
use pyo3::types::{PyFrozenSet, PyList, PySet, PyTuple};
use valgebra_core::{PathSegment, Schema, SeqKind, SeqShape, Violation};

use super::{
    Frame, Scan, homogeneous_scalar, is_fatal, member, mutated, record_fatal, scalar_admits,
    scalar_of, stop,
};
use crate::check::ctx::Ctx;
use crate::check::violation::{summarize_value, type_fail};
use crate::input::Value;

/// Membership for a sequence node: the value is a list or tuple whose elements
/// take the schema's shape — a fixed positional prefix then an optional repeated
/// tail. The elements are walked lazily against the shape the node holds, with
/// no automaton and no collection, identical in cost to a direct positional or
/// homogeneous check. JSON arrays are lists.
#[inline]
pub(super) fn check_seq(
    container: SeqKind,
    shape: &SeqShape,
    value: &Value<'_, '_>,
    frame: &mut Frame<'_, '_>,
) -> bool {
    let ctx = frame.ctx;
    let (kind_word, type_code, len_code) = match container {
        SeqKind::List => ("list", "list_type", "list_length"),
        SeqKind::Tuple => ("tuple", "tuple_type", "tuple_length"),
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
            // A list of one scalar kind -- `list[int]`, `list[str]` -- is the
            // shape whose per-element cost is almost all bookkeeping: the walk's
            // depth guard, its fatal-signal check and its dispatch, around a
            // single type test. None of the three is needed per element here: a
            // scalar cannot recurse, cannot run Python, and is the same schema at
            // every position, so they are paid once for the list.
            if let Some(kind) = homogeneous_scalar(prefix, tail, ctx) {
                // A snapshot of the list is a tuple, and a tuple's elements are
                // read borrowed. The reading it answers about is the list as it
                // was when the copy was taken, so the count is compared again
                // afterwards and a value that moved reports the move, exactly as
                // the in-place scan does. A copy the interpreter cannot make is
                // not a verdict: the walk reads in place instead.
                if snapshot_pays(list.len())
                    && let Ok(snapshot) = list.as_sequence().to_tuple()
                {
                    let ok = snapshot
                        .iter_borrowed()
                        .all(|item| scalar_admits(kind, &Value::Py(&item)));
                    if !ok {
                        return false;
                    }
                    return if list.len() == snapshot.len() {
                        true
                    } else {
                        mutated(value, frame)
                    };
                }
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
                    Scan::Unreadable => mutated(value, frame),
                };
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
            if !SeqArity::of(prefix.len(), tail).admits(tuple.len()) {
                return seq_length_fail(len_code, kind_word, prefix, tail, value, frame);
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
                ok &= seq_element(prefix, tail, i, &Value::Py(&item), frame);
                if !ok && stop(ctx) {
                    return false;
                }
            }
            ok
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
    // position, and nothing per element but the test.
    if let Some(kind) = homogeneous_scalar(prefix, tail, ctx) {
        return items
            .iter()
            .all(|item| scalar_admits(kind, &Value::Json(py, item)));
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
    len_code: &'static str,
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
            code: len_code,
            path: frame.path.clone(),
            expected,
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
#[inline]
pub(super) fn check_set(
    element: &Schema,
    value: &Value<'_, '_>,
    frame: &mut Frame<'_, '_>,
) -> bool {
    check_elements(&SET, element, value, frame)
}

/// A frozenset whose every element matches `element`. JSON has no frozensets.
#[inline]
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
    // per-element depth guard, signal check and dispatch.
    let scalar = scalar_of(element);
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
/// which costs a full scan of a set that is already failing.
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
/// A list of one scalar kind is read through a snapshot of it (see
/// [`snapshot_pays`]), and the two ends of that band are where the snapshot
/// stops paying. Below the first, its fixed cost -- one call, one allocation --
/// outweighs what it saves: sixteen elements is where the two meet, and a
/// four-element list reads thirteen percent dearer through a snapshot on
/// `CPython` 3.12. Above the second, the copy is large enough that walking it
/// costs more cache than the reference counts it avoids: measured on one box,
/// a snapshot reads a hundred thousand elements at 1.68 ns each against 4.57
/// in place, two hundred thousand at 1.73 against 4.74, four hundred thousand
/// at 3.88 against 4.68, and six hundred thousand at 5.76 against 4.61 -- so
/// the crossing is between four and six hundred thousand, and the cap sits
/// below it with margin, at two mebibytes of transient.
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
