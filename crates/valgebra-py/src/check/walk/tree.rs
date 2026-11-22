//! A definition read as a tree of scalars ([`TreePlan`]).
//!
//! The general walk reads a value against such a definition as it reads any
//! other: the reference enters its pair on the trail, the union tries its
//! members in turn, and a container's every element is the reference again.
//! Each value pays that frame around a question its type answers, and a float
//! pays the containers' walks too, which the union tries before the
//! refinement. Here the question is asked by the value's exact type, once,
//! and the walk's state is kept as the general walk would keep it, so a value
//! no type settles is handed to that walk where it would have reached it.

use std::ops::ControlFlow;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat, PyList, PyString};
use valgebra_core::{DefIx, Schema};

use super::record::scan_dict;
use super::scalar::{exact_of, leaf_admits};
use super::sequence::scan_list;
use super::{Frame, Scan, member, mutated, note_scalar_reach, record_move};
use crate::check::ctx::{Ctx, Entered, MAX_RECURSION_DEPTH, MAX_WALK_DEPTH, WalkMode};
use crate::check::index::{Exact, TreePlan};
use crate::input::Value;

/// The levels from a container's reference to each element's: the union, the
/// container's member, and the element's reference.
pub(super) const TO_AN_ELEMENT: usize = 3;

/// What [`member`] answers for `schema`, a reference to the definition `id`
/// that the index reads as a tree of scalars, at `value`, or `None` where the
/// general walk must read it: a walk in another mode, a parsed value, a value
/// of no exact builtin type, a container met inside itself or in a walk with
/// a memo armed, and a value where a bound may stop the walk.
///
/// Asked from the reference's arm, with its level open. A kept walk reads how
/// deep the walks inside it reached, which the reading of a container does not
/// count, so there a scalar alone is read ([`kept_scalar_read`]). Its answer is
/// the general walk's: a value is settled here only where the walk could not
/// stop -- the levels to its type tests free, and its reference's pair below
/// the trail's bound -- and such a type test runs no Python and records
/// nothing. Any other value is handed to the general walk at the level and
/// with the trail that walk would hold there, so it stops, records and runs
/// Python exactly as it would have. Where it answers, it enters the definition
/// as the reference's arm does where it walks one.
#[inline(never)]
pub(super) fn tree_member(
    id: DefIx,
    schema: &Schema,
    value: &Value<'_, '_>,
    frame: &mut Frame<'_, '_>,
) -> Option<bool> {
    let ctx = frame.ctx;
    let (Value::Py(value), WalkMode::Fast) = (value, ctx.mode) else {
        return None;
    };
    let plan = ctx.index.tree(id)?;
    let height = ctx.state.guard.borrow().height();
    if ctx.state.memo.is_armed() || !readable(plan, ctx.state.depth.get(), height) {
        return kept_scalar_read(plan, id, value, height, ctx);
    }
    let admitted = read(plan, id, schema, value, frame)?;
    ctx.state.memo.enter();
    Some(admitted)
}

/// Whether a value whose reference stands `depth` levels down, with `height`
/// pairs on the trail beside its own, is read here: where the levels to its
/// deepest type test are free -- the union the definition is, and its leaves'
/// own ([`LeafPlan::levels`](crate::check::index::LeafPlan::levels)) -- and its pair is below the trail's bound, so
/// the walk could not stop on it.
fn readable(plan: &TreePlan, depth: usize, height: usize) -> bool {
    depth + plan.leaf.levels <= MAX_WALK_DEPTH && height < MAX_RECURSION_DEPTH
}

/// What a value is, read by its exact type, with its reference's level open
/// and the levels and the trail entry its type tests take free; `None` for a
/// value the walk must read.
fn read(
    plan: &TreePlan,
    id: DefIx,
    schema: &Schema,
    value: &Bound<'_, PyAny>,
    frame: &mut Frame<'_, '_>,
) -> Option<bool> {
    match exact_of(value)? {
        Exact::Dict if plan.reads(Exact::Dict) => {
            entries_read(plan, id, schema, value.cast_exact::<PyDict>().ok()?, frame)
        }
        Exact::List if plan.reads(Exact::List) => {
            elements_read(plan, id, schema, value.cast_exact::<PyList>().ok()?, frame)
        }
        exact => leaf_admits(&plan.leaf, value, exact),
    }
}

/// Open a container's level, as its reference enters its pair, where each of
/// its values is [`readable`] beneath it. `None` where they are not, and where
/// the container is already on the trail, a value inside itself: the general
/// walk reads both.
fn opened(plan: &TreePlan, id: DefIx, container: &Bound<'_, PyAny>, ctx: Ctx<'_>) -> Option<usize> {
    let depth = ctx.state.depth.get();
    let mut trail = ctx.state.guard.borrow_mut();
    if !readable(plan, depth + TO_AN_ELEMENT, trail.height() + 1) {
        return None;
    }
    match trail.enter((container.as_ptr() as usize, id.get())) {
        Entered::Open => Some(depth),
        Entered::Cycle | Entered::Full => None,
    }
}

/// Close the level [`opened`] opened at `depth`, and say what the scan read.
fn closed(depth: usize, scan: &Scan, admitted: bool, ctx: Ctx<'_>) -> bool {
    ctx.state.depth.set(depth);
    ctx.state.guard.borrow_mut().leave();
    match scan {
        Scan::Complete => admitted,
        Scan::Stopped => false,
        Scan::Unreadable => {
            record_move(ctx);
            false
        }
    }
}

/// Whether every element of `list` is a member, read as the general walk reads
/// them: in order, against the count the scan took, to the first refused.
fn elements_read(
    plan: &TreePlan,
    id: DefIx,
    schema: &Schema,
    list: &Bound<'_, PyList>,
    frame: &mut Frame<'_, '_>,
) -> Option<bool> {
    let ctx = frame.ctx;
    let depth = opened(plan, id, list.as_any(), ctx)?;
    ctx.state.depth.set(depth + TO_AN_ELEMENT);
    let mut admitted = true;
    let scan = scan_list(list, |_, item| {
        admitted = held(plan, id, schema, item, frame);
        if admitted {
            ControlFlow::Continue(())
        } else {
            ControlFlow::Break(())
        }
    });
    Some(closed(depth, &scan, admitted, ctx))
}

/// Whether every entry of `dict` is a member, a `str` key and a value of the
/// definition, read as the general walk reads them: the key, then the value,
/// in order, to the first refused.
fn entries_read(
    plan: &TreePlan,
    id: DefIx,
    schema: &Schema,
    dict: &Bound<'_, PyDict>,
    frame: &mut Frame<'_, '_>,
) -> Option<bool> {
    let ctx = frame.ctx;
    let depth = opened(plan, id, dict.as_any(), ctx)?;
    ctx.state.depth.set(depth + TO_AN_ELEMENT);
    let mut admitted = true;
    let scan = scan_dict(dict, ctx.state, |key, item| {
        admitted = key.is_instance_of::<PyString>() && held(plan, id, schema, item, frame);
        if admitted {
            ControlFlow::Continue(())
        } else {
            ControlFlow::Break(())
        }
    });
    Some(closed(depth, &scan, admitted, ctx))
}

/// Whether a value a container holds is a member, read here where its type
/// settles it, and otherwise handed to the general walk from the container's
/// level, where that walk opens the reference's.
fn held(
    plan: &TreePlan,
    id: DefIx,
    schema: &Schema,
    item: &Bound<'_, PyAny>,
    frame: &mut Frame<'_, '_>,
) -> bool {
    if let Some(admitted) = read(plan, id, schema, item, frame) {
        return admitted;
    }
    let depth = &frame.ctx.state.depth;
    let at = depth.get();
    depth.set(at - 1);
    let admitted = member(schema, &Value::Py(item), frame);
    depth.set(at);
    admitted
}

/// A list whose element is a reference to a definition the index reads as a
/// tree of scalars -- `list[J]`, which a JSON Schema translation meets every
/// array with -- each element read as [`tree_member`] reads it at the element's
/// reference, or `None` where that reading would decline at every element: a
/// walk in another mode, a prefix, or a reference's level or its pair past a
/// bound. In a walk with a memo armed, [`kept_list_read`] reads it.
///
/// The general loop pays each element's level, its reference's arm and the
/// reading's own entry around a question the element's type answers. Here the
/// bounds are asked once, as each element's reference would ask them, and an
/// element the reading declines is walked whole from the list's level, where
/// the loop walks it, so it opens the same level and asks the same reading.
#[inline(never)]
pub(super) fn tree_list_read(
    id: DefIx,
    element: &Schema,
    prefix: &[Schema],
    list: &Bound<'_, PyList>,
    value: &Value<'_, '_>,
    frame: &mut Frame<'_, '_>,
) -> Option<bool> {
    #[cfg(test)]
    super::taken::note(super::taken::Reading::TreeList);
    let ctx = frame.ctx;
    if !prefix.is_empty() || ctx.mode != WalkMode::Fast {
        return None;
    }
    if ctx.state.memo.is_armed() {
        return kept_list_read(id, element, list, value, frame);
    }
    let plan = ctx.index.tree(id)?;
    let depth = ctx.state.depth.get();
    if !readable(plan, depth + 1, ctx.state.guard.borrow().height()) {
        return None;
    }
    let mut admitted = true;
    let mut seen = Seen::default();
    let scan = scan_list(list, |_, item| {
        admitted = element_read(plan, id, element, item, depth, frame, &mut seen);
        if admitted {
            ControlFlow::Continue(())
        } else {
            ControlFlow::Break(())
        }
    });
    Some(match scan {
        Scan::Complete => admitted,
        Scan::Stopped => false,
        Scan::Unreadable => mutated(value, frame),
    })
}

/// A mapping of `str` keys to a reference to a definition the index reads as
/// a tree of scalars -- `dict[str, J]`, which a JSON Schema translation meets
/// every object with -- each entry read as the general scan reads it, its key
/// an instance of `str` and its value read as [`tree_list_read`] reads a
/// list's element, or `None` where that reading would decline at every value.
/// In a walk with a memo armed, [`kept_dict_read`] reads it.
///
/// The scan asks each value through its clause, the reference's arm and the
/// reading's own entry; here the bounds are asked once, as each value's
/// reference would ask them, and a value the reading declines is walked whole
/// from the mapping's level, as the scan walks it.
#[inline(never)]
pub(super) fn tree_dict_read(
    id: DefIx,
    value_schema: &Schema,
    dict: &Bound<'_, PyDict>,
    frame: &mut Frame<'_, '_>,
) -> Option<bool> {
    let ctx = frame.ctx;
    if ctx.mode != WalkMode::Fast {
        return None;
    }
    if ctx.state.memo.is_armed() {
        return kept_dict_read(id, value_schema, dict, frame);
    }
    let plan = ctx.index.tree(id)?;
    let depth = ctx.state.depth.get();
    if !readable(plan, depth + 1, ctx.state.guard.borrow().height()) {
        return None;
    }
    let mut admitted = true;
    let mut seen = Seen::default();
    let scan = scan_dict(dict, ctx.state, |key, item| {
        admitted = key.is_instance_of::<PyString>()
            && element_read(plan, id, value_schema, item, depth, frame, &mut seen);
        if admitted {
            ControlFlow::Continue(())
        } else {
            ControlFlow::Break(())
        }
    });
    Some(match scan {
        Scan::Complete => admitted,
        Scan::Stopped => false,
        Scan::Unreadable => {
            record_move(ctx);
            false
        }
    })
}

/// Whether an element of a list of the definition `id` is a member: read at
/// its reference's level, one below the list's `depth`, and entering the
/// definition as [`tree_member`] does where it answers, or walked whole where
/// the reading declines. A memo armed for one element's walk is put back when
/// that walk returns, so the list's own reading of it holds for every element.
fn element_read(
    plan: &TreePlan,
    id: DefIx,
    element: &Schema,
    item: &Bound<'_, PyAny>,
    depth: usize,
    frame: &mut Frame<'_, '_>,
    seen: &mut Seen,
) -> bool {
    let ctx = frame.ctx;
    let read = match seen.name(plan, item) {
        Named::Kept => Some(true),
        Named::Answered(admitted) => Some(admitted),
        Named::Exact(_) => {
            ctx.state.depth.set(depth + 1);
            let read = read(plan, id, element, item, frame);
            ctx.state.depth.set(depth);
            read
        }
        Named::Other => None,
    };
    if let Some(admitted) = read {
        ctx.state.memo.enter();
        return admitted;
    }
    member(element, &Value::Py(item), frame)
}

/// [`tree_member`] where it declines, in a walk kept for the memo, one a node
/// armed: a scalar read by its exact type, with the reach its reference's walk
/// would have noted, where the walk could not stop on it, and `None` for a
/// container, which the general walk keeps. Asked where the reading declines,
/// so with no memo armed the bounds declined it, and decline it here.
///
/// A kept walk reads how far the walks inside it reached. A scalar's walk
/// enters no reference below it, so its reach is its reference's own; the
/// reading of a container reads levels below it that no reach would count.
/// And kept, a container is found by a deeper meet that reaches the definition
/// at it again, which a reading would read whole a second time. Asked from the
/// branch where the reading declines, so the reading's own entry reads as it
/// does in a walk that keeps nothing.
#[cold]
#[inline(never)]
fn kept_scalar_read(
    plan: &TreePlan,
    id: DefIx,
    value: &Bound<'_, PyAny>,
    height: usize,
    ctx: Ctx<'_>,
) -> Option<bool> {
    let depth = ctx.state.depth.get();
    if !readable(plan, depth, height) {
        return None;
    }
    let exact = exact_of(value)?;
    if plan.reads(exact) {
        return None;
    }
    let admitted = leaf_admits(&plan.leaf, value, exact)?;
    note_scalar_reach(id, (depth, height), ctx);
    ctx.state.memo.enter();
    Some(admitted)
}

/// [`tree_list_read`] in a walk kept for the memo, the fast walk of a list with
/// no prefix: each scalar element read as [`kept_scalar_read`] reads one, the
/// reach of its reference noted once for the list, and each other element
/// walked whole from the list's level; `None` where a reference's level or its
/// pair is past a bound.
///
/// Asked once the mode and the prefix are read, so it takes the five
/// arguments that pass in registers: asked where the reading declines, with
/// the prefix besides, it cost [`tree_list_read`]'s own loop an instruction an
/// element on the profiled `CPython` 3.15 wheel.
#[cold]
#[inline(never)]
fn kept_list_read(
    id: DefIx,
    element: &Schema,
    list: &Bound<'_, PyList>,
    value: &Value<'_, '_>,
    frame: &mut Frame<'_, '_>,
) -> Option<bool> {
    let ctx = frame.ctx;
    let plan = ctx.index.tree(id)?;
    let depth = ctx.state.depth.get();
    let height = ctx.state.guard.borrow().height();
    if !readable(plan, depth + 1, height) {
        return None;
    }
    let (mut admitted, mut read) = (true, false);
    let scan = scan_list(list, |_, item| {
        admitted = kept_element_read(plan, element, item, frame, &mut read);
        if admitted {
            ControlFlow::Continue(())
        } else {
            ControlFlow::Break(())
        }
    });
    if read {
        note_scalar_reach(id, (depth + 1, height), ctx);
    }
    Some(match scan {
        Scan::Complete => admitted,
        Scan::Stopped => false,
        Scan::Unreadable => mutated(value, frame),
    })
}

/// [`tree_dict_read`] in a walk kept for the memo, the fast walk, each value
/// read as [`kept_list_read`] reads a list's element.
#[cold]
#[inline(never)]
fn kept_dict_read(
    id: DefIx,
    value_schema: &Schema,
    dict: &Bound<'_, PyDict>,
    frame: &mut Frame<'_, '_>,
) -> Option<bool> {
    let ctx = frame.ctx;
    let plan = ctx.index.tree(id)?;
    let depth = ctx.state.depth.get();
    let height = ctx.state.guard.borrow().height();
    if !readable(plan, depth + 1, height) {
        return None;
    }
    let (mut admitted, mut read) = (true, false);
    let scan = scan_dict(dict, ctx.state, |key, item| {
        admitted = key.is_instance_of::<PyString>()
            && kept_element_read(plan, value_schema, item, frame, &mut read);
        if admitted {
            ControlFlow::Continue(())
        } else {
            ControlFlow::Break(())
        }
    });
    if read {
        note_scalar_reach(id, (depth + 1, height), ctx);
    }
    Some(match scan {
        Scan::Complete => admitted,
        Scan::Stopped => false,
        Scan::Unreadable => {
            record_move(ctx);
            false
        }
    })
}

/// Whether a value a kept list or mapping of the tree holds is a member: a
/// scalar read by its type, which sets `read`, and any other value walked
/// whole from the container's level.
///
/// It names no type ([`Seen`]): a second caller of [`Seen::name`] left it out
/// of line in [`element_read`], and a list of a thousand integers against the
/// companion's `integer` counted 28 instructions an element more on the
/// profiled `CPython` 3.15 wheel.
fn kept_element_read(
    plan: &TreePlan,
    element: &Schema,
    item: &Bound<'_, PyAny>,
    frame: &mut Frame<'_, '_>,
    read: &mut bool,
) -> bool {
    let answered = match exact_of(item) {
        Some(exact) if !plan.reads(exact) => leaf_admits(&plan.leaf, item, exact),
        _ => None,
    };
    if let Some(admitted) = answered {
        #[cfg(test)]
        super::taken::note(super::taken::Reading::KeptScalar);
        *read = true;
        frame.ctx.state.memo.enter();
        return admitted;
    }
    member(element, &Value::Py(item), frame)
}

/// The exact type of a scalar a container's scan admitted by its type alone,
/// so the next value of that type is admitted by one comparison.
///
/// A list or a mapping is mostly of one type, and the leaf's answer for a
/// type it decides whatever the number is a constant ([`LeafPlan`]'s
/// `decided`); naming the type again took a comparison per kind before it,
/// the levels set and reset, and the reading's call, around that constant.
/// Only such a type is kept, so a value whose number the leaf reads -- a
/// `float` under a bound -- is read every time, and so is a value of any other
/// type. A list or a mapping of the tree keeps one; the tree's own containers
/// do not, since a value of a tree is as often a container beside a scalar,
/// and kept there, the type that missed at every other value read
/// `--binding-recursive` 8% dearer.
///
/// [`LeafPlan`]: crate::check::index::LeafPlan
#[derive(Default)]
pub(super) struct Seen(Option<*mut pyo3::ffi::PyTypeObject>);

/// What [`Seen::name`] reads a value as.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Named {
    /// Admitted by its type alone.
    Kept,
    /// A `float` the leaf's interval answers for.
    Answered(bool),
    /// Of this exact builtin type, for the reading to take.
    Exact(Exact),
    /// Of no exact builtin type.
    Other,
}

impl Seen {
    /// What `item` is to a scan of the tree `plan`, keeping its type where the
    /// leaf admits every value of it. A tree's leaves refuse every list and
    /// dict, so a container is never kept and is read as a container.
    pub(super) fn name(&mut self, plan: &TreePlan, item: &Bound<'_, PyAny>) -> Named {
        let kind = item.get_type_ptr();
        if self.0 == Some(kind) {
            return Named::Kept;
        }
        if let Some(range) = plan.leaf.floats()
            && let Ok(number) = item.cast_exact::<PyFloat>()
        {
            return Named::Answered(range.holds(number.value()));
        }
        match exact_of(item) {
            Some(exact) if plan.leaf.decided(exact) == Some(true) => {
                self.0 = Some(kind);
                Named::Kept
            }
            Some(exact) => Named::Exact(exact),
            None => Named::Other,
        }
    }
}
