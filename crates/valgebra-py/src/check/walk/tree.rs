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
use pyo3::types::{PyDict, PyList, PyString};
use valgebra_core::{DefIx, Schema};

use super::record::scan_dict;
use super::scalar::{exact_of, leaf_admits};
use super::sequence::scan_list;
use super::{Frame, Scan, member, record_move};
use crate::check::ctx::{Ctx, Entered, MAX_RECURSION_DEPTH, MAX_WALK_DEPTH, WalkMode};
use crate::check::index::{Exact, TreePlan};
use crate::input::Value;

/// The levels from a container's reference to each element's: the union, the
/// container's member, and the element's reference.
pub(super) const TO_AN_ELEMENT: usize = 3;

/// What [`member`] answers for `schema`, a reference to the definition `id`
/// that the index reads as a tree of scalars, at `value`, or `None` where the
/// general walk must read it: a walk in another mode or with a memo armed, a
/// parsed value, a value of no exact builtin type, a container met inside
/// itself, and a value where a bound may stop the walk.
///
/// Asked from the reference's arm, with its level open. A kept walk reads how
/// deep the walks inside it reached, which this reading does not count, so
/// where one may be kept the general walk reads the definition. Its answer is
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
        return None;
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
