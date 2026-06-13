//! The validation walk: one membership test of a value against the IR.
//!
//! [`member`] is the single walk. It returns whether the value belongs to the
//! schema's set, and in *explain* mode (`ctx.explain`) it also aggregates a
//! [`Violation`] for each independent failure into `out` (each record field,
//! each sequence element, each mapping entry), unless `ctx.fail_fast` stops it at
//! the first. In *fast* mode it allocates nothing and short-circuits as soon as
//! membership is decided — the path it took before this module fused the two
//! walks into one. There is no second walk to keep in sync.
//!
//! The walk runs over a [`Value`], so the object path and the in-place JSON path
//! share one traversal. The explain side only ever sees a Python value (the JSON
//! entry points materialize before explaining), so building a violation always
//! has a Python object in hand. The per-child path bookkeeping is gated on
//! `ctx.explain`, a flag constant for a whole walk, so the fast path pays nothing
//! for it.

use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

use jiter::JsonValue;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFrozenSet, PyList, PySet, PyString, PyTuple};
use valgebra_core::{Constraint, Field, PathSegment, Schema, Violation};

use crate::errors::{class_label, summarize, truncate};
use crate::input::Value;

/// The read-only context threaded through a validation walk: the constants pool,
/// the recursion definitions, the active recursion guard, and the two mode flags.
/// The guard records `(object id, definition index)` pairs currently on the path
/// so a value that contains itself fails with `recursion_loop` instead of
/// looping.
#[derive(Clone, Copy)]
pub(crate) struct Ctx<'a> {
    pub(crate) pool: &'a [Py<PyAny>],
    pub(crate) defs: &'a [Schema],
    pub(crate) guard: &'a RefCell<HashSet<(usize, usize)>>,
    /// Build violations into `out`. When false the walk is the membership fast
    /// path: it never touches `out`, never builds a path, and short-circuits.
    pub(crate) explain: bool,
    /// In explain mode, stop at the first failure instead of aggregating siblings.
    pub(crate) fail_fast: bool,
}

/// Whether the sibling loop should stop after a failure: always in the fast path
/// (membership is already decided), and in explain mode only under `fail_fast`.
fn stop(ctx: Ctx<'_>) -> bool {
    !ctx.explain || ctx.fail_fast
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
    match schema {
        Schema::Anything | Schema::Any => true,
        // Bottom admits nothing; an unresolved self-reference is never a member.
        Schema::Nothing => admit(false, schema, value, path, ctx, out),
        Schema::SelfRef(_) => {
            if ctx.explain {
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
        Schema::Sequence(element) => check_sequence(element, value, path, ctx, out),
        Schema::FixedSequence(elements) => check_fixed_sequence(elements, value, path, ctx, out),
        Schema::Tuple(elements) => check_tuple(elements, value, path, ctx, out),
        Schema::VariadicTuple(element) => check_variadic_tuple(element, value, path, ctx, out),
        Schema::Set(element) => check_set(element, value, path, ctx, out),
        Schema::FrozenSet(element) => check_frozenset(element, value, path, ctx, out),
        Schema::Mapping { key, value: val } => check_mapping(key, val, value, path, ctx, out),
        Schema::Record { fields, open } => {
            // Membership is the single-pass fast check; on failure the explain
            // pass re-walks in declared order to aggregate ordered violations.
            let ok = record_matches(fields, *open, value, ctx);
            if !ok && ctx.explain {
                record_explain(fields, *open, value, path, ctx, out);
            }
            ok
        }
        Schema::Union(members) => check_union(members, value, path, ctx, out),
        Schema::Intersection(members) => check_intersection(members, value, path, ctx, out),
        Schema::Complement(inner) => check_complement(inner, value, path, ctx, out),
        Schema::Instance(index) => check_instance(*index, value, path, ctx, out),
        Schema::Object {
            class_index,
            fields,
        } => check_object(*class_index, fields, value, path, ctx, out),
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
    if !ok && ctx.explain {
        out.push(mismatch(schema, value, path));
    }
    ok
}

fn check_literal(
    index: usize,
    value: &Value<'_, '_>,
    path: &[PathSegment],
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
    let literal = ctx.pool[index].bind(value.py());
    let ok = value
        .to_python()
        .is_ok_and(|obj| literal_matches(&obj, literal));
    if !ok && ctx.explain {
        out.push(Violation {
            code: "literal_value",
            path: path.to_vec(),
            expected: format!("the literal {}", summarize(literal)),
            value_summary: summarize_value(value),
        });
    }
    ok
}

/// A list whose every element matches `element`. A JSON array is a list, like the
/// value `json.loads` produces.
fn check_sequence(
    element: &Schema,
    value: &Value<'_, '_>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
    match value {
        Value::Py(v) => match v.cast::<PyList>() {
            Ok(list) => {
                let mut ok = true;
                for (index, item) in list.iter().enumerate() {
                    if ctx.explain {
                        path.push(PathSegment::Index(index));
                    }
                    ok &= member(element, &Value::Py(&item), path, ctx, out);
                    if ctx.explain {
                        path.pop();
                    }
                    if !ok && stop(ctx) {
                        return false;
                    }
                }
                ok
            }
            Err(_) => type_fail("list_type", "list", value, path, ctx, out),
        },
        Value::Json(py, JsonValue::Array(items)) => {
            let mut ok = true;
            for (index, item) in items.iter().enumerate() {
                if ctx.explain {
                    path.push(PathSegment::Index(index));
                }
                ok &= member(element, &Value::Json(*py, item), path, ctx, out);
                if ctx.explain {
                    path.pop();
                }
                if !ok && stop(ctx) {
                    return false;
                }
            }
            ok
        }
        Value::Json(..) => type_fail("list_type", "list", value, path, ctx, out),
    }
}

fn check_fixed_sequence(
    elements: &[Schema],
    value: &Value<'_, '_>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
    match value {
        Value::Py(v) => match v.cast::<PyList>() {
            Ok(list) => {
                if list.len() != elements.len() {
                    return fixed_length_fail(elements.len(), value, path, ctx, out);
                }
                let mut ok = true;
                for (index, (schema, item)) in elements.iter().zip(list.iter()).enumerate() {
                    if ctx.explain {
                        path.push(PathSegment::Index(index));
                    }
                    ok &= member(schema, &Value::Py(&item), path, ctx, out);
                    if ctx.explain {
                        path.pop();
                    }
                    if !ok && stop(ctx) {
                        return false;
                    }
                }
                ok
            }
            Err(_) => type_fail("list_type", "list", value, path, ctx, out),
        },
        Value::Json(py, JsonValue::Array(items)) => {
            if items.len() != elements.len() {
                return fixed_length_fail(elements.len(), value, path, ctx, out);
            }
            let mut ok = true;
            for (index, (schema, item)) in elements.iter().zip(items.iter()).enumerate() {
                if ctx.explain {
                    path.push(PathSegment::Index(index));
                }
                ok &= member(schema, &Value::Json(*py, item), path, ctx, out);
                if ctx.explain {
                    path.pop();
                }
                if !ok && stop(ctx) {
                    return false;
                }
            }
            ok
        }
        Value::Json(..) => type_fail("list_type", "list", value, path, ctx, out),
    }
}

/// A fixed-length list whose length governs the positional match: a length
/// mismatch is terminal, since the per-index checks are then meaningless.
fn fixed_length_fail(
    expected_len: usize,
    value: &Value<'_, '_>,
    path: &[PathSegment],
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
    if ctx.explain {
        out.push(Violation {
            code: "list_length",
            path: path.to_vec(),
            expected: format!("list of length {expected_len}"),
            value_summary: summarize_value(value),
        });
    }
    false
}

/// A tuple matched positionally. JSON has no tuples, so a JSON value is never a
/// member.
fn check_tuple(
    elements: &[Schema],
    value: &Value<'_, '_>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
    let Value::Py(v) = value else {
        return type_fail("tuple_type", "tuple", value, path, ctx, out);
    };
    let Ok(tuple) = v.cast::<PyTuple>() else {
        return type_fail("tuple_type", "tuple", value, path, ctx, out);
    };
    if tuple.len() != elements.len() {
        if ctx.explain {
            out.push(Violation {
                code: "tuple_length",
                path: path.clone(),
                expected: format!("tuple of length {}", elements.len()),
                value_summary: summarize_value(value),
            });
        }
        return false;
    }
    let mut ok = true;
    for (index, (schema, item)) in elements.iter().zip(tuple.iter()).enumerate() {
        if ctx.explain {
            path.push(PathSegment::Index(index));
        }
        ok &= member(schema, &Value::Py(&item), path, ctx, out);
        if ctx.explain {
            path.pop();
        }
        if !ok && stop(ctx) {
            return false;
        }
    }
    ok
}

/// A tuple of any length whose every element matches `element`.
fn check_variadic_tuple(
    element: &Schema,
    value: &Value<'_, '_>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
    let Value::Py(v) = value else {
        return type_fail("tuple_type", "tuple", value, path, ctx, out);
    };
    let Ok(tuple) = v.cast::<PyTuple>() else {
        return type_fail("tuple_type", "tuple", value, path, ctx, out);
    };
    let mut ok = true;
    for (index, item) in tuple.iter().enumerate() {
        if ctx.explain {
            path.push(PathSegment::Index(index));
        }
        ok &= member(element, &Value::Py(&item), path, ctx, out);
        if ctx.explain {
            path.pop();
        }
        if !ok && stop(ctx) {
            return false;
        }
    }
    ok
}

/// A set whose every element matches `element`. Set order is not meaningful, so
/// element failures carry no index segment. JSON has no sets.
fn check_set(
    element: &Schema,
    value: &Value<'_, '_>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
    let Value::Py(v) = value else {
        return type_fail("set_type", "set", value, path, ctx, out);
    };
    let Ok(set) = v.cast::<PySet>() else {
        return type_fail("set_type", "set", value, path, ctx, out);
    };
    let mut ok = true;
    for item in set.iter() {
        ok &= member(element, &Value::Py(&item), path, ctx, out);
        if !ok && stop(ctx) {
            return false;
        }
    }
    ok
}

/// A frozenset whose every element matches `element`. JSON has no frozensets.
fn check_frozenset(
    element: &Schema,
    value: &Value<'_, '_>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
    let Value::Py(v) = value else {
        return type_fail("frozenset_type", "frozenset", value, path, ctx, out);
    };
    let Ok(set) = v.cast::<PyFrozenSet>() else {
        return type_fail("frozenset_type", "frozenset", value, path, ctx, out);
    };
    let mut ok = true;
    for item in set.iter() {
        ok &= member(element, &Value::Py(&item), path, ctx, out);
        if !ok && stop(ctx) {
            return false;
        }
    }
    ok
}

/// A dict whose keys all match `key_schema` and values all match `value_schema`.
/// A JSON object's keys are strings; a duplicate key keeps its last value, as
/// `json.loads` does. The key segment for an entry's path is built only in
/// explain mode, so the fast path allocates nothing per entry.
fn check_mapping(
    key_schema: &Schema,
    value_schema: &Schema,
    value: &Value<'_, '_>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
    match value {
        Value::Py(v) => match v.cast::<PyDict>() {
            Ok(dict) => {
                let mut ok = true;
                for (key, val) in dict.iter() {
                    if ctx.explain {
                        path.push(PathSegment::Key(key_label(&key)));
                    }
                    let entry_ok = member(key_schema, &Value::Py(&key), path, ctx, out)
                        & member(value_schema, &Value::Py(&val), path, ctx, out);
                    if ctx.explain {
                        path.pop();
                    }
                    ok &= entry_ok;
                    if !ok && stop(ctx) {
                        return false;
                    }
                }
                ok
            }
            Err(_) => type_fail("dict_type", "dict", value, path, ctx, out),
        },
        Value::Json(py, JsonValue::Object(entries)) => {
            let mut ok = true;
            for (i, (key, val)) in entries.iter().enumerate() {
                // A duplicate key keeps its last value, so skip an entry whose
                // key recurs later (json.loads semantics).
                if entries[i + 1..].iter().any(|(later, _)| later == key) {
                    continue;
                }
                let key_value = JsonValue::Str(Cow::Borrowed(key.as_ref()));
                if ctx.explain {
                    path.push(PathSegment::Key(truncate(key.as_ref(), 40)));
                }
                let entry_ok = member(key_schema, &Value::Json(*py, &key_value), path, ctx, out)
                    & member(value_schema, &Value::Json(*py, val), path, ctx, out);
                if ctx.explain {
                    path.pop();
                }
                ok &= entry_ok;
                if !ok && stop(ctx) {
                    return false;
                }
            }
            ok
        }
        Value::Json(..) => type_fail("dict_type", "dict", value, path, ctx, out),
    }
}

/// Membership for a record: the single-pass fast check. The walk is inverted —
/// it visits each entry once and matches the key against the declared fields —
/// rather than looking up every declared field and rescanning for extra keys.
fn record_matches(fields: &[Field], open: bool, value: &Value<'_, '_>, ctx: Ctx<'_>) -> bool {
    match value {
        Value::Py(v) => record_matches_py(fields, open, v, ctx),
        Value::Json(py, JsonValue::Object(entries)) => {
            record_matches_json(fields, open, *py, entries, ctx)
        }
        Value::Json(..) => false,
    }
}

/// The record fast path over a Python dict. The key's UTF-8 is borrowed without
/// allocating; a non-string key whose `str()` names a field never fills it,
/// exactly as a string-key lookup would miss it.
fn record_matches_py(fields: &[Field], open: bool, dict: &Bound<'_, PyAny>, ctx: Ctx<'_>) -> bool {
    let Ok(dict) = dict.cast::<PyDict>() else {
        return false;
    };
    let declared: HashMap<&str, &Field> = fields.iter().map(|f| (f.name.as_str(), f)).collect();
    let mut required_remaining = fields.iter().filter(|f| f.required).count();
    let sub = fast(ctx);
    for (key, val) in dict.iter() {
        match key.cast::<PyString>().ok().and_then(|s| s.to_str().ok()) {
            Some(name) => match declared.get(name) {
                Some(field) => {
                    if !member(
                        &field.schema,
                        &Value::Py(&val),
                        &mut Vec::new(),
                        sub,
                        &mut Vec::new(),
                    ) {
                        return false;
                    }
                    if field.required {
                        required_remaining -= 1;
                    }
                }
                None if open => {}
                None => return false,
            },
            None => {
                if !open && !stringifies_to_declared(&key, &declared) {
                    return false;
                }
            }
        }
    }
    required_remaining == 0
}

/// The record fast path over a JSON object. Keys are strings; a duplicate key
/// keeps its last value (a reverse find), as `json.loads` does. Records are
/// small, so a linear scan beats building a per-object map.
fn record_matches_json(
    fields: &[Field],
    open: bool,
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
    if open {
        return true;
    }
    // Closed: every object key must name a declared field.
    entries
        .iter()
        .all(|(key, _)| fields.iter().any(|f| f.name == key.as_ref()))
}

/// Whether the `str()` of a non-string key names a declared field, mirroring the
/// stringified extra-key check on the explain path.
fn stringifies_to_declared(key: &Bound<'_, PyAny>, declared: &HashMap<&str, &Field>) -> bool {
    key.str()
        .ok()
        .and_then(|text| text.to_str().ok().map(|name| declared.contains_key(name)))
        .unwrap_or(false)
}

/// The explain pass over a record, run only after [`record_matches`] has already
/// reported the value is not a member. It re-walks in declared order so the
/// aggregated violations read in declared order: present fields checked in order,
/// then absent required keys, then (for a closed record) extra keys.
fn record_explain(
    fields: &[Field],
    open: bool,
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
        if ctx.fail_fast && !out.is_empty() {
            return;
        }
    }
    if open {
        return;
    }
    let declared: HashSet<&str> = fields.iter().map(|field| field.name.as_str()).collect();
    for (key, _) in dict.iter() {
        let key_text = key
            .str()
            .map_or_else(|_| String::new(), |text| text.to_string());
        if !declared.contains(key_text.as_str()) {
            out.push(located(
                path,
                key_text.clone(),
                "extra_key",
                "no unexpected key".to_owned(),
                format!("{key_text:?}"),
            ));
            if ctx.fail_fast {
                return;
            }
        }
    }
}

fn check_union(
    members: &[Schema],
    value: &Value<'_, '_>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
    // A value is a member iff it matches at least one branch; decide that on the
    // fast path, where a discarded branch pays for no path or violation.
    let sub = fast(ctx);
    if members
        .iter()
        .any(|m| member(m, value, &mut Vec::new(), sub, &mut Vec::new()))
    {
        return true;
    }
    if !ctx.explain {
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
        explain: true,
        fail_fast: false,
        ..ctx
    };
    let mut best: Option<(usize, Vec<Violation>)> = None;
    for branch_schema in members {
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
            let labels: Vec<&str> = members.iter().map(Schema::expected).collect();
            out.push(Violation {
                code: "union_error",
                path: path.clone(),
                expected: format!("one of: {}", labels.join(", ")),
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
    // Every member must hold; in explain mode each member's failure is collected.
    let mut ok = true;
    for member_schema in members {
        ok &= member(member_schema, value, path, ctx, out);
        if !ok && stop(ctx) {
            return false;
        }
    }
    ok
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
        if ctx.explain {
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
    index: usize,
    value: &Value<'_, '_>,
    path: &[PathSegment],
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
    let class = ctx.pool[index].bind(value.py());
    let ok = value
        .to_python()
        .is_ok_and(|obj| obj.is_instance(class).unwrap_or(false));
    if !ok && ctx.explain {
        out.push(type_mismatch(
            "instance_type",
            &class_label(class),
            value,
            path,
        ));
    }
    ok
}

fn check_object(
    class_index: usize,
    fields: &[Field],
    value: &Value<'_, '_>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
    // A JSON value materializes to a builtin, never an instance of a user class.
    let Ok(obj) = value.to_python() else {
        return false;
    };
    let class = ctx.pool[class_index].bind(value.py());
    if !obj.is_instance(class).unwrap_or(false) {
        // Not an instance: the attribute checks below cannot be trusted.
        if ctx.explain {
            out.push(type_mismatch(
                "instance_type",
                &class_label(class),
                value,
                path,
            ));
        }
        return false;
    }
    let mut ok = true;
    for field in fields {
        if let Ok(attr) = obj.getattr(field.name.as_str()) {
            if ctx.explain {
                path.push(PathSegment::Key(field.name.clone()));
            }
            ok &= member(&field.schema, &Value::Py(&attr), path, ctx, out);
            if ctx.explain {
                path.pop();
            }
        } else {
            if ctx.explain {
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

/// Whether `value` (already a base member, materialized once) satisfies one
/// constraint, recording a violation on failure in explain mode.
fn check_constraint(
    constraint: &Constraint,
    value: &Bound<'_, PyAny>,
    path: &[PathSegment],
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
    let py = value.py();
    let (ok, code, expected): (bool, &'static str, String) = match constraint {
        Constraint::Ge(i) => {
            let bound = ctx.pool[*i].bind(py);
            (
                cmp(value.ge(bound)),
                "greater_than_equal",
                format!(">= {}", summarize(bound)),
            )
        }
        Constraint::Gt(i) => {
            let bound = ctx.pool[*i].bind(py);
            (
                cmp(value.gt(bound)),
                "greater_than",
                format!("> {}", summarize(bound)),
            )
        }
        Constraint::Le(i) => {
            let bound = ctx.pool[*i].bind(py);
            (
                cmp(value.le(bound)),
                "less_than_equal",
                format!("<= {}", summarize(bound)),
            )
        }
        Constraint::Lt(i) => {
            let bound = ctx.pool[*i].bind(py);
            (
                cmp(value.lt(bound)),
                "less_than",
                format!("< {}", summarize(bound)),
            )
        }
        Constraint::MinLen(n) => (
            value.len().is_ok_and(|len| len >= *n),
            "too_short",
            format!("length >= {n}"),
        ),
        Constraint::MaxLen(n) => (
            value.len().is_ok_and(|len| len <= *n),
            "too_long",
            format!("length <= {n}"),
        ),
        Constraint::MultipleOf(i) => {
            let operand = ctx.pool[*i].bind(py);
            (
                is_multiple_of(value, operand),
                "not_multiple_of",
                format!("a multiple of {}", summarize(operand)),
            )
        }
        Constraint::Predicate(i) => {
            // Slow path: the user's Python callable runs at the boundary. A
            // raising predicate is surfaced as a distinct `predicate_error`
            // rather than masked as an ordinary failed match.
            let predicate = ctx.pool[*i].bind(py);
            match predicate_passes(predicate, value) {
                Ok(passed) => (passed, "predicate_failed", "a passing predicate".to_owned()),
                Err(err) => (
                    false,
                    "predicate_error",
                    format!("a predicate that does not raise (raised {err})"),
                ),
            }
        }
    };
    if !ok && ctx.explain {
        out.push(Violation {
            code,
            path: path.to_vec(),
            expected,
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
    id: usize,
    value: &Value<'_, '_>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
    let key = (value.id(), id);
    let depth = {
        let mut guard = ctx.guard.borrow_mut();
        if !guard.insert(key) {
            if ctx.explain {
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
        if ctx.explain {
            out.push(Violation {
                code: "recursion_limit",
                path: path.clone(),
                expected: format!("at most {MAX_RECURSION_DEPTH} levels of recursion"),
                value_summary: summarize_value(value),
            });
        }
        return false;
    }
    let result = member(&ctx.defs[id], value, path, ctx, out);
    ctx.guard.borrow_mut().remove(&key);
    result
}

/// A copy of `ctx` switched to the membership fast path (no explanation), for the
/// speculative sub-checks of union, complement, and the record fast walk.
fn fast(ctx: Ctx<'_>) -> Ctx<'_> {
    Ctx {
        explain: false,
        fail_fast: true,
        ..ctx
    }
}

/// A type/value mismatch for a leaf schema.
fn mismatch(schema: &Schema, value: &Value<'_, '_>, path: &[PathSegment]) -> Violation {
    Violation {
        code: schema.error_code(),
        path: path.to_vec(),
        expected: schema.expected().to_owned(),
        value_summary: summarize_value(value),
    }
}

/// Record a structural type mismatch and report non-membership.
fn type_fail(
    code: &'static str,
    expected: &str,
    value: &Value<'_, '_>,
    path: &[PathSegment],
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
    if ctx.explain {
        out.push(type_mismatch(code, expected, value, path));
    }
    false
}

fn type_mismatch(
    code: &'static str,
    expected: &str,
    value: &Value<'_, '_>,
    path: &[PathSegment],
) -> Violation {
    Violation {
        code,
        path: path.to_vec(),
        expected: expected.to_owned(),
        value_summary: summarize_value(value),
    }
}

/// Build a violation whose path is `path` extended by one key segment.
fn located(
    path: &[PathSegment],
    key: String,
    code: &'static str,
    expected: String,
    value_summary: String,
) -> Violation {
    let mut full = path.to_vec();
    full.push(PathSegment::Key(key));
    Violation {
        code,
        path: full,
        expected,
        value_summary,
    }
}

/// A short repr-style summary of a value, materializing a JSON value first.
fn summarize_value(value: &Value<'_, '_>) -> String {
    match value.to_python() {
        Ok(obj) => summarize(&obj),
        Err(_) => "<unrepresentable>".to_owned(),
    }
}

/// Whether `value` is the typed singleton denoted by `literal`: same type and
/// equal. The same-type guard rules out Python's cross-type equality
/// (`1 == True == 1.0`), so `Literal[1]` denotes `{1}`, not `{1, True, 1.0}`.
fn literal_matches(value: &Bound<'_, PyAny>, literal: &Bound<'_, PyAny>) -> bool {
    value.get_type().is(literal.get_type()) && value.eq(literal).unwrap_or(false)
}

/// Whether `value % operand == 0`. A non-numeric value (whose modulo errors or
/// is not defined) is not a multiple. The remainder is zero iff it is falsy.
fn is_multiple_of(value: &Bound<'_, PyAny>, operand: &Bound<'_, PyAny>) -> bool {
    value
        .call_method1("__mod__", (operand,))
        .ok()
        .and_then(|remainder| remainder.is_truthy().ok())
        .is_some_and(|nonzero| !nonzero)
}

/// Run a user predicate and report whether it returned a truthy result.
fn predicate_passes(predicate: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<bool> {
    predicate.call1((value,))?.is_truthy()
}

/// Interpret a rich-comparison result, treating an error as "did not hold".
fn cmp(result: PyResult<bool>) -> bool {
    result.unwrap_or(false)
}

/// A short, printable label for a mapping key, used in error paths.
fn key_label(key: &Bound<'_, PyAny>) -> String {
    match key.str() {
        Ok(text) => truncate(&text.to_string(), 40),
        Err(_) => summarize(key),
    }
}
