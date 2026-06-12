//! The validation walk: membership testing of a Python value against the IR.
//!
//! [`check`] is the explain path: it walks the schema and *aggregates* every
//! independent failure into a `Vec<Violation>` (each record field, each
//! sequence element, each mapping entry), rather than stopping at the first.
//! With `ctx.fail_fast` it stops at the first failure instead. [`matches`] is
//! the membership fast path (a bool, no allocation) used by `is_valid` and by
//! the speculative combinators. `check` (whether any violation is produced) and
//! `matches` must stay membership-equivalent.

use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

use jiter::JsonValue;
use pyo3::prelude::*;
use pyo3::types::{
    PyBool, PyBytes, PyDict, PyFloat, PyFrozenSet, PyInt, PyList, PySet, PyString, PyTuple,
};
use valgebra_core::{Constraint, Field, PathSegment, Schema, Violation};

use crate::errors::{class_label, summarize, truncate};
use crate::input::Value;

/// The read-only context threaded through a validation walk: the constants
/// pool, the recursion definitions, the active recursion guard, and the
/// fail-fast flag. The guard records `(object id, definition index)` pairs
/// currently on the path so a value that contains itself fails with
/// `recursion_loop` instead of looping.
#[derive(Clone, Copy)]
pub(crate) struct Ctx<'a> {
    pub(crate) pool: &'a [Py<PyAny>],
    pub(crate) defs: &'a [Schema],
    pub(crate) guard: &'a RefCell<HashSet<(usize, usize)>>,
    /// Stop at the first failure instead of aggregating siblings.
    pub(crate) fail_fast: bool,
}

/// Whether the walk should stop: fail-fast is on and a failure is already
/// recorded. In the default (aggregating) mode this is always false, so the
/// walk visits every independent position.
fn aborted(ctx: Ctx<'_>, out: &[Violation]) -> bool {
    ctx.fail_fast && !out.is_empty()
}

/// Walk the schema against `value` in Rust, pushing a [`Violation`] for each
/// failure into `out`. `path` accumulates the location of the current value.
pub(crate) fn check(
    schema: &Schema,
    value: &Bound<'_, PyAny>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) {
    match schema {
        Schema::Anything | Schema::Any => {}
        Schema::Nothing => out.push(mismatch(schema, value, path)),
        Schema::NoneType => admit(value.is_none(), schema, value, path, out),
        Schema::Bool => admit(value.is_instance_of::<PyBool>(), schema, value, path, out),
        // bool subclasses int, so True/False are ints: Bool is a subset of Int.
        Schema::Int => admit(value.is_instance_of::<PyInt>(), schema, value, path, out),
        Schema::Float => admit(value.is_instance_of::<PyFloat>(), schema, value, path, out),
        Schema::Str => admit(value.is_instance_of::<PyString>(), schema, value, path, out),
        Schema::Bytes => admit(value.is_instance_of::<PyBytes>(), schema, value, path, out),
        Schema::Literal(index) => check_literal(*index, value, path, ctx, out),
        Schema::Sequence(element) => check_sequence(element, value, path, ctx, out),
        Schema::FixedSequence(elements) => check_fixed_sequence(elements, value, path, ctx, out),
        Schema::Tuple(elements) => check_tuple(elements, value, path, ctx, out),
        Schema::VariadicTuple(element) => check_variadic_tuple(element, value, path, ctx, out),
        Schema::Set(element) => check_set(element, value, path, ctx, out),
        Schema::FrozenSet(element) => check_frozenset(element, value, path, ctx, out),
        Schema::Mapping { key, value: val } => check_mapping(key, val, value, path, ctx, out),
        Schema::Record { fields, open } => check_record(fields, *open, value, path, ctx, out),
        Schema::Union(members) => check_union(members, value, path, ctx, out),
        Schema::Intersection(members) => check_intersection(members, value, path, ctx, out),
        Schema::Complement(inner) => check_complement(inner, value, path, ctx, out),
        Schema::Instance(index) => check_instance(*index, value, path, ctx, out),
        Schema::Object {
            class_index,
            fields,
        } => check_object(*class_index, fields, value, path, ctx, out),
        Schema::Refine { base, constraints } => {
            check_refine(base, constraints, value, path, ctx, out);
        }
        Schema::Ref(id) => check_ref(*id, value, path, ctx, out),
        // A SelfRef should have been resolved into a Ref at build time; reaching
        // one means an unresolved recursion marker leaked into a validator.
        Schema::SelfRef(_) => out.push(Violation {
            code: "unresolved_recursion",
            path: path.clone(),
            expected: "a resolved recursive value".to_owned(),
            value_summary: summarize(value),
        }),
    }
}

/// Decide membership without building a violation or tracking a path.
///
/// The fast path: `is_valid`/`is_valid_json` use it, and the speculative
/// combinators use it so a discarded branch never pays for a `Violation` or a
/// `repr`. It walks a [`Value`], dispatching per node on the input source, so
/// the object path and the in-place JSON path share one traversal. It must stay
/// membership-equivalent to [`check`].
pub(crate) fn matches(schema: &Schema, value: &Value<'_, '_>, ctx: Ctx<'_>) -> bool {
    match schema {
        Schema::Anything | Schema::Any => true,
        // Bottom admits nothing; an unresolved self-reference is never a member.
        Schema::Nothing | Schema::SelfRef(_) => false,
        Schema::NoneType => value.is_none(),
        Schema::Bool => value.is_bool(),
        Schema::Int => value.is_int(),
        Schema::Float => value.is_float(),
        Schema::Str => value.is_str(),
        Schema::Bytes => value.is_bytes(),
        Schema::Literal(i) => value
            .to_python()
            .is_ok_and(|obj| literal_matches(&obj, ctx.pool[*i].bind(value.py()))),
        Schema::Sequence(e) => seq_all(e, value, ctx),
        Schema::FixedSequence(es) => seq_positional(es, value, ctx),
        Schema::Tuple(es) => tuple_positional(es, value, ctx),
        Schema::VariadicTuple(e) => tuple_all(e, value, ctx),
        Schema::Set(e) => set_all(e, value, ctx),
        Schema::FrozenSet(e) => frozenset_all(e, value, ctx),
        Schema::Mapping { key, value: val } => mapping_all(key, val, value, ctx),
        Schema::Record { fields, open } => matches_record(fields, *open, value, ctx),
        Schema::Union(members) => members.iter().any(|m| matches(m, value, ctx)),
        Schema::Intersection(members) => members.iter().all(|m| matches(m, value, ctx)),
        Schema::Complement(inner) => !matches(inner, value, ctx),
        Schema::Instance(i) => value.to_python().is_ok_and(|obj| {
            obj.is_instance(ctx.pool[*i].bind(value.py()))
                .unwrap_or(false)
        }),
        Schema::Object {
            class_index,
            fields,
        } => matches_object(*class_index, fields, value, ctx),
        Schema::Refine { base, constraints } => {
            matches(base, value, ctx)
                && value
                    .to_python()
                    .is_ok_and(|obj| constraints.iter().all(|c| constraint_holds(c, &obj, ctx)))
        }
        Schema::Ref(id) => matches_ref(*id, value, ctx),
    }
}

/// A list whose every element matches `element`. JSON arrays are lists, like the
/// value `json.loads` produces.
fn seq_all(element: &Schema, value: &Value<'_, '_>, ctx: Ctx<'_>) -> bool {
    match value {
        Value::Py(v) => v.cast::<PyList>().is_ok_and(|list| {
            list.iter()
                .all(|item| matches(element, &Value::Py(&item), ctx))
        }),
        Value::Json(py, JsonValue::Array(items)) => items
            .iter()
            .all(|item| matches(element, &Value::Json(*py, item), ctx)),
        Value::Json(..) => false,
    }
}

/// A list matched positionally at exactly the schemas' length.
fn seq_positional(elements: &[Schema], value: &Value<'_, '_>, ctx: Ctx<'_>) -> bool {
    match value {
        Value::Py(v) => v.cast::<PyList>().is_ok_and(|list| {
            list.len() == elements.len()
                && elements
                    .iter()
                    .zip(list.iter())
                    .all(|(s, item)| matches(s, &Value::Py(&item), ctx))
        }),
        Value::Json(py, JsonValue::Array(items)) => {
            items.len() == elements.len()
                && elements
                    .iter()
                    .zip(items.iter())
                    .all(|(s, item)| matches(s, &Value::Json(*py, item), ctx))
        }
        Value::Json(..) => false,
    }
}

/// A tuple matched positionally. JSON has no tuples, so a JSON value is never a
/// member.
fn tuple_positional(elements: &[Schema], value: &Value<'_, '_>, ctx: Ctx<'_>) -> bool {
    match value {
        Value::Py(v) => v.cast::<PyTuple>().is_ok_and(|tuple| {
            tuple.len() == elements.len()
                && elements
                    .iter()
                    .zip(tuple.iter())
                    .all(|(s, item)| matches(s, &Value::Py(&item), ctx))
        }),
        Value::Json(..) => false,
    }
}

/// A tuple of any length whose every element matches `element`.
fn tuple_all(element: &Schema, value: &Value<'_, '_>, ctx: Ctx<'_>) -> bool {
    match value {
        Value::Py(v) => v.cast::<PyTuple>().is_ok_and(|tuple| {
            tuple
                .iter()
                .all(|item| matches(element, &Value::Py(&item), ctx))
        }),
        Value::Json(..) => false,
    }
}

/// A set whose every element matches `element`. JSON has no sets.
fn set_all(element: &Schema, value: &Value<'_, '_>, ctx: Ctx<'_>) -> bool {
    match value {
        Value::Py(v) => v.cast::<PySet>().is_ok_and(|set| {
            set.iter()
                .all(|item| matches(element, &Value::Py(&item), ctx))
        }),
        Value::Json(..) => false,
    }
}

/// A frozenset whose every element matches `element`. JSON has no frozensets.
fn frozenset_all(element: &Schema, value: &Value<'_, '_>, ctx: Ctx<'_>) -> bool {
    match value {
        Value::Py(v) => v.cast::<PyFrozenSet>().is_ok_and(|set| {
            set.iter()
                .all(|item| matches(element, &Value::Py(&item), ctx))
        }),
        Value::Json(..) => false,
    }
}

/// A dict whose keys all match `key_schema` and values all match `value_schema`.
/// A JSON object's keys are strings; a duplicate key keeps its last value, as
/// `json.loads` does.
fn mapping_all(
    key_schema: &Schema,
    value_schema: &Schema,
    value: &Value<'_, '_>,
    ctx: Ctx<'_>,
) -> bool {
    match value {
        Value::Py(v) => v.cast::<PyDict>().is_ok_and(|dict| {
            dict.iter().all(|(k, val)| {
                matches(key_schema, &Value::Py(&k), ctx)
                    && matches(value_schema, &Value::Py(&val), ctx)
            })
        }),
        Value::Json(py, JsonValue::Object(entries)) => {
            entries.iter().enumerate().all(|(i, (key, val))| {
                // A duplicate key keeps its last value, so skip an entry whose
                // key recurs later (json.loads semantics). No allocation.
                if entries[i + 1..].iter().any(|(later, _)| later == key) {
                    return true;
                }
                let key_value = JsonValue::Str(Cow::Borrowed(key.as_ref()));
                matches(key_schema, &Value::Json(*py, &key_value), ctx)
                    && matches(value_schema, &Value::Json(*py, val), ctx)
            })
        }
        Value::Json(..) => false,
    }
}

/// Membership for an object node (isinstance plus per-attribute checks). The
/// value is materialized once; a JSON value materializes to a builtin, which is
/// never an instance of a user class, so a JSON value never matches here.
fn matches_object(
    class_index: usize,
    fields: &[Field],
    value: &Value<'_, '_>,
    ctx: Ctx<'_>,
) -> bool {
    let Ok(obj) = value.to_python() else {
        return false;
    };
    obj.is_instance(ctx.pool[class_index].bind(value.py()))
        .unwrap_or(false)
        && fields.iter().all(|f| {
            obj.getattr(f.name.as_str())
                .is_ok_and(|attr| matches(&f.schema, &Value::Py(&attr), ctx))
        })
}

/// Membership for a record: declared fields match and required keys are present,
/// and no undeclared key is admitted unless the record is open.
fn matches_record(fields: &[Field], open: bool, value: &Value<'_, '_>, ctx: Ctx<'_>) -> bool {
    match value {
        Value::Py(v) => matches_record_py(fields, open, v, ctx),
        Value::Json(py, JsonValue::Object(entries)) => {
            matches_record_json(fields, open, *py, entries, ctx)
        }
        Value::Json(..) => false,
    }
}

/// The record fast path over a Python dict.
///
/// The walk is inverted: it visits each dict entry once and matches the key
/// against the declared fields, rather than looking up every declared field
/// (which builds a temporary Python string per field) and then scanning the
/// dict a second time for undeclared keys. The key's UTF-8 is borrowed without
/// allocating. A non-string key whose `str()` names a field never fills it,
/// exactly as a string-key lookup would miss it.
fn matches_record_py(fields: &[Field], open: bool, dict: &Bound<'_, PyAny>, ctx: Ctx<'_>) -> bool {
    let Ok(dict) = dict.cast::<PyDict>() else {
        return false;
    };
    let declared: HashMap<&str, &Field> = fields.iter().map(|f| (f.name.as_str(), f)).collect();
    let mut required_remaining = fields.iter().filter(|f| f.required).count();
    for (key, val) in dict.iter() {
        match key.cast::<PyString>().ok().and_then(|s| s.to_str().ok()) {
            Some(name) => match declared.get(name) {
                Some(field) => {
                    if !matches(&field.schema, &Value::Py(&val), ctx) {
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

/// The record fast path over a JSON object. Keys are strings, so the
/// non-string-key subtlety does not arise; a duplicate key keeps its last value,
/// as `json.loads` does. No allocation: records are small, so a linear scan with
/// a reverse find (the last occurrence wins) beats building a per-object map.
fn matches_record_json(
    fields: &[Field],
    open: bool,
    py: Python<'_>,
    entries: &[(Cow<'_, str>, JsonValue<'_>)],
    ctx: Ctx<'_>,
) -> bool {
    for field in fields {
        match entries
            .iter()
            .rev()
            .find(|(key, _)| field.name == key.as_ref())
        {
            Some((_, val)) => {
                if !matches(&field.schema, &Value::Json(py, val), ctx) {
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

/// The most levels of recursive descent allowed before a value is rejected. A
/// finite value never reaches this; the bound exists so a pathologically deep
/// value fails with `recursion_limit` instead of overflowing the native stack.
const MAX_RECURSION_DEPTH: usize = 128;

fn check_ref(
    id: usize,
    value: &Bound<'_, PyAny>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) {
    let key = (value.as_ptr() as usize, id);
    let depth = {
        let mut guard = ctx.guard.borrow_mut();
        if !guard.insert(key) {
            out.push(Violation {
                code: "recursion_loop",
                path: path.clone(),
                expected: "a finite (non-cyclic) value".to_owned(),
                value_summary: summarize(value),
            });
            return;
        }
        guard.len()
    };
    if depth > MAX_RECURSION_DEPTH {
        ctx.guard.borrow_mut().remove(&key);
        out.push(Violation {
            code: "recursion_limit",
            path: path.clone(),
            expected: format!("at most {MAX_RECURSION_DEPTH} levels of recursion"),
            value_summary: summarize(value),
        });
        return;
    }
    check(&ctx.defs[id], value, path, ctx, out);
    ctx.guard.borrow_mut().remove(&key);
}

/// Membership for a recursion reference, with the same cycle and depth guards
/// as [`check_ref`] but reporting only yes/no.
fn matches_ref(id: usize, value: &Value<'_, '_>, ctx: Ctx<'_>) -> bool {
    let key = (value.id(), id);
    let depth = {
        let mut guard = ctx.guard.borrow_mut();
        if !guard.insert(key) {
            return false;
        }
        guard.len()
    };
    if depth > MAX_RECURSION_DEPTH {
        ctx.guard.borrow_mut().remove(&key);
        return false;
    }
    let result = matches(&ctx.defs[id], value, ctx);
    ctx.guard.borrow_mut().remove(&key);
    result
}

fn check_intersection(
    members: &[Schema],
    value: &Bound<'_, PyAny>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) {
    // Every member must hold; collect each member's failure.
    for member in members {
        check(member, value, path, ctx, out);
        if aborted(ctx, out) {
            return;
        }
    }
}

fn check_complement(
    inner: &Schema,
    value: &Bound<'_, PyAny>,
    path: &[PathSegment],
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) {
    // The inner result is discarded either way, so decide membership on the
    // fast path; a value that matches the inner schema fails the complement.
    if matches(inner, &Value::Py(value), ctx) {
        out.push(Violation {
            code: "unexpected_match",
            path: path.to_vec(),
            expected: format!("not {}", inner.expected()),
            value_summary: summarize(value),
        });
    }
}

fn check_refine(
    base: &Schema,
    constraints: &[Constraint],
    value: &Bound<'_, PyAny>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) {
    // Constraints narrow the base set, so they are meaningful only on a base
    // member: if the base fails, report that and do not run the constraints.
    let before = out.len();
    check(base, value, path, ctx, out);
    if out.len() != before {
        return;
    }
    for constraint in constraints {
        check_constraint(constraint, value, path, ctx, out);
        if aborted(ctx, out) {
            return;
        }
    }
}

fn check_constraint(
    constraint: &Constraint,
    value: &Bound<'_, PyAny>,
    path: &[PathSegment],
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) {
    let py = value.py();
    let (ok, code, expected) = match constraint {
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
    if !ok {
        out.push(Violation {
            code,
            path: path.to_vec(),
            expected,
            value_summary: summarize(value),
        });
    }
}

/// Whether `value` satisfies `constraint`, for the membership fast path.
fn constraint_holds(constraint: &Constraint, value: &Bound<'_, PyAny>, ctx: Ctx<'_>) -> bool {
    let py = value.py();
    match constraint {
        Constraint::Ge(i) => cmp(value.ge(ctx.pool[*i].bind(py))),
        Constraint::Gt(i) => cmp(value.gt(ctx.pool[*i].bind(py))),
        Constraint::Le(i) => cmp(value.le(ctx.pool[*i].bind(py))),
        Constraint::Lt(i) => cmp(value.lt(ctx.pool[*i].bind(py))),
        Constraint::MinLen(n) => value.len().is_ok_and(|len| len >= *n),
        Constraint::MaxLen(n) => value.len().is_ok_and(|len| len <= *n),
        Constraint::MultipleOf(i) => is_multiple_of(value, ctx.pool[*i].bind(py)),
        Constraint::Predicate(i) => predicate_passes(ctx.pool[*i].bind(py), value).unwrap_or(false),
    }
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

fn check_instance(
    index: usize,
    value: &Bound<'_, PyAny>,
    path: &[PathSegment],
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) {
    let class = ctx.pool[index].bind(value.py());
    if !value.is_instance(class).unwrap_or(false) {
        out.push(type_mismatch(
            "instance_type",
            &class_label(class),
            value,
            path,
        ));
    }
}

fn check_object(
    class_index: usize,
    fields: &[Field],
    value: &Bound<'_, PyAny>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) {
    let class = ctx.pool[class_index].bind(value.py());
    if !value.is_instance(class).unwrap_or(false) {
        // Not an instance: the attribute checks below cannot be trusted.
        out.push(type_mismatch(
            "instance_type",
            &class_label(class),
            value,
            path,
        ));
        return;
    }
    for field in fields {
        match value.getattr(field.name.as_str()) {
            Ok(attr) => {
                path.push(PathSegment::Key(field.name.clone()));
                check(&field.schema, &attr, path, ctx, out);
                path.pop();
            }
            Err(_) => out.push(located(
                path,
                field.name.clone(),
                "missing_attribute",
                format!("attribute {:?}", field.name),
                "missing".to_owned(),
            )),
        }
        if aborted(ctx, out) {
            return;
        }
    }
}

fn check_union(
    members: &[Schema],
    value: &Bound<'_, PyAny>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) {
    // A value is a member iff it matches at least one branch; decide that on the
    // fast path.
    if members
        .iter()
        .any(|member| matches(member, &Value::Py(value), ctx))
    {
        return;
    }
    // No branch matches. Explain the *closest* branch — the one that descended
    // furthest into the value before failing — by reporting its aggregated
    // failures, rather than dumping every branch's noise. "Furthest" is the
    // greatest path depth past the union's own location. When no branch makes
    // any progress (every branch is a flat type mismatch, e.g. `int | str`
    // against a float), fall back to a single union error. Sub-checks aggregate
    // regardless of `fail_fast` so both the deepest progress and the full branch
    // detail are visible; this runs only on the error path, never on a match.
    let base_depth = path.len();
    let probe = Ctx {
        fail_fast: false,
        ..ctx
    };
    let mut best: Option<(usize, Vec<Violation>)> = None;
    for member in members {
        let mut branch = Vec::new();
        check(member, value, path, probe, &mut branch);
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
                value_summary: summarize(value),
            });
        }
    }
}

fn admit(
    ok: bool,
    schema: &Schema,
    value: &Bound<'_, PyAny>,
    path: &[PathSegment],
    out: &mut Vec<Violation>,
) {
    if !ok {
        out.push(mismatch(schema, value, path));
    }
}

fn mismatch(schema: &Schema, value: &Bound<'_, PyAny>, path: &[PathSegment]) -> Violation {
    Violation {
        code: schema.error_code(),
        path: path.to_vec(),
        expected: schema.expected().to_owned(),
        value_summary: summarize(value),
    }
}

fn type_mismatch(
    code: &'static str,
    expected: &str,
    value: &Bound<'_, PyAny>,
    path: &[PathSegment],
) -> Violation {
    Violation {
        code,
        path: path.to_vec(),
        expected: expected.to_owned(),
        value_summary: summarize(value),
    }
}

/// Whether `value` is the typed singleton denoted by `literal`: same type and
/// equal. The same-type guard rules out Python's cross-type equality
/// (`1 == True == 1.0`), so `Literal[1]` denotes `{1}`, not `{1, True, 1.0}`.
fn literal_matches(value: &Bound<'_, PyAny>, literal: &Bound<'_, PyAny>) -> bool {
    value.get_type().is(literal.get_type()) && value.eq(literal).unwrap_or(false)
}

fn check_literal(
    index: usize,
    value: &Bound<'_, PyAny>,
    path: &[PathSegment],
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) {
    let literal = ctx.pool[index].bind(value.py());
    if !literal_matches(value, literal) {
        out.push(Violation {
            code: "literal_value",
            path: path.to_vec(),
            expected: format!("the literal {}", summarize(literal)),
            value_summary: summarize(value),
        });
    }
}

fn check_sequence(
    element: &Schema,
    value: &Bound<'_, PyAny>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) {
    let Ok(list) = value.cast::<PyList>() else {
        out.push(type_mismatch("list_type", "list", value, path));
        return;
    };
    for (index, item) in list.iter().enumerate() {
        path.push(PathSegment::Index(index));
        check(element, &item, path, ctx, out);
        path.pop();
        if aborted(ctx, out) {
            return;
        }
    }
}

fn check_fixed_sequence(
    elements: &[Schema],
    value: &Bound<'_, PyAny>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) {
    let Ok(list) = value.cast::<PyList>() else {
        out.push(type_mismatch("list_type", "list", value, path));
        return;
    };
    if list.len() != elements.len() {
        // Length governs the positional match; without it the per-index checks
        // are meaningless, so this is terminal.
        out.push(Violation {
            code: "list_length",
            path: path.clone(),
            expected: format!("list of length {}", elements.len()),
            value_summary: summarize(value),
        });
        return;
    }
    for (index, (schema, item)) in elements.iter().zip(list.iter()).enumerate() {
        path.push(PathSegment::Index(index));
        check(schema, &item, path, ctx, out);
        path.pop();
        if aborted(ctx, out) {
            return;
        }
    }
}

fn check_tuple(
    elements: &[Schema],
    value: &Bound<'_, PyAny>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) {
    let Ok(tuple) = value.cast::<PyTuple>() else {
        out.push(type_mismatch("tuple_type", "tuple", value, path));
        return;
    };
    if tuple.len() != elements.len() {
        out.push(Violation {
            code: "tuple_length",
            path: path.clone(),
            expected: format!("tuple of length {}", elements.len()),
            value_summary: summarize(value),
        });
        return;
    }
    for (index, (schema, item)) in elements.iter().zip(tuple.iter()).enumerate() {
        path.push(PathSegment::Index(index));
        check(schema, &item, path, ctx, out);
        path.pop();
        if aborted(ctx, out) {
            return;
        }
    }
}

fn check_variadic_tuple(
    element: &Schema,
    value: &Bound<'_, PyAny>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) {
    let Ok(tuple) = value.cast::<PyTuple>() else {
        out.push(type_mismatch("tuple_type", "tuple", value, path));
        return;
    };
    for (index, item) in tuple.iter().enumerate() {
        path.push(PathSegment::Index(index));
        check(element, &item, path, ctx, out);
        path.pop();
        if aborted(ctx, out) {
            return;
        }
    }
}

fn check_set(
    element: &Schema,
    value: &Bound<'_, PyAny>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) {
    let Ok(set) = value.cast::<PySet>() else {
        out.push(type_mismatch("set_type", "set", value, path));
        return;
    };
    // Set order is not meaningful, so element failures carry no index segment.
    for item in set.iter() {
        check(element, &item, path, ctx, out);
        if aborted(ctx, out) {
            return;
        }
    }
}

fn check_frozenset(
    element: &Schema,
    value: &Bound<'_, PyAny>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) {
    let Ok(set) = value.cast::<PyFrozenSet>() else {
        out.push(type_mismatch("frozenset_type", "frozenset", value, path));
        return;
    };
    for item in set.iter() {
        check(element, &item, path, ctx, out);
        if aborted(ctx, out) {
            return;
        }
    }
}

fn check_mapping(
    key_schema: &Schema,
    value_schema: &Schema,
    value: &Bound<'_, PyAny>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) {
    let Ok(dict) = value.cast::<PyDict>() else {
        out.push(type_mismatch("dict_type", "dict", value, path));
        return;
    };
    for (key, val) in dict.iter() {
        path.push(PathSegment::Key(key_label(&key)));
        check(key_schema, &key, path, ctx, out);
        check(value_schema, &val, path, ctx, out);
        path.pop();
        if aborted(ctx, out) {
            return;
        }
    }
}

fn check_record(
    fields: &[Field],
    open: bool,
    value: &Bound<'_, PyAny>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) {
    let Ok(dict) = value.cast::<PyDict>() else {
        out.push(type_mismatch("dict_type", "dict", value, path));
        return;
    };

    // Declared fields, in declared order: present values are checked, absent
    // required keys fail. Each field's failure is collected independently.
    for field in fields {
        match dict.get_item(field.name.as_str()) {
            Ok(Some(item)) => {
                path.push(PathSegment::Key(field.name.clone()));
                check(&field.schema, &item, path, ctx, out);
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
        if aborted(ctx, out) {
            return;
        }
    }

    // An open (lax) record admits undeclared keys; a closed one collects each.
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
            if aborted(ctx, out) {
                return;
            }
        }
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

/// A short, printable label for a mapping key, used in error paths.
fn key_label(key: &Bound<'_, PyAny>) -> String {
    match key.str() {
        Ok(text) => truncate(&text.to_string(), 40),
        Err(_) => summarize(key),
    }
}
