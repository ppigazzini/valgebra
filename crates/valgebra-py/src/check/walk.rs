//! The validation walk: one membership test of a value against the IR.
//!
//! [`member`] is the single walk. It returns whether the value belongs to the
//! schema's set, and in *explain* mode (`ctx.explain`) it also aggregates a
//! [`Violation`] for each independent failure into `out` (each record field,
//! each sequence element, each mapping entry), unless `ctx.fail_fast` stops it at
//! the first. In *fast* mode it allocates nothing and short-circuits as soon as
//! membership is decided.

use std::borrow::Cow;

use jiter::JsonValue;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFrozenSet, PyList, PySet, PyString, PyTuple};
use rustc_hash::{FxHashMap, FxHashSet};
use valgebra_core::{Constraint, Field, PathSegment, Schema, SeqKind, SeqRegex, Violation};

use crate::check::Ctx;
use crate::check::index::compile_pattern;
use crate::check::violation::{
    key_label, located, mismatch, summarize_value, type_fail, type_mismatch,
};
use crate::errors::{class_label, summarize};
use crate::input::Value;

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
        Schema::Anything | Schema::Dynamic => true,
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
        Schema::Seq { container, regex } => check_seq(*container, regex, value, path, ctx, out),
        Schema::Set(element) => check_set(element, value, path, ctx, out),
        Schema::FrozenSet(element) => check_frozenset(element, value, path, ctx, out),
        Schema::KeyedMap { fields, defaults } => {
            // Membership is the single-pass fast check; on failure the explain
            // pass re-walks in declared order to aggregate ordered violations.
            let ok = keyed_map_matches(fields, defaults, value, ctx);
            if !ok && ctx.explain {
                keyed_map_explain(fields, defaults, value, path, ctx, out);
            }
            ok
        }
        Schema::Union(members) => check_union(members, value, path, ctx, out),
        Schema::Intersection(members) => check_intersection(members, value, path, ctx, out),
        Schema::Complement(inner) => check_complement(inner, value, path, ctx, out),
        Schema::Instance(index) => check_instance(*index, value, path, ctx, out),
        Schema::Attrs {
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
            code: "literal_error",
            path: path.to_vec(),
            expected: format!("the literal {}", summarize(literal)),
            value_summary: summarize_value(value),
        });
    }
    ok
}

/// Membership for a sequence node: the value is a list or tuple whose element
/// sequence matches the regex. The frontend emits only *linear* regexes — a
/// fixed positional prefix then an optional repeated tail — so the elements are
/// walked lazily with no automaton and no collection, identical in cost to a
/// direct positional or homogeneous check. JSON arrays are lists.
fn check_seq(
    container: SeqKind,
    regex: &SeqRegex,
    value: &Value<'_, '_>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
    let (kind_word, type_code, len_code) = match container {
        SeqKind::List => ("list", "list_type", "list_length"),
        SeqKind::Tuple => ("tuple", "tuple_type", "tuple_length"),
    };
    let Some((prefix, tail)) = regex.linear() else {
        // Alternation and nesting are built only inside the decision procedure;
        // such a regex never reaches value membership.
        return false;
    };
    match (container, value) {
        (SeqKind::List, Value::Py(v)) => {
            let Ok(list) = v.cast::<PyList>() else {
                return type_fail(type_code, kind_word, value, path, ctx, out);
            };
            if !seq_len_ok(list.len(), prefix.len(), tail.is_some()) {
                return seq_length_fail(len_code, kind_word, &prefix, tail, value, path, ctx, out);
            }
            let mut ok = true;
            for (i, item) in list.iter().enumerate() {
                ok &= seq_element(&prefix, tail, i, &Value::Py(&item), path, ctx, out);
                if !ok && stop(ctx) {
                    return false;
                }
            }
            ok
        }
        (SeqKind::List, Value::Json(py, JsonValue::Array(items))) => {
            if !seq_len_ok(items.len(), prefix.len(), tail.is_some()) {
                return seq_length_fail(len_code, kind_word, &prefix, tail, value, path, ctx, out);
            }
            let mut ok = true;
            for (i, item) in items.iter().enumerate() {
                ok &= seq_element(&prefix, tail, i, &Value::Json(*py, item), path, ctx, out);
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
            if !seq_len_ok(tuple.len(), prefix.len(), tail.is_some()) {
                return seq_length_fail(len_code, kind_word, &prefix, tail, value, path, ctx, out);
            }
            let mut ok = true;
            for (i, item) in tuple.iter().enumerate() {
                ok &= seq_element(&prefix, tail, i, &Value::Py(&item), path, ctx, out);
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

/// Whether the element count fits the regex: exactly the prefix length with no
/// tail, or at least the prefix length when a repeated tail follows.
fn seq_len_ok(len: usize, prefix_len: usize, has_tail: bool) -> bool {
    if has_tail {
        len >= prefix_len
    } else {
        len == prefix_len
    }
}

/// Match one element at position `i`: the prefix schema at `i`, or the repeated
/// tail past the prefix. The index segment is pushed only in explain mode.
fn seq_element(
    prefix: &[&Schema],
    tail: Option<&Schema>,
    i: usize,
    item: &Value<'_, '_>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
    let schema = prefix
        .get(i)
        .copied()
        .unwrap_or_else(|| tail.expect("length already checked"));
    if ctx.explain {
        path.push(PathSegment::Index(i));
    }
    let ok = member(schema, item, path, ctx, out);
    if ctx.explain {
        path.pop();
    }
    ok
}

/// A sequence-length mismatch: terminal, since the positional match is then
/// meaningless. A tailless regex wants an exact length; a tailed one a minimum.
#[allow(clippy::too_many_arguments)]
fn seq_length_fail(
    len_code: &'static str,
    kind_word: &str,
    prefix: &[&Schema],
    tail: Option<&Schema>,
    value: &Value<'_, '_>,
    path: &[PathSegment],
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
    if ctx.explain {
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
        return type_fail("frozen_set_type", "frozenset", value, path, ctx, out);
    };
    let Ok(set) = v.cast::<PyFrozenSet>() else {
        return type_fail("frozen_set_type", "frozenset", value, path, ctx, out);
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

/// Membership for a keyed map: named fields, then a default clause for every
/// other key. The walk is inverted — it visits each entry once — and a JSON
/// object's keys are strings, a duplicate keeping its last value as
/// `json.loads` does.
fn keyed_map_matches(
    fields: &[Field],
    defaults: &[(Schema, Schema)],
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
fn covered(
    defaults: &[(Schema, Schema)],
    key: &Value<'_, '_>,
    val: &Value<'_, '_>,
    ctx: Ctx<'_>,
) -> bool {
    let sub = fast(ctx);
    defaults.iter().any(|(key_schema, value_schema)| {
        member(key_schema, key, &mut Vec::new(), sub, &mut Vec::new())
            && member(value_schema, val, &mut Vec::new(), sub, &mut Vec::new())
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
    defaults: &[(Schema, Schema)],
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
    defaults: &[(Schema, Schema)],
    dict: &Bound<'_, PyDict>,
    ctx: Ctx<'_>,
    mut required_remaining: usize,
    lookup: impl Fn(&str) -> Option<usize>,
) -> bool {
    let sub = fast(ctx);
    for (key, val) in dict.iter() {
        // A non-string key, or a string carrying a lone surrogate (which cannot
        // equal a field name, since names are valid UTF-8 by build-time check),
        // resolves to no field and must instead be covered by a default clause.
        let index = key
            .cast::<PyString>()
            .ok()
            .and_then(|s| s.to_str().ok())
            .and_then(&lookup);
        match index {
            Some(i) => {
                let field = &fields[i];
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
            None => {
                if !covered(defaults, &Value::Py(&key), &Value::Py(&val), ctx) {
                    return false;
                }
            }
        }
    }
    required_remaining == 0
}

/// The keyed-map fast path over a JSON object. Keys are strings; a duplicate key
/// keeps its last value (a reverse find), as `json.loads` does. Records are
/// small, so a linear scan beats building a per-object map.
fn keyed_map_matches_json(
    fields: &[Field],
    defaults: &[(Schema, Schema)],
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
    for (i, (key, val)) in entries.iter().enumerate() {
        if fields.iter().any(|f| f.name == key.as_ref()) {
            continue;
        }
        if entries[i + 1..].iter().any(|(later, _)| later == key) {
            continue;
        }
        let key_value = JsonValue::Str(Cow::Borrowed(key.as_ref()));
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
    defaults: &[(Schema, Schema)],
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
        if ctx.fail_fast && !out.is_empty() {
            return;
        }
    }
    for (key, val) in dict.iter() {
        if let Some(name) = key.cast::<PyString>().ok().and_then(|s| s.to_str().ok())
            && declared.contains(name)
        {
            continue;
        }
        if covered(defaults, &Value::Py(&key), &Value::Py(&val), ctx) {
            continue;
        }
        if let Some((key_schema, value_schema)) = defaults.first() {
            // A clause exists but did not cover this key: surface the key and
            // value violations against it (the homogeneous-mapping error).
            path.push(PathSegment::Key(key_label(&key)));
            member(key_schema, &Value::Py(&key), path, ctx, out);
            member(value_schema, &Value::Py(&val), path, ctx, out);
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
        if ctx.fail_fast && !out.is_empty() {
            return;
        }
    }
}

/// Cap on how many branches the closest-branch error probe re-walks. The
/// membership decision has already scanned every branch to confirm non-matching;
/// this bounds the *second*, explain-mode pass so building the error for a
/// pathologically wide union (a large `Literal[...]`, say) stays linear in the
/// cap rather than the branch count. Beyond the cap the report falls back to the
/// union summary. Error-path only — the membership result is never affected.
const CLOSEST_BRANCH_PROBE_LIMIT: usize = 64;

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
    if !ctx.explain
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
                "multiple_of",
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
                .is_some_and(|text| match ctx.regexes.get(pattern) {
                    Some(compiled) => compiled.is_match(text),
                    None => compile_pattern(pattern).is_ok_and(|re| re.is_match(text)),
                });
            (
                matched,
                "string_pattern_mismatch",
                format!("a string matching {pattern:?}"),
            )
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

/// Whether `value` is the typed singleton denoted by `literal`: same type and
/// equal. The same-type guard rules out Python's cross-type equality
/// (`1 == True == 1.0`), so `Literal[1]` denotes `{1}`, not `{1, True, 1.0}`.
pub(crate) fn literal_matches(value: &Bound<'_, PyAny>, literal: &Bound<'_, PyAny>) -> bool {
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
