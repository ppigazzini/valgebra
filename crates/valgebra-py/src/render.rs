//! Render a compiled schema back to a readable annotation/combinator string.

use std::cell::RefCell;

use pyo3::prelude::*;
use rustc_hash::FxHashMap;
use valgebra_core::{CollKind, Constraint, DefIx, Field, MapClause, Schema, SeqKind, Spelling};

use crate::errors::{class_label, summarize};

/// The deepest render recursion before the walk stops and prints `...`. A cycle
/// guard already bounds a single recursive definition, but a chain of *distinct*
/// definitions, or a legitimately deep tree, recurses one native stack frame per
/// level with a string allocation each, the heaviest per-level frame in the
/// crate. This counter is the render path's own stack-safety guarantee: the
/// construction bounds keep real schemas well under it, and a chain past it
/// prints an ellipsis rather than overflowing the stack. It sits above the
/// schema-depth bound (a legal tree renders in full) and far enough below the
/// smallest platform thread stack to hold on it.
const MAX_RENDER_DEPTH: usize = 200;

/// Render a schema back to the annotation/combinator expression that produces
/// it. A recursive `Ref` renders as the `recursive` call that builds it, with
/// the back edge as the lambda's own parameter, so the printed form is finite
/// *and* rebuilds the schema. `depth` is the current recursion level; past
/// [`MAX_RENDER_DEPTH`] the walk prints `...`, which is the one form that does
/// not rebuild -- a pathological definition chain would otherwise overflow the
/// native stack, and a truncated render says so by being unreadable as Python.
pub(crate) fn render(
    py: Python<'_>,
    schema: &Schema,
    pool: &[Py<PyAny>],
    defs: &[Schema],
    active: &RefCell<FxHashMap<DefIx, String>>,
    depth: usize,
) -> String {
    if depth > MAX_RENDER_DEPTH {
        return "...".to_owned();
    }
    let r = |s: &Schema| render(py, s, pool, defs, active, depth + 1);
    match schema {
        // The one place the spelling is read: `typing.Any` and `anything` are
        // the same set and the same node, and this gives back what was written.
        Schema::Anything(Spelling::Any) => "Any".to_owned(),
        Schema::Anything(Spelling::Top) => "anything".to_owned(),
        Schema::Nothing => "nothing".to_owned(),
        Schema::NoneType => "None".to_owned(),
        Schema::Bool => "bool".to_owned(),
        Schema::Int => "int".to_owned(),
        Schema::Float => "float".to_owned(),
        Schema::Str => "str".to_owned(),
        Schema::Bytes => "bytes".to_owned(),
        Schema::Literal(i) => format!("Literal[{}]", pool_repr(py, pool, i.get())),
        Schema::Seq { container, shape } => {
            let list = matches!(container, SeqKind::List);
            match (shape.prefix.as_slice(), shape.tail.as_deref()) {
                // Homogeneous: list[T] / tuple[T, ...].
                ([], Some(t)) if list => format!("list[{}]", r(t)),
                ([], Some(t)) => format!("tuple[{}, ...]", r(t)),
                // Fixed positional: [A, B] / tuple[A, B].
                (ps, None) => {
                    let body = ps.iter().map(r).collect::<Vec<_>>().join(", ");
                    if list {
                        format!("[{body}]")
                    } else if ps.is_empty() {
                        // The nullary product, whose one member is `()`. Python
                        // spells the empty subscript `tuple[()]`; `tuple[]` is
                        // not an expression at all.
                        "tuple[()]".to_owned()
                    } else {
                        format!("tuple[{body}]")
                    }
                }
                // Fixed prefix then a repeated tail.
                (ps, Some(t)) => {
                    let mut parts: Vec<String> = ps.iter().map(r).collect();
                    parts.push(r(t));
                    parts.push("...".to_owned());
                    let body = parts.join(", ");
                    if list {
                        format!("[{body}]")
                    } else {
                        format!("tuple[{body}]")
                    }
                }
            }
        }
        Schema::Coll { container, element } => match container {
            CollKind::Set => format!("set[{}]", r(element)),
            CollKind::FrozenSet => format!("frozenset[{}]", r(element)),
        },
        Schema::KeyedMap { fields, defaults } => {
            render_keyed_map(py, fields, defaults, pool, defs, active, depth)
        }
        Schema::Union(members) => members.iter().map(&r).collect::<Vec<_>>().join(" | "),
        Schema::Intersection(members) => {
            render_meet(py, schema, members, pool, defs, active, depth)
        }
        Schema::Complement(inner) => format!("complement({})", r(inner)),
        Schema::Instance(i) => pool_class_name(py, pool, i.get()),
        Schema::AttrRecord { fields } => render_attr_record(py, fields, pool, defs, active, depth),
        Schema::Refine { base, constraints } => {
            let mut parts = vec![r(base)];
            parts.extend(constraints.iter().map(|c| render_constraint(py, c, pool)));
            format!("Annotated[{}]", parts.join(", "))
        }
        Schema::Ref(id) => {
            // A back edge into a definition already being rendered is the
            // lambda's parameter, which is what makes the form finite without
            // an ellipsis nothing can rebuild.
            if let Some(name) = active.borrow().get(id) {
                return name.clone();
            }
            let name = binder(active.borrow().len());
            active.borrow_mut().insert(*id, name.clone());
            let body = defs.get(id.get()).map_or_else(|| "...".to_owned(), &r);
            active.borrow_mut().remove(id);
            format!("recursive(lambda {name}: {body})")
        }
        // The transient marker `recursive` uses while its own body is being
        // built. A compiled validator holds no such node, so nothing a caller
        // can print reaches this; it renders as an ellipsis because there is no
        // definition to name yet.
        Schema::SelfRef(_) => "...".to_owned(),
    }
}

/// The name a recursive definition's back edge is bound to, by how many are
/// already open.
///
/// Single letters while they last, because one definition is the ordinary case
/// and `X` reads as a type variable does. Past them the letter carries the depth,
/// which keeps two nested definitions apart -- a name reused inside its own
/// scope would render a schema that rebuilds a different one.
fn binder(open: usize) -> String {
    match open {
        0 => "X".to_owned(),
        1 => "Y".to_owned(),
        2 => "Z".to_owned(),
        n => format!("T{n}"),
    }
}

/// Render a meet, reading a class's `isinstance` atom beside its attribute
/// record back as the class name the user wrote.
///
/// The pair is [`Schema::object_class`]; the members that are not the pair
/// render as themselves, so a meet that flattened others in beside a class still
/// names the class. Without this the annotation `Pt` would print as
/// `intersection(Pt, object(x=int))` -- the algebra's spelling of a thing the
/// user spelled with one name.
fn render_meet(
    py: Python<'_>,
    schema: &Schema,
    members: &[Schema],
    pool: &[Py<PyAny>],
    defs: &[Schema],
    active: &RefCell<FxHashMap<DefIx, String>>,
    depth: usize,
) -> String {
    let r = |s: &Schema| render(py, s, pool, defs, active, depth + 1);
    let Some(class) = schema.object_class() else {
        let kids = members.iter().map(r).collect::<Vec<_>>().join(", ");
        return format!("intersection({kids})");
    };
    let name = pool_class_name(py, pool, class.get());
    let mut parts = vec![name];
    parts.extend(
        members
            .iter()
            .filter(|m| !matches!(m, Schema::Instance(_) | Schema::AttrRecord { .. }))
            .map(r),
    );
    if parts.len() == 1 {
        parts.remove(0)
    } else {
        format!("intersection({})", parts.join(", "))
    }
}

/// Render an attribute record on its own: `object(x=int)`.
///
/// No annotation builds a record without a class beside it, and no combinator
/// takes one apart, so nothing a caller can write reaches this today. It is here
/// because `render` is total over the IR and the node is in it: the day a
/// spelling arrives, the form is readable rather than re-parsable, like the
/// `{...}` open-record marker.
fn render_attr_record(
    py: Python<'_>,
    fields: &[Field],
    pool: &[Py<PyAny>],
    defs: &[Schema],
    active: &RefCell<FxHashMap<DefIx, String>>,
    depth: usize,
) -> String {
    let entries: Vec<String> = fields
        .iter()
        .map(|field| {
            let suffix = if field.required { "" } else { "?" };
            let schema = render(py, &field.schema, pool, defs, active, depth + 1);
            format!("{}{}={}", field.name, suffix, schema)
        })
        .collect();
    format!("object({})", entries.join(", "))
}

fn render_keyed_map(
    py: Python<'_>,
    fields: &[Field],
    defaults: &[MapClause],
    pool: &[Py<PyAny>],
    defs: &[Schema],
    active: &RefCell<FxHashMap<DefIx, String>>,
    depth: usize,
) -> String {
    let r = |s: &Schema| render(py, s, pool, defs, active, depth + 1);
    // A pure mapping — no named fields, one clause — is dict[K, V].
    if fields.is_empty()
        && let [clause] = defaults
    {
        return format!("dict[{}, {}]", r(&clause.key), r(&clause.value));
    }
    // Otherwise a record/struct: named fields, then any catch-all clauses.
    let mut entries: Vec<String> = fields
        .iter()
        .map(|field| {
            let suffix = if field.required { "" } else { "?" };
            format!("'{}{}': {}", field.name, suffix, r(&field.schema))
        })
        .collect();
    for clause in defaults {
        // Every clause renders as the key-to-value entry it is, the catch-all
        // included. `{'a': int, ...}` read better and rebuilt a *different*
        // schema: `...` is a dict key like any other, so the frontend reads it
        // back as `Literal[Ellipsis]` and the record is closed with an odd
        // field rather than open.
        entries.push(format!("{}: {}", r(&clause.key), r(&clause.value)));
    }
    format!("{{{}}}", entries.join(", "))
}

fn render_constraint(py: Python<'_>, constraint: &Constraint, pool: &[Py<PyAny>]) -> String {
    match constraint {
        Constraint::Ge(i) => format!("Ge({})", pool_repr(py, pool, i.get())),
        Constraint::Gt(i) => format!("Gt({})", pool_repr(py, pool, i.get())),
        Constraint::Le(i) => format!("Le({})", pool_repr(py, pool, i.get())),
        Constraint::Lt(i) => format!("Lt({})", pool_repr(py, pool, i.get())),
        Constraint::MinLen(n) => format!("MinLen({n})"),
        Constraint::MaxLen(n) => format!("MaxLen({n})"),
        Constraint::MultipleOf(i) => format!("MultipleOf({})", pool_repr(py, pool, i.get())),
        Constraint::Predicate(_) => "Predicate(...)".to_owned(),
        Constraint::Regex(pattern) => format!("Regex({pattern:?})"),
    }
}

// Bounds-check the pool rather than indexing directly: a corrupt pool index
// degrades to a placeholder in the rendered string instead of panicking across
// the FFI boundary, matching the defensive `.get` posture in the walk. The index
// is pool-valid by construction, so the miss is an invariant break — loud in
// debug, a recognisable placeholder in release.
fn pool_repr(py: Python<'_>, pool: &[Py<PyAny>], index: usize) -> String {
    if let Some(constant) = pool.get(index) {
        summarize(constant.bind(py))
    } else {
        debug_assert!(false, "pool index {index} out of range");
        "<unknown>".to_owned()
    }
}

fn pool_class_name(py: Python<'_>, pool: &[Py<PyAny>], index: usize) -> String {
    if let Some(class) = pool.get(index) {
        class_label(class.bind(py))
    } else {
        debug_assert!(false, "pool index {index} out of range");
        "<unknown>".to_owned()
    }
}
