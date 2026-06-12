//! Render a compiled schema back to a readable annotation/combinator string.

use std::cell::RefCell;
use std::collections::HashSet;

use pyo3::prelude::*;
use valgebra_core::{Constraint, Field, Schema};

use crate::errors::{class_label, summarize};

/// Render a schema back to the annotation/combinator expression that produces
/// it. A recursive `Ref` is unfolded once; the back edge to a reference already
/// being rendered shows as `...`, so the printed form stays finite.
pub(crate) fn render(
    py: Python<'_>,
    schema: &Schema,
    pool: &[Py<PyAny>],
    defs: &[Schema],
    active: &RefCell<HashSet<usize>>,
) -> String {
    let r = |s: &Schema| render(py, s, pool, defs, active);
    let kids = |members: &[Schema]| members.iter().map(&r).collect::<Vec<_>>().join(", ");
    match schema {
        Schema::Anything => "anything".to_owned(),
        Schema::Any => "Any".to_owned(),
        Schema::Nothing => "nothing".to_owned(),
        Schema::NoneType => "None".to_owned(),
        Schema::Bool => "bool".to_owned(),
        Schema::Int => "int".to_owned(),
        Schema::Float => "float".to_owned(),
        Schema::Str => "str".to_owned(),
        Schema::Bytes => "bytes".to_owned(),
        Schema::Literal(i) => format!("Literal[{}]", pool_repr(py, pool, *i)),
        Schema::Sequence(e) => format!("list[{}]", r(e)),
        Schema::FixedSequence(es) => format!("[{}]", kids(es)),
        Schema::Tuple(es) => format!("tuple[{}]", kids(es)),
        Schema::VariadicTuple(e) => format!("tuple[{}, ...]", r(e)),
        Schema::Set(e) => format!("set[{}]", r(e)),
        Schema::FrozenSet(e) => format!("frozenset[{}]", r(e)),
        Schema::Mapping { key, value } => format!("dict[{}, {}]", r(key), r(value)),
        Schema::Record { fields } => render_record(py, fields, pool, defs, active),
        Schema::Union(members) => members.iter().map(&r).collect::<Vec<_>>().join(" | "),
        Schema::Intersection(members) => format!("intersect({})", kids(members)),
        Schema::Complement(inner) => format!("complement({})", r(inner)),
        Schema::Instance(i) | Schema::Object { class_index: i, .. } => {
            pool_class_name(py, pool, *i)
        }
        Schema::Refine { base, constraints } => {
            let mut parts = vec![r(base)];
            parts.extend(constraints.iter().map(|c| render_constraint(py, c, pool)));
            format!("Annotated[{}]", parts.join(", "))
        }
        Schema::Ref(id) => {
            // Unfold the definition once; a back-edge to a reference already
            // being rendered shows as `...`, so the form stays finite.
            if !active.borrow_mut().insert(*id) {
                return "...".to_owned();
            }
            let body = defs.get(*id).map_or_else(|| "...".to_owned(), &r);
            active.borrow_mut().remove(id);
            body
        }
        Schema::SelfRef(_) => "...".to_owned(),
    }
}

fn render_record(
    py: Python<'_>,
    fields: &[Field],
    pool: &[Py<PyAny>],
    defs: &[Schema],
    active: &RefCell<HashSet<usize>>,
) -> String {
    let entries: Vec<String> = fields
        .iter()
        .map(|field| {
            let suffix = if field.required { "" } else { "?" };
            format!(
                "'{}{}': {}",
                field.name,
                suffix,
                render(py, &field.schema, pool, defs, active)
            )
        })
        .collect();
    format!("{{{}}}", entries.join(", "))
}

fn render_constraint(py: Python<'_>, constraint: &Constraint, pool: &[Py<PyAny>]) -> String {
    match constraint {
        Constraint::Ge(i) => format!("Ge({})", pool_repr(py, pool, *i)),
        Constraint::Gt(i) => format!("Gt({})", pool_repr(py, pool, *i)),
        Constraint::Le(i) => format!("Le({})", pool_repr(py, pool, *i)),
        Constraint::Lt(i) => format!("Lt({})", pool_repr(py, pool, *i)),
        Constraint::MinLen(n) => format!("MinLen({n})"),
        Constraint::MaxLen(n) => format!("MaxLen({n})"),
        Constraint::Predicate(_) => "Predicate(...)".to_owned(),
    }
}

fn pool_repr(py: Python<'_>, pool: &[Py<PyAny>], index: usize) -> String {
    summarize(pool[index].bind(py))
}

fn pool_class_name(py: Python<'_>, pool: &[Py<PyAny>], index: usize) -> String {
    class_label(pool[index].bind(py))
}
