//! Render a compiled schema back to a readable annotation/combinator string.

use std::cell::{Cell, RefCell};
use std::mem::{Discriminant, discriminant};

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyComplex, PyFloat, PyInt, PyString, PyType};
use pyo3::{PyTypeCheck, PyTypeInfo};
use rustc_hash::{FxHashMap, FxHashSet};
use valgebra_core::{
    Attribute, CollKind, Constraint, DefIx, Field, MapClause, Schema, SeqKind, Spelling,
};

use crate::build::declares_fields;
use crate::errors::{UNREPRESENTABLE, class_label, stored, unless_fatal};

/// The deepest render recursion before the walk stops and gives up. A cycle
/// guard already bounds a single recursive definition, but a chain of *distinct*
/// definitions, or a legitimately deep tree, recurses one native stack frame per
/// level with a string allocation each, the heaviest per-level frame in the
/// crate. This counter is the render path's own stack-safety guarantee: a chain
/// past it prints [`GAVE_UP`] rather than overflowing the stack. It sits above
/// the schema-depth bound, so a legal *tree* renders in full, and far enough
/// below the smallest platform thread stack to hold on it.
///
/// A chain of definitions reaches it: `MAX_DEFINITIONS` links of a few levels
/// each is more levels than this, and `tests/test_repr.py` asserts the step
/// across for three shapes of link.
const MAX_RENDER_DEPTH: usize = 200;

/// What the renderer prints where it has no expression to give: past
/// [`MAX_RENDER_DEPTH`], for a `Ref` naming no definition, and for the
/// transient self-reference a compiled validator never holds.
///
/// Not an ellipsis. A render either re-parses to the schema it came from or
/// names what it cannot spell -- a class by its name, a predicate as
/// `Predicate(...)`, a constant in angle brackets where its `repr` is not an
/// expression of it -- and `...` is valid Python inside a subscript, so a
/// truncated one would parse and build a *different* validator with nothing to
/// say it had been truncated. This form is a syntax error wherever it lands,
/// which is what makes a lossy render announce itself. `tuple[T, ...]` is
/// unrelated: that ellipsis is the annotation's own spelling and rebuilds
/// exactly.
const GAVE_UP: &str = "<...>";

/// The two characters around a binder's mark while a render is built, with the
/// number of definitions open around the back edge between them.
///
/// Control characters, because a printed name or repr is the one place either
/// can come from, and both are what [`name_binders`] reads before it replaces a
/// mark.
const MARK_OPEN: char = '\u{1}';
const MARK_CLOSE: char = '\u{2}';

/// Render a schema back to the annotation/combinator expression that produces
/// it. A recursive `Ref` renders as the `recursive` call that builds it, with
/// the back edge as the lambda's own parameter, so the printed form is finite
/// *and* rebuilds the schema. Past [`MAX_RENDER_DEPTH`] the walk prints
/// [`GAVE_UP`] -- a chain of definitions would otherwise overflow the native
/// stack, and a truncated render says so by being unreadable as Python.
///
/// The bound is reachable. A single annotation cannot be written deep enough --
/// the frontend refuses past `MAX_SCHEMA_DEPTH`, which is lower -- but a chain
/// of definitions composes: the render descends into each in turn, so a hundred
/// shallow links reach a depth no one annotation can.
///
/// A constant or a class is printed by its own `__repr__` or `__name__`, and a
/// fatal signal one of them raises propagates, as it does from every other
/// question valgebra asks of user code.
pub(crate) fn render(
    py: Python<'_>,
    schema: &Schema,
    pool: &[Py<PyAny>],
    defs: &[Schema],
) -> PyResult<String> {
    let binders = Binders::default();
    let text = render_at(py, schema, pool, defs, &binders, 0)?;
    name_binders(py, text, binders.printed.get())
}

/// The definitions open while a render descends, by how many enclose each, and
/// how many binder marks it has printed.
#[derive(Default)]
struct Binders {
    open: RefCell<FxHashMap<DefIx, usize>>,
    printed: Cell<usize>,
}

impl Binders {
    /// The mark for the binder of a definition with `level` others open around
    /// it.
    fn mark(&self, level: usize) -> String {
        self.printed.set(self.printed.get() + 1);
        format!("{MARK_OPEN}{level}{MARK_CLOSE}")
    }
}

/// [`render`] at a recursion level, `depth`, printing each binder as its mark.
fn render_at(
    py: Python<'_>,
    schema: &Schema,
    pool: &[Py<PyAny>],
    defs: &[Schema],
    active: &Binders,
    depth: usize,
) -> PyResult<String> {
    if depth > MAX_RENDER_DEPTH {
        return Ok(GAVE_UP.to_owned());
    }
    let r = |s: &Schema| render_at(py, s, pool, defs, active, depth + 1);
    Ok(match schema {
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
        Schema::Literal(i) => format!("Literal[{}]", literal_repr(py, pool, i.get())?),
        Schema::Seq { container, shape } => {
            let list = matches!(container, SeqKind::List);
            match (&shape.prefix[..], shape.tail.as_deref()) {
                // Homogeneous: list[T] / tuple[T, ...].
                ([], Some(t)) if list => format!("list[{}]", r(t)?),
                ([], Some(t)) => format!("tuple[{}, ...]", r(t)?),
                // Fixed positional: [A, B] / tuple[A, B].
                (ps, None) => {
                    let body = ps.iter().map(r).collect::<PyResult<Vec<_>>>()?.join(", ");
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
                    let mut parts: Vec<String> = ps.iter().map(r).collect::<PyResult<_>>()?;
                    parts.push(r(t)?);
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
            CollKind::Set => format!("set[{}]", r(element)?),
            CollKind::FrozenSet => format!("frozenset[{}]", r(element)?),
        },
        Schema::KeyedMap { fields, defaults } => {
            render_keyed_map(py, fields, defaults, pool, defs, active, depth)?
        }
        Schema::Union(members) => render_union(members, &r)?,
        Schema::Intersection(members) => {
            render_meet(py, schema, members, pool, defs, active, depth)?
        }
        Schema::Complement(inner) => format!("complement({})", r(inner)?),
        Schema::Instance(i) => render_instance(py, pool, i.get())?,
        Schema::AttrRecord { fields } => render_attr_record(py, fields, pool, defs, active, depth)?,
        Schema::Refine { base, constraints } => {
            let mut markers = constraints
                .iter()
                .map(|constraint| {
                    Ok((
                        discriminant(constraint),
                        render_constraint(py, constraint, pool)?,
                    ))
                })
                .collect::<PyResult<Vec<_>>>()?;
            in_reading_order(&mut markers);
            let mut parts = vec![r(base)?];
            parts.extend(markers.into_iter().map(|(_, text)| text));
            format!("Annotated[{}]", parts.join(", "))
        }
        Schema::Ref(id) => {
            // A back edge into a definition already being rendered is the
            // lambda's parameter, which is what makes the form finite without
            // an ellipsis nothing can rebuild.
            let open = active.open.borrow().get(id).copied();
            if let Some(level) = open {
                return Ok(active.mark(level));
            }
            let level = active.open.borrow().len();
            active.open.borrow_mut().insert(*id, level);
            let body = defs.get(id.get()).map_or_else(|| Ok(GAVE_UP.to_owned()), r);
            active.open.borrow_mut().remove(id);
            format!("recursive(lambda {}: {})", active.mark(level), body?)
        }
        // The transient marker `recursive` uses while its own body is being
        // built. A compiled validator holds no such node, so nothing a caller
        // can print reaches this; it renders as the mark of a render that gave
        // up, because there is no definition to name yet.
        Schema::SelfRef(_) => GAVE_UP.to_owned(),
    })
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

/// The finished render, each binder mark replaced by a name it prints nowhere
/// else.
///
/// A binder is a name in the scope of everything its body prints, so one equal
/// to a name the body prints captures it: a class named `X` in the body of a
/// `lambda X:` reads back as the back edge, and the render rebuilds another
/// set. Every name a render prints comes from a class's name or a constant's
/// repr, which are known only once printed, so the binders are named last, each
/// the first of its level's names that no word of the render spells. A word is
/// a run of characters no ASCII punctuation or space breaks, read after the
/// interpreter's NFKC normalisation, which reads `Ｘ` as `X`. That is wider
/// than Python's identifiers -- a word inside a quoted string is one too -- and
/// the cost of the width is a suffixed name where a plain one would have read
/// back as well.
///
/// The names of different levels never meet: a level's names are its own
/// letter, alone or with a numbered suffix after an underscore, and no letter
/// has an underscore, so two nested definitions stay apart as [`binder`]
/// keeps them.
///
/// A printed name or repr holding the mark's character is not rewritten, which
/// would edit what was written: the render keeps its marks, and their control
/// characters make it a syntax error wherever it is parsed.
fn name_binders(py: Python<'_>, text: String, printed: usize) -> PyResult<String> {
    if printed == 0 || text.matches(MARK_OPEN).count() != printed {
        return Ok(text);
    }
    let normalized;
    let read = if text.is_ascii() {
        text.as_str()
    } else {
        normalized = py
            .import(intern!(py, "unicodedata"))?
            .call_method1(intern!(py, "normalize"), (intern!(py, "NFKC"), &text))?
            .extract::<String>()?;
        normalized.as_str()
    };
    let taken: FxHashSet<&str> = read
        .split(|c: char| c.is_ascii() && !c.is_ascii_alphanumeric() && c != '_')
        .collect();
    let mut by_level: Vec<String> = Vec::new();
    let mut named = String::with_capacity(text.len());
    let mut rest = text.as_str();
    while let Some((before, after)) = rest.split_once(MARK_OPEN) {
        let Some((level, after)) = after.split_once(MARK_CLOSE) else {
            return Ok(text);
        };
        let Ok(level) = level.parse::<usize>() else {
            return Ok(text);
        };
        while by_level.len() <= level {
            by_level.push(unprinted(binder(by_level.len()), &taken));
        }
        named.push_str(before);
        named.push_str(by_level.get(level).map_or("", String::as_str));
        rest = after;
    }
    named.push_str(rest);
    Ok(named)
}

/// `name`, or the first of its numbered spellings no word of the render takes.
///
/// One more spelling than there are words is always enough, and the bound keeps
/// the search finite whatever it is asked.
fn unprinted(name: String, taken: &FxHashSet<&str>) -> String {
    (0..=taken.len())
        .map(|n| {
            if n == 0 {
                name.clone()
            } else {
                format!("{name}_{n}")
            }
        })
        .find(|spelling| !taken.contains(spelling.as_str()))
        .unwrap_or(name)
}

/// Put each run of one kind of item in the order its renderings sort in.
///
/// Construction sorts a set the IR holds as a list -- a union's or a meet's
/// members, a refinement's constraints, a map's clauses -- with the node's own
/// `Ord`, which compares the variant first and then the payload. A payload that
/// names a constant or a class names it by its **pool slot**, which is the
/// order the constants were first seen, not a property of the schema. So
/// `Literal[1, 2]` and `Literal[2, 1]`, `A | B` and `B | A`, and `Annotated[int,
/// Ge(3), Ge(5)]` and `Annotated[int, Ge(5), Ge(3)]` are each one schema by `==`
/// and by `hash`, and printed in the slot order they would be two.
/// `docs/04-algebra.md` says "`repr` shows it and `==` compares it", so each
/// pair has to be one rendering.
///
/// The variant order is the slot-free part of the sort, so the runs keep it --
/// `int | str` prints in the order every scalar union prints in -- and within a
/// run the rendering is the key, because it is what the reader sees: two items
/// that print the same print the same wherever they sit. A set's order is not
/// part of what it names, so any order reads back as the same schema.
fn in_reading_order<K: PartialEq>(rendered: &mut [(K, String)]) {
    for run in rendered.chunk_by_mut(|a, b| a.0 == b.0) {
        run.sort_by(|a, b| a.1.cmp(&b.1));
    }
}

/// Render a union's members, each run of one kind in the order
/// [`in_reading_order`] gives it.
///
/// A member printed as a display -- a record `{...}` or a fixed list `[...]` --
/// is not a type, and `|` between it and the next member is Python's own
/// operator on the two values: `{'a': int} | {'b': str}` is a dict merge, read
/// back quietly as the record `{'a': int, 'b': str}`, and a list beside
/// anything is a `TypeError`. Such a union is printed as the `union(...)` call,
/// which reads every display back as the schema it spells.
fn render_union(
    members: &[Schema],
    render: &impl Fn(&Schema) -> PyResult<String>,
) -> PyResult<String> {
    let mut rendered = rendered_members(members, render)?;
    in_reading_order(&mut rendered);
    let displayed = rendered
        .iter()
        .any(|(_, text)| text.starts_with('{') || text.starts_with('['));
    let texts = rendered
        .into_iter()
        .map(|(_, text)| text)
        .collect::<Vec<_>>();
    Ok(if displayed {
        format!("union({})", texts.join(", "))
    } else {
        texts.join(" | ")
    })
}

/// Render a meet, reading a class's `isinstance` atom beside its attribute
/// record back as the class name the user wrote.
///
/// The pair is [`Schema::object_class`]; the members that are not the pair
/// render as themselves, so a meet that flattened others in beside a class still
/// names the class. Without this the annotation `Pt` would print as
/// `intersection(Pt, object(x=int))` -- the algebra's spelling of a thing the
/// user spelled with one name.
/// Each member beside its variant, the key [`in_reading_order`] groups by.
fn rendered_members<'a>(
    members: impl IntoIterator<Item = &'a Schema>,
    render: &impl Fn(&Schema) -> PyResult<String>,
) -> PyResult<Vec<(Discriminant<Schema>, String)>> {
    members
        .into_iter()
        .map(|member| Ok((discriminant(member), render(member)?)))
        .collect()
}

fn render_meet(
    py: Python<'_>,
    schema: &Schema,
    members: &[Schema],
    pool: &[Py<PyAny>],
    defs: &[Schema],
    active: &Binders,
    depth: usize,
) -> PyResult<String> {
    let r = |s: &Schema| render_at(py, s, pool, defs, active, depth + 1);
    let Some(class) = schema.object_class() else {
        let mut kids = rendered_members(members, &r)?;
        in_reading_order(&mut kids);
        let kids: Vec<String> = kids.into_iter().map(|(_, text)| text).collect();
        return Ok(format!("intersection({})", kids.join(", ")));
    };
    let name = pool_class_name(py, pool, class.get())?;
    // The atom, and the deep check of what the class declares: both are what
    // the name already says. The declaration is named where the fields have
    // names and positional where they do not, and `object_class` answers only
    // where there is exactly one of the two -- so dropping both kinds drops
    // exactly the member it read the class out of. Keeping the positional one
    // printed a `NamedTuple` as `intersection(Point, tuple[int, str])` beside a
    // dataclass printing as `DC`.
    let mut rest = rendered_members(
        members.iter().filter(|member| {
            !matches!(
                member,
                Schema::Instance(_) | Schema::AttrRecord { .. } | Schema::Seq { .. }
            )
        }),
        &r,
    )?;
    in_reading_order(&mut rest);
    Ok(if rest.is_empty() {
        name
    } else {
        let mut parts = vec![name];
        parts.extend(rest.into_iter().map(|(_, text)| text));
        format!("intersection({})", parts.join(", "))
    })
}

/// Render an attribute record on its own: `object(x=int)`.
///
/// A `Protocol` is the record with no class beside it, and a meet of two
/// classes with attributes holds records the render does not pair with their
/// classes. The form is readable rather than re-parsable, like the `{...}`
/// open-record marker.
fn render_attr_record(
    py: Python<'_>,
    fields: &[Attribute],
    pool: &[Py<PyAny>],
    defs: &[Schema],
    active: &Binders,
    depth: usize,
) -> PyResult<String> {
    let entries: Vec<String> = fields
        .iter()
        .map(|field| {
            let schema = render_at(py, &field.schema, pool, defs, active, depth + 1)?;
            Ok(format!("{}={}", field.name, schema))
        })
        .collect::<PyResult<_>>()?;
    Ok(format!("object({})", entries.join(", ")))
}

fn render_keyed_map(
    py: Python<'_>,
    fields: &[Field],
    defaults: &[MapClause],
    pool: &[Py<PyAny>],
    defs: &[Schema],
    active: &Binders,
    depth: usize,
) -> PyResult<String> {
    let r = |s: &Schema| render_at(py, s, pool, defs, active, depth + 1);
    // A pure mapping — no named fields, one clause — is dict[K, V].
    if fields.is_empty()
        && let [clause] = defaults
    {
        return Ok(format!("dict[{}, {}]", r(&clause.key)?, r(&clause.value)?));
    }
    // Otherwise a record/struct: named fields, then any catch-all clauses. A
    // name is spelled by Python's own `repr`, as a pattern is: written between
    // two quotes as it stood, a name carrying a quote closed the string early,
    // and `a': str, 'zz` printed as two fields.
    let mut entries: Vec<String> = fields
        .iter()
        .map(|field| {
            let suffix = if field.required { "" } else { "?" };
            let name = python_repr(py, &format!("{}{suffix}", field.name));
            Ok(format!("{name}: {}", r(&field.schema)?))
        })
        .collect::<PyResult<_>>()?;
    // Every clause renders as the key-to-value entry it is, the catch-all
    // included. `{'a': int, ...}` read better and rebuilt a *different* schema:
    // `...` is a dict key like any other, so the frontend reads it back as
    // `Literal[Ellipsis]` and the record is closed with an odd field rather
    // than open. The clauses are a set, ordered by a key that reads a literal
    // key's pool slot, so they print in reading order.
    let mut clauses = defaults
        .iter()
        .map(|clause| {
            Ok((
                discriminant(&clause.key),
                format!("{}: {}", r(&clause.key)?, r(&clause.value)?),
            ))
        })
        .collect::<PyResult<Vec<_>>>()?;
    in_reading_order(&mut clauses);
    entries.extend(clauses.into_iter().map(|(_, text)| text));
    Ok(format!("{{{}}}", entries.join(", ")))
}

fn render_constraint(
    py: Python<'_>,
    constraint: &Constraint,
    pool: &[Py<PyAny>],
) -> PyResult<String> {
    Ok(match constraint {
        Constraint::Ge(i) => format!("Ge({})", pool_repr(py, pool, i.get())?),
        Constraint::Gt(i) => format!("Gt({})", pool_repr(py, pool, i.get())?),
        Constraint::Le(i) => format!("Le({})", pool_repr(py, pool, i.get())?),
        Constraint::Lt(i) => format!("Lt({})", pool_repr(py, pool, i.get())?),
        Constraint::MinLen(n) => format!("MinLen({n})"),
        Constraint::MaxLen(n) => format!("MaxLen({n})"),
        Constraint::MultipleOf(i) => format!("MultipleOf({})", pool_repr(py, pool, i.get())?),
        Constraint::Predicate(_) => "Predicate(...)".to_owned(),
        Constraint::Regex(pattern) => format!("Regex({})", python_repr(py, pattern)),
    })
}

/// A string spelled the way Python spells it.
///
/// Rust's `{:?}` is Rust's escaping: it writes `\u{7}` for a control character,
/// which is a truncated escape to Python, and double quotes where Python's own
/// repr picks single. A rendered schema claims to read back, and the marker it
/// renders has a repr of its own, so the pattern inside it is spelled by the
/// interpreter that will read it rather than by the language that printed it.
///
/// A `str`'s `__repr__` does not raise, so the fallback is unreachable; it is
/// the Rust spelling rather than a placeholder, because a pattern a reader can
/// see is worth more here than one they cannot.
fn python_repr(py: Python<'_>, text: &str) -> String {
    PyString::new(py, text)
        .repr()
        .map_or_else(|_| format!("{text:?}"), |repr| repr.to_string())
}

/// A pooled constant, spelled by its own `repr` in full.
///
/// Whole, because a cut constant does not read back: an error message keeps
/// eighty characters of a value, and a render that kept them printed
/// `Literal[10**100]` as a syntax error. A render is as long as its schema, a
/// pattern and a field name already print whole, and a constant is part of the
/// schema as they are. A `repr` raising an ordinary exception -- an `int` past
/// the interpreter's digit limit -- prints [`UNREPRESENTABLE`], a syntax error
/// where it is read; a fatal signal propagates.
///
/// Bounds-checked rather than indexed: a corrupt pool index degrades to a
/// placeholder in the rendered string instead of panicking across the FFI
/// boundary, matching the defensive `.get` posture in the walk. The index is
/// pool-valid by construction, so the miss is an invariant break -- loud in
/// debug, a recognisable placeholder in release.
fn pool_repr(py: Python<'_>, pool: &[Py<PyAny>], index: usize) -> PyResult<String> {
    if let Some(constant) = pool.get(index) {
        let text = constant.bind(py).repr().map(|text| stored(&text));
        unless_fatal(text, py, UNREPRESENTABLE.to_owned())
    } else {
        debug_assert!(false, "pool index {index} out of range");
        Ok("<unknown>".to_owned())
    }
}

/// A literal's constant, marked where its `repr` reads back as another value.
///
/// A literal admits the values of its constant's own type, so a literal of
/// `S("a")`, `S` a subclass of `str`, refuses the `str` `"a"`. A subclass that
/// writes no `repr` of its own prints `'a'`, which reads back as the `str`: the
/// render would rebuild a schema admitting what this one refuses. So a
/// constant of a subclass of a builtin whose values are written as literals --
/// `int`, `float`, `complex`, `str`, `bytes` -- whose `repr` is the builtin's
/// spelling of it prints as `<S: 'a'>`, a syntax error where it is read, which
/// says what it holds and refuses to stand for anything else. A subclass whose
/// `repr` spells it some other way is printed by that `repr`, as every other
/// constant is: an enum member's `<Level.LOW: 1>`, and a `Shown(1)` that reads
/// back as itself. A bound is not marked: it is read as Python's comparison,
/// and the schema a bound of `S(3)` and one of `3` give is one schema.
fn literal_repr(py: Python<'_>, pool: &[Py<PyAny>], index: usize) -> PyResult<String> {
    let text = pool_repr(py, pool, index)?;
    let Some(constant) = pool.get(index).map(|constant| constant.bind(py)) else {
        return Ok(text);
    };
    let Some(builtin) = derived_builtin(constant) else {
        return Ok(text);
    };
    // The builtin's own `repr` of the value, which runs no user code; one that
    // raises, `int`'s past the digit limit, spells nothing the text can equal.
    let spelled = builtin
        .call_method1(intern!(py, "__repr__"), (constant,))
        .map(|spelled| spelled.str().map(|spelled| stored(&spelled)).ok());
    if unless_fatal(spelled, py, None)?.as_deref() != Some(text.as_str()) {
        return Ok(text);
    }
    let kind = class_label(constant.get_type().as_any())?;
    Ok(format!("<{kind}: {text}>"))
}

/// The builtin with a literal spelling `constant`'s type strictly derives from.
///
/// `bool` is such a builtin itself, though it derives from `int`: `True`
/// reads back as the `bool`, and no class derives from `bool`. Read through
/// `int`, it would be marked wherever `int`'s `repr` spells it `True`, as it
/// does on `PyPy`.
fn derived_builtin<'py>(constant: &Bound<'py, PyAny>) -> Option<Bound<'py, PyType>> {
    let py = constant.py();
    if constant.is_exact_instance_of::<PyBool>() {
        None
    } else if derives_from::<PyInt>(constant) {
        Some(PyInt::type_object(py))
    } else if derives_from::<PyFloat>(constant) {
        Some(PyFloat::type_object(py))
    } else if derives_from::<PyComplex>(constant) {
        Some(PyComplex::type_object(py))
    } else if derives_from::<PyString>(constant) {
        Some(PyString::type_object(py))
    } else if derives_from::<PyBytes>(constant) {
        Some(PyBytes::type_object(py))
    } else {
        None
    }
}

/// Whether `constant` is an instance of a strict subclass of `T`.
fn derives_from<T: PyTypeCheck + PyTypeInfo>(constant: &Bound<'_, PyAny>) -> bool {
    constant.is_instance_of::<T>() && !constant.is_exact_instance_of::<T>()
}

/// Render a class atom: the class's name, or `instance_of(Name)` where the name
/// alone would read back as more.
///
/// The name of a dataclass or a named tuple builds the class met with the
/// fields it declares, so the atom on its own prints as the call that builds
/// it. The class is asked, rather than the node: an atom carries no record of
/// how it was spelled.
fn render_instance(py: Python<'_>, pool: &[Py<PyAny>], index: usize) -> PyResult<String> {
    let name = pool_class_name(py, pool, index)?;
    let declares = match pool.get(index).map(|class| class.bind(py).cast::<PyType>()) {
        Some(Ok(class)) => declares_fields(class)?,
        _ => false,
    };
    Ok(if declares {
        format!("instance_of({name})")
    } else {
        name
    })
}

fn pool_class_name(py: Python<'_>, pool: &[Py<PyAny>], index: usize) -> PyResult<String> {
    if let Some(class) = pool.get(index) {
        class_label(class.bind(py))
    } else {
        debug_assert!(false, "pool index {index} out of range");
        Ok("<unknown>".to_owned())
    }
}
