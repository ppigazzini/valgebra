//! What a value is asked without descending into it.
//!
//! A scalar kind, a literal and a constraint are the walk's leaves: each is
//! decided by the value in hand, and none of them opens the value up and asks
//! about a part. A container's length belongs here too, for the same reason --
//! `MinLen` counts what a value holds without reading any of it, which is why
//! it is answered beside the kinds rather than beside the sequence walk.
//!
//! The dispatcher keeps its own scalar arms, and this module states the same
//! rules a second time in [`scalar_of`] and [`scalar_admits`], deliberately:
//! see the note there for what routing the arms through here costs and what
//! test holds the two statements together.

use jiter::JsonValue;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyFrozenSet, PyList, PySet, PyString, PyTuple};
use valgebra_core::{CollKind, ConstIx, Constraint, OperandIx, Schema, SeqKind, Violation};

use super::{
    Base, Frame, const_at, fold, held_len, is_fatal, member, operand_at, predicate_at,
    reads_its_length, record_fatal, record_if_fatal, stop,
};
use crate::check::ctx::Ctx;
use crate::check::index::compile_pattern;
use crate::check::violation::{mismatch, summarize_in, summarize_value};
use crate::codes::{
    Code, GREATER_THAN, GREATER_THAN_EQUAL, LESS_THAN, LESS_THAN_EQUAL, LITERAL_ERROR, MULTIPLE_OF,
    PREDICATE_ERROR, PREDICATE_FAILED, STRING_PATTERN_MISMATCH, TOO_LONG, TOO_SHORT,
};
use crate::errors::{SUMMARY_CHARS, shorten};
use crate::input::Value;

/// A schema whose membership is decided by the value alone.
///
/// Named as a kind rather than answered as a test, so a caller that asks the
/// same schema about many values -- a homogeneous sequence -- reads the schema
/// once and tests many times.
///
/// This *is* a second statement of the scalar rules, beside the arms of
/// [`member`], and it is one deliberately: routing those arms through here cost
/// the call boundary 2.8% and the record walk 2.4%, because a match the
/// compiler can see through is worth more there than the sharing is. The two
/// are held together by a test rather than by construction --
/// `the_scalar_loop_and_the_walk_admit_the_same_values` asks both about every
/// scalar schema and every kind of value, so a rule changed in one place and
/// not the other fails rather than deciding one value two ways.
#[derive(Clone, Copy)]
pub(super) enum Scalar {
    /// The top and the bottom, which admit everything and nothing.
    Everything,
    Nothing,
    NoneType,
    Bool,
    Int,
    Float,
    Str,
    Bytes,
}

/// The scalar kind of a schema, or `None` for one the walk must descend into.
#[inline]
pub(super) fn scalar_of(schema: &Schema) -> Option<Scalar> {
    Some(match schema {
        Schema::Anything(_) => Scalar::Everything,
        Schema::Nothing => Scalar::Nothing,
        Schema::NoneType => Scalar::NoneType,
        Schema::Bool => Scalar::Bool,
        Schema::Int => Scalar::Int,
        Schema::Float => Scalar::Float,
        Schema::Str => Scalar::Str,
        Schema::Bytes => Scalar::Bytes,
        _ => return None,
    })
}

/// Whether a scalar kind admits a value.
#[inline]
pub(super) fn scalar_admits(kind: Scalar, value: &Value<'_, '_>) -> bool {
    match kind {
        Scalar::Everything => true,
        Scalar::Nothing => false,
        Scalar::NoneType => value.is_none(),
        Scalar::Bool => value.is_bool(),
        // bool subclasses int, so True/False are ints: Bool is a subset of Int.
        Scalar::Int => value.is_int(),
        Scalar::Float => value.is_float(),
        Scalar::Str => value.is_str(),
        Scalar::Bytes => value.is_bytes(),
    }
}

/// The scalar kind every position of a sequence takes, where the walk of it
/// needs no path and reports no violation.
///
/// The shape a homogeneous list or tuple of a builtin type takes -- `list[int]`,
/// `tuple[str, ...]` -- and the one whose per-element cost is almost all
/// bookkeeping: a depth guard, a fatal-signal check and a dispatch around a
/// single type test. An explaining walk is not this shape, since it records the
/// position of each element it rejects.
///
/// **The depth is read, and the level is not held.** Every element sits one
/// level below the container, and the explaining walk reaches each through
/// [`member`], which takes that level and refuses at the bound. The two must
/// refuse together, so this declines to the general path wherever no level is
/// available and lets that path refuse. Holding one is what a caller that can
/// descend needs, and a scalar cannot.
#[inline]
pub(super) fn homogeneous_scalar(
    prefix: &[Schema],
    tail: Option<&Schema>,
    ctx: Ctx<'_>,
) -> Option<Scalar> {
    if !prefix.is_empty() || ctx.mode.explains() || !ctx.room_to_descend() {
        return None;
    }
    scalar_of(tail?)
}

/// A leaf decision: pass `ok` through, recording a type/value mismatch when it is
/// false in explain mode.
///
/// Inlined, and its recording half is not. Every scalar arm of the walk ends
/// here, so on a list of integers this is called once per element and does
/// nothing but return the bool it was handed: measured at twenty-two
/// instructions of call and return around a single test, which was a fifth of
/// the per-element cost. Inlining leaves the test where the answer already is.
/// The other half is a `Vec` push and a value summary, which only a failing
/// element in explain mode reaches -- it is marked cold so the branch predicts
/// the accepting path and the code sits away from it.
#[inline]
pub(super) fn admit(
    ok: bool,
    schema: &Schema,
    value: &Value<'_, '_>,
    frame: &mut Frame<'_, '_>,
) -> bool {
    let ctx = frame.ctx;
    if !ok && ctx.mode.explains() {
        record_mismatch(schema, value, frame);
    }
    ok
}

/// Record a leaf's type or value mismatch. The half of [`admit`] that allocates.
#[cold]
#[inline(never)]
fn record_mismatch(schema: &Schema, value: &Value<'_, '_>, frame: &mut Frame<'_, '_>) {
    frame
        .out
        .push(mismatch(schema, value, frame.path, frame.ctx));
}

pub(super) fn check_literal(
    index: ConstIx,
    value: &Value<'_, '_>,
    frame: &mut Frame<'_, '_>,
) -> bool {
    let ctx = frame.ctx;
    let Some(literal) = const_at(ctx, index, value.py()) else {
        return false;
    };
    let ok = fold(
        value
            .to_python()
            .and_then(|obj| literal_matches(&obj, literal)),
        value.py(),
        ctx,
    );
    if !ok && ctx.mode.explains() {
        frame.out.push(Violation {
            code: LITERAL_ERROR.as_str(),
            path: frame.path.clone(),
            expected: format!("the literal {}", summarize_in(literal, ctx)),
            value_summary: summarize_value(value, ctx),
        });
    }
    ok
}

/// Whether `value` is the typed singleton denoted by `literal`: same type and
/// equal. The same-type guard rules out Python's cross-type equality
/// (`1 == True == 1.0`), so `Literal[1]` denotes `{1}`, not `{1, True, 1.0}`.
/// Returns the comparison result so a raising `__eq__` is folded by the caller.
/// The three probe helpers below take the **value first** and the pooled object
/// second. All three arguments are `&Bound<'_, PyAny>`, so every transposition
/// typechecks, and a reader who has just read two of them carries a prior into
/// the third -- which is the condition under which a transposition gets written.
pub(crate) fn literal_matches(
    value: &Bound<'_, PyAny>,
    literal: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    Ok(value.get_type().is(literal.get_type()) && value.eq(literal)?)
}

/// Whether `value % operand == 0`. The remainder is zero iff it is falsy. Returns
/// the result so a raising `%` is folded by the caller (a non-numeric value whose
/// modulo is not defined is then a non-multiple).
///
/// The operator, not `__mod__` by name: the dunder is only half of what `%`
/// means. A type that does not know the operand answers `NotImplemented` and the
/// operand's `__rmod__` is asked next, which is how a `Fraction` or a `Decimal`
/// divides an `int` — and `NotImplemented` is truthy, so reading the dunder's
/// result directly reports every such pair a non-multiple.
fn is_multiple_of(value: &Bound<'_, PyAny>, operand: &Bound<'_, PyAny>) -> PyResult<bool> {
    // The remainder compared against zero, which is what the node denotes:
    // `value % operand == 0`. Reading the remainder's truthiness instead asks a
    // different question of any type whose `__bool__` and `__eq__` disagree --
    // `timedelta(0)` is falsy and does not equal `0` -- and the denotation is
    // the one a caller reads.
    value.rem(operand)?.eq(0)
}

/// Run a user predicate and report whether it returned a truthy result.
fn predicate_passes(value: &Bound<'_, PyAny>, predicate: &Bound<'_, PyAny>) -> PyResult<bool> {
    predicate.call1((value,))?.is_truthy()
}

/// Membership for a refinement: the base, and every constraint on a member of it.
///
/// Constraints narrow the base set, so they are asked only of a base member,
/// with one exception: a length bound on a container, which
/// [`lengths_before_elements`] asks before the elements. The loop below asks it
/// again, where it holds and records nothing unless the walk ran code that
/// changed the value.
///
/// Which bases have elements is a property of the node, so a refinement of a
/// scalar -- asked once per element of a list of them -- pays one comparison
/// of the base's variant. The loop is the one place [`check_constraint`] is
/// called, which is what keeps it inlined here: a second call site moves it out
/// of line and costs a list of refined elements 7% of its instructions.
pub(super) fn check_refine(
    base: &Schema,
    constraints: &[Constraint],
    value: &Value<'_, '_>,
    frame: &mut Frame<'_, '_>,
) -> bool {
    let ctx = frame.ctx;
    if matches!(
        base,
        Schema::Seq { .. } | Schema::Coll { .. } | Schema::KeyedMap { .. }
    ) && !lengths_before_elements(base, constraints, value, frame)
    {
        return false;
    }
    if !member(base, value, frame) {
        return false;
    }
    let obj = match value.to_python() {
        Ok(obj) => obj,
        Err(err) => {
            record_if_fatal(err, value.py(), ctx);
            return false;
        }
    };
    let mut ok = true;
    for constraint in constraints {
        ok &= check_constraint(constraint, &obj, ctx, frame);
        if !ok && stop(ctx) {
            return false;
        }
    }
    ok
}

/// Ask a container's length bounds where a fixed shape asks its arity: after
/// the kind test the base makes first, and before any element.
///
/// `[int, int]` refuses five elements with `list_length` alone, and
/// `Annotated[list[int], MinLen(1)]` is decided equal to `[int, int, ...]`, so
/// the two spellings of one set are refused at the same step -- and a list too
/// long for its bound is refused without reading what it holds, as a wrong
/// length ends the walk of a fixed shape.
#[inline(never)]
fn lengths_before_elements(
    base: &Schema,
    constraints: &[Constraint],
    value: &Value<'_, '_>,
    frame: &mut Frame<'_, '_>,
) -> bool {
    let ctx = frame.ctx;
    if !is_of_its_kind(base, value) {
        return true;
    }
    // The length every bound reads, taken once and the way the constraint takes
    // it. A JSON array holds its items as the list built from it would, so its
    // count is read without building one; a JSON object may repeat a key, which
    // the dict built from it does not, so it is built and asked.
    let len = match value {
        Value::Json(_, JsonValue::Array(items)) => Some(items.len()),
        Value::Json(..) | Value::Py(_) => {
            match value.to_python().and_then(|obj| stored_len(&obj, ctx)) {
                Ok(len) => Some(len),
                Err(err) => {
                    record_if_fatal(err, value.py(), ctx);
                    None
                }
            }
        }
    };
    let mut ok = true;
    for constraint in constraints {
        ok &= length_holds(constraint, len, value, frame);
        if !ok && stop(ctx) {
            return false;
        }
    }
    ok
}

/// Whether a length satisfies `constraint` where it bounds one, recorded as
/// [`check_constraint`] records it; any other constraint holds here. A length
/// that could not be read satisfies no bound.
fn length_holds(
    constraint: &Constraint,
    len: Option<usize>,
    value: &Value<'_, '_>,
    frame: &mut Frame<'_, '_>,
) -> bool {
    let ctx = frame.ctx;
    let (ok, code, expected) = match constraint {
        Constraint::MinLen(n) => (
            len.is_some_and(|len| len >= *n),
            TOO_SHORT,
            Expected::Length(">=", *n),
        ),
        Constraint::MaxLen(n) => (
            len.is_some_and(|len| len <= *n),
            TOO_LONG,
            Expected::Length("<=", *n),
        ),
        _ => return true,
    };
    if !ok && ctx.mode.explains() {
        record_failure(code, &expected, summarize_value(value, ctx), frame);
    }
    ok
}

/// Whether `value` passes the kind test a container base makes before reading
/// any element: a list or a JSON array, a tuple, a set, a frozenset, a dict or
/// a JSON object. The tests are the ones each container's walk makes first. One
/// that disagreed with its walk would move a report and never a verdict, since
/// the value must satisfy the base and every bound either way.
fn is_of_its_kind(base: &Schema, value: &Value<'_, '_>) -> bool {
    match (base, value) {
        (Schema::Seq { container, .. }, Value::Py(v)) => match container {
            SeqKind::List => v.is_instance_of::<PyList>(),
            SeqKind::Tuple => v.is_instance_of::<PyTuple>(),
        },
        (
            Schema::Seq {
                container: SeqKind::List,
                ..
            },
            Value::Json(_, JsonValue::Array(_)),
        )
        | (Schema::KeyedMap { .. }, Value::Json(_, JsonValue::Object(_))) => true,
        (Schema::Coll { container, .. }, Value::Py(v)) => match container {
            CollKind::Set => v.is_instance_of::<PySet>(),
            CollKind::FrozenSet => v.is_instance_of::<PyFrozenSet>(),
        },
        (Schema::KeyedMap { .. }, Value::Py(v)) => v.is_instance_of::<PyDict>(),
        _ => false,
    }
}

/// A builtin container the walk counts, with the tests that recognise one.
///
/// Two tests rather than one, and the first is what keeps the common reading
/// cheap: an exact container overrides nothing, so its own `__len__` *is* the
/// base's and asking its type for the slot buys a dictionary lookup per length
/// bound and learns nothing.
struct Sized {
    base: Base,
    is_exact: fn(&Bound<'_, PyAny>) -> bool,
    is_kind: fn(&Bound<'_, PyAny>) -> bool,
}

/// Every container whose length is the count of what it holds, most common
/// first: a length bound meets a string more often than a frozenset.
const SIZED: [Sized; 7] = [
    Sized {
        base: Base::Str,
        is_exact: |value| value.is_exact_instance_of::<PyString>(),
        is_kind: |value| value.is_instance_of::<PyString>(),
    },
    Sized {
        base: Base::List,
        is_exact: |value| value.is_exact_instance_of::<PyList>(),
        is_kind: |value| value.is_instance_of::<PyList>(),
    },
    Sized {
        base: Base::Dict,
        is_exact: |value| value.is_exact_instance_of::<PyDict>(),
        is_kind: |value| value.is_instance_of::<PyDict>(),
    },
    Sized {
        base: Base::Tuple,
        is_exact: |value| value.is_exact_instance_of::<PyTuple>(),
        is_kind: |value| value.is_instance_of::<PyTuple>(),
    },
    Sized {
        base: Base::Bytes,
        is_exact: |value| value.is_exact_instance_of::<PyBytes>(),
        is_kind: |value| value.is_instance_of::<PyBytes>(),
    },
    Sized {
        base: Base::Set,
        is_exact: |value| value.is_exact_instance_of::<PySet>(),
        is_kind: |value| value.is_instance_of::<PySet>(),
    },
    Sized {
        base: Base::FrozenSet,
        is_exact: |value| value.is_exact_instance_of::<PyFrozenSet>(),
        is_kind: |value| value.is_instance_of::<PyFrozenSet>(),
    },
];

/// The length of a value, read the way the rest of the walk reads it.
///
/// **One value has one length.** A container subclass may override `__len__`
/// and say anything; before this, `MinLen(5)` believed it and the shape beside
/// it counted the storage, so the two constraints described different sets and a
/// value could satisfy each in a different sense. A length that two parts of one
/// schema disagree about is not a property of the value, and a set defined by
/// one is not a set.
///
/// **And a length has one source.** The C accessor is not it: `PyTuple_Size`
/// reads the storage on `CPython` and goes through the object's own `__len__`
/// on `PyPy`'s `cpyext`, which is the overridden answer again under another
/// name. An exact container overrides nothing and is asked directly; a subclass
/// that overrides is asked of the base type's slot, through [`held_len`].
///
/// An object that is none of these -- one with a `__len__` and no builtin
/// container behind it -- answers for itself, because there is no storage to
/// read past it and `__len__` is the whole of what it holds.
pub(super) fn stored_len(value: &Bound<'_, PyAny>, ctx: Ctx<'_>) -> PyResult<usize> {
    for sized in &SIZED {
        if (sized.is_exact)(value) {
            return value.len();
        }
        if (sized.is_kind)(value) {
            return if reads_its_length(value, sized.base, ctx) {
                value.len()
            } else {
                held_len(value, sized.base)
            };
        }
    }
    value.len()
}

/// What a violation would say, carried unrendered until one is recorded.
///
/// Naming a bound takes the bound's `repr`, and a value that belongs produces no
/// violation to name it in. Rendering eagerly therefore makes accepting a value
/// cost the size of the schema's *operand* rather than the size of the value.
enum Expected<'py> {
    /// A comparison against a pool operand, as `symbol operand`.
    Order(&'static str, &'py Bound<'py, PyAny>),
    /// A length bound, as `length symbol n`.
    Length(&'static str, usize),
    /// A divisibility operand from the pool.
    Multiple(&'py Bound<'py, PyAny>),
    /// A pattern the whole string must match.
    Pattern(&'py str),
    /// A message with nothing to render into it.
    Fixed(&'static str),
    /// A predicate that raised, carrying the error it raised. Only ever built on
    /// the failing path, so it renders no sooner than the rest.
    Raised(String),
}

impl Expected<'_> {
    /// Render the message. Called once per recorded violation, never per check.
    fn render(&self, ctx: Ctx<'_>) -> String {
        match self {
            Self::Order(symbol, operand) => format!("{symbol} {}", summarize_in(operand, ctx)),
            Self::Length(symbol, n) => format!("length {symbol} {n}"),
            Self::Multiple(operand) => format!("a multiple of {}", summarize_in(operand, ctx)),
            Self::Pattern(pattern) => format!("a string matching {pattern:?}"),
            Self::Fixed(text) => (*text).to_owned(),
            Self::Raised(error) => {
                format!("a predicate that does not raise (raised {error})")
            }
        }
    }
}

/// Check one order bound (`Ge`/`Gt`/`Le`/`Lt`) against `value`: resolve the pool
/// constant and run the rich comparison at the boundary, folding an ordinary
/// error to a non-match. Returns `None` when the pool constant is unavailable,
/// the signal the caller turns into a non-member.
fn order_bound<'py>(
    value: &Bound<'py, PyAny>,
    index: OperandIx,
    ctx: Ctx<'py>,
    py: Python<'py>,
    compare: impl Fn(&Bound<'py, PyAny>, &Bound<'py, PyAny>) -> PyResult<bool>,
    code: Code,
    symbol: &'static str,
) -> Option<(bool, Code, Expected<'py>)> {
    let bound = operand_at(ctx, index, py)?;
    let ok = fold(compare(value, bound), py, ctx);
    // Borrowed from the pool for as long as the walk's context lives, so a
    // passing check -- every check, in fast mode -- takes no reference to the
    // bound; a free-threaded interpreter would make that an atomic on one
    // counter every thread sharing the validator touches.
    Some((ok, code, Expected::Order(symbol, bound)))
}

/// Whether `value` (already a base member, materialized once) satisfies one
/// constraint, recording a violation on failure in explain mode.
fn check_constraint<'py>(
    constraint: &'py Constraint,
    value: &Bound<'py, PyAny>,
    ctx: Ctx<'py>,
    frame: &mut Frame<'_, '_>,
) -> bool {
    let py = value.py();
    let (ok, code, expected): (bool, Code, Expected<'py>) = match constraint {
        Constraint::Ge(i) => {
            let Some(t) = order_bound(value, *i, ctx, py, |v, b| v.ge(b), GREATER_THAN_EQUAL, ">=")
            else {
                return false;
            };
            t
        }
        Constraint::Gt(i) => {
            let Some(t) = order_bound(value, *i, ctx, py, |v, b| v.gt(b), GREATER_THAN, ">") else {
                return false;
            };
            t
        }
        Constraint::Le(i) => {
            let Some(t) = order_bound(value, *i, ctx, py, |v, b| v.le(b), LESS_THAN_EQUAL, "<=")
            else {
                return false;
            };
            t
        }
        Constraint::Lt(i) => {
            let Some(t) = order_bound(value, *i, ctx, py, |v, b| v.lt(b), LESS_THAN, "<") else {
                return false;
            };
            t
        }
        Constraint::MinLen(n) => (
            fold(stored_len(value, ctx).map(|len| len >= *n), py, ctx),
            TOO_SHORT,
            Expected::Length(">=", *n),
        ),
        Constraint::MaxLen(n) => (
            fold(stored_len(value, ctx).map(|len| len <= *n), py, ctx),
            TOO_LONG,
            Expected::Length("<=", *n),
        ),
        Constraint::MultipleOf(i) => {
            let Some(operand) = operand_at(ctx, *i, py) else {
                return false;
            };
            let ok = fold(is_multiple_of(value, operand), py, ctx);
            (ok, MULTIPLE_OF, Expected::Multiple(operand))
        }
        Constraint::Predicate(i) => {
            // Slow path: the user's Python callable runs at the boundary. A
            // raising predicate is surfaced as a distinct `predicate_error`
            // rather than masked as an ordinary failed match.
            let Some(predicate) = predicate_at(ctx, *i, py) else {
                return false;
            };
            match predicate_passes(value, predicate) {
                Ok(passed) => (
                    passed,
                    PREDICATE_FAILED,
                    Expected::Fixed("a passing predicate"),
                ),
                // A fatal signal raised inside the predicate is the interpreter
                // unwinding, not a predicate that merely errored: propagate it.
                Err(err) if is_fatal(&err, py) => {
                    record_fatal(err, ctx);
                    return false;
                }
                // Bounded like every other value a message carries: a predicate
                // raising a megabyte of text is a message nobody reads.
                Err(err) => (
                    false,
                    PREDICATE_ERROR,
                    Expected::Raised(shorten(err.to_string(), SUMMARY_CHARS)),
                ),
            }
        }
        Constraint::Regex(pattern) => {
            // Native fast path: the precompiled, anchored pattern matches the
            // borrowed string UTF-8 in Rust. A non-string never matches (the base
            // of a pattern refinement is a string, so this is reached only after
            // a string base check, but stays defensive).
            //
            // A pattern absent from the index is compiled here, per value. A
            // validator's own walk never misses -- its index compiles every
            // pattern the schema holds -- but the relation oracle walks schemas
            // no validator owns, with an empty index, to ask whether a literal
            // belongs; answering "no match" there refutes an inclusion that
            // holds. `a_literal_is_asked_of_a_pattern_the_probe_compiles_itself`
            // holds the case.
            let matched = value
                .cast::<PyString>()
                .ok()
                .and_then(|s| s.to_str().ok())
                .is_some_and(|text| match ctx.regexes.get(&(pattern.as_ptr() as usize)) {
                    Some(compiled) => compiled.is_match(text),
                    None => compile_pattern(pattern).is_ok_and(|re| re.is_match(text)),
                });
            (matched, STRING_PATTERN_MISMATCH, Expected::Pattern(pattern))
        }
    };
    if !ok && ctx.mode.explains() {
        record_failure(code, &expected, summarize_in(value, ctx), frame);
    }
    ok
}

/// Record a constraint's failure. The message is rendered here and nowhere
/// else: a check that passes never reads the operand it would have named, and
/// never calls this.
#[cold]
fn record_failure(
    code: Code,
    expected: &Expected<'_>,
    value_summary: String,
    frame: &mut Frame<'_, '_>,
) {
    frame.out.push(Violation {
        code: code.as_str(),
        path: frame.path.clone(),
        expected: expected.render(frame.ctx),
        value_summary,
    });
}
