//! How `Annotated` metadata is read: the marker protocol, and whether the
//! constraint it carries fits the base it refines.
//!
//! A marker is read by *attribute* and never by name, so any library's marker of
//! the right shape works and none is imported. The section "How `Annotated`
//! metadata is read" in `docs/dev/03-frontend.md` is this module.

use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBytes, PyDict, PyString, PyTuple, PyType};
use valgebra_core::{Constraint, OperandIx, Schema};

use super::{Pool, build_schema, not_implemented};
use crate::errors::summarize;

/// Refuse an order bound against `nan`, which orders nothing.
///
/// Every comparison with `nan` is false, so `Annotated[float, Ge(nan)]` admits
/// **no value at all** -- and it is the empty set written as a bound, which no
/// caller means and neither decider proves empty. Dropping the bound would
/// admit every float instead; refusing says which mistake was made.
///
/// Only `nan` is refused, not every empty bound: `Gt(inf)` is also empty, but it
/// is empty because the order says so, which is an answer. `nan` is the absence
/// of an order.
pub(super) fn refuse_unordered_bound(attr: &str, bound: &Bound<'_, PyAny>) -> PyResult<()> {
    let unordered = bound.extract::<f64>().is_ok_and(f64::is_nan);
    if unordered {
        return Err(PyValueError::new_err(format!(
            "{attr} cannot be nan: every comparison with nan is false, so the \
             bound admits no value at all. Write the bound you mean, or `nothing` \
             for the empty set"
        )));
    }
    Ok(())
}

/// Build a Refine node from an `Annotated` base and its metadata markers.
///
/// Markers are read structurally (annotated-types style): an object exposing
/// `ge`/`gt`/`le`/`lt` contributes a comparison bound, `min_length`/
/// `max_length` contribute length bounds, and `func` (or a bare callable)
/// contributes a predicate. Unrecognized metadata is ignored, per the typing
/// spec. With no recognized constraint the base schema is returned as-is.
pub(super) fn build_refine(
    base: &Bound<'_, PyAny>,
    metadata: &Bound<'_, PyTuple>,
    lits: &mut Pool,
    defs: &mut Vec<Schema>,
) -> PyResult<Schema> {
    let base_schema = build_schema(base, lits, defs)?;
    let mut constraints = Vec::new();
    for marker in metadata.iter() {
        parse_constraint(&marker, &mut constraints, lits)?;
    }
    for constraint in &constraints {
        check_constraint_fits(&base_schema, constraint, lits)?;
    }
    Ok(Schema::refine(base_schema, constraints))
}

/// Whether the values a base admits can answer a constraint.
///
/// Three answers, because a refusal needs certainty. A constraint is refused only
/// where *no* value of the base can answer it, since that is the case where the
/// refinement denotes the empty set and the marker was written to narrow rather
/// than to empty. Where the base is opaque — a class, the gradual atom, a
/// literal's pooled constant, a recursive reference — the check stands aside.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Carries {
    /// Every value of the base can answer the constraint.
    Yes,
    /// No value of the base can, so the refinement admits nothing.
    No,
    /// The base does not say.
    Maybe,
}

impl Carries {
    /// The answer for a base that is a union of the two.
    ///
    /// A member that can answer makes the constraint a narrowing of the union
    /// rather than an emptying of it: `Annotated[int | str, MinLen(1)]` is the
    /// non-empty strings, which is a set a reader can mean.
    fn or(self, other: Carries) -> Carries {
        match (self, other) {
            (Carries::Yes, _) | (_, Carries::Yes) => Carries::Yes,
            (Carries::Maybe, _) | (_, Carries::Maybe) => Carries::Maybe,
            (Carries::No, Carries::No) => Carries::No,
        }
    }
}

/// Fold `answer` over the members of a union, and stand aside anywhere else that
/// is not a plain base: an intersection or a complement narrows a set this check
/// does not compute, and a refinement's answer is its own base's.
pub(super) fn carries_through(
    base: &Schema,
    answer: &impl Fn(&Schema) -> Carries,
) -> Option<Carries> {
    match base {
        Schema::Union(members) => Some(members.iter().map(answer).fold(Carries::No, Carries::or)),
        Schema::Refine { base, .. } => Some(answer(base)),
        Schema::Intersection(_) | Schema::Complement(_) | Schema::Ref(_) | Schema::SelfRef(_) => {
            Some(Carries::Maybe)
        }
        _ => None,
    }
}

/// Whether the base's values have a length.
pub(super) fn carries_length(base: &Schema) -> Carries {
    if let Some(answer) = carries_through(base, &carries_length) {
        return answer;
    }
    match base {
        Schema::Str
        | Schema::Bytes
        | Schema::Seq { .. }
        | Schema::Coll { .. }
        | Schema::KeyedMap { .. } => Carries::Yes,
        Schema::NoneType | Schema::Bool | Schema::Int | Schema::Float => Carries::No,
        _ => Carries::Maybe,
    }
}

/// Whether the base's values are text a pattern can be matched against.
pub(super) fn carries_pattern(base: &Schema) -> Carries {
    if let Some(answer) = carries_through(base, &carries_pattern) {
        return answer;
    }
    match base {
        Schema::Str => Carries::Yes,
        Schema::NoneType
        | Schema::Bool
        | Schema::Int
        | Schema::Float
        | Schema::Bytes
        | Schema::Seq { .. }
        | Schema::Coll { .. }
        | Schema::KeyedMap { .. } => Carries::No,
        _ => Carries::Maybe,
    }
}

/// Whether the base's values are numbers, which is what a divisor needs.
pub(super) fn carries_division(base: &Schema) -> Carries {
    if let Some(answer) = carries_through(base, &carries_division) {
        return answer;
    }
    match base {
        Schema::Bool | Schema::Int | Schema::Float => Carries::Yes,
        Schema::NoneType
        | Schema::Str
        | Schema::Bytes
        | Schema::Seq { .. }
        | Schema::Coll { .. }
        | Schema::KeyedMap { .. } => Carries::No,
        _ => Carries::Maybe,
    }
}

/// Whether the base's values are ordered against `operand`.
///
/// Python orders numbers with numbers, text with text and bytes with bytes, and
/// raises across those groups. A bound whose operand is in another group than the
/// base compares nothing, so the refinement it builds admits nothing.
pub(super) fn carries_order(base: &Schema, operand: &Bound<'_, PyAny>) -> Carries {
    let by_group = |base: &Schema| carries_order(base, operand);
    if let Some(answer) = carries_through(base, &by_group) {
        return answer;
    }
    let number = operand
        .py()
        .import("numbers")
        .and_then(|numbers| numbers.getattr("Number"))
        .is_ok_and(|class| operand.is_instance(&class).unwrap_or(false));
    let matches = match base {
        Schema::Bool | Schema::Int | Schema::Float => number,
        Schema::Str => operand.is_instance_of::<PyString>(),
        Schema::Bytes => operand.is_instance_of::<PyBytes>(),
        _ => return Carries::Maybe,
    };
    if matches { Carries::Yes } else { Carries::No }
}

/// Refuse a constraint no value of the base can answer.
///
/// A constraint that cannot be asked of a value is not a narrowing: reading a
/// length off an `int` raises, and the walk reads a raise as a non-member, so
/// `Annotated[int, MinLen(1)]` compiles to a schema that admits nothing at all
/// and says nothing about why. That is a schema nobody writes on purpose, so it
/// is refused where it is written rather than at the first value that meets it.
pub(super) fn check_constraint_fits(
    base: &Schema,
    constraint: &Constraint,
    lits: &Pool,
) -> PyResult<()> {
    let operand = |index: OperandIx| lits.items().get(index.get());
    let (answer, what) = match constraint {
        Constraint::MinLen(_) | Constraint::MaxLen(_) => (carries_length(base), "length"),
        Constraint::Regex(_) => (carries_pattern(base), "text for a pattern to match"),
        Constraint::MultipleOf(index) => match operand(*index) {
            Some(_) => (carries_division(base), "number for a divisor"),
            None => (Carries::Maybe, ""),
        },
        Constraint::Ge(index)
        | Constraint::Gt(index)
        | Constraint::Le(index)
        | Constraint::Lt(index) => match operand(*index) {
            Some(bound) => (
                Python::attach(|py| carries_order(base, bound.bind(py))),
                "order against that bound",
            ),
            None => (Carries::Maybe, ""),
        },
        Constraint::Predicate(_) => (Carries::Maybe, ""),
    };
    if answer == Carries::No {
        return Err(not_implemented(&format!(
            "{} values have no {what}, so this constraint admits none of them; \
             constrain a base the constraint can be asked of",
            base.expected()
        )));
    }
    Ok(())
}

/// The compilation flags a `re.Pattern` carries, folded into the pattern itself.
///
/// A compiled pattern keeps its flags beside its source, and the source alone is
/// a different expression: `re.compile("abc", re.I)` matches `"ABC"` and `"abc"`
/// does not. Dropping them silently narrows the set the marker was written for,
/// so each flag is either written into the pattern -- the engine here reads the
/// same inline spellings -- or refused by name.
///
/// `re.UNICODE` is the default for a `str` pattern in both engines and says
/// nothing extra, so it is not named here at all. `re.ASCII` and `re.LOCALE` change what a
/// character class means in ways this engine spells differently, and `re.DEBUG`
/// asks the other engine to talk about itself, so all three are refused rather
/// than approximated.
pub(super) fn with_inline_flags(
    marker: &Bound<'_, PyAny>,
    probes: &Probes<'_>,
    pattern: String,
) -> PyResult<String> {
    // `re`'s own bit values, which are part of its published interface.
    const IGNORECASE: u32 = 2;
    const LOCALE: u32 = 4;
    const MULTILINE: u32 = 8;
    const DOTALL: u32 = 16;
    const DEBUG: u32 = 128;
    const VERBOSE: u32 = 64;
    const ASCII: u32 = 256;

    let Some(flags) = probes.get(marker, Probe::Flags)? else {
        return Ok(pattern);
    };
    let Ok(flags) = flags.extract::<u32>() else {
        return Ok(pattern);
    };
    for (bit, name, why) in [
        (
            ASCII,
            "re.ASCII",
            "write the ASCII spellings ([0-9], [A-Za-z0-9_]) instead",
        ),
        (
            LOCALE,
            "re.LOCALE",
            "a pattern here does not depend on a locale",
        ),
        (
            DEBUG,
            "re.DEBUG",
            "it asks the other engine to report on itself",
        ),
    ] {
        if flags & bit != 0 {
            return Err(not_implemented(&format!(
                "{name} cannot be carried into this pattern: {why}"
            )));
        }
    }
    let mut inline = String::new();
    for (bit, letter) in [
        (IGNORECASE, 'i'),
        (MULTILINE, 'm'),
        (DOTALL, 's'),
        (VERBOSE, 'x'),
    ] {
        if flags & bit != 0 {
            inline.push(letter);
        }
    }
    if inline.is_empty() {
        Ok(pattern)
    } else {
        Ok(format!("(?{inline}){pattern}"))
    }
}

/// Whether `marker` comes from `annotated_types`, whose vocabulary a reader
/// expects this frontend to know.
///
/// The typing spec says to ignore metadata a consumer does not recognise, and
/// that is right for metadata written for someone else. A marker from the
/// constraint vocabulary is not that: it was written to narrow this schema, and
/// ignoring it leaves a validator that admits everything the marker excludes.
/// The one member carrying no constraint is the documentation marker, which says
/// nothing about which values belong.
pub(super) fn is_unhandled_constraint(marker: &Bound<'_, PyAny>) -> bool {
    let ty = marker.get_type();
    let from_vocabulary = ty
        .getattr("__module__")
        .ok()
        .and_then(|module| module.extract::<String>().ok())
        .is_some_and(|module| module == "annotated_types");
    let documentation = ty
        .getattr("__name__")
        .ok()
        .and_then(|name| name.extract::<String>().ok())
        .is_some_and(|name| name == "DocInfo");
    from_vocabulary && !documentation
}

/// An optional attribute a refinement marker is read through.
///
/// A marker carries one or two of these and not the other eight: `Ge(0)` has a
/// `ge` and nothing else, and a compiled pattern has a `pattern` and `flags`.
/// Absence is the common answer, and it was asked for by *trying*: a `getattr`
/// for an absent name answers by raising, and `func` was asked with a bare one
/// on every interpreter while the rest were asked with
/// `PyObject_GetOptionalAttr` -- the first non-raising spelling the interpreter
/// offers, and there is none before 3.13. A fifty-field record of
/// `Annotated[int, Ge(0)]` built, threw and dropped four hundred exceptions to
/// learn nine times over that a marker was what it said it was.
///
/// Which names a marker can carry is a property of its *type*, so the question
/// is asked there, once, and the answer kept: see [`Probes`].
#[derive(Clone, Copy)]
pub(super) enum Probe {
    Pattern,
    Flags,
    Ge,
    Gt,
    Le,
    Lt,
    MinLength,
    MaxLength,
    MultipleOf,
    Func,
}

impl Probe {
    /// Every probe, which is what a type is read for when it is first seen.
    const ALL: [Self; 10] = [
        Self::Pattern,
        Self::Flags,
        Self::Ge,
        Self::Gt,
        Self::Le,
        Self::Lt,
        Self::MinLength,
        Self::MaxLength,
        Self::MultipleOf,
        Self::Func,
    ];

    /// The attribute's name, as a handle the interpreter already holds: text
    /// would be decoded into a fresh `PyString` and hashed before the lookup
    /// could begin, once per name per marker.
    fn name(self, py: Python<'_>) -> &Bound<'_, PyString> {
        match self {
            Self::Pattern => intern!(py, "pattern"),
            Self::Flags => intern!(py, "flags"),
            Self::Ge => intern!(py, "ge"),
            Self::Gt => intern!(py, "gt"),
            Self::Le => intern!(py, "le"),
            Self::Lt => intern!(py, "lt"),
            Self::MinLength => intern!(py, "min_length"),
            Self::MaxLength => intern!(py, "max_length"),
            Self::MultipleOf => intern!(py, "multiple_of"),
            Self::Func => intern!(py, "func"),
        }
    }

    /// This probe's bit in a type's mask.
    const fn bit(self) -> u16 {
        1 << (self as u16)
    }
}

/// The mask each marker type reads under, one bit per [`Probe`] plus the two
/// above. Keyed by the type, which is what the answer is a property of.
///
/// A shared, *mutable* Python object, which the other caches on this crate's
/// path are not, so the free-threading argument is worth writing down: a
/// `dict`'s reads and writes are atomic under a free-threaded interpreter's
/// per-object lock, and two threads that miss on one type both compute the same
/// mask and write it, since the mask is a property of the type and not of
/// either thread. The `PyOnceLock` around it makes the dict itself arrive once.
static CARRIED: PyOnceLock<Py<PyDict>> = PyOnceLock::new();

/// How many marker types the mask cache keeps.
///
/// A cache entry holds a type, so it keeps one alive. The markers a program
/// uses are a handful of module-level classes and the map stops growing after
/// them; a program that *builds* a marker class per call would grow it without
/// a bound, and `typing` memoises `Annotated[...]` in a cache of its own that
/// holds the last hundred and twenty-eight, so this map would be the one that
/// grows. Past the bound a type is read each time rather than remembered --
/// slower than the cache, and no worse than having none.
///
/// The far side is exercised by `tests/test_adversarial_bounds.py`, which
/// builds marker types well past this and reads the same answers from them.
const MAX_MARKER_TYPES: usize = 256;

/// How a marker's optional attributes are read, decided once per marker.
///
/// Two questions settle it. Which names the *type* carries is asked of the type
/// and kept, because it cannot differ between two markers of one class: every
/// `annotated_types` marker is a `slots` dataclass, so `Ge.ge` is the
/// descriptor that reads the slot and `Ge.gt` does not exist. And a marker that
/// is an ordinary object keeps its values in a dictionary of its own, which is
/// read directly -- a dictionary answers for a name it does not hold without
/// raising, which is the whole point.
///
/// A type with a `__getattr__` hook answers for names neither holds, so it is
/// asked for everything, exactly as the frontend asked before this. That path
/// is correct and gives up only the saving.
pub(super) struct Probes<'py> {
    mask: u16,
    own: Option<Bound<'py, PyDict>>,
}

impl<'py> Probes<'py> {
    /// The type answers for names no dictionary of its own holds -- a
    /// `__getattr__` hook -- so every name is asked of the marker, as before.
    const HAS_HOOK: u16 = 1 << 14;

    /// Instances of the type keep their values in a dictionary of their own,
    /// which is where a marker that is not a `slots` class puts them.
    const HAS_OWN_DICT: u16 = 1 << 15;

    /// Read how this marker answers, from its type and its own dictionary.
    pub(super) fn of(marker: &Bound<'py, PyAny>) -> PyResult<Self> {
        let py = marker.py();
        let ty = marker.get_type();
        let cache = CARRIED
            .get_or_try_init(py, || Ok::<_, PyErr>(PyDict::new(py).unbind()))?
            .bind(py);
        let mask = if let Some(held) = cache.get_item(&ty)? {
            held.extract()?
        } else {
            let read = Self::read_type(&ty, marker)?;
            if cache.len() < MAX_MARKER_TYPES {
                cache.set_item(&ty, read)?;
            }
            read
        };
        let own = if mask & Self::HAS_OWN_DICT == 0 {
            None
        } else {
            marker
                .getattr_opt(intern!(py, "__dict__"))?
                .and_then(|dict| dict.cast_into::<PyDict>().ok())
        };
        Ok(Self { mask, own })
    }

    /// Which names the type carries, and how its instances keep the rest.
    fn read_type(ty: &Bound<'py, PyType>, marker: &Bound<'py, PyAny>) -> PyResult<u16> {
        let py = ty.py();
        let mut mask = 0;
        for probe in Probe::ALL {
            if ty.getattr_opt(probe.name(py))?.is_some() {
                mask |= probe.bit();
            }
        }
        if ty.getattr_opt(intern!(py, "__getattr__"))?.is_some() {
            mask |= Self::HAS_HOOK;
        }
        if marker
            .getattr_opt(intern!(py, "__dict__"))?
            .is_some_and(|dict| dict.is_instance_of::<PyDict>())
        {
            mask |= Self::HAS_OWN_DICT;
        }
        Ok(mask)
    }

    /// Read one of the marker's attributes, asking only where an answer can be.
    fn get(&self, marker: &Bound<'py, PyAny>, probe: Probe) -> PyResult<Option<Bound<'py, PyAny>>> {
        let name = probe.name(marker.py());
        if self.mask & (probe.bit() | Self::HAS_HOOK) != 0 {
            return marker.getattr_opt(name);
        }
        match &self.own {
            Some(own) => own.get_item(name),
            None => Ok(None),
        }
    }
}

/// How deep a marker may group other markers before the frontend refuses.
///
/// A grouped marker answers with the constraints it stands for, and one of
/// those may be grouped in turn. The vocabulary's own nest one deep at most; a
/// marker that answers with itself nests forever, and following it is a stack
/// this library does not have. The bound is a refusal rather than a truncation
/// for the reason every other refusal here is: a marker read in part leaves a
/// schema admitting what the rest of it excludes.
const MAX_GROUPING_DEPTH: u32 = 8;

pub(super) fn parse_constraint(
    marker: &Bound<'_, PyAny>,
    out: &mut Vec<Constraint>,
    lits: &mut Pool,
) -> PyResult<()> {
    parse_constraint_within(marker, out, lits, 0)
}

/// Whether this marker stands for the constraints it yields.
///
/// `annotated_types` marks one with an attribute, and reading the attribute is
/// how the rest of this module reads a marker too: an embedded interpreter
/// starts on the base prefix and sees no virtual environment, so importing the
/// package to ask `isinstance` would make the answer depend on how the process
/// was launched.
fn groups_other_markers(marker: &Bound<'_, PyAny>) -> bool {
    let py = marker.py();
    marker
        .getattr_opt(intern!(py, "__is_annotated_types_grouped_metadata__"))
        .ok()
        .flatten()
        .is_some_and(|flag| flag.is_truthy().unwrap_or(false))
}

/// Read a marker that stands for the constraints it yields.
///
/// The comment the caller carries, in the one place that acts on it: a grouped
/// marker is several constraints, each read the way a marker written on its own
/// is, so a group of groups terminates at [`MAX_GROUPING_DEPTH`] rather than at
/// the stack.
fn parse_grouped(
    marker: &Bound<'_, PyAny>,
    out: &mut Vec<Constraint>,
    lits: &mut Pool,
    depth: u32,
) -> PyResult<()> {
    if depth >= MAX_GROUPING_DEPTH {
        return Err(PyValueError::new_err(format!(
            "{} groups markers nested too deeply: a marker stands for the \
             constraints it yields, and following this one does not bottom out",
            summarize(marker)
        )));
    }
    for inner in marker.try_iter()? {
        parse_constraint_within(&inner?, out, lits, depth + 1)?;
    }
    Ok(())
}

fn parse_constraint_within(
    marker: &Bound<'_, PyAny>,
    out: &mut Vec<Constraint>,
    lits: &mut Pool,
    depth: u32,
) -> PyResult<()> {
    // A class is metadata this frontend does not recognise, and the typing spec
    // says to ignore what a consumer does not recognise.
    //
    // It has to be refused before any attribute is read, not only before the
    // predicate arms. A marker *class* exposes descriptors where an instance
    // exposes values: `at.Ge(0)` carries `ge = 0`, while `at.Ge` carries the
    // slot descriptor that reads it, and taking that for a bound builds a
    // comparison no value is ordered against. Calling one is the same trap a
    // step later — `Kilograms(1.5)` constructs a unit marker rather than
    // answering whether 1.5 belongs.
    if marker.is_instance_of::<PyType>() {
        return Ok(());
    }
    let before = out.len();
    // How this marker answers, read once per marker and mostly once per type.
    // Every question below goes through it, so a name nothing carries is
    // answered from a dictionary rather than by the interpreter raising.
    let probes = Probes::of(marker)?;

    // A string-pattern marker: valgebra's `Regex(...)` or a compiled
    // `re.Pattern`, both carrying the source pattern as `.pattern`. The pattern
    // is validated (anchored) here so an invalid expression fails at compile
    // time, not at first validation; the compiled regex is cached per validator.
    if let Some(attr) = probes.get(marker, Probe::Pattern)? {
        let Ok(pattern) = attr.extract::<String>() else {
            // A pattern this frontend cannot read as text. `re` compiles one
            // against `bytes` values, and a pattern constraint here matches the
            // text of a `str`; anything else carrying a `pattern` attribute is
            // a marker whose pattern is not a pattern at all. Reading the marker
            // and dropping it would leave a schema that admits every value of
            // its base, which is the opposite of what a pattern is written for,
            // so the refusal names what was found rather than a kind it guessed.
            let what = if attr.is_instance_of::<PyBytes>() {
                "a bytes pattern".to_owned()
            } else {
                format!("the pattern {}", summarize(&attr))
            };
            return Err(not_implemented(&format!(
                "{what} cannot constrain a schema: a pattern is matched against \
                 text, so write the pattern as a str",
            )));
        };
        let pattern = with_inline_flags(marker, &probes, pattern)?;
        // Refused before the compile, because this engine compiles these and
        // reads them as a different set than `re` does. The parse error the
        // compile gives is for the loud direction; this is the quiet one.
        super::dialect::reject_reserved_class_syntax(&pattern)?;
        crate::check::compile_pattern(&pattern).map_err(|err| {
            PyValueError::new_err(format!("invalid regular expression {pattern:?}: {err}"))
        })?;
        out.push(Constraint::Regex(pattern));
        return Ok(());
    }
    // Comparison bounds. One marker may carry several (e.g. an interval).
    //
    // A marker carries one of these four and not the other three, so the three
    // absences are the common answer -- and the name each is asked by is a
    // *handle* rather than Rust text, because text is decoded into a fresh
    // `PyString` and hashed before the lookup can begin, once per name per
    // marker.
    let py = marker.py();
    for (probe, make) in [
        (Probe::Ge, Constraint::Ge as fn(OperandIx) -> Constraint),
        (Probe::Gt, Constraint::Gt),
        (Probe::Le, Constraint::Le),
        (Probe::Lt, Constraint::Lt),
    ] {
        if let Some(bound) = probes.get(marker, probe)?
            && !bound.is_none()
        {
            refuse_unordered_bound(&probe.name(py).to_string_lossy(), &bound)?;
            out.push(make(lits.intern_operand(&bound)));
        }
    }
    // Length bounds. A bound no length can be compared against -- negative, or
    // past what a container can hold -- is refused rather than dropped: dropping
    // it leaves a schema that admits every value of its base, and the marker was
    // written to admit fewer.
    for (probe, make) in [
        (
            Probe::MinLength,
            Constraint::MinLen as fn(usize) -> Constraint,
        ),
        (Probe::MaxLength, Constraint::MaxLen),
    ] {
        if let Some(bound) = probes.get(marker, probe)?
            && !bound.is_none()
        {
            // A `bool` is read as the length it equals: `MinLen(True)` is
            // `MinLen(1)`. Refusing it was tried and withdrawn -- `MinLen(0)` and
            // `MinLen(False)` are equal and hash alike, so `typing` returns one
            // `Annotated` object for both and whichever spelling a process built
            // first is the one every later spelling gets. A refusal would fail
            // correct `MinLen(0)` code because of a `MinLen(False)` somewhere
            // else, which is worse than reading the value the marker holds.
            let n = bound.extract::<usize>().map_err(|_| {
                PyValueError::new_err(format!(
                    "{} must be a length a value can have, and {} is not",
                    probe.name(py),
                    summarize(&bound)
                ))
            })?;
            out.push(make(n));
        }
    }
    // Numeric multiple-of bound. A zero divisor is rejected here: no value is a
    // multiple of zero, and checking one would divide by zero at validation time,
    // so the schema is unsatisfiable and the error belongs at construction.
    if let Some(multiple) = probes.get(marker, Probe::MultipleOf)?
        && !multiple.is_none()
    {
        if multiple.extract::<f64>().is_ok_and(f64::is_nan) {
            return Err(PyValueError::new_err(
                "MultipleOf(nan) is not a valid constraint: no value is a multiple \
                 of nan, because every comparison with nan is false. Write the \
                 step you mean",
            ));
        }
        if multiple.eq(0).unwrap_or(false) {
            return Err(PyValueError::new_err(
                "MultipleOf(0) is not a valid constraint: no value is a multiple of \
                 zero. Use a nonzero divisor.",
            ));
        }
        out.push(Constraint::MultipleOf(lits.intern_operand(&multiple)));
    }
    // Predicate escape hatch: a callable marker, or `annotated_types.Predicate`,
    // which carries its callable on `.func` and is not callable itself.
    //
    // Callability is how `annotated_types` tells its two marker shapes apart:
    // `Not` defines `__call__` so a consumer calls it, and calling is what
    // applies the negation, while `Predicate` deliberately does not. Reading
    // `.func` from whichever marker has one drops `Not`'s negation and strips a
    // `functools.partial` of its bound arguments — both carry a `.func` too.
    if marker.is_callable() {
        out.push(Constraint::Predicate(lits.intern_predicate(marker)));
    } else if let Some(func) = probes.get(marker, Probe::Func)?
        && func.is_callable()
    {
        out.push(Constraint::Predicate(lits.intern_predicate(&func)));
    } else if out.len() == before && groups_other_markers(marker) {
        // A marker standing for several constraints answers with them, which is
        // the protocol `annotated_types` documents and what a caller writes
        // their own against. `Interval` and `Len` carry their bounds as
        // attributes too, so the probes above have already read them and this
        // arm is reached only by a marker whose constraints live nowhere else --
        // which is why the question is asked here rather than before them. Read
        // first, it cost an attribute lookup on every marker a schema carries,
        // and a build of one annotated record spends that on each of them.
        //
        // Without the arm, a marker written this way is metadata the frontend
        // does not recognise, which the typing spec says to ignore -- leaving a
        // schema that admits everything the marker was written to exclude.
        return parse_grouped(marker, out, lits, depth);
    } else if out.len() == before && is_unhandled_constraint(marker) {
        return Err(not_implemented(&format!(
            "{} is a constraint this frontend does not check; a schema carrying \
             it would admit the values it excludes, so it is refused rather than \
             ignored",
            summarize(marker)
        )));
    }
    Ok(())
}
