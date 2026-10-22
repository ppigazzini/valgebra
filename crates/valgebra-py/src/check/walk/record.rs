//! The keyed maps and attribute records a value is read as.
//!
//! A record is the shape whose membership is a question per key rather than
//! per position: which keys the value carries, which of them the schema
//! declares, and what a key the schema does not declare is covered by. The
//! walk's other containers ask about elements and share none of that, which is
//! why these live together and beside rather than inside the dispatcher.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::ops::ControlFlow;
use std::sync::Arc;

use ahash::RandomState;
use jiter::JsonValue;
use pyo3::prelude::*;
use pyo3::sync::critical_section::with_critical_section;
use pyo3::types::{PyDict, PyString};
use rustc_hash::FxHashMap;
use valgebra_core::{Attribute, Field, MapClause, PathSegment, Schema, quoted};

use super::scalar::{Scalar, admitted_quietly, scalar_admits, scalar_member, scalar_of};
#[cfg(PyPy)]
use super::{Base, held_dict, reads_its_length, reads_its_values};
use super::{
    Frame, Scan, Unwritten, again, fast, fold, is_fatal, member, mutated, record_fatal,
    record_if_fatal, record_move, stop,
};
use crate::check::ctx::{Ctx, WalkState};
use crate::check::index::{RecordPlan, decided_by_its_value};
use crate::check::violation::{at_key, key_segment, located, summarize_value, type_mismatch};
use crate::codes::{DICT_TYPE, EXTRA_FORBIDDEN, MISSING_ATTRIBUTE, MISSING_KEY};
use crate::input::Value;

/// Visit a dict's entries, refusing rather than panicking when the dict changes
/// size underneath the scan.
///
/// The iterator `PyO3` hands out panics when the dict's length moves, and the walk
/// runs Python at every entry, so that panic is reachable from an ordinary
/// schema — and a panic is not one of the answers this library gives: it crosses
/// the boundary as a `BaseException` that no caller catches as a validation
/// failure. The scan asks the same question one step earlier, before each step
/// rather than inside it, and stops at the entry count it began with, so the
/// iterator is never advanced into either of the states it panics in.
///
/// The loop is spelled once for a build with a global lock and once for a
/// free-threaded one (`scan_held_dict`), and the two read the same entries
/// against the same count. The critical section is no lock on this build, and
/// the loop keeps both it and its own spelling because the record walk the
/// scan is inlined into lays out differently without either: spelled through
/// `find_map`, or without the closure the section takes, the record shapes of
/// the instruction gate read 1.4% dearer on 3.14, and 1.9% where the keys are
/// interned.
///
/// `PyPy`'s `PyDict_Next` does not survive the change at all: a key replaced by
/// another at the same size, while the scan runs Python, is a fatal error inside
/// `cpyext` and takes the process with it. There the entries are read from a
/// copy taken before the first visit, which nothing else can reach, while the
/// size check still reads the value itself. `dict.copy` asks no key for its
/// hash, so taking it runs no code of the value's.
#[cfg(not(Py_GIL_DISABLED))]
pub(super) fn scan_dict<'py>(
    dict: &Bound<'py, PyDict>,
    state: &WalkState,
    mut visit: impl FnMut(&Bound<'py, PyAny>, &Bound<'py, PyAny>) -> ControlFlow<()>,
) -> Scan {
    // The state is read where the copy is taken, which is `PyPy` alone. A
    // reference rather than the walk's context: the context is seven words,
    // and handed by value to a function this size it is copied to the stack
    // by every caller, which the profile-guided build hoisted to the entry of
    // `member`, a copy on every level of every walk.
    #[cfg(not(PyPy))]
    let _ = state;
    with_critical_section(dict.as_any(), || {
        let entries = dict.len();
        // A copy that cannot be taken is a dict the scan could not read, and a
        // fatal signal taking it raised is recorded, as [`stored`] records one.
        #[cfg(PyPy)]
        let copied = match held_dict(dict) {
            Ok(copied) => copied,
            Err(err) => {
                if is_fatal(&err, dict.py()) {
                    state.keep_fatal(err);
                }
                return Scan::Unreadable;
            }
        };
        #[cfg(PyPy)]
        let mut iter = copied.iter();
        #[cfg(not(PyPy))]
        let mut iter = dict.iter();
        let mut seen = 0;
        while seen < entries {
            if dict.len() != entries {
                return Scan::Unreadable;
            }
            let Some((key, value)) = iter.next() else {
                break;
            };
            seen += 1;
            if visit(&key, &value).is_break() {
                return Scan::Stopped;
            }
        }
        if dict.len() == entries {
            Scan::Complete
        } else {
            Scan::Unreadable
        }
    })
}

/// [`scan_dict`] on a free-threaded build, where the dict is held for the
/// whole scan and read by `scan_held_dict`. The critical section keeps a
/// second thread out of the dict for the parts of the scan that do not call
/// back into the interpreter.
#[cfg(Py_GIL_DISABLED)]
pub(super) fn scan_dict<'py>(
    dict: &Bound<'py, PyDict>,
    state: &WalkState,
    visit: impl FnMut(&Bound<'py, PyAny>, &Bound<'py, PyAny>) -> ControlFlow<()>,
) -> Scan {
    // Read where `PyPy` copies the dict, which a free-threaded build is not.
    let _ = state;
    with_critical_section(dict.as_any(), || scan_held_dict(dict, visit))
}

/// Visit a dict's entries inside the critical section [`scan_dict`] holds on
/// it, on a free-threaded build.
///
/// `PyO3`'s dict iterator takes the dict's section around every step it is
/// asked for, and inside the section the scan already holds each of those is
/// a re-entry: a call into the interpreter, a compare-and-swap that fails, a
/// second call that finds the section already held, and a third to end it, at
/// every entry. Its `find_map` takes the section once and steps inside it, so
/// the scan is spelled through it and pays the re-entry once a dict:
/// `is_valid` on a hundred-entry `dict[str, int]` reads 16% fewer instructions
/// on 3.14t and 17% fewer on 3.15t, and the 3.14t PGO wheel takes 18% less
/// time over it.
///
/// `find_map` steps the iterator before it hands over each entry, so the
/// question [`scan_dict`] asks before a step is asked here after each visit,
/// which is the same point in the scan: nothing runs between taking the count
/// and the first step, and the scan stops at the count it began with.
#[cfg(Py_GIL_DISABLED)]
fn scan_held_dict<'py>(
    dict: &Bound<'py, PyDict>,
    mut visit: impl FnMut(&Bound<'py, PyAny>, &Bound<'py, PyAny>) -> ControlFlow<()>,
) -> Scan {
    let entries = dict.len();
    let read_to_end = || {
        if dict.len() == entries {
            Scan::Complete
        } else {
            Scan::Unreadable
        }
    };
    let mut seen = 0;
    dict.iter()
        .find_map(|(key, value)| {
            if visit(&key, &value).is_break() {
                return Some(Scan::Stopped);
            }
            seen += 1;
            (seen == entries || dict.len() != entries).then(read_to_end)
        })
        .unwrap_or_else(read_to_end)
}

/// A dict the C-level reads reach the storage of, on `PyPy`: the value itself,
/// or a copy of a subclass `cpyext` would read through its own methods.
///
/// `CPython` answers `PyDict_Size`, `PyDict_Next` and `PyDict_GetItem` from the
/// storage of any dict. `PyPy`'s `cpyext` answers the first through the
/// object's own `__len__`, and reads each value `PyDict_Next` yields through
/// its own `__getitem__`, so read through those, a subclass overriding either
/// would be another dict to the walk than the one it holds: an overridden
/// `__len__` would read as an extra key and refuse `{"a": 1}` from a closed
/// record, and an overridden `__getitem__` would refuse it from
/// `dict[str, int]`. The base's own `dict.copy` reads the storage there, and
/// the walk reads the copy. An exact dict, and a subclass
/// overriding neither, are read where they lie. Compiled for `PyPy` alone: the
/// walk on `CPython` reads the value itself.
///
/// `None` where the copy cannot be made, which the caller reports as a value it
/// could not read; a fatal signal the copy raised is recorded.
#[cfg(PyPy)]
pub(super) fn stored<'a, 'py>(
    dict: &'a Bound<'py, PyDict>,
    ctx: Ctx<'_>,
) -> Option<Cow<'a, Bound<'py, PyDict>>> {
    if dict.is_exact_instance_of::<PyDict>()
        || (reads_its_length(dict, Base::Dict, ctx) && reads_its_values(dict, ctx))
    {
        return Some(Cow::Borrowed(dict));
    }
    stored_copy(dict, ctx).map(Cow::Owned)
}

/// The copy [`stored`] walks, out of line: only a `PyPy` subclass overriding
/// what `cpyext` reads reaches it.
#[cfg(PyPy)]
#[cold]
#[inline(never)]
fn stored_copy<'py>(dict: &Bound<'py, PyDict>, ctx: Ctx<'_>) -> Option<Bound<'py, PyDict>> {
    if ctx.state.signals.fatal_seen.get() {
        return None;
    }
    held_dict(dict)
        .map_err(|err| record_if_fatal(err, dict.py(), ctx))
        .ok()
}

/// Where the deciding walk stopped, so the explaining walk need not redo what it
/// already read.
///
/// A keyed map that fails is walked twice: once to decide, once to say which
/// field. Both walks resolve the declared keys in the same order, and a probe is
/// the dear half -- about 143 instructions each through the interpreter, against
/// a handful for the check that follows. For a fifty-field record refused at the
/// thirty-second position, thirty-one probes and thirty-one checks are repeated
/// for nothing: a field the deciding walk *passed* has no violation to report,
/// so the explaining walk has nothing to learn by asking again.
///
/// What that costs to know is three integers and no allocation, which is why it
/// is carried rather than recomputed. Carrying the probed values themselves
/// instead saves the probes and not the checks, and charges every *accepting*
/// validation a clone per field -- 13.5% on the explaining accept path and
/// 2.35% even in the fast mode that records nothing, against 15.7% saved on the
/// path that fails. `BindingShape::ExplainAccept` is the shape that measures the
/// accepting side, which is the side such a trade charges.
///
/// The guard is the one the deciding walk already applies to itself: the entry
/// count. A value that changed size between the passes resumes from the start,
/// and one that changed a *value* without changing its size is what
/// [`mutated`] is for -- the explaining walk then finds nothing and says so,
/// which is a truer report than a violation about a value that has moved on.
#[derive(Default)]
pub(super) struct Decided {
    /// The dict's entry count when the deciding walk read it.
    entries: usize,
    /// How many fields the deciding walk admitted before it refused, in the
    /// order it read them, or how many entries where it scanned the dict: what
    /// a union reads to choose the record branch it explains.
    admitted: usize,
    /// Whether the refusal lay inside the value of a field that is not decided
    /// by its own value -- a record, a sequence, a reference -- rather than at
    /// a field: the walk went below the record there, as far as the union's
    /// choice can tell without explaining it.
    inside: bool,
    /// Where the explaining walk may resume, and what the deciding walk had
    /// counted by then. `None` when no by-keys reading ran.
    resume: Option<Resume>,
}

/// The point the deciding walk reached.
#[derive(Clone, Copy)]
struct Resume {
    /// The first field position the explaining walk must read for itself. Every
    /// earlier field was read and did not refuse.
    position: usize,
    /// Declared keys found before that position, so the count that lets a
    /// complete record skip the trailing scan survives the resumption. Losing it
    /// is what sends a record to a scan that reports what the count would have.
    present: usize,
}

impl Decided {
    /// How far the deciding walk got before it refused: the fields or entries
    /// it admitted, then whether it refused inside a structured field's value.
    /// A union explains the record branch that got furthest.
    pub(super) fn progress(&self) -> (usize, bool) {
        (self.admitted, self.inside)
    }

    /// Where to start, and with what count, in a value still holding `entries`
    /// keys.
    fn resume_at(&self, entries: usize) -> (usize, usize) {
        match self.resume {
            Some(at) if self.entries == entries => (at.position, at.present),
            _ => (0, 0),
        }
    }
}

/// Membership for a keyed map: a record or a mapping, read as a dict.
///
/// The fast check decides; on failure an explaining walk re-reads in declared
/// order to report every violation, resuming where the first pass stopped
/// rather than re-reading the fields it found to match.
///
/// Inlined into the reader of a list of records, which asks it of every
/// element: called there, a list of two hundred records of three fields reads
/// 6.5% to 7% more instructions on the profiled 3.12 and 3.14 wheels.
/// [`member`] asks it through [`check_keyed_map_apart`].
#[expect(
    clippy::inline_always,
    reason = "measured: called from the list reader, a list of records read \
              6.5% to 7% more instructions"
)]
#[inline(always)]
pub(super) fn check_keyed_map(
    fields: &[Field],
    defaults: &[MapClause],
    value: &Value<'_, '_>,
    frame: &mut Frame<'_, '_>,
) -> bool {
    let ctx = frame.ctx;
    let mut decided = Decided::default();
    let keep = ctx.mode.explains().then_some(&mut decided);
    let ok = keyed_map_matches(fields, defaults, value, ctx, keep);
    if !ok && ctx.mode.explains() {
        let before = frame.out.len();
        keyed_map_explain(fields, defaults, value, frame, &decided);
        if frame.out.len() == before {
            // Two passes read the same dict and disagreed, so the dict did not
            // stay still between them: report that rather than a failure with
            // nothing behind it.
            mutated(value, frame);
        }
    }
    ok
}

/// [`check_keyed_map`] in a frame of its own, for [`member`]'s arm.
///
/// The walk is one function under fat LTO with its arms inlined, so code
/// inlined into an arm lays out every shape crossing the walk anew: inlined
/// there, the record's check took a list nested twenty-five deep 10% more
/// cycles than behind this call in the profiled 3.14 wheel, and 16% more in
/// the 3.12 one. A record the walk reaches alone pays the call, about
/// thirty-six instructions and 3% of a three-field record's cycles on 3.14; a
/// list of records reads its elements inlined.
#[inline(never)]
pub(super) fn check_keyed_map_apart(
    fields: &[Field],
    defaults: &[MapClause],
    value: &Value<'_, '_>,
    frame: &mut Frame<'_, '_>,
) -> bool {
    check_keyed_map(fields, defaults, value, frame)
}

/// Membership for a keyed map: named fields, then a default clause for every
/// other key. The walk is inverted — it visits each entry once — and a JSON
/// object's keys are strings, a duplicate keeping its last value as
/// `json.loads` does.
pub(super) fn keyed_map_matches(
    fields: &[Field],
    defaults: &[MapClause],
    value: &Value<'_, '_>,
    ctx: Ctx<'_>,
    decided: Option<&mut Decided>,
) -> bool {
    match value {
        Value::Py(v) => keyed_map_matches_py(fields, defaults, v, ctx, decided),
        Value::Json(py, JsonValue::Object(entries)) => {
            keyed_map_matches_json(fields, defaults, *py, entries, ctx)
        }
        Value::Json(..) => false,
    }
}

/// Whether a field's `value` belongs to its `schema`, as the deciding walk in
/// `sub` answers: a scalar field is its type test, asked without the walk
/// around it ([`scalar_member`]), and any other field is walked.
///
/// A record's fields are mostly scalars -- a `TypedDict` of `str`, `int` and
/// `float` -- and each paid `member`'s dispatch, its level and its signal check
/// around the one test. `room` is whether a level is free under the record,
/// read once by the caller: the walk does not move between fields.
#[inline]
fn field_holds(
    schema: &Schema,
    value: &Value<'_, '_>,
    room: bool,
    sub: &mut Frame<'_, '_>,
) -> bool {
    scalar_member(schema, value, sub.ctx, room).unwrap_or_else(|| member(schema, value, sub))
}

/// Whether `(key, val)` is covered by some default clause: the key belongs to a
/// clause's key schema and the value to that clause's value schema. The clauses
/// denote a union of key×value rectangles.
///
/// Decided in `sub`, a fast walk's frame the caller holds for the whole map:
/// a fast walk writes neither of its buffers, but each is a value with a
/// destructor, and a pair built and dropped per key is work the answer does
/// not depend on. Each half is a field's question ([`field_holds`]), so a
/// scalar key or value is its type test -- `dict[str, int]` reads both halves
/// of every entry that way -- and the frame's context is read where it lies
/// rather than copied once an entry.
///
/// Two clauses can read one entry, so each clause after the first is one more
/// of the map's walks of it, armed as a union's next branch is ([`again`]).
/// A map of one clause, which is every `dict[K, V]`, asks it alone.
fn covered(
    defaults: &[MapClause],
    key: &Value<'_, '_>,
    val: &Value<'_, '_>,
    sub: &mut Frame<'_, '_>,
) -> bool {
    let room = sub.ctx.room_to_descend();
    let holds = |clause: &MapClause, sub: &mut Frame<'_, '_>| {
        field_holds(&clause.key, key, room, sub) && field_holds(&clause.value, val, room, sub)
    };
    if let [clause] = defaults {
        return holds(clause, sub);
    }
    let node = defaults.as_ptr() as usize;
    let entered = sub.ctx.state.memo.entered();
    defaults
        .iter()
        .any(|clause| again(node, entered, sub.ctx, || holds(clause, sub)))
}

/// A record's declared names, the two ways [`declared_position`] reads them:
/// by text, and as the interned keys the record's plan holds, where it holds
/// them.
struct Names<'a, F> {
    by_text: F,
    interned: Option<&'a [Py<PyString>]>,
}

/// The declared field a key resolves to, read the way the dict resolves one.
///
/// An exact `str` is every key a caller writes, and its text is the name,
/// looked up through `by_name`. Any other key is the field a dict would find
/// it under: the declared name whose hash equals the key's and which the key
/// equals, the two questions a lookup asks, the key's `__eq__` first as a dict
/// asks it of the key it holds. That reads a `str` subclass with a `__hash__`
/// or `__eq__` of its own, and an object of another type equal to a name, as
/// the field they reach. Read any other way, the scan and the reading by keys
/// answer differently for one value -- which of them runs is decided by
/// whether a clause reads the undeclared keys -- and the explaining walk, which
/// asks the dict, finds nothing to report and names a mutation.
///
/// The relations read a field's key as a `str`, so a key of another type that
/// reaches a name is outside them, which `docs/14-soundness.md` states.
///
/// The names come from the plan's interned keys where it holds them, which
/// carry their hashes, and are built from the field otherwise. The questions
/// run the key's own code: an error reads as not that field, and a fatal
/// signal is recorded.
fn declared_position(
    key: &Bound<'_, PyAny>,
    fields: &[Field],
    names: &Names<'_, impl Fn(&str) -> Option<usize>>,
    ctx: Ctx<'_>,
) -> Option<usize> {
    let py = key.py();
    if let Ok(text) = key.cast::<PyString>()
        && text.is_exact_instance_of::<PyString>()
    {
        return match text.to_str() {
            Ok(name) => (names.by_text)(name),
            Err(err) => {
                record_if_fatal(err, py, ctx);
                None
            }
        };
    }
    reached_by(key, fields, names.interned, ctx)
}

/// [`declared_position`] for a key that is not an exact `str`: the declared
/// name its hash and `==` reach.
///
/// Out of line, because records rarely hold such a key and every scan reads
/// its entries through the caller: inlined, the search would make the loop over
/// a `dict[str, int]`, which resolves no key at all, 2.6% dearer by count.
#[cold]
#[inline(never)]
fn reached_by(
    key: &Bound<'_, PyAny>,
    fields: &[Field],
    interned: Option<&[Py<PyString>]>,
    ctx: Ctx<'_>,
) -> Option<usize> {
    let py = key.py();
    let own = match key.hash() {
        Ok(own) => own,
        Err(err) => {
            record_if_fatal(err, py, ctx);
            return None;
        }
    };
    fields.iter().enumerate().position(|(position, field)| {
        let name = interned.and_then(|keys| keys.get(position)).map_or_else(
            || PyString::new(py, &field.name),
            |key| key.bind(py).clone(),
        );
        let reached = name
            .hash()
            .and_then(|theirs| Ok(theirs == own && key.eq(&name)?));
        fold(reached, py, ctx)
    })
}

/// What a record's clauses say about a key it does not declare, where that can
/// be said without reading the key and its value together.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Undeclared {
    /// No clause: a closed record, and a key to spare refuses it.
    Refused,
    /// The top clause, `anything: anything`: any key, any value, by definition.
    Admitted,
    /// `str: anything` -- the clause a `TypedDict` carries, since its keys are
    /// strings by the typing spec -- admits a key exactly when the key is a
    /// `str`, whatever its value. Every declared name is a `str` too, so a
    /// `str` key needs no resolving; a key of another type can still reach a
    /// declared name, as a dict resolves keys, and is asked whether it does.
    AnyStr,
    /// A clause that reads the key or the value: each undeclared key is asked of
    /// it, which is the scan.
    Read,
}

impl Undeclared {
    /// The reading, from the clause list alone.
    ///
    /// Asked once per record **value** rather than once per record node -- a
    /// document of a hundred objects asks it a hundred times -- so it is a
    /// match over the clause's two constructors rather than a comparison
    /// against schemas built to be compared against. `anything` and
    /// `typing.Any` are one node wearing two spellings and
    /// [`Spelling`](valgebra_core::Spelling) makes them equal, so the variant
    /// alone decides this: the payload is bound to `_` and never read, which
    /// is what keeps the spelling out of the answer.
    fn of(defaults: &[MapClause]) -> Undeclared {
        match defaults {
            [] => Undeclared::Refused,
            [clause]
                if matches!(clause.key, Schema::Anything(_))
                    && matches!(clause.value, Schema::Anything(_)) =>
            {
                Undeclared::Admitted
            }
            [clause]
                if matches!(clause.key, Schema::Str)
                    && matches!(clause.value, Schema::Anything(_)) =>
            {
                Undeclared::AnyStr
            }
            _ => Undeclared::Read,
        }
    }
}

/// A record's membership, asked key by key rather than read entry by entry, or
/// `None` where that reading does not settle it.
///
/// A dict cannot repeat a key, so the value holds nothing but declared keys
/// exactly when the count of declared keys found equals the count it holds.
/// Asking for the declared keys costs one probe each with the key's own hash,
/// where scanning the value costs an iteration step, a decode of the key's
/// bytes, a second hash of those bytes and a comparison against the name they
/// matched. What the value holds beyond them is answered by the clause alone
/// when [`Undeclared::of`] can read it without the key's value.
///
/// A key is resolved the way Python resolves one -- by the dict's own lookup --
/// rather than by decoding its bytes and matching those, so a key of a `str`
/// subclass with an `__eq__` of its own is found exactly where indexing the
/// dict would find it.
///
/// `None` means "ask the scan instead": a clause reads an undeclared key
/// together with its value; the plan has no interned key for a field; or a
/// probe raised something other than a fatal signal, which is not an answer. A
/// value that changes size under the probes is not one of those: it is answered
/// here, as the scan answers it, because there is no reading of it left to fall
/// back to.
fn keyed_map_asks_for_its_keys(
    fields: &[Field],
    defaults: &[MapClause],
    dict: &Bound<'_, PyDict>,
    ctx: Ctx<'_>,
    plan: &RecordPlan,
    decided: Option<&mut Decided>,
) -> Option<bool> {
    // By its keys where the keys settle it. A record whose clause reads an
    // undeclared key together with its value has to be scanned, and the scan
    // walks the declared fields as it goes, so reading those by key first would
    // be work done twice; such a record is not read here.
    let undeclared = Undeclared::of(defaults);
    if undeclared == Undeclared::Read || plan.keys.len() != fields.len() {
        return None;
    }
    // One pair of scratch buffers for the record, as the scan takes: a fast
    // walk writes to neither. Dropped as ordinary vectors, unlike the scan's
    // `Unwritten` pair: without the drop, the release build lays this loop out
    // three instructions a field dearer -- 1.4% of `--binding-record` -- which
    // outweighs the call the drop costs a small record.
    let (mut path, mut out) = (Vec::new(), Vec::new());
    let mut sub = Frame::new(&mut path, &mut out, fast(ctx));
    with_critical_section(dict.as_any(), || {
        let entries = dict.len();
        // Where the explaining walk would resume if this one refuses, and how
        // far this one got, kept as it goes so that refusing costs nothing
        // extra to record.
        let stopped = |admitted: usize, position: usize, present: usize, inside: bool| {
            if let Some(keep) = decided {
                keep.entries = entries;
                keep.admitted = admitted;
                keep.inside = inside;
                keep.resume = Some(Resume { position, present });
            }
        };
        let room = ctx.room_to_descend();
        // A field decided by its own value and declared after one that is not
        // -- a tag beside a recursive child -- is read first, so a branch the
        // tag refuses walks nothing below it ([`RecordPlan::early`]). A refusal
        // here read no declared prefix, and the explaining walk starts at the
        // first field.
        for (passed, &position) in plan.early.iter().enumerate() {
            let field = fields.get(position)?;
            let key = plan.keys.get(position)?.bind(dict.py());
            let refused = match dict.get_item(key) {
                Ok(Some(value)) => !field_holds(&field.schema, &Value::Py(&value), room, &mut sub),
                Ok(None) => field.required,
                Err(err) if is_fatal(&err, dict.py()) => {
                    record_fatal(err, ctx);
                    return Some(false);
                }
                Err(_) => return None,
            };
            if refused {
                stopped(passed, 0, 0, false);
                return Some(false);
            }
        }
        // How far a refusal at `position` got: the declared fields before it,
        // and the early ones after it, which were read first.
        let admitted = |position: usize| {
            position + plan.early.iter().filter(|&&early| early > position).count()
        };
        let mut present = 0usize;
        for (position, field) in fields.iter().enumerate() {
            let key = plan.keys.get(position)?.bind(dict.py());
            match dict.get_item(key) {
                Ok(Some(value)) => {
                    present += 1;
                    if !field_holds(&field.schema, &Value::Py(&value), room, &mut sub) {
                        let inside = !decided_by_its_value(&field.schema);
                        stopped(admitted(position), position, present - 1, inside);
                        return Some(false);
                    }
                }
                // A key the value does not carry: the record still matches when
                // the field is optional.
                Ok(None) if !field.required => {}
                Ok(None) => {
                    stopped(admitted(position), position, present, false);
                    return Some(false);
                }
                // A failed probe is not an answer -- an unhashable key cannot be
                // in a dict, but a `__eq__` that raises can stop the lookup --
                // and the scan answers instead, unless what stopped it is a fatal
                // signal, which the scan would not see raised again.
                Err(err) if is_fatal(&err, dict.py()) => {
                    record_fatal(err, ctx);
                    return Some(false);
                }
                Err(_) => return None,
            }
        }
        stopped(fields.len(), fields.len(), present, false);
        if dict.len() != entries {
            // The value changed while it was being read, so there is no reading
            // to answer from: not a member, exactly as the scan answers it, and
            // the explain pass names the mutation.
            record_move(ctx);
            return Some(false);
        }
        // Every key the value carries is one of the declared ones exactly when
        // the count of declared keys found equals the count it holds; there is
        // then nothing for a clause to govern, whatever the clause.
        if present == entries {
            return Some(true);
        }
        match undeclared {
            Undeclared::Refused => Some(false),
            Undeclared::Admitted => Some(true),
            // A `str` key is admitted whatever it names, so it is read for its
            // type and nothing else, and no value is walked. A key of another
            // type is a declared field where it reaches one, as the probes above
            // found it, and the first that reaches none refuses the record: the
            // scan stops at the first such key, and [`other_keys_reach_names`]
            // reads the entries again.
            Undeclared::AnyStr => {
                let keys = scan_dict(dict, ctx.state, |key, _| {
                    if key.is_instance_of::<PyString>() {
                        ControlFlow::Continue(())
                    } else {
                        ControlFlow::Break(())
                    }
                });
                match keys {
                    Scan::Stopped => Some(other_keys_reach_names(dict, fields, plan, &sub.ctx)),
                    _ => Some(read_whole(&keys, ctx)),
                }
            }
            // Kept out by the guard above; were it not, the scan still decides,
            // at the cost of the reading just done.
            Undeclared::Read => None,
        }
    })
}

/// Whether every key of `dict` that is not a `str` reaches a declared name, for
/// a record whose clause admits any `str` key ([`Undeclared::AnyStr`]).
///
/// The second reading of a record whose first stopped at such a key, which
/// records rarely hold. Out of line, and handed the frame's context, which
/// already lives in memory, rather than `member`'s: the first reading then
/// stays a visitor that captures nothing, where one carrying the context and
/// the record's names to the scan cost a list of records 2.5% on the profiled
/// wheel by count, and one holding a copy of them 3% where the records carry an
/// undeclared key.
#[cold]
#[inline(never)]
fn other_keys_reach_names(
    dict: &Bound<'_, PyDict>,
    fields: &[Field],
    plan: &RecordPlan,
    ctx: &Ctx<'_>,
) -> bool {
    let ctx = *ctx;
    let keys = scan_dict(dict, ctx.state, |key, _| {
        if key.is_instance_of::<PyString>()
            || reached_by(key, fields, Some(&plan.keys), ctx).is_some()
        {
            ControlFlow::Continue(())
        } else {
            ControlFlow::Break(())
        }
    });
    read_whole(&keys, ctx)
}

/// The keyed-map fast path over a Python dict. A string key naming a declared
/// field is checked against it; any other key (non-string, or undeclared) must
/// be covered by a default clause. Closed records have no clauses, so an
/// undeclared key is rejected; an open record's `anything` clause covers it.
///
/// The declared-field lookup comes from the validator's precomputed
/// [`RecordIndex`](crate::check::index::RecordIndex) when present, so a wide
/// record skips rebuilding its name map on every call; a record not in the
/// index (an empty one, or a node the build-time traversal did not reach) falls
/// back to building the map here.
fn keyed_map_matches_py(
    fields: &[Field],
    defaults: &[MapClause],
    dict: &Bound<'_, PyAny>,
    ctx: Ctx<'_>,
    decided: Option<&mut Decided>,
) -> bool {
    let Ok(dict) = dict.cast::<PyDict>() else {
        return false;
    };
    #[cfg(PyPy)]
    let Some(ref dict) = stored(dict, ctx) else {
        return false;
    };
    let mut decided = decided;
    if fields.is_empty()
        && let Some(answered) = scalar_map_matches(defaults, dict, ctx, decided.as_deref_mut())
    {
        return answered;
    }
    if let Some(plan) = ctx.index.records.get(&(fields.as_ptr() as usize)) {
        if let Some(answered) =
            keyed_map_asks_for_its_keys(fields, defaults, dict, ctx, plan, decided.as_deref_mut())
        {
            return answered;
        }
        keyed_map_scan(
            fields,
            defaults,
            dict,
            ctx,
            plan.required,
            decided,
            &Names {
                by_text: |name: &str| plan.by_name.get(name).copied(),
                interned: Some(&plan.keys),
            },
        )
    } else {
        let declared: FxHashMap<&str, usize> = fields
            .iter()
            .enumerate()
            .map(|(i, f)| (&*f.name, i))
            .collect();
        let required = fields.iter().filter(|f| f.required).count();
        keyed_map_scan(
            fields,
            defaults,
            dict,
            ctx,
            required,
            decided,
            &Names {
                by_text: |name: &str| declared.get(name).copied(),
                interned: None,
            },
        )
    }
}

/// A mapping of one clause whose key and value are each a scalar kind --
/// `dict[str, int]` -- decided as two type tests an entry, or `None` where the
/// general scan must read it.
///
/// [`keyed_map_scan`] reads such a mapping through [`covered`], which asks
/// each half of an entry as a field ([`field_holds`]): the schema's kind, the
/// level, the fatal-signal flag and the clause count are read again at every
/// entry, around the two tests that decide it. Read once here, a hundred-entry
/// `dict[str, int]` takes about forty-five fewer instructions an entry on the
/// profiled 3.12 and 3.14 wheels, and about fifty on the free-threaded ones,
/// whose profiled scan lays those reads out dearest.
///
/// The answers are the scan's. A type test runs no Python, so nothing an entry
/// does can record a signal or move the dict, and the scan, its count and its
/// stop are [`keyed_map_scan`]'s own: a dict another thread moves reads as
/// unreadable there and here. A recorded signal declines to the scan, which
/// refuses every entry, and so does a map with no level free under it, where
/// the scan walks each half through [`member`] to refuse it at the bound and
/// record why. What the deciding walk keeps for a union is what the scan
/// keeps: the entries admitted before the refusal, and a refusal that is never
/// inside a field.
#[expect(
    clippy::redundant_closure_for_method_calls,
    reason = "the loop takes a test over a `Value` of any lifetime, and the \
              method path names one lifetime; each closure is the method, \
              generic over the lifetime the loop hands it"
)]
pub(super) fn scalar_map_matches(
    defaults: &[MapClause],
    dict: &Bound<'_, PyDict>,
    ctx: Ctx<'_>,
    decided: Option<&mut Decided>,
) -> Option<bool> {
    let [clause] = defaults else {
        return None;
    };
    let (key_kind, value_kind) = (scalar_of(&clause.key)?, scalar_of(&clause.value)?);
    if !ctx.room_to_descend() || ctx.state.signals.fatal_seen.get() {
        return None;
    }
    let state = ctx.state;
    let (admitted, scan) = match (key_kind, value_kind) {
        (Scalar::Str, Scalar::Int) => {
            scalar_map_loop(dict, state, |key| key.is_str(), |value| value.is_int())
        }
        _ => scalar_map_loop(
            dict,
            state,
            |key| scalar_admits(key_kind, key),
            |value| scalar_admits(value_kind, value),
        ),
    };
    if let Some(keep) = decided {
        keep.admitted = admitted;
        keep.inside = false;
    }
    Some(read_whole(&scan, ctx))
}

/// The scan [`scalar_map_matches`] reads a mapping with, asking `key_admits`
/// and `value_admits` of each entry: how many entries both admitted, and how
/// the scan ended.
///
/// Generic over the two tests, so the pair the instruction gate counts -- a
/// `str` key with an `int` value -- is a loop of its own with each test written
/// where it is asked, as a list of one kind is
/// ([`scalar_list_matches`](super::sequence)). Asked through [`scalar_admits`],
/// each half of every entry dispatches on its kind: four or five instructions
/// an entry more on the profiled 3.12 and 3.14 wheels.
fn scalar_map_loop(
    dict: &Bound<'_, PyDict>,
    state: &WalkState,
    key_admits: impl Fn(&Value<'_, '_>) -> bool,
    value_admits: impl Fn(&Value<'_, '_>) -> bool,
) -> (usize, Scan) {
    let mut admitted = 0usize;
    let scan = scan_dict(dict, state, |key, val| {
        if key_admits(&Value::Py(key)) && value_admits(&Value::Py(val)) {
            admitted += 1;
            ControlFlow::Continue(())
        } else {
            ControlFlow::Break(())
        }
    });
    (admitted, scan)
}

/// Walk a dict once against a record's fields, resolving each string key to a
/// declared-field index through `lookup` (a precomputed plan or a freshly built
/// map). A key that resolves checks its value against that field; any other key
/// must be covered by a default clause. The record matches iff every entry
/// matches and every required field was seen. How many entries it admitted is
/// kept in `decided`, where a union asks, as the by-keys reading keeps how many
/// fields it admitted.
fn keyed_map_scan(
    fields: &[Field],
    defaults: &[MapClause],
    dict: &Bound<'_, PyDict>,
    ctx: Ctx<'_>,
    mut required_remaining: usize,
    decided: Option<&mut Decided>,
    names: &Names<'_, impl Fn(&str) -> Option<usize>>,
) -> bool {
    // Scratch buffers for the whole record, not one pair per field: a fast walk
    // writes to neither, and a pair per field would build and drop a hundred of
    // them for a fifty-field record, to answer one membership question.
    let mut unwritten = Unwritten::default();
    let mut sub = unwritten.frame(ctx);
    let room = ctx.room_to_descend();
    let mut admitted = 0usize;
    let mut inside = false;
    let scan = scan_dict(dict, ctx.state, |key, val| {
        // A key no declared name is reached by -- among them a string carrying
        // a lone surrogate, which cannot equal a name, since names are valid
        // UTF-8 by build-time check -- must instead be covered by a default
        // clause. A mapping declares no field, so no key resolves to one, and
        // its keys are not read as names: decoding each and asking its type
        // would be a fifth of reading a `dict[str, int]`.
        let index = if fields.is_empty() {
            None
        } else {
            declared_position(key, fields, names, ctx)
        };
        match index.and_then(|i| fields.get(i)) {
            Some(field) => {
                if !field_holds(&field.schema, &Value::Py(val), room, &mut sub) {
                    inside = !decided_by_its_value(&field.schema);
                    return ControlFlow::Break(());
                }
                if field.required {
                    // Saturating: the counter is the precomputed required-field
                    // count, so it cannot legitimately pass zero, but a malformed
                    // index must not wrap a release build into a false pass.
                    required_remaining = required_remaining.saturating_sub(1);
                }
            }
            None => {
                if !covered(defaults, &Value::Py(key), &Value::Py(val), &mut sub) {
                    return ControlFlow::Break(());
                }
            }
        }
        admitted += 1;
        ControlFlow::Continue(())
    });
    if let Some(keep) = decided {
        keep.admitted = admitted;
        keep.inside = inside;
    }
    read_whole(&scan, ctx) && required_remaining == 0
}

/// Whether a deciding scan read the dict to the end. One the dict moved under
/// answers a non-member, as one that stopped does, and records the stop: the
/// explaining pass names the move, and a complement over this walk reads it.
fn read_whole(scan: &Scan, ctx: Ctx<'_>) -> bool {
    match scan {
        Scan::Complete => true,
        Scan::Stopped => false,
        Scan::Unreadable => {
            record_move(ctx);
            false
        }
    }
}

/// How wide a record's per-object field table may be and still sit on the
/// stack: sixteen slots of one pointer each.
const FOUND_ON_STACK: usize = 16;

/// The keyed-map fast path over a JSON object. Keys are strings; a duplicate key
/// keeps its last value (a reverse find), as `json.loads` does. Records are
/// small, so a linear scan beats building a per-object map.
pub(super) fn keyed_map_matches_json(
    fields: &[Field],
    defaults: &[MapClause],
    py: Python<'_>,
    entries: &[(Cow<'_, str>, JsonValue<'_>)],
    ctx: Ctx<'_>,
) -> bool {
    let mut unwritten = Unwritten::default();
    let mut sub = unwritten.frame(ctx);
    let room = ctx.room_to_descend();
    // A record whose keys settle it resolves the document's keys through the
    // plan instead of searching the document once per field. The search is
    // quadratic in the width -- a fifty-field record read a fifty-entry object
    // fifty times -- and the plan already holds the name-to-position map the
    // resolution wants. A closed record refuses an undeclared key; the top
    // clause admits it, and so does `str: anything`, since a JSON key is a
    // string. A record whose clause reads a key asks it of each undeclared key
    // once the fields are read, as the search below does, where it is wide
    // enough to take the plan at all ([`resolves_through_the_plan`]).
    let undeclared = Undeclared::of(defaults);
    if let Some(plan) = ctx
        .index
        .records
        .get(&(fields.as_ptr() as usize))
        .filter(|plan| {
            plan.by_name.len() == fields.len()
                && resolves_through_the_plan(undeclared, fields.len())
        })
    {
        // The document's value for each declared field, last occurrence winning
        // as `json.loads` does, gathered before any of them is checked: an
        // earlier duplicate that fails is not the entry the document means.
        //
        // Held on the stack for a record as wide as `FOUND_ON_STACK`, which is
        // most of them: the table is asked for once per object in a document,
        // and a heap table is a trip to the allocator and back per object.
        let mut on_stack: [Option<&JsonValue<'_>>; FOUND_ON_STACK] = [None; FOUND_ON_STACK];
        let mut on_heap: Vec<Option<&JsonValue<'_>>>;
        let found: &mut [Option<&JsonValue<'_>>] =
            if let Some(slots) = on_stack.get_mut(..fields.len()) {
                slots
            } else {
                on_heap = vec![None; fields.len()];
                &mut on_heap
            };
        for (key, value) in entries {
            match plan.by_name.get(key.as_ref()) {
                Some(&at) => match found.get_mut(at) {
                    Some(slot) => *slot = Some(value),
                    // The plan and the field list disagree about a position,
                    // which the filter above rules out; answer conservatively
                    // rather than indexing.
                    None => return false,
                },
                // A closed record has no clause to cover an undeclared key; an
                // open one's top clause covers it by definition, and a clause
                // that reads keys is asked below.
                None if undeclared == Undeclared::Refused => return false,
                None => {}
            }
        }
        // The early fields first, as the object path reads them.
        for &position in &plan.early {
            let (Some(field), Some(value)) = (fields.get(position), found.get(position)) else {
                return false;
            };
            match value {
                Some(value) => {
                    if !field_holds(&field.schema, &Value::Json(py, value), room, &mut sub) {
                        return false;
                    }
                }
                None if field.required => return false,
                None => {}
            }
        }
        for (field, value) in fields.iter().zip(found.iter().copied()) {
            match value {
                Some(value) => {
                    if !field_holds(&field.schema, &Value::Json(py, value), room, &mut sub) {
                        return false;
                    }
                }
                None if field.required => return false,
                None => {}
            }
        }
        // The context is `sub`'s, which already lives in memory, rather than
        // `ctx`: a context goes to a call out of line by address, and `member`
        // inlines this function, so passing `ctx` puts `member`'s own copy in
        // memory and copies the whole context in at every node -- a nested list
        // on the profiled wheel reads 2.2% dearer by count.
        if undeclared == Undeclared::Read {
            return undeclared_covered(defaults, py, entries, sub.ctx, |name| {
                plan.by_name.contains_key(name)
            });
        }
        return true;
    }
    for field in fields {
        match entries
            .iter()
            .rev()
            .find(|(key, _)| &*field.name == key.as_ref())
        {
            Some((_, val)) => {
                if !field_holds(&field.schema, &Value::Json(py, val), room, &mut sub) {
                    return false;
                }
            }
            None if field.required => return false,
            None => {}
        }
    }
    // Every key that is not a declared field must be covered by a default clause,
    // testing each key's last value (json.loads semantics).
    //
    // Whether a key is a declared field is a question about the *schema*, so it
    // is answered from the record plan built once per validator rather than from
    // a name set rebuilt per object. A schema absent from the plan falls back to
    // scanning the field list, so correctness never depends on the plan being
    // complete.
    let plan = ctx.index.records.get(&(fields.as_ptr() as usize));
    let declares = |name: &str| match plan {
        Some(plan) => plan.by_name.contains_key(name),
        None => fields.iter().any(|f| &*f.name == name),
    };
    // A closed record has no clause to cover an undeclared key with, so the first
    // one decides and there is nothing to collapse.
    if defaults.is_empty() {
        return entries.iter().all(|(key, _)| declares(key.as_ref()));
    }
    undeclared_covered(defaults, py, entries, ctx, declares)
}

/// Whether a parsed object meets a record of `fields` declared fields through
/// the record's plan, rather than by a search of the object per field.
///
/// Every record but one whose clause reads keys, and that one where it declares
/// more fields than [`SMALL_OBJECT`]: the clause hashes every key a second time
/// to tell the undeclared ones, and against a few fields the comparisons a
/// search makes cost less than the hashes.
pub(super) fn resolves_through_the_plan(undeclared: Undeclared, fields: usize) -> bool {
    undeclared != Undeclared::Read || fields > SMALL_OBJECT
}

/// Whether every key of a parsed object that no field declares is covered by a
/// default clause, testing each key's last value (`json.loads` semantics).
///
/// Split from [`keyed_map_matches_json`] because it is the half with two
/// readings of its own -- which clause question to ask, and where to answer the
/// duplicate-key question -- and the caller is the field walk.
fn undeclared_covered(
    defaults: &[MapClause],
    py: Python<'_>,
    entries: &[(Cow<'_, str>, JsonValue<'_>)],
    ctx: Ctx<'_>,
    declares: impl Fn(&str) -> bool,
) -> bool {
    // A parsed object's keys are strings by construction, so a lone clause whose
    // key schema admits every string governs every undeclared key by its value
    // alone: the key half of the coverage question is already answered, and
    // asking it builds a `JsonValue` and walks a schema per key for a fixed yes.
    // `dict[str, V]` -- the shape a document's free-form section is written as
    // -- is exactly this clause.
    let value_only = match defaults {
        [clause] if matches!(clause.key, Schema::Str | Schema::Anything(_)) => Some(&clause.value),
        _ => None,
    };
    let mut unwritten = Unwritten::default();
    let mut sub = unwritten.frame(ctx);
    let mut covers = |key: &str, val: &JsonValue<'_>| {
        if let Some(schema) = value_only {
            return member(schema, &Value::Json(py, val), &mut sub);
        }
        let key_value = JsonValue::Str(Cow::Borrowed(key));
        covered(
            defaults,
            &Value::Json(py, &key_value),
            &Value::Json(py, val),
            &mut sub,
        )
    };
    // A narrow object is covered where it lies. The entry a document means by a
    // key is its last, and an entry is that one exactly when no entry after it
    // repeats the key -- which is a look forward, not a map of the whole
    // object. The map below answers the same question by allocating a table and
    // hashing every key into it, and a mapping of one or two keys pays that in
    // full: the free-form section of a record is written `dict[str, V]` and
    // usually carries a handful.
    if entries.len() <= SMALL_OBJECT {
        for (position, (key, val)) in entries.iter().enumerate() {
            if declares(key.as_ref())
                || entries
                    .iter()
                    .skip(position + 1)
                    .any(|(later, _)| later.as_ref() == key.as_ref())
            {
                continue;
            }
            if !covers(key.as_ref(), val) {
                return false;
            }
        }
        return true;
    }
    // Which entries the document means, found in one pass, so a document with
    // many keys (or many duplicates) is covered linearly rather than by
    // rescanning the tail per key: an entry a field declares is not this
    // check's, and one a later entry repeats is not the document's -- the table
    // hands back the position a repeat displaces. The table is keyed by the
    // document's own keys, so it hashes them with a hash keyed per process:
    // `rustc-hash` is unseeded, its collisions are the same in every process,
    // and a document can carry keys computed offline to all hash alike -- each
    // insert would then walk the whole chain, and 20,000 such keys, 0.74 MB,
    // take half a second where ordinary ones take 2 ms. The entries are covered
    // in the document's order, so the clauses are asked in an order the seed
    // does not choose.
    let mut last: HashMap<&str, usize, RandomState> =
        HashMap::with_capacity_and_hasher(entries.len(), RandomState::new());
    let mut skipped = vec![false; entries.len()];
    for (position, (key, _)) in entries.iter().enumerate() {
        let displaced = if declares(key.as_ref()) {
            Some(position)
        } else {
            last.insert(key.as_ref(), position)
        };
        if let Some(slot) = displaced.and_then(|at| skipped.get_mut(at)) {
            *slot = true;
        }
    }
    entries
        .iter()
        .zip(&skipped)
        .all(|((key, val), &skipped)| skipped || covers(key.as_ref(), val))
}

/// How many entries an object may carry and still be covered where it lies.
///
/// The look forward is quadratic in the entries and the map is linear with a
/// table and a hash per key, so the two cross somewhere -- but the constant on
/// the map is an allocation, and the comparisons the look forward makes are
/// mostly a length test that fails. The bound is set low enough that the
/// crossover is not the question: an object wider than this is a document's
/// payload rather than a record's free-form section, and pays for the table it
/// then uses.
///
/// The same crossing decides how a parsed object meets a record whose clause
/// reads keys: a record declaring no more fields than this searches the object
/// for each, and a wider one resolves the object's keys through its plan
/// ([`keyed_map_matches_json`]).
pub(super) const SMALL_OBJECT: usize = 8;

/// How many keys a parsed object holds, counted as the dict `json.loads`
/// builds from it counts them: a key the document repeats is one key.
///
/// Counted where the object lies, by the two readings [`undeclared_covered`]
/// finds a document's last entries with: a look forward up to
/// [`SMALL_OBJECT`] entries, and past it a table hashed with a key per
/// process, for the reason given there. The dict is built from these same
/// decoded keys, so the two counts are one count.
pub(super) fn distinct_keys(entries: &[(Cow<'_, str>, JsonValue<'_>)]) -> usize {
    if entries.len() <= SMALL_OBJECT {
        return entries
            .iter()
            .enumerate()
            .filter(|(position, (key, _))| {
                !entries
                    .iter()
                    .skip(position + 1)
                    .any(|(later, _)| later.as_ref() == key.as_ref())
            })
            .count();
    }
    let mut seen: HashSet<&str, RandomState> =
        HashSet::with_capacity_and_hasher(entries.len(), RandomState::new());
    entries
        .iter()
        .filter(|(key, _)| seen.insert(key.as_ref()))
        .count()
}

/// The explain pass over a keyed map, run only after [`keyed_map_matches`] has
/// reported the value is not a member. It walks in declared order — present
/// fields checked in order, then absent required keys — then reports each
/// undeclared key: an uncovered key with no clauses reads as an unexpected key,
/// and with clauses its key and value are checked against the first clause.
pub(super) fn keyed_map_explain(
    fields: &[Field],
    defaults: &[MapClause],
    value: &Value<'_, '_>,
    frame: &mut Frame<'_, '_>,
    decided: &Decided,
) {
    let ctx = frame.ctx;
    let Value::Py(v) = value else {
        // The explain pass only ever sees a Python value; a JSON value here is
        // unreachable, but keep the false-implies-a-violation invariant.
        frame
            .out
            .push(type_mismatch(DICT_TYPE, "dict", value, frame.path, ctx));
        return;
    };
    let Ok(dict) = v.cast::<PyDict>() else {
        frame
            .out
            .push(type_mismatch(DICT_TYPE, "dict", value, frame.path, ctx));
        return;
    };
    #[cfg(PyPy)]
    let Some(ref dict) = stored(dict, ctx) else {
        mutated(value, frame);
        return;
    };
    // The interned keys, in field order. Asking the dict by Rust text decodes a
    // fresh `PyString` and hashes it before the probe can start, once per field
    // per call; an interned key carries its hash. A schema absent from the index
    // falls back to its own text, so correctness never depends on the plan being
    // complete -- the two spellings name the same key.
    let interned = ctx.index.records.get(&(fields.as_ptr() as usize));
    let entries = dict.len();
    // Every field the deciding walk read before it refused matched, so it has no
    // violation to report and nothing is learned by probing it again. Resuming
    // takes its count with it: the count is what lets a record holding exactly
    // its own keys skip the scan below, and a walk that lost it falls into one.
    let (from, mut present) = decided.resume_at(entries);
    for (position, field) in fields.iter().enumerate().skip(from) {
        let key = interned.and_then(|plan| plan.keys.get(position));
        let found = match key {
            Some(interned) => dict.get_item(interned.bind(dict.py())),
            None => dict.get_item(&*field.name),
        };
        match found {
            Ok(Some(item)) => {
                present += 1;
                frame.path.push(PathSegment::Key(Arc::clone(&field.name)));
                member(&field.schema, &Value::Py(&item), frame);
                frame.path.pop();
            }
            Ok(None) if field.required => frame.out.push(located(
                frame.path,
                Arc::clone(&field.name),
                MISSING_KEY,
                format!("required key {}", quoted(&field.name)),
                "missing".to_owned(),
            )),
            Ok(None) => {}
            // A lookup that raises is a key that will not say whether it is
            // this field -- the comparison-raises rule makes the value a
            // non-member, and the value is a dict all the same. Reporting it as
            // the wrong type sends a reader to the wrong place; a required field
            // the dict cannot be shown to hold is reported missing, as an absent
            // one is. A fatal signal from the comparison is carried out instead.
            Err(err) => {
                if is_fatal(&err, dict.py()) {
                    record_fatal(err, ctx);
                    return;
                }
                if field.required {
                    frame.out.push(located(
                        frame.path,
                        Arc::clone(&field.name),
                        MISSING_KEY,
                        format!("required key {}", quoted(&field.name)),
                        "missing".to_owned(),
                    ));
                }
            }
        }
        if ctx.mode.stops_at_first() && !frame.out.is_empty() {
            return;
        }
    }
    // A record that holds exactly the keys it declares has no undeclared key to
    // find, and the field loop above has already established it: a dict cannot
    // repeat a key, so finding as many declared keys as the value has entries
    // accounts for every one of them. Open or closed -- a clause governs the
    // keys a record does not declare, and there are none. The length is
    // re-read because the walk of a field's value runs Python, which can
    // resize the dict -- and a value that moved under the reading falls through
    // to the scan, which is where a mutation is reported.
    if present == entries && dict.len() == entries {
        return;
    }
    undeclared_explained(fields, defaults, dict, value, frame);
}

/// Report each key a record does not declare and no clause of it covers: the
/// key, and its value, against the clause where the record has one, and the
/// key alone where it is closed.
fn undeclared_explained(
    fields: &[Field],
    defaults: &[MapClause],
    dict: &Bound<'_, PyDict>,
    value: &Value<'_, '_>,
    frame: &mut Frame<'_, '_>,
) {
    let ctx = frame.ctx;
    // Built here rather than in the caller, because the scan is the only reader
    // and a record that answers by count never reaches it.
    let declared: FxHashMap<&str, usize> = fields
        .iter()
        .enumerate()
        .map(|(position, field)| (&*field.name, position))
        .collect();
    let names = Names {
        by_text: |name: &str| declared.get(name).copied(),
        interned: ctx
            .index
            .records
            .get(&(fields.as_ptr() as usize))
            .map(|plan| &plan.keys[..]),
    };
    // Whether a clause covers a key is asked in a fast walk, whose frame is
    // held for the scan rather than built per key.
    let mut unwritten = Unwritten::default();
    let mut deciding = unwritten.frame(ctx);
    let scan = scan_dict(dict, ctx.state, |key, val| {
        if declared_position(key, fields, &names, ctx).is_some() {
            return ControlFlow::Continue(());
        }
        if covered(defaults, &Value::Py(key), &Value::Py(val), &mut deciding) {
            return ControlFlow::Continue(());
        }
        if let Some(clause) = defaults.first() {
            // A clause exists but did not cover this key: surface the key and
            // value violations against it (the homogeneous-mapping error). The
            // two are two failures, so a mode that stops at the first reports
            // the key alone.
            frame.path.push(key_segment(key, ctx));
            member(&clause.key, &Value::Py(key), frame);
            if !ctx.mode.stops_at_first() || frame.out.is_empty() {
                member(&clause.value, &Value::Py(val), frame);
            }
            frame.path.pop();
        } else {
            // A closed record: the key is not allowed. It is named the way every
            // other key in a path is -- an integer key as an integer -- so a
            // caller walking the path back down reaches the entry rather than
            // one spelled like it.
            frame.out.push(at_key(
                frame.path,
                key_segment(key, ctx),
                EXTRA_FORBIDDEN,
                "no unexpected key".to_owned(),
                summarize_value(&Value::Py(key), ctx),
            ));
        }
        if ctx.mode.stops_at_first() && !frame.out.is_empty() {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    });
    // The fast pass reported a non-member; if the dict moved underneath this one
    // there is nothing else to report, and the mutation is the finding.
    if matches!(scan, Scan::Unreadable) {
        mutated(value, frame);
    }
}

pub(super) fn check_attr_record(
    fields: &[Attribute],
    value: &Value<'_, '_>,
    frame: &mut Frame<'_, '_>,
) -> bool {
    let ctx = frame.ctx;
    // Attributes are read off a Python object, so a value that will not
    // materialize into one carries none and belongs to no record. A Python
    // value is borrowed, as the walk holds it, so reading a record writes no
    // reference count on the object.
    let built;
    let obj = match value {
        Value::Py(obj) => *obj,
        Value::Json(..) => match value.to_python() {
            Ok(obj) => {
                built = obj;
                &built
            }
            Err(err) => {
                record_if_fatal(err, value.py(), ctx);
                return false;
            }
        },
    };
    // The interned names, in field order. A schema absent from the index (an
    // incomplete build traversal) falls back to the field's own text, so
    // correctness never depends on the plan being complete.
    let interned = ctx.index.attrs.get(&(fields.as_ptr() as usize));
    // Whether a level is free under the record, which no field's walk moves.
    let room = ctx.room_to_descend();
    let mut ok = true;
    for (position, field) in fields.iter().enumerate() {
        let name = interned.and_then(|plan| plan.names.get(position));
        let attribute = match name {
            Some(interned) => obj.getattr(interned.bind(value.py())),
            None => obj.getattr(&*field.name),
        };
        match attribute {
            Ok(attr) => {
                // An attribute the walk would admit without recording anything
                // -- a scalar that passes its test, a union of scalars one of
                // whose tests does -- is answered by the test, in either mode,
                // before its location is pushed. A scalar's test is asked here
                // and any other field's in `admitted_quietly`, which is out of
                // line and asked of every attribute would be a call a field.
                let attr = Value::Py(&attr);
                let quiet = match scalar_member(&field.schema, &attr, ctx, room) {
                    Some(admits) => admits,
                    None => admitted_quietly(&field.schema, &attr, ctx),
                };
                if quiet {
                    continue;
                }
                if ctx.mode.explains() {
                    frame.path.push(PathSegment::Key(Arc::clone(&field.name)));
                }
                ok &= member(&field.schema, &attr, frame);
                if ctx.mode.explains() {
                    frame.path.pop();
                }
            }
            Err(err) => {
                if attribute_missing(err, field, value.py(), frame).is_break() {
                    return false;
                }
                ok = false;
            }
        }
        if !ok && stop(ctx) {
            return false;
        }
    }
    ok
}

/// The attribute `field` names is missing, its reading having raised `err`,
/// and an explaining walk records it. A break where `err` is a fatal signal --
/// the interpreter unwinding, not a missing attribute -- recorded for the walk
/// to stop on.
///
/// Out of line and cold. Dropping an error reads a thread-local, whether the
/// thread is attached to the interpreter, and inlined into
/// [`check_attr_record`]'s loop over the fields, that read's address is taken
/// once per record, ahead of the loop, on the path where every attribute is
/// there: a call into the dynamic linker per value.
#[cold]
#[inline(never)]
fn attribute_missing(
    err: PyErr,
    field: &Attribute,
    py: Python<'_>,
    frame: &mut Frame<'_, '_>,
) -> ControlFlow<()> {
    let ctx = frame.ctx;
    if is_fatal(&err, py) {
        record_fatal(err, ctx);
        return ControlFlow::Break(());
    }
    if ctx.mode.explains() {
        frame.out.push(located(
            frame.path,
            Arc::clone(&field.name),
            MISSING_ATTRIBUTE,
            format!("attribute {}", quoted(&field.name)),
            "missing".to_owned(),
        ));
    }
    ControlFlow::Continue(())
}
