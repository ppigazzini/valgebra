//! What one membership test keeps of a definition it has walked at a value, for
//! a second member of one node to find rather than walk again.
//!
//! A node that reads one value by several members -- a union's branches, a
//! meet's members, a map's clauses -- asks each in turn, and two of them can
//! reach the same definition at the same value: `int | list[T] | list[T | str]`
//! reads a list's element as `T` once per list branch. Each level of the value
//! then walks the level below once per such member, so the time is the number
//! of members to the power of the depth, and the walk's bounds cap the depth,
//! not the product. A value the schema admits pays it as well as one it
//! refuses, and so does the explaining walk, which explains each branch of a
//! refused union. Walked that way, a refused document of a few dozen bytes
//! takes seconds.
//!
//! The memo answers a definition at a value from the walk that answered it
//! before, so each pair is walked once. It is kept where it can be read as the
//! value's own answer, which is four conditions:
//!
//! - **A walk that stopped answers only a walk with no more room.** A walk
//!   that decided answers for the value, and is found by any later one. A walk
//!   that stopped -- at the depth bound, at a value inside itself, at a value
//!   that moved -- and refused has answered for the room it had: the walk
//!   levels open and the definitions unfolded where it began. Every answer a
//!   walk decides it decides with more room as well, so a walk with no more
//!   room refuses too, and the refusal is kept with its room and its stop,
//!   found only by such a walk, and said again with the stop. A value nested
//!   past the bounds stops every walk of it, and a memo keeping decided walks
//!   alone would keep nothing for it. A fatal signal unwinds the walk, and
//!   nothing is kept.
//! - **Only a definition that reaches no code the walk calls at each
//!   occurrence**: no predicate, and no class whose metaclass answers
//!   `isinstance` itself. Both are promised a call wherever the walk reaches
//!   them, and a memo would choose which of an impure answer's readings counts.
//! - **For the length of the call.** The memo lives in the call's
//!   [`WalkState`](super::WalkState) and is dropped with it.
//! - **Holding each Python value it keys.** A key is the value's address, and
//!   user code the walk runs -- an `__eq__` behind a literal, a key's
//!   `__hash__` -- can replace an element of a list at the same length, which
//!   no size check sees. Freed, the element's address could be a later
//!   value's, which the memo would answer for. A parsed document frees nothing
//!   while it is walked.
//!
//! Remembering costs a hash insert and a reference a walk, so it runs only
//! where it can be found again: in the walk of a member after the first, of a
//! node whose members reach a definition in common ([`MemoPlan::shares`]),
//! once an earlier member has entered a definition ([`Memo::entered`]). A
//! member that refuses by its kind -- `list[T]` against a dict -- enters none,
//! so a JSON schema's branches, which share their element definition and
//! never read one value twice, arm nothing. A remembered answer is looked up
//! by every later walk of the call, armed or not, which is what keeps a
//! branch's first walk from walking again what a later one kept.

use std::cell::Cell;

use pyo3::prelude::*;
use pyo3::types::PyType;
use rustc_hash::{FxHashMap, FxHashSet};
use valgebra_core::{Constraint, DefIx, PathSegment, Schema, Violation};

use crate::codes::Code;
use crate::input::Value;

/// The definitions a plan can name, one bit each in a `u128`. The validator
/// refuses a table longer than that (`MAX_DEFINITIONS`), so every table it
/// builds is planned; one past it would be planned as remembering nothing.
pub(crate) const PLANNED_DEFINITIONS: usize = u128::BITS as usize;

const _: () = assert!(crate::validator::MAX_DEFINITIONS <= PLANNED_DEFINITIONS);

/// Which definitions a walk may remember, and which nodes arm it: built once
/// per validator, beside the other plans in its index.
#[derive(Default)]
pub(crate) struct MemoPlan {
    /// One bit per definition whose walk may be kept: no predicate and no
    /// hooked class in its body or in any definition it reaches.
    remembered: u128,
    /// The member buffers, by address, of the unions and meets two of whose
    /// members reach a remembered definition, and the clause buffers of the
    /// maps two of whose clauses do.
    shared: FxHashSet<usize>,
}

impl MemoPlan {
    /// Whether a walk of this definition may be kept.
    #[inline]
    pub(crate) fn remembers(&self, def: DefIx) -> bool {
        def.get() < PLANNED_DEFINITIONS && (self.remembered >> def.get()) & 1 == 1
    }

    /// Whether two members of the node owning this buffer reach a remembered
    /// definition in common.
    pub(crate) fn shares(&self, members: usize) -> bool {
        !self.shared.is_empty() && self.shared.contains(&members)
    }
}

/// Plan the memo for a finished schema and its definitions.
pub(crate) fn plan(
    py: Python<'_>,
    schema: &Schema,
    defs: &[Schema],
    pool: &[Py<PyAny>],
) -> MemoPlan {
    if defs.len() > PLANNED_DEFINITIONS {
        return MemoPlan::default();
    }
    // What each body names, closed under naming: a definition reaches what
    // the definitions it names reach. A pass follows each reach through the
    // reaches it names, which doubles the chain of names it covers, so as
    // many passes as the table is long cover every chain; a pass that adds
    // nothing ends it sooner.
    let mut reach: Vec<u128> = defs.iter().map(names).collect();
    for _ in 0..defs.len() {
        let closed: Vec<u128> = reach.iter().map(|&named| through(&reach, named)).collect();
        if closed == reach {
            break;
        }
        reach = closed;
    }
    let calling = defs
        .iter()
        .enumerate()
        .filter(|(_, body)| calls_code(py, body, pool))
        .fold(0u128, |bits, (at, _)| bits | 1 << at);
    let remembered = reach
        .iter()
        .enumerate()
        .filter(|(at, reached)| (calling >> at) & 1 == 0 && *reached & calling == 0)
        .fold(0u128, |bits, (at, _)| bits | 1 << at);
    let mut shared = FxHashSet::default();
    if remembered != 0 {
        for node in std::iter::once(schema).chain(defs) {
            collect_shared(node, &reach, remembered, &mut shared);
        }
    }
    MemoPlan { remembered, shared }
}

/// `named` together with everything the definitions it names reach.
fn through(reach: &[u128], named: u128) -> u128 {
    reach
        .iter()
        .enumerate()
        .filter(|(at, _)| (named >> at) & 1 == 1)
        .fold(named, |bits, (_, reached)| bits | reached)
}

/// The definitions a schema names, one bit each, without following them.
fn names(schema: &Schema) -> u128 {
    let mut bits = 0;
    name_into(schema, &mut bits);
    bits
}

fn name_into(schema: &Schema, bits: &mut u128) {
    if let Schema::Ref(id) = schema {
        if id.get() < PLANNED_DEFINITIONS {
            *bits |= 1 << id.get();
        }
        return;
    }
    for_each_child(schema, &mut |child| name_into(child, bits));
}

/// Whether a schema, read without following its references -- a reference has
/// no children to read -- carries code the walk calls at each occurrence: a
/// predicate, or a class whose metaclass is not `type`. Such a metaclass may
/// answer `isinstance` itself; telling one that does from one that does not
/// would read its namespace, and an unreadable class is read as calling code.
fn calls_code(py: Python<'_>, schema: &Schema, pool: &[Py<PyAny>]) -> bool {
    match schema {
        Schema::Refine { base, constraints } => {
            constraints
                .iter()
                .any(|c| matches!(c, Constraint::Predicate(_)))
                || calls_code(py, base, pool)
        }
        Schema::Instance(index) => !pool
            .get(index.get())
            .is_some_and(|class| class.bind(py).get_type().is(py.get_type::<PyType>())),
        _ => {
            let mut calls = false;
            for_each_child(schema, &mut |child| {
                calls = calls || calls_code(py, child, pool);
            });
            calls
        }
    }
}

/// Record every union, meet and clause list below `schema` two of whose
/// members reach a remembered definition in common.
fn collect_shared(
    schema: &Schema,
    reach: &[u128],
    remembered: u128,
    shared: &mut FxHashSet<usize>,
) {
    let reached = |named: u128| through(reach, named) & remembered;
    match schema {
        Schema::Union(members) | Schema::Intersection(members) => {
            if overlap(members.iter().map(|m| reached(names(m)))) {
                shared.insert(members.as_ptr() as usize);
            }
        }
        Schema::KeyedMap { defaults, .. } => {
            let clauses = defaults
                .iter()
                .map(|clause| reached(names(&clause.key) | names(&clause.value)));
            if overlap(clauses) {
                shared.insert(defaults.as_ptr() as usize);
            }
        }
        _ => {}
    }
    for_each_child(schema, &mut |child| {
        collect_shared(child, reach, remembered, shared);
    });
}

/// Whether two of these definition sets have one in common.
fn overlap(mut reaches: impl Iterator<Item = u128>) -> bool {
    let mut seen = 0u128;
    reaches.any(|bits| {
        let met = seen & bits != 0;
        seen |= bits;
        met
    })
}

/// Call `visit` on each schema directly below `schema`, references not
/// followed.
fn for_each_child(schema: &Schema, visit: &mut impl FnMut(&Schema)) {
    match schema {
        Schema::Seq { shape, .. } => {
            shape.prefix.iter().for_each(&mut *visit);
            shape.tail.as_deref().into_iter().for_each(visit);
        }
        Schema::Coll { element, .. } => visit(element),
        Schema::KeyedMap { fields, defaults } => {
            for field in fields.iter() {
                visit(&field.schema);
            }
            for clause in defaults.iter() {
                visit(&clause.key);
                visit(&clause.value);
            }
        }
        Schema::AttrRecord { fields } => {
            for field in fields.iter() {
                visit(&field.schema);
            }
        }
        Schema::Union(members) | Schema::Intersection(members) => {
            members.iter().for_each(visit);
        }
        Schema::Complement(inner) => visit(inner),
        Schema::Refine { base, .. } => visit(base),
        Schema::Anything(_)
        | Schema::Nothing
        | Schema::NoneType
        | Schema::Bool
        | Schema::Int
        | Schema::Float
        | Schema::Str
        | Schema::Bytes
        | Schema::Literal(_)
        | Schema::Instance(_)
        | Schema::Ref(_)
        | Schema::SelfRef(_) => {}
    }
}

/// How much of the walk's two bounds a walk had left where it began: the walk
/// levels already open, and the definitions already unfolded on its path. A
/// walk begun deeper on either has less room.
#[derive(Clone, Copy)]
pub(crate) struct Room {
    pub(crate) levels: usize,
    pub(crate) unfolded: usize,
}

impl Room {
    /// Whether a walk begun here has no more room than one begun at `kept`.
    fn within(self, kept: Room) -> bool {
        self.levels >= kept.levels && self.unfolded >= kept.unfolded
    }
}

/// What a walk of a definition at a value answered: whether it admitted the
/// value, and, where it stopped, the first stop and the room it began with.
#[derive(Clone, Copy)]
pub(crate) struct Answer {
    pub(crate) admitted: bool,
    pub(crate) stopped: Option<(Code, Room)>,
}

impl Answer {
    /// Whether this answer is a later walk's, begun with `room`: a walk that
    /// decided, or admitted, answers any, and one that stopped and refused
    /// answers a walk with no more room than it had.
    fn answers(&self, room: Room) -> bool {
        match self.stopped {
            Some((_, kept)) if !self.admitted => room.within(kept),
            _ => true,
        }
    }

    /// The stop a walk answered from this says it met: the one the kept walk
    /// met, so a complement or a kept walk above reads it as that walk did.
    pub(crate) fn stop(&self) -> Option<Code> {
        self.stopped.map(|(code, _)| code)
    }
}

/// What the deciding walks kept for a pair: an answer that holds for any
/// later walk, or a refusal that stopped, with the rooms the walks that met it
/// began with, none of them covering another.
enum Decided {
    Answered(Answer),
    Stopped(Code, Vec<Room>),
}

/// A definition explained at a value: its answer, and what it reported, with
/// the path it was reported at.
struct Explained {
    answer: Answer,
    /// The path's length where the walk began. What a violation's path holds
    /// past it is the violation's own, and a later walk reaching the value at
    /// another path reports it there.
    from: usize,
    violations: Box<[Violation]>,
}

/// Where one membership test's memo stands: how many definitions the walk has
/// entered, and whether a node has armed it. One word, the count above two
/// flags, so the walk writes one word for it a call and reads one where it
/// enters a definition.
///
/// What the memo keeps is [`Kept`], held with the state's other rare
/// resource, its fatal signal: every call carries a `Memo` and nearly none
/// arms it, and tables held here would be words to write and a drop to run on
/// each call.
#[derive(Default)]
pub(crate) struct Memo {
    /// [`Self::USED`], [`Self::ARMED`], and the definitions entered times
    /// [`Self::ENTERED`]. The
    /// count only grows, and a node about to walk a value by another member
    /// reads whether it moved since the node began: if it did not, no earlier
    /// member entered a definition, and there is nothing a later one could
    /// find again.
    bits: Cell<usize>,
}

impl Memo {
    /// Set once any node has armed the memo in this call. Until one has, there
    /// is nothing to find, and a definition is walked without a lookup.
    const USED: usize = 1;
    /// Set while the walk is inside a member a node armed.
    const ARMED: usize = 2;
    /// One definition entered.
    const ENTERED: usize = 4;

    /// Count a definition entered, and answer whether a node has armed the
    /// memo, so the definition may be found in it.
    #[inline]
    pub(crate) fn enter(&self) -> bool {
        let bits = self.bits.get();
        self.bits.set(bits.wrapping_add(Self::ENTERED));
        bits & Self::USED != 0
    }

    /// Where the walk stands in the definitions it has entered: a reading that
    /// moves when one is entered, compared between two points of the walk and
    /// never counted from. The flags ride in it unmasked: they change only
    /// inside a member a node armed, and a node arms one only after an earlier
    /// member entered a definition, which moved the count already.
    #[inline]
    pub(crate) fn entered(&self) -> usize {
        self.bits.get()
    }

    /// How many definitions the walk has entered, for the tests that hold the
    /// walk to a count.
    #[cfg(all(test, feature = "interpreter-tests"))]
    pub(crate) fn count(&self) -> usize {
        self.bits.get() / Self::ENTERED
    }

    /// Whether a node has armed the memo in this call, for the tests that
    /// hold a walk to arming nothing.
    #[cfg(all(test, feature = "interpreter-tests"))]
    pub(crate) fn in_use(&self) -> bool {
        self.bits.get() & Self::USED != 0
    }

    /// Whether a walk is kept.
    pub(crate) fn is_armed(&self) -> bool {
        self.bits.get() & Self::ARMED != 0
    }

    /// Run `walk` with the memo armed.
    pub(crate) fn armed<R>(&self, walk: impl FnOnce() -> R) -> R {
        let outer = self.bits.get() & Self::ARMED;
        self.bits.set(self.bits.get() | Self::USED | Self::ARMED);
        let answer = walk();
        self.bits.set(self.bits.get() & !Self::ARMED | outer);
        answer
    }
}

/// What one membership test keeps, keyed by the value's address and the
/// definition's index, built by the first walk kept.
#[derive(Default)]
pub(crate) struct Kept {
    decided: FxHashMap<(usize, usize), Decided>,
    /// The stops settling walks met, with their rooms. A settling walk reads
    /// past a stop, so its stop is the third value of the walk it ended, and a
    /// settling walk with no more room ends the same way.
    settled: FxHashMap<(usize, usize), Decided>,
    explained: FxHashMap<(usize, usize), Explained>,
    /// One reference per Python value a key names.
    held: Vec<Py<PyAny>>,
}

impl Kept {
    /// The answer a deciding walk kept for this pair, where it answers a walk
    /// begun with `room`.
    pub(crate) fn decided(&self, key: (usize, usize), room: Room) -> Option<Answer> {
        Self::read(&self.decided, key, room)
    }

    /// The stop a settling walk kept for this pair, where it answers a settling
    /// walk begun with `room`.
    pub(crate) fn settled(&self, key: (usize, usize), room: Room) -> Option<Answer> {
        Self::read(&self.settled, key, room)
    }

    fn read(
        table: &FxHashMap<(usize, usize), Decided>,
        key: (usize, usize),
        room: Room,
    ) -> Option<Answer> {
        match table.get(&key)? {
            Decided::Answered(answer) => Some(*answer),
            Decided::Stopped(code, rooms) => {
                rooms
                    .iter()
                    .any(|kept| room.within(*kept))
                    .then_some(Answer {
                        admitted: false,
                        stopped: Some((*code, room)),
                    })
            }
        }
    }

    /// The answer an explaining walk kept for this pair, where it answers a
    /// walk begun with `room`, its violations reported again at `path`.
    pub(crate) fn explained(
        &self,
        key: (usize, usize),
        room: Room,
        path: &[PathSegment],
        out: &mut Vec<Violation>,
    ) -> Option<Answer> {
        self.reported(key, |answer| answer.answers(room), path, out)
    }

    /// Whether an explaining walk kept a refusal for this pair where it
    /// stopped, which a walk with more room than it had does not find.
    pub(crate) fn explained_a_stop(&self, key: (usize, usize)) -> bool {
        self.explained
            .get(&key)
            .is_some_and(|found| found.answer.stopped.is_some() && !found.answer.admitted)
    }

    /// The refusal an explaining walk kept for this pair where it stopped,
    /// its violations reported again at `path`, whatever room it had: for a
    /// walk whose deciding walk refuses where it stops too.
    pub(crate) fn explained_refusal(
        &self,
        key: (usize, usize),
        path: &[PathSegment],
        out: &mut Vec<Violation>,
    ) -> Option<Answer> {
        self.reported(key, |answer| !answer.admitted, path, out)
    }

    fn reported(
        &self,
        key: (usize, usize),
        found_by: impl FnOnce(&Answer) -> bool,
        path: &[PathSegment],
        out: &mut Vec<Violation>,
    ) -> Option<Answer> {
        let found = self
            .explained
            .get(&key)
            .filter(|found| found_by(&found.answer))?;
        out.extend(found.violations.iter().map(|violation| {
            let own = violation.path.get(found.from..).unwrap_or_default();
            Violation {
                code: violation.code,
                path: path.iter().chain(own).cloned().collect(),
                expected: violation.expected.clone(),
                value_summary: violation.value_summary.clone(),
            }
        }));
        Some(found.answer)
    }

    /// Keep what a deciding walk answered for this pair.
    ///
    /// A refusal that stopped joins the rooms the pair's earlier stopped
    /// refusals were kept with, and drops any it answers for: a room counts
    /// two things -- walk levels and definitions unfolded -- so two paths can
    /// each have more of one than the other, and a table holding the latest
    /// alone walks the pair again each time a walk alternates between them.
    /// The rooms kept are those no other covers, so a later walk is answered
    /// by any it has no more room than.
    pub(crate) fn keep_decided(
        &mut self,
        key: (usize, usize),
        value: &Value<'_, '_>,
        answer: Answer,
        settling: bool,
    ) {
        self.hold(value);
        let Some((code, room)) = answer.stopped.filter(|_| !answer.admitted) else {
            self.decided.insert(key, Decided::Answered(answer));
            return;
        };
        if settling {
            Self::keep_stop(&mut self.settled, key, code, room);
        }
        Self::keep_stop(&mut self.decided, key, code, room);
    }

    fn keep_stop(
        table: &mut FxHashMap<(usize, usize), Decided>,
        key: (usize, usize),
        code: Code,
        room: Room,
    ) {
        match table.get_mut(&key) {
            Some(Decided::Stopped(_, rooms)) => {
                rooms.retain(|kept| !kept.within(room));
                rooms.push(room);
            }
            _ => {
                table.insert(key, Decided::Stopped(code, vec![room]));
            }
        }
    }

    /// Keep what an explaining walk answered and reported for this pair; the
    /// walk began where the path was `from` long.
    pub(crate) fn keep_explained(
        &mut self,
        key: (usize, usize),
        value: &Value<'_, '_>,
        answer: Answer,
        from: usize,
        violations: &[Violation],
    ) {
        self.hold(value);
        let explained = Explained {
            answer,
            from,
            violations: violations.into(),
        };
        self.explained.insert(key, explained);
    }

    /// Hold a reference to the Python value a key names.
    fn hold(&mut self, value: &Value<'_, '_>) {
        if let Value::Py(obj) = value {
            self.held.push((*obj).clone().unbind());
        }
    }
}
