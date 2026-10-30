//! Lowering a schema to a descriptor.
//!
//! A schema is a *term*: a tree the walk reads and the printer renders. A
//! descriptor is a *set*. Lowering is the map between them, and the two live
//! side by side rather than one replacing the other -- the walk still decides
//! membership, and the descriptor is what the algebra reasons in.
//!
//! **The map is partial, and the type says so.** A form the descriptor cannot
//! yet hold lowers to `None` rather than to something close: a descriptor that
//! stood for a schema it does not denote would be complemented into one that is
//! wrong the other way, and a refusal is the only sound answer to give.
//!
//! The core holds no Python objects, so the constants a schema names are indices
//! into a pool the bindings keep. [`Constants`] is the way in: the caller reads
//! the pool and answers what a comparison operand or a literal carries.

use super::budget;
use super::classes::Class;
use super::floats::FloatSet;
use super::maps::{KEY_KINDS, Label};
use super::{BoolSet, Descr, EVERY_PART, Parts, integers::IntSet, part};
use crate::ir::{
    ClassIx, CollKind, ConstIx, Constraint, Field, MapClause, OperandIx, Polarity, Schema, SeqKind,
    SeqShape,
};
use crate::kind::Kind;
use std::cell::Cell;

/// A pooled value, as far as a descriptor can read one.
#[derive(Debug, Clone, PartialEq)]
pub enum Operand {
    /// One of the two booleans.
    Boolean(bool),
    /// An integer, which is never a boolean: `bool` is its own kind here.
    Integer(i64),
    /// A float, `nan` included.
    Float(f64),
    /// A word, under the kind that reads it -- a `str` as its UTF-8 bytes.
    Word(Vec<u8>, Kind),
    /// `None`.
    NoneType,
    /// An instance of a pure class.
    Instance(Class),
}

/// What a lowering needs from the pool the core does not hold.
///
/// The core stays free of Python objects, so a schema names its constants by
/// index. Answering `None` is the honest reading of an operand this cannot see
/// -- an impure class, a callable, an object with a custom `__eq__` -- and it
/// refuses the lowering rather than guessing.
pub trait Constants {
    /// What a comparison or `MultipleOf` operand carries.
    ///
    /// Every question here defaults to `None`, the answer of a pool that cannot
    /// see this one. That is the conservative direction -- it refuses the
    /// lowering rather than describing a set the schema does not denote -- and it
    /// is what lets an oracle that holds no objects be written as an empty impl
    /// instead of three declines.
    fn operand(&self, _index: OperandIx) -> Option<Operand> {
        None
    }

    /// What a `Literal` names.
    fn constant(&self, _index: ConstIx) -> Option<Operand> {
        None
    }

    /// The class an `Instance` atom names, as the order snapshot the core reads.
    ///
    /// Only the bindings can walk an `__mro__`, and only they can tell a **pure**
    /// class from one whose metaclass answers `isinstance` by running code. A
    /// hooked class is not a set this algebra holds, so answering `None` for one
    /// refuses the lowering rather than describing a set the class does not have.
    fn class(&self, _index: ClassIx) -> Option<Class> {
        None
    }
}

/// The nodes this will lower before refusing.
///
/// A schema is a tree the caller writes, and lowering both recurses over it and
/// *builds* at every node -- a sequence node determinizes an automaton, a set
/// node takes a powerset. Without a bound a deep enough schema exhausts the
/// stack and a wide one exhausts the clock, which is why the procedure beside
/// this one carries a work budget too.
///
/// A refusal past the bound is the same refusal as for a form with nowhere to
/// land, and it reaches the same place: the caller decides by the rules. It is a
/// bound on the *lowering*, not on the algebra -- the components have their own,
/// and those are limits of what they can represent rather than of what they will
/// spend.
///
/// Sized by what can be *built* rather than by what can be parsed. Complementing
/// a descriptor complements the guards inside it, so a chain of complements
/// deepens the descriptor as well as the schema, and the operations recurse
/// through that nesting: past roughly a hundred the stack goes rather than the
/// clock.
///
/// **This is the bound an ordinary annotation reaches first**, and it reaches
/// it by width. A record whose fields each take two types, against the union of
/// the records that fix every field, reads 22 nodes at two fields, 45 at three
/// and 96 at four -- so three fields decide and four are refused before they
/// are answered. `tests/test_completeness_ledger.py` carries all three as rows,
/// which is what makes these figures re-derivable: the row fails on the commit
/// that moves the number either way.
///
/// **This bound is debt.** It counts work rather than limiting what the
/// representation can hold, and it is here for the reason the decision's own
/// budget is: a lowering repeats itself over structurally equal subtrees.
/// Construction shares those subtrees where they are built alike
/// (`ir/intern.rs`), which is what gives the memo that would replace this
/// ceiling a cheap key; the memo is the half not written.
pub(crate) const BUDGET: u32 = 64;

/// The set a schema denotes, or `None` where the descriptor cannot yet hold it.
///
/// # Errors
///
/// Refuses rather than approximating. It refuses a recursive reference, which a
/// finite descriptor has no room for; a predicate, which runs code the core
/// cannot read; a key or a bound with no component to land in, such as a length
/// over a set or a dict; an operand or a class the pool could not read; and a
/// schema past the bounds one lowering spends.
pub fn lower(schema: &Schema, pool: &dyn Constants) -> Option<Descr> {
    lower_within(Bounds::DEFAULT, schema, pool)
}

/// How many times a reference is unfolded before the descriptor is built.
///
/// One. A single unfolding puts the fixpoint's own body in front of the
/// representation, which settles every relation that turns on *what kinds* a
/// recursive schema admits -- a meet with a disjoint kind, an inclusion in a
/// wider union -- and that is the whole of what the structural rules cannot
/// read. Each further unfolding multiplies the schema this must build against
/// [`BUDGET`]'s 64 nodes, for relations nobody has asked for.
///
/// A bound like the three in [`Bounds`], and unlike them in one way worth
/// stating: those three say what a build may spend, and this says what it is
/// given to build *from*. A `Ref` has no component to land in, so without the
/// unfolding a recursive schema does not lower at all.
pub(crate) const UNFOLDS: u32 = 1;

/// [`lower`], resolving a recursive schema's references through `definitions`.
///
/// `polarity` is the side the schema is read on, and it is the caller's to
/// know: unfolding grows the set on one side and shrinks it on the other, so a
/// difference stays sound only when the two sides are unfolded in opposite
/// directions -- and only in the direction that proves it empty.
///
/// # Errors
///
/// Refuses what [`lower`] refuses. A reference `definitions` does not resolve
/// is one of those: the unfolding leaves it in place and no component holds it.
pub(crate) fn lower_unfolded(
    schema: &Schema,
    definitions: &[Schema],
    polarity: Polarity,
    pool: &dyn Constants,
) -> Option<Descr> {
    // The empty check first: it is one comparison, and it is true for every
    // schema that carries no fixpoint at all -- which is almost all of them, and
    // all of the ones on the workload the decision budget is measured over. The
    // walk that follows costs a pass over the tree, and paying it per relation
    // for a schema with no definitions would be ten percent of that workload.
    if definitions.is_empty() || !schema.has_reference() {
        return lower(schema, pool);
    }
    lower(&schema.unfolded(definitions, UNFOLDS, polarity), pool)
}

/// The parts of the universe whose values the unfolding of `schema` cuts a
/// reference beneath: for each cut, the kind of the constructor nearest the
/// root it sits under, or every part for a cut beneath none.
///
/// A constructor builds values of its one kind, and a union, a meet and a
/// complement are read a kind at a time [FCB08, (4)], so a cut beneath a
/// `list[...]` widens or narrows the lists and leaves every other kind's values
/// as the schema denotes them: a finite approximation of a recursive type is
/// exact up to the depth it cuts at [AC93, §3.4], and the depth starts at the
/// constructor. A refinement, a meet's member and an attribute record are no
/// constructor of a kind, the last holding values of every kind.
///
/// The unfolding is [`lower_unfolded`]'s and [`read_widened`]'s, [`UNFOLDS`]
/// deep, and a reference to a definition `definitions` does not hold is cut
/// where it stands, as theirs is. The walk reads at most [`CUT_WALK`] nodes,
/// past which it answers every part, which refutes nothing.
pub(crate) fn parts_a_cut_reaches(schema: &Schema, definitions: &[Schema]) -> Parts {
    let mut reached = 0;
    let mut left = CUT_WALK;
    let mut pending = vec![(schema, None, UNFOLDS)];
    while let Some((node, under, unfolds)) = pending.pop() {
        let Some(rest) = left.checked_sub(1) else {
            return EVERY_PART;
        };
        left = rest;
        let cut = || under.map_or(EVERY_PART, part);
        match node {
            Schema::Ref(index) => match definitions.get(index.get()) {
                Some(body) if unfolds > 0 => pending.push((body, under, unfolds - 1)),
                _ => reached |= cut(),
            },
            Schema::SelfRef(_) => reached |= cut(),
            _ => {
                let below = under.or_else(|| constructed_kind(node));
                let mut children = Vec::new();
                node.push_children(&mut children);
                pending.extend(children.into_iter().map(|child| (child, below, unfolds)));
            }
        }
    }
    reached
}

/// The nodes [`parts_a_cut_reaches`] reads before it answers every part.
///
/// **This bound is cost.** The walk unfolds each reference once and so ends
/// without it; what it caps is the walk of a schema far past any a lowering
/// reads -- one the set representation lowers whole is read in [`BUDGET`]
/// nodes of the lowering's own count, which reads fewer children than this
/// walk does, and is far inside the figure.
const CUT_WALK: u32 = 1 << 16;

/// The kind of the values a constructor builds, or `None` for a node that
/// builds none of its own.
fn constructed_kind(schema: &Schema) -> Option<Kind> {
    Some(match schema {
        Schema::Seq {
            container: SeqKind::List,
            ..
        } => Kind::List,
        Schema::Seq {
            container: SeqKind::Tuple,
            ..
        } => Kind::Tuple,
        Schema::Coll {
            container: CollKind::Set,
            ..
        } => Kind::Set,
        Schema::Coll {
            container: CollKind::FrozenSet,
            ..
        } => Kind::FrozenSet,
        Schema::KeyedMap { .. } => Kind::Dict,
        _ => return None,
    })
}

/// [`lower_unfolded`] on the widened side, decided by the caller one reading
/// at a time: the schema built whole, and the schema without the conjuncts at
/// its top a lowering cannot read -- a predicate among a refinement's
/// constraints, at the top or as a member, and a meet's member holding a
/// predicate below its top or nested deeper than the meet leaves room for.
///
/// A meet is below each of its members, and a refinement below the one its
/// constraints are a subset of, so the thinned schema denotes a superset,
/// which is what a widened reading may be: proving it empty proves the schema
/// empty [FCB08, Def. 4.1]. That makes emptiness monotone in a meet's members:
/// `A & B` proved empty stays proved beside a predicate, or beside a member an
/// unfolding has nested past [`DEPTH`]. Nothing below a union, a complement or
/// a container is thinned, since a member there is no conjunct of the whole.
///
/// `decide` is asked of each reading in turn, with the conjuncts it kept --
/// `None` for the schema read whole -- and the first answer it gives is the
/// answer. A reading it answers nothing of is no answer, so the next reading
/// is asked, and since each reading is sound on its own the order is a matter
/// of cost alone. **Each is decided under an allowance of its own**, [`WORK`]
/// as the build was: reading a descriptor is a build ([`budget`]), and a
/// negated union's emptiness expands it by a product -- a complemented meet of
/// three records, read unbudgeted, multiplies out for minutes where the
/// allowance refuses at once.
///
/// A schema nesting a container past [`DEPTH`] is read thinned and never
/// built whole -- [`reach`] reads that off the schema -- whatever else in it
/// the build would have refused first. One holding a scalar past the depth is
/// read thinned first: the readings that leave the scalar's member out build
/// less, a meet beside a pattern's automata being the product they skip, and
/// where they answer nothing -- a scalar tower whose deep member is what decides
/// it -- the schema is built whole. Any other schema is built whole first, so
/// one the lowering takes pays a walk and nothing more, and one refused for a
/// predicate is read thinned. A refusal of any other kind -- a class the pool
/// cannot read, a key in part of a kind, a bound on what a build spends --
/// refuses the reading as it refuses the schema, unless it sits in a conjunct
/// the reading leaves out. **A thinned reading widens like a cut**, so an
/// inhabited answer refutes nothing.
///
/// **A reading that can only be inhabited is not built for a caller that
/// cannot read one.** A thinned reading of conjuncts whose meet the rules find a
/// value of is an inhabited set, which proves nothing empty. It answers one
/// question, a relation's subject below a supertype that lowers, so `answers`
/// is the caller's word, given the conjuncts kept, on whether their reading
/// can answer it -- the rules being the caller's to ask -- and the reading is
/// refused where it says no. The whole build a scalar past the depth defers is
/// asked the same of the whole schema: an emptiness the rules already find
/// inhabited is asked of every meet the rules reach, and would build the same
/// schema each time to read a value the rules have.
#[inline]
pub(crate) fn read_widened<T>(
    schema: &Schema,
    definitions: &[Schema],
    pool: &dyn Constants,
    answers: impl FnMut(&[Schema]) -> bool,
    mut decide: impl FnMut(&Descr, Option<&[Schema]>) -> Option<T>,
) -> Option<T> {
    let unfolded;
    let schema = if definitions.is_empty() || !schema.has_reference() {
        schema
    } else {
        unfolded = schema.unfolded(definitions, UNFOLDS, Polarity::Widen);
        &unfolded
    };
    let past = reach(schema, DEPTH, &Cell::new(BUDGET));
    let mut answers = answers;
    match past {
        Reach::Container => return thinned_readings(schema, pool, answers, decide),
        Reach::Scalar => {
            if let Some(answer) = thinned_readings(schema, pool, &mut answers, &mut decide) {
                return Some(answer);
            }
            if !answers(core::slice::from_ref(schema)) {
                return None;
            }
        }
        Reach::Within => {}
    }
    // Decided where it is built: a set returned out of here to be decided
    // would be copied, and the copy is most of what this costs a schema that
    // lowers.
    let room = Room::new(BUDGET);
    if let Some(set) = budget::under(WORK, || descend(schema, pool, &room, DEPTH)) {
        return budget::under(WORK, || decide(&set, None));
    }
    if past == Reach::Scalar || !room.unread.get() {
        return None;
    }
    thinned_readings(schema, pool, answers, decide)
}

/// The first reading [`read_widened`] builds, and the conjuncts it kept: what
/// a test reads of one reading.
#[cfg(test)]
pub(crate) fn lower_widened(
    schema: &Schema,
    definitions: &[Schema],
    pool: &dyn Constants,
    kept: &mut Option<Vec<Schema>>,
    answers: impl FnMut(&[Schema]) -> bool,
) -> Option<Descr> {
    read_widened(schema, definitions, pool, answers, |set, read| {
        *kept = read.map(<[Schema]>::to_vec);
        Some(set.clone())
    })
}

/// The thinned half of [`read_widened`], out of line so the schemas that lower
/// whole carry none of it.
///
/// Three times at most, keeping less each time: the members a lowering reads
/// whole -- no predicate, no container past the depth -- then the members
/// nested within the depth counting every node, a scalar included, which is
/// the reading a schema holding a scalar past the depth is decided by where
/// the build that holds it is not, and last the same counting a clause's key
/// as a level, which the build does not spend. A member at the edge of the
/// depth by its key alone is kept by the first two and is the one the meet's
/// emptiness is not proved beside, where the meet without it is proved empty;
/// a reading of fewer members holds more values, so the proof stands. A
/// thinning that keeps what the one before kept is skipped. One look at each
/// member first, allocating nothing: most refusals leave here as they came.
#[inline(never)]
fn thinned_readings<T>(
    schema: &Schema,
    pool: &dyn Constants,
    mut answers: impl FnMut(&[Schema]) -> bool,
    mut decide: impl FnMut(&Descr, Option<&[Schema]>) -> Option<T>,
) -> Option<T> {
    let mut kept_before = None;
    for (every_node, keys) in [(false, false), (true, false), (true, true)] {
        // Each thinning keeps a part of what the one before kept, so as many
        // members are the same members.
        let Some(read) = thinned_members(schema, every_node, keys)
            .filter(|read| Some(read.len()) != kept_before)
        else {
            continue;
        };
        kept_before = Some(read.len());
        // Fewer members hold more values, so a reading the caller refuses for
        // the first thinning is refused for the second.
        if !answers(&read) {
            return None;
        }
        let Some(set) = lower(&Schema::Intersection(read.clone().into()), pool) else {
            continue;
        };
        if let Some(answer) = budget::under(WORK, || decide(&set, Some(&read))) {
            return Some(answer);
        }
    }
    None
}

/// The members of `schema` a thinned reading keeps, or `None` where it would
/// keep them all: each one [`readable`] as it is or with its predicates left
/// out. A meet keeping none reads as the universe, which a relation against a
/// supertype the reading proves universal still decides.
fn thinned_members(schema: &Schema, every_node: bool, keys: bool) -> Option<Vec<Schema>> {
    let read = |member: &Schema, room: u32| readable(member, room, every_node, keys);
    match schema {
        Schema::Intersection(members) => {
            let room = DEPTH.checked_sub(1)?;
            if members.iter().all(|member| read(member, room)) {
                return None;
            }
            Some(
                members
                    .iter()
                    .filter_map(|member| {
                        if read(member, room) {
                            return Some(member.clone());
                        }
                        without_predicates(member).filter(|thinned| read(thinned, room))
                    })
                    .collect(),
            )
        }
        schema => {
            let thinned = without_predicates(schema)?;
            read(&thinned, DEPTH).then(|| vec![thinned])
        }
    }
}

/// A refinement with its predicates left out, or `None` where `schema` is no
/// refinement carrying one.
fn without_predicates(schema: &Schema) -> Option<Schema> {
    let Schema::Refine { base, constraints } = schema else {
        return None;
    };
    if !constraints
        .iter()
        .any(|constraint| matches!(constraint, Constraint::Predicate(_)))
    {
        return None;
    }
    Some(Schema::Refine {
        base: base.clone(),
        constraints: constraints
            .iter()
            .filter(|constraint| !matches!(constraint, Constraint::Predicate(_)))
            .cloned()
            .collect::<Vec<_>>()
            .into(),
    })
}

/// Whether `schema` is one a lowering can read at all: its containers nested
/// within the `room` it descends -- a level for each node and the rest for what
/// it holds -- or, where `every_node`, every node of it, a clause's key
/// counted too where `keys`, and holding no predicate.
fn readable(schema: &Schema, room: u32, every_node: bool, keys: bool) -> bool {
    let room = match room.checked_sub(1) {
        Some(room) => room,
        None if every_node || starts_a_build(schema) => return false,
        None => 0,
    };
    if let Schema::Refine { constraints, .. } = schema
        && constraints
            .iter()
            .any(|constraint| matches!(constraint, Constraint::Predicate(_)))
    {
        return false;
    }
    every_child_read(schema, keys, |child| {
        readable(child, room, every_node, keys)
    })
}

/// Whether a build of `schema` under `bounds` reaches a container past their
/// depth before it has read their nodes: a refusal read off the schema before
/// anything is built.
///
/// The walk is the build's without the building. It spends a node and a level
/// where [`descend`] spends one, and reads the schemas it reads in the order it
/// reads them, so a schema found past the depth is one the build refuses, at
/// that node or before it, and one cleared is one the build never refuses for
/// depth. What the walk saves is everything ahead of the deep node: the build
/// reaches it only after constructing the sets before it -- an automaton at a
/// sequence, a powerset at a set, a product at a meet -- and a walk of the
/// same nodes allocates nothing.
fn nests_past_the_depth(schema: &Schema, bounds: Bounds) -> bool {
    reach(schema, bounds.depth, &Cell::new(bounds.nodes)) == Reach::Container
}

/// The farthest a build reads past the depth: nowhere, a scalar, or a
/// container, which refuses it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Reach {
    Within,
    Scalar,
    Container,
}

/// [`nests_past_the_depth`] one node down, and what it found on the way: the
/// farthest past `depth` the nodes it read reach.
///
/// One walk answers both questions [`read_widened`] asks: whether the schema
/// is refused for depth, and whether a reading that leaves a scalar past the
/// depth out could decide what the whole one does not. It stops at a
/// container past the depth, and where the node bound runs out it reports
/// what it read before: a node past the bound is read as within the depth
/// and nothing below it is walked, so a member past the node bound is refused
/// for its nodes, as the build refuses it.
fn reach(schema: &Schema, depth: u32, nodes: &Cell<u32>) -> Reach {
    let Some(left) = nodes.get().checked_sub(1) else {
        return Reach::Within;
    };
    nodes.set(left);
    let (depth, mut far) = match depth.checked_sub(1) {
        Some(depth) => (depth, Reach::Within),
        None if starts_a_build(schema) => return Reach::Container,
        None => (0, Reach::Scalar),
    };
    every_child_read(schema, false, |child| {
        far = far.max(reach(child, depth, nodes));
        far != Reach::Container
    });
    far
}

/// Whether `read` holds of every schema [`descend`] reads below `schema`,
/// asked in the order it reads them, up to the first it does not hold of.
///
/// Every child but a clause's key: [`map`] reads a key through [`key_cover`],
/// which names the labels and parts the key covers without descending into
/// it. A walk over [`Schema::children`] counts the key's nesting, and would
/// refuse a map whose key is its deepest node, which the build reads. Where
/// `keys`, the key is read as well, for the last thinned reading
/// ([`thinned_readings`]).
fn every_child_read(schema: &Schema, keys: bool, mut read: impl FnMut(&Schema) -> bool) -> bool {
    match schema {
        Schema::Union(members) | Schema::Intersection(members) => members.iter().all(&mut read),
        Schema::Complement(inner)
        | Schema::Coll { element: inner, .. }
        | Schema::Refine { base: inner, .. } => read(inner),
        Schema::Seq { shape, .. } => shape.elements().all(&mut read),
        Schema::KeyedMap { fields, defaults } => {
            fields.iter().all(|field| read(&field.schema))
                && defaults
                    .iter()
                    .all(|clause| (!keys || read(&clause.key)) && read(&clause.value))
        }
        Schema::AttrRecord { fields } => fields.iter().all(|field| read(&field.schema)),
        // The leaves, listed as `Schema::push_children` lists them, so a
        // variant that holds a schema is a compile error here too.
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
        | Schema::SelfRef(_) => true,
    }
}

/// Whether lowering `schema` starts a build of its own, the cost [`DEPTH`]
/// bounds: a container, whose set is an automaton, a powerset or a record
/// lattice built over its parts. A meet, a union, a complement and a
/// refinement are operations on the sets below them, whose products
/// [`WORK`] charges, and a scalar is an interval or a word set.
fn starts_a_build(schema: &Schema) -> bool {
    matches!(
        schema,
        Schema::Seq { .. }
            | Schema::Coll { .. }
            | Schema::KeyedMap { .. }
            | Schema::AttrRecord { .. }
    )
}

/// What a lowering may spend, in the three quantities it can run out of.
///
/// They are three because a build can be too big in three ways, and no one of
/// them implies another: a schema of a dozen nodes can nest ten deep, a shallow
/// one can be a hundred wide, and either can build a whole product and then
/// refuse anyway. Named together so a caller measuring one can
/// lift the other two, which is what the benchmark behind these numbers does --
/// a bound whose figure lives only in a comment cannot be re-derived on another
/// machine, and cannot fail when the shape it guards against changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bounds {
    /// The schema nodes a lowering will read.
    pub(crate) nodes: u32,
    /// The schema nesting a lowering will descend.
    pub(crate) depth: u32,
    /// The units of multiplying work a build may spend.
    pub(crate) work: u64,
}

impl Bounds {
    /// The bounds every relation lowers under. See the three constants for what
    /// each number is measured from.
    pub(crate) const DEFAULT: Bounds = Bounds {
        nodes: BUDGET,
        depth: DEPTH,
        work: WORK,
    };

    /// Bounds no build reaches, for measuring what one costs unheld.
    ///
    /// Not a mode the library runs in: an unheld build is the thing the three
    /// numbers above exist to prevent, and this exists so a benchmark can show
    /// what they prevent.
    pub const UNHELD: Bounds = Bounds {
        nodes: u32::MAX,
        depth: u32::MAX,
        work: u64::MAX,
    };
}

/// The work a build may spend before it refuses.
///
/// `BUDGET` bounds the schema this reads; this bounds what reading it costs,
/// and the two are different quantities. A schema of a dozen nodes can build a
/// whole product and then refuse anyway, because the bounds on a descriptor say
/// how large a result may be and say nothing about the work of reaching one.
///
/// Sized from the shapes on both sides of the question. The differences the
/// descriptor decides and the structural rules do not -- a container meet, a
/// meet of two tuples, one regular language inside another -- each
/// spend under
/// 256 units. The shapes that blow up spend tens or hundreds of thousands: a
/// record nested eight deep spends 22,806, and a union of four records nested
/// three deep minus a union of its siblings spends 170,597.
///
/// On the shapes measured above, `DEPTH` refuses first for anything deep, so
/// this number stops a build only where a shape is shallow and wide -- which is
/// the one nesting cannot catch, and what it is kept for.
///
/// **The figure is what a record split across a union of records costs**, which
/// is the widest shallow shape the sets are asked to decide. A two-field record
/// against the four records that fix both keys spends 1,173 units; the
/// three-field record against its eight corners spends 2,241. Both are
/// relations a caller writes and the rules decline, so both are the sets' to
/// answer, and a bound below them refuses an ordinary question. The blow-ups
/// the number exists to stop are an order of magnitude further out, so the
/// margin is real rather than nominal.
///
/// The field after that is outside on both counts: four fields against sixteen
/// corners spend 9,965 units and read 96 nodes against a `BUDGET` of 64, so
/// raising either bound alone leaves the refusal where it is. That is the width
/// of the decided fragment, and `docs/15-decidability.md` states it as one.
///
/// `tests/test_completeness_ledger.py` carries both shapes as rows, which is
/// what makes this figure re-derivable: the row fails on the commit that
/// lowers it.
///
/// **This bound is cost.** A build multiplies finitely many atoms and ends
/// without it; what it caps is how long a caller waits on a product that is
/// finite and exponential, which a memo over equal subtrees would not shrink.
pub const WORK: u64 = 4096;

/// The schema nesting this will descend before refusing a container.
///
/// [`WORK`] bounds what a build spends once it has started, and it cannot bound
/// what starting costs: a lowering builds an automaton at every sequence node
/// and a powerset at every set node, and none of that is a product to charge
/// for. On the shapes a relation is asked about, that floor is around a hundred
/// times what the structural rules spend answering the whole question -- so a
/// build that will not pay for itself has to be refused *before* it is walked,
/// and depth is what says which. [`nests_past_the_depth`] reads it off the
/// schema, so a refusal for depth costs a walk of the nodes ahead of the deep
/// one rather than the sets built from them.
///
/// **A container is what it refuses.** Every node above one counts, a meet, a
/// union, a complement or a refinement as much as another container, so a
/// container more than this deep is refused as it always was; a node past the
/// depth that starts no build ([`starts_a_build`]) is read, since an interval,
/// a word set or an operation on them is what [`WORK`] charges. A scalar
/// tower -- `int & ~(int & ~(Literal[0, 1]))`, six deep at its constants --
/// lowers, and where the reading holding the deep scalar decides nothing, the
/// one leaving it out is asked ([`read_widened`]).
///
/// Nesting is the exponential, and `benches/core.rs` measures it:
/// `lower_nested_records_depth{0,2,4,6}`, run under [`Bounds::UNHELD`]. Breadth
/// is not the exponential and is bounded by [`BUDGET`] instead.
///
/// Set from both sides. Every relation the descriptor decides and the rules do
/// not nests its containers five deep or less -- `lower_container_meet`,
/// `lower_tuple_meet`, `lower_regular_language` and `lower_step_divides` are
/// those four. The shapes that blow up nest ten and deeper.
/// `lower_sibling_union_difference_{unheld,held}` is the pair that shows what
/// the bound buys, on one shape, from this number alone.
///
/// **This bound is cost.** A lowering is a recursion over a finite tree and ends
/// without it; nesting multiplies what it builds, over distinct levels, which a
/// memo over equal subtrees would not shrink, so the bound caps the exponential
/// rather than standing in for a termination argument.
pub(crate) const DEPTH: u32 = 5;

/// [`lower`] under explicit bounds, for a caller measuring one of them.
///
/// # Errors
///
/// Refuses what [`lower`] refuses, and also a build that would exceed any of
/// `bounds`. That refusal is not a statement about the schema: the same schema
/// lowers under larger bounds. It is what makes the cost of asking bounded,
/// which is the only thing that makes asking safe.
pub fn lower_within(bounds: Bounds, schema: &Schema, pool: &dyn Constants) -> Option<Descr> {
    if nests_past_the_depth(schema, bounds) {
        return None;
    }
    budget::under(bounds.work, || {
        descend(schema, pool, &Room::new(bounds.nodes), bounds.depth)
    })
}

/// What one lowering has left of the nodes it may read, and whether it refused
/// at a predicate, the constraint no lowering reads.
///
/// The second is what [`read_widened`] asks before it thins a refused schema,
/// so a refusal of any other kind costs it nothing. The depth bound, the other
/// refusal a thinned reading answers, is read off the schema before the build
/// ([`reach`]), so a build that starts never reaches it. The
/// flag is set on that one refusal alone, which leaves the paths that lower
/// untouched but for the store that clears it: the matrix decision workload,
/// whose subjects are refused for a class and an operand its pool cannot read,
/// measures the whole reading at +0.03%, where a syntactic look at every
/// refused schema measured +0.2%.
struct Room {
    nodes: Cell<u32>,
    unread: Cell<bool>,
}

impl Room {
    fn new(nodes: u32) -> Self {
        Self {
            nodes: Cell::new(nodes),
            unread: Cell::new(false),
        }
    }
}

/// Whether a union's members hold the top, a member that is a union read as
/// its members at any depth.
fn holds_the_top(members: &[Schema]) -> bool {
    let mut unread: Vec<&Schema> = members.iter().collect();
    while let Some(member) = unread.pop() {
        match member {
            Schema::Anything(_) => return true,
            Schema::Union(inner) => unread.extend(inner.iter()),
            _ => {}
        }
    }
    false
}

/// [`lower`] with the nodes left to spend, which every node spends one of.
///
/// Each node lowers to the set it denotes: the bounds to the universe and the
/// empty set, and a union, a meet and a complement to the union, intersection
/// and complement of their members' sets [FCB08, Def. 4.1].
fn descend(schema: &Schema, pool: &dyn Constants, budget: &Room, depth: u32) -> Option<Descr> {
    budget.nodes.set(budget.nodes.get().checked_sub(1)?);
    let depth = match depth.checked_sub(1) {
        Some(depth) => depth,
        None if starts_a_build(schema) => return None,
        None => 0,
    };
    match schema {
        Schema::Anything(_) => Some(Descr::anything()),
        Schema::Nothing => Some(Descr::nothing()),
        Schema::NoneType => Some(Descr::of_kind(Kind::NoneType)),
        Schema::Bool => Some(Descr::of_kind(Kind::Bool)),
        // `bool` subclasses `int`, so every boolean is an integer. The two are
        // separate *kinds* here, which is what makes their components
        // independent -- so the schema that denotes both spells both.
        Schema::Int => Descr::of_kind(Kind::Int).union(&Descr::of_kind(Kind::Bool)),
        Schema::Float => Some(Descr::of_kind(Kind::Float)),
        Schema::Str => Some(Descr::of_kind(Kind::Str)),
        Schema::Bytes => Some(Descr::of_kind(Kind::Bytes)),
        Schema::Literal(index) => singleton(&pool.constant(*index)?),
        Schema::Seq { container, shape } => sequence(*container, shape, pool, budget, depth),
        Schema::Coll { container, element } => {
            let kind = match container {
                CollKind::Set => Kind::Set,
                CollKind::FrozenSet => Kind::FrozenSet,
            };
            Descr::set(&descend(element, pool, budget, depth)?, kind)
        }
        // A union holding the top is the top whatever its other members are,
        // so a member the lowering refuses refuses no such union. The
        // constructors fold one; a term built raw carries it, and is read on a
        // refusal alone, so a union that lowers pays nothing for the look.
        Schema::Union(members) => members
            .iter()
            .try_fold(Descr::nothing(), |whole, member| {
                whole.union(&descend(member, pool, budget, depth)?)
            })
            .or_else(|| holds_the_top(members).then(Descr::anything)),
        Schema::Intersection(members) => {
            members.iter().try_fold(Descr::anything(), |whole, member| {
                whole.intersect(&descend(member, pool, budget, depth)?)
            })
        }
        Schema::Complement(inner) => Some(descend(inner, pool, budget, depth)?.complement()),
        Schema::Refine { base, constraints } => refine(base, constraints, pool, budget, depth),
        // Every attribute narrows the same value, so the record is their meet.
        // Each attribute's type is a descriptor in its own right, which is what
        // makes the attribute half recursive, and each is required: the value
        // carries it.
        Schema::AttrRecord { fields } => {
            fields.iter().try_fold(Descr::anything(), |whole, field| {
                let ty = descend(&field.schema, pool, budget, depth)?;
                whole.intersect(&Descr::attribute(&field.name, &ty, false))
            })
        }
        Schema::KeyedMap { fields, defaults } => map(fields, defaults, pool, budget, depth),
        Schema::Instance(index) => Some(Descr::instance_of(pool.class(*index)?)),
        // A reference is a cycle a finite descriptor has no room for.
        Schema::Ref(_) | Schema::SelfRef(_) => None,
    }
}

/// The dicts a keyed map spells.
///
/// A record's field and a mapping's literal key both name a key and say what
/// it maps to, so both become labels -- and they differ in one thing the atom
/// has to be told. A field is read *instead of* the clauses: a key a field
/// names is governed by the field alone. A literal-keyed clause is read
/// *beside* them: clauses are a disjunction, so a key the clause names is
/// admitted when it or any other clause covering that key admits the value,
/// and the label's type is the join of every clause that reads it. A clause
/// whose key is a *kind* opens that part of the key partition [Cas23, §4.4].
/// What no clause opens stays shut, which is what makes a record with no
/// catch-all closed [Cas23, §3.2].
fn map(
    fields: &[Field],
    defaults: &[MapClause],
    pool: &dyn Constants,
    budget: &Room,
    depth: u32,
) -> Option<Descr> {
    let mut labels: Vec<(Label, Descr, bool)> = Vec::with_capacity(fields.len());
    for field in fields {
        let ty = descend(&field.schema, pool, budget, depth)?;
        labels.push((Label::str(&field.name), ty, !field.required));
    }
    // Every clause first -- what it names, what it opens, what it maps to --
    // because a named key's type is read off all of them.
    let mut clauses: Vec<(Vec<Label>, Vec<Option<Kind>>, Descr)> =
        Vec::with_capacity(defaults.len());
    for clause in defaults {
        let value = descend(&clause.value, pool, budget, depth)?;
        let (named, parts) = key_cover(&clause.key, pool)?;
        clauses.push((named, parts, value));
    }
    let mut opened: Vec<(Option<Kind>, Descr)> = Vec::new();
    for (index, (named, parts, value)) in clauses.iter().enumerate() {
        for label in named {
            // A field of this name is read instead of the clause.
            if fields.iter().any(|field| Label::str(&field.name) == *label) {
                continue;
            }
            // A clause names a key without requiring it: `dict[Literal["a"], V]`
            // admits the dict that carries no `a` at all. And the key is read
            // by every clause that covers it, so its type is the join.
            let mut ty = value.clone();
            for (other, (other_named, other_parts, other_value)) in clauses.iter().enumerate() {
                let covers = other != index
                    && (other_named.contains(label)
                        || other_parts.iter().any(|part| *part == Some(label.kind())));
                if covers {
                    ty = ty.union(other_value)?;
                }
            }
            labels.push((label.clone(), ty, true));
        }
        for part in parts {
            opened.push((*part, value.clone()));
        }
    }
    Descr::keyed_map(labels, opened)
}

/// The keys a clause governs: the constants it names, and the parts of the key
/// partition it covers whole.
///
/// `None` where it is neither -- a key schema covering *part* of a kind, such as
/// a regex over strings, would need the default to hold a set of keys rather
/// than a part of the partition, and that is the theory of overlapping domains
/// the paper sets aside [Cas23, §4.5], "it is possible to define a theory for
/// maps with overlapping domains, but in that case, there would not be any
/// difference between record types and an intersection of function types whose
/// codomain may contain an undefined value". Refusing is what keeps the default
/// a function.
///
/// A complement of whole parts is the other parts. The parts are a partition of
/// the keys, so a key outside every part `inner` opens is in one of the rest,
/// the part for a key of no listed kind among them: `~str` opens seven kinds
/// and the classes, `~int` neither `int` nor `bool`, and `~bool` opens `int`,
/// whose part holds no boolean. That is a union of key types, which
/// [Cas23, fn. 15] lets a domain be, and the remaining parts are a complement's
/// keys because the key types cover the keys [Cas23, §4.4], each part with its
/// own default [Cas23, Def. 4.6]. A complement of a constant cuts a part, so it
/// is refused with the constant.
pub(crate) fn key_cover(
    key: &Schema,
    pool: &dyn Constants,
) -> Option<(Vec<Label>, Vec<Option<Kind>>)> {
    let part = |kind: Kind| Some((Vec::new(), vec![Some(kind)]));
    let every_part = || KEY_KINDS.iter().copied().map(Some).chain([None]);
    match key {
        Schema::Nothing => Some((Vec::new(), Vec::new())),
        // Every part, the one for a key of no listed kind included.
        Schema::Anything(_) => Some((Vec::new(), every_part().collect())),
        Schema::NoneType => part(Kind::NoneType),
        Schema::Bool => part(Kind::Bool),
        // A `bool` key **is** an `int` key: `{True: 1}` is a dict whose key is
        // an integer, and the walk admits it under `dict[int, V]`. Opening only
        // the `Int` part would name a set smaller than the schema denotes.
        Schema::Int => Some((Vec::new(), vec![Some(Kind::Int), Some(Kind::Bool)])),
        Schema::Float => part(Kind::Float),
        Schema::Str => part(Kind::Str),
        Schema::Bytes => part(Kind::Bytes),
        // A tuple or a frozenset of anything is every key of its kind: a key
        // is hashable, and a hashable tuple is still a tuple. Any other shape
        // holds part of the kind or none of it: `tuple[list[int]]` is no key.
        Schema::Seq {
            container: SeqKind::Tuple,
            shape,
        } if shape.prefix.is_empty()
            && matches!(shape.tail.as_deref(), Some(Schema::Anything(_))) =>
        {
            part(Kind::Tuple)
        }
        Schema::Coll {
            container: CollKind::FrozenSet,
            element,
        } if matches!(**element, Schema::Anything(_)) => part(Kind::FrozenSet),
        Schema::Complement(inner) => {
            let (named, opened) = key_cover(inner, pool)?;
            named.is_empty().then(|| {
                let rest = every_part().filter(|slot| !opened.contains(slot));
                (Vec::new(), rest.collect())
            })
        }
        Schema::Literal(index) => Some((vec![label_of(&pool.constant(*index)?)?], Vec::new())),
        Schema::Union(members) => {
            let mut labels = Vec::new();
            let mut parts = Vec::new();
            for member in members.iter() {
                let (mine, theirs) = key_cover(member, pool)?;
                labels.extend(mine);
                parts.extend(theirs);
            }
            Some((labels, parts))
        }
        _ => None,
    }
}

/// The constant a pooled operand is, where a map atom could name it as a key.
///
/// A label is a key's value and not its type: the label `"a"` is every `str`
/// key equal to `"a"`, a subclass's included, which is the key a record's field
/// names, since the walk looks the name up. A literal key is the exact type
/// alone, so reading one as a label reads it wider; [`an_empty_reading_stands`] says
/// where that costs a proof.
fn label_of(operand: &Operand) -> Option<Label> {
    match operand {
        Operand::NoneType => Some(Label::NoneType),
        Operand::Boolean(value) => Some(Label::Bool(*value)),
        Operand::Integer(value) => Some(Label::Int(*value)),
        Operand::Word(word, kind) => Some(Label::Word(word.clone(), *kind)),
        // A float is not a `Literal` the typing spec allows, and an instance is
        // not a constant.
        Operand::Float(_) | Operand::Instance(_) => None,
    }
}

/// Whether an empty reading of a difference stands, where literal keys were
/// read as labels.
///
/// `widened` are the schemas the difference holds the values of, and
/// `narrowed` the ones it subtracts: `a ≤ b` reads `a` widened and `b`
/// narrowed, or `a` and `c` both widened where `b` is `¬c`.
///
/// A literal key of a kind with subclasses -- an `int`, a `str`, a `bytes` --
/// becomes a label, which also holds the subclass keys equal to it
/// (`label_of`). No descriptor tells such a key from the exact one, so
/// swapping one for the other moves no value in or out of any descriptor, and
/// on a dict whose keys are exact a label and its literal agree. A difference
/// read *inhabited* therefore holds a value with exact keys, which is a value
/// of the real difference: a refutation stands wherever the label is.
///
/// A difference read *empty* is a proof where every such label widens it,
/// which is the widened side. On the narrowed side the label subtracts a
/// subclass key's dict the real set keeps, and the proof stands only where
/// the difference holds no such dict: every narrowed label acts on the dict at
/// the top of the value, and a widened schema, read as a meet, has a member
/// whose every key a literal names -- every value of the difference then has
/// exact keys at the top. A label on a dict nested in another value, or a top
/// a record's field or a kind's part can key, leaves the reading undecided.
///
/// Unfolds a reference as [`lower_unfolded`] does, so it reads the schemas
/// that were lowered; a reference no definition resolves is cut, as lowering
/// one refuses.
#[must_use]
pub(crate) fn an_empty_reading_stands(
    widened: &[&Schema],
    narrowed: &[&Schema],
    definitions: &[Schema],
    pool: &dyn Constants,
) -> bool {
    let unfold = |schema: &Schema, side| schema.unfolded(definitions, UNFOLDS, side);
    let widened: Vec<Schema> = widened
        .iter()
        .map(|schema| unfold(schema, Polarity::Widen))
        .collect();
    let narrowed: Vec<Schema> = narrowed
        .iter()
        .map(|schema| unfold(schema, Polarity::Narrow))
        .collect();
    let reach = widened
        .iter()
        .map(|schema| narrowed_labels(schema, Polarity::Widen, pool))
        .chain(
            narrowed
                .iter()
                .map(|schema| narrowed_labels(schema, Polarity::Narrow, pool)),
        )
        .max()
        .unwrap_or(Labels::None);
    match reach {
        Labels::None => true,
        Labels::Top => widened.iter().any(|schema| keys_are_literals(schema, pool)),
        Labels::Nested => false,
    }
}

/// Where a schema read on one side subtracts a literal key read as a label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Labels {
    /// Nowhere.
    None,
    /// On the dict at the top of the value, and nowhere deeper.
    Top,
    /// On a dict nested in another value.
    Nested,
}

/// The farthest a literal key read as a label is subtracted, reading `schema`
/// on `side`: a complement flips the side, and a node holding other values --
/// a container's element, a field, a clause's value -- puts them below the top.
/// A refinement is read as such a node: over a map it lowers to nothing, so
/// what it holds is never a top the reading could keep.
fn narrowed_labels(schema: &Schema, side: Polarity, pool: &dyn Constants) -> Labels {
    let mut reach = Labels::None;
    let mut pending = vec![(schema, side, true)];
    while let Some((node, side, top)) = pending.pop() {
        match node {
            Schema::Complement(inner) => pending.push((inner, side.flipped(), top)),
            Schema::Union(members) | Schema::Intersection(members) => {
                pending.extend(members.iter().map(|member| (member, side, top)));
            }
            _ => {
                if let Schema::KeyedMap { defaults, .. } = node
                    && side == Polarity::Narrow
                    && defaults
                        .iter()
                        .any(|clause| names_a_wide_key(&clause.key, pool))
                {
                    reach = reach.max(if top { Labels::Top } else { Labels::Nested });
                }
                pending.extend(node.children().map(|child| (child, side, false)));
            }
        }
    }
    reach
}

/// Whether every value of `schema`, read as a meet, is a dict whose every key
/// a literal clause names: a keyed map with no field and only literal keys, or
/// a meet holding one. Such a dict's keys are exact.
fn keys_are_literals(schema: &Schema, pool: &dyn Constants) -> bool {
    match schema {
        Schema::KeyedMap { fields, defaults } => {
            fields.is_empty()
                && defaults
                    .iter()
                    .all(|clause| names_only_literals(&clause.key, pool))
        }
        Schema::Intersection(members) => {
            members.iter().any(|member| keys_are_literals(member, pool))
        }
        _ => false,
    }
}

/// Whether a clause's key is literals and nothing else, each one a label.
fn names_only_literals(key: &Schema, pool: &dyn Constants) -> bool {
    match key {
        Schema::Literal(index) => pool.constant(*index).as_ref().and_then(label_of).is_some(),
        Schema::Union(members) => members
            .iter()
            .all(|member| names_only_literals(member, pool)),
        _ => false,
    }
}

/// Whether a clause's key names a literal its label reads wider.
fn names_a_wide_key(key: &Schema, pool: &dyn Constants) -> bool {
    match key {
        Schema::Literal(index) => matches!(
            pool.constant(*index),
            Some(Operand::Integer(_) | Operand::Word(..))
        ),
        Schema::Union(members) => members.iter().any(|member| names_a_wide_key(member, pool)),
        _ => false,
    }
}

/// The set holding one pooled value and nothing else, the model's singleton
/// [FCB08, §4.2].
///
/// The value at its exact type: a pooled constant is an exact `int`, `float`,
/// `str` or `bytes`, and a kind also holds the builtin's subclasses, so the
/// value is met with [`Class::exact`]. `bool` and `None` have no subclass.
fn singleton(constant: &Operand) -> Option<Descr> {
    let exactly =
        |value: Descr, kind: Kind| value.intersect(&Descr::instance_of(Class::exact(kind)));
    match constant {
        Operand::Boolean(value) => Some(Descr::boolean(*value)),
        Operand::Integer(value) => exactly(Descr::integer(*value), Kind::Int),
        Operand::Float(value) => exactly(Descr::float(*value), Kind::Float),
        Operand::Word(word, kind) => exactly(Descr::word(word, *kind)?, *kind),
        Operand::NoneType => Some(Descr::of_kind(Kind::NoneType)),
        // A class is a set of objects, not one value; a literal naming an
        // instance also pins *which* instance, which the descriptor cannot say.
        Operand::Instance(_) => None,
    }
}

/// The sequences a shape spells, under the kind that reads them.
fn sequence(
    container: SeqKind,
    shape: &SeqShape,
    pool: &dyn Constants,
    budget: &Room,
    depth: u32,
) -> Option<Descr> {
    let kind = match container {
        SeqKind::List => Kind::List,
        SeqKind::Tuple => Kind::Tuple,
    };
    let prefix: Option<Vec<Descr>> = shape
        .prefix
        .iter()
        .map(|element| descend(element, pool, budget, depth))
        .collect();
    let tail = match &shape.tail {
        Some(element) => Some(descend(element, pool, budget, depth)?),
        None => None,
    };
    Descr::sequence(&prefix?, tail.as_ref(), kind)
}

/// The base narrowed by every constraint, which is a meet, as a refined base
/// type is its base cut down by its refinement [JV20, §3.2].
///
/// Each constraint is a *set* here rather than a test, so narrowing is
/// intersection and the order the constraints are written in carries no meaning
/// -- which is what makes `Ge(0) ∧ Lt(0)` decide rather than run.
///
/// A constraint is asked of one value at a time, so a refinement of a union is
/// the union of its refined branches, as a meet distributes over a union
/// [FCB08, Def. 4.1]. `Annotated[int | float, Ge(0)]` is the non-negative ints
/// and the non-negative floats, each in the component that holds it, where the
/// union as one base lands in neither. And a constraint is Python's own
/// operation, whose error the walk reads as a non-match: `None` orders against
/// no builtin value, has no length, takes no remainder and matches no pattern,
/// and no class derives from `NoneType` to answer otherwise, so `None` under
/// any constraint but a predicate is the empty set.
fn refine(
    base: &Schema,
    constraints: &[Constraint],
    pool: &dyn Constants,
    budget: &Room,
    depth: u32,
) -> Option<Descr> {
    if let Schema::Union(branches) = base {
        return branches.iter().try_fold(Descr::nothing(), |whole, branch| {
            whole.union(&refine(branch, constraints, pool, budget, depth)?)
        });
    }
    if matches!(base, Schema::NoneType)
        && constraints
            .iter()
            .any(|constraint| holds_of_no_none(constraint, pool))
    {
        return Some(Descr::nothing());
    }
    let mut narrowed = descend(base, pool, budget, depth)?;
    for constraint in constraints {
        let Some(set) = constrained(constraint, &narrowed, pool) else {
            if matches!(constraint, Constraint::Predicate(_)) {
                budget.unread.set(true);
            }
            return None;
        };
        narrowed = narrowed.intersect(&set)?;
    }
    Some(narrowed)
}

/// Whether `constraint` holds of no `None`: it raises against `None`, and the
/// walk reads its error as a non-match. Every constraint but a predicate does,
/// whose operand is a builtin value -- a comparison is asked of the operand's
/// own type too, reflected, and a class may answer `None` there.
fn holds_of_no_none(constraint: &Constraint, pool: &dyn Constants) -> bool {
    match constraint {
        Constraint::Ge(index)
        | Constraint::Gt(index)
        | Constraint::Le(index)
        | Constraint::Lt(index)
        | Constraint::MultipleOf(index) => matches!(
            pool.operand(*index),
            Some(
                Operand::Boolean(_)
                    | Operand::Integer(_)
                    | Operand::Float(_)
                    | Operand::Word(..)
                    | Operand::NoneType
            )
        ),
        Constraint::MinLen(_) | Constraint::MaxLen(_) | Constraint::Regex(_) => true,
        Constraint::Predicate(_) => false,
    }
}

/// The set one constraint denotes, read under the kinds the base admits.
///
/// A bound is a set of *numbers* and a length bound a set of *words* or
/// *sequences*, so which component a constraint lands in depends on what it is
/// narrowing. A constraint the descriptor cannot read refuses: a predicate, a
/// bound over a base that is not whole numbers alone or floats alone, an
/// operand the base's component cannot spell, a length bound over a base
/// admitting a set, a dict or a value with no length, and a pattern whose
/// automaton is past its bound.
fn constrained(constraint: &Constraint, base: &Descr, pool: &dyn Constants) -> Option<Descr> {
    // A bound is a set of whole numbers, and `bool` is a kind of its own here,
    // so the set has to be spelled in both slots: `True` is `1` to every
    // comparison Python makes, and a bound that admits `1` admits it.
    //
    // **Refuses unless the base is whole numbers and nothing else**, for the
    // reason [`lengths`] refuses a length bound over a set or a dict. A bound
    // orders whatever a value's type orders -- a float, a string, a date -- and
    // these two components speak for two of those. Narrowing a float base to a
    // set of integers gives a *smaller* set than the schema denotes, and a
    // smaller set has a larger complement, which is a subtype proof no value
    // supports.
    let integers = |set: IntSet| {
        if !base.within(&[Kind::Int, Kind::Bool]) {
            return None;
        }
        let mut descr = Descr::nothing();
        descr.integers(set.clone());
        descr.booleans(
            [false, true]
                .into_iter()
                .filter(|boolean| set.holds(i64::from(*boolean)))
                .fold(BoolSet::EMPTY, |held, boolean| {
                    held.union(BoolSet::just(boolean))
                }),
        );
        Some(descr)
    };
    // The same shape for a float base, so a relation needing the descriptor
    // and mentioning `Annotated[float, Gt(0)]` is decided rather than declined.
    //
    // **Refuses unless the base is floats and nothing else**, for the reason the
    // integer side refuses: narrowing a wider base to a set of floats gives a
    // smaller set than the schema denotes, and a smaller set has a larger
    // complement. `nan` is outside every interval, which is what `FloatSet`'s
    // constructors already say and what Python's own comparisons do.
    let floats = |set: FloatSet| {
        if !base.within(&[Kind::Float]) {
            return None;
        }
        let mut descr = Descr::nothing();
        descr.floats(set);
        Some(descr)
    };
    // A bound's operand is whatever the caller wrote: `Annotated[float, Gt(0)]`
    // carries the *integer* zero, and it orders the floats all the same. So the
    // side a bound lands on is chosen by the **base**, not by the operand's own
    // type, and the operand is read as a number either way -- an integer past
    // 2^53 against its neighbors, as [`float_bound`] says.
    let float_bound = |constraint: &Constraint, index: &OperandIx| {
        float_bound(constraint, &pool.operand(*index)?)
    };
    let base_is_floats = base.within(&[Kind::Float]);
    // A word orders the words of its own kind, which the word component holds
    // as an automaton: read where the bound is a word and the base holds no
    // value but of that word's kind.
    if let Constraint::Ge(index)
    | Constraint::Gt(index)
    | Constraint::Le(index)
    | Constraint::Lt(index) = constraint
        && let Some(Operand::Word(bound, kind)) = pool.operand(*index)
        && base.within(&[kind])
    {
        let above = matches!(constraint, Constraint::Ge(_) | Constraint::Gt(_));
        let inclusive = matches!(constraint, Constraint::Ge(_) | Constraint::Le(_));
        return Descr::words_ordered(&bound, kind, above, inclusive);
    }
    match constraint {
        Constraint::Ge(index) | Constraint::Gt(index) if base_is_floats => {
            floats(float_bound(constraint, index)?)
        }
        Constraint::Le(index) | Constraint::Lt(index) if base_is_floats => {
            floats(float_bound(constraint, index)?)
        }
        Constraint::Ge(index)
        | Constraint::Gt(index)
        | Constraint::Le(index)
        | Constraint::Lt(index) => integers(integer_bound(constraint, &pool.operand(*index)?)?),
        Constraint::MultipleOf(index) => {
            let Operand::Integer(step) = pool.operand(*index)? else {
                return None;
            };
            integers(IntSet::multiple_of(step)?)
        }
        Constraint::MinLen(least) => {
            lengths(base, &|kind| Descr::words_at_least(*least, kind), &|kind| {
                Descr::sequences_at_least(*least, kind)
            })
        }
        Constraint::MaxLen(most) => {
            lengths(base, &|kind| Descr::words_at_most(*most, kind), &|kind| {
                Descr::sequences_at_most(*most, kind)
            })
        }
        Constraint::Regex(pattern) => words(pattern, base),
        // A callback is a leaf the core cannot read, which is what makes it
        // opaque to the procedure beside this one too.
        Constraint::Predicate(_) => None,
    }
}

/// Where an integer bound falls among the floats.
///
/// Every integer up to 2^53 is a float. Past that a bound may have *no* float
/// equal to it, and then the set it cuts is bounded by a neighbor rather than
/// by the bound itself -- which is a different set from the one the nearest
/// float would cut, in whichever direction that rounding went.
///
/// The neighbor is the one on the side the bound keeps, so the type has no
/// reading for the other side and no caller has a crossed case to answer.
enum Straddle {
    /// A float equals the bound.
    Exact(f64),
    /// No float does; this is the nearest one on the side the bound keeps --
    /// the smallest above it for a lower bound, the largest below it for an
    /// upper one.
    Beside(f64),
}

/// Which floats an integer bound cuts between.
///
/// `lower` picks the side: a lower bound is answered by the smallest float above
/// an inexact value, an upper bound by the largest below it, and both by the
/// value itself where it is exact.
fn straddle(value: i64, lower: bool) -> Straddle {
    #[expect(
        clippy::cast_precision_loss,
        reason = "the rounding is the thing measured: its direction picks the neighbor"
    )]
    let nearest = value as f64;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "an integral float inside the i64 range converts exactly"
    )]
    let round_trip = nearest as i128;
    match round_trip.cmp(&i128::from(value)) {
        core::cmp::Ordering::Equal => Straddle::Exact(nearest),
        // The nearest float overshot, so it *is* the one above.
        core::cmp::Ordering::Greater if lower => Straddle::Beside(nearest),
        core::cmp::Ordering::Greater => Straddle::Beside(nearest.next_down()),
        core::cmp::Ordering::Less if lower => Straddle::Beside(nearest.next_up()),
        core::cmp::Ordering::Less => Straddle::Beside(nearest),
    }
}

/// The integers a comparison bound admits, for a base that is not the floats.
///
/// Python compares an `int` with a `float` exactly, so a float operand is read
/// as the integer on the near side of it, and then as an integer bound is:
/// `n > x` is `n > floor(x)` and `n >= x` is `n >= ceil(x)`, so `n > 0.5` and
/// `n > 0.0` are both `n >= 1`; the upper bounds mirror them. An infinite
/// operand admits every integer or none, and `nan`, which orders nothing,
/// declines, as does a bound past the range an integer here holds.
fn integer_bound(constraint: &Constraint, operand: &Operand) -> Option<IntSet> {
    let lower = matches!(constraint, Constraint::Ge(_) | Constraint::Gt(_));
    let strict = matches!(constraint, Constraint::Gt(_) | Constraint::Lt(_));
    let bound = match *operand {
        Operand::Integer(value) => value,
        Operand::Boolean(value) => i64::from(value),
        Operand::Float(value) if value.is_infinite() => {
            return Some(if value.is_sign_positive() == lower {
                IntSet::empty()
            } else {
                IntSet::all()
            });
        }
        Operand::Float(value) => {
            let near = if lower == strict {
                value.floor()
            } else {
                value.ceil()
            };
            if !(-I64_EDGE..I64_EDGE).contains(&near) {
                return None;
            }
            #[expect(
                clippy::cast_possible_truncation,
                reason = "an integral float inside the i64 range converts exactly"
            )]
            let near = near as i64;
            near
        }
        _ => return None,
    };
    Some(match (lower, strict) {
        (true, false) => IntSet::between(Some(bound), None),
        (true, true) => IntSet::between(Some(bound.checked_add(1)?), None),
        (false, false) => IntSet::between(None, Some(bound)),
        (false, true) => IntSet::between(None, Some(bound.checked_sub(1)?)),
    })
}

/// `2^63`, the first float past the integers an `i64` holds; `-2^63` is the
/// least one it holds. `nan` is inside no range, so it declines here too.
const I64_EDGE: f64 = 9_223_372_036_854_775_808.0;

/// The floats a comparison bound admits, for a base that is the floats.
///
/// An integer operand past 2^53 has no float equal to it, and rounding it to the
/// nearest one moves the boundary whichever way that rounding went -- naming a
/// set of floats the schema does not denote, in either direction. It is read
/// against its neighbors instead: a float is `>= n` exactly when it is at or
/// above the smallest float above `n`, and `<= n` exactly when it is at or below
/// the largest float below `n`. Strictness costs nothing there, since no float
/// equals the bound for `>` or `<` to exclude. Python compares an `int` with a
/// `float` exactly, and so does this.
fn float_bound(constraint: &Constraint, operand: &Operand) -> Option<FloatSet> {
    let lower = matches!(constraint, Constraint::Ge(_) | Constraint::Gt(_));
    let strict = matches!(constraint, Constraint::Gt(_) | Constraint::Lt(_));
    let value = match *operand {
        Operand::Float(value) => Straddle::Exact(value),
        Operand::Integer(value) => straddle(value, lower),
        Operand::Boolean(value) => straddle(i64::from(value), lower),
        _ => return None,
    };
    // No float equals an inexact bound, so `>` and `>=` admit the same ones and
    // the neighbor carries what the strictness would have.
    Some(match value {
        Straddle::Exact(bound) if strict => {
            if lower {
                FloatSet::above(bound)
            } else {
                FloatSet::below(bound)
            }
        }
        Straddle::Exact(bound) | Straddle::Beside(bound) => {
            if lower {
                FloatSet::at_least(bound)
            } else {
                FloatSet::at_most(bound)
            }
        }
    })
}

/// The values of the base's kinds whose length the bound admits.
///
/// A length is not a word's alone, and for two of the kinds that have one it is
/// a property the representation can state: a word's length is a pattern over
/// its alphabet, and a sequence's is "any element, that many times", which is
/// as regular as any other shape the automaton holds. Both are built here and
/// unioned, so a base admitting words *and* sequences is bounded on each.
///
/// **Refuses for a base admitting anything else.** A set and a dict have a
/// length their components do not count, and a value with no length at all
/// fails the bound by raising -- which the walk reads as a non-member and this
/// cannot express. Lowering the bound while ignoring those kinds would give a
/// set larger than the schema denotes, and a larger set has a smaller
/// complement: a subtype proof no value supports.
fn lengths(
    base: &Descr,
    words: &dyn Fn(Kind) -> Option<Descr>,
    sequences: &dyn Fn(Kind) -> Option<Descr>,
) -> Option<Descr> {
    const WORDS: [Kind; 2] = [Kind::Str, Kind::Bytes];
    const SEQUENCES: [Kind; 2] = [Kind::List, Kind::Tuple];

    if !base.within(&[Kind::Str, Kind::Bytes, Kind::List, Kind::Tuple]) {
        return None;
    }
    // The first kind's part is the union so far, rather than a union with the
    // empty set: that union is a product of automata, one state per element
    // of a sequence bound, spent to rebuild the part it was given.
    let mut whole: Option<Descr> = None;
    let parts = WORDS
        .iter()
        .map(|kind| (*kind, words))
        .chain(SEQUENCES.iter().map(|kind| (*kind, sequences)));
    for (kind, part) in parts {
        if base.reaches(kind) {
            let part = part(kind)?;
            whole = Some(match whole {
                Some(whole) => whole.union(&part)?,
                None => part,
            });
        }
    }
    // The base lies within the counted kinds, so no part is built exactly when
    // the base is empty -- and a bound over an empty base denotes the empty
    // set, which is an answer rather than a refusal.
    Some(whole.unwrap_or_else(Descr::nothing))
}

/// The strings a pattern matches whole, whatever else the base admits.
///
/// A pattern matches the text of a string, and no other value has any: the
/// set is the pattern's language over `str`, `bytes` included among the kinds
/// it leaves out. [`refine`] meets it with the base, which drops the strings
/// the base does not hold and every value of another kind, so no kind beside
/// `str` needs reading here. Read over `bytes` as well, the set would be
/// larger than the schema denotes, and a larger set on the supertype side is
/// a subtype proof that no value supports: `Annotated[bytes, MaxLen(0)]`
/// below `Annotated[str | bytes, Regex("a*")]`, with `b""` standing against
/// it.
///
/// A base holding no string meets the pattern in nothing. That is an answer,
/// given before the automaton is built, so a pattern past its size bound
/// still decides it.
fn words(pattern: &str, base: &Descr) -> Option<Descr> {
    if !base.reaches(Kind::Str) {
        return Some(Descr::nothing());
    }
    Descr::pattern(pattern, Kind::Str)
}

#[cfg(test)]
mod tests;
