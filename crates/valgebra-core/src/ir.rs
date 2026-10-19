//! The schema intermediate representation: the IR node definitions, the
//! construction that keeps them canonical, and the questions asked of a node
//! that need no value -- its depth, its node count, whether a reference occurs
//! under a guard.
//!
//! Rewriting one node into another -- shifting an index into another
//! validator's pool, resolving a self-reference, unfolding a recursive
//! definition, opening a record, pruning the definitions table -- is
//! [`transform`], which is a child of this module because every rewrite is
//! written against the node definitions here and none of them adds a shape.
//!
//! ## The index spaces
//!
//! Five payloads in this module are integers that address something the
//! validator holds, and four of them address the *same* constants pool: a
//! literal's constant, a class, a comparison operand, and a user predicate. A
//! bare index used against the wrong one of those retrieves a real object of the
//! wrong kind, so the failure is a plausible wrong verdict rather than a panic.
//! Each therefore has its own type, minted by one named constructor and opened
//! by [`get`](ConstIx::get), which is the line a reviewer reads. The fifth,
//! [`DefIx`], addresses the definitions table instead.
//!
//! The two *shifts* applied when two validators are composed are typed for the
//! same reason: [`Schema::shifted`] takes one per space, where two adjacent
//! `usize` arguments would let a caller transpose them in silence.
//!
//! The types stop a shift reaching the wrong space. They say nothing about
//! whether a payload was shifted at all, which is why one walk serves both ways
//! of combining two pools ([`Remap`]) and why the child set of each variant is
//! stated in two walks and no others: the one a rewrite goes through
//! ([`Schema::map_children`]) and `push_children`, the one a reading goes
//! through.

use std::cell::RefCell;
use std::sync::{Arc, OnceLock};

use crate::codes;
use crate::oracle::{Connective, NoLeafRelations, has_complementary_pair_within};

mod intern;
mod transform;

pub(crate) use transform::Polarity;
pub use transform::pruned;

/// Remap a pool index through the map built when two validators merge.
///
/// Every index is in range by construction, so a miss is an internal invariant
/// break; the map keeps the original index rather than panicking, so a malformed
/// merge degrades to a (later bounds-checked) wrong lookup instead of aborting.
fn remap(lit_map: &[usize], index: usize) -> usize {
    debug_assert!(
        index < lit_map.len(),
        "literal index {index} out of remap range"
    );
    lit_map.get(index).copied().unwrap_or(index)
}

/// How far every constants-pool index moves when two pools are appended.
///
/// Distinct from [`DefShift`] so the two cannot be transposed at
/// [`Schema::shifted`], which takes one of each.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct PoolShift(usize);

impl PoolShift {
    /// A shift of `by` pool slots -- in practice the length of the pool the
    /// second validator's constants are appended to.
    ///
    /// There is no accessor: a shift is only ever *applied*, by the index types
    /// that know which space they belong to, so nothing outside this module has
    /// a use for the distance as a bare integer.
    #[must_use]
    pub const fn new(by: usize) -> Self {
        Self(by)
    }
}

/// How far every definitions-table index moves when a second validator's
/// definitions are appended to a first. See [`PoolShift`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct DefShift(usize);

impl DefShift {
    /// A shift of `by` definition slots. No accessor, for the reason
    /// [`PoolShift::new`] gives.
    #[must_use]
    pub const fn new(by: usize) -> Self {
        Self(by)
    }
}

/// Define one index space over the constants pool.
///
/// Every such type is a `usize` in layout and a distinct set of values to the
/// compiler. `new` is the only way in and `get` the only way out, so a
/// conversion is always a place a reader can see; there is deliberately no
/// `From<usize>`.
macro_rules! pool_index {
    ($(#[$meta:meta])* $name:ident, $what:literal) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        #[repr(transparent)]
        pub struct $name(usize);

        impl $name {
            #[doc = concat!("The pool slot holding ", $what, ".")]
            ///
            /// The caller is the code that put the object in the pool, so this
            /// is where the index acquires its meaning.
            #[must_use]
            pub const fn new(index: usize) -> Self {
                Self(index)
            }

            /// The underlying pool slot. Every call is a place the index stops
            /// carrying which space it belongs to, so there should be few.
            #[must_use]
            pub const fn get(self) -> usize {
                self.0
            }

            /// This index in a pool that has been appended to another.
            ///
            /// The sum cannot overflow: both terms are bounded by a live pool's
            /// length, so their sum is bounded by the length of the pool they
            /// are being combined into. The release profile wraps rather than
            /// traps, so there that bound is the whole argument: a wrapped index
            /// would read a real object of the wrong kind.
            #[must_use]
            fn shifted(self, by: PoolShift) -> Self {
                debug_assert!(
                    self.0.checked_add(by.0).is_some(),
                    "pool index {} shifted by {} overflows",
                    self.0,
                    by.0
                );
                Self(self.0 + by.0)
            }

            /// This index after the pool was interned into another, collapsing
            /// identity-shared constants.
            #[must_use]
            fn remapped(self, lit_map: &[usize]) -> Self {
                Self(remap(lit_map, self.0))
            }

            /// This index under either way of combining two pools. The single
            /// entry point the schema walk calls, so a payload is remapped by
            /// asking *this* space how it moves rather than by the walk knowing
            /// which combination it is performing.
            #[must_use]
            fn remapped_by(self, remap: Remap<'_>) -> Self {
                match remap {
                    Remap::Append { pool, .. } => self.shifted(pool),
                    Remap::Intern { lit_map, .. } => self.remapped(lit_map),
                }
            }
        }
    };
}

/// How the top was written, kept so `repr` can give it back.
///
/// `typing.Any` and `anything` denote the same set — every Python value — and
/// one node holds both. A gradual type is an atom apart from the top for a
/// *second* question, consistency, which a static checker asks at every site
/// where a value crosses between typed and untyped code; a validator has no
/// such site and never asks it (`docs/dev/01-schema-ir.md`, "What `Any` is").
/// What is left is a spelling, and a spelling is worth giving back.
///
/// **It is not part of the set.** `PartialEq`, `Ord` and `Hash` are written by
/// hand so every spelling compares and hashes alike, which is what keeps the
/// flag out of the algebra: every rule that keys on equality, ordering or a
/// hash — which is every rule here — is looking at two values it cannot tell
/// apart. A direct `match` on the payload still can, and that is the one thing
/// this type cannot stop. In this crate only the sharing table does it
/// (`ir/intern.rs`), so a shared node never hands a caller back the spelling it
/// did not write; `render` in the bindings does it to print that spelling; no
/// relation or law may.
#[derive(Debug, Clone, Copy)]
pub enum Spelling {
    /// `anything`, the lattice top written as itself.
    Top,
    /// `typing.Any`.
    Any,
}

impl PartialEq for Spelling {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl Eq for Spelling {}

impl PartialOrd for Spelling {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Spelling {
    fn cmp(&self, _: &Self) -> core::cmp::Ordering {
        core::cmp::Ordering::Equal
    }
}

impl core::hash::Hash for Spelling {
    fn hash<H: core::hash::Hasher>(&self, _: &mut H) {}
}

pool_index!(
    /// The pool slot holding a [`Schema::Literal`]'s constant.
    ConstIx,
    "a literal's constant"
);
pool_index!(
    /// The pool slot holding the class of a [`Schema::Instance`].
    ClassIx,
    "a class"
);
pool_index!(
    /// The pool slot holding a comparison constraint's operand.
    OperandIx,
    "a comparison operand"
);
pool_index!(
    /// The pool slot holding a [`Constraint::Predicate`]'s callable.
    PredIx,
    "a user predicate"
);

/// A slot in the validator's definitions table: the target of a
/// [`Schema::Ref`] back edge.
///
/// A different table from the constants pool, and therefore a different type:
/// carried as a bare `usize` it would reach the pool's accessors, which answer
/// with a Python object rather than a schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct DefIx(usize);

impl DefIx {
    /// The definition at this slot of the validator's definitions table.
    #[must_use]
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    /// The underlying slot.
    #[must_use]
    pub const fn get(self) -> usize {
        self.0
    }

    /// This index in a definitions table that has been appended to another.
    ///
    /// Cannot overflow, for the reason [`ConstIx::shifted`] gives: both terms
    /// are bounded by a live definitions table's length.
    #[must_use]
    fn shifted(self, by: DefShift) -> Self {
        debug_assert!(
            self.0.checked_add(by.0).is_some(),
            "definition index {} shifted by {} overflows",
            self.0,
            by.0
        );
        Self(self.0 + by.0)
    }

    /// This index under either way of combining two validators. A definitions
    /// table is only ever appended to, never interned, so both combinations move
    /// a definition index the same way -- which is the fact that makes one
    /// `Remap` enough for both.
    #[must_use]
    fn remapped_by(self, remap: Remap<'_>) -> Self {
        self.shifted(remap.defs())
    }
}

/// How the two index spaces move when one validator's schema is rebuilt against
/// another's pools.
///
/// The constants pool combines in two ways -- appended, or interned so
/// identity-shared constants collapse -- and the definitions table in one. Two
/// whole structural walks over the IR, identical but for the leaf action, would
/// make a payload site reached by one and missed by the other a wrong index the
/// compiler cannot see. Naming the difference leaves one walk and puts the
/// difference at the leaf.
#[derive(Debug, Clone, Copy)]
enum Remap<'a> {
    /// The second pool is appended to the first: every pool index moves along by
    /// the first pool's length.
    Append { pool: PoolShift, defs: DefShift },
    /// The second pool is interned into the first: every pool index moves to the
    /// slot `lit_map` records for it.
    Intern {
        lit_map: &'a [usize],
        defs: DefShift,
    },
}

impl Remap<'_> {
    /// How far the definitions table moves, which both combinations carry.
    fn defs(self) -> DefShift {
        match self {
            Remap::Append { defs, .. } | Remap::Intern { defs, .. } => defs,
        }
    }
}

/// Whether a record admits keys it does not declare.
///
/// A name rather than a positional `bool`: `Schema::record(fields, false)` would
/// read as a flag whose polarity a caller has to remember, and the two openness
/// senses -- "closed to extra keys" and "open to them" -- are exactly the pair a
/// reader gets backwards.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Openness {
    /// Only the declared keys are admitted.
    Closed,
    /// Any further key is admitted, with any value.
    Open,
}

/// Whether a structural constructor has been crossed on the way to a recursive
/// reference.
///
/// A `recursive` definition is contractive only when every occurrence of its
/// self-reference sits under one. The condition travelled as a positional
/// `bool`, where `occurs_unguarded(id, false)` reads as neither "no guard yet"
/// nor "not guarded" without checking the callee.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Guarded {
    /// No structural constructor has been crossed yet.
    No,
    /// A structural constructor has been crossed, so anything below it is
    /// productive.
    Yes,
}

impl Guarded {
    /// The join, in which [`Yes`](Guarded::Yes) **absorbs**: crossing a structural
    /// constructor guards everything below it, however deeply it nests.
    ///
    /// The absorbing element is why the guardedness check answers what it does --
    /// only the arm demanding [`No`](Guarded::No) can report an unguarded
    /// occurrence -- and this is the one place it is stated, rather than in
    /// every structural arm.
    #[must_use]
    pub(crate) const fn join(self, other: Guarded) -> Guarded {
        match (self, other) {
            (Guarded::No, Guarded::No) => Guarded::No,
            _ => Guarded::Yes,
        }
    }
}

/// A single step in the location of a value inside a composite structure.
///
/// Scalar schemas never produce a path; structural schemas (records, sequences,
/// tuples, sets, mappings) push a segment per level as they descend. Ordered,
/// so a report can be put in an order that is a property of what it says: a
/// set's failing elements carry no position of their own.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum PathSegment {
    /// A mapping or record key that is a string.
    ///
    /// Shared rather than owned outright, for the reason a field's name is
    /// ([`Field::name`]): a record's failure path carries the very name the
    /// schema declares, and copying it per violation is a heap allocation per
    /// failing field of every failing record. A key read from a value rather
    /// than from the schema is copied once, into the same shared form.
    Key(Arc<str>),
    /// A mapping key that is an integer.
    ///
    /// Separate from [`PathSegment::Key`] because the path is what a caller
    /// walks back down to the value, and `d[2]` and `d["2"]` are different
    /// entries of the same dict: rendering an integer key as its text made the
    /// two indistinguishable and the path unusable on a dict keyed by numbers.
    /// Separate from [`PathSegment::Index`] because it is not a position.
    IntKey(i64),
    /// A mapping key that is an integer too large for [`PathSegment::IntKey`].
    ///
    /// Python's integers are unbounded and a dict may be keyed by any of them,
    /// so a key of `2**70` has to reach a caller as an integer or the path stops
    /// walking back down. Carried as its digits, a Python integer literal,
    /// because the core holds no Python object and has no big integer of its
    /// own: decimal, and `0x` hexadecimal past the interpreter's limit on the
    /// decimal digits it converts. The binding turns it back into the `int` it
    /// came from, which is the only form a caller can index with.
    BigIntKey(String),
    /// A sequence, tuple, or set position.
    Index(usize),
}

/// The clause list an open record carries, allocated once for the process.
///
/// Every open record's catch-all is the same pair -- any key, any value -- so
/// every record built or reopened would otherwise allocate its own copy. A shared
/// list makes the carry a refcount bump, which is what sharing the list is for;
/// the same goes for the empty lists a closed record and a fieldless one hold.
fn open_clauses() -> Clauses {
    static OPEN: OnceLock<Clauses> = OnceLock::new();
    OPEN.get_or_init(|| Clauses::from([MapClause::top()]))
        .clone()
}

/// The empty clause list, shared: a closed record's catch-all.
fn no_clauses() -> Clauses {
    static NONE: OnceLock<Clauses> = OnceLock::new();
    NONE.get_or_init(|| Clauses::from([])).clone()
}

/// The empty field list, shared: a mapping's declared names.
fn no_fields() -> Fields {
    static NONE: OnceLock<Fields> = OnceLock::new();
    NONE.get_or_init(|| Fields::from([])).clone()
}

/// The empty member list, shared: a sequence shape with no fixed prefix.
fn no_members() -> Members {
    static NONE: OnceLock<Members> = OnceLock::new();
    NONE.get_or_init(|| Members::from([])).clone()
}

/// The clause list for an openness, shared either way.
fn clauses_for(open: Openness) -> Clauses {
    match open {
        Openness::Open => open_clauses(),
        Openness::Closed => no_clauses(),
    }
}

/// The region no clause of `defaults` claims, as a key schema.
///
/// The clauses denote a union of key-by-value rectangles, so the keys they
/// speak for are the union of their key schemas and the rest is its complement.
/// A record claims nothing, so its unclaimed region is the whole key space --
/// which is why a bare catch-all is what openness reads as there.
fn unclaimed_region(defaults: &[MapClause]) -> Schema {
    Schema::union(defaults.iter().map(|clause| clause.key.clone())).complement()
}

/// What of `clause` stands once the default of the unclaimed region is taken
/// out: the clause, nothing, or the keys it claims beside that region.
///
/// That default is what `close` removes and `open` writes, and it is recognised
/// rather than marked: a clause carrying every value over exactly the keys the
/// others leave over. Recognising it structurally keeps this file below the
/// decision procedure, which is where the representation belongs.
///
/// **`open` folds the region into a clause carrying every value**, since two
/// clauses carrying one value are one clause over the union of their keys:
/// `{int: anything, str: int}` opens to `{int | ~(int | str): anything, str:
/// int}`. The complement in that key is the region `open` freed -- the
/// complement of every key claimed, the clause's own beside it included --
/// and the rest of the key is what the clause claims. So the complement alone
/// goes and the claimed keys stand, which is ICFP 2023's reading of openness
/// as the default of the key-types a record does not name, the named ones
/// untouched either way. Read as the whole clause, closing an opened mapping
/// over a class, or beside a clause with another value, would keep the freed
/// region and admit every key. Where the complement law has folded the key to
/// the top, the clause's own claim is not in the term to keep: what stays is
/// the clause over the keys the other clauses claim, at which it admits every
/// value, since closing touches no claimed region.
fn claimed_beside_the_region(
    clause: &MapClause,
    defaults: &[MapClause],
    index: usize,
) -> Option<MapClause> {
    if clause.value != Schema::ANYTHING {
        return Some(clause.clone());
    }
    let others = || {
        defaults
            .iter()
            .enumerate()
            .filter(move |(at, _)| *at != index)
            .map(|(_, other)| other.key.clone())
    };
    let claimed_by_others = Schema::union(others());
    if clause.key == claimed_by_others.clone().complement() {
        return None;
    }
    // Every key: the region and what the clause claimed, folded by the
    // complement law, so its own claim is not in the term to read back. What
    // the other clauses claim it admits every value at too, and that stays.
    if clause.key == Schema::ANYTHING {
        return (claimed_by_others != Schema::Nothing).then(|| MapClause {
            key: claimed_by_others,
            value: clause.value.clone(),
        });
    }
    let Schema::Union(parts) = &clause.key else {
        return Some(clause.clone());
    };
    for (at, part) in parts.iter().enumerate() {
        let Schema::Complement(region) = part else {
            continue;
        };
        let claimed = || {
            parts
                .iter()
                .enumerate()
                .filter(move |(other, _)| *other != at)
                .map(|(_, kept)| kept.clone())
        };
        if **region == Schema::union(claimed().chain(others())) {
            return Some(MapClause {
                key: Schema::union(claimed()),
                value: clause.value.clone(),
            });
        }
    }
    Some(clause.clone())
}

/// The clause list an openness gives a node that already carries `defaults`.
///
/// **Openness is the default of the region no clause claims** (ICFP Definition
/// 4.6): a clause is a key-type region with its own default, and these two
/// operators are about the region the clauses leave over. Closing sends it to
/// bottom -- the clause that frees it goes -- and opening sends it to the top,
/// which is one clause admitting every value at every key the others do not
/// claim. The claimed regions are untouched either way, so the paper's own
/// closed `{|"a" = Int, String => Bool|}` keeps its `String => Bool`. The
/// clauses admitting every value are read as the one clause they are first
/// ([`tops_merged`]), so the region does not turn on how they are split.
fn clauses_over(defaults: &[MapClause], open: Openness) -> Clauses {
    // A node with no clauses claims no key, so the unclaimed region is the
    // whole key space and the answer is the shared list either way: the top
    // clause, or none. That is a record, which is most of what is opened, and
    // computing the region for it is a complement of an empty union.
    if defaults.is_empty() {
        return clauses_for(open);
    }
    let merged;
    let defaults = if defaults
        .iter()
        .filter(|clause| clause.value == Schema::ANYTHING)
        .nth(1)
        .is_some()
    {
        merged = tops_merged(defaults);
        &merged[..]
    } else {
        defaults
    };
    let mut kept: Vec<MapClause> = defaults
        .iter()
        .enumerate()
        .filter_map(|(at, clause)| claimed_beside_the_region(clause, defaults, at))
        .collect();
    if open == Openness::Open {
        let region = unclaimed_region(&kept);
        // A clause list already claiming every key leaves nothing to free, and
        // an empty region would be a rectangle over no key at all.
        if region != Schema::Nothing {
            kept.push(MapClause {
                key: region,
                value: Schema::ANYTHING,
            });
        }
        merge_by_value(&mut kept);
    }
    // In the constructor's order, since two orders of one clause list are one
    // schema: the freed clause folds wherever its value first appears, and an
    // opened map is equal to its own clauses written out again.
    canonical_clauses(&mut kept);
    share_clauses(&mut kept)
}

/// The clauses admitting every value, read as one clause over the union of
/// their keys, the others as they stand.
///
/// Two clauses each admitting every value at their own keys are one rectangle
/// over the join of the keys, and the region `open` frees is such a clause, so
/// which of them is the region is not a question the pair can answer apart:
/// `{str: anything, complement(str): anything}` is the top clause written
/// twice, ICFP 2023's wildcard (l.1509-1511) spelled as the union footnote 15
/// reads one as. Read one at a time, each would be tested against the
/// complement of the others' keys, and the answer would turn on whether that
/// complement folds: the pair alone closes to `{}` as the top does, and beside
/// a `bytes: int` clause, or split over `complement(int)` and
/// `complement(str)`, it would keep every clause and admit every dict. Merged,
/// the clause is read as the top is -- the part over the keys the others claim
/// stays, and the rest is the region -- however it is split.
fn tops_merged(defaults: &[MapClause]) -> Vec<MapClause> {
    let mut merged: Vec<MapClause> = Vec::with_capacity(defaults.len());
    let mut top: Option<usize> = None;
    for clause in defaults {
        if clause.value != Schema::ANYTHING {
            merged.push(clause.clone());
            continue;
        }
        if let Some(kept) = top.and_then(|at| merged.get_mut(at)) {
            kept.key = Schema::union([kept.key.clone(), clause.key.clone()]);
        } else {
            top = Some(merged.len());
            merged.push(clause.clone());
        }
    }
    merged
}

/// Fold clauses carrying one value into one clause over the union of their keys.
///
/// The clauses denote a union of key-by-value rectangles, so two sharing a value
/// are one rectangle over the join of their keys -- the same set, one clause
/// shorter. Read after the unclaimed region is freed, because that is where the
/// pair arises: a record declaring `str => anything` for the keys it does not
/// name is what a `TypedDict` builds, and freeing the key-types beside it gives
/// two clauses carrying `anything` between them.
///
/// Leaving them apart costs a pair its answer where the region claimed holds a
/// literal: the region freed is then the literal's complement, a key holding
/// part of a kind, which the set representation declines
/// (`docs/15-decidability.md`), so the same set written the long way stops
/// being decided. Over whole kinds the long way is decided too, and the fold
/// is the shorter term. Where the values differ there is nothing to fold and
/// the complement stands, which is the mapping case the page records.
fn merge_by_value(clauses: &mut Vec<MapClause>) {
    let mut folded: Vec<MapClause> = Vec::with_capacity(clauses.len());
    for clause in clauses.drain(..) {
        // First occurrence keeps the position; the caller puts the list in
        // the constructor's order afterwards, as every spelling of it is.
        match folded.iter_mut().find(|kept| kept.value == clause.value) {
            Some(kept) => kept.key = Schema::union([kept.key.clone(), clause.key]),
            None => folded.push(clause),
        }
    }
    *clauses = folded;
}

/// A field list, shared when it is empty and owned when it is not.
fn share_fields(fields: &mut Vec<Field>) -> Fields {
    if fields.is_empty() {
        no_fields()
    } else {
        intern::fields(fields)
    }
}

/// An attribute list, shared by the same rule as [`share_fields`].
fn share_attributes(attributes: &mut Vec<Attribute>) -> Arc<[Attribute]> {
    intern::attributes(attributes)
}

/// A clause list, shared when it is one the whole tree already holds.
fn share_clauses(defaults: &mut Vec<MapClause>) -> Clauses {
    if defaults.is_empty() {
        no_clauses()
    } else if defaults.len() == 1 && defaults.first() == Some(&MapClause::top()) {
        open_clauses()
    } else {
        intern::clauses(defaults)
    }
}

/// A member list, shared when it is empty and interned when it is not.
///
/// The counterpart of [`share_fields`] for the list a join, a meet or a fixed
/// sequence prefix carries.
pub(crate) fn share_members(members: &[Schema]) -> Members {
    if members.is_empty() {
        no_members()
    } else {
        intern::members(members)
    }
}

/// A child node, shared: the counterpart of [`share_members`] for the single
/// schema a complement, a collection, a sequence tail or a refinement holds.
pub(crate) fn share_node(schema: Schema) -> Arc<Schema> {
    intern::node(schema)
}

/// Run `build` with a scratch field buffer, returning it to the pool after.
///
/// The same pooling as [`with_member_buffer`], for the list a record rebuild
/// assembles. A pass over a wide record filters and maps its fields into a
/// buffer and then copies the result into the shared slice the node holds; the
/// buffer is dead as soon as the node exists, so it is borrowed rather than
/// allocated.
fn with_field_buffer<R>(build: impl FnOnce(&mut Vec<Field>) -> R) -> R {
    thread_local! {
        static POOL: RefCell<Vec<Vec<Field>>> = const { RefCell::new(Vec::new()) };
    }
    let mut buffer = POOL
        .with(|pool| pool.borrow_mut().pop())
        .unwrap_or_default();
    buffer.clear();
    let made = build(&mut buffer);
    buffer.clear();
    POOL.with(|pool| pool.borrow_mut().push(buffer));
    made
}

/// Run `build` with a scratch member buffer, returning it to the pool after.
///
/// A join or a meet is assembled in a growable buffer -- members flattened in,
/// then ordered and deduplicated -- and the finished list is copied into the
/// shared slice the node holds. Allocating that buffer per construction is the
/// price of the shared slice, and it is a price nothing collects: the buffer is
/// dead the moment the node exists. Pooling it per thread makes the buffer free
/// after the first use and leaves one allocation per node, which is what an
/// owned `Vec` would cost.
///
/// The pool is a stack rather than a single spare because construction nests
/// deeply -- a member is built while the join holding it is being assembled --
/// and a single spare leaves every inner call allocating its own.
pub(crate) fn with_member_buffer<R>(build: impl FnOnce(&mut Vec<Schema>) -> R) -> R {
    thread_local! {
        static POOL: RefCell<Vec<Vec<Schema>>> = const { RefCell::new(Vec::new()) };
    }
    let mut buffer = POOL
        .with(|pool| pool.borrow_mut().pop())
        .unwrap_or_default();
    buffer.clear();
    let made = build(&mut buffer);
    buffer.clear();
    POOL.with(|pool| pool.borrow_mut().push(buffer));
    made
}

/// The members of a join or a meet.
///
/// A shared, immutable slice rather than a `Vec`: the members are written once
/// by the constructor that normalises them and read for the rest of the node's
/// life, so what the representation has to be cheap at is *carrying* the list,
/// not editing it.
pub type Members = Arc<[Schema]>;

/// A record's declared fields, shared for the same reason as [`Members`].
pub type Fields = Arc<[Field]>;

/// A map's default clauses, shared for the same reason as [`Members`].
pub type Clauses = Arc<[MapClause]>;

/// A refinement's constraints, shared for the same reason as [`Members`].
pub type Constraints = Arc<[Constraint]>;

/// The schema intermediate representation.
///
/// Each variant documents its denotation: the set of Python values it accepts.
/// `Ord`/`Eq` are structural; the constructors use them to put union and
/// intersection members in one order and to drop a repeated one.
///
/// Adding a variant means handling it in every walk over the IR. The compiler
/// forces an arm in the matches that name every variant:
/// - core: the two statements of a node's children, `mapped_children` for a
///   walk that rebuilds and `push_children` for one that only reads, which
///   every structural walk goes through (`map_children`, [`Schema::measure`],
///   [`Schema::node_count`], `has_reference` among them);
///   `remapped_where_moved`, which moves the pooled and definitions indices;
///   [`Schema::expected`], [`Schema::error_code`] and `guards_children`, which
///   [`Schema::occurs_unguarded_under`] reads; the interner's `hash_node` and `tag`;
///   and `descend`, the lowering to the set representation;
/// - bindings (`valgebra-py`): the `member` membership walk, which decides
///   membership and in explain mode aggregates the violation, and `render`.
///
/// Every other match over the IR ends in a wildcard and compiles unchanged: the
/// interner's `same_node` reads a new variant as never shared, and the decision
/// procedures read it through their wildcard arms. `tests/test_node_matrix.py`
/// and `tests/test_closure_ledger.py` read the variants out of this enum and
/// fail on one without a row, which is what makes a new variant visible there.
///
/// The compiler forces an arm; it cannot check the arm recursed into every child.
/// That is why the child set is stated in the two child walks rather than per
/// walk, and `the_functor_and_the_traversal_describe_the_same_children` holds
/// the two to one set.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Schema {
    /// Top. Denotes every Python value; membership always holds. The payload
    /// records how the top was written, and is read by `repr` to print it and
    /// by the sharing table to keep the two apart ([`Spelling`]).
    Anything(Spelling),
    /// Bottom. Denotes the empty set; membership never holds.
    Nothing,
    /// Denotes the singleton set `{None}`.
    NoneType,
    /// Denotes `{True, False}`, exactly the `bool` instances.
    ///
    /// Because `bool` is a subclass of `int`, this set is a subset of
    /// [`Schema::Int`]: `Bool` is a subtype of `Int`.
    Bool,
    /// Denotes every value whose type is `int` or a subclass of it:
    /// `issubclass(type(x), int)`. The type is the real one, not what a
    /// `__class__` property answers, so a proxy `isinstance` admits is not a
    /// member (`docs/14-soundness.md`).
    ///
    /// In Python `bool` is a subclass of `int`, so `True` and `False` are
    /// integers and are members of this set. No value is carved out: subtyping
    /// is subset inclusion, so [`Schema::Bool`] is a subtype of `Int` rather
    /// than disjoint from it.
    Int,
    /// Denotes every value whose type is `float` or a subclass of it:
    /// `issubclass(type(x), float)`, read as [`Schema::Int`] reads it.
    ///
    /// `int` does not subclass `float`, so `Int` and `Float` are disjoint and
    /// an integer is not a member.
    Float,
    /// Denotes every value whose type is `str` or a subclass of it, read as
    /// [`Schema::Int`] reads it.
    Str,
    /// Denotes every value whose type is `bytes` or a subclass of it, read as
    /// [`Schema::Int`] reads it.
    Bytes,
    /// Denotes the values of `c`'s exact type that equal it, for a fixed
    /// constant `c`: a value is a member iff it has the *same type* as `c` and is
    /// equal to it. That is the typed singleton `{c}` for a constant whose `==`
    /// is Python's usual one, and the empty set for `nan`, which equals nothing.
    /// Same-type is what makes this a singleton — Python's `==` conflates across
    /// types (`1 == True == 1.0`), so equality alone would make `Literal[1]`
    /// also admit `True` and `1.0`. Requiring `type(x) is type(c)` keeps the
    /// typing spec's distinction between `Literal[1]`, `Literal[True]`, and
    /// `Literal[1.0]`.
    ///
    /// A set under the trust base, which takes a value's `==` to answer one way
    /// (`docs/14-soundness.md`): a value whose `==` answers two calls
    /// differently is outside it, and `Literal[c] & ~Literal[c]` is `Nothing`
    /// whatever that value answers.
    ///
    /// The constant itself is not stored here — the core stays free of Python
    /// objects. The payload is an index into a constants pool held alongside the
    /// compiled validator. The same-type test is applied in the bindings, where
    /// the Python value is in hand.
    Literal(ConstIx),
    /// Denotes lists or tuples whose elements take a [`SeqShape`].
    ///
    /// One node subsumes the homogeneous `list[T]`/`tuple[T, ...]`, the fixed
    /// `[A, B]`/`tuple[A, B]`, and the prefix-plus-tail forms: one shape to walk
    /// and one to relate, rather than four ad-hoc nodes.
    ///
    /// A sequence composes in the Boolean algebra the way every other node does,
    /// through [`Schema::Union`], [`Schema::Intersection`] and
    /// [`Schema::Complement`] over the sequence node. The shape itself is *not*
    /// closed under those operations here -- it has no complement or intersection
    /// constructor -- so the closure is at the schema level and the regular
    /// languages the shape is drawn from are the model, not a computation this
    /// crate performs.
    Seq {
        /// Whether the value is a list or a tuple.
        container: SeqKind,
        /// The prefix and optional tail the value's elements must take.
        shape: SeqShape,
    },
    /// Denotes sets or frozensets whose every element belongs to `element`.
    ///
    /// One node with a kind, as [`Schema::Seq`] is: the two containers hold
    /// their members the same way and differ only in which class a value has, so
    /// two variants meant every walk over the node set carried the pair
    /// together and every arm had to remember to name both.
    Coll {
        /// Whether the value is a set or a frozenset.
        container: CollKind,
        /// The set every member of the value must belong to.
        element: Arc<Schema>,
    },
    /// Denotes dicts with named fields and key-schema-keyed defaults for the
    /// rest.
    ///
    /// A dict is a member iff every required field's key is present with a
    /// matching value, every present optional field's value matches, and every
    /// key that is *not* a declared field name is covered by some default
    /// clause — a `(key-schema, value-schema)` pair the key and its value both
    /// satisfy. Named fields take precedence over the defaults.
    ///
    /// One node subsumes the record, the homogeneous mapping, the heterogeneous
    /// mapping, and their combination: a closed record has no default clause, an
    /// open record a single `(Anything, Anything)` clause, `dict[K, V]` a
    /// single `(K, V)` clause with no fields, and a typed catch-all a record's
    /// fields plus a typed clause. The empty closed map denotes only the empty
    /// dict.
    KeyedMap {
        /// The declared string-named fields, in order.
        fields: Fields,
        /// `(key-schema, value-schema)` clauses governing every key that is not
        /// a declared field name. A key belongs when **some** clause admits both
        /// it and its value, so the clauses are a disjunction and not a
        /// precedence list: the order is the order they are rendered in, and
        /// carries no meaning to membership or to subtyping. Both consumers ask
        /// `any`; a clause list read as ordered is a semantics no code here
        /// implements.
        defaults: Clauses,
    },
    /// Denotes the union of the member sets: a value is a member iff it belongs
    /// to at least one member schema.
    Union(Members),
    /// Denotes the intersection of the member sets: a value is a member iff it
    /// belongs to every member schema.
    Intersection(Members),
    /// Denotes the complement of the inner set: a value is a member iff it is
    /// not a member of the inner schema.
    Complement(Arc<Schema>),
    /// Denotes instances of a class, by `isinstance`. The class is held in the
    /// validator's object pool; the payload is its index.
    Instance(ClassIx),
    /// Denotes every value whose attributes satisfy the given ones: each named
    /// attribute must be present on the value and belong to its set.
    ///
    /// Every attribute is required, and the type says so: an [`Attribute`]
    /// carries no presence flag. An optional one would denote `AttrRecord{x: S}
    /// | ~AttrRecord{x: anything}`, which the other nodes reach, so it is no
    /// generator and no normal form names it. No frontend path would build one
    /// -- a protocol member and a dataclass field are both present on an
    /// instance -- and a state only a test can build is one the walk, the
    /// lowering and the rules can read three ways unseen
    /// (`docs/dev/01-schema-ir.md`).
    ///
    /// An attribute is read as the value's own, so the record is a set under the
    /// trust base, which takes what a value answers to be a fact about it
    /// (`docs/14-soundness.md`). Construction folds it with its complement on
    /// that ground, and a property whose reads alternate is outside the sets
    /// the fold is about.
    ///
    /// The record carries **no class**. A class is a separate narrowing, and a
    /// dataclass is the meet of the two: `build_object` emits
    /// `Instance(C) ∧ AttrRecord`, which is what the form has always meant.
    /// Holding them in one node made the pair inseparable, so the algebra could
    /// neither relate a class to a record nor complement one without the other;
    /// split, each half is a set the rules already know.
    AttrRecord {
        /// The attributes, by name.
        fields: Arc<[Attribute]>,
    },
    /// Denotes the subset of the base set satisfying every constraint
    /// (`{ x in [[base]] | all constraints hold }`). The base is checked first.
    Refine {
        /// The base schema; a value must belong to it before constraints apply.
        base: Arc<Schema>,
        /// Constraints that further narrow the base set, checked in order.
        constraints: Constraints,
    },
    /// A reference to a recursive definition: denotes the same set as the
    /// definition at this index in the validator's definitions table. The back
    /// edge of a fixpoint, produced by `recursive`.
    Ref(DefIx),
    /// Denotes the **empty set**: the walk admits no value against one, and
    /// every decider reads it as a schema it cannot relate.
    ///
    /// The marker a `recursive` body carries where its own back edge will go,
    /// resolved to a [`Schema::Ref`] before the validator is returned. A
    /// finished schema holds none, so the denotation is what a value meets only
    /// where that resolution did not happen -- which is a build this refuses
    /// rather than a set a caller can write.
    SelfRef(u64),
}

/// Whether a [`Schema::Coll`] denotes sets or frozensets.
///
/// The container is part of the value: a `set` is never a `frozenset`, so the
/// two are disjoint and this is what says which one a schema names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CollKind {
    /// `set` values.
    Set,
    /// `frozenset` values.
    FrozenSet,
}

/// Whether a [`Schema::Seq`] denotes lists or tuples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SeqKind {
    /// `list` values.
    List,
    /// `tuple` values.
    Tuple,
}

/// The element shape of a [`Schema::Seq`]: a fixed prefix, then an optional
/// repeated tail.
///
/// A value's element sequence is a member iff its first `prefix.len()` elements
/// belong to the prefix schemas positionally and every element past them belongs
/// to `tail` -- with no element past the prefix at all when there is no tail.
/// The three forms a caller can spell are the three this shape takes:
/// homogeneous (`list[T]`) is an empty prefix and a tail, fixed (`tuple[A, B]`)
/// is a prefix and no tail, and prefix-plus-tail (`tuple[A, *tuple[B, ...]]`) is
/// both.
///
/// This is a *linear* language rather than a regular one, and deliberately so.
/// The general form is a regular expression over element schemas, as
/// Hosoya-Vouillon-Pierce give it, and deciding inclusion between two of those
/// wants the automaton construction `docs/15-decidability.md` records as
/// unbuilt. Nothing in this crate or the bindings builds an alternation or a
/// nested repetition, so the general form would make every walk answer for
/// shapes no value reaches, and every membership check prove first that the
/// shape it holds is one of the three. This shape is the only one there is.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct SeqShape {
    /// The positional element schemas, matched in order from the front.
    pub prefix: Members,
    /// The schema every element past the prefix must belong to, or `None` when
    /// the sequence ends at the prefix.
    pub tail: Option<Arc<Schema>>,
}

impl SeqShape {
    /// Every element schema this shape holds: the prefix in order, then the tail.
    ///
    /// The sequence half of [`Schema::children`]: a measure over a sequence reads
    /// its elements from here rather than restating where a sequence keeps one.
    pub(crate) fn elements(&self) -> impl Iterator<Item = &Schema> {
        self.prefix.iter().chain(self.tail.as_deref())
    }
}

impl Schema {
    /// A list whose elements take `shape`.
    #[must_use]
    pub fn list(shape: SeqShape) -> Schema {
        Schema::Seq {
            container: SeqKind::List,
            shape,
        }
    }

    /// A tuple whose elements take `shape`.
    #[must_use]
    pub fn tuple(shape: SeqShape) -> Schema {
        Schema::Seq {
            container: SeqKind::Tuple,
            shape,
        }
    }

    /// A set whose every member belongs to `element`.
    #[must_use]
    pub fn set(element: Schema) -> Schema {
        Schema::Coll {
            container: CollKind::Set,
            element: share_node(element),
        }
    }

    /// A frozenset whose every member belongs to `element`.
    #[must_use]
    pub fn frozen_set(element: Schema) -> Schema {
        Schema::Coll {
            container: CollKind::FrozenSet,
            element: share_node(element),
        }
    }

    /// A homogeneous mapping `dict[K, V]`: every key in the clause's `key`,
    /// every value in its `value`.
    #[must_use]
    pub fn mapping(clause: MapClause) -> Schema {
        Schema::KeyedMap {
            fields: no_fields(),
            defaults: share_clauses(&mut vec![clause]),
        }
    }

    /// The union of `members`, in the lattice normal form.
    ///
    /// A union of no members is the bottom, which is this operation's identity.
    /// The constructor owns that, so a caller folding an empty collection gets an
    /// atom every consumer already reads rather than a node each must special-case
    /// -- the render printed the empty join of no members as an empty string.
    ///
    /// The normal form is what the associativity, commutativity, idempotence and
    /// identity laws say a join is: a nested union is flattened into this one,
    /// the bottom drops out, the top absorbs, the members are ordered and
    /// deduplicated, and a single member is returned unwrapped. So
    /// `union(int, int)` is `int` and `union(str, int)` is the schema
    /// `union(int, str)` is, which is what makes `==` equality of the set's one
    /// spelling rather than of whichever the caller happened to write.
    ///
    /// Ordering is by [`Ord`], which is the variant order and then the payload:
    /// a literal sorts by its pool slot, so a `Literal[...]` keeps the order it
    /// was written in and the message a failed union reports lists its branches
    /// that way.
    ///
    /// **Absorption is not applied.** `A | (A & B)` and `A` are one set and stay
    /// two terms. Absorption is one case of `A | B` folding to `A` wherever `A`
    /// contains `B`; that fold needs a containment, and containment is the
    /// decision procedure -- running it wherever a schema is built is the cost
    /// this design refuses everywhere else. A rule downstream may assume the form
    /// below and no more.
    #[must_use]
    pub fn union(members: impl IntoIterator<Item = Schema>) -> Schema {
        Schema::union_within(members, &[])
    }

    /// The join, with the definitions its members may refer into.
    ///
    /// Only the complement law reads them, and only for a recursive member: a
    /// `Ref` is not a set on its own evidence, so without the definitions
    /// `json | ~json` is left standing while `int | ~int` folds to the top. A
    /// caller that holds the definitions -- which is every caller that combines
    /// two compiled validators -- passes them and gets the law for both.
    ///
    /// **The table is required to be contractive**, because the law is about
    /// sets and a non-contractive definition names none: `t = ~t` is satisfied
    /// by no set, and folding `t | ~t` over it applies a law to nothing. Every
    /// caller in the bindings has run `occurs_unguarded_under` on the body
    /// before a `Ref` into it exists; a caller assembling a table by hand owes
    /// that check. `oracle::denotes_a_set_within` is where a reference is read
    /// and where the precondition bites.
    #[must_use]
    pub fn union_within(
        members: impl IntoIterator<Item = Schema>,
        definitions: &[Schema],
    ) -> Schema {
        with_member_buffer(|flat| {
            for member in members {
                match member {
                    // The top absorbs, and keeps the spelling of the member it
                    // absorbed: `Any | int` renders `Any`.
                    top @ Schema::Anything(_) => return top,
                    Schema::Nothing => {}
                    Schema::Union(inner) => flat.extend(inner.iter().cloned()),
                    other => flat.push(other),
                }
            }
            flat.sort();
            flat.dedup();
            // A join carrying a schema together with its complement is the top,
            // whatever those are, so no such join survives construction and no rule
            // downstream may assume one does -- a union flattened into the list
            // included, whose members stand where it did. `has_complementary_pair_within`
            // is the one statement of the law, read here and by the emptiness
            // decision from `oracle.rs` -- which is below both, because a lattice
            // law belongs to neither of them. It searches the sorted list only for
            // a member that is itself a complement.
            // With no oracle here the law declines for an atom only the bindings can
            // read, which is the safe direction: a join left unfolded still denotes
            // what it denotes. `definitions` is what a *recursive* member needs: a
            // reference is not a set on its own evidence, so without them the law
            // declines for every fixpoint, `json | ~json` included.
            let members: &[Schema] = flat;
            if has_complementary_pair_within(
                members,
                Connective::Join,
                &NoLeafRelations,
                definitions,
                |wanted| members.binary_search(wanted).is_ok(),
            ) {
                return Schema::ANYTHING;
            }
            match flat.len() {
                0 => Schema::Nothing,
                1 => flat.swap_remove(0),
                _ => Schema::Union(share_members(flat)),
            }
        })
    }

    /// The meet of `members`, in the lattice normal form dual to
    /// [`union`](Self::union).
    ///
    /// A meet of no members is the top. A nested meet is flattened, the top drops
    /// out, the bottom absorbs, and the members are ordered and deduplicated, so
    /// `intersection(int, anything)` is `int` and `intersection(int, int)` is
    /// `int`.
    #[must_use]
    pub fn meet(members: impl IntoIterator<Item = Schema>) -> Schema {
        Schema::meet_within(members, &[])
    }

    /// The meet, with the definitions its members may refer into. The dual of
    /// [`union_within`](Self::union_within), and for the same reason -- and
    /// under the same precondition, that the table is contractive.
    #[must_use]
    pub fn meet_within(
        members: impl IntoIterator<Item = Schema>,
        definitions: &[Schema],
    ) -> Schema {
        with_member_buffer(|flat| {
            for member in members {
                match member {
                    Schema::Nothing => return Schema::Nothing,
                    Schema::Anything(_) => {}
                    Schema::Intersection(inner) => flat.extend(inner.iter().cloned()),
                    other => flat.push(other),
                }
            }
            flat.sort();
            flat.dedup();
            // The dual of the fold in [`union`](Self::union), and the same law read
            // the other way: a meet carrying a schema together with its complement
            // is the bottom. Both constructors state it or neither does -- a law
            // folded on one side and left standing on the other is two answers to
            // one question.
            let members: &[Schema] = flat;
            if has_complementary_pair_within(
                members,
                Connective::Meet,
                &NoLeafRelations,
                definitions,
                |wanted| members.binary_search(wanted).is_ok(),
            ) {
                return Schema::Nothing;
            }
            match flat.len() {
                0 => Schema::ANYTHING,
                1 => flat.swap_remove(0),
                _ => Schema::Intersection(share_members(flat)),
            }
        })
    }

    /// The complement: every value this schema does not admit.
    ///
    /// Named for the operation rather than spelled as `!`, matching `Region` and
    /// the Python surface: a one-character operator in a fold is a one-character
    /// defect, and typing has no operator for this one.
    ///
    /// `~~A` is `A`, so a complement of a complement cancels. Nothing downstream
    /// carries a double negation, and no rule anywhere may assume one exists.
    ///
    /// The two bounds cancel with it: `~anything` is `nothing` and `~nothing` is
    /// `anything`. Without those, dropping the top from a meet -- which the
    /// identity law does -- would take the complementary pair apart before the
    /// pair rule saw it, and `Any & ~Any` would come out as `~Any` rather than
    /// the bottom. The spelling is not carried across, because neither bound is
    /// the other's spelling of itself.
    #[must_use]
    pub fn complement(self) -> Schema {
        match self {
            Schema::Complement(inner) => Arc::unwrap_or_clone(inner),
            Schema::Anything(_) => Schema::Nothing,
            Schema::Nothing => Schema::ANYTHING,
            other => Schema::Complement(share_node(other)),
        }
    }

    /// A record of named fields. An open record admits any other key with any
    /// value; a closed one admits none.
    #[must_use]
    pub fn record(fields: Vec<Field>, open: Openness) -> Schema {
        Schema::keyed_map_within(fields, clauses_for(open))
    }

    /// A dict node: named fields, and the catch-all clauses governing every other
    /// key.
    ///
    /// The constructor the other two are written in terms of, so a caller never
    /// builds the variant raw. `record` and `mapping` are the two shapes the
    /// frontend spells; this is the general one they are special cases of, and a
    /// mixed record-and-catch-all needs it.
    #[must_use]
    pub fn keyed_map(mut fields: Vec<Field>, mut defaults: Vec<MapClause>) -> Schema {
        canonical_clauses(&mut defaults);
        canonical_fields(&mut fields);
        Schema::KeyedMap {
            fields: share_fields(&mut fields),
            defaults: share_clauses(&mut defaults),
        }
    }

    /// The same node with its clause list already shared and canonical.
    ///
    /// The two clause lists a record can carry -- the open catch-all and the
    /// empty one -- are the same lists every time, so a caller that knows which
    /// it wants passes the shared one rather than building a list for this node
    /// alone.
    #[must_use]
    pub(crate) fn keyed_map_within(mut fields: Vec<Field>, defaults: Clauses) -> Schema {
        canonical_fields(&mut fields);
        Schema::KeyedMap {
            fields: share_fields(&mut fields),
            defaults,
        }
    }

    /// The same, from a buffer the caller keeps rather than owns, so a rebuild
    /// that borrowed its buffer from the pool can hand it back. What the buffer
    /// holds afterwards is the sharing's business: a list the table already
    /// holds leaves the fields where they are and drops them, and one it does
    /// not takes them.
    #[must_use]
    // A `Vec` rather than a slice because interning drains the fields into the
    // shared list when the table does not hold them, which a slice cannot give.
    fn keyed_map_from(fields: &mut Vec<Field>, defaults: Clauses) -> Schema {
        canonical_fields(fields);
        Schema::KeyedMap {
            fields: share_fields(fields),
            defaults,
        }
    }

    /// An object node: the attributes an instance must carry, in canonical order.
    ///
    /// The record half of a class schema, met with the `Instance` atom that
    /// names the class. Named attributes, ordered by name as a dict record's
    /// fields are, so one record written in two orders is one schema.
    #[must_use]
    pub fn attr_record(mut fields: Vec<Attribute>) -> Schema {
        fields.sort_by(|a, b| a.name.cmp(&b.name));
        Schema::AttrRecord {
            fields: share_attributes(&mut fields),
        }
    }

    /// A refinement node: `base` narrowed by `constraints`, in canonical order.
    ///
    /// With no constraint there is nothing to narrow, so the base is returned as
    /// it stands rather than wrapped in a node that says nothing -- which is the
    /// identity law, settled where the schema is built like the others.
    #[must_use]
    pub fn refine(base: Schema, mut constraints: Vec<Constraint>) -> Schema {
        constraints.sort();
        constraints.dedup();
        if constraints.is_empty() {
            return base;
        }
        Schema::Refine {
            base: share_node(base),
            constraints: constraints.into(),
        }
    }
}

/// Put a record's fields in the order two spellings of one record agree on.
///
/// A record's fields are a *map*: `{"a": int, "b": str}` and `{"b": str, "a":
/// int}` admit exactly the same dicts. Unordered they are two terms -- `==`
/// says so, `hash` says so, and a union of the pair keeps both members, since
/// the dedup that folds a repeated member compares terms. Ordering by name is
/// what keeps the written order out of the schema.
///
/// The name alone is the key: a record cannot declare one name twice (the
/// frontend refuses it, and a dict literal cannot express it), so the order is
/// total and the sort is stable in the only way it needs to be.
fn canonical_fields(fields: &mut [Field]) {
    fields.sort_by(|a, b| a.name.cmp(&b.name));
}

/// Put a map's catch-all clauses in the order two spellings agree on.
///
/// The clauses are unordered here -- a deliberate narrowing of the model, which
/// orders them -- so `{str: int, int: bool}` and `{int: bool, str: int}` govern
/// exactly the same dicts and are one schema.
fn canonical_clauses(defaults: &mut Vec<MapClause>) {
    defaults.sort();
    defaults.dedup();
}

impl SeqShape {
    /// The homogeneous form: any number of elements, each in `element`.
    #[must_use]
    pub fn homogeneous(element: Schema) -> SeqShape {
        SeqShape {
            prefix: no_members(),
            tail: Some(share_node(element)),
        }
    }

    /// The fixed form: each element positionally, and no element past them.
    #[must_use]
    pub fn fixed(elements: impl IntoIterator<Item = Schema>) -> SeqShape {
        SeqShape {
            prefix: share_members(&elements.into_iter().collect::<Vec<_>>()),
            tail: None,
        }
    }

    /// The prefix-plus-tail form: a fixed positional prefix, then zero or more
    /// elements in `tail`.
    #[must_use]
    pub fn prefix_tail(prefix: impl IntoIterator<Item = Schema>, tail: Schema) -> SeqShape {
        SeqShape {
            prefix: share_members(&prefix.into_iter().collect::<Vec<_>>()),
            tail: Some(share_node(tail)),
        }
    }
}

/// A constraint narrowing a [`Schema::Refine`] base set.
///
/// Comparison and predicate operands live in the validator's object pool; the
/// payload is an index. Length bounds carry the length directly.
///
/// Every constraint but [`Constraint::Predicate`] is a set under the trust base
/// (`docs/14-soundness.md`). The comparison, the length and the `%` it reads
/// are operators Python's rules make facts about the value -- the language
/// reference's rules for comparisons, the data model's `__len__` -- with
/// not-a-number the exception they name, read as one. So construction folds a
/// refinement with its complement, and a value whose operator answers two
/// calls differently is outside the sets that fold is about. A predicate is
/// code Python asks nothing of, and `denotes_a_set_within` declines it.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Constraint {
    /// `value >= pool[i]`.
    Ge(OperandIx),
    /// `value > pool[i]`.
    Gt(OperandIx),
    /// `value <= pool[i]`.
    Le(OperandIx),
    /// `value < pool[i]`.
    Lt(OperandIx),
    /// The value holds at least `n` items: the count a builtin container
    /// stores, and `len(value)` for any other value.
    MinLen(usize),
    /// The value holds at most `n` items, counted as [`Constraint::MinLen`]
    /// counts them.
    MaxLen(usize),
    /// `value % pool[i] == 0`: a numeric multiple of the operand.
    MultipleOf(OperandIx),
    /// `pool[i](value)` is truthy. The documented Python-callback slow path.
    Predicate(PredIx),
    /// The string fully matches this regular expression, anchored at both ends
    /// as `re.fullmatch` anchors, in the Rust engine's dialect rather than `re`'s
    /// (`docs/05-refinements.md`). A value that is not a `str` has no text to
    /// match and is outside the set, a `bytes` value among them, as is a string
    /// holding a lone surrogate (`docs/14-soundness.md`). The pattern is held
    /// inline rather than pooled; the bindings compile it once and match
    /// natively. The structural rules relate two patterns only when they are
    /// written identically; the set representation reads a pattern as the
    /// regular language over `str` it denotes (`descr/regular.rs`), so
    /// `Regex("a+")` is below `Regex("a*")`.
    Regex(String),
}

/// The key and value schemas of one catch-all clause of a [`Schema::KeyedMap`].
///
/// Two positional `Schema`s are the shape a type cannot distinguish: a caller
/// writing `dict[K, V]` with them transposed builds `dict[V, K]`, which
/// typechecks and validates real values. Naming them cannot be transposed, which
/// is the only remedy that applies here -- a key schema and a value schema are
/// genuinely two schemas, and neither carries which position it is in.
///
/// The IR stores these rather than pairs. A `(Schema, Schema)` in the node put
/// the hazard back at every site that read one: each `|(k, v)|` closure is a
/// place the two could be bound the wrong way round, and there is no arity or
/// type to catch it.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MapClause {
    /// Every key the clause governs must belong to this set.
    pub key: Schema,
    /// Every value under such a key must belong to this set.
    pub value: Schema,
}

impl MapClause {
    /// The clause admitting every key with every value: what an open record's
    /// catch-all is.
    #[must_use]
    pub fn top() -> MapClause {
        MapClause {
            key: Schema::ANYTHING,
            value: Schema::ANYTHING,
        }
    }

    /// A clause from a key and a value schema.
    ///
    /// Spelled here rather than at each pass over a clause list, for the reason
    /// [`Field::with_schema`] is: the two schemas a clause holds are in a fixed
    /// order and a pass that rebuilds one from the wrong parts swaps what a key
    /// must be with what its value must be.
    pub(crate) fn of(key: Schema, value: Schema) -> MapClause {
        MapClause { key, value }
    }
}

/// Whether a record's own clauses already say what this field says.
///
/// The paper's `dom` read as a set rather than as a syntax: a field that
/// constrains its key to exactly what the record gives every key it does not
/// name adds nothing, and two records differing only in such a field are one
/// record. A closed record says every unnamed key is absent, which is what an
/// optional field admitting nothing says; an open one says every unnamed key is
/// free, which is what an optional field admitting everything says.
///
/// A *required* field is never covered: it says the key must be there, and no
/// clause says that of a key it does not name.
fn already_said(field: &Field, defaults: &[MapClause]) -> bool {
    if field.required {
        return false;
    }
    match defaults {
        [] => field.schema == Schema::Nothing,
        [clause] if *clause == MapClause::top() => field.schema == Schema::ANYTHING,
        _ => false,
    }
}

/// A named attribute of a [`Schema::AttrRecord`]: present on the value, and in
/// its set. There is no optional attribute, so there is no flag for one.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Attribute {
    /// The attribute's name, shared for the reason a field's is
    /// ([`Field::name`]).
    pub name: Arc<str>,
    /// The set the attribute's value must belong to.
    pub schema: Schema,
}

impl Attribute {
    /// The same attribute with its schema replaced, the name carried across.
    pub(crate) fn with_schema(&self, schema: Schema) -> Attribute {
        Attribute {
            name: Arc::clone(&self.name),
            schema,
        }
    }
}

/// A named field of a [`Schema::KeyedMap`].
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Field {
    /// The key name, shared rather than owned outright.
    ///
    /// A field name is written once, when the frontend reads the annotation,
    /// and never edited; every pass over a schema afterwards rebuilds the tree
    /// and carries the name across unchanged. Owning it outright would make each
    /// of those rebuilds allocate and copy the name again, once per field of a
    /// wide record. A shared pointer makes the carry a refcount bump, and it is
    /// atomic because a validator is shared across threads on a free-threaded
    /// interpreter.
    ///
    /// Comparison, ordering and hashing read *through* to the string, so a name
    /// is still a name: two fields with the same spelling are equal whether or
    /// not they came from the same read.
    pub name: Arc<str>,
    /// Schema the field's value must satisfy.
    pub schema: Schema,
    /// Whether the key must be present.
    pub required: bool,
}

impl Field {
    /// This field with its schema mapped through `f`, keeping its name and its
    /// required-ness.
    ///
    /// Every pass over a field list rebuilds its fields, and a struct literal
    /// spelled out in each pass is one more place to drop a field's
    /// required-ness by rebuilding it from the wrong parts.
    pub(crate) fn with_schema(&self, schema: Schema) -> Field {
        Field {
            name: Arc::clone(&self.name),
            schema,
            required: self.required,
        }
    }
}

impl Schema {
    /// The top, written as itself.
    pub const ANYTHING: Schema = Schema::Anything(Spelling::Top);
    /// The top, as the user wrote `typing.Any`. The same set as
    /// [`ANYTHING`](Self::ANYTHING), and equal to it; the two differ only in
    /// what `repr` gives back.
    pub const ANY: Schema = Schema::Anything(Spelling::Any);

    /// The class of an object schema, when this node is one: a meet carrying
    /// exactly one `isinstance` atom beside exactly one deep check of what the
    /// class declares.
    ///
    /// That pair is what the frontend builds for a class whose fields it reads,
    /// so it is what a user who wrote a class name is holding. `repr` and a
    /// union's branch label both read the class back out of it, and a later meet
    /// may flatten other members in beside the pair without changing which class
    /// it is. The shape is recognised here, once, so the two readers cannot
    /// recognise different shapes.
    ///
    /// The deep check is an attribute record where the fields are named and a
    /// sequence where they are positions: a `NamedTuple` is its class met with
    /// `tuple[...]`. Reading the named shape alone made those two class forms
    /// print differently -- a dataclass as `DC`, a `NamedTuple` as
    /// `intersection(tuple[int], NT)` -- while the page states one rule for
    /// both, and a union naming one branch said the same thing twice over.
    #[must_use]
    pub fn object_class(&self) -> Option<ClassIx> {
        let Schema::Intersection(members) = self else {
            return None;
        };
        let mut class = None;
        let mut declarations = 0usize;
        for member in members.iter() {
            match member {
                // A second atom of either kind is a meet of two objects, not one
                // object: there is no single class to name.
                Schema::Instance(_) if class.is_some() => return None,
                Schema::Instance(index) => class = Some(*index),
                Schema::AttrRecord { .. } | Schema::Seq { .. } => declarations += 1,
                _ => {}
            }
        }
        class.filter(|_| declarations == 1)
    }

    /// A short, stable label naming the expected set, shown in violations.
    #[must_use]
    pub fn expected(&self) -> &'static str {
        match self {
            Schema::Anything(_) => "anything",
            Schema::Nothing => "nothing",
            Schema::NoneType => "None",
            Schema::Bool => "bool",
            Schema::Int => "int",
            Schema::Float => "float",
            Schema::Str => "str",
            Schema::Bytes => "bytes",
            // The py layer renders the concrete constant; this is a fallback.
            Schema::Literal(_) => "literal",
            Schema::Seq {
                container: SeqKind::List,
                ..
            } => "list",
            Schema::Seq {
                container: SeqKind::Tuple,
                ..
            } => "tuple",
            Schema::Coll { container, .. } => match container {
                CollKind::Set => "set",
                CollKind::FrozenSet => "frozenset",
            },
            Schema::KeyedMap { .. } => "dict",
            Schema::Union(_) => "union",
            Schema::Intersection(_) => "intersection",
            Schema::Complement(_) => "complement",
            // The py layer renders the concrete class name; these are fallbacks.
            Schema::Instance(_) => "instance",
            Schema::AttrRecord { .. } => "object",
            // A refinement's type is its base; constraints report their own.
            Schema::Refine { base, .. } => base.expected(),
            // A reference reports through its definition at validation time.
            Schema::Ref(_) => "value",
            Schema::SelfRef(_) => "recursive value",
        }
    }

    /// The stable, machine-readable code emitted when membership fails.
    ///
    /// The ten codes a walk also writes without the node in hand are read from
    /// [`codes`], where the binding's table reads them too.
    #[must_use]
    pub fn error_code(&self) -> &'static str {
        match self {
            // Anything and Any never fail; the codes are for completeness.
            Schema::Anything(_) => "anything",
            Schema::Nothing => "no_match",
            Schema::NoneType => "none_type",
            Schema::Bool => "bool_type",
            Schema::Int => "int_type",
            Schema::Float => "float_type",
            Schema::Str => "string_type",
            Schema::Bytes => "bytes_type",
            Schema::Literal(_) => codes::LITERAL_ERROR,
            Schema::Seq {
                container: SeqKind::List,
                ..
            } => codes::LIST_TYPE,
            Schema::Seq {
                container: SeqKind::Tuple,
                ..
            } => codes::TUPLE_TYPE,
            Schema::Coll { container, .. } => match container {
                CollKind::Set => codes::SET_TYPE,
                CollKind::FrozenSet => codes::FROZEN_SET_TYPE,
            },
            Schema::KeyedMap { .. } => codes::DICT_TYPE,
            Schema::Union(_) => codes::UNION_ERROR,
            Schema::Intersection(_) => "intersection_error",
            Schema::Complement(_) => codes::UNEXPECTED_MATCH,
            Schema::Instance(_) => codes::INSTANCE_TYPE,
            // A record's failures are its fields' -- a missing attribute, or the
            // field schema's own code -- so the walk never emits this one; it is
            // here because every node has a code.
            Schema::AttrRecord { .. } => "object_type",
            Schema::Refine { base, .. } => base.error_code(),
            Schema::Ref(_) => "recursion",
            Schema::SelfRef(_) => codes::UNRESOLVED_RECURSION,
        }
    }

    /// Whether any reference is reachable from here, so a caller can tell the
    /// schemas [`unfolded`](Self::unfolded) would rebuild from the ones it would
    /// hand straight back.
    #[must_use]
    pub(crate) fn has_reference(&self) -> bool {
        let mut pending = vec![self];
        while let Some(node) = pending.pop() {
            if matches!(node, Schema::Ref(_) | Schema::SelfRef(_)) {
                return true;
            }
            node.push_children(&mut pending);
        }
        false
    }

    /// Push every schema this one holds onto `out`, one level down.
    ///
    /// The read-only companion of [`map_children`](Self::map_children): a walk
    /// that only looks does not need to rebuild what it looked at.
    pub(crate) fn push_children<'a>(&'a self, out: &mut Vec<&'a Schema>) {
        match self {
            Schema::Union(members) | Schema::Intersection(members) => out.extend(members.iter()),
            Schema::Complement(inner) | Schema::Coll { element: inner, .. } => {
                out.push(inner);
            }
            // The sequence's own element set, read from the shape rather than
            // restated here: where a sequence keeps an element is the shape's
            // to say.
            Schema::Seq { shape, .. } => out.extend(shape.elements()),
            Schema::KeyedMap { fields, defaults } => {
                out.extend(fields.iter().map(|field| &field.schema));
                for clause in defaults.iter() {
                    out.push(&clause.key);
                    out.push(&clause.value);
                }
            }
            Schema::AttrRecord { fields } => {
                out.extend(fields.iter().map(|field| &field.schema));
            }
            Schema::Refine { base, .. } => out.push(base),
            // The leaves: the scalars, the bounds, the pooled atoms and the two
            // reference forms hold no schema. Listed rather than caught by `_`,
            // so a variant that holds one is a compile error here as it is in
            // `mapped_children`.
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

    /// Every child schema of this node, in declaration order.
    ///
    /// The reading counterpart of [`map_children`](Self::map_children), and a
    /// second spelling of [`push_children`](Self::push_children) rather than a
    /// second statement of what a node holds: a reading walk matches on the
    /// child set there alone, and this is the shape a caller that wants an
    /// iterator takes. A measure that restated the set could read a different
    /// set than the map writes, and neither the types nor an exhaustive `match`
    /// would see the difference; a test holds the two that remain together.
    ///
    /// A leaf allocates nothing, since an empty `Vec` does not.
    pub(crate) fn children(&self) -> std::vec::IntoIter<&Schema> {
        let mut out = Vec::new();
        self.push_children(&mut out);
        out.into_iter()
    }

    /// The nodes this node contributes besides its children: itself, plus a
    /// refinement's constraints, which are payloads rather than child schemas.
    pub(crate) fn own_nodes(&self) -> usize {
        match self {
            Schema::Refine { constraints, .. } => 1 + constraints.len(),
            _ => 1,
        }
    }

    /// The structural nesting depth of this schema, by its definition: one
    /// level for this node, plus its deepest child's.
    ///
    /// The reading [`Measure::depth`] states, written as the recursion it is
    /// rather than as [`measure`](Self::measure)'s loop, so the property that
    /// holds the two together compares two algorithms rather than one with
    /// itself.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn depth(&self) -> usize {
        1 + self.children().map(Schema::depth).max().unwrap_or(0)
    }

    /// The total number of schema nodes in this tree, counting this node plus
    /// every node in its children.
    ///
    /// A `Ref` back edge is one node -- the definition it points at is counted
    /// where it lives in the definitions table, not re-counted through the edge --
    /// so the count of a recursive schema is finite. Combined with a per-tree
    /// depth bound, a total-node bound rejects a schema that is shallow but
    /// exponentially wide (a doubling union) without rejecting a legitimately deep
    /// or wide one. A refinement's constraints count one node each
    /// (`own_nodes`), so a long list of them is held to the bound too.
    #[must_use]
    pub fn node_count(&self) -> usize {
        // A worklist rather than the obvious recursion: the sum needs no order,
        // so the whole tree is counted against one buffer instead of one per
        // node. The root is held beside the worklist so a leaf -- which is most
        // of a schema -- allocates nothing at all.
        let mut total = 0;
        let mut pending: Vec<&Schema> = Vec::new();
        let mut root = Some(self);
        while let Some(node) = root.take().or_else(|| pending.pop()) {
            total += node.own_nodes();
            node.push_children(&mut pending);
        }
        total
    }

    /// Whether this constructor guards its children.
    ///
    /// A structural constructor does: a fixpoint unfolds one of them per step, so
    /// a self-reference below one is productive. An algebraic combinator does not:
    /// it consumes no unfolding, so a reference below it is as unguarded as the
    /// combinator itself. The match takes no wildcard, so a new variant is a
    /// compile error here, which is where the decision belongs.
    #[must_use]
    pub(crate) fn guards_children(&self) -> Guarded {
        match self {
            Schema::Seq { .. }
            | Schema::Coll { .. }
            | Schema::KeyedMap { .. }
            | Schema::AttrRecord { .. } => Guarded::Yes,
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
            | Schema::SelfRef(_)
            | Schema::Union(_)
            | Schema::Intersection(_)
            | Schema::Complement(_)
            | Schema::Refine { .. } => Guarded::No,
        }
    }

    /// [`occurs_unguarded_under`](Self::occurs_unguarded_under) over the term
    /// alone, with no definitions to follow.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn occurs_unguarded(&self, target: DefIx, guarded: Guarded) -> bool {
        self.occurs_unguarded_under(target, guarded, &[])
    }

    /// Whether `Ref(target)` occurs without a structural guard above it,
    /// following `Ref` edges through `defs`, so an occurrence one definition
    /// away is reached.
    ///
    /// A `recursive` definition is contractive (productive) only when every
    /// occurrence of its self-reference sits under a structural constructor;
    /// `guarded` records whether such a constructor has been crossed.
    ///
    /// One rule over the shared child traversal: a node joins its own guard onto
    /// the one it inherited and asks its children under the result. Because
    /// [`Guarded::Yes`] absorbs, nothing below a structural constructor is ever
    /// reported unguarded however deeply it nests -- which is the whole of the
    /// check, rather than an arm per constructor stating it again.
    ///
    /// A body that names only itself is answered by the term: every path from it
    /// to the reference is spelled out in the term. A body that names *another*
    /// definition is not — an inner fixpoint whose own body mentions this one
    /// puts the occurrence behind a `Ref`, where a term walk sees a leaf. The
    /// contractivity condition is about the whole system of definitions, so the
    /// check reads the whole system. With no definitions the two questions are
    /// the same one, so the term's question is this one over an empty graph
    /// rather than a second traversal.
    ///
    /// **A definition is entered once per check, not once per path.** Its body
    /// is always read from an unguarded root -- a guarded reference is answered
    /// before it is entered -- so what the body reaches is the same from every
    /// path to it: the check is reachability over the references met unguarded.
    /// A definition entered once and found not to reach the target is not
    /// entered again, and one still open is the walk already looking from it.
    /// Forgetting a definition when its walk returns would walk it once per
    /// path: a chain whose every link names the one before it twice is 2^n
    /// walks for n definitions that way, 0.27 s at eighteen in a debug build,
    /// of the 128 a validator may hold.
    #[must_use]
    pub fn occurs_unguarded_under(&self, target: DefIx, guarded: Guarded, defs: &[Schema]) -> bool {
        self.occurs_unguarded_within(target, guarded, defs, &mut Vec::new())
    }

    /// The walk of [`occurs_unguarded_under`](Self::occurs_unguarded_under),
    /// with the definitions it has entered so far.
    pub(crate) fn occurs_unguarded_within(
        &self,
        target: DefIx,
        guarded: Guarded,
        defs: &[Schema],
        entered: &mut Vec<DefIx>,
    ) -> bool {
        if let Schema::Ref(id) = self {
            if guarded == Guarded::Yes {
                return false;
            }
            if *id == target {
                return true;
            }
            if entered.contains(id) {
                return false;
            }
            let Some(definition) = defs.get(id.get()) else {
                return false;
            };
            entered.push(*id);
            return definition.occurs_unguarded_within(target, Guarded::No, defs, entered);
        }
        let below = guarded.join(self.guards_children());
        self.children()
            .any(|child| child.occurs_unguarded_within(target, below, defs, entered))
    }

    /// The three facts a validator is accepted on, read in one walk: how deep
    /// the tree nests, how many nodes it spans, and whether it carries a
    /// self-reference marker whose token `is_open` does not recognise.
    ///
    /// The depth is the one [`Measure::depth`] states, the count is what
    /// [`node_count`](Self::node_count) returns, and the marker is the test
    /// below; a property test holds the depth and the marker to recursions over
    /// the children and the count to `node_count`. They are one walk because
    /// each of them visits every node and building a validator asks for all
    /// three, which apart would be three passes over one tree.
    ///
    /// **The marker.** A finished schema carries none: `recursive` resolves the
    /// one it minted into a back edge before it returns. A marker reaches
    /// construction two ways, and only one of them is legitimate -- from inside
    /// the builder of the definition it stands for, where the schemas the caller
    /// composes carry it until that definition closes; or from a placeholder kept
    /// past the call it was handed to, which stands for a fixpoint nobody is
    /// defining any more. Which token is which is the caller's fact, so the
    /// caller brings the test and this walk brings the traversal.
    ///
    /// Read level by level against two buffers rather than by recursion. The
    /// obvious spelling asks each node for its `children`, which collects them
    /// into a vector of that node's own, so every node holding a child
    /// allocates one and frees it; two buffers serve the whole walk, and a leaf
    /// allocates neither. The walk is also iterative, so reading the depth off
    /// a deep schema descends none of the native stack the depth bound is there
    /// to protect.
    ///
    /// **The level is left before it is counted, rather than counted while the
    /// list is not empty.** The tidier spelling is `while !level.is_empty()`,
    /// and the mutation that deletes that `!` is a schema measure that never
    /// returns: the sweep spends its whole budget on it and reports no verdict,
    /// which `scripts/mutation_gate.py` reads as a rig fault rather than as a
    /// result, and the lane fails having judged nothing. Breaking out of a
    /// `loop` reads the same condition and compiles to the same walk, and no
    /// single edit to it removes the exit.
    #[must_use]
    pub fn measure(&self, is_open: &dyn Fn(u64) -> bool) -> Measure {
        let escapes = |node: &Schema| matches!(node, Schema::SelfRef(token) if !is_open(*token));
        let mut measure = Measure {
            depth: 1,
            nodes: self.own_nodes(),
            escaped_self_ref: escapes(self),
        };
        let mut level: Vec<&Schema> = Vec::new();
        let mut next: Vec<&Schema> = Vec::new();
        self.push_children(&mut level);
        loop {
            if level.is_empty() {
                break;
            }
            measure.depth += 1;
            for node in level.drain(..) {
                measure.nodes += node.own_nodes();
                measure.escaped_self_ref |= escapes(node);
                node.push_children(&mut next);
            }
            std::mem::swap(&mut level, &mut next);
        }
        measure
    }
}

/// What [`Schema::measure`] reads off a tree: the facts a validator is accepted
/// on, taken in one walk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Measure {
    /// How deep the tree nests: the longest chain of nested constructors from
    /// the root down to a leaf, each node one level.
    ///
    /// A `Ref` counts as a leaf even though it names a recursive definition: the
    /// back edge is not followed, so the depth of a recursive schema is finite.
    /// The count mirrors the native stack a recursive walk over the tree
    /// descends -- one frame per level in clone, drop, the decision procedure,
    /// and the render back to an annotation -- so composition can be bounded to
    /// a depth every such walk survives. A sequence is no exception: a
    /// [`SeqShape`] holds its elements directly, so reaching one costs the same
    /// descent as reaching a set's element or a field's schema.
    pub depth: usize,
    /// How many nodes the tree spans, as [`Schema::node_count`] counts it.
    pub nodes: usize,
    /// Whether a self-reference marker the caller does not recognise is
    /// anywhere in the tree.
    pub escaped_self_ref: bool,
}

/// The sizes every walk and decision copies, pinned so a layout change is a
/// compile error rather than a cost a gate reads as a percent: a schema node is
/// cloned into every narrowing and a field into every rebuilt record, and the
/// layout of a `repr(Rust)` enum is the compiler's to change between releases.
#[cfg(target_pointer_width = "64")]
const _: () = {
    assert!(size_of::<Schema>() == 40);
    assert!(size_of::<Field>() == 64);
};
