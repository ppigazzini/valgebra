//! The schema intermediate representation: the IR node definitions and the
//! pure structural operations over them (construction, index shifting,
//! self-reference resolution, and the structural guardedness check).
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
//! same reason: [`Schema::shifted`] takes one per space, and they used to be two
//! adjacent `usize` arguments a caller could transpose in silence.
//!
//! The types stop a shift reaching the wrong space. They say nothing about
//! whether a payload was shifted at all, which is why one walk serves both ways
//! of combining two pools ([`Remap`]) and why the child set of each variant is
//! declared in one place ([`Schema::map_children`]).

/// Remap a pool index through the reindexing map built when two validators merge.
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

/// How far every constants-pool index moves when a second validator's pool is
/// appended to a first. Distinct from [`DefShift`] so the two cannot be
/// transposed at [`Schema::shifted`], which takes one of each.
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
            /// are being combined into. Asserted rather than assumed, because
            /// the release profile wraps and a wrapped index would read a real
            /// object of the wrong kind.
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

pool_index!(
    /// The pool slot holding a [`Schema::Literal`]'s constant.
    ConstIx,
    "a literal's constant"
);
pool_index!(
    /// The pool slot holding the class of a [`Schema::Instance`] or a
    /// [`Schema::Attrs`]. One type for both: they address the same kind of
    /// object and no call site carries one of each, so a second type would be a
    /// distinction with no swap behind it.
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
/// before the split both travelled as `usize`, so either reached either.
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
/// identity-shared constants collapse -- and the definitions table in one. The
/// two ways were two whole structural walks over the IR, identical but for the
/// leaf action, so a payload site reached by one and missed by the other was a
/// wrong index the compiler could not see. Naming the difference leaves one walk
/// and puts the difference at the leaf.
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
/// A name rather than a positional `bool`: `Schema::record(fields, Openness::Closed)` reads
/// as a flag whose polarity a caller has to remember, and the two openness
/// senses -- "closed to extra keys" and "open to them" -- are exactly the pair a
/// reader gets backwards.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Openness {
    /// Only the declared keys are admitted.
    Closed,
    /// Any further key is admitted, with any value.
    Open,
}

impl Openness {
    /// The openness a boolean flag denotes, for a caller that has one in hand
    /// (the Python surface takes `open=True`/`False`).
    #[must_use]
    pub const fn from_flag(open: bool) -> Self {
        if open {
            Openness::Open
        } else {
            Openness::Closed
        }
    }
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
    /// occurrence -- and it was threaded by hand through every structural arm.
    #[must_use]
    pub const fn join(self, other: Guarded) -> Guarded {
        match (self, other) {
            (Guarded::No, Guarded::No) => Guarded::No,
            _ => Guarded::Yes,
        }
    }
}

/// A single step in the location of a value inside a composite structure.
///
/// Scalar schemas never produce a path; structural schemas (records, sequences,
/// tuples, sets, mappings) push a segment per level as they descend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PathSegment {
    /// A mapping or record key.
    Key(String),
    /// A sequence, tuple, or set position.
    Index(usize),
}

/// The schema intermediate representation.
///
/// Each variant documents its denotation: the set of Python values it accepts.
/// `Ord`/`Eq` are structural; the simplifier uses them to canonicalize the
/// order of union and intersection members and to deduplicate.
///
/// Adding a variant means handling it in every walk over the IR; the compiler
/// forces the exhaustive `match`es. Checklist:
/// - core: `Schema::map_children`, which is where the variant's child schemas
///   are declared and which every purely structural walk reads them from;
///   `Schema::remapped_by`, if it carries a pooled or definitions index;
///   [`Schema::expected`], [`Schema::error_code`], [`Schema::depth`],
///   [`Schema::node_count`], [`Schema::occurs_unguarded`],
///   [`Schema::simplify`];
/// - bindings (`valgebra-py`): the single `member` membership walk (which
///   decides membership and, in explain mode, aggregates the violation) plus
///   `render`.
///
/// The compiler forces an arm; it cannot check the arm recursed into every child.
/// That is why the child set is declared once, in `map_children`, rather than per
/// walk.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Schema {
    /// Top. Denotes every Python value; membership always holds.
    Anything,
    /// The gradual dynamic type (the user spells it `typing.Any`). At runtime it
    /// admits every value like the top, but it is a distinct atom: the simplifier
    /// must not rewrite it by the lattice laws, so `Dynamic` and
    /// [`Schema::Anything`] are kept separate. Named for the gradual-typing
    /// term (Siek-Taha; ty's `Dynamic`), not the Python surface spelling.
    Dynamic,
    /// Bottom. Denotes the empty set; membership never holds.
    Nothing,
    /// Denotes the singleton set `{None}`.
    NoneType,
    /// Denotes `{True, False}`, exactly the `bool` instances.
    ///
    /// Because `bool` is a subclass of `int`, this set is a subset of
    /// [`Schema::Int`]: `Bool` is a subtype of `Int`.
    Bool,
    /// Denotes every `int` instance: `isinstance(x, int)`.
    ///
    /// In Python `bool` is a subclass of `int`, so `True` and `False` are
    /// integers and are members of this set. No value is carved out: subtyping
    /// is subset inclusion, so [`Schema::Bool`] is a subtype of `Int` rather
    /// than disjoint from it.
    Int,
    /// Denotes every `float` instance: `isinstance(x, float)`.
    ///
    /// `int` does not subclass `float`, so `Int` and `Float` are disjoint and
    /// an integer is not a member.
    Float,
    /// Denotes the `str` instances.
    Str,
    /// Denotes the `bytes` instances.
    Bytes,
    /// Denotes the typed singleton `{c}` for a fixed constant `c`: a value is a
    /// member iff it has the *same type* as `c` and is equal to it. Same-type is
    /// what makes this a singleton — Python's `==` conflates across types
    /// (`1 == True == 1.0`), so equality alone would make `Literal[1]` also
    /// admit `True` and `1.0`. Requiring `type(x) is type(c)` keeps the typing
    /// spec's distinction between `Literal[1]`, `Literal[True]`, and
    /// `Literal[1.0]`.
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
    /// Denotes sets whose every element belongs to the inner schema.
    Set(Box<Schema>),
    /// Denotes frozensets whose every element belongs to the inner schema.
    FrozenSet(Box<Schema>),
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
        fields: Vec<Field>,
        /// `(key-schema, value-schema)` clauses governing every key that is not
        /// a declared field name. A key belongs when **some** clause admits both
        /// it and its value, so the clauses are a disjunction and not a
        /// precedence list: the order is the order they are rendered in, and
        /// carries no meaning to membership or to subtyping. Both consumers ask
        /// `any`, and this comment once said "ordered", which is a semantics no
        /// code here implements.
        defaults: Vec<MapClause>,
    },
    /// Denotes the union of the member sets: a value is a member iff it belongs
    /// to at least one member schema.
    Union(Vec<Schema>),
    /// Denotes the intersection of the member sets: a value is a member iff it
    /// belongs to every member schema.
    Intersection(Vec<Schema>),
    /// Denotes the complement of the inner set: a value is a member iff it is
    /// not a member of the inner schema.
    Complement(Box<Schema>),
    /// Denotes instances of a class, by `isinstance`. The class is held in the
    /// validator's object pool; the payload is its index.
    Instance(ClassIx),
    /// An instance of a class whose attributes satisfy the given fields — an
    /// `isinstance` atom intersected with an attribute record (`Instance ∧
    /// attrs`). Named `Attrs` so it does not collide with `object`, the lattice
    /// top, which the frontend maps to [`Schema::Anything`].
    ///
    /// `isinstance` against the pooled class at `class_index` must hold, and
    /// every field's attribute must be present and match. This is the deep
    /// check for dataclasses and named tuples.
    Attrs {
        /// Index of the class in the validator's object pool.
        class_index: ClassIx,
        /// Per-attribute field schemas; all required.
        fields: Vec<Field>,
    },
    /// Denotes the subset of the base set satisfying every constraint
    /// (`{ x in [[base]] | all constraints hold }`). The base is checked first.
    Refine {
        /// The base schema; a value must belong to it before constraints apply.
        base: Box<Schema>,
        /// Constraints that further narrow the base set, checked in order.
        constraints: Vec<Constraint>,
    },
    /// A reference to a recursive definition: denotes the same set as the
    /// definition at this index in the validator's definitions table. The back
    /// edge of a fixpoint, produced by `recursive`.
    Ref(DefIx),
    /// A transient self-reference marker used only while a `recursive` definition is
    /// being built; it is resolved to a [`Schema::Ref`] before the validator is
    /// returned and never appears in a finished schema.
    SelfRef(u64),
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
/// nested repetition, so carrying the general form meant every walk answering
/// for shapes no value could reach and every membership check first proving the
/// shape it held was one of the three. The shape it holds is now the only shape
/// there is.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct SeqShape {
    /// The positional element schemas, matched in order from the front.
    pub prefix: Vec<Schema>,
    /// The schema every element past the prefix must belong to, or `None` when
    /// the sequence ends at the prefix.
    pub tail: Option<Box<Schema>>,
}

impl SeqShape {
    /// Map every element schema through `f`, preserving the shape.
    pub(crate) fn map_elems(&self, f: &impl Fn(&Schema) -> Schema) -> SeqShape {
        SeqShape {
            prefix: self.prefix.iter().map(f).collect(),
            tail: self.tail.as_deref().map(|t| Box::new(f(t))),
        }
    }

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

    /// A homogeneous mapping `dict[K, V]`: every key in `key`, every value in
    /// `value`.
    #[must_use]
    pub fn mapping(clause: MapClause) -> Schema {
        Schema::KeyedMap {
            fields: Vec::new(),
            defaults: vec![clause],
        }
    }

    /// The union of `members`: a value belongs when it belongs to at least one.
    ///
    /// A union of no members is the bottom, which is this operation's identity.
    /// The constructor owns that, so a caller folding an empty collection gets an
    /// atom every consumer already reads rather than a node each must special-case
    /// -- the render printed the empty join of no members as an empty string.
    ///
    /// Members are kept in the order given and are not deduplicated: the order is
    /// observable through the render and through structural equality, and
    /// [`simplify`](Self::simplify) is where the lattice laws apply.
    #[must_use]
    pub fn union(members: impl IntoIterator<Item = Schema>) -> Schema {
        let members: Vec<Schema> = members.into_iter().collect();
        if members.is_empty() {
            return Schema::Nothing;
        }
        // A join carrying a schema together with its complement is the top,
        // whatever those are, so no such join survives construction and no rule
        // downstream may assume one does. `has_complementary_pair` is the one
        // statement of the law, shared with the simplifier; it reaches a pairwise
        // comparison only for a member that is itself a complement.
        // With no oracle here the law declines for an atom only the bindings can
        // read, which is the safe direction: a join left unfolded still denotes
        // what it denotes.
        if crate::decision::has_complementary_pair(&members, &crate::decision::NoLeafRelations) {
            return Schema::Anything;
        }
        Schema::Union(members)
    }

    /// The meet of `members`: a value belongs when it belongs to every one.
    ///
    /// A meet of no members is the top, dually to [`union`](Self::union).
    #[must_use]
    pub fn meet(members: impl IntoIterator<Item = Schema>) -> Schema {
        let members: Vec<Schema> = members.into_iter().collect();
        if members.is_empty() {
            Schema::Anything
        } else {
            Schema::Intersection(members)
        }
    }

    /// The complement: every value this schema does not admit.
    ///
    /// Named for the operation rather than spelled as `!`, matching `Region` and
    /// the Python surface: a one-character operator in a fold is a one-character
    /// defect, and typing has no operator for this one.
    ///
    /// `~~A` is `A`, so a complement of a complement cancels. Nothing downstream
    /// carries a double negation, and no rule anywhere may assume one exists.
    #[must_use]
    pub fn complement(self) -> Schema {
        match self {
            Schema::Complement(inner) => *inner,
            other => Schema::Complement(Box::new(other)),
        }
    }

    /// A record of named fields. An open record admits any other key with any
    /// value; a closed one admits none.
    #[must_use]
    pub fn record(fields: Vec<Field>, open: Openness) -> Schema {
        Schema::keyed_map(
            fields,
            match open {
                Openness::Open => vec![MapClause::top()],
                Openness::Closed => Vec::new(),
            },
        )
    }

    /// A dict node: named fields, and the catch-all clauses governing every other
    /// key.
    ///
    /// The constructor the other two are written in terms of, so a caller never
    /// builds the variant raw. `record` and `mapping` are the two shapes the
    /// frontend spells; this is the general one they are special cases of, and a
    /// mixed record-and-catch-all needs it.
    #[must_use]
    pub fn keyed_map(fields: Vec<Field>, defaults: Vec<MapClause>) -> Schema {
        Schema::KeyedMap { fields, defaults }
    }
}

impl SeqShape {
    /// The homogeneous form: any number of elements, each in `element`.
    #[must_use]
    pub fn homogeneous(element: Schema) -> SeqShape {
        SeqShape {
            prefix: Vec::new(),
            tail: Some(Box::new(element)),
        }
    }

    /// The fixed form: each element positionally, and no element past them.
    #[must_use]
    pub fn fixed(elements: impl IntoIterator<Item = Schema>) -> SeqShape {
        SeqShape {
            prefix: elements.into_iter().collect(),
            tail: None,
        }
    }

    /// The prefix-plus-tail form: a fixed positional prefix, then zero or more
    /// elements in `tail`.
    #[must_use]
    pub fn prefix_tail(prefix: impl IntoIterator<Item = Schema>, tail: Schema) -> SeqShape {
        SeqShape {
            prefix: prefix.into_iter().collect(),
            tail: Some(Box::new(tail)),
        }
    }
}

/// A constraint narrowing a [`Schema::Refine`] base set.
///
/// Comparison and predicate operands live in the validator's object pool; the
/// payload is an index. Length bounds carry the length directly.
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
    /// `len(value) >= n`.
    MinLen(usize),
    /// `len(value) <= n`.
    MaxLen(usize),
    /// `value % pool[i] == 0`: a numeric multiple of the operand.
    MultipleOf(OperandIx),
    /// `pool[i](value)` is truthy. The documented Python-callback slow path.
    Predicate(PredIx),
    /// The string fully matches this regular expression (anchored, `re.fullmatch`
    /// semantics). The pattern is held inline rather than pooled; the bindings
    /// compile it once and match natively. Like [`Constraint::Predicate`] it is a
    /// leaf the decision procedure treats opaquely: two regex constraints relate
    /// only when their patterns are identical.
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
            key: Schema::Anything,
            value: Schema::Anything,
        }
    }

    /// This clause with both schemas mapped through `f`.
    pub(crate) fn map_schemas(&self, f: &impl Fn(&Schema) -> Schema) -> MapClause {
        MapClause {
            key: f(&self.key),
            value: f(&self.value),
        }
    }
}

/// A named field of a [`Schema::KeyedMap`] or [`Schema::Attrs`].
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Field {
    /// The key name.
    pub name: String,
    /// Schema the field's value must satisfy.
    pub schema: Schema,
    /// Whether the key must be present.
    pub required: bool,
}

impl Field {
    /// This field with its schema mapped through `f`, keeping its name and its
    /// required-ness.
    ///
    /// The three-line struct literal that spells this out was written at every
    /// pass over a field list, and it is the one place a pass could drop a
    /// field's required-ness by rebuilding it from the wrong parts.
    pub(crate) fn map_schema(&self, f: &impl Fn(&Schema) -> Schema) -> Field {
        Field {
            name: self.name.clone(),
            schema: f(&self.schema),
            required: self.required,
        }
    }
}

impl Schema {
    /// A short, stable label naming the expected set, shown in violations.
    #[must_use]
    pub fn expected(&self) -> &'static str {
        match self {
            Schema::Anything => "anything",
            Schema::Dynamic => "any",
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
            Schema::Set(_) => "set",
            Schema::FrozenSet(_) => "frozenset",
            Schema::KeyedMap { .. } => "dict",
            Schema::Union(_) => "union",
            Schema::Intersection(_) => "intersection",
            Schema::Complement(_) => "complement",
            // The py layer renders the concrete class name; these are fallbacks.
            Schema::Instance(_) => "instance",
            Schema::Attrs { .. } => "object",
            // A refinement's type is its base; constraints report their own.
            Schema::Refine { base, .. } => base.expected(),
            // A reference reports through its definition at validation time.
            Schema::Ref(_) => "value",
            Schema::SelfRef(_) => "recursive value",
        }
    }

    /// The stable, machine-readable code emitted when membership fails.
    #[must_use]
    pub fn error_code(&self) -> &'static str {
        match self {
            // Anything and Any never fail; the codes are for completeness.
            Schema::Anything => "anything",
            Schema::Dynamic => "any",
            Schema::Nothing => "no_match",
            Schema::NoneType => "none_type",
            Schema::Bool => "bool_type",
            Schema::Int => "int_type",
            Schema::Float => "float_type",
            Schema::Str => "string_type",
            Schema::Bytes => "bytes_type",
            Schema::Literal(_) => "literal_error",
            Schema::Seq {
                container: SeqKind::List,
                ..
            } => "list_type",
            Schema::Seq {
                container: SeqKind::Tuple,
                ..
            } => "tuple_type",
            Schema::Set(_) => "set_type",
            Schema::FrozenSet(_) => "frozen_set_type",
            Schema::KeyedMap { .. } => "dict_type",
            Schema::Union(_) => "union_error",
            Schema::Intersection(_) => "intersection_error",
            Schema::Complement(_) => "unexpected_match",
            Schema::Instance(_) | Schema::Attrs { .. } => "instance_type",
            Schema::Refine { base, .. } => base.error_code(),
            Schema::Ref(_) => "recursion",
            Schema::SelfRef(_) => "unresolved_recursion",
        }
    }

    /// Rebuild this node with every child schema mapped through `f`, leaving the
    /// node's own payloads -- the container kind, a field's name and
    /// required-ness, a pooled index, a constraint -- exactly as they are.
    ///
    /// **This is the one place the child set of each variant is written down.** A
    /// walk that only descends -- moving indices, resolving a self-reference --
    /// used to spell the whole descent out per pass, and the compiler forced an
    /// arm without being able to check the arm recursed into everything: a
    /// forgotten child was a silent stale subtree. Written once, every such pass
    /// inherits the child set.
    ///
    /// A new variant carrying a child schema must map it here, or every pass
    /// built on this drops it. A new variant carrying a *pooled index* must also
    /// be handled in [`remapped_by`](Self::remapped_by), which is why that match
    /// takes no wildcard.
    pub(crate) fn map_children(&self, f: &impl Fn(&Schema) -> Schema) -> Schema {
        let field = |field: &Field| field.map_schema(f);
        match self {
            Schema::Anything
            | Schema::Dynamic
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
            | Schema::SelfRef(_) => self.clone(),
            Schema::Seq { container, shape } => Schema::Seq {
                container: *container,
                shape: shape.map_elems(f),
            },
            Schema::Set(inner) => Schema::Set(Box::new(f(inner))),
            Schema::FrozenSet(inner) => Schema::FrozenSet(Box::new(f(inner))),
            Schema::Complement(inner) => Schema::Complement(Box::new(f(inner))),
            Schema::Union(members) => Schema::Union(members.iter().map(f).collect()),
            Schema::Intersection(members) => Schema::Intersection(members.iter().map(f).collect()),
            Schema::KeyedMap { fields, defaults } => Schema::KeyedMap {
                fields: fields.iter().map(field).collect(),
                defaults: defaults.iter().map(|c| c.map_schemas(f)).collect(),
            },
            Schema::Attrs {
                class_index,
                fields,
            } => Schema::Attrs {
                class_index: *class_index,
                fields: fields.iter().map(field).collect(),
            },
            Schema::Refine { base, constraints } => Schema::Refine {
                base: Box::new(f(base)),
                constraints: constraints.clone(),
            },
        }
    }

    /// Rebuild this schema against another validator's pools, moving every
    /// payload the way `remap` moves its index space.
    ///
    /// The arms here are exactly the nodes that *carry* an index; the structural
    /// descent is [`map_children`](Self::map_children). The match deliberately
    /// takes **no wildcard**: a variant that carries a pooled index and reaches
    /// this by a catch-all would keep its index and read a real object of the
    /// wrong kind, which is a plausible wrong verdict rather than a crash. Listing
    /// the structural variants costs a line each and makes a new variant a
    /// compile error here, where the decision belongs.
    fn remapped_by(&self, remap: Remap<'_>) -> Schema {
        match self {
            Schema::Literal(index) => Schema::Literal(index.remapped_by(remap)),
            Schema::Instance(index) => Schema::Instance(index.remapped_by(remap)),
            Schema::Ref(index) => Schema::Ref(index.remapped_by(remap)),
            Schema::Attrs {
                class_index,
                fields,
            } => Schema::Attrs {
                class_index: class_index.remapped_by(remap),
                fields: fields
                    .iter()
                    .map(|field| Field {
                        name: field.name.clone(),
                        schema: field.schema.remapped_by(remap),
                        required: field.required,
                    })
                    .collect(),
            },
            Schema::Refine { base, constraints } => Schema::Refine {
                base: Box::new(base.remapped_by(remap)),
                constraints: constraints
                    .iter()
                    .map(|constraint| constraint.remapped_by(remap))
                    .collect(),
            },
            // No payload of its own: descend, and let the child set live in one
            // place. Spelled out rather than caught by `_` so a new variant with
            // an index cannot arrive here silently.
            Schema::Anything
            | Schema::Dynamic
            | Schema::Nothing
            | Schema::NoneType
            | Schema::Bool
            | Schema::Int
            | Schema::Float
            | Schema::Str
            | Schema::Bytes
            | Schema::SelfRef(_)
            | Schema::Seq { .. }
            | Schema::Set(_)
            | Schema::FrozenSet(_)
            | Schema::Complement(_)
            | Schema::Union(_)
            | Schema::Intersection(_)
            | Schema::KeyedMap { .. } => self.map_children(&|s| s.remapped_by(remap)),
        }
    }

    /// Return a copy with pool indices shifted by `pool` and definition
    /// references shifted by `defs`.
    ///
    /// Used when composing two compiled validators: their constants pools and
    /// definitions tables are concatenated, so the second schema's
    /// `Literal`/`Instance`/`Attrs`/`Refine` indices move past the first
    /// pool's length and its `Ref` indices past the first definitions' length.
    #[must_use]
    pub fn shifted(&self, pool: PoolShift, defs: DefShift) -> Schema {
        self.remapped_by(Remap::Append { pool, defs })
    }

    /// Every child schema of this node, in declaration order.
    ///
    /// The one place a *reading* walk learns what a node contains, as
    /// [`map_children`](Self::map_children) is the one place a *rebuilding* walk
    /// does. A measure that restates the child set can read a different set than
    /// the map writes, and neither the types nor an exhaustive `match` sees the
    /// difference; a test holds the two together.
    pub(crate) fn children(&self) -> Box<dyn Iterator<Item = &Schema> + '_> {
        match self {
            Schema::Anything
            | Schema::Dynamic
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
            | Schema::SelfRef(_) => Box::new(core::iter::empty()),
            Schema::Seq { shape, .. } => Box::new(shape.elements()),
            Schema::Set(inner) | Schema::FrozenSet(inner) | Schema::Complement(inner) => {
                Box::new(core::iter::once(inner.as_ref()))
            }
            Schema::Union(members) | Schema::Intersection(members) => Box::new(members.iter()),
            Schema::KeyedMap { fields, defaults } => Box::new(
                fields.iter().map(|field| &field.schema).chain(
                    defaults
                        .iter()
                        .flat_map(|clause| [&clause.key, &clause.value].into_iter()),
                ),
            ),
            Schema::Attrs { fields, .. } => Box::new(fields.iter().map(|field| &field.schema)),
            Schema::Refine { base, .. } => Box::new(core::iter::once(base.as_ref())),
        }
    }

    /// The nodes this node contributes besides its children: itself, plus a
    /// refinement's constraints, which are payloads rather than child schemas.
    pub(crate) fn own_nodes(&self) -> usize {
        match self {
            Schema::Refine { constraints, .. } => 1 + constraints.len(),
            _ => 1,
        }
    }

    /// The structural nesting depth of this schema: the longest chain of nested
    /// constructors from here down to a leaf.
    ///
    /// A `Ref` counts as a leaf even though it names a recursive definition: the
    /// back edge is not followed, so the depth of a recursive schema is finite and
    /// this terminates. The count mirrors the native stack a recursive walk over
    /// the tree descends -- one frame per level in clone, drop, the decision
    /// procedure, and the render back to an annotation -- so composition can be
    /// bounded to a depth every such walk survives.
    ///
    /// Every node is one such level, sequences included: a [`SeqShape`] holds its
    /// elements directly, so reaching one costs the same descent as reaching a
    /// set's element or a field's schema. While a sequence carried a regular
    /// expression a sequence node counted its constructor nesting too, because a
    /// walk descended those boxes as well.
    #[must_use]
    pub fn depth(&self) -> usize {
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
    /// or wide one. A regex constructor is not a schema node and is not counted.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.own_nodes() + self.children().map(Schema::node_count).sum::<usize>()
    }

    /// Like [`shifted`](Self::shifted), but remapping pool indices through
    /// `lit_map` (an old->new table from interning one pool into another, so
    /// identity-shared constants collapse to one index) while still offsetting
    /// definition indices by `def_offset`.
    ///
    /// The same walk as `shifted`, over the same payload sites: the two differ
    /// only in how a pool index moves, which is what `Remap` names.
    #[must_use]
    pub fn reindexed(&self, lit_map: &[usize], def_offset: DefShift) -> Schema {
        self.remapped_by(Remap::Intern {
            lit_map,
            defs: def_offset,
        })
    }

    /// Replace each `SelfRef(token)` with `Ref(ref_id)`, leaving other tokens
    /// (from enclosing `recursive` definitions) untouched.
    #[must_use]
    pub fn resolve_self(&self, token: u64, ref_id: DefIx) -> Schema {
        match self {
            Schema::SelfRef(t) if *t == token => Schema::Ref(ref_id),
            // Every other node keeps its payloads and passes the rewrite down. A
            // wildcard is right here, unlike in `remapped_by`: this pass rewrites
            // one marker and touches nothing else, so it is already correct for a
            // variant that does not exist yet -- provided that variant's children
            // are mapped in `map_children`, which is the one place to add them.
            other => other.map_children(&|s| s.resolve_self(token, ref_id)),
        }
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
            | Schema::Set(_)
            | Schema::FrozenSet(_)
            | Schema::KeyedMap { .. }
            | Schema::Attrs { .. } => Guarded::Yes,
            Schema::Anything
            | Schema::Dynamic
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

    /// Whether `Ref(target)` occurs without a structural guard above it.
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
    #[must_use]
    pub fn occurs_unguarded(&self, target: DefIx, guarded: Guarded) -> bool {
        self.occurs_unguarded_under(target, guarded, &[])
    }

    /// [`occurs_unguarded`](Self::occurs_unguarded) following `Ref` edges through
    /// `defs`, so an occurrence one definition away is reached.
    ///
    /// A body that names only itself is answered by the term: every path from it
    /// to the reference is spelled out in the term. A body that names *another*
    /// definition is not — an inner fixpoint whose own body mentions this one
    /// puts the occurrence behind a `Ref`, where a term walk sees a leaf. The
    /// contractivity condition is about the whole system of definitions, so the
    /// check reads the whole system; with no definitions the two questions are
    /// the same one, which is why the term entry point is this with an empty
    /// graph rather than a second traversal.
    ///
    /// A definition already open on the path is not re-entered: the walk that
    /// opened it is still looking for the target, so a second visit adds no path
    /// to it and would not terminate.
    #[must_use]
    pub fn occurs_unguarded_under(&self, target: DefIx, guarded: Guarded, defs: &[Schema]) -> bool {
        self.occurs_unguarded_within(target, guarded, defs, &mut Vec::new())
    }

    fn occurs_unguarded_within(
        &self,
        target: DefIx,
        guarded: Guarded,
        defs: &[Schema],
        visiting: &mut Vec<DefIx>,
    ) -> bool {
        if let Schema::Ref(id) = self {
            if guarded == Guarded::Yes {
                return false;
            }
            if *id == target {
                return true;
            }
            if visiting.contains(id) {
                return false;
            }
            let Some(definition) = defs.get(id.get()) else {
                return false;
            };
            visiting.push(*id);
            let reaches = definition.occurs_unguarded_within(target, Guarded::No, defs, visiting);
            visiting.pop();
            return reaches;
        }
        let below = guarded.join(self.guards_children());
        self.children()
            .any(|child| child.occurs_unguarded_within(target, below, defs, visiting))
    }

    /// Whether this schema carries a self-reference marker whose token `is_open`
    /// does not recognise.
    ///
    /// A finished schema carries no marker at all: `recursive` resolves the one
    /// it minted into a back edge before it returns. A marker reaches
    /// construction two ways, and only one of them is legitimate — from inside
    /// the builder of the definition it stands for, where the schemas the caller
    /// composes carry it until that definition closes; or from a placeholder kept
    /// past the call it was handed to, which stands for a fixpoint nobody is
    /// defining any more. Which token is which is the caller's fact, so the
    /// caller brings the test and this walk brings the traversal.
    #[must_use]
    pub fn has_escaped_self_ref(&self, is_open: &dyn Fn(u64) -> bool) -> bool {
        match self {
            Schema::SelfRef(token) => !is_open(*token),
            _ => self
                .children()
                .any(|child| child.has_escaped_self_ref(is_open)),
        }
    }

    /// Return a copy with every record-shaped [`Schema::KeyedMap`] in the tree
    /// set to `open`.
    ///
    /// This backs the `open`/`close` methods: `open` opens every record in a
    /// subtree (undeclared keys allowed via an `anything` catch-all), `close`
    /// closes them. A pure mapping (no named fields) is not a record and keeps
    /// its clauses.
    #[must_use]
    pub fn with_records_open(&self, open: Openness) -> Schema {
        match self {
            // The one node this transform is about: a record (named fields)
            // replaces its catch-all. A pure mapping has no fields, so it is not
            // a record and falls through to the descent below.
            Schema::KeyedMap { fields, .. } if !fields.is_empty() => Schema::KeyedMap {
                fields: fields
                    .iter()
                    .map(|field| field.map_schema(&|s| s.with_records_open(open)))
                    .collect(),
                defaults: match open {
                    Openness::Open => vec![MapClause::top()],
                    Openness::Closed => Vec::new(),
                },
            },
            // Every other node carries the transform to its children and keeps
            // its own payloads. Spelling the descent out here again is what let
            // it end in a wildcard, where a new child-carrying variant would be
            // cloned unopened rather than failing to compile.
            _ => self.map_children(&|s| s.with_records_open(open)),
        }
    }
}

impl Constraint {
    /// This constraint against another validator's pool.
    ///
    /// A length bound and a regex pattern are not pool indices and are carried
    /// through untouched. The type says so for the length: a `usize` has no
    /// `remapped_by`, so an arm that must not move an index cannot.
    fn remapped_by(&self, remap: Remap<'_>) -> Constraint {
        match self {
            Constraint::Ge(index) => Constraint::Ge(index.remapped_by(remap)),
            Constraint::Gt(index) => Constraint::Gt(index.remapped_by(remap)),
            Constraint::Le(index) => Constraint::Le(index.remapped_by(remap)),
            Constraint::Lt(index) => Constraint::Lt(index.remapped_by(remap)),
            Constraint::MinLen(n) => Constraint::MinLen(*n),
            Constraint::MaxLen(n) => Constraint::MaxLen(*n),
            Constraint::MultipleOf(index) => Constraint::MultipleOf(index.remapped_by(remap)),
            Constraint::Predicate(index) => Constraint::Predicate(index.remapped_by(remap)),
            Constraint::Regex(pattern) => Constraint::Regex(pattern.clone()),
        }
    }
}
