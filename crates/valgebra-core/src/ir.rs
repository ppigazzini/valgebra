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
    #[must_use]
    pub const fn new(by: usize) -> Self {
        Self(by)
    }

    /// The underlying distance.
    #[must_use]
    pub const fn get(self) -> usize {
        self.0
    }
}

/// How far every definitions-table index moves when a second validator's
/// definitions are appended to a first. See [`PoolShift`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct DefShift(usize);

impl DefShift {
    /// A shift of `by` definition slots.
    #[must_use]
    pub const fn new(by: usize) -> Self {
        Self(by)
    }

    /// The underlying distance.
    #[must_use]
    pub const fn get(self) -> usize {
        self.0
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
            #[must_use]
            fn shifted(self, by: PoolShift) -> Self {
                Self(self.0 + by.0)
            }

            /// This index after the pool was interned into another, collapsing
            /// identity-shared constants.
            #[must_use]
            fn remapped(self, lit_map: &[usize]) -> Self {
                Self(remap(lit_map, self.0))
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
    #[must_use]
    fn shifted(self, by: DefShift) -> Self {
        Self(self.0 + by.0)
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
/// - core: [`Schema::expected`], [`Schema::error_code`], [`Schema::shifted`],
///   [`Schema::resolve_self`], [`Schema::occurs_unguarded`],
///   [`Schema::simplify`];
/// - bindings (`valgebra-py`): the single `member` membership walk (which
///   decides membership and, in explain mode, aggregates the violation) plus
///   `render`.
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
    /// Denotes lists or tuples whose element sequence matches a regular
    /// expression over element schemas.
    ///
    /// One node subsumes the homogeneous `list[T]`/`tuple[T, ...]`, the fixed
    /// `[A, B]`/`tuple[A, B]`, and the prefix-plus-tail forms. Regular languages
    /// are closed under union, intersection, and complement, so a sequence type
    /// is a first-class member of the Boolean algebra rather than four ad-hoc,
    /// non-composable nodes.
    Seq {
        /// Whether the value is a list or a tuple.
        container: SeqKind,
        /// The regular expression over element schemas the sequence must match.
        regex: SeqRegex,
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
        /// Ordered `(key-schema, value-schema)` clauses governing every key that
        /// is not a declared field name.
        defaults: Vec<(Schema, Schema)>,
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

/// A regular expression over element schemas, the body of a [`Schema::Seq`].
///
/// A value's element sequence is a member iff it is in the regular language this
/// expression denotes, where a single element symbol "matches" `Elem(s)` when the
/// element belongs to `s`. The homogeneous form is `Star(Elem(t))`, the fixed
/// form is `Cat([Elem(a), Elem(b), ...])`, and the prefix-plus-tail form appends a
/// trailing `Star`. `Or` and nesting are produced only by the decision procedure
/// (closure under the Boolean operations); the frontend emits linear shapes only,
/// which [`SeqRegex::linear`] recognizes for the membership walk.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SeqRegex {
    /// The empty sequence.
    Empty,
    /// A single element belonging to the schema.
    Elem(Box<Schema>),
    /// Concatenation: each part in order.
    Cat(Vec<SeqRegex>),
    /// Alternation: any one branch.
    Or(Vec<SeqRegex>),
    /// Zero or more repetitions.
    Star(Box<SeqRegex>),
}

impl SeqRegex {
    /// Map every element schema through `f`, preserving the regex structure.
    pub(crate) fn map_elems(&self, f: &impl Fn(&Schema) -> Schema) -> SeqRegex {
        match self {
            SeqRegex::Empty => SeqRegex::Empty,
            SeqRegex::Elem(s) => SeqRegex::Elem(Box::new(f(s))),
            SeqRegex::Cat(parts) => SeqRegex::Cat(parts.iter().map(|p| p.map_elems(f)).collect()),
            SeqRegex::Or(parts) => SeqRegex::Or(parts.iter().map(|p| p.map_elems(f)).collect()),
            SeqRegex::Star(inner) => SeqRegex::Star(Box::new(inner.map_elems(f))),
        }
    }

    /// Whether any element schema satisfies `pred`.
    fn any_elem(&self, pred: &impl Fn(&Schema) -> bool) -> bool {
        match self {
            SeqRegex::Empty => false,
            SeqRegex::Elem(s) => pred(s),
            SeqRegex::Cat(parts) | SeqRegex::Or(parts) => parts.iter().any(|p| p.any_elem(pred)),
            SeqRegex::Star(inner) => inner.any_elem(pred),
        }
    }

    fn shifted(&self, pool: PoolShift, defs: DefShift) -> SeqRegex {
        self.map_elems(&|s| s.shifted(pool, defs))
    }

    fn reindexed(&self, lit_map: &[usize], def_offset: DefShift) -> SeqRegex {
        self.map_elems(&|s| s.reindexed(lit_map, def_offset))
    }

    fn resolve_self(&self, token: u64, ref_id: DefIx) -> SeqRegex {
        self.map_elems(&|s| s.resolve_self(token, ref_id))
    }

    fn with_records_open(&self, open: Openness) -> SeqRegex {
        self.map_elems(&|s| s.with_records_open(open))
    }

    /// A `Seq` guards its element schemas, so a recursive reference inside one is
    /// guarded; report whether `target` occurs (necessarily guarded here).
    fn occurs_guarded(&self, target: DefIx) -> bool {
        self.any_elem(&|s| s.occurs_unguarded(target, Guarded::Yes))
    }

    /// The structural nesting depth this regex contributes: one level per regex
    /// constructor plus the depth of the deepest element schema. The frontend's
    /// linear shapes are shallow; the count still tracks every native stack frame
    /// a recursive walk descends through the regex.
    fn depth(&self) -> usize {
        match self {
            SeqRegex::Empty => 0,
            SeqRegex::Elem(s) => s.depth(),
            SeqRegex::Cat(parts) | SeqRegex::Or(parts) => {
                1 + parts.iter().map(SeqRegex::depth).max().unwrap_or(0)
            }
            SeqRegex::Star(inner) => 1 + inner.depth(),
        }
    }

    /// The number of schema nodes this regex contributes, counting each element
    /// schema's whole subtree. Mirrors [`Schema::node_count`] for the sequence
    /// body; a regex constructor itself is not a schema node.
    fn node_count(&self) -> usize {
        match self {
            SeqRegex::Empty => 0,
            SeqRegex::Elem(s) => s.node_count(),
            SeqRegex::Cat(parts) | SeqRegex::Or(parts) => {
                parts.iter().map(SeqRegex::node_count).sum()
            }
            SeqRegex::Star(inner) => inner.node_count(),
        }
    }

    /// If this regex is a *linear* sequence — a fixed prefix of element schemas
    /// followed by an optional repeated tail element — return `(prefix, tail)`.
    ///
    /// The frontend's forms are all linear: homogeneous (`Star(Elem)`), fixed
    /// (`Cat` of `Elem`s), and prefix-plus-tail (`Cat` of `Elem`s ending in
    /// `Star(Elem)`). `Or` and nested forms, built only inside the decision
    /// procedure, are not linear and never reach value membership.
    #[must_use]
    pub fn linear(&self) -> Option<(Vec<&Schema>, Option<&Schema>)> {
        match self {
            SeqRegex::Empty => Some((Vec::new(), None)),
            SeqRegex::Elem(s) => Some((vec![s.as_ref()], None)),
            SeqRegex::Star(inner) => match inner.as_ref() {
                SeqRegex::Elem(s) => Some((Vec::new(), Some(s.as_ref()))),
                _ => None,
            },
            SeqRegex::Cat(parts) => {
                let mut prefix = Vec::new();
                let mut tail = None;
                for (i, part) in parts.iter().enumerate() {
                    match part {
                        SeqRegex::Elem(s) => prefix.push(s.as_ref()),
                        SeqRegex::Star(inner) if i + 1 == parts.len() => match inner.as_ref() {
                            SeqRegex::Elem(s) => tail = Some(s.as_ref()),
                            _ => return None,
                        },
                        _ => return None,
                    }
                }
                Some((prefix, tail))
            }
            SeqRegex::Or(_) => None,
        }
    }
}

impl Schema {
    /// A list whose element sequence matches `regex`.
    #[must_use]
    pub fn list(regex: SeqRegex) -> Schema {
        Schema::Seq {
            container: SeqKind::List,
            regex,
        }
    }

    /// A tuple whose element sequence matches `regex`.
    #[must_use]
    pub fn tuple(regex: SeqRegex) -> Schema {
        Schema::Seq {
            container: SeqKind::Tuple,
            regex,
        }
    }

    /// A homogeneous mapping `dict[K, V]`: every key in `key`, every value in
    /// `value`.
    #[must_use]
    pub fn mapping(key: Schema, value: Schema) -> Schema {
        Schema::KeyedMap {
            fields: Vec::new(),
            defaults: vec![(key, value)],
        }
    }

    /// A record of named fields, closed (`open` false) or open (`open` true). An
    /// open record admits any other key; a closed one admits none.
    #[must_use]
    pub fn record(fields: Vec<Field>, open: Openness) -> Schema {
        let defaults = if open == Openness::Open {
            vec![(Schema::Anything, Schema::Anything)]
        } else {
            Vec::new()
        };
        Schema::KeyedMap { fields, defaults }
    }
}

impl SeqRegex {
    /// The homogeneous form `Star(Elem(element))`: any number of `element`s.
    #[must_use]
    pub fn homogeneous(element: Schema) -> SeqRegex {
        SeqRegex::Star(Box::new(SeqRegex::Elem(Box::new(element))))
    }

    /// The fixed form `Cat([Elem(e0), Elem(e1), ...])`: each element positionally.
    #[must_use]
    pub fn fixed(elements: impl IntoIterator<Item = Schema>) -> SeqRegex {
        SeqRegex::Cat(
            elements
                .into_iter()
                .map(|s| SeqRegex::Elem(Box::new(s)))
                .collect(),
        )
    }

    /// The prefix-plus-tail form `Cat([Elem(p0), ..., Star(Elem(tail))])`: a fixed
    /// positional prefix, then zero or more elements matching `tail`.
    #[must_use]
    pub fn prefix_tail(prefix: impl IntoIterator<Item = Schema>, tail: Schema) -> SeqRegex {
        let mut parts: Vec<SeqRegex> = prefix
            .into_iter()
            .map(|s| SeqRegex::Elem(Box::new(s)))
            .collect();
        parts.push(SeqRegex::Star(Box::new(SeqRegex::Elem(Box::new(tail)))));
        SeqRegex::Cat(parts)
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

    /// Return a copy with pool indices shifted by `pool` and definition
    /// references shifted by `defs`.
    ///
    /// Used when composing two compiled validators: their constants pools and
    /// definitions tables are concatenated, so the second schema's
    /// `Literal`/`Instance`/`Attrs`/`Refine` indices move past the first
    /// pool's length and its `Ref` indices past the first definitions' length.
    #[must_use]
    pub fn shifted(&self, pool: PoolShift, defs: DefShift) -> Schema {
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
            | Schema::SelfRef(_) => self.clone(),
            Schema::Literal(i) => Schema::Literal(i.shifted(pool)),
            Schema::Instance(i) => Schema::Instance(i.shifted(pool)),
            Schema::Ref(i) => Schema::Ref(i.shifted(defs)),
            Schema::Seq { container, regex } => Schema::Seq {
                container: *container,
                regex: regex.shifted(pool, defs),
            },
            Schema::Set(e) => Schema::Set(Box::new(e.shifted(pool, defs))),
            Schema::FrozenSet(e) => Schema::FrozenSet(Box::new(e.shifted(pool, defs))),
            Schema::Complement(e) => Schema::Complement(Box::new(e.shifted(pool, defs))),
            Schema::Union(es) => Schema::Union(es.iter().map(|s| s.shifted(pool, defs)).collect()),
            Schema::Intersection(es) => {
                Schema::Intersection(es.iter().map(|s| s.shifted(pool, defs)).collect())
            }
            Schema::KeyedMap { fields, defaults } => Schema::KeyedMap {
                fields: fields.iter().map(|f| f.shifted(pool, defs)).collect(),
                defaults: defaults
                    .iter()
                    .map(|(k, v)| (k.shifted(pool, defs), v.shifted(pool, defs)))
                    .collect(),
            },
            Schema::Attrs {
                class_index,
                fields,
            } => Schema::Attrs {
                class_index: class_index.shifted(pool),
                fields: fields.iter().map(|f| f.shifted(pool, defs)).collect(),
            },
            Schema::Refine { base, constraints } => Schema::Refine {
                base: Box::new(base.shifted(pool, defs)),
                constraints: constraints.iter().map(|c| c.shifted(pool)).collect(),
            },
        }
    }

    /// The structural nesting depth of this schema: the longest chain of nested
    /// constructors from here down to a leaf. Leaves — the scalars, `Literal`,
    /// `Instance`, `Ref`, and `SelfRef` — have depth 1; every constructor is one
    /// more than the deepest schema it contains.
    ///
    /// A `Ref` counts as a leaf even though it names a recursive definition: the
    /// back edge is not followed, so the depth of a recursive schema is finite
    /// and this terminates. The count mirrors the native stack a recursive walk
    /// over the tree descends — one frame per level in clone, drop, the decision
    /// procedure, and the render back to an annotation — so composition can be
    /// bounded to a depth every such walk survives.
    #[must_use]
    pub fn depth(&self) -> usize {
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
            | Schema::SelfRef(_) => 1,
            Schema::Seq { regex, .. } => 1 + regex.depth(),
            Schema::Set(e) | Schema::FrozenSet(e) | Schema::Complement(e) => 1 + e.depth(),
            Schema::Union(es) | Schema::Intersection(es) => {
                1 + es.iter().map(Schema::depth).max().unwrap_or(0)
            }
            Schema::KeyedMap { fields, defaults } => {
                let fields_depth = fields.iter().map(|f| f.schema.depth()).max().unwrap_or(0);
                let defaults_depth = defaults
                    .iter()
                    .map(|(k, v)| k.depth().max(v.depth()))
                    .max()
                    .unwrap_or(0);
                1 + fields_depth.max(defaults_depth)
            }
            Schema::Attrs { fields, .. } => {
                1 + fields.iter().map(|f| f.schema.depth()).max().unwrap_or(0)
            }
            Schema::Refine { base, .. } => 1 + base.depth(),
        }
    }

    /// The total number of schema nodes in this tree, counting this node plus
    /// every node in its children. A `Ref` back edge is one node — the
    /// definition it points at is counted where it lives in the definitions
    /// table, not re-counted through the edge — so the count of a recursive
    /// schema is finite. Combined with a per-tree depth bound, a total-node
    /// bound rejects a schema that is shallow but exponentially wide (a doubling
    /// union) without rejecting a legitimately deep or wide one.
    #[must_use]
    pub fn node_count(&self) -> usize {
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
            | Schema::SelfRef(_) => 1,
            Schema::Seq { regex, .. } => 1 + regex.node_count(),
            Schema::Set(e) | Schema::FrozenSet(e) | Schema::Complement(e) => 1 + e.node_count(),
            Schema::Union(es) | Schema::Intersection(es) => {
                1 + es.iter().map(Schema::node_count).sum::<usize>()
            }
            Schema::KeyedMap { fields, defaults } => {
                let fields_nodes: usize = fields.iter().map(|f| f.schema.node_count()).sum();
                let defaults_nodes: usize = defaults
                    .iter()
                    .map(|(k, v)| k.node_count() + v.node_count())
                    .sum();
                1 + fields_nodes + defaults_nodes
            }
            Schema::Attrs { fields, .. } => {
                1 + fields.iter().map(|f| f.schema.node_count()).sum::<usize>()
            }
            Schema::Refine { base, constraints } => 1 + base.node_count() + constraints.len(),
        }
    }

    /// Like [`shifted`](Self::shifted), but remapping pool indices through
    /// `lit_map` (an old→new table from interning one pool into another, so
    /// identity-shared constants collapse to one index) while still offsetting
    /// definition indices by `def_offset`.
    #[must_use]
    pub fn reindexed(&self, lit_map: &[usize], def_offset: DefShift) -> Schema {
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
            | Schema::SelfRef(_) => self.clone(),
            Schema::Literal(i) => Schema::Literal(i.remapped(lit_map)),
            Schema::Instance(i) => Schema::Instance(i.remapped(lit_map)),
            Schema::Ref(i) => Schema::Ref(i.shifted(def_offset)),
            Schema::Seq { container, regex } => Schema::Seq {
                container: *container,
                regex: regex.reindexed(lit_map, def_offset),
            },
            Schema::Set(e) => Schema::Set(Box::new(e.reindexed(lit_map, def_offset))),
            Schema::FrozenSet(e) => Schema::FrozenSet(Box::new(e.reindexed(lit_map, def_offset))),
            Schema::Complement(e) => Schema::Complement(Box::new(e.reindexed(lit_map, def_offset))),
            Schema::Union(es) => Schema::Union(
                es.iter()
                    .map(|s| s.reindexed(lit_map, def_offset))
                    .collect(),
            ),
            Schema::Intersection(es) => Schema::Intersection(
                es.iter()
                    .map(|s| s.reindexed(lit_map, def_offset))
                    .collect(),
            ),
            Schema::KeyedMap { fields, defaults } => Schema::KeyedMap {
                fields: fields
                    .iter()
                    .map(|f| f.reindexed(lit_map, def_offset))
                    .collect(),
                defaults: defaults
                    .iter()
                    .map(|(k, v)| {
                        (
                            k.reindexed(lit_map, def_offset),
                            v.reindexed(lit_map, def_offset),
                        )
                    })
                    .collect(),
            },
            Schema::Attrs {
                class_index,
                fields,
            } => Schema::Attrs {
                class_index: class_index.remapped(lit_map),
                fields: fields
                    .iter()
                    .map(|f| f.reindexed(lit_map, def_offset))
                    .collect(),
            },
            Schema::Refine { base, constraints } => Schema::Refine {
                base: Box::new(base.reindexed(lit_map, def_offset)),
                constraints: constraints.iter().map(|c| c.reindexed(lit_map)).collect(),
            },
        }
    }

    /// Replace each `SelfRef(token)` with `Ref(ref_id)`, leaving other tokens
    /// (from enclosing `recursive` definitions) untouched.
    #[must_use]
    pub fn resolve_self(&self, token: u64, ref_id: DefIx) -> Schema {
        let recur = |s: &Schema| s.resolve_self(token, ref_id);
        match self {
            Schema::SelfRef(t) if *t == token => Schema::Ref(ref_id),
            Schema::Seq { container, regex } => Schema::Seq {
                container: *container,
                regex: regex.resolve_self(token, ref_id),
            },
            Schema::Set(e) => Schema::Set(Box::new(recur(e))),
            Schema::FrozenSet(e) => Schema::FrozenSet(Box::new(recur(e))),
            Schema::Complement(e) => Schema::Complement(Box::new(recur(e))),
            Schema::Union(es) => Schema::Union(es.iter().map(recur).collect()),
            Schema::Intersection(es) => Schema::Intersection(es.iter().map(recur).collect()),
            Schema::KeyedMap { fields, defaults } => Schema::KeyedMap {
                fields: fields
                    .iter()
                    .map(|f| Field {
                        name: f.name.clone(),
                        schema: recur(&f.schema),
                        required: f.required,
                    })
                    .collect(),
                defaults: defaults.iter().map(|(k, v)| (recur(k), recur(v))).collect(),
            },
            Schema::Attrs {
                class_index,
                fields,
            } => Schema::Attrs {
                class_index: *class_index,
                fields: fields
                    .iter()
                    .map(|f| Field {
                        name: f.name.clone(),
                        schema: recur(&f.schema),
                        required: f.required,
                    })
                    .collect(),
            },
            Schema::Refine { base, constraints } => Schema::Refine {
                base: Box::new(recur(base)),
                constraints: constraints.clone(),
            },
            other => other.clone(),
        }
    }

    /// Whether `Ref(target)` occurs without a structural guard above it.
    ///
    /// A `recursive` definition is contractive (productive) only when every
    /// occurrence of its self-reference sits under a structural constructor;
    /// `guarded` records whether such a constructor has been crossed.
    #[must_use]
    pub fn occurs_unguarded(&self, target: DefIx, guarded: Guarded) -> bool {
        match self {
            Schema::Ref(id) => *id == target && guarded == Guarded::No,
            // Structural constructors guard their children.
            Schema::Seq { regex, .. } => regex.occurs_guarded(target),
            Schema::Set(e) | Schema::FrozenSet(e) => e.occurs_unguarded(target, Guarded::Yes),
            Schema::KeyedMap { fields, defaults } => {
                fields
                    .iter()
                    .any(|f| f.schema.occurs_unguarded(target, Guarded::Yes))
                    || defaults.iter().any(|(k, v)| {
                        k.occurs_unguarded(target, Guarded::Yes)
                            || v.occurs_unguarded(target, Guarded::Yes)
                    })
            }
            Schema::Attrs { fields, .. } => fields
                .iter()
                .any(|f| f.schema.occurs_unguarded(target, Guarded::Yes)),
            // Algebraic combinators do not guard: they pass `guarded` through.
            Schema::Union(es) | Schema::Intersection(es) => {
                es.iter().any(|s| s.occurs_unguarded(target, guarded))
            }
            Schema::Complement(e) => e.occurs_unguarded(target, guarded),
            Schema::Refine { base, .. } => base.occurs_unguarded(target, guarded),
            _ => false,
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
        let recur = |s: &Schema| s.with_records_open(open);
        let fields_open = |fields: &[Field]| -> Vec<Field> {
            fields
                .iter()
                .map(|f| Field {
                    name: f.name.clone(),
                    schema: recur(&f.schema),
                    required: f.required,
                })
                .collect()
        };
        match self {
            // A record (named fields) opens or closes its catch-all; a pure
            // mapping (no fields) is not a record, so only its clause schemas are
            // recursed.
            Schema::KeyedMap { fields, .. } if !fields.is_empty() => Schema::KeyedMap {
                fields: fields_open(fields),
                defaults: if open == Openness::Open {
                    vec![(Schema::Anything, Schema::Anything)]
                } else {
                    Vec::new()
                },
            },
            Schema::KeyedMap { defaults, .. } => Schema::KeyedMap {
                fields: Vec::new(),
                defaults: defaults.iter().map(|(k, v)| (recur(k), recur(v))).collect(),
            },
            Schema::Attrs {
                class_index,
                fields,
            } => Schema::Attrs {
                class_index: *class_index,
                fields: fields_open(fields),
            },
            Schema::Seq { container, regex } => Schema::Seq {
                container: *container,
                regex: regex.with_records_open(open),
            },
            Schema::Set(e) => Schema::Set(Box::new(recur(e))),
            Schema::FrozenSet(e) => Schema::FrozenSet(Box::new(recur(e))),
            Schema::Complement(e) => Schema::Complement(Box::new(recur(e))),
            Schema::Union(es) => Schema::Union(es.iter().map(recur).collect()),
            Schema::Intersection(es) => Schema::Intersection(es.iter().map(recur).collect()),
            Schema::Refine { base, constraints } => Schema::Refine {
                base: Box::new(recur(base)),
                constraints: constraints.clone(),
            },
            other => other.clone(),
        }
    }
}

impl Field {
    fn shifted(&self, pool: PoolShift, defs: DefShift) -> Field {
        Field {
            name: self.name.clone(),
            schema: self.schema.shifted(pool, defs),
            required: self.required,
        }
    }

    fn reindexed(&self, lit_map: &[usize], def_offset: DefShift) -> Field {
        Field {
            name: self.name.clone(),
            schema: self.schema.reindexed(lit_map, def_offset),
            required: self.required,
        }
    }
}

impl Constraint {
    fn shifted(&self, pool: PoolShift) -> Constraint {
        match self {
            Constraint::Ge(i) => Constraint::Ge(i.shifted(pool)),
            Constraint::Gt(i) => Constraint::Gt(i.shifted(pool)),
            Constraint::Le(i) => Constraint::Le(i.shifted(pool)),
            Constraint::Lt(i) => Constraint::Lt(i.shifted(pool)),
            // A length is not a pool index and takes no pool shift. The type
            // says so: a usize has no `shifted(PoolShift)`.
            Constraint::MinLen(n) => Constraint::MinLen(*n),
            Constraint::MaxLen(n) => Constraint::MaxLen(*n),
            Constraint::MultipleOf(i) => Constraint::MultipleOf(i.shifted(pool)),
            Constraint::Predicate(i) => Constraint::Predicate(i.shifted(pool)),
            Constraint::Regex(p) => Constraint::Regex(p.clone()),
        }
    }

    fn reindexed(&self, lit_map: &[usize]) -> Constraint {
        match self {
            Constraint::Ge(i) => Constraint::Ge(i.remapped(lit_map)),
            Constraint::Gt(i) => Constraint::Gt(i.remapped(lit_map)),
            Constraint::Le(i) => Constraint::Le(i.remapped(lit_map)),
            Constraint::Lt(i) => Constraint::Lt(i.remapped(lit_map)),
            Constraint::MinLen(n) => Constraint::MinLen(*n),
            Constraint::MaxLen(n) => Constraint::MaxLen(*n),
            Constraint::MultipleOf(i) => Constraint::MultipleOf(i.remapped(lit_map)),
            Constraint::Predicate(i) => Constraint::Predicate(i.remapped(lit_map)),
            Constraint::Regex(p) => Constraint::Regex(p.clone()),
        }
    }
}
