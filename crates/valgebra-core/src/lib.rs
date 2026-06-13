//! valgebra schema intermediate representation.
//!
//! A schema denotes a set of Python values; validation is membership. This
//! crate is pure Rust: it defines the IR, the denotation of every node, and the
//! structured [`Violation`] produced when membership fails. Inspecting a Python
//! object requires `PyO3`, so the validator walk itself lives in the bindings
//! crate; this crate is the stable, language-agnostic core.

use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};

/// Fresh, process-unique tokens for the transient [`Schema::SelfRef`] marker, so
/// nested `lazy` definitions never resolve each other's self-references.
static NEXT_SELF_TOKEN: AtomicU64 = AtomicU64::new(0);

/// Allocate a fresh self-reference token for a `lazy` definition.
#[must_use]
pub fn fresh_self_token() -> u64 {
    NEXT_SELF_TOKEN.fetch_add(1, Ordering::Relaxed)
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
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Schema {
    /// Top. Denotes every Python value; membership always holds.
    Anything,
    /// The gradual dynamic type. At runtime it admits every value like the top,
    /// but it is a distinct atom: the simplifier must not rewrite it by the
    /// lattice laws, so `Any` and [`Schema::Anything`] are kept separate.
    Any,
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
    Literal(usize),
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
    /// open (lax) record a single `(Anything, Anything)` clause, `dict[K, V]` a
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
    Instance(usize),
    /// Denotes instances of a class whose attributes satisfy the given fields.
    ///
    /// `isinstance` against the pooled class at `class_index` must hold, and
    /// every field's attribute must be present and match. This is the deep
    /// check for dataclasses and named tuples.
    Object {
        /// Index of the class in the validator's object pool.
        class_index: usize,
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
    /// edge of a fixpoint, produced by `lazy`.
    Ref(usize),
    /// A transient self-reference marker used only while a `lazy` definition is
    /// being built; it is resolved to a [`Schema::Ref`] before the validator is
    /// returned and never appears in a finished schema.
    SelfRef(u64),
}

/// Whether a [`Schema::Seq`] denotes lists or tuples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
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
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
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
    fn map_elems(&self, f: &impl Fn(&Schema) -> Schema) -> SeqRegex {
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

    /// Whether the regex matches **no** sequence at all — its language is empty.
    /// `Empty` and `Star` always match the empty sequence, so they are never
    /// empty; a single element is empty when its schema is; a concatenation is
    /// empty when any part is (every part must be matchable); an alternation is
    /// empty only when every alternative is.
    fn language_is_empty(&self, defs: &[Schema], visiting: &mut Vec<usize>) -> bool {
        match self {
            SeqRegex::Empty | SeqRegex::Star(_) => false,
            SeqRegex::Elem(schema) => schema.is_empty_rec(defs, visiting),
            SeqRegex::Cat(parts) => parts.iter().any(|p| p.language_is_empty(defs, visiting)),
            SeqRegex::Or(parts) => parts.iter().all(|p| p.language_is_empty(defs, visiting)),
        }
    }

    /// Whether every sequence this regex matches is also matched by `other` —
    /// language inclusion. Alternation distributes; everything else reduces to
    /// the prefix-and-tail [`linear`](Self::linear) form and
    /// [`linear_subtype`]. Sound throughout: a regex shape it cannot put in
    /// linear form (a nested non-trailing star) yields `false`, never a guess.
    fn regex_subtype(
        &self,
        other: &SeqRegex,
        cx: SubtypeCx<'_>,
        assumptions: &mut Vec<(Schema, Schema)>,
    ) -> bool {
        if self == other {
            return true;
        }
        // A union of languages is included iff every branch is; a language is
        // included in a union if it lands in one branch (sound, conservative —
        // it may instead split across several).
        if let SeqRegex::Or(parts) = self {
            return parts
                .iter()
                .all(|part| part.regex_subtype(other, cx, assumptions));
        }
        if let SeqRegex::Or(parts) = other {
            return parts
                .iter()
                .any(|part| self.regex_subtype(part, cx, assumptions));
        }
        match (self.linear(), other.linear()) {
            (Some((pa, ta)), Some((pb, tb))) => linear_subtype(&pa, ta, &pb, tb, cx, assumptions),
            _ => false,
        }
    }

    fn shifted(&self, pool: usize, defs: usize) -> SeqRegex {
        self.map_elems(&|s| s.shifted(pool, defs))
    }

    fn resolve_self(&self, token: u64, ref_id: usize) -> SeqRegex {
        self.map_elems(&|s| s.resolve_self(token, ref_id))
    }

    fn with_records_open(&self, open: bool) -> SeqRegex {
        self.map_elems(&|s| s.with_records_open(open))
    }

    fn simplify(&self) -> SeqRegex {
        self.map_elems(&Schema::simplify)
    }

    /// A `Seq` guards its element schemas, so a recursive reference inside one is
    /// guarded; report whether `target` occurs (necessarily guarded here).
    fn occurs_guarded(&self, target: usize) -> bool {
        self.any_elem(&|s| s.occurs_unguarded(target, true))
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

    /// A record of named fields, closed (`open` false) or lax (`open` true). An
    /// open record admits any other key; a closed one admits none.
    #[must_use]
    pub fn record(fields: Vec<Field>, open: bool) -> Schema {
        let defaults = if open {
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
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Constraint {
    /// `value >= pool[i]`.
    Ge(usize),
    /// `value > pool[i]`.
    Gt(usize),
    /// `value <= pool[i]`.
    Le(usize),
    /// `value < pool[i]`.
    Lt(usize),
    /// `len(value) >= n`.
    MinLen(usize),
    /// `len(value) <= n`.
    MaxLen(usize),
    /// `value % pool[i] == 0`: a numeric multiple of the operand.
    MultipleOf(usize),
    /// `pool[i](value)` is truthy. The documented Python-callback slow path.
    Predicate(usize),
}

/// A named field of a [`Schema::KeyedMap`] or [`Schema::Object`].
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
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
            Schema::Any => "any",
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
            Schema::Object { .. } => "object",
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
            Schema::Any => "any",
            Schema::Nothing => "no_match",
            Schema::NoneType => "none_type",
            Schema::Bool => "bool_type",
            Schema::Int => "int_type",
            Schema::Float => "float_type",
            Schema::Str => "string_type",
            Schema::Bytes => "bytes_type",
            Schema::Literal(_) => "literal_value",
            Schema::Seq {
                container: SeqKind::List,
                ..
            } => "list_type",
            Schema::Seq {
                container: SeqKind::Tuple,
                ..
            } => "tuple_type",
            Schema::Set(_) => "set_type",
            Schema::FrozenSet(_) => "frozenset_type",
            Schema::KeyedMap { .. } => "dict_type",
            Schema::Union(_) => "union_error",
            Schema::Intersection(_) => "intersection_error",
            Schema::Complement(_) => "unexpected_match",
            Schema::Instance(_) | Schema::Object { .. } => "instance_type",
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
    /// `Literal`/`Instance`/`Object`/`Refine` indices move past the first
    /// pool's length and its `Ref` indices past the first definitions' length.
    #[must_use]
    pub fn shifted(&self, pool: usize, defs: usize) -> Schema {
        match self {
            Schema::Anything
            | Schema::Any
            | Schema::Nothing
            | Schema::NoneType
            | Schema::Bool
            | Schema::Int
            | Schema::Float
            | Schema::Str
            | Schema::Bytes
            | Schema::SelfRef(_) => self.clone(),
            Schema::Literal(i) => Schema::Literal(i + pool),
            Schema::Instance(i) => Schema::Instance(i + pool),
            Schema::Ref(i) => Schema::Ref(i + defs),
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
            Schema::Object {
                class_index,
                fields,
            } => Schema::Object {
                class_index: class_index + pool,
                fields: fields.iter().map(|f| f.shifted(pool, defs)).collect(),
            },
            Schema::Refine { base, constraints } => Schema::Refine {
                base: Box::new(base.shifted(pool, defs)),
                constraints: constraints.iter().map(|c| c.shifted(pool)).collect(),
            },
        }
    }

    /// Replace each `SelfRef(token)` with `Ref(ref_id)`, leaving other tokens
    /// (from enclosing `lazy` definitions) untouched.
    #[must_use]
    pub fn resolve_self(&self, token: u64, ref_id: usize) -> Schema {
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
            Schema::Object {
                class_index,
                fields,
            } => Schema::Object {
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
    /// A `lazy` definition is contractive (productive) only when every
    /// occurrence of its self-reference sits under a structural constructor;
    /// `guarded` records whether such a constructor has been crossed.
    #[must_use]
    pub fn occurs_unguarded(&self, target: usize, guarded: bool) -> bool {
        match self {
            Schema::Ref(id) => *id == target && !guarded,
            // Structural constructors guard their children.
            Schema::Seq { regex, .. } => regex.occurs_guarded(target),
            Schema::Set(e) | Schema::FrozenSet(e) => e.occurs_unguarded(target, true),
            Schema::KeyedMap { fields, defaults } => {
                fields
                    .iter()
                    .any(|f| f.schema.occurs_unguarded(target, true))
                    || defaults.iter().any(|(k, v)| {
                        k.occurs_unguarded(target, true) || v.occurs_unguarded(target, true)
                    })
            }
            Schema::Object { fields, .. } => fields
                .iter()
                .any(|f| f.schema.occurs_unguarded(target, true)),
            // Algebraic combinators do not guard: they pass `guarded` through.
            Schema::Union(es) | Schema::Intersection(es) => {
                es.iter().any(|s| s.occurs_unguarded(target, guarded))
            }
            Schema::Complement(e) => e.occurs_unguarded(target, guarded),
            Schema::Refine { base, .. } => base.occurs_unguarded(target, guarded),
            _ => false,
        }
    }

    /// Return a membership-equivalent schema reduced by the lattice laws.
    ///
    /// Every rewrite preserves the set of admitted values: nested unions and
    /// intersections are flattened, members sorted and deduplicated
    /// (associativity, commutativity, idempotence), the top and bottom
    /// identities are applied, complements are pushed inward to negation-normal
    /// form (De Morgan) and double negations cancelled. `Any` (gradual) is left
    /// untouched: it is never treated as the top. Conservative by design — it
    /// never claims an equivalence it cannot justify structurally.
    #[must_use]
    pub fn simplify(&self) -> Schema {
        match self {
            Schema::Seq { container, regex } => Schema::Seq {
                container: *container,
                regex: regex.simplify(),
            },
            Schema::Set(e) => Schema::Set(Box::new(e.simplify())),
            Schema::FrozenSet(e) => Schema::FrozenSet(Box::new(e.simplify())),
            Schema::KeyedMap { fields, defaults } => Schema::KeyedMap {
                fields: fields
                    .iter()
                    .map(|f| Field {
                        name: f.name.clone(),
                        schema: f.schema.simplify(),
                        required: f.required,
                    })
                    .collect(),
                defaults: defaults
                    .iter()
                    .map(|(k, v)| (k.simplify(), v.simplify()))
                    .collect(),
            },
            Schema::Object {
                class_index,
                fields,
            } => Schema::Object {
                class_index: *class_index,
                fields: fields
                    .iter()
                    .map(|f| Field {
                        name: f.name.clone(),
                        schema: f.schema.simplify(),
                        required: f.required,
                    })
                    .collect(),
            },
            Schema::Refine { base, constraints } => Schema::Refine {
                base: Box::new(base.simplify()),
                constraints: constraints.clone(),
            },
            Schema::Union(members) => simplify_union(members),
            Schema::Intersection(members) => simplify_intersection(members),
            Schema::Complement(inner) => simplify_complement(inner),
            // Atoms (including Any and Literal/Instance) reduce to themselves.
            other => other.clone(),
        }
    }

    /// Return a copy with every record-shaped [`Schema::KeyedMap`] in the tree
    /// set to `open`.
    ///
    /// This backs the `lax`/`strict` wrappers: `lax` opens every record in a
    /// subtree (undeclared keys allowed via an `anything` catch-all), `strict`
    /// closes them. A pure mapping (no named fields) is not a record and keeps
    /// its clauses.
    #[must_use]
    pub fn with_records_open(&self, open: bool) -> Schema {
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
                defaults: if open {
                    vec![(Schema::Anything, Schema::Anything)]
                } else {
                    Vec::new()
                },
            },
            Schema::KeyedMap { defaults, .. } => Schema::KeyedMap {
                fields: Vec::new(),
                defaults: defaults.iter().map(|(k, v)| (recur(k), recur(v))).collect(),
            },
            Schema::Object {
                class_index,
                fields,
            } => Schema::Object {
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

    /// Whether this schema and `other` are *provably* disjoint: no value belongs
    /// to both. Sound, not complete — it returns true only when the concrete
    /// types cannot overlap (distinct builtin scalars, distinct container kinds,
    /// a refinement's base versus another), and false (conservatively) for the
    /// cases it cannot decide in the core: `Literal` and `Instance` (a class may
    /// subclass a builtin), `Any`, references, and combinators.
    #[must_use]
    pub fn disjoint(&self, other: &Schema) -> bool {
        if matches!(self, Schema::Nothing) || matches!(other, Schema::Nothing) {
            return true;
        }
        match (self.type_tag(), other.type_tag()) {
            // Distinct concrete types are disjoint, except bool ⊆ int.
            (Some(a), Some(b)) => {
                a != b
                    && !matches!(
                        (a, b),
                        (TypeTag::Bool, TypeTag::Int) | (TypeTag::Int, TypeTag::Bool)
                    )
            }
            _ => false,
        }
    }

    /// A concrete type tag for nodes whose disjointness the core can decide
    /// soundly. `None` for nodes it cannot (`Literal`/`Instance`/`Any`/...).
    fn type_tag(&self) -> Option<TypeTag> {
        Some(match self {
            Schema::NoneType => TypeTag::NoneType,
            Schema::Bool => TypeTag::Bool,
            Schema::Int => TypeTag::Int,
            Schema::Float => TypeTag::Float,
            Schema::Str => TypeTag::Str,
            Schema::Bytes => TypeTag::Bytes,
            Schema::Seq {
                container: SeqKind::List,
                ..
            } => TypeTag::List,
            Schema::Seq {
                container: SeqKind::Tuple,
                ..
            } => TypeTag::Tuple,
            Schema::Set(_) => TypeTag::Set,
            Schema::FrozenSet(_) => TypeTag::FrozenSet,
            Schema::KeyedMap { .. } => TypeTag::Dict,
            // A refinement is a subset of its base, so its base's disjointness
            // is sound for it.
            Schema::Refine { base, .. } => return base.type_tag(),
            _ => return None,
        })
    }

    /// The value-universe regions this schema denotes, as a bitset over the
    /// `REGION_*` partition, or `None` when the schema is not *scalar-decidable*
    /// — built only from the scalar atoms, `Nothing`, `Anything`, and the
    /// `Union`/`Intersection`/`Complement` combinators. On that fragment the
    /// bitset is exact, so emptiness and subtyping are decided completely;
    /// elsewhere the caller stays conservative. The gradual `Any`, literals,
    /// instances, refinements, content-bearing containers, and references are
    /// not scalar-decidable, so any combination containing one yields `None`.
    fn region_set(&self) -> Option<u16> {
        Some(match self {
            Schema::Nothing => 0,
            Schema::Anything => REGION_ALL,
            Schema::NoneType => REGION_NONE,
            Schema::Bool => REGION_BOOL,
            Schema::Int => REGION_BOOL | REGION_INT, // bool ⊆ int
            Schema::Float => REGION_FLOAT,
            Schema::Str => REGION_STR,
            Schema::Bytes => REGION_BYTES,
            Schema::Union(members) => {
                let mut acc = 0;
                for member in members {
                    acc |= member.region_set()?;
                }
                acc
            }
            Schema::Intersection(members) => {
                let mut acc = REGION_ALL;
                for member in members {
                    acc &= member.region_set()?;
                }
                acc
            }
            Schema::Complement(inner) => REGION_ALL & !inner.region_set()?,
            _ => return None,
        })
    }

    /// Whether this schema is provably empty — denotes no value. Complete on the
    /// scalar fragment (every Boolean combination of scalar atoms) and on the
    /// structural fragment reached here — a sequence whose regex matches no
    /// sequence, a keyed map with an impossible required field, and a union of
    /// empties — and sound everywhere else: it never reports a non-empty schema
    /// as empty. A set or frozenset is never empty (the empty collection is
    /// always a member). The gradual `Any`, instances, literals, refinements,
    /// and unresolved recursive references are not decided, so a combination
    /// containing one is never reported empty. To resolve recursive references,
    /// use [`is_empty_under`](Self::is_empty_under).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.is_empty_under(&[])
    }

    /// Like [`is_empty`](Self::is_empty), but resolving recursive references
    /// through `defs`, so an uninhabited recursive schema — a mandatory
    /// self-reference with no base case — is detected. A reference `defs` does
    /// not resolve stays conservative (never reported empty).
    #[must_use]
    pub fn is_empty_under(&self, defs: &[Schema]) -> bool {
        self.is_empty_rec(defs, &mut Vec::new())
    }

    fn is_empty_rec(&self, defs: &[Schema], visiting: &mut Vec<usize>) -> bool {
        match self {
            Schema::Ref(id) => {
                // A reference reached again while resolving it is a cycle: this
                // occurrence demands an infinite unfolding, so on its own it has
                // no finite inhabitant. A union base case or an optional or
                // starred position escapes before reaching here.
                if visiting.contains(id) {
                    return true;
                }
                match defs.get(*id) {
                    Some(def) => {
                        visiting.push(*id);
                        let empty = def.is_empty_rec(defs, visiting);
                        visiting.pop();
                        empty
                    }
                    None => false,
                }
            }
            Schema::Seq { regex, .. } => regex.language_is_empty(defs, visiting),
            Schema::Set(_) | Schema::FrozenSet(_) => false,
            Schema::KeyedMap { fields, .. } => fields
                .iter()
                .any(|field| field.required && field.schema.is_empty_rec(defs, visiting)),
            Schema::Union(members) => members.iter().all(|m| m.is_empty_rec(defs, visiting)),
            // Scalars and their Boolean combinations decide via the region set;
            // a combination with an opaque leaf yields `None`, hence not empty.
            _ => matches!(self.region_set(), Some(0)),
        }
    }

    /// Whether every value of `self` is also a value of `other` — set inclusion,
    /// the semantic-subtyping relation. Complete on the scalar fragment via
    /// `self ∧ ¬other = ∅`, and decided structurally past it by recursion on
    /// matching constructors (the lattice rules, set/frozenset element
    /// inclusion, and sequence inclusion on the prefix-and-tail form). Every
    /// rule is **sound** — it never reports a subtype it cannot justify — and
    /// conservative where it cannot decide (`Or` regexes, recursive references,
    /// instances, literals): there it returns `false` rather than guess.
    #[must_use]
    pub fn is_subtype(&self, other: &Schema) -> bool {
        self.is_subtype_under(other, &NoLeafRelations, &[])
    }

    /// [`is_subtype`](Self::is_subtype) with a [`LeafRelations`] oracle deciding
    /// the leaf relations the structural rules cannot (an `Instance` class or a
    /// `Literal` value), and the `defs` that resolve recursive references so
    /// subtyping is decided between recursive schemas too. The oracle's `None`
    /// and an unresolved reference both keep the conservative `false`.
    #[must_use]
    pub fn is_subtype_under(
        &self,
        other: &Schema,
        oracle: &dyn LeafRelations,
        defs: &[Schema],
    ) -> bool {
        self.is_subtype_rec(other, SubtypeCx { oracle, defs }, &mut Vec::new())
    }

    fn is_subtype_rec(
        &self,
        other: &Schema,
        cx: SubtypeCx<'_>,
        assumptions: &mut Vec<(Schema, Schema)>,
    ) -> bool {
        // Coinductive hypothesis: a goal already being proven on this path is
        // assumed to hold, so two recursive types are compared at their greatest
        // fixpoint rather than unfolded forever.
        if assumptions.iter().any(|(a, b)| a == self && b == other) {
            return true;
        }
        // Scalar fragment: exact via the region partition.
        if let (Some(a), Some(b)) = (self.region_set(), other.region_set()) {
            return a & !b & REGION_ALL == 0;
        }
        if self == other {
            return true;
        }
        match (self, other) {
            // ∅ is a subset of every set, and every set is a subset of the top.
            (Schema::Nothing, _) | (_, Schema::Anything) => true,
            (_, Schema::Nothing) => self.is_empty(), // A ⊆ ∅ exactly when A is empty
            // Unfold a recursive reference, recording the goal so a cycle back to
            // it is caught by the coinductive hypothesis above.
            (Schema::Ref(id), _) => match cx.defs.get(*id) {
                Some(def) => {
                    assumptions.push((self.clone(), other.clone()));
                    let holds = def.is_subtype_rec(other, cx, assumptions);
                    assumptions.pop();
                    holds
                }
                None => false,
            },
            (_, Schema::Ref(id)) => match cx.defs.get(*id) {
                Some(def) => {
                    assumptions.push((self.clone(), other.clone()));
                    let holds = self.is_subtype_rec(def, cx, assumptions);
                    assumptions.pop();
                    holds
                }
                None => false,
            },
            // (X ∪ Y) ⊆ Z iff X ⊆ Z and Y ⊆ Z; A ⊆ (Y ∩ Z) iff A ⊆ Y and A ⊆ Z.
            (Schema::Union(members), _) => members
                .iter()
                .all(|m| m.is_subtype_rec(other, cx, assumptions)),
            (_, Schema::Intersection(members)) => members
                .iter()
                .all(|m| self.is_subtype_rec(m, cx, assumptions)),
            // Sound one-directional rules for the remaining lattice shapes.
            (_, Schema::Union(members)) => members
                .iter()
                .any(|m| self.is_subtype_rec(m, cx, assumptions)),
            (Schema::Intersection(members), _) => members
                .iter()
                .any(|m| m.is_subtype_rec(other, cx, assumptions)),
            // Set and frozenset inclusion reduces to element inclusion.
            (Schema::Set(a), Schema::Set(b)) | (Schema::FrozenSet(a), Schema::FrozenSet(b)) => {
                a.is_subtype_rec(b, cx, assumptions)
            }
            // Same-kind sequence inclusion is language inclusion on the regexes.
            (
                Schema::Seq {
                    container: ka,
                    regex: ra,
                },
                Schema::Seq {
                    container: kb,
                    regex: rb,
                },
            ) if ka == kb => ra.regex_subtype(rb, cx, assumptions),
            // Record and mapping inclusion.
            (
                Schema::KeyedMap {
                    fields: fa,
                    defaults: da,
                },
                Schema::KeyedMap {
                    fields: fb,
                    defaults: db,
                },
            ) => keyed_map_subtype(fa, da, fb, db, cx, assumptions),
            // A leaf the structural rules cannot relate (an instance or literal):
            // defer to the oracle, conservative when it declines.
            _ => cx.oracle.leaf_subtype(self, other).unwrap_or(false),
        }
    }

    /// Whether `self` and `other` denote the same set — mutual inclusion.
    #[must_use]
    pub fn equivalent(&self, other: &Schema) -> bool {
        self.equivalent_under(other, &NoLeafRelations, &[])
    }

    /// [`equivalent`](Self::equivalent) under a [`LeafRelations`] oracle and the
    /// recursive definitions.
    #[must_use]
    pub fn equivalent_under(
        &self,
        other: &Schema,
        oracle: &dyn LeafRelations,
        defs: &[Schema],
    ) -> bool {
        self.is_subtype_under(other, oracle, defs) && other.is_subtype_under(self, oracle, defs)
    }
}

/// Threaded state for the subtyping decision: the leaf-relation oracle and the
/// definitions that resolve recursive references.
#[derive(Clone, Copy)]
struct SubtypeCx<'a> {
    oracle: &'a dyn LeafRelations,
    defs: &'a [Schema],
}

/// Resolves the leaf relations the structural subtyping decision cannot: those
/// that depend on the Python class hierarchy (an `Instance`) or on a concrete
/// value (a `Literal`). The bindings implement it with `issubclass` and
/// membership; the core defaults to [`NoLeafRelations`].
pub trait LeafRelations {
    /// Whether leaf schema `sub` is a subtype of `sup`, or `None` to leave the
    /// relation conservatively undecided.
    fn leaf_subtype(&self, sub: &Schema, sup: &Schema) -> Option<bool>;
}

/// The trivial [`LeafRelations`] that decides nothing — the core default, under
/// which `Instance` and `Literal` relations stay conservative.
pub struct NoLeafRelations;

impl LeafRelations for NoLeafRelations {
    fn leaf_subtype(&self, _sub: &Schema, _sup: &Schema) -> Option<bool> {
        None
    }
}

/// Whether the linear language `pa · ta*` is included in `pb · tb*` — a fixed
/// prefix optionally followed by a repeated tail, the shape
/// [`SeqRegex::linear`] returns. `ta`/`tb` of `None` mean no repeated tail.
fn linear_subtype(
    pa: &[&Schema],
    ta: Option<&Schema>,
    pb: &[&Schema],
    tb: Option<&Schema>,
    cx: SubtypeCx<'_>,
    assumptions: &mut Vec<(Schema, Schema)>,
) -> bool {
    // A repeated tail with an empty element language never repeats, so the left
    // side is then just its fixed prefix.
    let ta = ta.filter(|element| !element.is_empty());
    // A's fixed prefix must align with B: against B's prefix where they overlap,
    // then against B's repeated tail past it (which B must therefore have).
    let prefix_aligns = pa.len() >= pb.len()
        && pa.iter().enumerate().all(|(i, element)| match pb.get(i) {
            Some(expected) => element.is_subtype_rec(expected, cx, assumptions),
            None => tb.is_some_and(|tail| element.is_subtype_rec(tail, cx, assumptions)),
        });
    match (ta, tb) {
        (None, None) => pa.len() == pb.len() && prefix_aligns,
        (None, Some(_)) => prefix_aligns,
        // A repeats without bound but B is finite-length: impossible.
        (Some(_), None) => false,
        // A's repeated element must also land in B's repeated tail.
        (Some(a), Some(tail)) => prefix_aligns && a.is_subtype_rec(tail, cx, assumptions),
    }
}

/// Whether keyed-map `a` (fields `fa`, default clauses `da`) is a subtype of
/// keyed-map `b`. For two closed records (no defaults) it holds by width and
/// depth — every field of `a` is a field of `b` with a subtype schema — and by
/// required-ness — every field `b` requires is required in `a`. For two pure
/// mappings (no fields, one default clause each) the key and value schemas
/// covary. Every other shape is conservative.
fn keyed_map_subtype(
    fa: &[Field],
    da: &[(Schema, Schema)],
    fb: &[Field],
    db: &[(Schema, Schema)],
    cx: SubtypeCx<'_>,
    assumptions: &mut Vec<(Schema, Schema)>,
) -> bool {
    if da.is_empty() && db.is_empty() {
        let width_and_depth = fa.iter().all(|a| {
            fb.iter()
                .find(|b| b.name == a.name)
                .is_some_and(|b| a.schema.is_subtype_rec(&b.schema, cx, assumptions))
        });
        let required = fb
            .iter()
            .filter(|b| b.required)
            .all(|b| fa.iter().any(|a| a.name == b.name && a.required));
        width_and_depth && required
    } else if fa.is_empty() && fb.is_empty() && da.len() == 1 && db.len() == 1 {
        da[0].0.is_subtype_rec(&db[0].0, cx, assumptions)
            && da[0].1.is_subtype_rec(&db[0].1, cx, assumptions)
    } else {
        false
    }
}

/// The value universe partitioned into mutually-disjoint regions, so a Boolean
/// combination of scalar atoms denotes a set computed by bitset operations. The
/// scalar atoms occupy `NONE`..`BYTES` (with `int` covering `BOOL | INT`); the
/// container kinds and `OTHER` complete the partition so a complement of a
/// scalar correctly includes every non-scalar value.
const REGION_NONE: u16 = 1 << 0;
const REGION_BOOL: u16 = 1 << 1;
const REGION_INT: u16 = 1 << 2; // int values other than bool
const REGION_FLOAT: u16 = 1 << 3;
const REGION_STR: u16 = 1 << 4;
const REGION_BYTES: u16 = 1 << 5;
const REGION_ALL: u16 = (1 << 12) - 1; // 6 scalar + 5 container kinds + OTHER

/// A concrete runtime type, for the sound fragment of disjointness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TypeTag {
    NoneType,
    Bool,
    Int,
    Float,
    Str,
    Bytes,
    List,
    Tuple,
    Set,
    FrozenSet,
    Dict,
}

/// Whether the members contain a schema and its complement — `X` and `¬X` — so
/// the intersection is empty (`X ∩ ¬X = ⊥`) or the union is everything
/// (`X ∪ ¬X = ⊤`). The gradual `Any` is excluded: `Any ∩ ¬Any` must not
/// collapse, preserving "deliberately unchecked".
fn has_complementary_pair(members: &[Schema]) -> bool {
    members.iter().any(|member| {
        if let Schema::Complement(inner) = member {
            !matches!(**inner, Schema::Any) && members.iter().any(|other| other == &**inner)
        } else {
            false
        }
    })
}

/// Whether two members are provably disjoint, so their intersection is empty.
fn has_disjoint_pair(members: &[Schema]) -> bool {
    members
        .iter()
        .enumerate()
        .any(|(i, a)| members[i + 1..].iter().any(|b| a.disjoint(b)))
}

/// Whether two members are the complements of disjoint schemas, so the union is
/// everything: `¬A ∪ ¬B = ¬(A ∩ B) = ⊤` when `A` and `B` are disjoint. This is
/// the De Morgan dual of [`has_disjoint_pair`], so the two simplifiers stay
/// consistent under negation.
fn has_disjoint_complement_pair(members: &[Schema]) -> bool {
    let inners: Vec<&Schema> = members
        .iter()
        .filter_map(|m| match m {
            Schema::Complement(inner) => Some(inner.as_ref()),
            _ => None,
        })
        .collect();
    inners
        .iter()
        .enumerate()
        .any(|(i, a)| inners[i + 1..].iter().any(|b| a.disjoint(b)))
}

/// Flatten, absorb the top, drop the bottom, dedup, and collapse a union.
fn simplify_union(members: &[Schema]) -> Schema {
    let mut flat = Vec::new();
    for member in members {
        match member.simplify() {
            Schema::Anything => return Schema::Anything,
            Schema::Nothing => {}
            Schema::Union(inner) => flat.extend(inner),
            other => flat.push(other),
        }
    }
    flat.sort();
    flat.dedup();
    // X ∪ ¬X is everything, as is ¬A ∪ ¬B for disjoint A and B; and a union is
    // everything once its scalar-decidable members alone cover every region of
    // the value universe (opaque members can only add coverage, so they are
    // ignored — keeping the decision independent of grouping).
    let covers_universe = flat
        .iter()
        .filter_map(Schema::region_set)
        .fold(0u16, |acc, regions| acc | regions)
        == REGION_ALL;
    if has_complementary_pair(&flat) || has_disjoint_complement_pair(&flat) || covers_universe {
        return Schema::Anything;
    }
    match flat.len() {
        0 => Schema::Nothing,
        1 => flat.swap_remove(0),
        _ => Schema::Union(flat),
    }
}

/// Flatten, absorb the bottom, drop the top, dedup, and collapse an intersection.
fn simplify_intersection(members: &[Schema]) -> Schema {
    let mut flat = Vec::new();
    for member in members {
        match member.simplify() {
            Schema::Nothing => return Schema::Nothing,
            Schema::Anything => {}
            Schema::Intersection(inner) => flat.extend(inner),
            other => flat.push(other),
        }
    }
    flat.sort();
    flat.dedup();
    // X ∩ ¬X is empty, as is an intersection of two provably disjoint members;
    // and an intersection is empty once its scalar-decidable members alone
    // cancel to no region (opaque members only narrow further, so they are
    // ignored — keeping the decision independent of grouping).
    let region_empty = flat
        .iter()
        .filter_map(Schema::region_set)
        .fold(REGION_ALL, |acc, regions| acc & regions)
        == 0;
    if has_complementary_pair(&flat) || has_disjoint_pair(&flat) || region_empty {
        return Schema::Nothing;
    }
    match flat.len() {
        0 => Schema::Anything,
        1 => flat.swap_remove(0),
        _ => Schema::Intersection(flat),
    }
}

/// Push a complement to negation-normal form and cancel double negation.
fn simplify_complement(inner: &Schema) -> Schema {
    match inner.simplify() {
        Schema::Complement(x) => *x,
        Schema::Anything => Schema::Nothing,
        Schema::Nothing => Schema::Anything,
        Schema::Union(members) => simplify_intersection(&complement_each(members)),
        Schema::Intersection(members) => simplify_union(&complement_each(members)),
        other => Schema::Complement(Box::new(other)),
    }
}

fn complement_each(members: Vec<Schema>) -> Vec<Schema> {
    members
        .into_iter()
        .map(|m| Schema::Complement(Box::new(m)))
        .collect()
}

impl Field {
    fn shifted(&self, pool: usize, defs: usize) -> Field {
        Field {
            name: self.name.clone(),
            schema: self.schema.shifted(pool, defs),
            required: self.required,
        }
    }
}

impl Constraint {
    fn shifted(&self, pool: usize) -> Constraint {
        match self {
            Constraint::Ge(i) => Constraint::Ge(i + pool),
            Constraint::Gt(i) => Constraint::Gt(i + pool),
            Constraint::Le(i) => Constraint::Le(i + pool),
            Constraint::Lt(i) => Constraint::Lt(i + pool),
            Constraint::MinLen(n) => Constraint::MinLen(*n),
            Constraint::MaxLen(n) => Constraint::MaxLen(*n),
            Constraint::MultipleOf(i) => Constraint::MultipleOf(i + pool),
            Constraint::Predicate(i) => Constraint::Predicate(i + pool),
        }
    }
}

/// A validation failure: a value did not belong to a schema's set.
#[derive(Debug, Clone)]
pub struct Violation {
    /// Stable, machine-readable code.
    pub code: &'static str,
    /// Location of the offending value from the validation root; empty at root.
    pub path: Vec<PathSegment>,
    /// Short label of the expected set (e.g. `int`).
    pub expected: String,
    /// Short repr-style summary of the offending value.
    pub value_summary: String,
}

impl Violation {
    /// Render the path as a location string (`name[2].id`); empty at the root.
    #[must_use]
    pub fn location(&self) -> String {
        let mut out = String::new();
        for segment in &self.path {
            match segment {
                PathSegment::Key(key) => {
                    if !out.is_empty() {
                        out.push('.');
                    }
                    out.push_str(key);
                }
                PathSegment::Index(index) => {
                    let _ = write!(out, "[{index}]");
                }
            }
        }
        out
    }
}

impl std::fmt::Display for Violation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let location = self.location();
        if location.is_empty() {
            write!(
                f,
                "expected {}, got {} [{}]",
                self.expected, self.value_summary, self.code
            )
        } else {
            write!(
                f,
                "at {}: expected {}, got {} [{}]",
                location, self.expected, self.value_summary, self.code
            )
        }
    }
}

impl std::error::Error for Violation {}

#[cfg(test)]
mod tests {
    use super::*;

    /// The element schema of a homogeneous (`[T, ...]`) sequence node.
    fn homogeneous_elem(schema: &Schema) -> &Schema {
        match schema {
            Schema::Seq { regex, .. } => {
                regex.linear().expect("linear").1.expect("homogeneous tail")
            }
            _ => panic!("not a sequence: {schema:?}"),
        }
    }

    #[test]
    fn violation_renders_root_message() {
        let v = Violation {
            code: "int_type",
            path: Vec::new(),
            expected: "int".to_owned(),
            value_summary: "'x'".to_owned(),
        };
        assert_eq!(v.location(), "");
        assert_eq!(v.to_string(), "expected int, got 'x' [int_type]");
    }

    #[test]
    fn violation_renders_nested_location() {
        let v = Violation {
            code: "string_type",
            path: vec![PathSegment::Key("name".to_owned()), PathSegment::Index(2)],
            expected: "str".to_owned(),
            value_summary: "5".to_owned(),
        };
        assert_eq!(v.location(), "name[2]");
        assert!(v.to_string().starts_with("at name[2]: expected str"));
    }

    #[test]
    fn labels_and_codes_for_every_variant() {
        let cases = [
            (Schema::Anything, "anything", "anything"),
            (Schema::Nothing, "nothing", "no_match"),
            (Schema::NoneType, "None", "none_type"),
            (Schema::Bool, "bool", "bool_type"),
            (Schema::Int, "int", "int_type"),
            (Schema::Float, "float", "float_type"),
            (Schema::Str, "str", "string_type"),
            (Schema::Bytes, "bytes", "bytes_type"),
            (Schema::Literal(0), "literal", "literal_value"),
            (
                Schema::list(SeqRegex::homogeneous(Schema::Int)),
                "list",
                "list_type",
            ),
            (
                Schema::tuple(SeqRegex::fixed([Schema::Int])),
                "tuple",
                "tuple_type",
            ),
            (Schema::Set(Box::new(Schema::Int)), "set", "set_type"),
            (
                Schema::mapping(Schema::Str, Schema::Int),
                "dict",
                "dict_type",
            ),
            (
                Schema::record(
                    vec![Field {
                        name: "k".to_owned(),
                        schema: Schema::Int,
                        required: true,
                    }],
                    false,
                ),
                "dict",
                "dict_type",
            ),
        ];
        for (schema, expected, code) in cases {
            assert_eq!(schema.expected(), expected, "expected for {schema:?}");
            assert_eq!(schema.error_code(), code, "code for {schema:?}");
        }
    }

    #[test]
    fn location_renders_keys_indices_and_their_mix() {
        let key_only = Violation {
            code: "x",
            path: vec![
                PathSegment::Key("a".to_owned()),
                PathSegment::Key("b".to_owned()),
            ],
            expected: String::new(),
            value_summary: String::new(),
        };
        assert_eq!(key_only.location(), "a.b");

        let index_only = Violation {
            code: "x",
            path: vec![PathSegment::Index(0), PathSegment::Index(3)],
            expected: String::new(),
            value_summary: String::new(),
        };
        assert_eq!(index_only.location(), "[0][3]");

        let mixed = Violation {
            code: "x",
            path: vec![
                PathSegment::Key("items".to_owned()),
                PathSegment::Index(2),
                PathSegment::Key("id".to_owned()),
            ],
            expected: "int".to_owned(),
            value_summary: "'x'".to_owned(),
        };
        assert_eq!(mixed.location(), "items[2].id");
        assert_eq!(
            mixed.to_string(),
            "at items[2].id: expected int, got 'x' [x]"
        );
    }

    #[test]
    fn mapping_and_record_share_the_dict_label() {
        let mapping = Schema::mapping(Schema::Str, Schema::Int);
        let record = Schema::record(Vec::new(), false);
        assert_eq!(mapping.expected(), record.expected());
        assert_eq!(mapping.error_code(), record.error_code());
    }

    /// Whether a record-shaped keyed map admits undeclared keys (has a default).
    fn record_is_open(schema: &Schema) -> bool {
        match schema {
            Schema::KeyedMap { defaults, .. } => !defaults.is_empty(),
            _ => panic!("not a keyed map: {schema:?}"),
        }
    }

    #[test]
    fn with_records_open_flips_every_record_in_the_tree() {
        let record = Schema::record(
            vec![Field {
                name: "k".to_owned(),
                schema: Schema::Int,
                required: true,
            }],
            false,
        );
        let schema = Schema::list(SeqRegex::homogeneous(record));
        let opened = schema.with_records_open(true);
        assert!(record_is_open(homogeneous_elem(&opened)));
        // strict flips it back.
        let closed = schema.with_records_open(true).with_records_open(false);
        assert!(!record_is_open(homogeneous_elem(&closed)));
    }

    #[test]
    fn schema_equality_is_structural() {
        assert_eq!(
            Schema::list(SeqRegex::homogeneous(Schema::Int)),
            Schema::list(SeqRegex::homogeneous(Schema::Int))
        );
        assert_ne!(
            Schema::list(SeqRegex::homogeneous(Schema::Int)),
            Schema::list(SeqRegex::homogeneous(Schema::Str))
        );
        assert_ne!(Schema::Literal(0), Schema::Literal(1));
    }

    #[test]
    fn resolve_self_replaces_only_the_matching_token() {
        let body = Schema::list(SeqRegex::homogeneous(Schema::SelfRef(1)));
        let resolved = body.resolve_self(1, 3);
        assert!(matches!(homogeneous_elem(&resolved), Schema::Ref(3)));
        assert!(matches!(
            Schema::SelfRef(2).resolve_self(1, 3),
            Schema::SelfRef(2)
        ));
    }

    #[test]
    fn contractivity_requires_a_structural_guard() {
        assert!(!Schema::list(SeqRegex::homogeneous(Schema::Ref(0))).occurs_unguarded(0, false));
        assert!(Schema::Ref(0).occurs_unguarded(0, false));
        assert!(Schema::Union(vec![Schema::Int, Schema::Ref(0)]).occurs_unguarded(0, false));
        assert!(
            !Schema::list(SeqRegex::homogeneous(Schema::Union(vec![
                Schema::Int,
                Schema::Ref(0)
            ])))
            .occurs_unguarded(0, false)
        );
    }

    #[test]
    fn shifted_remaps_ref_by_the_definition_offset() {
        let shifted = Schema::list(SeqRegex::homogeneous(Schema::Ref(0))).shifted(7, 4);
        assert!(matches!(homogeneous_elem(&shifted), Schema::Ref(4)));
        assert!(matches!(
            Schema::SelfRef(9).shifted(1, 1),
            Schema::SelfRef(9)
        ));
    }

    #[test]
    fn refine_delegates_label_and_code_to_its_base() {
        let refined = Schema::Refine {
            base: Box::new(Schema::Str),
            constraints: vec![Constraint::MinLen(1)],
        };
        assert_eq!(refined.expected(), "str");
        assert_eq!(refined.error_code(), "string_type");
    }

    #[test]
    fn field_is_cloneable_and_carries_its_flag() {
        let field = Field {
            name: "n".to_owned(),
            schema: Schema::Int,
            required: false,
        };
        let copy = field.clone();
        assert_eq!(copy.name, "n");
        assert!(!copy.required);
        assert_eq!(copy.schema, Schema::Int);
    }

    #[test]
    fn linear_recognizes_the_frontend_sequence_shapes() {
        let homogeneous = SeqRegex::homogeneous(Schema::Int);
        let (prefix, tail) = homogeneous.linear().expect("homogeneous is linear");
        assert!(prefix.is_empty());
        assert!(matches!(tail, Some(Schema::Int)));

        let fixed = SeqRegex::fixed([Schema::Int, Schema::Str]);
        let (prefix, tail) = fixed.linear().expect("fixed is linear");
        assert_eq!(prefix.len(), 2);
        assert!(tail.is_none());

        let prefix_tail = SeqRegex::prefix_tail([Schema::Str], Schema::Int);
        let (prefix, tail) = prefix_tail.linear().expect("prefix-plus-tail is linear");
        assert_eq!(prefix.len(), 1);
        assert!(matches!(tail, Some(Schema::Int)));

        let (prefix, tail) = SeqRegex::Empty.linear().expect("empty is linear");
        assert!(prefix.is_empty() && tail.is_none());
    }

    #[test]
    fn linear_rejects_the_non_linear_shapes() {
        let elem = || SeqRegex::Elem(Box::new(Schema::Int));
        // Alternation is not a linear sequence.
        assert!(SeqRegex::Or(vec![elem()]).linear().is_none());
        // A repetition of something other than a single element.
        assert!(
            SeqRegex::Star(Box::new(SeqRegex::Cat(vec![])))
                .linear()
                .is_none()
        );
        // A repetition that is not in tail position.
        let star_first = SeqRegex::Cat(vec![SeqRegex::Star(Box::new(elem())), elem()]);
        assert!(star_first.linear().is_none());
        // Alternation nested inside a concatenation.
        let cat_or = SeqRegex::Cat(vec![SeqRegex::Or(vec![SeqRegex::Empty])]);
        assert!(cat_or.linear().is_none());
    }

    #[test]
    fn sequence_transforms_recurse_through_every_regex_arm() {
        // A regex touching Or, Cat, Star, and Elem, with a Ref element and a
        // SelfRef under a repetition, so every arm of the transforms is walked.
        let regex = SeqRegex::Or(vec![
            SeqRegex::Cat(vec![
                SeqRegex::Elem(Box::new(Schema::Ref(0))),
                SeqRegex::Star(Box::new(SeqRegex::Elem(Box::new(Schema::SelfRef(7))))),
            ]),
            SeqRegex::Empty,
        ]);
        let seq = Schema::list(regex);

        // The Ref sits under the sequence guard, so it is not unguarded.
        assert!(!seq.occurs_unguarded(0, false));
        // simplify and with_records_open preserve the sequence shape.
        assert!(matches!(seq.simplify(), Schema::Seq { .. }));
        assert!(matches!(seq.with_records_open(true), Schema::Seq { .. }));

        // shifted moves the Ref element by the definitions offset.
        let Schema::Seq {
            regex: SeqRegex::Or(branches),
            ..
        } = seq.shifted(0, 5)
        else {
            panic!("shape preserved")
        };
        let SeqRegex::Cat(parts) = &branches[0] else {
            panic!("Or branch is a Cat")
        };
        let SeqRegex::Elem(head) = &parts[0] else {
            panic!("first part is an element")
        };
        assert!(matches!(**head, Schema::Ref(5)));

        // resolve_self rewrites the SelfRef under the repetition into a Ref.
        let Schema::Seq {
            regex: SeqRegex::Or(branches),
            ..
        } = seq.resolve_self(7, 3)
        else {
            panic!("shape preserved")
        };
        let SeqRegex::Cat(parts) = &branches[0] else {
            panic!("Or branch is a Cat")
        };
        let SeqRegex::Star(inner) = &parts[1] else {
            panic!("second part is a repetition")
        };
        let SeqRegex::Elem(tail) = inner.as_ref() else {
            panic!("repetition wraps an element")
        };
        assert!(matches!(**tail, Schema::Ref(3)));
    }

    #[test]
    fn keyed_map_transforms_recurse_through_fields_and_defaults() {
        let schema = Schema::KeyedMap {
            fields: vec![Field {
                name: "f".to_owned(),
                schema: Schema::Ref(0),
                required: true,
            }],
            defaults: vec![(Schema::Str, Schema::SelfRef(7))],
        };
        // Both the field's Ref and the default's SelfRef sit under the map guard.
        assert!(!schema.occurs_unguarded(0, false));
        // shifted moves the field's Ref by the definitions offset.
        let Schema::KeyedMap { fields, .. } = schema.shifted(0, 5) else {
            panic!("shape preserved")
        };
        assert!(matches!(fields[0].schema, Schema::Ref(5)));
        // resolve_self rewrites the default clause's SelfRef into a Ref.
        let Schema::KeyedMap { defaults, .. } = schema.resolve_self(7, 3) else {
            panic!("shape preserved")
        };
        assert!(matches!(defaults[0].1, Schema::Ref(3)));
    }

    fn not(s: Schema) -> Schema {
        Schema::Complement(Box::new(s))
    }

    #[test]
    fn simplify_decides_the_complement_laws() {
        // X ∩ ¬X = ⊥ and X ∪ ¬X = ⊤.
        assert_eq!(
            Schema::Intersection(vec![Schema::Int, not(Schema::Int)]).simplify(),
            Schema::Nothing
        );
        assert_eq!(
            Schema::Union(vec![Schema::Int, not(Schema::Int)]).simplify(),
            Schema::Anything
        );
        // Disjoint basics and disjoint container kinds give an empty intersection.
        assert_eq!(
            Schema::Intersection(vec![Schema::Int, Schema::Str]).simplify(),
            Schema::Nothing
        );
        assert_eq!(
            Schema::Intersection(vec![
                Schema::list(SeqRegex::homogeneous(Schema::Int)),
                Schema::Set(Box::new(Schema::Int)),
            ])
            .simplify(),
            Schema::Nothing
        );
        // bool ⊆ int, so their intersection is not empty.
        assert_ne!(
            Schema::Intersection(vec![Schema::Bool, Schema::Int]).simplify(),
            Schema::Nothing
        );
    }

    #[test]
    fn simplify_preserves_gradual_any_under_complement() {
        // The gradual `Any` must not be rewritten by the complement laws.
        assert_ne!(
            Schema::Intersection(vec![Schema::Any, not(Schema::Any)]).simplify(),
            Schema::Nothing
        );
        assert_ne!(
            Schema::Union(vec![Schema::Any, not(Schema::Any)]).simplify(),
            Schema::Anything
        );
    }

    #[test]
    fn disjoint_is_sound_for_the_decidable_fragment() {
        assert!(Schema::Int.disjoint(&Schema::Str));
        assert!(Schema::Int.disjoint(&Schema::Float));
        // Every concrete tag is disjoint from a distinct one.
        assert!(Schema::NoneType.disjoint(&Schema::Int));
        assert!(Schema::Bytes.disjoint(&Schema::Str));
        let list_int = Schema::list(SeqRegex::homogeneous(Schema::Int));
        let tuple_empty = Schema::tuple(SeqRegex::fixed([]));
        assert!(tuple_empty.disjoint(&list_int)); // tuple vs list
        assert!(
            Schema::FrozenSet(Box::new(Schema::Int)).disjoint(&Schema::Set(Box::new(Schema::Int)))
        );
        assert!(Schema::mapping(Schema::Str, Schema::Int).disjoint(&Schema::Int)); // dict vs int
        // Nothing is disjoint from everything.
        assert!(Schema::Nothing.disjoint(&Schema::Int));
        assert!(Schema::Int.disjoint(&Schema::Nothing));
        // Same tag is not disjoint: two list types share the empty list.
        assert!(!list_int.disjoint(&Schema::list(SeqRegex::homogeneous(Schema::Str))));
        assert!(!Schema::Bool.disjoint(&Schema::Int)); // bool is a subtype of int
        assert!(!Schema::Int.disjoint(&Schema::Int));
        // Conservative where the core cannot decide soundly.
        assert!(!Schema::Literal(0).disjoint(&Schema::Int));
        assert!(!Schema::Instance(0).disjoint(&Schema::Int));
        assert!(!Schema::Any.disjoint(&Schema::Int));
        // A refinement is disjoint exactly when its base is.
        let refined = Schema::Refine {
            base: Box::new(Schema::Int),
            constraints: vec![Constraint::Ge(0)],
        };
        assert!(refined.disjoint(&Schema::Str));
        assert!(!refined.disjoint(&Schema::Int));
    }
}

#[cfg(test)]
mod laws {
    use super::*;
    use proptest::prelude::*;

    /// A small schema generator: atoms combined by union, intersection, and
    /// complement. Pool indices are arbitrary but consistent across a value.
    fn schema() -> impl Strategy<Value = Schema> {
        let atom = prop_oneof![
            Just(Schema::Anything),
            Just(Schema::Nothing),
            Just(Schema::Any),
            Just(Schema::Int),
            Just(Schema::Str),
            Just(Schema::Bool),
            Just(Schema::Literal(0)),
            Just(Schema::Instance(1)),
        ];
        atom.prop_recursive(4, 24, 3, |inner| {
            prop_oneof![
                proptest::collection::vec(inner.clone(), 1..4).prop_map(Schema::Union),
                proptest::collection::vec(inner.clone(), 1..4).prop_map(Schema::Intersection),
                inner.prop_map(|s| Schema::Complement(Box::new(s))),
            ]
        })
    }

    fn union(a: Schema, b: Schema) -> Schema {
        Schema::Union(vec![a, b])
    }
    fn intersection(a: Schema, b: Schema) -> Schema {
        Schema::Intersection(vec![a, b])
    }
    fn not(a: Schema) -> Schema {
        Schema::Complement(Box::new(a))
    }

    /// One representative value per distinguishable scalar region. The five
    /// container kinds and `OTHER` are indistinguishable to a scalar schema (no
    /// scalar atom touches them, and a complement includes them together), so a
    /// single `Other` sample stands for that whole class.
    #[derive(Clone, Copy)]
    enum Sample {
        None,
        Bool,
        Int,
        Float,
        Str,
        Bytes,
        Other,
    }

    const SAMPLES: [Sample; 7] = [
        Sample::None,
        Sample::Bool,
        Sample::Int,
        Sample::Float,
        Sample::Str,
        Sample::Bytes,
        Sample::Other,
    ];

    /// A reference membership predicate for the scalar fragment, independent of
    /// the region-set decision under test, used as its oracle.
    fn member(schema: &Schema, value: Sample) -> bool {
        match schema {
            Schema::Anything => true,
            Schema::Nothing => false,
            Schema::NoneType => matches!(value, Sample::None),
            Schema::Bool => matches!(value, Sample::Bool),
            Schema::Int => matches!(value, Sample::Bool | Sample::Int), // bool ⊆ int
            Schema::Float => matches!(value, Sample::Float),
            Schema::Str => matches!(value, Sample::Str),
            Schema::Bytes => matches!(value, Sample::Bytes),
            Schema::Union(members) => members.iter().any(|m| member(m, value)),
            Schema::Intersection(members) => members.iter().all(|m| member(m, value)),
            Schema::Complement(inner) => !member(inner, value),
            other => unreachable!("oracle is scalar-only, got {other:?}"),
        }
    }

    /// A generator over the scalar-decidable fragment: scalar atoms combined by
    /// union, intersection, and complement.
    fn scalar_schema() -> impl Strategy<Value = Schema> {
        let atom = prop_oneof![
            Just(Schema::Anything),
            Just(Schema::Nothing),
            Just(Schema::NoneType),
            Just(Schema::Bool),
            Just(Schema::Int),
            Just(Schema::Float),
            Just(Schema::Str),
            Just(Schema::Bytes),
        ];
        atom.prop_recursive(4, 24, 3, |inner| {
            prop_oneof![
                proptest::collection::vec(inner.clone(), 1..4).prop_map(Schema::Union),
                proptest::collection::vec(inner.clone(), 1..4).prop_map(Schema::Intersection),
                inner.prop_map(|s| Schema::Complement(Box::new(s))),
            ]
        })
    }

    #[test]
    fn decides_scalar_emptiness_subtyping_and_equivalence() {
        // Multi-way emptiness the pairwise checks cannot reach.
        assert!(
            Schema::Intersection(vec![Schema::Int, not(Schema::Bool), not(Schema::Int)]).is_empty()
        );
        assert!(
            Schema::Intersection(vec![
                Schema::Union(vec![Schema::Int, Schema::Str]),
                not(Schema::Int),
                not(Schema::Str),
            ])
            .is_empty()
        );
        assert!(!Schema::Intersection(vec![Schema::Int, not(Schema::Bool)]).is_empty());
        // Subtyping, with bool ⊆ int.
        assert!(Schema::Bool.is_subtype(&Schema::Int));
        assert!(!Schema::Int.is_subtype(&Schema::Bool));
        assert!(!Schema::Float.is_subtype(&Schema::Int));
        // Equivalence between structurally different schemas: bool ∪ int = int.
        assert!(Schema::Union(vec![Schema::Bool, Schema::Int]).equivalent(&Schema::Int));
    }

    #[test]
    fn is_empty_and_subtype_are_sound_off_the_scalar_fragment() {
        // Non-scalar leaves are never decided empty.
        assert!(!Schema::Any.is_empty());
        assert!(!Schema::Literal(0).is_empty());
        assert!(!Schema::Instance(0).is_empty());
        assert!(!Schema::Set(Box::new(Schema::Int)).is_empty());
        assert!(!Schema::list(SeqRegex::homogeneous(Schema::Int)).is_empty());
        // A scalar mixed with a non-scalar leaf is undecidable here, so it is
        // never claimed empty (an instance could subclass the scalar's type).
        assert!(!Schema::Intersection(vec![Schema::Int, Schema::Instance(0)]).is_empty());
        // The gradual `Any` is never collapsed.
        assert!(!Schema::Intersection(vec![Schema::Any, not(Schema::Any)]).is_empty());
        // Subtyping off the fragment is reflexive only.
        assert!(Schema::Instance(0).is_subtype(&Schema::Instance(0)));
        assert!(!Schema::Instance(0).is_subtype(&Schema::Instance(1)));
    }

    #[test]
    fn decides_structural_container_emptiness() {
        // A fixed sequence with an impossible element matches no sequence.
        let empty_pair = Schema::tuple(SeqRegex::fixed([Schema::Int, Schema::Nothing]));
        assert!(empty_pair.is_empty());
        // A list or tuple that admits the empty sequence is never empty.
        assert!(!Schema::list(SeqRegex::homogeneous(Schema::Nothing)).is_empty());
        assert!(!Schema::tuple(SeqRegex::fixed([Schema::Int])).is_empty());
        // A set or frozenset is never empty: the empty collection is a member.
        assert!(!Schema::Set(Box::new(Schema::Nothing)).is_empty());
        assert!(!Schema::FrozenSet(Box::new(Schema::Nothing)).is_empty());
        // A keyed map is empty exactly when a required field is impossible.
        let field = |required| Field {
            name: "x".to_owned(),
            schema: Schema::Nothing,
            required,
        };
        assert!(
            Schema::KeyedMap {
                fields: vec![field(true)],
                defaults: Vec::new(),
            }
            .is_empty()
        );
        assert!(
            !Schema::KeyedMap {
                fields: vec![field(false)],
                defaults: Vec::new(),
            }
            .is_empty()
        );
        // A union is empty only when every member is.
        assert!(Schema::Union(vec![Schema::Nothing, empty_pair.clone()]).is_empty());
        assert!(!Schema::Union(vec![Schema::Int, empty_pair]).is_empty());
    }

    #[test]
    fn decides_structural_subtyping_between_containers() {
        let set = |s| Schema::Set(Box::new(s));
        let frozenset = |s| Schema::FrozenSet(Box::new(s));
        // Sets and frozensets reduce to element inclusion (bool ⊆ int).
        assert!(set(Schema::Bool).is_subtype(&set(Schema::Int)));
        assert!(!set(Schema::Int).is_subtype(&set(Schema::Bool)));
        assert!(frozenset(Schema::Bool).is_subtype(&frozenset(Schema::Int)));
        // Different container kinds are never subtypes.
        assert!(!set(Schema::Int).is_subtype(&frozenset(Schema::Int)));
        // Homogeneous sequences: list[bool] ⊆ list[int], not list[int] ⊆ list[str].
        let list = |r| Schema::list(r);
        let tuple = |r| Schema::tuple(r);
        assert!(
            list(SeqRegex::homogeneous(Schema::Bool))
                .is_subtype(&list(SeqRegex::homogeneous(Schema::Int)))
        );
        assert!(
            !list(SeqRegex::homogeneous(Schema::Int))
                .is_subtype(&list(SeqRegex::homogeneous(Schema::Str)))
        );
        // Fixed sequences compare pointwise; a tuple is not a list.
        assert!(
            tuple(SeqRegex::fixed([Schema::Bool, Schema::Str]))
                .is_subtype(&tuple(SeqRegex::fixed([Schema::Int, Schema::Str])))
        );
        assert!(
            !tuple(SeqRegex::fixed([Schema::Int]))
                .is_subtype(&list(SeqRegex::homogeneous(Schema::Int)))
        );
        // A fixed list is a subtype of a homogeneous list when each element is.
        assert!(
            list(SeqRegex::fixed([Schema::Bool, Schema::Int]))
                .is_subtype(&list(SeqRegex::homogeneous(Schema::Int)))
        );
        // Equivalence between structurally different container schemas.
        assert!(set(Schema::Union(vec![Schema::Bool, Schema::Int])).equivalent(&set(Schema::Int)));
    }

    #[test]
    fn decides_record_and_mapping_subtyping() {
        let field = |name: &str, schema, required| Field {
            name: name.to_owned(),
            schema,
            required,
        };
        let record = |fields| Schema::KeyedMap {
            fields,
            defaults: Vec::new(),
        };
        let mapping = |k, v| Schema::KeyedMap {
            fields: Vec::new(),
            defaults: vec![(k, v)],
        };

        // Width: a closed record with fewer keys is a subtype of one with more.
        let narrow = record(vec![field("x", Schema::Int, true)]);
        let wide = record(vec![
            field("x", Schema::Int, true),
            field("y", Schema::Str, false),
        ]);
        assert!(narrow.is_subtype(&wide));
        assert!(!wide.is_subtype(&narrow)); // wide admits key y; narrow (closed) forbids it
        // Depth: shared field schemas covary (bool ⊆ int).
        assert!(
            record(vec![field("x", Schema::Bool, true)]).is_subtype(&record(vec![field(
                "x",
                Schema::Int,
                true
            )]))
        );
        // Required: a field the supertype requires must be required in the subtype.
        let required = record(vec![field("x", Schema::Int, true)]);
        let optional = record(vec![field("x", Schema::Int, false)]);
        assert!(required.is_subtype(&optional));
        assert!(!optional.is_subtype(&required));
        // Mappings covary in key and value.
        assert!(mapping(Schema::Str, Schema::Bool).is_subtype(&mapping(Schema::Str, Schema::Int)));
        assert!(!mapping(Schema::Str, Schema::Int).is_subtype(&mapping(Schema::Str, Schema::Bool)));
        // A record and a mapping are not compared — conservative.
        assert!(!narrow.is_subtype(&mapping(Schema::Str, Schema::Int)));
    }

    #[test]
    fn decides_sequence_subtyping_with_prefix_tail_and_alternation() {
        // A list `[head, tail*]`: a one-element fixed prefix then a repeated tail.
        let prefix_tail = |head, tail| {
            Schema::list(SeqRegex::Cat(vec![
                SeqRegex::Elem(Box::new(head)),
                SeqRegex::Star(Box::new(SeqRegex::Elem(Box::new(tail)))),
            ]))
        };
        // Prefix and tail covary (bool ⊆ int), in both positions.
        assert!(
            prefix_tail(Schema::Bool, Schema::Bool)
                .is_subtype(&prefix_tail(Schema::Int, Schema::Int))
        );
        assert!(
            !prefix_tail(Schema::Int, Schema::Int)
                .is_subtype(&prefix_tail(Schema::Int, Schema::Bool))
        );
        // A fixed-length list is a subtype of a prefix-and-tail one it fits.
        assert!(
            Schema::list(SeqRegex::fixed([Schema::Bool, Schema::Int]))
                .is_subtype(&prefix_tail(Schema::Int, Schema::Int))
        );
        // Alternation distributes: (bool* | int*) ⊆ int*, but int* ⊄ (bool* | str*).
        let alternation = |a, b| {
            Schema::list(SeqRegex::Or(vec![
                SeqRegex::homogeneous(a),
                SeqRegex::homogeneous(b),
            ]))
        };
        assert!(
            alternation(Schema::Bool, Schema::Int)
                .is_subtype(&Schema::list(SeqRegex::homogeneous(Schema::Int)))
        );
        assert!(
            !Schema::list(SeqRegex::homogeneous(Schema::Int))
                .is_subtype(&alternation(Schema::Bool, Schema::Str))
        );
    }

    #[test]
    fn detects_uninhabited_recursive_schemas() {
        let field = |name: &str, schema, required| Field {
            name: name.to_owned(),
            schema,
            required,
        };
        // t = {value: int, next: t} — a mandatory self-reference, no base case:
        // no finite value satisfies it.
        let uninhabited = [Schema::KeyedMap {
            fields: vec![
                field("value", Schema::Int, true),
                field("next", Schema::Ref(0), true),
            ],
            defaults: Vec::new(),
        }];
        assert!(Schema::Ref(0).is_empty_under(&uninhabited));
        // t = None | {next: t} — a base case makes it inhabited.
        let inhabited = [Schema::Union(vec![
            Schema::NoneType,
            Schema::KeyedMap {
                fields: vec![field("next", Schema::Ref(0), true)],
                defaults: Vec::new(),
            },
        ])];
        assert!(!Schema::Ref(0).is_empty_under(&inhabited));
        // t = {next?: t} — an optional self-reference is inhabited by the empty map.
        let optional = [Schema::KeyedMap {
            fields: vec![field("next", Schema::Ref(0), false)],
            defaults: Vec::new(),
        }];
        assert!(!Schema::Ref(0).is_empty_under(&optional));
        // t = [t] — a list of itself is inhabited by the empty list.
        let list_of_self = [Schema::list(SeqRegex::homogeneous(Schema::Ref(0)))];
        assert!(!Schema::Ref(0).is_empty_under(&list_of_self));
        // An unresolved reference stays conservative.
        assert!(!Schema::Ref(9).is_empty_under(&uninhabited));
        // Without the definitions, recursion is not resolved (no-arg is_empty).
        assert!(!Schema::Ref(0).is_empty());
    }

    #[test]
    fn decides_recursive_subtyping_coinductively() {
        let field = |name: &str, schema, required| Field {
            name: name.to_owned(),
            schema,
            required,
        };
        let list_of = |value, next| {
            Schema::Union(vec![
                Schema::NoneType,
                Schema::KeyedMap {
                    fields: vec![
                        field("value", value, true),
                        field("next", Schema::Ref(next), true),
                    ],
                    defaults: Vec::new(),
                },
            ])
        };
        // Two structurally identical recursive linked-list types are equivalent.
        let identical = [list_of(Schema::Int, 0), list_of(Schema::Int, 1)];
        assert!(Schema::Ref(0).equivalent_under(&Schema::Ref(1), &NoLeafRelations, &identical));
        // Depth covariance through the recursion: a bool-valued list is a subtype
        // of an int-valued one (bool ⊆ int), but not the reverse.
        let covary = [list_of(Schema::Bool, 0), list_of(Schema::Int, 1)];
        assert!(Schema::Ref(0).is_subtype_under(&Schema::Ref(1), &NoLeafRelations, &covary));
        assert!(!Schema::Ref(1).is_subtype_under(&Schema::Ref(0), &NoLeafRelations, &covary));
    }

    /// A sample value for the subtyping oracle: a scalar, or a set whose element
    /// kinds are listed. Sets suffice to exercise the container rule without a
    /// regex matcher; sequence rules are covered by the unit test above.
    #[derive(Clone)]
    enum Val {
        Scalar(Sample),
        SetOf(Vec<Sample>),
    }

    fn samples_v() -> Vec<Val> {
        let mut values: Vec<Val> = SAMPLES.iter().map(|&s| Val::Scalar(s)).collect();
        values.push(Val::SetOf(vec![]));
        values.push(Val::SetOf(vec![Sample::Bool]));
        values.push(Val::SetOf(vec![Sample::Int]));
        values.push(Val::SetOf(vec![Sample::Str]));
        values.push(Val::SetOf(vec![Sample::Int, Sample::Str]));
        values
    }

    /// Reference membership for the scalar-and-set fragment, the oracle the
    /// structural subtyping decision is checked against.
    fn member_v(schema: &Schema, value: &Val) -> bool {
        match schema {
            Schema::Anything => true,
            Schema::Nothing => false,
            Schema::Set(element) => match value {
                Val::SetOf(elements) => elements.iter().all(|&e| member(element, e)),
                Val::Scalar(_) => false,
            },
            Schema::Union(members) => members.iter().any(|m| member_v(m, value)),
            Schema::Intersection(members) => members.iter().all(|m| member_v(m, value)),
            Schema::Complement(inner) => !member_v(inner, value),
            scalar => match value {
                Val::Scalar(sample) => member(scalar, *sample),
                Val::SetOf(_) => false,
            },
        }
    }

    /// A generator over scalars, sets of scalar schemas, and their Boolean
    /// combinations — the fragment the `member_v` oracle covers.
    fn scalar_or_set_schema() -> impl Strategy<Value = Schema> {
        let leaf = prop_oneof![
            Just(Schema::Anything),
            Just(Schema::Nothing),
            Just(Schema::NoneType),
            Just(Schema::Bool),
            Just(Schema::Int),
            Just(Schema::Float),
            Just(Schema::Str),
            Just(Schema::Bytes),
            scalar_schema().prop_map(|s| Schema::Set(Box::new(s))),
        ];
        leaf.prop_recursive(3, 16, 3, |inner| {
            prop_oneof![
                proptest::collection::vec(inner.clone(), 1..3).prop_map(Schema::Union),
                proptest::collection::vec(inner.clone(), 1..3).prop_map(Schema::Intersection),
                inner.prop_map(|s| Schema::Complement(Box::new(s))),
            ]
        })
    }

    proptest! {
        #[test]
        fn scalar_decision_matches_the_value_oracle(a in scalar_schema(), b in scalar_schema()) {
            let a_empty = SAMPLES.iter().all(|&v| !member(&a, v));
            prop_assert_eq!(a.is_empty(), a_empty);

            let a_sub_b = SAMPLES.iter().all(|&v| !member(&a, v) || member(&b, v));
            let b_sub_a = SAMPLES.iter().all(|&v| !member(&b, v) || member(&a, v));
            prop_assert_eq!(a.is_subtype(&b), a_sub_b);
            prop_assert_eq!(a.equivalent(&b), a_sub_b && b_sub_a);
        }

        #[test]
        fn structural_subtyping_is_sound(a in scalar_or_set_schema(), b in scalar_or_set_schema()) {
            prop_assert!(a.is_subtype(&a)); // reflexivity holds everywhere
            // Soundness: a claimed subtype never accepts a sample the supertype rejects.
            if a.is_subtype(&b) {
                for value in &samples_v() {
                    prop_assert!(!member_v(&a, value) || member_v(&b, value));
                }
            }
        }

        #[test]
        fn simplify_is_idempotent(a in schema()) {
            let once = a.simplify();
            prop_assert_eq!(once.clone(), once.simplify());
        }

        #[test]
        fn union_and_intersection_commute(a in schema(), b in schema()) {
            prop_assert_eq!(union(a.clone(), b.clone()).simplify(), union(b.clone(), a.clone()).simplify());
            prop_assert_eq!(intersection(a.clone(), b.clone()).simplify(), intersection(b, a).simplify());
        }

        #[test]
        fn union_and_intersection_associate(a in schema(), b in schema(), c in schema()) {
            prop_assert_eq!(
                union(a.clone(), union(b.clone(), c.clone())).simplify(),
                union(union(a.clone(), b.clone()), c.clone()).simplify()
            );
            prop_assert_eq!(
                intersection(a.clone(), intersection(b.clone(), c.clone())).simplify(),
                intersection(intersection(a, b), c).simplify()
            );
        }

        #[test]
        fn idempotence(a in schema()) {
            prop_assert_eq!(union(a.clone(), a.clone()).simplify(), a.clone().simplify());
            prop_assert_eq!(intersection(a.clone(), a.clone()).simplify(), a.simplify());
        }

        #[test]
        fn identities(a in schema()) {
            prop_assert_eq!(union(a.clone(), Schema::Nothing).simplify(), a.clone().simplify());
            prop_assert_eq!(intersection(a.clone(), Schema::Anything).simplify(), a.clone().simplify());
            prop_assert_eq!(union(a.clone(), Schema::Anything).simplify(), Schema::Anything);
            prop_assert_eq!(intersection(a, Schema::Nothing).simplify(), Schema::Nothing);
        }

        #[test]
        fn double_negation(a in schema()) {
            prop_assert_eq!(not(not(a.clone())).simplify(), a.simplify());
        }

        #[test]
        fn de_morgan(a in schema(), b in schema()) {
            prop_assert_eq!(
                not(union(a.clone(), b.clone())).simplify(),
                intersection(not(a.clone()), not(b)).simplify()
            );
        }
    }
}
