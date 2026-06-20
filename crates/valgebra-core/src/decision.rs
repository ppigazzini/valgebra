//! The decision procedures over the IR: emptiness, subtyping, equivalence, and
//! disjointness, with the leaf-relation oracle and the scalar region partition.

use crate::ir::{Constraint, Field, Schema, SeqKind, SeqRegex};

impl SeqRegex {
    /// Whether the regex matches **no** sequence at all — its language is empty.
    /// `Empty` and `Star` always match the empty sequence, so they are never
    /// empty; a single element is empty when its schema is; a concatenation is
    /// empty when any part is (every part must be matchable); an alternation is
    /// empty only when every alternative is.
    fn language_is_empty(
        &self,
        oracle: &dyn LeafRelations,
        defs: &[Schema],
        visiting: &mut Vec<usize>,
    ) -> bool {
        match self {
            SeqRegex::Empty | SeqRegex::Star(_) => false,
            SeqRegex::Elem(schema) => schema.is_empty_rec(oracle, defs, visiting),
            SeqRegex::Cat(parts) => parts
                .iter()
                .any(|p| p.language_is_empty(oracle, defs, visiting)),
            SeqRegex::Or(parts) => parts
                .iter()
                .all(|p| p.language_is_empty(oracle, defs, visiting)),
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
}

impl Schema {
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
    pub(crate) fn region_set(&self) -> Option<u16> {
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
        self.is_empty_rec(&NoLeafRelations, &[], &mut Vec::new())
    }

    /// Like [`is_empty`](Self::is_empty), but resolving recursive references
    /// through `defs`, so an uninhabited recursive schema — a mandatory
    /// self-reference with no base case — is detected. A reference `defs` does
    /// not resolve stays conservative (never reported empty).
    #[must_use]
    pub fn is_empty_under(&self, defs: &[Schema]) -> bool {
        self.is_empty_rec(&NoLeafRelations, defs, &mut Vec::new())
    }

    /// Like [`is_empty_under`](Self::is_empty_under), but with an `oracle` that
    /// can order the pool values behind refinement bounds, so an unsatisfiable
    /// bound conjunction (a lower bound above an upper bound) is detected.
    pub fn is_empty_with(&self, oracle: &dyn LeafRelations, defs: &[Schema]) -> bool {
        self.is_empty_rec(oracle, defs, &mut Vec::new())
    }

    fn is_empty_rec(
        &self,
        oracle: &dyn LeafRelations,
        defs: &[Schema],
        visiting: &mut Vec<usize>,
    ) -> bool {
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
                        let empty = def.is_empty_rec(oracle, defs, visiting);
                        visiting.pop();
                        empty
                    }
                    None => false,
                }
            }
            Schema::Seq { regex, .. } => regex.language_is_empty(oracle, defs, visiting),
            // A refinement is a subset of its base: an empty base empties it, and
            // so does an unsatisfiable bound conjunction (decided by the oracle).
            Schema::Refine { base, constraints } => {
                base.is_empty_rec(oracle, defs, visiting)
                    || bounds_unsatisfiable(constraints, oracle)
            }
            // An intersection is empty if a member is, if the scalar regions cancel,
            // or if the refinement bounds across its members cannot hold together.
            Schema::Intersection(members) => {
                members
                    .iter()
                    .any(|m| m.is_empty_rec(oracle, defs, visiting))
                    || matches!(self.region_set(), Some(0))
                    || intersection_bounds_unsatisfiable(members, oracle)
            }
            Schema::Set(_) | Schema::FrozenSet(_) => false,
            Schema::KeyedMap { fields, .. } => fields
                .iter()
                .any(|field| field.required && field.schema.is_empty_rec(oracle, defs, visiting)),
            Schema::Union(members) => members
                .iter()
                .all(|m| m.is_empty_rec(oracle, defs, visiting)),
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
    pub fn is_subtype_of(&self, other: &Schema) -> bool {
        self.is_subtype_of_under(other, &NoLeafRelations, &[])
    }

    /// [`is_subtype_of`](Self::is_subtype_of) with a [`LeafRelations`] oracle deciding
    /// the leaf relations the structural rules cannot (an `Instance` class or a
    /// `Literal` value), and the `defs` that resolve recursive references so
    /// subtyping is decided between recursive schemas too. The oracle's `None`
    /// and an unresolved reference both keep the conservative `false`.
    #[must_use]
    pub fn is_subtype_of_under(
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
            // A ⊆ ∅ exactly when A is empty, decided with the same oracle so a
            // refinement with unsatisfiable bounds is recognised here too.
            (_, Schema::Nothing) => self.is_empty_rec(cx.oracle, cx.defs, &mut Vec::new()),
            // (X ∪ Y) ⊆ Z iff X ⊆ Z and Y ⊆ Z; A ⊆ (Y ∩ Z) iff A ⊆ Y and A ⊆ Z.
            (Schema::Union(members), _) => members
                .iter()
                .all(|m| m.is_subtype_rec(other, cx, assumptions)),
            (_, Schema::Intersection(members)) => members
                .iter()
                .all(|m| self.is_subtype_rec(m, cx, assumptions)),
            // (A ∩ B) ⊆ C if some conjunct already is. When C is a union, the meet
            // may instead land in one branch, so that sound rule is tried too —
            // ahead of the plain `_ ⊆ (Y ∪ Z)` rule, so a meet that contains its
            // own supertype (a reference beside that union) decides, which is what
            // lets such a meet be recognised as a subtype of itself.
            (Schema::Intersection(members), _) => {
                members
                    .iter()
                    .any(|m| m.is_subtype_rec(other, cx, assumptions))
                    || matches!(other, Schema::Union(branches)
                        if branches.iter().any(|b| self.is_subtype_rec(b, cx, assumptions)))
            }
            // A ⊆ (Y ∪ Z) if A lands in one branch (sound, conservative).
            (_, Schema::Union(members)) => members
                .iter()
                .any(|m| self.is_subtype_rec(m, cx, assumptions)),
            // Unfold a recursive reference — after the lattice rules, so an
            // intersection or union meeting a reference decomposes first (which
            // lets a recursive member be compared against the reference rather
            // than the reference being unfolded past it). The goal is recorded so
            // a cycle back to it is caught by the coinductive hypothesis above.
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
            // Complement is contravariant: ¬A ⊆ ¬B exactly when B ⊆ A.
            (Schema::Complement(a), Schema::Complement(b)) => b.is_subtype_rec(a, cx, assumptions),
            // A refinement is a subset of its base. Against another refinement the
            // base must subtype and every constraint of the supertype must be
            // present, since more constraints denote a smaller set; equal bounds
            // share a pool index, so this decides nested bounds and lengths.
            // Bound-value entailment beyond syntactic containment is conservative.
            (
                Schema::Refine {
                    base: narrow_base,
                    constraints: narrow_cons,
                },
                Schema::Refine {
                    base: wide_base,
                    constraints: wide_cons,
                },
            ) => {
                narrow_base.is_subtype_rec(wide_base, cx, assumptions)
                    && wide_cons
                        .iter()
                        .all(|constraint| narrow_cons.contains(constraint))
            }
            // Against a non-refinement, a refinement inherits its base's supertypes.
            (Schema::Refine { base, .. }, _) => base.is_subtype_rec(other, cx, assumptions),
            // A leaf the structural rules cannot relate (an instance or literal):
            // defer to the oracle, conservative when it declines.
            _ => cx.oracle.leaf_subtype(self, other).unwrap_or(false),
        }
    }

    /// Whether `self` and `other` denote the same set — mutual inclusion.
    #[must_use]
    pub fn is_equivalent(&self, other: &Schema) -> bool {
        self.is_equivalent_under(other, &NoLeafRelations, &[])
    }

    /// [`is_equivalent`](Self::is_equivalent) under a [`LeafRelations`] oracle and
    /// the recursive definitions.
    #[must_use]
    pub fn is_equivalent_under(
        &self,
        other: &Schema,
        oracle: &dyn LeafRelations,
        defs: &[Schema],
    ) -> bool {
        self.is_subtype_of_under(other, oracle, defs)
            && other.is_subtype_of_under(self, oracle, defs)
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

    /// Order the two pool values behind refinement bounds at indices `left` and
    /// `right`, or `None` when the core cannot or the values are not comparable.
    /// The default decides nothing, so bound satisfiability stays conservative.
    fn compare(&self, _left: usize, _right: usize) -> Option<core::cmp::Ordering> {
        None
    }
}

/// The trivial [`LeafRelations`] that decides nothing — the core default, under
/// which `Instance` and `Literal` relations stay conservative.
pub struct NoLeafRelations;

impl LeafRelations for NoLeafRelations {
    fn leaf_subtype(&self, _sub: &Schema, _sup: &Schema) -> Option<bool> {
        None
    }
}

/// Whether a refinement's bound and length constraints cannot hold together: a
/// required minimum length above the allowed maximum, or a numeric lower bound
/// above the upper bound (or equal with a strict end). Sound: it reports
/// unsatisfiable only when the ordering the oracle returns forces it, and stays
/// conservative when the oracle cannot compare two bounds.
fn bounds_unsatisfiable(constraints: &[Constraint], oracle: &dyn LeafRelations) -> bool {
    use core::cmp::Ordering;
    let min_len = constraints
        .iter()
        .filter_map(|c| match c {
            Constraint::MinLen(n) => Some(*n),
            _ => None,
        })
        .max();
    let max_len = constraints
        .iter()
        .filter_map(|c| match c {
            Constraint::MaxLen(n) => Some(*n),
            _ => None,
        })
        .min();
    if let (Some(lo), Some(hi)) = (min_len, max_len)
        && lo > hi
    {
        return true;
    }
    let mut lower: Option<(usize, bool)> = None;
    let mut upper: Option<(usize, bool)> = None;
    for constraint in constraints {
        match constraint {
            Constraint::Ge(i) => lower = Some(tighter_bound(lower, (*i, false), oracle, true)),
            Constraint::Gt(i) => lower = Some(tighter_bound(lower, (*i, true), oracle, true)),
            Constraint::Le(i) => upper = Some(tighter_bound(upper, (*i, false), oracle, false)),
            Constraint::Lt(i) => upper = Some(tighter_bound(upper, (*i, true), oracle, false)),
            _ => {}
        }
    }
    if let (Some((lo, lo_strict)), Some((hi, hi_strict))) = (lower, upper) {
        return match oracle.compare(lo, hi) {
            Some(Ordering::Greater) => true,
            Some(Ordering::Equal) => lo_strict || hi_strict,
            _ => false,
        };
    }
    false
}

/// Keep the tighter of two one-sided bounds: the greater value for a lower bound,
/// the lesser for an upper bound; on equal values the strict end wins, and on an
/// incomparable pair the current bound is kept (conservative).
fn tighter_bound(
    current: Option<(usize, bool)>,
    candidate: (usize, bool),
    oracle: &dyn LeafRelations,
    is_lower: bool,
) -> (usize, bool) {
    use core::cmp::Ordering;
    let Some(current) = current else {
        return candidate;
    };
    match oracle.compare(candidate.0, current.0) {
        Some(Ordering::Equal) => (current.0, current.1 || candidate.1),
        Some(Ordering::Greater) => {
            if is_lower {
                candidate
            } else {
                current
            }
        }
        Some(Ordering::Less) => {
            if is_lower {
                current
            } else {
                candidate
            }
        }
        None => current,
    }
}

/// Whether the refinement constraints across the members of an intersection
/// cannot hold together. A value in the intersection satisfies every member, so
/// all their refinement constraints apply to it at once.
fn intersection_bounds_unsatisfiable(members: &[Schema], oracle: &dyn LeafRelations) -> bool {
    let merged: Vec<Constraint> = members
        .iter()
        .filter_map(|m| match m {
            Schema::Refine { constraints, .. } => Some(constraints.clone()),
            _ => None,
        })
        .flatten()
        .collect();
    !merged.is_empty() && bounds_unsatisfiable(&merged, oracle)
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
/// keyed-map `b`. Sound everywhere; complete on three shapes, conservative
/// (returns `false`) outside them:
///
/// 1. **Closed record ≤ anything** (`da` empty): holds by width and depth (each
///    field of `a` maps into a like-named field of `b` with a subtype schema) and
///    by required-ness (every field `b` requires is required in `a`).
/// 2. **Pure mapping ≤ pure mapping** (`fa` and `fb` empty): every clause of `a`
///    is subsumed by a clause of `b` with both key and value narrower.
/// 3. **Mixed record-and-catch-all ≤ mixed**, *when `a` declares every field `b`
///    declares*: shared fields narrow, `b`'s required fields are required in `a`,
///    each extra field of `a` is covered by a `str`/`anything`-keyed catch-all of
///    `b`, and every clause of `a` is subsumed by one of `b`.
///
/// The decided boundary's one deliberate gap: when the **supertype declares a
/// field the subtype lacks**, the relation needs the full set-theoretic decision
/// (the subtype's catch-all would have to cover that exact key name), so it stays
/// conservative. Extending that direction is the planned route toward
/// completeness; until then a true relation there is reported `false`, never an
/// unsound `true`.
fn keyed_map_subtype(
    fa: &[Field],
    da: &[(Schema, Schema)],
    fb: &[Field],
    db: &[(Schema, Schema)],
    cx: SubtypeCx<'_>,
    assumptions: &mut Vec<(Schema, Schema)>,
) -> bool {
    if da.is_empty() {
        // A closed record carries only its declared fields, so it is a subtype
        // when every field maps into a like-named field of the supertype (width
        // and depth) and the supertype's required fields are required here too. A
        // field the supertype does not declare would need the supertype's
        // catch-all to cover that exact key name, which stays conservative.
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
    } else if fa.is_empty() && fb.is_empty() {
        // Pure mappings: every key-pattern clause of the subtype must be subsumed
        // by a clause of the supertype, with both its keys and its values
        // narrower, so every entry the subtype admits the supertype admits too.
        da.iter().all(|(ka, va)| {
            db.iter().any(|(kb, vb)| {
                ka.is_subtype_rec(kb, cx, assumptions) && va.is_subtype_rec(vb, cx, assumptions)
            })
        })
    } else if fb.iter().all(|b| fa.iter().any(|a| a.name == b.name)) {
        // A record mixed with a catch-all whose fields include every field of the
        // supertype: a subtype when each shared field narrows, the supertype's
        // required fields are required here, every extra subtype field is covered
        // by a supertype catch-all whose keys include all field names (a `str` or
        // `anything` key), and every subtype catch-all clause is subsumed. A field
        // key is matched as a field, so the catch-all clauses govern the non-field
        // keys; the extra-field coverage handles the keys the supertype reads
        // through its catch-all. The reverse direction -- the supertype declaring a
        // field the subtype lacks -- needs the full comparison and stays
        // conservative.
        let shared = fb.iter().all(|b| {
            fa.iter()
                .find(|a| a.name == b.name)
                .is_some_and(|a| a.schema.is_subtype_rec(&b.schema, cx, assumptions))
        });
        let required = fb
            .iter()
            .filter(|b| b.required)
            .all(|b| fa.iter().any(|a| a.name == b.name && a.required));
        let extra_covered = fa
            .iter()
            .filter(|a| !fb.iter().any(|b| b.name == a.name))
            .all(|a| {
                db.iter().any(|(kb, vb)| {
                    matches!(kb, Schema::Str | Schema::Anything)
                        && a.schema.is_subtype_rec(vb, cx, assumptions)
                })
            });
        let defaults = da.iter().all(|(ka, va)| {
            db.iter().any(|(kb, vb)| {
                ka.is_subtype_rec(kb, cx, assumptions) && va.is_subtype_rec(vb, cx, assumptions)
            })
        });
        shared && required && extra_covered && defaults
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
pub(crate) const REGION_ALL: u16 = (1 << 12) - 1; // 6 scalar + 5 container kinds + OTHER

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
