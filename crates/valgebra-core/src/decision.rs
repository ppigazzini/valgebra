//! The decision procedures over the IR: emptiness, subtyping, equivalence, and
//! disjointness, with the leaf-relation oracle and the scalar region partition.

use crate::ir::{Constraint, Field, Schema, SeqKind, SeqRegex};
use std::cell::Cell;

/// The most decision steps one top-level query may take before it stops and
/// returns the conservative answer. Subtyping distributes over unions and
/// intersections and emptiness recurses the structural fragment, so a deeply
/// nested Boolean combination can demand work exponential in its depth; without
/// interning to share equal subtrees there is no cheap memo, so the procedure
/// bounds its own work. One budget is threaded through a whole top-level query —
/// subtyping and the emptiness checks it calls into share it, and the two
/// directions of an equivalence share it — so the bound cannot be escaped through
/// a side door or spent twice. The ceiling is far above any schema a real
/// annotation produces, so a legitimate relation is always decided; only an
/// adversarial schema built to blow up the decision reaches it, and there a
/// `false` ("not proven") is sound by the conservative contract. A complete,
/// work-sharing decision is the interning-based procedure.
const DECISION_BUDGET: u32 = 1_000_000;

/// Spend one unit of `budget`; returns `false` when it is already exhausted, the
/// signal a budgeted decision uses to stop and report the conservative answer.
fn spend(budget: &Cell<u32>) -> bool {
    match budget.get().checked_sub(1) {
        Some(remaining) => {
            budget.set(remaining);
            true
        }
        None => false,
    }
}

/// Intersect two region bitsets bottom-up. A missing region (an opaque, not
/// scalar-decidable child) makes the combination opaque too, matching
/// [`region_set`](Schema::region_set)'s `?` short-circuit.
fn and_region(a: Option<u8>, b: Option<u8>) -> Option<u8> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a & b),
        _ => None,
    }
}

/// Union two region bitsets bottom-up, with the same opaque-propagation rule as
/// [`and_region`].
fn or_region(a: Option<u8>, b: Option<u8>) -> Option<u8> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a | b),
        _ => None,
    }
}

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
        budget: &Cell<u32>,
    ) -> bool {
        match self {
            SeqRegex::Empty | SeqRegex::Star(_) => false,
            SeqRegex::Elem(schema) => schema.is_empty_rec(oracle, defs, visiting, budget),
            SeqRegex::Cat(parts) => parts
                .iter()
                .any(|p| p.language_is_empty(oracle, defs, visiting, budget)),
            SeqRegex::Or(parts) => parts
                .iter()
                .all(|p| p.language_is_empty(oracle, defs, visiting, budget)),
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
    pub(crate) fn region_set(&self) -> Option<u8> {
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
    ///
    /// The decision is bounded: a deeply nested adversarial schema that would take
    /// more than a fixed number of steps stops and returns `false`, so a `false`
    /// means "not proven empty within the work bound", not necessarily "non-empty".
    /// A real schema decides far inside the bound. The scalar-region check is folded
    /// bottom-up from each node's children, so nested Boolean structure is decided
    /// in time linear in its size rather than by re-walking each subtree per level.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.is_empty_rec(
            &NoLeafRelations,
            &[],
            &mut Vec::new(),
            &Cell::new(DECISION_BUDGET),
        )
    }

    /// Like [`is_empty`](Self::is_empty), but resolving recursive references
    /// through `defs`, so an uninhabited recursive schema — a mandatory
    /// self-reference with no base case — is detected. A reference `defs` does
    /// not resolve stays conservative (never reported empty).
    #[must_use]
    pub fn is_empty_under(&self, defs: &[Schema]) -> bool {
        self.is_empty_rec(
            &NoLeafRelations,
            defs,
            &mut Vec::new(),
            &Cell::new(DECISION_BUDGET),
        )
    }

    /// Like [`is_empty_under`](Self::is_empty_under), but with an `oracle` that
    /// can order the pool values behind refinement bounds, so an unsatisfiable
    /// bound conjunction (a lower bound above an upper bound) is detected.
    #[must_use]
    pub fn is_empty_with(&self, oracle: &dyn LeafRelations, defs: &[Schema]) -> bool {
        self.is_empty_rec(oracle, defs, &mut Vec::new(), &Cell::new(DECISION_BUDGET))
    }

    fn is_empty_rec(
        &self,
        oracle: &dyn LeafRelations,
        defs: &[Schema],
        visiting: &mut Vec<usize>,
        budget: &Cell<u32>,
    ) -> bool {
        self.empty_and_region(oracle, defs, visiting, budget).0
    }

    /// The emptiness verdict and the value-region bitset of `self`, decided in a
    /// single bottom-up pass: a Boolean node folds its children's already-computed
    /// regions in O(1) each instead of re-deriving its region by re-walking the
    /// whole subtree with [`region_set`](Self::region_set). The returned bitset is
    /// exactly what `region_set` would return (`None` off the scalar-decidable
    /// fragment), so emptiness on the scalar fragment is decided identically — but
    /// a deeply nested intersection is now decided in time linear in its size
    /// rather than quadratically (each level no longer re-walks the levels below).
    ///
    /// The work is bounded by the shared `budget`, so the region computation cannot
    /// run unbounded down a side door any more than the rest of the decision can;
    /// on exhaustion it returns the conservative "not proven empty" with an unknown
    /// region.
    fn empty_and_region(
        &self,
        oracle: &dyn LeafRelations,
        defs: &[Schema],
        visiting: &mut Vec<usize>,
        budget: &Cell<u32>,
    ) -> (bool, Option<u8>) {
        // Bound the work, sharing the budget with the caller (the subtyping
        // decision passes its own `cx.budget` in), so emptiness cannot escape the
        // ceiling subtyping advertises. Exhaustion returns "not proven empty".
        if !spend(budget) {
            return (false, None);
        }
        match self {
            // Scalar atoms and the lattice bounds carry a known region; their
            // emptiness is exactly "the region is empty".
            Schema::Nothing => (true, Some(0)),
            Schema::Anything => (false, Some(REGION_ALL)),
            Schema::NoneType => (false, Some(REGION_NONE)),
            Schema::Bool => (false, Some(REGION_BOOL)),
            Schema::Int => (false, Some(REGION_BOOL | REGION_INT)), // bool ⊆ int
            Schema::Float => (false, Some(REGION_FLOAT)),
            Schema::Str => (false, Some(REGION_STR)),
            Schema::Bytes => (false, Some(REGION_BYTES)),
            Schema::Ref(id) => {
                // A reference reached again while resolving it is a cycle: this
                // occurrence demands an infinite unfolding, so on its own it has
                // no finite inhabitant. A union base case or an optional or
                // starred position escapes before reaching here.
                if visiting.contains(id) {
                    return (true, None);
                }
                match defs.get(*id) {
                    Some(def) => {
                        visiting.push(*id);
                        let empty = def.is_empty_rec(oracle, defs, visiting, budget);
                        visiting.pop();
                        (empty, None)
                    }
                    None => (false, None),
                }
            }
            Schema::Seq { regex, .. } => (
                regex.language_is_empty(oracle, defs, visiting, budget),
                None,
            ),
            // A refinement is a subset of its base: an empty base empties it, and
            // so does an unsatisfiable bound conjunction (decided by the oracle).
            Schema::Refine { base, constraints } => {
                // The discreteness rule applies only when the base is exactly the
                // integer atom: `bool` is excluded (its two values are already
                // covered by the ordering check), and floats are dense.
                let int_discrete = base.type_tag() == Some(TypeTag::Int);
                let empty = base.is_empty_rec(oracle, defs, visiting, budget)
                    || bounds_unsatisfiable(constraints.iter(), oracle, int_discrete);
                (empty, None)
            }
            // An intersection is empty if a member is, if the scalar regions cancel,
            // or if the refinement bounds across its members cannot hold together.
            // Its region is the intersection of its members' regions — combined from
            // the children's results, never by re-walking the subtree.
            Schema::Intersection(members) => {
                let mut any_empty = false;
                let mut region = Some(REGION_ALL);
                for m in members {
                    let (empty, member_region) = m.empty_and_region(oracle, defs, visiting, budget);
                    any_empty |= empty;
                    region = and_region(region, member_region);
                }
                let empty = any_empty
                    || region == Some(0)
                    || intersection_bounds_unsatisfiable(members, oracle);
                (empty, region)
            }
            // A set or frozenset is *known* non-empty — the empty collection is
            // always a member — so it is never empty for a different reason than the
            // opaque wildcard below (which is non-empty only by conservatism).
            #[allow(clippy::match_same_arms)]
            Schema::Set(_) | Schema::FrozenSet(_) => (false, None),
            Schema::KeyedMap { fields, .. } => {
                let empty = fields.iter().any(|field| {
                    field.required && field.schema.is_empty_rec(oracle, defs, visiting, budget)
                });
                (empty, None)
            }
            // A structural-attribute schema requires every field, so an empty
            // required field's schema empties it — the same rule as a keyed map.
            // An uninhabited dataclass-style schema is detected here; the nominal
            // `isinstance` part stays opaque, so it never narrows to empty.
            Schema::Attrs { fields, .. } => {
                let empty = fields.iter().any(|field| {
                    field.required && field.schema.is_empty_rec(oracle, defs, visiting, budget)
                });
                (empty, None)
            }
            // A union is empty when every member is; its region is the union of the
            // members' regions, again folded from the children.
            Schema::Union(members) => {
                let mut all_empty = true;
                let mut region = Some(0);
                for m in members {
                    let (empty, member_region) = m.empty_and_region(oracle, defs, visiting, budget);
                    all_empty &= empty;
                    region = or_region(region, member_region);
                }
                (all_empty, region)
            }
            // A complement's region is the partition minus its inner's region; it is
            // empty exactly when that region is empty (`¬⊤ = ∅`).
            Schema::Complement(inner) => {
                let (_, inner_region) = inner.empty_and_region(oracle, defs, visiting, budget);
                let region = inner_region.map(|r| REGION_ALL & !r);
                (region == Some(0), region)
            }
            // The gradual `Any`, literals, and instances are not scalar-decidable:
            // an unknown region, never reported empty.
            _ => (false, None),
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
    ///
    /// The decision is bounded: an adversarial schema that would take more than a
    /// fixed number of steps stops and returns `false`, so a `false` can mean
    /// "not proven a subtype within the work bound". A real schema decides far
    /// inside the bound.
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
        let budget = Cell::new(DECISION_BUDGET);
        self.is_subtype_rec(
            other,
            SubtypeCx {
                oracle,
                defs,
                budget: &budget,
            },
            &mut Vec::new(),
        )
    }

    fn is_subtype_rec(
        &self,
        other: &Schema,
        cx: SubtypeCx<'_>,
        assumptions: &mut Vec<(Schema, Schema)>,
    ) -> bool {
        // Bound the total work: the distribution rules below can demand effort
        // exponential in the schema depth, so once the shared budget is spent the
        // decision stops and returns the conservative `false` rather than running
        // unbounded. A real annotation decides in a few steps; only an adversarial
        // schema reaches the ceiling.
        if !spend(cx.budget) {
            return false;
        }
        // Coinductive hypothesis: a goal already being proven on this path is
        // assumed to hold, so two recursive types are compared at their greatest
        // fixpoint rather than unfolded forever. The scan is empty (free) for a
        // non-recursive query; under recursion the stack holds one entry per
        // reference goal still being unfolded — bounded by the distinct goals
        // before a cycle and, overall, by the shared work budget. Every recorded
        // goal has a `Ref` on one side, so the structural compare rejects a
        // mismatched goal on the discriminant before walking either subtree.
        if assumptions.iter().any(|(a, b)| a == self && b == other) {
            return true;
        }
        // Scalar fragment: exact via the region partition. `a` is already a
        // subset of `REGION_ALL`, so `a & !b` needs no further mask: it holds
        // exactly when every region of `self` is also a region of `other`.
        if let (Some(a), Some(b)) = (self.region_set(), other.region_set()) {
            return a & !b == 0;
        }
        if self == other {
            return true;
        }
        self.subtype_decide(other, cx, assumptions)
    }

    /// The structural subtyping decision: the lattice, recursion, and
    /// constructor-matching rules. Reached from [`is_subtype_rec`] after the
    /// coinductive, scalar, identity, and memo fast paths.
    fn subtype_decide(
        &self,
        other: &Schema,
        cx: SubtypeCx<'_>,
        assumptions: &mut Vec<(Schema, Schema)>,
    ) -> bool {
        match (self, other) {
            // ∅ is a subset of every set, and every set is a subset of the top.
            (Schema::Nothing, _) | (_, Schema::Anything) => true,
            // A ⊆ ∅ exactly when A is empty, decided with the same oracle so a
            // refinement with unsatisfiable bounds is recognised here too.
            (_, Schema::Nothing) => {
                self.is_empty_rec(cx.oracle, cx.defs, &mut Vec::new(), cx.budget)
            }
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
            // Two structural-attribute schemas over the same class: a subtype when
            // it carries every attribute the supertype requires with a narrower
            // schema (width and depth; all attributes are required). Across
            // different classes the relation needs the nominal class hierarchy,
            // which the core cannot decide, so it stays conservative.
            (
                Schema::Attrs {
                    class_index: ca,
                    fields: fa,
                },
                Schema::Attrs {
                    class_index: cb,
                    fields: fb,
                },
            ) if ca == cb => {
                let a_by_name = field_index(fa);
                fb.iter().all(|b| {
                    a_by_name
                        .get(b.name.as_str())
                        .is_some_and(|a| a.schema.is_subtype_rec(&b.schema, cx, assumptions))
                })
            }
            // Complement is contravariant: ¬A ⊆ ¬B exactly when B ⊆ A.
            (Schema::Complement(a), Schema::Complement(b)) => b.is_subtype_rec(a, cx, assumptions),
            // A refinement is a subset of its base. Against another refinement the
            // base must subtype and every constraint of the supertype must hold of
            // every subtype value: either it appears verbatim, or it is entailed by
            // the subtype's bounds (a tighter lower/upper/length bound entails a
            // looser one, decided through the ordering oracle). A bound the oracle
            // cannot compare, and a non-order constraint, stay on the verbatim path.
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
                    && wide_cons.iter().all(|constraint| {
                        narrow_cons.contains(constraint)
                            || constraint_entailed(constraint, narrow_cons, cx.oracle)
                    })
            }
            // Against a non-refinement, a refinement inherits its base's supertypes.
            (Schema::Refine { base, .. }, _) => base.is_subtype_rec(other, cx, assumptions),
            // A leaf the structural rules cannot relate (an instance or literal):
            // defer to the oracle, conservative when it declines.
            _ => cx.oracle.leaf_subtype(self, other).unwrap_or(false),
        }
    }

    /// Whether `self` and `other` denote the same set — mutual inclusion.
    ///
    /// Like the relations it composes, the decision is bounded; a `false` can mean
    /// "not proven equivalent within the work bound" for an adversarial schema.
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
        // Both inclusion directions share one budget, so equivalence cannot spend
        // twice the ceiling, and its verdict does not depend on which direction
        // happened to allocate a fresh allowance first.
        let budget = Cell::new(DECISION_BUDGET);
        let cx = SubtypeCx {
            oracle,
            defs,
            budget: &budget,
        };
        self.is_subtype_rec(other, cx, &mut Vec::new())
            && other.is_subtype_rec(self, cx, &mut Vec::new())
    }
}

/// Threaded state for the subtyping decision: the leaf-relation oracle, the
/// definitions that resolve recursive references, and the remaining work budget
/// shared across the whole query. The budget counts decision steps down to zero,
/// at which point the procedure stops and returns the conservative `false`,
/// bounding the cost of a deeply nested Boolean combination.
#[derive(Clone, Copy)]
struct SubtypeCx<'a> {
    oracle: &'a dyn LeafRelations,
    defs: &'a [Schema],
    budget: &'a Cell<u32>,
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

    /// Whether no integer lies between the pool values at `lo` and `hi`, under the
    /// strictness of each bound (`lo_strict` excludes `lo`, `hi_strict` excludes
    /// `hi`). The core asks this only for an integer-discrete refinement base, so a
    /// `Some(true)` proves the interval admits no integer and the refinement is
    /// empty. `None` leaves the discreteness rule conservative — the default, so a
    /// core with no value oracle never decides on integer adjacency.
    fn no_int_between(
        &self,
        _lo: usize,
        _lo_strict: bool,
        _hi: usize,
        _hi_strict: bool,
    ) -> Option<bool> {
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
fn bounds_unsatisfiable<'a>(
    constraints: impl Iterator<Item = &'a Constraint> + Clone,
    oracle: &dyn LeafRelations,
    int_discrete: bool,
) -> bool {
    use core::cmp::Ordering;
    let min_len = constraints
        .clone()
        .filter_map(|c| match c {
            Constraint::MinLen(n) => Some(*n),
            _ => None,
        })
        .max();
    let max_len = constraints
        .clone()
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
        match oracle.compare(lo, hi) {
            Some(Ordering::Greater) => return true,
            Some(Ordering::Equal) => return lo_strict || hi_strict,
            _ => {}
        }
        // An integer-discrete base bounds the integers in the interval, so the
        // refinement is empty when no integer lies between the bounds even though
        // the endpoints themselves are ordered `lo < hi` — `Annotated[int, Gt(0),
        // Lt(1)]` admits no value. The oracle answers only for a real numeric
        // pair and stays `None` otherwise, so floats and incomparable bounds keep
        // the interval conservatively non-empty.
        if int_discrete && oracle.no_int_between(lo, lo_strict, hi, hi_strict) == Some(true) {
            return true;
        }
    }
    false
}

/// Whether a single supertype refinement constraint is *entailed* by the subtype's
/// constraint set: every value satisfying all of `narrow` also satisfies `wide`.
/// Order and length bounds entail by value (a tighter lower bound entails a looser
/// one, dually for upper and length), decided through the ordering `oracle`; the
/// remaining kinds (`MultipleOf`, `Predicate`, `Regex`) have no sound value
/// entailment and require the constraint to appear verbatim, handled by the
/// caller's syntactic-containment check. A bound the oracle cannot compare is not
/// entailed (conservative).
fn constraint_entailed(
    wide: &Constraint,
    narrow: &[Constraint],
    oracle: &dyn LeafRelations,
) -> bool {
    use core::cmp::Ordering;
    let ge = |o: Option<Ordering>| matches!(o, Some(Ordering::Greater | Ordering::Equal));
    let gt = |o: Option<Ordering>| matches!(o, Some(Ordering::Greater));
    let le = |o: Option<Ordering>| matches!(o, Some(Ordering::Less | Ordering::Equal));
    let lt = |o: Option<Ordering>| matches!(o, Some(Ordering::Less));
    match wide {
        // x >= w holds if the subtype forces a lower bound at value >= w.
        Constraint::Ge(w) => narrow.iter().any(|c| match c {
            Constraint::Ge(n) | Constraint::Gt(n) => ge(oracle.compare(*n, *w)),
            _ => false,
        }),
        // x > w holds from Gt(n>=w), or Ge(n>w).
        Constraint::Gt(w) => narrow.iter().any(|c| match c {
            Constraint::Gt(n) => ge(oracle.compare(*n, *w)),
            Constraint::Ge(n) => gt(oracle.compare(*n, *w)),
            _ => false,
        }),
        // x <= w holds if the subtype forces an upper bound at value <= w.
        Constraint::Le(w) => narrow.iter().any(|c| match c {
            Constraint::Le(n) | Constraint::Lt(n) => le(oracle.compare(*n, *w)),
            _ => false,
        }),
        // x < w holds from Lt(n<=w), or Le(n<w).
        Constraint::Lt(w) => narrow.iter().any(|c| match c {
            Constraint::Lt(n) => le(oracle.compare(*n, *w)),
            Constraint::Le(n) => lt(oracle.compare(*n, *w)),
            _ => false,
        }),
        // Length bounds compare by their raw counts.
        Constraint::MinLen(w) => narrow
            .iter()
            .any(|c| matches!(c, Constraint::MinLen(n) if n >= w)),
        Constraint::MaxLen(w) => narrow
            .iter()
            .any(|c| matches!(c, Constraint::MaxLen(n) if n <= w)),
        // No sound value entailment without an exact match (handled by the caller).
        Constraint::MultipleOf(_) | Constraint::Predicate(_) | Constraint::Regex(_) => false,
    }
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

/// Whether the refinement constraints of the intersection's **directly refined
/// members** cannot hold together. A value in the intersection satisfies every
/// member, so the constraints of each top-level `Refine` member apply to it at
/// once. This gathers only those top-level constraints — a refinement nested
/// inside a member (say under a union arm) is not collected here; the decision
/// stays sound, since missing a contradiction only forgoes reporting emptiness,
/// never reports a non-empty intersection empty.
fn intersection_bounds_unsatisfiable(members: &[Schema], oracle: &dyn LeafRelations) -> bool {
    // Gather the top-level refine members' constraints by reference — no clone, so
    // a `Regex` constraint's pattern string is not copied per intersection node.
    let merged: Vec<&Constraint> = members
        .iter()
        .filter_map(|m| match m {
            Schema::Refine { constraints, .. } => Some(constraints.as_slice()),
            _ => None,
        })
        .flatten()
        .collect();
    // The integer-discreteness rule stays off across an intersection: the base of
    // the merged bounds is the members' meet, not a single declared atom, so the
    // narrow single-`Refine` discreteness premise does not transfer here. Bound
    // contradiction across members is still decided through the ordering oracle.
    !merged.is_empty() && bounds_unsatisfiable(merged.iter().copied(), oracle, false)
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
    // side is then just its fixed prefix. Emptiness is decided with the same
    // oracle and definitions as the rest of the decision, so a tail empty only
    // under a refinement bound or an uninhabited recursive reference is
    // recognised here too, consistent with the context-aware recursion around it.
    let ta =
        ta.filter(|element| !element.is_empty_rec(cx.oracle, cx.defs, &mut Vec::new(), cx.budget));
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
/// 3. **Mixed record-and-catch-all ≤ mixed** (general): each shared field narrows
///    and respects required-ness; each field `a` declares that `b` does not is
///    covered by `b`'s catch-all; each field `b` declares that `a` lacks is
///    governed by `a`'s catch-all — decidable only when it is **optional** (a
///    catch-all guarantees a key's value type, never its presence, so a required
///    such field stays `false`) and every catch-all value of `a` fits it; and
///    every catch-all clause of `a` is subsumed by one of `b`.
///
/// Sound throughout — a required supertype field the subtype cannot guarantee
/// present, or a clause an oracle cannot relate, is reported `false`, never an
/// unsound `true`.
fn keyed_map_subtype(
    fa: &[Field],
    da: &[(Schema, Schema)],
    fb: &[Field],
    db: &[(Schema, Schema)],
    cx: SubtypeCx<'_>,
    assumptions: &mut Vec<(Schema, Schema)>,
) -> bool {
    // Index both field lists by name once, so the cross-list lookups below are O(1)
    // each rather than a fresh linear scan per field (O(fields²) per comparison).
    let a_by_name = field_index(fa);
    let b_by_name = field_index(fb);
    if da.is_empty() {
        // A closed record carries only its declared fields, so it is a subtype
        // when every field maps into a like-named field of the supertype (width
        // and depth) and the supertype's required fields are required here too. A
        // field the supertype does not declare would need the supertype's
        // catch-all to cover that exact key name, which stays conservative.
        let width_and_depth = fa.iter().all(|a| {
            b_by_name
                .get(a.name.as_str())
                .is_some_and(|b| a.schema.is_subtype_rec(&b.schema, cx, assumptions))
        });
        let required = fb
            .iter()
            .filter(|b| b.required)
            .all(|b| a_by_name.get(b.name.as_str()).is_some_and(|a| a.required));
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
    } else {
        // General mixed record-and-catch-all subtyping (`a` carries a catch-all and
        // is not the pure-mapping case). Every supertype field is checked against
        // `a`: a field `a` declares is matched field-wise; a field `a` lacks is
        // governed by `a`'s catch-all.
        let fields_ok = fb.iter().all(|b_field| {
            match a_by_name.get(b_field.name.as_str()) {
                // Shared field: it must narrow in depth, and a field `b` requires
                // must be required in `a` too.
                Some(a_field) => {
                    a_field
                        .schema
                        .is_subtype_rec(&b_field.schema, cx, assumptions)
                        && (!b_field.required || a_field.required)
                }
                // A field `b` declares that `a` lacks: a catch-all guarantees a
                // key's value type but never its presence, so a *required* such
                // field stays undecided; an *optional* one holds when every value
                // `a`'s catch-all could place at that key fits `b`'s field schema.
                None => {
                    !b_field.required
                        && da
                            .iter()
                            .all(|(_, va)| va.is_subtype_rec(&b_field.schema, cx, assumptions))
                }
            }
        });
        // Each field `a` declares that `b` does not is read by `b` through its
        // catch-all, so a `str`/`anything`-keyed clause of `b` must cover it.
        let extra_covered = fa
            .iter()
            .filter(|a_field| !b_by_name.contains_key(a_field.name.as_str()))
            .all(|a_field| {
                db.iter().any(|(kb, vb)| {
                    matches!(kb, Schema::Str | Schema::Anything)
                        && a_field.schema.is_subtype_rec(vb, cx, assumptions)
                })
            });
        // Every catch-all clause of `a` (governing its non-field keys) is subsumed
        // by a clause of `b` with both key and value narrower.
        let defaults = da.iter().all(|(ka, va)| {
            db.iter().any(|(kb, vb)| {
                ka.is_subtype_rec(kb, cx, assumptions) && va.is_subtype_rec(vb, cx, assumptions)
            })
        });
        fields_ok && extra_covered && defaults
    }
}

/// Index a field list by name for O(1) cross-list lookup during subtyping.
///
/// Unique field names are a hard caller invariant: `collect` into a map keeps the
/// last entry per key, so a duplicate name would silently shadow an earlier field
/// and could make the `required`/width checks that consume this index unsound. The
/// frontend rejects duplicates; the `debug_assert` makes that dependency explicit
/// and catches a malformed IR in debug rather than deciding on a shadowed field.
fn field_index(fields: &[Field]) -> std::collections::HashMap<&str, &Field> {
    let index: std::collections::HashMap<&str, &Field> =
        fields.iter().map(|f| (f.name.as_str(), f)).collect();
    debug_assert_eq!(
        index.len(),
        fields.len(),
        "record has duplicate field names; the frontend must reject them"
    );
    index
}

/// The value universe partitioned into mutually-disjoint regions, so a Boolean
/// combination of scalar atoms denotes a set computed by bitset operations. The
/// scalar atoms occupy `NONE`..`BYTES` (with `int` covering `BOOL | INT`); the
/// container kinds and `OTHER` complete the partition so a complement of a
/// scalar correctly includes every non-scalar value.
const REGION_NONE: u8 = 1 << 0;
const REGION_BOOL: u8 = 1 << 1;
const REGION_INT: u8 = 1 << 2; // int values other than bool
const REGION_FLOAT: u8 = 1 << 3;
const REGION_STR: u8 = 1 << 4;
const REGION_BYTES: u8 = 1 << 5;
// Bit 6 is the single non-scalar region: every value that is not one of the six
// scalar kinds (containers, instances, callables, everything else) lumped
// together. No atom ever names it on its own, so the non-scalar kinds need no
// further bits; it exists so the complement of a scalar includes every non-scalar
// value, which keeps emptiness sound (e.g. the meet of all six scalar complements
// is the non-empty non-scalar region, not the empty set).
// 6 scalar regions + the non-scalar region: 7 of `u8`'s 8 bits. The bitset type
// is sized to the partition, so adding an 8th region still fits but a 9th would
// overflow `1 << 8` at compile time — the type is the guard against a silent wrap.
pub(crate) const REGION_ALL: u8 = (1 << 7) - 1;

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
