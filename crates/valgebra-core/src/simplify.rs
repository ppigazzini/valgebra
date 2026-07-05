//! The membership-preserving simplifier: the lattice-law normalisation of the
//! IR (flattening, identities, De Morgan, deduplication).

use crate::decision::{REGION_ALL, has_complementary_pair, has_disjoint_pair};
use crate::ir::{Constraint, Field, Schema, SeqRegex};

impl SeqRegex {
    fn simplify(&self) -> SeqRegex {
        self.map_elems(&Schema::simplify)
    }
}

impl Schema {
    /// Return a membership-equivalent schema reduced by the lattice laws.
    ///
    /// Every rewrite preserves the set of admitted values: nested unions and
    /// intersections are flattened, members sorted and deduplicated
    /// (associativity, commutativity, idempotence), the top and bottom
    /// identities are applied, complements are pushed inward to negation-normal
    /// form (De Morgan) and double negations cancelled. `Any` (gradual) is left
    /// untouched: it is never treated as the top. Conservative by design — it
    /// never claims an equivalence it cannot justify structurally.
    ///
    /// This is purely the lattice-law normal form: it does **not** run the
    /// emptiness/subtyping decision. So an intersection that is empty only by a
    /// deeper argument — contradictory refinement bounds like
    /// `int & Ge(10) & Le(0)`, or two disjoint refined bases — survives
    /// simplification unchanged, even though [`is_empty`](Self::is_empty) reports
    /// it empty. A caller must not treat a simplified schema as fully reduced and
    /// then read membership relations off its structure;
    /// [`is_empty`](Self::is_empty), [`is_subtype_of`](Self::is_subtype_of), and
    /// [`is_equivalent`](Self::is_equivalent) are the stronger, separate decision
    /// procedures, deciding a wider fragment than `simplify` folds.
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
            Schema::Attrs {
                class_index,
                fields,
            } => Schema::Attrs {
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
            Schema::Refine { base, constraints } => {
                canonical_refine(base.simplify(), constraints.clone())
            }
            Schema::Union(members) => simplify_union(members),
            Schema::Intersection(members) => simplify_intersection(members),
            Schema::Complement(inner) => simplify_complement(inner),
            // Atoms (including Any and Literal/Instance) reduce to themselves.
            other => other.clone(),
        }
    }
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

/// Build a refinement in canonical form from an already-simplified `base`.
///
/// A refinement of a refinement flattens into one refinement over the shared
/// base (`{x in {y in b | c1} | c2}` is `{x in b | c1 and c2}`), and the merged
/// constraints are sorted and deduplicated so two refinements that list the same
/// constraints in any order, or repeat one, share a single normal form. This is
/// idempotence and commutativity over the conjunction of constraints, the same
/// laws `simplify` already applies to union and intersection members. A
/// refinement left with no constraints is exactly its base.
fn canonical_refine(mut base: Schema, mut constraints: Vec<Constraint>) -> Schema {
    while let Schema::Refine {
        base: inner_base,
        constraints: mut inner_constraints,
    } = base
    {
        inner_constraints.append(&mut constraints);
        constraints = inner_constraints;
        base = *inner_base;
    }
    constraints.sort_unstable();
    constraints.dedup();
    if constraints.is_empty() {
        base
    } else {
        Schema::Refine {
            base: Box::new(base),
            constraints,
        }
    }
}

/// Simplify a union from its raw members: each member is normalised lazily inside
/// the loop, so the top identity short-circuits the moment a member reduces to it,
/// without normalising the members that follow. This is the entry from the top of
/// `simplify`, where the members are not yet normal.
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
    finish_union(flat)
}

/// Collapse a union of already-normal members without re-normalising them. The
/// De Morgan path builds these from members `simplify` has already produced, so
/// re-running `simplify` on each would repeat the whole subtree's work once per
/// level it is nested under — the exponential `simplify` blowup. Pushing the
/// complement inward over already-normal members keeps the pass linear.
fn union_of_simplified(members: Vec<Schema>) -> Schema {
    let mut flat = Vec::new();
    for member in members {
        match member {
            Schema::Anything => return Schema::Anything,
            Schema::Nothing => {}
            Schema::Union(inner) => flat.extend(inner),
            other => flat.push(other),
        }
    }
    finish_union(flat)
}

/// Sort, dedup, apply the union completeness laws, and collapse a flattened set
/// of normal union members to a single schema. Shared by both union entries.
fn finish_union(mut flat: Vec<Schema>) -> Schema {
    flat.sort();
    flat.dedup();
    // X ∪ ¬X is everything, as is ¬A ∪ ¬B for disjoint A and B; and a union is
    // everything once its scalar-decidable members alone cover every region of
    // the value universe (opaque members can only add coverage, so they are
    // ignored — keeping the decision independent of grouping). `region_set` walks
    // each already-flattened member once, so this fold is linear in the flattened
    // breadth; nested unions are flattened above, and a deep region-decidable nest
    // collapses to an atom under the laws here before it can accumulate depth.
    let covers_universe = flat
        .iter()
        .filter_map(Schema::region_set)
        .fold(0u8, |acc, regions| acc | regions)
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

/// Simplify an intersection from its raw members: each member is normalised
/// lazily inside the loop, so the bottom identity short-circuits the moment a
/// member reduces to it, without normalising the members that follow. This is the
/// entry from the top of `simplify`, where the members are not yet normal.
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
    finish_intersection(flat)
}

/// Collapse an intersection of already-normal members without re-normalising
/// them — the De Morgan dual of [`union_of_simplified`], used along the complement
/// path so a nested complement is not re-simplified once per level.
fn intersection_of_simplified(members: Vec<Schema>) -> Schema {
    let mut flat = Vec::new();
    for member in members {
        match member {
            Schema::Nothing => return Schema::Nothing,
            Schema::Anything => {}
            Schema::Intersection(inner) => flat.extend(inner),
            other => flat.push(other),
        }
    }
    finish_intersection(flat)
}

/// Sort, dedup, apply the intersection emptiness laws, and collapse a flattened
/// set of normal intersection members. Shared by both intersection entries.
fn finish_intersection(mut flat: Vec<Schema>) -> Schema {
    flat.sort();
    flat.dedup();
    // X ∩ ¬X is empty, as is an intersection of two provably disjoint members;
    // and an intersection is empty once its scalar-decidable members alone
    // cancel to no region (opaque members only narrow further, so they are
    // ignored — keeping the decision independent of grouping). As in `finish_union`,
    // `region_set` walks each already-flattened member once, so this fold is linear
    // in the flattened breadth.
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

/// Simplify a complement from its raw inner schema: normalise the inner once, then
/// push the complement inward. The inner is normalised lazily here (its own
/// short-circuits intact); the De Morgan arms then operate on its already-normal
/// members through [`complement_of_simplified`], so no member is normalised twice.
fn simplify_complement(inner: &Schema) -> Schema {
    complement_of_simplified(inner.simplify())
}

/// Push a complement to negation-normal form and cancel double negation over an
/// already-normal inner schema. The De Morgan arms complement each already-normal
/// member (which only inspects its top constructor) and re-collapse the result,
/// so the traversal stays linear in the tree size — never re-running `simplify` on
/// a member, which is the rewrite that made nested complements blow up.
fn complement_of_simplified(inner: Schema) -> Schema {
    match inner {
        Schema::Complement(x) => *x,
        Schema::Anything => Schema::Nothing,
        Schema::Nothing => Schema::Anything,
        Schema::Union(members) => intersection_of_simplified(complement_each(members)),
        Schema::Intersection(members) => union_of_simplified(complement_each(members)),
        other => Schema::Complement(Box::new(other)),
    }
}

/// Complement each already-normal member, keeping the result normal: a member that
/// is itself a complement cancels, a union or intersection pushes the complement
/// further inward by De Morgan, and an atom gains one complement.
fn complement_each(members: Vec<Schema>) -> Vec<Schema> {
    members.into_iter().map(complement_of_simplified).collect()
}
