//! The membership-preserving simplifier: the lattice-law normalisation of the
//! IR (flattening, identities, De Morgan, deduplication).

use crate::decision::REGION_ALL;
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

fn has_complementary_pair(members: &[Schema]) -> bool {
    members.iter().any(|member| {
        if let Schema::Complement(inner) = member {
            !matches!(**inner, Schema::Dynamic) && members.iter().any(|other| other == &**inner)
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
