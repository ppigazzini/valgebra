//! Shared fuzz scaffolding: a depth-bounded `Arbitrary` generator that maps raw
//! fuzzer bytes onto the core schema IR, and the sound invariants the targets
//! assert. Indices into the (absent) object pool are kept small so distinct
//! atoms collide often enough for the relational checks to bite. Recursion is
//! capped so a target exercises algebra and decision logic rather than the
//! stack-depth limits, which the adversarial-bounds work measures separately.
//!
//! The targets assert procedure-agnostic *laws* — panic-freedom, `simplify`
//! idempotence, and relational soundness (reflexivity, top/bottom bounds,
//! equivalence as mutual inclusion) — over the **full** IR, including the opaque
//! fragment (instances, gradual `Any`, recursion) a value oracle cannot model.
//! Value-level denotation preservation is deliberately not re-checked here: it is
//! oracle-tested against an independent membership predicate over the decidable
//! fragment in the core law suite (`simplify_preserves_membership_over_values`).
//! Duplicating it here would only cover the sub-fragment the fuzzer's wide
//! generator is built to exceed, so the split is intentional, not a gap.

use arbitrary::{Arbitrary, Result, Unstructured};
use valgebra_core::{
    ClassIx, ConstIx, Constraint, Field, MapClause, OperandIx, PredIx, Schema, SeqKind, SeqShape,
};

const NAMES: [&str; 4] = ["a", "b", "c", "d"];
const PATTERNS: [&str; 3] = ["a+", "[0-9]*", "x"];

/// A small pool slot. Deliberately narrow so distinct atoms collide often enough
/// for the relational checks to bite.
///
/// The four index spaces are minted through their own constructors here, exactly
/// as the frontend does: a fuzzer that produced bare integers would be building
/// schemas the frontend cannot, and the laws under test are about the schemas it
/// can.
fn small(u: &mut Unstructured) -> Result<usize> {
    Ok(usize::from(u.arbitrary::<u8>()?) % 3)
}

fn operand(u: &mut Unstructured) -> Result<OperandIx> {
    Ok(OperandIx::new(small(u)?))
}

fn count(u: &mut Unstructured, max: usize) -> Result<usize> {
    Ok(usize::from(u.arbitrary::<u8>()?) % max)
}

fn build_constraint(u: &mut Unstructured) -> Result<Constraint> {
    Ok(match u.arbitrary::<u8>()? % 9 {
        0 => Constraint::Ge(operand(u)?),
        1 => Constraint::Gt(operand(u)?),
        2 => Constraint::Le(operand(u)?),
        3 => Constraint::Lt(operand(u)?),
        4 => Constraint::MinLen(count(u, 8)?),
        5 => Constraint::MaxLen(count(u, 8)?),
        6 => Constraint::MultipleOf(operand(u)?),
        7 => Constraint::Predicate(PredIx::new(small(u)?)),
        _ => Constraint::Regex(PATTERNS[usize::from(u.arbitrary::<u8>()?) % PATTERNS.len()].into()),
    })
}

/// Build one sequence shape: a prefix of up to four element schemas, and a tail
/// on half the draws.
///
/// The generator covers the shape's whole space, which is what it did not do
/// while the sequence body was a regular expression: three of its five branches
/// then built alternations and nested repetitions that nothing else in the tree
/// produces, so the fuzzer spent its budget on languages no value could reach.
fn build_shape(u: &mut Unstructured, depth: u32) -> Result<SeqShape> {
    let arity = if depth == 0 || u.is_empty() {
        0
    } else {
        count(u, 4)?
    };
    let element_depth = depth.saturating_sub(1);
    let mut prefix = Vec::with_capacity(arity);
    for _ in 0..arity {
        prefix.push(build_schema(u, element_depth)?);
    }
    let tail = if u.arbitrary::<bool>()? {
        Some(Box::new(build_schema(u, element_depth)?))
    } else {
        None
    };
    Ok(SeqShape { prefix, tail })
}

/// Build one schema from the fuzzer's bytes, bounded by `depth` recursion levels.
pub fn build_schema(u: &mut Unstructured, depth: u32) -> Result<Schema> {
    // Atoms are always reachable; composites only while the depth budget holds.
    // The universe has one arm, not two: `typing.Any` is the top with a spelling
    // the term keeps for `repr`, and a fuzz target reads verdicts, not spellings.
    let atoms = 10u8;
    let composites = 8u8;
    let span = if depth == 0 || u.is_empty() {
        atoms
    } else {
        atoms + composites
    };
    Ok(match u.arbitrary::<u8>()? % span {
        0 => Schema::ANYTHING,
        1 => Schema::Nothing,
        2 => Schema::NoneType,
        3 => Schema::Bool,
        4 => Schema::Int,
        5 => Schema::Float,
        6 => Schema::Str,
        7 => Schema::Bytes,
        8 => Schema::Literal(ConstIx::new(small(u)?)),
        9 => Schema::Instance(ClassIx::new(small(u)?)),
        10 => {
            let n = 1 + count(u, 3)?;
            let mut members = Vec::with_capacity(n);
            for _ in 0..n {
                members.push(build_schema(u, depth - 1)?);
            }
            Schema::Union(members)
        }
        11 => {
            let n = 1 + count(u, 3)?;
            let mut members = Vec::with_capacity(n);
            for _ in 0..n {
                members.push(build_schema(u, depth - 1)?);
            }
            Schema::Intersection(members)
        }
        12 => Schema::Complement(Box::new(build_schema(u, depth - 1)?)),
        13 => {
            let base = Box::new(build_schema(u, depth - 1)?);
            let n = count(u, 3)?;
            let mut constraints = Vec::with_capacity(n);
            for _ in 0..n {
                constraints.push(build_constraint(u)?);
            }
            Schema::Refine { base, constraints }
        }
        14 => Schema::Set(Box::new(build_schema(u, depth - 1)?)),
        15 => Schema::FrozenSet(Box::new(build_schema(u, depth - 1)?)),
        16 => Schema::Seq {
            container: if u.arbitrary()? {
                SeqKind::List
            } else {
                SeqKind::Tuple
            },
            shape: build_shape(u, depth - 1)?,
        },
        _ => {
            // Unique field names are a caller invariant the frontend guarantees
            // (a record's fields come from a Python type-hints dict, keyed by
            // name), and the decision procedures assert it. Drop a field whose
            // name a sibling already took, while still consuming its schema bytes
            // so the byte-to-IR mapping stays total and deterministic.
            let nf = count(u, 3)?;
            let mut fields: Vec<Field> = Vec::with_capacity(nf);
            for _ in 0..nf {
                let name = NAMES[usize::from(u.arbitrary::<u8>()?) % NAMES.len()];
                let schema = build_schema(u, depth - 1)?;
                let required = u.arbitrary()?;
                if fields.iter().any(|f| f.name == name) {
                    continue;
                }
                fields.push(Field {
                    name: name.into(),
                    schema,
                    required,
                });
            }
            let nd = count(u, 3)?;
            let mut defaults = Vec::with_capacity(nd);
            for _ in 0..nd {
                defaults.push(MapClause {
                    key: build_schema(u, depth - 1)?,
                    value: build_schema(u, depth - 1)?,
                });
            }
            Schema::KeyedMap { fields, defaults }
        }
    })
}

/// One fuzzer-built schema (depth 5).
#[derive(Debug)]
pub struct SchemaPlan(pub Schema);

impl<'a> Arbitrary<'a> for SchemaPlan {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        Ok(Self(build_schema(u, 5)?))
    }
}

/// A pair of fuzzer-built schemas for the relational invariants.
#[derive(Debug)]
pub struct SchemaPair(pub Schema, pub Schema);

impl<'a> Arbitrary<'a> for SchemaPair {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        Ok(Self(build_schema(u, 4)?, build_schema(u, 4)?))
    }
}

/// Invariants that hold of `simplify` for every schema: it terminates without
/// panicking and reaches a fixpoint after one application (a lattice normal form
/// is stable under re-simplification).
///
/// Membership preservation is *not* asserted here through `is_equivalent`: that
/// decision is deliberately conservative, so it can answer `false` for a genuine
/// equality `simplify` produced (an empty-constraint refinement equals its base,
/// for instance). The denotation-preservation of `simplify` is oracle-tested
/// against an independent membership predicate in the core's law suite; the
/// fuzzer's job here is panic-freedom and idempotence over the full IR fragment.
pub fn check_simplify(schema: &Schema) {
    let once = schema.simplify();
    let twice = once.simplify();
    assert_eq!(once, twice, "simplify is not idempotent on {schema:?}");
}

/// Sound relational invariants any subtype/equivalence/emptiness procedure must
/// satisfy. A violation is a defect, not conservatism.
pub fn check_relations(a: &Schema, b: &Schema) {
    // Reflexivity of the order and the equivalence it induces.
    assert!(a.is_subtype_of(a), "subtyping not reflexive on {a:?}");
    assert!(a.is_equivalent(a), "equivalence not reflexive on {a:?}");
    // Top and bottom bound every schema. Stated over the ATOMS...
    assert!(
        a.is_subtype_of(&Schema::ANYTHING),
        "{a:?} not below the top"
    );
    assert!(Schema::Nothing.is_subtype_of(a), "bottom not below {a:?}");
    // ...and over the PROPERTY, which is the law the atoms are only one case of.
    //
    // Written over the atoms alone this pair confirms the syntactic rule using
    // the syntactic rule: `Nothing` is hardcoded on the left, so a schema that
    // denotes the empty set without being spelled `Nothing` is never the
    // subject, however long the fuzzer runs. Both sides are generated below, so
    // a cancelling intersection and an uninhabited record are reached -- and
    // they are common in this generator's space, not rare.
    if a.is_empty() {
        assert!(
            a.is_subtype_of(b),
            "empty {a:?} not below {b:?}: the empty set is a subset of every set"
        );
    }
    if Schema::Complement(Box::new(b.clone())).is_empty() {
        assert!(
            a.is_subtype_of(b),
            "{a:?} not below universal {b:?}: every set is a subset of the universe"
        );
    }
    // Equivalence is exactly mutual inclusion.
    let sub_ab = a.is_subtype_of(b);
    let sub_ba = b.is_subtype_of(a);
    if a.is_equivalent(b) {
        assert!(
            sub_ab && sub_ba,
            "equivalent {a:?} and {b:?} are not mutually included"
        );
    }
    if sub_ab && sub_ba {
        assert!(
            a.is_equivalent(b),
            "mutually included {a:?} and {b:?} are not equivalent"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every record the generator emits has unique field names, the caller
    /// invariant the decision procedures assert. Walking a wide range of byte
    /// inputs covers nested records and the records nested inside catch-all
    /// clauses, the shapes the live crash seed exercises.
    fn assert_unique_field_names(schema: &Schema) {
        match schema {
            Schema::KeyedMap { fields, defaults } => {
                let mut seen = std::collections::HashSet::new();
                for field in fields {
                    assert!(
                        seen.insert(field.name.as_str()),
                        "duplicate field name {:?} in {schema:?}",
                        field.name
                    );
                    assert_unique_field_names(&field.schema);
                }
                for clause in defaults {
                    assert_unique_field_names(&clause.key);
                    assert_unique_field_names(&clause.value);
                }
            }
            Schema::Union(members) | Schema::Intersection(members) => {
                members.iter().for_each(assert_unique_field_names);
            }
            Schema::Complement(inner)
            | Schema::Set(inner)
            | Schema::FrozenSet(inner)
            | Schema::Refine { base: inner, .. } => assert_unique_field_names(inner),
            Schema::Seq { shape, .. } => {
                for element in shape.prefix.iter().chain(shape.tail.as_deref()) {
                    assert_unique_field_names(element);
                }
            }
            _ => {}
        }
    }

    #[test]
    fn generator_records_have_unique_field_names() {
        // A spread of byte fills drives the generator down different branches;
        // 0xe3-heavy fills mirror the live crash seed that first tripped this.
        for fill in [0x00u8, 0x01, 0x1f, 0x31, 0x9a, 0xb7, 0xe3, 0xff] {
            for len in [4usize, 16, 64, 256, 1024] {
                let bytes = vec![fill; len];
                let mut u = Unstructured::new(&bytes);
                while let Ok(schema) = build_schema(&mut u, 4) {
                    assert_unique_field_names(&schema);
                    if u.is_empty() {
                        break;
                    }
                }
            }
        }
    }
}
