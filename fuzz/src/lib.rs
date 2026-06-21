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
use valgebra_core::{Constraint, Field, Schema, SeqKind, SeqRegex};

const NAMES: [&str; 4] = ["a", "b", "c", "d"];
const PATTERNS: [&str; 3] = ["a+", "[0-9]*", "x"];

fn small(u: &mut Unstructured) -> Result<usize> {
    Ok(usize::from(u.arbitrary::<u8>()?) % 3)
}

fn count(u: &mut Unstructured, max: usize) -> Result<usize> {
    Ok(usize::from(u.arbitrary::<u8>()?) % max)
}

fn build_constraint(u: &mut Unstructured) -> Result<Constraint> {
    Ok(match u.arbitrary::<u8>()? % 9 {
        0 => Constraint::Ge(small(u)?),
        1 => Constraint::Gt(small(u)?),
        2 => Constraint::Le(small(u)?),
        3 => Constraint::Lt(small(u)?),
        4 => Constraint::MinLen(count(u, 8)?),
        5 => Constraint::MaxLen(count(u, 8)?),
        6 => Constraint::MultipleOf(small(u)?),
        7 => Constraint::Predicate(small(u)?),
        _ => Constraint::Regex(PATTERNS[usize::from(u.arbitrary::<u8>()?) % PATTERNS.len()].into()),
    })
}

fn build_regex(u: &mut Unstructured, depth: u32) -> Result<SeqRegex> {
    if depth == 0 || u.is_empty() {
        return Ok(match u.arbitrary::<u8>()? % 2 {
            0 => SeqRegex::Empty,
            _ => SeqRegex::Elem(Box::new(build_schema(u, 0)?)),
        });
    }
    Ok(match u.arbitrary::<u8>()? % 5 {
        0 => SeqRegex::Empty,
        1 => SeqRegex::Elem(Box::new(build_schema(u, depth - 1)?)),
        2 => {
            let n = 1 + count(u, 3)?;
            let mut parts = Vec::with_capacity(n);
            for _ in 0..n {
                parts.push(build_regex(u, depth - 1)?);
            }
            SeqRegex::Cat(parts)
        }
        3 => {
            let n = 1 + count(u, 3)?;
            let mut parts = Vec::with_capacity(n);
            for _ in 0..n {
                parts.push(build_regex(u, depth - 1)?);
            }
            SeqRegex::Or(parts)
        }
        _ => SeqRegex::Star(Box::new(build_regex(u, depth - 1)?)),
    })
}

/// Build one schema from the fuzzer's bytes, bounded by `depth` recursion levels.
pub fn build_schema(u: &mut Unstructured, depth: u32) -> Result<Schema> {
    // Atoms are always reachable; composites only while the depth budget holds.
    let atoms = 11u8;
    let composites = 8u8;
    let span = if depth == 0 || u.is_empty() {
        atoms
    } else {
        atoms + composites
    };
    Ok(match u.arbitrary::<u8>()? % span {
        0 => Schema::Anything,
        1 => Schema::Dynamic,
        2 => Schema::Nothing,
        3 => Schema::NoneType,
        4 => Schema::Bool,
        5 => Schema::Int,
        6 => Schema::Float,
        7 => Schema::Str,
        8 => Schema::Bytes,
        9 => Schema::Literal(small(u)?),
        10 => Schema::Instance(small(u)?),
        11 => {
            let n = 1 + count(u, 3)?;
            let mut members = Vec::with_capacity(n);
            for _ in 0..n {
                members.push(build_schema(u, depth - 1)?);
            }
            Schema::Union(members)
        }
        12 => {
            let n = 1 + count(u, 3)?;
            let mut members = Vec::with_capacity(n);
            for _ in 0..n {
                members.push(build_schema(u, depth - 1)?);
            }
            Schema::Intersection(members)
        }
        13 => Schema::Complement(Box::new(build_schema(u, depth - 1)?)),
        14 => {
            let base = Box::new(build_schema(u, depth - 1)?);
            let n = count(u, 3)?;
            let mut constraints = Vec::with_capacity(n);
            for _ in 0..n {
                constraints.push(build_constraint(u)?);
            }
            Schema::Refine { base, constraints }
        }
        15 => Schema::Set(Box::new(build_schema(u, depth - 1)?)),
        16 => Schema::FrozenSet(Box::new(build_schema(u, depth - 1)?)),
        17 => Schema::Seq {
            container: if u.arbitrary()? {
                SeqKind::List
            } else {
                SeqKind::Tuple
            },
            regex: build_regex(u, depth - 1)?,
        },
        _ => {
            let nf = count(u, 3)?;
            let mut fields = Vec::with_capacity(nf);
            for _ in 0..nf {
                fields.push(Field {
                    name: NAMES[usize::from(u.arbitrary::<u8>()?) % NAMES.len()].into(),
                    schema: build_schema(u, depth - 1)?,
                    required: u.arbitrary()?,
                });
            }
            let nd = count(u, 3)?;
            let mut defaults = Vec::with_capacity(nd);
            for _ in 0..nd {
                defaults.push((build_schema(u, depth - 1)?, build_schema(u, depth - 1)?));
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
    // Top and bottom bound every schema.
    assert!(
        a.is_subtype_of(&Schema::Anything),
        "{a:?} not below the top"
    );
    assert!(Schema::Nothing.is_subtype_of(a), "bottom not below {a:?}");
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
