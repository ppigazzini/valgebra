//! A fixed, deterministic workload over the core schema operations.
//!
//! This is the body of the instruction-count regression gate. It runs the
//! simplifier, the composition index remap, and the open/closed record
//! transform a fixed number of times over a fixed corpus, then prints a
//! checksum so the optimizer cannot discard the work. Run under cachegrind, its
//! instruction count is deterministic for a given build, so a committed budget
//! catches a regression without depending on a noisy wall clock.
//!
//! The count must be dominated by the work, not process startup, so the
//! iteration count is large. Keep the corpus and `ITERATIONS` fixed; changing
//! either moves the budget and requires re-recording it.

use valgebra_core::{Field, Schema};

/// Iterations per operation. Large enough that startup is a rounding error.
const ITERATIONS: usize = 2_000;

/// A redundant Boolean expression exercising every simplifier rewrite.
fn boolean_corpus(depth: usize) -> Schema {
    let mut node = Schema::Union(vec![
        Schema::Int,
        Schema::Int,
        Schema::Nothing,
        Schema::Complement(Box::new(Schema::Complement(Box::new(Schema::Str)))),
    ]);
    for _ in 0..depth {
        node = Schema::Complement(Box::new(Schema::Intersection(vec![
            node.clone(),
            Schema::Union(vec![Schema::Bool, Schema::Anything, node]),
        ])));
    }
    node
}

/// A wide record whose fields carry pool-indexed leaves.
fn wide_record(width: usize) -> Schema {
    let fields = (0..width)
        .map(|i| Field {
            name: format!("f{i}"),
            schema: Schema::Literal(i),
            required: i % 2 == 0,
        })
        .collect();
    Schema::Record {
        fields,
        open: false,
    }
}

/// A record nested `depth` levels deep.
fn nested_records(depth: usize) -> Schema {
    let mut inner = Schema::Record {
        fields: vec![Field {
            name: "leaf".to_owned(),
            schema: Schema::Int,
            required: true,
        }],
        open: false,
    };
    for _ in 0..depth {
        inner = Schema::Record {
            fields: vec![Field {
                name: "child".to_owned(),
                schema: Schema::Sequence(Box::new(inner)),
                required: true,
            }],
            open: false,
        };
    }
    inner
}

fn main() {
    let boolean = boolean_corpus(8);
    let record = wide_record(64);
    let nested = nested_records(32);

    // Fold a checksum through each result so nothing is optimized away.
    let mut checksum: usize = 0;
    for _ in 0..ITERATIONS {
        checksum = checksum.wrapping_add(std::hint::black_box(&boolean).simplify().depth_marker());
        checksum =
            checksum.wrapping_add(std::hint::black_box(&record).shifted(100, 0).depth_marker());
        checksum = checksum.wrapping_add(
            std::hint::black_box(&nested)
                .with_records_open(true)
                .depth_marker(),
        );
    }
    // Printing forces the checksum to be observed.
    println!("checksum={checksum}");
}

/// A cheap structural fingerprint, just enough to keep results observable.
trait DepthMarker {
    fn depth_marker(&self) -> usize;
}

impl DepthMarker for Schema {
    fn depth_marker(&self) -> usize {
        match self {
            Schema::Union(members)
            | Schema::Intersection(members)
            | Schema::Tuple(members)
            | Schema::FixedSequence(members) => members.len(),
            Schema::Record { fields, .. } | Schema::Object { fields, .. } => fields.len(),
            Schema::Complement(inner)
            | Schema::Sequence(inner)
            | Schema::Set(inner)
            | Schema::FrozenSet(inner)
            | Schema::VariadicTuple(inner) => 1 + inner.depth_marker(),
            _ => 0,
        }
    }
}
