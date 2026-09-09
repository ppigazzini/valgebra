//! A fixed, deterministic workload over the decision procedures' *refutations*.
//!
//! The sibling `decision_workload` asks relations that hold, or that the rules
//! decline and the descriptor settles. Every reading it takes is of the path a
//! proof walks, so a change to the path a *refutation* walks costs nothing it
//! can see -- which is how work landed on that path with no instrument watching
//! it.
//!
//! The corpus here is the shapes a rule refutes by a mismatch: two fixed
//! lengths that differ, a prefix shorter than the one it must align with, a
//! sequence that repeats without bound against one that cannot, a record
//! lacking a key its supertype requires, and a record that makes optional a key
//! its supertype requires. One subject holds no value at all, which is the case
//! a mismatch is not a witness for: it is below every set, the shape it cannot
//! match included.
//!
//! Run under cachegrind, the count is deterministic for a given build. Keep the
//! corpus and `ITERATIONS` fixed; changing either moves the budget and requires
//! re-recording it.

use std::sync::Arc;

use valgebra_core::{Constraint, Field, Openness, Schema, SeqShape};

/// Iterations per relation. Fewer than the sibling workload takes, because a
/// refuted relation asks the descriptor and a proved one does not, so each of
/// these costs a hundred times what a proof costs.
const ITERATIONS: usize = 100;

/// A closed record of `width` integer fields, without the first `absent` of
/// them, and with the first `optional` of the rest not required.
fn record(width: usize, absent: usize, optional: usize) -> Schema {
    let fields = (0..width)
        .filter(|i| *i >= absent)
        .map(|i| Field {
            name: format!("f{i}").into(),
            schema: Schema::Int,
            required: i >= optional,
        })
        .collect();
    Schema::record(fields, Openness::Closed)
}

fn main() {
    let ints = |n: usize| Schema::list(SeqShape::fixed((0..n).map(|_| Schema::Int)));
    let two = ints(2);
    let three = ints(3);
    let one = ints(1);
    let repeating = Schema::list(SeqShape::homogeneous(Schema::Int));
    // A sequence whose first position admits no value, so the sequence admits
    // none either: below every set, and no arity it fails to match refutes it.
    let unfillable = Schema::list(SeqShape::fixed([
        Schema::refine(
            Schema::list(SeqShape::homogeneous(Schema::Nothing)),
            vec![Constraint::MinLen(1)],
        ),
        Schema::ANYTHING,
    ]));
    let empty_list = Schema::list(SeqShape::fixed([]));
    let complete = record(8, 0, 0);
    let missing = record(8, 1, 0);
    let loosened = record(8, 0, 1);
    let not_int = Schema::Complement(Arc::new(Schema::Int));

    let mut checksum: usize = 0;
    for _ in 0..ITERATIONS {
        let b = std::hint::black_box;
        // Two fixed lengths that differ, in both directions: one refutes on the
        // prefix being shorter, the other on the lengths not matching.
        checksum += usize::from(b(&one).is_subtype_of(b(&two)));
        checksum += usize::from(b(&three).is_subtype_of(b(&two)));
        // A sequence that repeats without bound against one that cannot.
        checksum += usize::from(b(&repeating).is_subtype_of(b(&two)));
        // A record lacking a key its supertype requires, and one that makes
        // such a key optional.
        checksum += usize::from(b(&missing).is_subtype_of(b(&complete)));
        checksum += usize::from(b(&loosened).is_subtype_of(b(&complete)));
        // A subject with no value, below a shape it can never take.
        checksum += usize::from(b(&unfillable).is_subtype_of(b(&empty_list)));
        // A refuted inclusion asked as an equivalence, which asks it twice.
        checksum += usize::from(b(&one).is_equivalent(b(&two)));
        // And one the complement rule refutes rather than a shape rule.
        checksum += usize::from(b(&repeating).is_subtype_of(b(&not_int)));
    }
    println!("checksum={checksum}");
}
