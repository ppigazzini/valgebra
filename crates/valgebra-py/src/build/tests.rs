use super::*;

#[test]
fn intern_deduplicates_by_identity() {
    Python::attach(|py| {
        let mut pool = Pool::default();
        let a = PyString::new(py, "x").into_any();

        // The same object interns to one slot.
        let first = pool.intern(&a);
        let again = pool.intern(&a);
        assert_eq!(first, again);
        assert_eq!(pool.items().len(), 1);

        // A distinct object takes a new slot.
        let b = PyList::empty(py).into_any();
        let second = pool.intern(&b);
        assert_ne!(first, second);
        assert_eq!(pool.items().len(), 2);

        // Dedup is by identity, not value: a fresh equal-but-distinct object
        // gets its own slot rather than collapsing onto the first.
        let c = PyList::empty(py).into_any();
        let third = pool.intern(&c);
        assert_ne!(second, third);
        assert_eq!(pool.items().len(), 3);
    });
}

/// A definition block is placed once: a block already present at some offset
/// is reused at that offset, shifted references included, and a new one is
/// appended at the end.
#[test]
fn a_definition_block_is_placed_once() {
    let body = |index: usize| {
        Schema::union([
            Schema::Int,
            Schema::list(SeqShape::homogeneous(Schema::Ref(DefIx::new(index)))),
        ])
    };
    let mut defs = Vec::new();

    // Into an empty list: offset zero.
    assert_eq!(place_definitions(&[Schema::Str], &[], &mut defs), 0);
    assert_eq!(defs, vec![Schema::Str]);

    // A block whose body names itself is shifted to the offset it lands at.
    assert_eq!(place_definitions(&[body(0)], &[], &mut defs), 1);
    assert_eq!(defs, vec![Schema::Str, body(1)]);

    // The same block again is found where it already is, and nothing grows.
    assert_eq!(place_definitions(&[body(0)], &[], &mut defs), 1);
    assert_eq!(defs.len(), 2);

    // A block that matches nowhere is appended after the last one.
    assert_eq!(place_definitions(&[Schema::Bytes], &[], &mut defs), 2);
    assert_eq!(defs, vec![Schema::Str, body(1), Schema::Bytes]);
}
