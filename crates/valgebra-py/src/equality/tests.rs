use super::*;

#[test]
fn a_multiset_matches_a_permutation_and_refuses_a_repeat() {
    assert!(same_multiset(&[1, 2, 3], &[3, 1, 2], |a, b| a == b));
    assert!(!same_multiset(&[1, 1, 2], &[1, 2, 2], |a, b| a == b));
    assert!(!same_multiset(&[1, 2], &[1, 2, 3], |a, b| a == b));
    assert!(same_multiset::<u8>(&[], &[], |a, b| a == b));
}

/// The shape hash ignores what the pool holds and follows what equality
/// reads: two spellings of one union hash alike, and two different literals
/// are allowed to collide.
#[test]
fn the_shape_hash_is_blind_to_the_slot_and_to_the_order() {
    let digest = |schema: &Schema| {
        let mut hasher = DefaultHasher::new();
        hash_shape(schema, &mut hasher);
        hasher.finish()
    };
    let literal = |at: usize| Schema::Literal(valgebra_core::ConstIx::new(at));
    // Two slots, one shape: equality tells these apart by value, and a hash
    // that read the slot would deny two equal validators one bucket.
    assert_eq!(digest(&literal(0)), digest(&literal(7)));
    // Order is not part of a union, so the fold over its members is not
    // allowed to see one.
    let left = Schema::Union(vec![Schema::Int, Schema::Str]);
    let right = Schema::Union(vec![Schema::Str, Schema::Int]);
    assert_eq!(digest(&left), digest(&right));
    // And a hash that ignored everything would pass the two lines above
    // having read nothing.
    assert_ne!(digest(&Schema::Int), digest(&Schema::Str));
    assert_ne!(digest(&left), digest(&Schema::Union(vec![Schema::Int])));
    // What a complement holds is part of its shape, and so is which collection
    // kind a node names.
    let not_int = Schema::Complement(Box::new(Schema::Int));
    let not_str = Schema::Complement(Box::new(Schema::Str));
    assert_ne!(digest(&not_int), digest(&not_str));
    assert_ne!(digest(&not_int), digest(&Schema::Int));
    assert_ne!(
        digest(&Schema::set(Schema::Int)),
        digest(&Schema::frozen_set(Schema::Int))
    );
    assert_ne!(
        digest(&Schema::set(Schema::Int)),
        digest(&Schema::set(Schema::Str))
    );
    // A reference names a definition, and two references to different ones
    // are different shapes.
    assert_ne!(
        digest(&Schema::Ref(valgebra_core::DefIx::new(0))),
        digest(&Schema::Ref(valgebra_core::DefIx::new(1)))
    );
    // An attribute record's fields are its shape. No annotation builds one
    // without a class, so it is built here.
    let attributes = |name: &str, schema: Schema| {
        Schema::attr_record(vec![Field {
            name: name.to_owned(),
            schema,
            required: true,
        }])
    };
    assert_ne!(
        digest(&attributes("x", Schema::Int)),
        digest(&attributes("y", Schema::Int))
    );
    assert_ne!(
        digest(&attributes("x", Schema::Int)),
        digest(&attributes("x", Schema::Str))
    );
    assert_eq!(
        digest(&attributes("x", Schema::Int)),
        digest(&attributes("x", Schema::Int))
    );
}
