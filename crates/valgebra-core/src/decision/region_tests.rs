use super::{Kind, Region, Regions};

/// Every region operation is a set operation, and each is pinned here rather
/// than inside the folds that use it. The bits used to be combined at each
/// call site with a raw `|`, `&`, `!`, or `|=`, where the wrong operator is a
/// one-character defect no test in that fold could distinguish; concentrating
/// them into five methods is only worth it if the five are tested, so they
/// are, over the boundary cases the folds start and end at.
#[test]
fn the_region_operations_are_the_set_operations() {
    let a = Kind::Bool.region().union(Kind::Int.region());
    let b = Kind::Bool.region().union(Kind::Str.region());

    // Union and intersection, distinguished: a wrong operator swaps these.
    assert_eq!(
        a.union(b),
        Kind::Bool
            .region()
            .union(Kind::Int.region())
            .union(Kind::Str.region())
    );
    assert_eq!(a.intersect(b), Kind::Bool.region());
    assert_ne!(a.union(b), a.intersect(b));

    // The bounds are the identities of their operations, and absorb the other.
    assert_eq!(a.union(Region::EMPTY), a);
    assert_eq!(a.intersect(Region::ALL), a);
    assert_eq!(a.union(Region::ALL), Region::ALL);
    assert_eq!(a.intersect(Region::EMPTY), Region::EMPTY);

    // The complement is bounded to the partition, so no result ever carries
    // the unused eighth bit, and it is an involution.
    assert_eq!(Region::EMPTY.complement(), Region::ALL);
    assert_eq!(Region::ALL.complement(), Region::EMPTY);
    assert_eq!(a.complement().complement(), a);
    assert_eq!(a.intersect(a.complement()), Region::EMPTY);
    assert_eq!(a.union(a.complement()), Region::ALL);

    // Emptiness is the fold's own verdict, so it is pinned on both sides.
    assert!(Region::EMPTY.is_empty());
    assert!(!Region::ALL.is_empty());
    assert!(!a.is_empty());
    assert!(a.intersect(Kind::Str.region()).is_empty());

    // Inclusion is the subtyping relation on the scalar fragment, and it is
    // not symmetric: `bool` is below `int`, and `int` is not below `bool`.
    assert!(Kind::Bool.region().subset_of(a));
    assert!(!a.subset_of(Kind::Bool.region()));
    assert!(a.subset_of(a));
    assert!(Region::EMPTY.subset_of(a));
    assert!(a.subset_of(Region::ALL));
    assert!(!Kind::Str.region().subset_of(a));
}

/// The region set an emptiness fold accumulates is a monoid under each lattice
/// operation, and `Unknown` absorbs both. The absorbing element is what lets a
/// fold over members stop: once it appears, no later member can change the
/// result, and the walk that continues spends the decision budget on an answer
/// already fixed.
#[test]
fn the_region_set_is_a_monoid_with_an_absorbing_element() {
    let known = |r| Regions::Known(r);
    let unknown = Regions::Unknown;
    let bool_int = known(Kind::Bool.region().union(Kind::Int.region()));

    // Each operation has its identity.
    assert_eq!(bool_int.union(Regions::UNION_UNIT), bool_int);
    assert_eq!(bool_int.intersect(Regions::MEET_UNIT), bool_int);

    // Unknown absorbs both operations, from either side.
    for combine in [Regions::union, Regions::intersect] {
        assert_eq!(combine(unknown, bool_int), unknown);
        assert_eq!(combine(bool_int, unknown), unknown);
        assert_eq!(combine(unknown, unknown), unknown);
    }

    // Only the absorbing element reports itself as one, so a fold cannot stop
    // on a known region it still has to combine.
    assert!(unknown.is_absorbing());
    assert!(!Regions::UNION_UNIT.is_absorbing());
    assert!(!Regions::MEET_UNIT.is_absorbing());
    assert!(!bool_int.is_absorbing());

    // The operations are the region's own where both sides are known.
    assert_eq!(
        known(Kind::Bool.region()).union(known(Kind::Str.region())),
        known(Kind::Bool.region().union(Kind::Str.region()))
    );
    assert_eq!(
        bool_int.intersect(known(Kind::Bool.region())),
        known(Kind::Bool.region())
    );
}

/// The six scalar regions and the non-scalar remainder partition the
/// universe: they are pairwise disjoint and together cover it. Emptiness
/// soundness rests on the cover — the meet of all six scalar complements must
/// be the non-empty non-scalar region, not the empty set.
#[test]
fn every_kind_lands_in_the_partition_and_the_scalars_land_apart() {
    // The claim the region fold rests on, read off the kinds rather than off
    // a second list beside them. Six scalar kinds, each with a region of its
    // own; five container kinds sharing the one that is left. A kind added
    // without a region fails to compile, and one given a region that
    // collapses or collides fails here.
    let scalars = [
        Kind::NoneType,
        Kind::Bool,
        Kind::Int,
        Kind::Float,
        Kind::Str,
        Kind::Bytes,
    ];
    let containers = [
        Kind::List,
        Kind::Tuple,
        Kind::Set,
        Kind::FrozenSet,
        Kind::Dict,
    ];

    // Non-empty and pairwise disjoint together force six distinct bits, which
    // is what makes the partition a partition. Either half alone is satisfied
    // by a region that collapsed to nothing.
    for (i, one) in scalars.iter().enumerate() {
        assert!(!one.region().is_empty(), "{one:?} has no region");
        for other in &scalars[i + 1..] {
            assert!(
                one.region().intersect(other.region()).is_empty(),
                "{one:?} and {other:?} share a region"
            );
        }
    }

    // The container kinds share one region, and it is none of the scalars'.
    // No schema names it alone -- `list[int]` is a proper part of the lists --
    // which is why one bit carries all five.
    let non_scalar = Kind::List.region();
    for kind in containers {
        assert_eq!(kind.region(), non_scalar, "{kind:?} left the shared region");
    }
    for kind in scalars {
        assert!(kind.region().intersect(non_scalar).is_empty());
    }

    // The scalars do not cover the universe: what is left is the region a
    // complement must keep, which is what makes the meet of all six scalar
    // complements inhabited rather than empty.
    let union = scalars
        .iter()
        .fold(Region::EMPTY, |acc, kind| acc.union(kind.region()));
    assert_ne!(union, Region::ALL);
    assert_eq!(union.complement(), non_scalar);
    let all_complements = scalars.iter().fold(Region::ALL, |acc, kind| {
        acc.intersect(kind.region().complement())
    });
    assert!(!all_complements.is_empty());
    assert_eq!(all_complements, union.complement());
}
