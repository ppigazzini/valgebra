//! The pool a reading's builds draw on between them (`pooled`).

use super::{pooled, spend, under};

/// The builds of a pooled reading share its pool: each arms what it asks
/// for or what is left, the smaller, and the pool is charged what each
/// spent. Outside a pool a build arms what it asks for.
#[test]
fn a_pooled_reading_s_builds_share_its_pool() {
    let ((), left) = pooled(5, || {
        under(3, || {
            assert!(spend() && spend() && spend());
            assert!(!spend(), "the build asked for three");
        });
        under(10, || {
            assert!(spend() && spend());
            assert!(!spend(), "the pool held two after the first build");
        });
    });
    assert_eq!(left, 0);
    under(3, || {
        assert!(
            spend() && spend() && spend(),
            "no pool once the reading ends"
        );
        assert!(!spend());
    });
}

/// A build armed inside another sees the pool before the one around it is
/// charged, so a reading spends at most the pool once per level its builds
/// nest, and both are charged once they end.
#[test]
fn a_nested_build_sees_the_pool_before_its_enclosing_one_is_charged() {
    let ((), left) = pooled(4, || {
        under(4, || {
            assert!(spend());
            under(4, || {
                for _ in 0..4 {
                    assert!(spend(), "the inner build armed the whole pool");
                }
                assert!(!spend());
            });
            assert!(spend() && spend() && spend(), "the outer kept its own");
        });
    });
    assert_eq!(left, 0, "eight units spent, charged against four");
    let ((), left) = pooled(4, || under(4, || assert!(spend())));
    assert_eq!(left, 3);
}
