use super::{Ctx, MAX_WALK_DEPTH, WalkMode, WalkState};
use pyo3::exceptions::PyKeyboardInterrupt;
use rustc_hash::FxHashMap;

/// Build a context over an empty validator. Only the depth counter is read
/// here, and it is the one piece of the context that needs no interpreter:
/// a `Cell<usize>` and the guard that returns a level to it.
fn with_ctx(state: &WalkState, run: impl FnOnce(Ctx<'_>)) {
    let records = FxHashMap::default();
    let attrs = FxHashMap::default();
    let unions = FxHashMap::default();
    let regexes = FxHashMap::default();
    run(Ctx {
        pool: &[],
        defs: &[],
        records: &records,
        attrs: &attrs,
        unions: &unions,
        regexes: &regexes,
        guard: &state.guard,
        depth: &state.depth,
        fatal: &state.fatal,
        fatal_seen: &state.fatal_seen,
        mode: WalkMode::Fast,
    });
}

/// The counter is a *depth*, not a total: a level is returned when its guard
/// drops, so a wide value takes and returns one level per child and only
/// nesting accumulates.
///
/// This is the whole reason the bound can be a fixed number. A counter that
/// only ever rose would refuse a flat list of 512 integers, which is not a
/// value that risks the stack.
#[test]
fn a_level_is_returned_when_its_descent_ends() {
    let state = WalkState::new();
    with_ctx(&state, |ctx| {
        assert_eq!(state.depth.get(), 0);
        {
            let _outer = ctx.descend().expect("the first level is open");
            assert_eq!(state.depth.get(), 1);
            {
                let _inner = ctx.descend().expect("a second level nests");
                assert_eq!(state.depth.get(), 2);
            }
            assert_eq!(state.depth.get(), 1, "the inner level was returned");
        }
        assert_eq!(state.depth.get(), 0, "the outer level was returned");

        // Width, which is the case the bound must not refuse: a thousand
        // siblings, each taking the level the last one gave back.
        for _ in 0..1_000 {
            let _sibling = ctx.descend().expect("a sibling reuses the level");
            assert_eq!(state.depth.get(), 1);
        }
        assert_eq!(state.depth.get(), 0);
    });
}

/// The ceiling admits exactly `MAX_WALK_DEPTH` open levels and refuses the
/// next, and refusing costs nothing: the walk that gets `None` has taken no
/// level, so the counter is where it was and the levels above it still close.
#[test]
fn the_ceiling_admits_its_own_number_of_levels_and_no_more() {
    let state = WalkState::new();
    with_ctx(&state, |ctx| {
        let open: Vec<_> = (0..MAX_WALK_DEPTH)
            .map(|level| {
                ctx.descend()
                    .unwrap_or_else(|| panic!("level {level} is inside the bound"))
            })
            .collect();
        assert_eq!(state.depth.get(), MAX_WALK_DEPTH);
        assert!(
            ctx.descend().is_none(),
            "the level past the bound must be refused"
        );
        assert_eq!(
            state.depth.get(),
            MAX_WALK_DEPTH,
            "a refused descent takes no level"
        );
        drop(open);
        assert_eq!(state.depth.get(), 0);
        assert!(ctx.descend().is_some(), "the bound is not a one-way latch");
    });
}

/// The recorded signal reaches the entry point that re-raises it. Without
/// that hand-off a fatal interpreter signal -- a `KeyboardInterrupt` raised
/// inside a predicate -- is swallowed and the value reads as a non-member.
#[test]
fn the_recorded_signal_leaves_with_the_state() {
    assert!(
        WalkState::new().into_fatal().is_none(),
        "a walk that saw no signal carries none out"
    );
    let state = WalkState::new();
    *state.fatal.borrow_mut() = Some(PyKeyboardInterrupt::new_err("stop"));
    state.fatal_seen.set(true);
    assert!(
        state.into_fatal().is_some(),
        "the signal must reach the entry point that re-raises it"
    );
}
