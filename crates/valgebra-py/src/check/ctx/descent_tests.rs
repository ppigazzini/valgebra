use super::{
    Ctx, Entered, FIRST_TRAIL, MAX_RECURSION_DEPTH, MAX_WALK_DEPTH, Trail, WalkMode, WalkState,
};
use crate::check::ValidatorIndex;
use pyo3::exceptions::PyKeyboardInterrupt;

/// Build a context over an empty validator. Only the depth counter is read
/// here, and it is the one piece of the context that needs no interpreter:
/// a `Cell<usize>` and the guard that returns a level to it.
fn with_ctx(state: &WalkState, run: impl FnOnce(Ctx<'_>)) {
    let index = ValidatorIndex::default();
    run(Ctx {
        pool: &[],
        defs: &[],
        index: &index,
        state,
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
        WalkState::new().take_fatal().is_none(),
        "a walk that saw no signal carries none out"
    );
    let state = WalkState::new();
    state.keep_fatal(PyKeyboardInterrupt::new_err("stop"));
    state.signals.fatal_seen.set(true);
    assert!(
        state.take_fatal().is_some(),
        "the signal must reach the entry point that re-raises it"
    );
}

/// Asking for room answers exactly what taking a level would do.
///
/// A leaf loop reads [`Ctx::room_to_descend`] instead of holding a level, so
/// the two have to agree at the boundary or a scalar element is refused where
/// its container's child would be admitted, or admitted where it would not --
/// which is the two walks parting at the depth edge, in the other direction.
#[test]
fn asking_for_room_answers_what_taking_a_level_does() {
    let state = WalkState::new();
    with_ctx(&state, |ctx| {
        assert!(ctx.room_to_descend(), "an empty walk has room");
        let mut open = Vec::new();
        for level in 0..MAX_WALK_DEPTH {
            assert!(
                ctx.room_to_descend(),
                "level {level} is inside the bound, so there is room for it"
            );
            open.push(ctx.descend().expect("a level inside the bound is open"));
        }
        assert!(
            !ctx.room_to_descend(),
            "at the bound there is no room, and descend refuses here too"
        );
        assert!(ctx.descend().is_none(), "the two answers must not part");
        drop(open);
        assert!(ctx.room_to_descend(), "the levels came back");
    });
}

/// A level leaves the pair it entered, so a sibling may enter the same one.
///
/// The trail is what refuses a value reached from inside itself. One object in
/// two *sibling* positions is not that: it is a value a caller writes without
/// thinking about it, and a level that entered its pair and did not leave would
/// report the second sibling as cyclic. The height is the recursion depth for
/// the same reason.
#[test]
fn a_level_leaves_the_pair_it_entered() {
    let mut trail = Trail::default();
    let pair = (0x1234, 0);

    assert!(matches!(trail.enter(pair), Entered::Open));
    assert!(
        matches!(trail.enter(pair), Entered::Cycle),
        "the same pair inside itself is a cycle"
    );
    trail.leave();
    assert!(
        matches!(trail.enter(pair), Entered::Open),
        "a sibling enters the pair the level before it left"
    );
    trail.leave();

    // The height is what the bound is read against, so a trail that does not
    // come back down refuses a value at a depth it never reached.
    for level in 0..MAX_RECURSION_DEPTH {
        assert!(
            matches!(trail.enter((level, 0)), Entered::Open),
            "level {level} is inside the bound"
        );
    }
    assert!(
        matches!(trail.enter((MAX_RECURSION_DEPTH, 0)), Entered::Full),
        "the level past the bound is refused"
    );
    for _ in 0..MAX_RECURSION_DEPTH {
        trail.leave();
    }
    assert!(
        matches!(trail.enter(pair), Entered::Open),
        "and the bound is not a one-way latch"
    );
}

/// The first level a trail opens reserves room for the levels that follow.
///
/// A value nested `FIRST_TRAIL` deep enters every level without the trail
/// growing again, so a membership test against a recursive schema asks the
/// allocator once rather than once per doubling. Nothing an answer depends on
/// sees the reservation -- the recursive binding shape counts it -- so the
/// capacity is the thing held here.
#[test]
fn the_first_level_reserves_the_levels_after_it() {
    let mut trail = Trail::default();
    assert_eq!(
        trail.0.capacity(),
        0,
        "a walk that enters no reference allocates nothing"
    );
    assert!(matches!(trail.enter((0, 0)), Entered::Open));
    let reserved = trail.0.capacity();
    assert!(reserved >= FIRST_TRAIL);
    for level in 1..FIRST_TRAIL {
        assert!(matches!(trail.enter((level, 0)), Entered::Open));
    }
    assert_eq!(
        trail.0.capacity(),
        reserved,
        "the trail did not grow inside the reservation"
    );
}

/// Before 3.14 the walk calls a predicate only with [`CALL_OUT_RESERVE`] of
/// the thread's stack left, and short of it answers the `RecursionError` that
/// ends a chain of re-entries. The short call is made by recursing on a thread
/// of its own until the thread's bounds say the reserve is not left, which
/// reads the same bounds the question does.
#[cfg(all(not(PyPy), not(Py_3_14), not(target_env = "musl")))]
#[test]
fn a_predicate_is_called_only_with_the_reserve_left() {
    use super::CALL_OUT_RESERVE;

    fn asked_with(left: usize) -> bool {
        let filler = [0_u8; 1_024];
        std::hint::black_box(&filler);
        let room = if stacker::remaining_stack().is_some_and(|now| now > left) {
            asked_with(left)
        } else {
            WalkState::new().room_for_python().is_ok()
        };
        std::hint::black_box(&filler);
        room
    }

    let asked = |left: usize| {
        std::thread::Builder::new()
            .stack_size(4 * CALL_OUT_RESERVE)
            .spawn(move || asked_with(left))
            .unwrap()
            .join()
            .unwrap()
    };
    assert!(
        !asked(CALL_OUT_RESERVE / 2),
        "short of the reserve, no call"
    );
    assert!(
        asked(2 * CALL_OUT_RESERVE),
        "with the reserve left, the call"
    );
}

/// Before 3.14 a predicate is short of the reserve from an address between the
/// thread's limit and the limit plus the reserve, both ends held, and the
/// error counts the bytes above the limit in kilobytes; above the reserve,
/// below the limit, or on a thread whose limit is not read, it has room.
#[cfg(all(not(PyPy), not(Py_3_14), not(target_env = "musl")))]
#[test]
fn a_predicate_is_short_only_between_the_limit_and_the_reserve_above_it() {
    use super::{CALL_OUT_RESERVE, NO_LIMIT, overflow_message, short_of_the_reserve};

    let limit = 1_000_000;
    let top = limit + CALL_OUT_RESERVE;
    assert_eq!(short_of_the_reserve(top + 1, limit), None);
    assert_eq!(short_of_the_reserve(top, limit), Some(CALL_OUT_RESERVE));
    assert_eq!(short_of_the_reserve(limit + 7, limit), Some(7));
    assert_eq!(short_of_the_reserve(limit, limit), Some(0));
    assert_eq!(short_of_the_reserve(limit - 1, limit), None);
    assert_eq!(short_of_the_reserve(0, limit), None);
    for here in [0, 5, limit, top, 1 << 47] {
        assert_eq!(short_of_the_reserve(here, NO_LIMIT), None);
    }
    assert_eq!(
        short_of_the_reserve(usize::MAX, NO_LIMIT),
        Some(CALL_OUT_RESERVE),
        "an unread limit's reserve is the top of the address space"
    );
    assert_eq!(
        overflow_message(40 * 1024 + 7),
        "Stack overflow (40 kB of the thread's stack left)"
    );
}
