//! What a panic that reaches an entry point says.
//!
//! `PyO3`'s constructors whose signature cannot fail panic on the null a refused
//! allocation returns, and the entry points read that panic back as the
//! `MemoryError` the interpreter raises for a refusal of its own
//! ([`refused_memory_is_memory_error`](crate::errors::refused_memory_is_memory_error)).
//! Which panic that is is read here, on its payload alone, so a test can hand it
//! each payload a panic carries: the shipped extension reaches it only where
//! Python's allocator refuses an object, which an interpreter without
//! `_testcapi` cannot arrange.

use std::any::Any;

/// Whether a panic's `payload` is `PyO3`'s on a null pointer: the text a refused
/// allocation panics with, carried as the `&'static str` a literal message
/// gives or the `String` a formatted one does. The message is the only mark,
/// and a `PyO3` release could reword it; `tests/test_fatal_signals.py` refuses
/// allocations where the interpreter can and reads `MemoryError` back.
pub(crate) fn a_refused_allocation(payload: &(dyn Any + Send)) -> bool {
    let says = |text: &str| text.contains("pointer is null");
    payload
        .downcast_ref::<&str>()
        .is_some_and(|text| says(text))
        || payload
            .downcast_ref::<String>()
            .is_some_and(|text| says(text))
}

#[cfg(test)]
mod tests {
    use super::a_refused_allocation;

    /// The payload a panic carries, caught.
    fn payload_of(panics: impl FnOnce() + std::panic::UnwindSafe) -> Box<dyn std::any::Any + Send> {
        std::panic::catch_unwind(panics).expect_err("the closure panics")
    }

    /// A null pointer's panic is a refused allocation whichever payload carries
    /// its text, a literal's `&str` or a formatted message's `String`, and a
    /// panic saying anything else is not.
    #[test]
    fn a_null_pointer_panic_is_read_on_either_payload() {
        let literal = payload_of(|| panic!("the pointer is null"));
        assert!(literal.downcast_ref::<&str>().is_some());
        assert!(a_refused_allocation(&*literal));
        let where_ = "PyTuple_New";
        let formatted = payload_of(move || panic!("{where_}: the pointer is null"));
        assert!(formatted.downcast_ref::<String>().is_some());
        assert!(a_refused_allocation(&*formatted));
        let other_text = payload_of(|| panic!("an index past the end"));
        assert!(!a_refused_allocation(&*other_text));
        let code = 7_u8;
        let no_text = payload_of(move || std::panic::panic_any(code));
        assert!(!a_refused_allocation(&*no_text));
    }
}
