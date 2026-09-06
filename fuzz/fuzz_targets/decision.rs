//! Fuzz the decision procedures: subtyping, equivalence, and emptiness must run
//! without panicking and obey the order laws (reflexivity, top/bottom bounds,
//! equivalence as mutual inclusion).
#![no_main]

use libfuzzer_sys::fuzz_target;
use valgebra_core_fuzz::{SchemaPair, check_relations};

fuzz_target!(|pair: SchemaPair| {
    let SchemaPair(a, b) = &pair;
    check_relations(a, b);
});
