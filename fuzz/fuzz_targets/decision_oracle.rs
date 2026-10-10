//! Fuzz the decision procedures under an oracle that answers: the order laws
//! of the `decision` target, asked of a drawn pool of constants, bounds and
//! classes, and a refutation never meeting a proof of the same inclusion
//! reached without it.
#![no_main]

use libfuzzer_sys::fuzz_target;
use valgebra_core_fuzz::{SchemaPair, WorldPair, check_relations_in};

fuzz_target!(|input: WorldPair| {
    let WorldPair(SchemaPair(a, b, defs), world) = &input;
    check_relations_in(a, b, defs, world);
});
