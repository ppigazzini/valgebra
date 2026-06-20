//! Fuzz the simplifier: it must terminate without panicking, reach a fixpoint
//! after one application, and preserve the denotation.
#![no_main]

use libfuzzer_sys::fuzz_target;
use valgebra_core_fuzz::{SchemaPlan, check_simplify};

fuzz_target!(|plan: SchemaPlan| {
    check_simplify(&plan.0);
});
