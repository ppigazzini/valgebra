//! valgebra schema intermediate representation.
//!
//! A schema denotes a set of Python values; validation is membership. This
//! crate is pure Rust: it defines the IR and the denotation of every node.
//! Inspecting a Python object requires `PyO3`, so the validator walk itself
//! lives in the bindings crate; this crate is the stable, language-agnostic
//! core.

/// The schema intermediate representation.
///
/// Each variant documents its denotation: the set of Python values it accepts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Schema {
    /// Denotes every `int` instance: `isinstance(x, int)`.
    ///
    /// In Python `bool` is a subclass of `int`, so `True` and `False` are
    /// integers and members of this set: subtyping is subset inclusion. `int`
    /// does not subclass `float`, so a float is not a member.
    Int,
}

impl Schema {
    /// A short, stable label naming the expected set, shown when membership
    /// fails.
    #[must_use]
    pub fn expected(&self) -> &'static str {
        match self {
            Schema::Int => "int",
        }
    }

    /// The stable, machine-readable code emitted when membership fails.
    #[must_use]
    pub fn error_code(&self) -> &'static str {
        match self {
            Schema::Int => "int_type",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int_reports_its_label_and_code() {
        assert_eq!(Schema::Int.expected(), "int");
        assert_eq!(Schema::Int.error_code(), "int_type");
    }
}
