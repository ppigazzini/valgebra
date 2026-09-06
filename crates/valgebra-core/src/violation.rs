//! The structured validation failure: the [`Violation`] value produced when a
//! value does not belong to a schema's set, and its rendering.

use crate::ir::PathSegment;

/// A validation failure: a value did not belong to a schema's set.
#[derive(Debug, Clone)]
pub struct Violation {
    /// Stable, machine-readable code.
    pub code: &'static str,
    /// Location of the offending value from the validation root; empty at root.
    pub path: Vec<PathSegment>,
    /// Short label of the expected set (e.g. `int`).
    pub expected: String,
    /// Short repr-style summary of the offending value.
    pub value_summary: String,
}

impl Violation {
    /// Render the path as a location string (`name[2].id`); empty at the root.
    #[must_use]
    pub fn location(&self) -> String {
        let mut out = String::new();
        // Infallible: a `String` writer never errors.
        let _ = self.write_location(&mut out);
        out
    }

    /// Write the location into any formatter, so the message need not build one.
    ///
    /// A failing `validate` formats a message per violation, and formatting it
    /// through an owned location string is an allocation per violation that only
    /// ever feeds another allocation. Aggregating a wide record's failures makes
    /// that a per-field cost.
    fn write_location(&self, out: &mut impl core::fmt::Write) -> core::fmt::Result {
        let mut first = true;
        for segment in &self.path {
            match segment {
                PathSegment::Key(key) => {
                    if !first {
                        out.write_char('.')?;
                    }
                    out.write_str(key)?;
                }
                PathSegment::IntKey(key) => write!(out, "[{key}]")?,
                PathSegment::BigIntKey(key) => write!(out, "[{key}]")?,
                PathSegment::Index(index) => write!(out, "[{index}]")?,
            }
            first = false;
        }
        Ok(())
    }
}

impl std::fmt::Display for Violation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if !self.path.is_empty() {
            f.write_str("at ")?;
            self.write_location(f)?;
            f.write_str(": ")?;
        }
        write!(
            f,
            "expected {}, got {} [{}]",
            self.expected, self.value_summary, self.code
        )
    }
}

impl std::error::Error for Violation {}
