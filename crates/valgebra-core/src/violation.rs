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
    ///
    /// A string key that reads as a bare name is written as one. Any other --
    /// empty, or holding a `.`, a bracket, whitespace or a control character --
    /// is written as a subscript of its [`quoted`] text, `name['a.b']`: bare,
    /// the key `"a.b"` read as the path `a` then `b`, `"[0]"` as the index 0,
    /// and a key holding a newline broke the one-line message across two.
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
                PathSegment::Key(key) if is_bare(key) => {
                    if !first {
                        out.write_char('.')?;
                    }
                    out.write_str(key)?;
                }
                PathSegment::Key(key) => {
                    out.write_char('[')?;
                    write_quoted(out, key)?;
                    out.write_char(']')?;
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

/// Whether a key reads as itself in a location: it has a character, and none the
/// location's grammar uses or that ends a line.
fn is_bare(key: &str) -> bool {
    !key.is_empty()
        && !key
            .chars()
            .any(|c| matches!(c, '.' | '[' | ']') || c.is_whitespace() || c.is_control())
}

/// `text` as a Python string literal, the spelling a caller writes it in.
///
/// Quoted as `repr` quotes: single quotes, or double where the text holds a
/// single quote and no double one. The backslash, the quote, and every control
/// or whitespace character but the space are escaped, so the literal is one
/// line and reads back as `text`.
#[must_use]
pub fn quoted(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    // Infallible: a `String` writer never errors.
    let _ = write_quoted(&mut out, text);
    out
}

fn write_quoted(out: &mut impl core::fmt::Write, text: &str) -> core::fmt::Result {
    let quote = if text.contains('\'') && !text.contains('"') {
        '"'
    } else {
        '\''
    };
    out.write_char(quote)?;
    for c in text.chars() {
        match c {
            '\\' => out.write_str("\\\\")?,
            '\n' => out.write_str("\\n")?,
            '\r' => out.write_str("\\r")?,
            '\t' => out.write_str("\\t")?,
            c if c == quote => {
                out.write_char('\\')?;
                out.write_char(c)?;
            }
            // Every control and whitespace character lies in the first plane,
            // so the four-digit escape is the widest one needed.
            c if c != ' ' && (c.is_control() || c.is_whitespace()) => match u32::from(c) {
                code @ 0..=0xff => write!(out, "\\x{code:02x}")?,
                code => write!(out, "\\u{code:04x}")?,
            },
            c => out.write_char(c)?,
        }
    }
    out.write_char(quote)
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

/// A violation's size, pinned beside the node sizes in `ir.rs`: an explaining
/// walk builds one per failure and moves it into the report.
#[cfg(target_pointer_width = "64")]
const _: () = assert!(size_of::<Violation>() == 88);
