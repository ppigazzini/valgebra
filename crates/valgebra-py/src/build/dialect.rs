//! Where the two regex engines read one pattern as two languages, and the
//! forms refused for it.
//!
//! The engine here is Rust's, and a pattern Python spells and this one does not
//! raises at build: the parse fails and the caller reads the parse error. That
//! is the loud direction, and it needs no help.
//!
//! The quiet direction is a pattern **both** engines accept and read
//! differently. Inside a character class, `--`, `&&` and `~~` combine classes
//! here and are literal characters to Python, and a nested `[` opens a class
//! here and is a literal `[` there. `[\w--\d]` is the non-digit word characters
//! under this engine and the word characters plus `-` under `re`, and nothing
//! about compiling it says so.
//!
//! `re` gives three different answers to the four forms, and that is the
//! argument for refusing all of them rather than trusting it to object:
//!
//! - `[\w--\d]` **raises**, so a ported pattern fails there at once;
//! - `[\w&&\d]` and `[\w~~\d]` **warn** that `re` may one day read them as
//!   operators, which is a reservation rather than a verdict;
//! - `[a[bc]]` says **nothing at all**, which is the case this is most for.
//!
//! A pattern read with another engine's meaning is a wrong answer a caller
//! cannot see, so each is refused by the name *this* engine would have read it
//! under -- the fact a reader porting the pattern needs, and for two of the
//! three a sentence `re` has already shown them.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Refuse a pattern whose character classes carry an operator `re` reserves.
///
/// The scan is a character walk rather than a parse: it needs only to know
/// whether it stands inside a class, which `\` escapes and `[`/`]` delimit, and
/// a pattern whose classes do not balance is refused by the compile that
/// follows this. A form found outside a class is nothing -- `a--b` is three
/// literals to both engines -- so the position is what decides, as it does in
/// `re`.
pub(crate) fn reject_reserved_class_syntax(pattern: &str) -> PyResult<()> {
    let chars: Vec<char> = pattern.chars().collect();
    let mut depth = 0_usize;
    let mut index = 0;
    // Where the open class's members begin, past the `^` and the `]` that are
    // literal there. A doubled operator at that first position is two literal
    // characters to both engines, so the position is what tells an operator
    // from a member -- `[--/]` is the range from `-` to `/` either way.
    let mut content = 0;
    while index < chars.len() {
        let here = chars[index];
        if here == '\\' {
            index += 2;
            continue;
        }
        if depth == 0 {
            if here == '[' {
                depth = 1;
                // A class opening with `^` and a `]` immediately inside are
                // both literal, and `re` reads them the same way.
                index += 1;
                if chars.get(index) == Some(&'^') {
                    index += 1;
                }
                if chars.get(index) == Some(&']') {
                    index += 1;
                }
                content = index;
                continue;
            }
            index += 1;
            continue;
        }
        if here == ']' {
            depth -= 1;
            index += 1;
            continue;
        }
        if here == '[' {
            // `[:name:]` is a POSIX class, which this engine reads and `re` does
            // not -- a divergence `docs/05-refinements.md` names, with the
            // example that shows how the two read it. Skipping to its close
            // keeps the class depth right; every other `[` opens a nested set,
            // which is a union here and four literals there.
            if chars.get(index + 1) == Some(&':') {
                index = posix_class_end(&chars, index);
                continue;
            }
            return Err(reserved("nested set", pattern, index));
        }
        // A doubled operator is an operator only past the first position of the
        // class, which is where `re` warns and where this engine reads one.
        if matches!(here, '-' | '&' | '~') && chars.get(index + 1) == Some(&here) && index > content
        {
            let what = match here {
                '-' => "set difference",
                '&' => "set intersection",
                _ => "set symmetric difference",
            };
            return Err(reserved(what, pattern, index));
        }
        index += 1;
    }
    Ok(())
}

/// Just past the `:]` closing the POSIX class opening at `start`, or just past
/// the `[` when nothing closes it -- an unterminated class is refused by the
/// compile that follows, and this scan only has to stop reading it.
fn posix_class_end(chars: &[char], start: usize) -> usize {
    let mut index = start + 2;
    while index + 1 < chars.len() {
        if chars[index] == ':' && chars[index + 1] == ']' {
            return index + 2;
        }
        index += 1;
    }
    start + 1
}

/// The refusal, in the words `re` warns with, plus what to write instead.
fn reserved(what: &str, pattern: &str, index: usize) -> PyErr {
    PyValueError::new_err(format!(
        "possible {what} at position {index} in {pattern:?}: this is a \
         class-set operator here and a literal character to `re`, so the two \
         read the pattern as different sets. Escape the characters to mean \
         them literally, or write the classes out."
    ))
}

#[cfg(test)]
mod tests;
