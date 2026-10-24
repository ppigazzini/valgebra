use super::{FieldCursor, Named};
use core::cell::Cell;

/// An entry that counts how often its name is read: every compare the cursor
/// makes reads one, so the count is the walk's length.
struct Counted<'c> {
    name: String,
    reads: &'c Cell<usize>,
}

impl Named for Counted<'_> {
    fn name(&self) -> &str {
        self.reads.set(self.reads.get() + 1);
        &self.name
    }
}

/// The names read to look up every name of `queries` in `fields`, both in name
/// order, the cursor's own construction left out of the count.
fn reads_for(fields: &[&str], queries: &[&str]) -> usize {
    let reads = Cell::new(0);
    let list = |names: &[&str]| -> Vec<Counted<'_>> {
        names
            .iter()
            .map(|name| Counted {
                name: (*name).to_owned(),
                reads: &reads,
            })
            .collect()
    };
    let (fields, queries) = (list(fields), list(queries));
    let order = Cell::new(None);
    let mut cursor = FieldCursor::over(&fields, &queries, &order);
    let wanted: Vec<String> = queries.iter().map(|query| query.name.clone()).collect();
    reads.set(0);
    for name in &wanted {
        let _ = cursor.named(name);
    }
    reads.get()
}

/// A cursor over two lists in name order reads each name a bounded number of
/// times, misses included.
///
/// The walk moves forward only, so it reads each field it passes once and the
/// field each lookup stops on once: `n + m` reads for `n` fields and `m`
/// lookups. The order it rests on is read once, at the first miss, two names a
/// neighboring pair in each list: `2(n - 1) + 2(m - 1)` more. A cursor that
/// fell back to its scan for every lookup reads a field per field before the
/// one it finds, `n` a lookup, which is what the bound tells apart.
#[test]
fn a_cursor_over_two_ordered_lists_reads_each_name_a_bounded_number_of_times() {
    let names: Vec<String> = (0..64).map(|i| format!("f{i:03}")).collect();
    let fields: Vec<&str> = names.iter().step_by(2).map(String::as_str).collect();
    let queries: Vec<&str> = names.iter().map(String::as_str).collect();
    let reads = reads_for(&fields, &queries);
    let bound = 3 * (fields.len() + queries.len()) - 4;
    assert!(
        reads <= bound,
        "{reads} name reads for {} fields and {} lookups; a forward walk reads at most {bound}",
        fields.len(),
        queries.len()
    );
}
