use super::{FieldCursor, Labels, Named, Slot, labels_met};
use crate::ir::Schema;
use core::cell::Cell;
use proptest::prelude::*;

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

/// Each name the maps label, met across every map's slot for it, a slot a map:
/// the definition [`labels_met`] builds without the slots it repeats.
fn every_maps_slot<'s>(maps: &[&Labels<'s>]) -> Vec<(&'s str, Schema, bool)> {
    let mut spelled: Vec<&'s str> = maps
        .iter()
        .flat_map(|labels| labels.named.iter().map(|(name, _)| *name))
        .collect();
    spelled.sort_unstable();
    spelled.dedup();
    spelled
        .into_iter()
        .map(|name| {
            let slots: Vec<Slot> = maps.iter().map(|labels| labels.slot(name)).collect();
            let absent = slots.iter().all(|slot| slot.absent);
            (
                name,
                Schema::meet(slots.into_iter().map(|slot| slot.ty)),
                absent,
            )
        })
        .collect()
}

/// The same, read off what [`labels_met`] builds.
fn met<'s>(maps: &[&Labels<'s>]) -> Vec<(&'s str, Schema, bool)> {
    labels_met(maps)
        .into_iter()
        .map(|(name, slot)| (name, slot.ty, slot.absent))
        .collect()
}

const NAMES: [&str; 4] = ["a", "b", "c", "d"];

/// A map as drawn: each name it labels, by index, with the type it gives it
/// and whether it requires it, and its default.
type DrawnMap = (Vec<(usize, usize, bool)>, Option<usize>);

/// The maps drawn: each name a map labels, with the type it gives it and
/// whether it requires it, and the map's default, by index into the types.
fn drawn_maps() -> impl Strategy<Value = Vec<DrawnMap>> {
    let map = (
        prop::collection::btree_map(0..NAMES.len(), (0..5usize, any::<bool>()), 0..4),
        prop::option::of(0..5usize),
    )
        .prop_map(|(named, rest)| {
            let named = named
                .into_iter()
                .map(|(name, (ty, required))| (name, ty, required))
                .collect();
            (named, rest)
        });
    prop::collection::vec(map, 1..7)
}

proptest! {
    // THEORY: a-record-splits-across-a-union
    /// A name's meet built from the maps that label it and the defaults of the
    /// maps left, each default once, is the meet of every map's slot, and so is
    /// whether it may be absent: a meet is the set of its members.
    #[test]
    fn a_names_meet_is_the_meet_of_every_maps_slot(maps in drawn_maps()) {
        let types = [Schema::Int, Schema::Str, Schema::Bool, Schema::Float, Schema::ANYTHING];
        let labels: Vec<Labels<'_>> = maps
            .iter()
            .map(|(named, rest)| Labels {
                named: named
                    .iter()
                    .filter_map(|(name, ty, required)| {
                        Some((
                            *NAMES.get(*name)?,
                            Slot { ty: types.get(*ty)?.clone(), absent: !*required },
                        ))
                    })
                    .collect(),
                rest: rest.and_then(|rest| types.get(rest)),
            })
            .collect();
        let maps: Vec<&Labels<'_>> = labels.iter().collect();
        prop_assert_eq!(met(&maps), every_maps_slot(&maps));
    }
}

// THEORY: a-record-splits-across-a-union
/// A thousand maps of one field each, one of them with no default: each name
/// meets its own type with the one default the others give and, but for the
/// name that map labels itself, with nothing, as the slot of every map gives
/// it.
#[test]
fn a_wide_meet_of_maps_reads_each_name_from_the_maps_that_label_it() {
    let names: Vec<String> = (0..1000).map(|at| format!("k{at}")).collect();
    let labels: Vec<Labels<'_>> = names
        .iter()
        .enumerate()
        .map(|(at, name)| Labels {
            named: vec![(
                name.as_str(),
                Slot {
                    ty: Schema::Int,
                    absent: at % 2 == 0,
                },
            )],
            rest: (at != 7).then_some(&Schema::Str),
        })
        .collect();
    let maps: Vec<&Labels<'_>> = labels.iter().collect();
    let met = met(&maps);
    assert_eq!(met.len(), 1000);
    assert_eq!(met, every_maps_slot(&maps));
    let held: Vec<&str> = met
        .iter()
        .filter(|(_, ty, _)| *ty != Schema::Nothing)
        .map(|(name, _, _)| *name)
        .collect();
    assert_eq!(held, ["k7"]);
}
