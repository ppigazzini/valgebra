"""A build must not change what it does because the project has a harness.

A Cargo feature that reaches production code makes the shipped artifact differ
from the one the tests exercised, and every gate in this tree passes on the
difference: the tests run one build, the wheel is another, and both are green.

There is one such feature, ``interpreter-tests``. It enables an embedded
interpreter so the binding's own tests can acquire the GIL under ``cargo test``,
every one of its sites is inside a test module, and the shipped wheel is built
without it. Nothing held it that way.

Held in both directions:

* a ``cfg(feature = ...)`` on a production path fails, so the shipped build
  cannot silently diverge from the tested one;
* a ledger entry naming a feature the manifest no longer declares fails, so an
  excuse cannot outlive its subject.

An option that is neither allowed nor banned is a decision nobody has made, and
that is what the ledger refuses.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BINDING = ROOT / "crates" / "valgebra-py"

# Features whose `cfg` sites are allowed OUTSIDE a test module, each with the
# reason. Empty: every site of the one feature this crate declares is test-only.
# An entry here is a shipped build that differs from the tested one, so it
# carries an argument rather than a name alone.
PRODUCTION_FEATURES: dict[str, str] = {}

CFG_FEATURE = re.compile(
    r'#\[cfg\((?:all\()?([^)]*feature\s*=\s*"([^"]+)"[^)]*)\)?\)\]'
)


def _declared_features() -> set[str]:
    text = (BINDING / "Cargo.toml").read_text(encoding="utf-8")
    match = re.search(r"^\[features\](.*?)(?=^\[|\Z)", text, re.DOTALL | re.MULTILINE)
    if match is None:
        return set()
    return set(re.findall(r"^([a-z0-9-]+)\s*=", match.group(1), re.MULTILINE))


def _cfg_sites() -> list[tuple[str, str, str]]:
    """Every `cfg(feature = ...)` site: (path, feature, the whole predicate)."""
    sites = []
    for path in sorted(ROOT.rglob("*.rs")):
        if "target" in path.parts:
            continue
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        for predicate, feature in CFG_FEATURE.findall(path.read_text(encoding="utf-8")):
            sites.append((rel, feature, predicate))
    return sites


def _inside_test_module(path: str, feature: str) -> bool:
    """Whether every site of `feature` in `path` sits under a `#[cfg(test)]`.

    Two shapes count: the site itself says `all(test, feature = ..)`, or it sits
    below a `#[cfg(test)] mod` in the same file. The second is what lets a nested
    interpreter module inside a test module carry the feature alone.
    """
    lines = (ROOT / path).read_text(encoding="utf-8").splitlines()
    test_module_at = [
        i for i, line in enumerate(lines) if line.strip().startswith("#[cfg(test)]")
    ]
    first_test_module = min(test_module_at) if test_module_at else None
    for i, line in enumerate(lines):
        if f'feature = "{feature}"' not in line or not line.strip().startswith("#[cfg"):
            continue
        if "test" in line.split("feature")[0]:
            continue  # all(test, feature = ..)
        if first_test_module is None or i < first_test_module:
            return False
    return True


def test_every_feature_site_is_test_only_or_named() -> None:
    sites = _cfg_sites()
    # The scan is the detector: no sites at all would pass having read nothing.
    assert sites, "no cfg(feature = ...) site found in any Rust source"

    leaked = sorted(
        {
            (path, feature)
            for path, feature, _ in sites
            if feature not in PRODUCTION_FEATURES
            and not _inside_test_module(path, feature)
        }
    )
    assert not leaked, (
        f"cfg(feature = ...) on a production path: {leaked}. A feature that "
        "reaches production makes the shipped build differ from the tested one. "
        "Move the site into a test module, or record it with the reason."
    )


def test_no_ledger_entry_is_stale() -> None:
    gone = sorted(set(PRODUCTION_FEATURES) - _declared_features())
    assert not gone, f"ledger names features the manifest does not declare: {gone}"


def test_every_ledger_entry_carries_a_reason() -> None:
    for feature, why in PRODUCTION_FEATURES.items():
        assert len(why) > 40, f"{feature}: a ledger entry with no reason"


def test_the_manifest_says_the_feature_never_ships() -> None:
    # The claim the whole arrangement rests on, kept where a reader of the
    # manifest meets it.
    text = (BINDING / "Cargo.toml").read_text(encoding="utf-8")
    assert "interpreter-tests" in text
    assert "never reaches the shipped wheel" in text


def test_the_test_module_detector_distinguishes_the_two_shapes() -> None:
    # `check/index.rs` carries the feature alone, nested inside a `#[cfg(test)]`
    # module; the others say `all(test, feature = ..)`. Both must read as
    # test-only, and the detector is what decides.
    assert _inside_test_module(
        "crates/valgebra-py/src/check/index.rs", "interpreter-tests"
    )
    assert _inside_test_module(
        "crates/valgebra-py/src/check/walk.rs", "interpreter-tests"
    )
