"""The README's two sentences about what a deployment pins, held to the metadata.

The README states under its title the licenses a user may take valgebra under,
and in its install section that valgebra has no runtime dependency, so a
deployment pins its version alone. Both are read by someone deciding whether to
adopt the package -- a legal review, a lock file -- and both are facts the
package's metadata already states: the SPDX expression in `[project] license`,
and the absence of `[project] dependencies`. A sentence the metadata can
contradict is held to it here, so a dependency added or a license changed takes
the README with it.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from _toml import load
from _tree import ROOT

# The repository checks are not the product suite: this file reads the README
# and the project's metadata, neither of which ships as code in a wheel.
pytestmark = pytest.mark.repository

README = ROOT / "README.md"
PYPROJECT = ROOT / "pyproject.toml"

#: The README's sentence that a deployment pins valgebra's version alone.
NO_RUNTIME_DEPENDENCY = "valgebra has no runtime dependency"


def _project() -> dict[str, Any]:
    return load(PYPROJECT)["project"]


def _license_line() -> str:
    """Return the README's first line of text under its title."""
    lines = README.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("# "), "the README opens with its title"
    return next(line for line in lines[1:] if line.strip())


def test_the_readme_names_every_license_the_metadata_offers() -> None:
    expression = _project()["license"]
    assert isinstance(expression, str)
    licenses = re.split(r"\s+OR\s+", expression)
    line = _license_line()
    # An `OR` expression is a choice the user makes, which the line says.
    assert len(licenses) == 1 or line.endswith("at your option."), line
    for spdx in licenses:
        assert spdx in line, f"{spdx} is offered and not named: {line!r}"
    for path in _project()["license-files"]:
        assert (ROOT / path).is_file(), path


def test_the_readme_says_no_runtime_dependency_only_while_there_is_none() -> None:
    assert NO_RUNTIME_DEPENDENCY in README.read_text(encoding="utf-8")
    assert not _project().get("dependencies"), (
        "pyproject.toml declares a runtime dependency, so the README's install "
        f"section no longer holds: {NO_RUNTIME_DEPENDENCY!r}"
    )
