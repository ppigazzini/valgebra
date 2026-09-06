"""`__version__` comes from the crate, and must still be the distribution's.

Reading it with `importlib.metadata.version()` answered the same question and
cost 20 ms of the 32 ms `import valgebra` took: the reader pulls `email`,
`zipfile`, `inspect` and the compression modules to read a file that says what
`Cargo.toml` already said. The extension carries the crate's version instead,
which is the manifest `maturin` builds the wheel's metadata from -- so the two
cannot disagree, and this holds them to each other rather than trusting that.
"""

from __future__ import annotations

import subprocess
import sys
from importlib.metadata import version

import pytest

import valgebra


def test_the_version_is_the_distribution_version() -> None:
    assert valgebra.__version__ == version("valgebra")


def test_the_version_looks_like_one() -> None:
    parts = valgebra.__version__.split(".")
    assert len(parts) >= 2, valgebra.__version__
    assert all(part[:1].isdigit() for part in parts[:2]), valgebra.__version__


def test_importing_valgebra_pulls_in_almost_nothing() -> None:
    """The reason the version moved, stated as a bound.

    A validation library is imported by things that are themselves imported at
    startup, so what it drags in is part of its cost. `importlib.metadata`
    brought a dozen modules for a version string.
    """
    listing = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; before = set(sys.modules); import valgebra; "
                "print(len(set(sys.modules) - before))"
            ),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    pulled = int(listing.stdout.strip())
    assert pulled <= 10, (
        f"importing valgebra pulled in {pulled} modules; it was four when this "
        "bound was written, and a jump means something heavy joined the import"
    )


@pytest.mark.repository
def test_the_crate_and_the_package_agree_on_the_version() -> None:
    """The two files a release has to move together.

    `pyproject.toml` names `Cargo.toml` as the source, so a mismatch here is a
    release that half-happened.
    """
    manifest = (
        subprocess.run(
            ["cargo", "metadata", "--no-deps", "--format-version", "1"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        ).stdout,
    )
    assert f'"version":"{valgebra.__version__}"' in manifest[0].replace(" ", "")
