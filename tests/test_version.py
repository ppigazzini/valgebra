from importlib.metadata import version

import valgebra


def test_version_is_a_nonempty_string() -> None:
    assert isinstance(valgebra.__version__, str)
    assert valgebra.__version__


def test_version_matches_distribution_metadata() -> None:
    assert valgebra.__version__ == version("valgebra")
