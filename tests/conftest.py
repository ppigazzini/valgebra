"""Shared pytest configuration: hypothesis example budgets.

Three property-test profiles select the example budget by the
``HYPOTHESIS_PROFILE`` environment variable, defaulting to ``dev``:

- ``dev`` (default, local): the fast budget for an edit-test loop.
- ``ci`` (pull requests): a wider budget that still finishes a merge gate.
- ``nightly`` (scheduled): the deep budget that hunts the long tail.

The decision and completeness suites read these so the hole-hunting fuzzers run
shallow locally and deep on a schedule without per-test settings.

The interactive profiles carry a per-example deadline so an algorithmic blowup in
``simplify`` or a decision procedure surfaces as a failing example rather than a
silent multi-second one. The ceiling is generous -- every healthy example
finishes well under it -- so it flags a superlinear regression without flaking on
a loaded runner. The scheduled profile leaves the deadline off, since its deepest
generated schemas can legitimately take longer to compile and validate.
"""

import os
from datetime import timedelta

from hypothesis import HealthCheck, settings

_DEADLINE = timedelta(seconds=2)

settings.register_profile("dev", max_examples=100, deadline=_DEADLINE)
settings.register_profile(
    "ci",
    max_examples=500,
    deadline=_DEADLINE,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "nightly",
    max_examples=5000,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))
