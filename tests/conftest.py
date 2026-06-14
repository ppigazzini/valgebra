"""Shared pytest configuration: hypothesis example budgets.

Three property-test profiles select the example budget by the
``HYPOTHESIS_PROFILE`` environment variable, defaulting to ``dev``:

- ``dev`` (default, local): the fast budget for an edit-test loop.
- ``ci`` (pull requests): a wider budget that still finishes a merge gate.
- ``nightly`` (scheduled): the deep budget that hunts the long tail.

The decision and completeness suites read these so the hole-hunting fuzzers run
shallow locally and deep on a schedule without per-test settings.
"""

import os

from hypothesis import HealthCheck, settings

settings.register_profile("dev", max_examples=100, deadline=None)
settings.register_profile(
    "ci",
    max_examples=500,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "nightly",
    max_examples=5000,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))
