"""Shared test fixtures for the IBL-Brain-Wide-Map test suite.

Centralises deterministic RNG seeds, synthetic population tensors, and
temporary paths so that individual test modules can focus on assertions
rather than data construction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from decision_geometry.data import PopulationDataset


@pytest.fixture
def rng() -> np.random.Generator:
    """Return a deterministic NumPy random generator seeded at 42.

    Useful whenever a test needs reproducible random numbers without
    hard-coding its own seed.  Sharing a single canonical seed across
    the suite makes it easy to recreate failures.
    """
    return np.random.default_rng(42)


@pytest.fixture
def decodable_population() -> dict:
    """Synthetic firing-rate tensor with an embedded class signal.

    Layout
    ------
    * 80 trials (40 per class), 10 units, 8 time bins.
    * Labels are ``[0]*40 + [1]*40``.
    * For class-1 trials the first 4 units have a +2.0 boost in time
      bins 4-7 (the second half), making the two classes separable in
      that region.

    Returns a dict with keys ``rates``, ``labels``, ``n_trials``,
    ``n_units``, and ``n_bins`` so callers can destructure only what
    they need.

    This is the shared counterpart of the per-file ``_decodable_fixture``
    helpers that several test modules used to define locally.
    """
    seed = 42
    rng = np.random.default_rng(seed)
    n_trials, n_units, n_bins = 80, 10, 8
    labels = np.repeat([0, 1], n_trials // 2)
    rates = rng.normal(size=(n_trials, n_units, n_bins))
    # Inject a detectable signal: boost the first 4 units in the
    # second half of time bins for class 1 only.
    rates[labels == 1, :4, n_bins // 2 :] += 2.0
    return {
        "rates": rates,
        "labels": labels,
        "n_trials": n_trials,
        "n_units": n_units,
        "n_bins": n_bins,
    }


@pytest.fixture
def minimal_population_dataset() -> PopulationDataset:
    """A small ``PopulationDataset`` suitable for cache / serialization tests.

    Contains 4 trials, 3 units, and 2 time bins -- the minimum needed
    to exercise ``save`` / ``from_cache`` round-trips without the
    overhead of a realistic tensor size.  All trial-level metadata
    arrays are consistent with the ``PopulationDataset`` constructor
    validation so this fixture is guaranteed not to raise.
    """
    return PopulationDataset(
        rates=np.ones((4, 3, 2), dtype=np.float32),
        time=np.array([-0.1, 0.1]),
        choice=np.array([0, 1, 0, 1]),
        stimulus_side=np.array([0, 1, 1, 0]),
        prior_side=np.array([-1, 0, 1, 0]),
        contrast=np.array([0.0, 25.0, 100.0, 12.5]),
        rewarded=np.array([True, True, False, True]),
        reaction_time=np.array([0.2, 0.3, 0.4, 0.25]),
        trial_ids=np.arange(4),
        unit_ids=np.arange(3),
        unit_regions=np.array(["A", "A", "B"]),
    )


@pytest.fixture
def tmp_cache_path(tmp_path: Path) -> Path:
    """A temporary ``.npz`` file path for population-cache tests.

    The file does *not* exist on disk yet -- the fixture only provides
    the path.  Built on top of pytest's ``tmp_path`` so the directory
    is automatically cleaned up after the session.
    """
    return tmp_path / "test_population.npz"
