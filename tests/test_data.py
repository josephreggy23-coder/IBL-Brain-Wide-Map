from pathlib import Path

import numpy as np

from decision_geometry.data import PopulationDataset, select_unit_indices


def test_quality_filter_selects_only_stable_units():
    selected = select_unit_indices(
        quality=np.array([1.0, 0.67, 1.0, 1.0]),
        presence_ratio=np.array([0.95, 0.99, 0.5, 0.95]),
        firing_rate=np.array([4.0, 5.0, 8.0, 150.0]),
        max_electrode=np.array([0, 1, 2, 3]),
        n_electrodes=4,
        max_units=None,
    )
    np.testing.assert_array_equal(selected, [0])


def test_population_cache_round_trip(tmp_path: Path):
    dataset = PopulationDataset(
        rates=np.ones((4, 3, 2), dtype=np.float32),
        time=np.array([-0.1, 0.1]),
        choice=np.array([0, 1, 0, 1]),
        stimulus_side=np.array([0, 1, 1, 0]),
        prior_side=np.array([-1, 0, 1, 0]),
        contrast=np.array([0, 25, 100, 12.5]),
        rewarded=np.array([True, True, False, True]),
        reaction_time=np.array([0.2, 0.3, 0.4, 0.25]),
        trial_ids=np.arange(4),
        unit_ids=np.arange(3),
        unit_regions=np.array(["A", "A", "B"]),
    )
    path = tmp_path / "population.npz"
    dataset.save(path)
    restored = PopulationDataset.from_cache(path)
    np.testing.assert_array_equal(restored.rates, dataset.rates)
    assert str(restored.subject_id) == dataset.subject_id
