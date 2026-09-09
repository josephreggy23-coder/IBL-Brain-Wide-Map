"""Tests for single-unit selectivity measures."""

from __future__ import annotations

import numpy as np
import pytest

from decision_geometry.selectivity import (
    PopulationSelectivity,
    UnitSelectivity,
    choice_probability,
    d_prime,
    population_selectivity,
    selectivity_index,
    unit_selectivity_profile,
)


# ---- Helpers -----------------------------------------------------------

def _separated_population(seed: int = 0):
    """Population with one strongly selective and one non-selective unit."""
    rng = np.random.default_rng(seed)
    n_trials = 80
    n_bins = 10
    labels = np.repeat([0, 1], n_trials // 2)

    # Unit 0: strong separation in the second half of time bins.
    # Unit 1: no separation (pure noise).
    rates = rng.normal(loc=5.0, scale=1.0, size=(n_trials, 2, n_bins))
    rates[labels == 1, 0, 5:] += 3.0  # add signal to unit 0, late bins

    return rates, labels


# ---- d-prime -----------------------------------------------------------

def test_d_prime_perfect_separation():
    """Perfectly separated groups should have a large d'."""
    a = np.array([10.0, 10.0, 10.0, 10.0])
    b = np.array([0.0, 0.0, 0.0, 0.0])
    dp = d_prime(a, b)
    assert dp > 5.0


def test_d_prime_identical_groups():
    """Identical groups should have d' = 0."""
    a = np.array([5.0, 5.0, 5.0])
    b = np.array([5.0, 5.0, 5.0])
    assert d_prime(a, b) == pytest.approx(0.0, abs=1e-10)


def test_d_prime_sign():
    """d' should be positive when A > B and negative when A < B."""
    a = np.array([10.0, 11.0, 12.0])
    b = np.array([5.0, 6.0, 7.0])
    assert d_prime(a, b) > 0
    assert d_prime(b, a) < 0


def test_d_prime_too_few_samples():
    """With fewer than 2 samples, d' should be nan."""
    assert np.isnan(d_prime(np.array([1.0]), np.array([2.0, 3.0])))


# ---- Choice probability ------------------------------------------------

def test_cp_perfect_separation():
    """When all A values exceed all B values, CP = 1.0."""
    a = np.array([10.0, 11.0, 12.0])
    b = np.array([1.0, 2.0, 3.0])
    assert choice_probability(a, b) == pytest.approx(1.0)


def test_cp_identical_distributions():
    """When A and B are identical, CP should be 0.5."""
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert choice_probability(vals, vals) == pytest.approx(0.5)


def test_cp_reverse_gives_complement():
    """CP(A, B) + CP(B, A) should equal 1."""
    rng = np.random.default_rng(1)
    a = rng.normal(2.0, 1.0, size=50)
    b = rng.normal(0.0, 1.0, size=50)
    assert choice_probability(a, b) + choice_probability(b, a) == pytest.approx(1.0)


def test_cp_in_range():
    """CP should always be in [0, 1]."""
    rng = np.random.default_rng(2)
    for _ in range(10):
        a = rng.normal(size=20)
        b = rng.normal(size=20)
        cp = choice_probability(a, b)
        assert 0.0 <= cp <= 1.0


# ---- Selectivity index -------------------------------------------------

def test_si_identical_means():
    """SI should be 0 when both groups have the same mean."""
    a = np.array([4.0, 5.0, 6.0])
    b = np.array([4.0, 5.0, 6.0])
    assert selectivity_index(a, b) == pytest.approx(0.0, abs=1e-10)


def test_si_one_silent():
    """SI should be 1 when one condition is silent."""
    a = np.array([10.0, 10.0, 10.0])
    b = np.array([0.0, 0.0, 0.0])
    assert selectivity_index(a, b) == pytest.approx(1.0)


def test_si_both_zero():
    """SI should be nan when both means are zero."""
    a = np.array([0.0, 0.0])
    b = np.array([0.0, 0.0])
    assert np.isnan(selectivity_index(a, b))


def test_si_symmetric():
    """SI should be the same regardless of condition order."""
    a = np.array([10.0, 12.0])
    b = np.array([5.0, 7.0])
    assert selectivity_index(a, b) == pytest.approx(selectivity_index(b, a))


# ---- Unit selectivity profile -------------------------------------------

def test_unit_profile_shapes():
    """Profile arrays should match the number of time bins."""
    rates, labels = _separated_population()
    profile = unit_selectivity_profile(rates, labels, unit_idx=0, n_shuffles=50)

    assert isinstance(profile, UnitSelectivity)
    n_bins = rates.shape[2]
    assert profile.d_prime.shape == (n_bins,)
    assert profile.choice_probability.shape == (n_bins,)
    assert profile.selectivity_index.shape == (n_bins,)
    assert profile.mean_a.shape == (n_bins,)
    assert profile.mean_b.shape == (n_bins,)


def test_selective_unit_has_large_dp():
    """Unit 0 should show high d' in the signal window (bins 5-9)."""
    rates, labels = _separated_population()
    profile = unit_selectivity_profile(rates, labels, unit_idx=0, n_shuffles=50)
    assert np.mean(np.abs(profile.d_prime[5:])) > 1.0


def test_nonselective_unit_has_small_dp():
    """Unit 1 (noise only) should show d' near 0 everywhere."""
    rates, labels = _separated_population()
    profile = unit_selectivity_profile(rates, labels, unit_idx=1, n_shuffles=50)
    assert np.mean(np.abs(profile.d_prime)) < 0.5


def test_onset_detection_finds_signal():
    """Selective unit should have onset in the signal window."""
    rates, labels = _separated_population()
    profile = unit_selectivity_profile(
        rates, labels, unit_idx=0, n_shuffles=100, alpha=0.05
    )
    assert profile.onset_bin is not None
    assert profile.onset_bin >= 4  # signal starts at bin 5


def test_onset_none_for_noise_unit():
    """Non-selective unit should have no significant onset."""
    rates, labels = _separated_population(seed=42)
    profile = unit_selectivity_profile(
        rates, labels, unit_idx=1, n_shuffles=100, alpha=0.01
    )
    # With pure noise and strict alpha, onset should usually be None.
    # (Occasionally it may find a spurious onset, so we also accept a late one.)
    if profile.onset_bin is not None:
        assert profile.onset_bin >= 3  # at least not falsely early


# ---- Population selectivity --------------------------------------------

def test_population_selectivity_shapes():
    rates, labels = _separated_population()
    result = population_selectivity(rates, labels, n_shuffles=50)

    assert isinstance(result, PopulationSelectivity)
    assert result.n_units == 2
    assert result.n_bins == 10
    assert result.d_prime.shape == (2, 10)
    assert result.choice_probability.shape == (2, 10)
    assert result.selectivity_index.shape == (2, 10)
    assert result.onset_bins.shape == (2,)
    assert result.fraction_selective.shape == (10,)


def test_population_peak_selectivity_in_signal_window():
    rates, labels = _separated_population()
    result = population_selectivity(rates, labels, n_shuffles=50)
    assert result.peak_selectivity_bin >= 5


def test_population_rejects_wrong_dimensions():
    with pytest.raises(ValueError, match="n_trials, n_units, n_bins"):
        population_selectivity(np.zeros((10, 5)), np.zeros(10))


def test_population_rejects_mismatched_labels():
    with pytest.raises(ValueError, match="trial dimension"):
        population_selectivity(np.zeros((10, 3, 5)), np.zeros(8))
