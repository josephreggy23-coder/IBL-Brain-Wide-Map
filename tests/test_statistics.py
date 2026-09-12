"""Tests for statistical inference on neural decoding results."""

from __future__ import annotations

import numpy as np
import pytest

from decision_geometry.statistics import (
    ChanceLevelCI,
    ClusterResult,
    PermutationResult,
    chance_level_ci,
    cluster_permutation_test,
    permutation_test,
)


# ---- Fixtures ----------------------------------------------------------

def _decodable_fixture(seed: int = 3):
    """Population with a decodable signal in the second half of time bins.

    Mirrors the fixture in test_analysis.py: 80 trials (40 per class),
    10 units, 8 time bins.  Units 0-3 shift their firing rate by +2.0
    for label=1 in bins 4-7, giving a strong decodable signal in that
    window and noise-level decoding in bins 0-3.
    """
    rng = np.random.default_rng(seed)
    labels = np.repeat([0, 1], 40)
    rates = rng.normal(size=(80, 10, 8))
    rates[labels == 1, :4, 4:] += 2.0
    return rates, labels


def _noise_fixture(seed: int = 42):
    """Population with no decodable signal (pure noise)."""
    rng = np.random.default_rng(seed)
    labels = np.repeat([0, 1], 40)
    rates = rng.normal(size=(80, 10, 8))
    return rates, labels


# ---- Permutation test ---------------------------------------------------

class TestPermutationTest:

    def test_decodable_signal_is_significant(self):
        """Signal bins (4-7) should have low p-values."""
        rates, labels = _decodable_fixture()
        result = permutation_test(rates, labels, n_permutations=99, seed=7)

        assert isinstance(result, PermutationResult)
        assert result.observed.shape == (8,)
        assert result.p_values.shape == (8,)
        assert result.null_distribution.shape == (99, 8)

        # Signal bins should be significant at alpha=0.05.
        assert np.all(result.p_values[4:] < 0.05)

    def test_noise_is_not_significant(self):
        """With pure noise, no bin should be consistently significant."""
        rates, labels = _noise_fixture()
        result = permutation_test(rates, labels, n_permutations=99, seed=7)

        # With no signal, p-values should generally be high.
        # We don't require *all* p > 0.05 (false positives happen), but
        # the mean should be well above typical significance thresholds.
        assert result.p_values.mean() > 0.2

    def test_p_values_in_valid_range(self):
        """P-values should be in (0, 1]."""
        rates, labels = _decodable_fixture()
        result = permutation_test(rates, labels, n_permutations=49, seed=0)
        assert np.all(result.p_values > 0)
        assert np.all(result.p_values <= 1)

    def test_observed_matches_direct_decode(self):
        """Observed scores should match what analysis.decode_timecourse gives."""
        from decision_geometry.analysis import decode_timecourse

        rates, labels = _decodable_fixture()
        result = permutation_test(rates, labels, n_permutations=19, seed=7)
        direct = decode_timecourse(rates, labels, seed=7)
        np.testing.assert_array_almost_equal(result.observed, direct)

    def test_rejects_bad_n_permutations(self):
        rates, labels = _decodable_fixture()
        with pytest.raises(ValueError, match="at least 1"):
            permutation_test(rates, labels, n_permutations=0)

    def test_deterministic(self):
        """Same seed should give identical results."""
        rates, labels = _decodable_fixture()
        a = permutation_test(rates, labels, n_permutations=19, seed=42)
        b = permutation_test(rates, labels, n_permutations=19, seed=42)
        np.testing.assert_array_equal(a.p_values, b.p_values)
        np.testing.assert_array_equal(a.null_distribution, b.null_distribution)


# ---- Cluster-based permutation test -------------------------------------

class TestClusterPermutationTest:

    def test_identifies_signal_window(self):
        """Cluster correction should find significant clusters in bins 4-7."""
        rates, labels = _decodable_fixture()
        result = cluster_permutation_test(
            rates, labels, n_permutations=99, seed=7
        )

        assert isinstance(result, ClusterResult)
        assert result.observed.shape == (8,)
        assert result.p_values.shape == (8,)

        # At least one cluster should cover the signal window.
        assert len(result.clusters) >= 1
        # The significant cluster(s) should have low p-values.
        sig_clusters = [
            i for i, p in enumerate(result.cluster_p_values) if p < 0.05
        ]
        assert len(sig_clusters) >= 1

        # Bins 4-7 should have cluster-corrected p < 0.05.
        assert np.all(result.p_values[4:] < 0.05)

    def test_noise_no_significant_clusters(self):
        """Pure noise should not produce significant clusters."""
        rates, labels = _noise_fixture()
        result = cluster_permutation_test(
            rates, labels, n_permutations=99, seed=7
        )

        # No cluster should be significant.
        if len(result.cluster_p_values) > 0:
            assert np.all(result.cluster_p_values > 0.05)

    def test_cluster_mass_is_positive_for_signal(self):
        """Cluster mass in the signal region should be positive and large."""
        rates, labels = _decodable_fixture()
        result = cluster_permutation_test(
            rates, labels, n_permutations=49, seed=7
        )

        if len(result.cluster_stats) > 0:
            # The largest cluster mass should be well above 0.
            assert np.max(result.cluster_stats) > 0.1

    def test_null_cluster_max_shape(self):
        """Null cluster-max distribution should have correct length."""
        rates, labels = _decodable_fixture()
        n_perm = 49
        result = cluster_permutation_test(
            rates, labels, n_permutations=n_perm, seed=7
        )
        assert result.null_cluster_max.shape == (n_perm,)

    def test_rejects_bad_n_permutations(self):
        rates, labels = _decodable_fixture()
        with pytest.raises(ValueError, match="at least 1"):
            cluster_permutation_test(rates, labels, n_permutations=0)


# ---- Chance-level confidence interval ------------------------------------

class TestChanceLevelCI:

    def test_balanced_binary(self):
        """For balanced binary classification, chance is 0.5."""
        ci = chance_level_ci(100, n_classes=2)
        assert isinstance(ci, ChanceLevelCI)
        assert ci.expected == pytest.approx(0.5)
        assert ci.ci_low < 0.5 < ci.ci_high
        assert ci.n_trials == 100
        assert ci.n_classes == 2

    def test_ci_contains_half(self):
        """The CI should contain 0.5 for balanced binary classification."""
        for n in [20, 50, 100, 500]:
            ci = chance_level_ci(n, n_classes=2)
            assert ci.ci_low < 0.5 < ci.ci_high

    def test_ci_narrows_with_more_trials(self):
        """Larger samples should give tighter confidence intervals."""
        ci_small = chance_level_ci(20, n_classes=2)
        ci_large = chance_level_ci(200, n_classes=2)
        width_small = ci_small.ci_high - ci_small.ci_low
        width_large = ci_large.ci_high - ci_large.ci_low
        assert width_large < width_small

    def test_multiclass(self):
        """For 4 classes, chance is 0.25."""
        ci = chance_level_ci(200, n_classes=4)
        assert ci.expected == pytest.approx(0.25)
        assert ci.ci_low < 0.25 < ci.ci_high

    def test_symmetric_around_chance(self):
        """The CI should be symmetric around the expected value."""
        ci = chance_level_ci(100, n_classes=2)
        lower_dist = ci.expected - ci.ci_low
        upper_dist = ci.ci_high - ci.expected
        assert lower_dist == pytest.approx(upper_dist, abs=1e-10)

    def test_rejects_too_few_trials(self):
        with pytest.raises(ValueError, match="n_trials"):
            chance_level_ci(1, n_classes=2)

    def test_rejects_single_class(self):
        with pytest.raises(ValueError, match="n_classes"):
            chance_level_ci(100, n_classes=1)

    def test_rejects_bad_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            chance_level_ci(100, alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            chance_level_ci(100, alpha=1.0)

    def test_wider_at_stricter_alpha(self):
        """A smaller alpha (more confidence) should give a wider CI."""
        ci_90 = chance_level_ci(100, alpha=0.10)
        ci_99 = chance_level_ci(100, alpha=0.01)
        width_90 = ci_90.ci_high - ci_90.ci_low
        width_99 = ci_99.ci_high - ci_99.ci_low
        assert width_99 > width_90
