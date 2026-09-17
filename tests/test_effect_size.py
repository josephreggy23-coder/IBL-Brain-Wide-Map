"""Tests for effect size measures on time-resolved neural decoding."""

from __future__ import annotations

import numpy as np
import pytest

from decision_geometry.effect_size import (
    EffectSizeResult,
    cohens_h,
    decoding_effect_sizes,
    null_normalized_advantage,
    omega_squared,
)


# ---- Cohen's h -------------------------------------------------------------

class TestCohensH:
    def test_equal_proportions_give_zero(self) -> None:
        assert cohens_h(0.5, 0.5) == pytest.approx(0.0)

    def test_sign_is_positive_when_p1_exceeds_p2(self) -> None:
        h = cohens_h(0.8, 0.5)
        assert h > 0

    def test_sign_is_negative_when_p1_below_p2(self) -> None:
        h = cohens_h(0.3, 0.5)
        assert h < 0

    def test_antisymmetric(self) -> None:
        """h(p1, p2) = -h(p2, p1)."""
        h_forward = cohens_h(0.7, 0.5)
        h_backward = cohens_h(0.5, 0.7)
        assert h_forward == pytest.approx(-h_backward)

    def test_perfect_vs_chance_is_large(self) -> None:
        """h(1.0, 0.5) should be a large effect size."""
        h = cohens_h(1.0, 0.5)
        assert abs(h) > 1.0

    def test_vectorized(self) -> None:
        p1 = np.array([0.5, 0.6, 0.8, 1.0])
        h = cohens_h(p1, 0.5)
        assert h.shape == (4,)
        assert h[0] == pytest.approx(0.0)
        assert np.all(np.diff(h) > 0)

    def test_invalid_proportion_raises(self) -> None:
        with pytest.raises(ValueError, match="proportions"):
            cohens_h(1.2, 0.5)
        with pytest.raises(ValueError, match="proportions"):
            cohens_h(0.5, -0.1)


# ---- Null-normalized advantage --------------------------------------------

class TestNullNormalizedAdvantage:
    def test_at_chance_gives_near_zero(self) -> None:
        """When observed equals the null mean, advantage should be ~0."""
        rng = np.random.default_rng(0)
        null = 0.5 + rng.normal(0, 0.02, size=(200, 5))
        observed = null.mean(axis=0)
        adv = null_normalized_advantage(observed, null)
        np.testing.assert_allclose(adv, 0.0, atol=0.1)

    def test_above_null_gives_positive(self) -> None:
        rng = np.random.default_rng(1)
        null = rng.normal(0.5, 0.02, size=(200, 5))
        observed = np.full(5, 0.8)
        adv = null_normalized_advantage(observed, null)
        assert np.all(adv > 5.0)  # well above the null

    def test_shape_matches(self) -> None:
        observed = np.array([0.5, 0.6, 0.7])
        null = np.random.default_rng(2).normal(0.5, 0.02, (100, 3))
        adv = null_normalized_advantage(observed, null)
        assert adv.shape == (3,)

    def test_mismatched_shapes_raise(self) -> None:
        with pytest.raises(ValueError, match="columns"):
            null_normalized_advantage(np.array([0.5, 0.6]), np.zeros((10, 3)))


# ---- Omega-squared --------------------------------------------------------

class TestOmegaSquared:
    def test_at_chance_gives_zero(self) -> None:
        w2 = omega_squared(0.5, n_classes=2)
        assert w2 == pytest.approx(0.0)

    def test_perfect_accuracy_gives_one(self) -> None:
        w2 = omega_squared(1.0, n_classes=2)
        assert w2 == pytest.approx(1.0)

    def test_above_chance_gives_positive(self) -> None:
        w2 = omega_squared(0.7, n_classes=2)
        assert w2 > 0

    def test_multiclass_chance(self) -> None:
        """With K=4, chance = 0.25, so accuracy of 0.25 gives 0."""
        w2 = omega_squared(0.25, n_classes=4)
        assert w2 == pytest.approx(0.0)

    def test_clipped_to_unit_interval(self) -> None:
        """Result should never exceed 1 or go below 0."""
        w2 = omega_squared(np.array([0.0, 0.5, 1.0]), n_classes=2)
        assert np.all(w2 >= 0)
        assert np.all(w2 <= 1)

    def test_monotone_increasing(self) -> None:
        accuracies = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        w2 = omega_squared(accuracies)
        assert np.all(np.diff(w2) >= 0)


# ---- Composite: decoding_effect_sizes -------------------------------------

class TestDecodingEffectSizes:
    def test_returns_correct_type(self) -> None:
        observed = np.array([0.5, 0.6, 0.7, 0.8])
        result = decoding_effect_sizes(observed, label="choice")
        assert isinstance(result, EffectSizeResult)
        assert result.label == "choice"

    def test_shapes_are_consistent(self) -> None:
        observed = np.array([0.5, 0.55, 0.6, 0.7, 0.75])
        result = decoding_effect_sizes(observed)
        assert result.cohens_h.shape == (5,)
        assert result.advantage.shape == (5,)
        assert result.omega_squared.shape == (5,)

    def test_advantage_is_nan_without_null(self) -> None:
        """Without a null distribution, advantage should be NaN."""
        observed = np.array([0.6, 0.7])
        result = decoding_effect_sizes(observed)
        assert np.all(np.isnan(result.advantage))

    def test_advantage_is_computed_with_null(self) -> None:
        rng = np.random.default_rng(3)
        observed = np.array([0.8, 0.85])
        null = rng.normal(0.5, 0.02, size=(100, 2))
        result = decoding_effect_sizes(observed, null_distribution=null)
        assert np.all(np.isfinite(result.advantage))
        assert np.all(result.advantage > 5)

    def test_peak_h_identifies_strongest_bin(self) -> None:
        observed = np.array([0.5, 0.5, 0.9, 0.5])
        result = decoding_effect_sizes(observed)
        assert result.peak_bin == 2
        assert result.peak_h > 0.8

    def test_mean_omega_squared_is_in_range(self) -> None:
        observed = np.array([0.6, 0.7, 0.8])
        result = decoding_effect_sizes(observed)
        assert 0 < result.mean_omega_squared < 1

    def test_rejects_non_1d_observed(self) -> None:
        with pytest.raises(ValueError, match="one-dimensional"):
            decoding_effect_sizes(np.zeros((3, 2)))
