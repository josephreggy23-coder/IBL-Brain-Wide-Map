"""Tests for cross-temporal dynamics measures."""

from __future__ import annotations

import numpy as np
import pytest

from decision_geometry.dynamics import (
    StabilityCurve,
    PeakLatency,
    coding_dimensionality,
    generalization_index,
    onset_latency,
    peak_decode_latency,
    temporal_stability_curve,
)


# ------------------------------------------------------------------
# Helpers: synthetic cross-temporal matrices with known properties
# ------------------------------------------------------------------

def _uniform_ctm(value: float, n: int = 6) -> np.ndarray:
    """A cross-temporal matrix where every entry is the same value.

    A perfectly stable decoder: GI should be 1 everywhere, and the
    stability curve should be flat.
    """
    return np.full((n, n), value)


def _diagonal_ctm(diag_value: float, off_value: float, n: int = 6) -> np.ndarray:
    """Diagonal-only decoder: high accuracy on-diagonal, low off-diagonal.

    GI should be off_value / diag_value at every bin.
    """
    ctm = np.full((n, n), off_value)
    np.fill_diagonal(ctm, diag_value)
    return ctm


def _rank1_ctm(n: int = 6) -> np.ndarray:
    """Rank-1 matrix: coding dimensionality should be 1."""
    v = np.linspace(0.1, 1.0, n)
    return np.outer(v, v)


def _identity_ctm(n: int = 6) -> np.ndarray:
    """Identity matrix: all singular values equal, dimensionality = n."""
    return np.eye(n)


# ------------------------------------------------------------------
# generalization_index
# ------------------------------------------------------------------

class TestGeneralizationIndex:
    def test_uniform_matrix_gives_gi_one(self):
        ctm = _uniform_ctm(0.8)
        gi = generalization_index(ctm)
        np.testing.assert_allclose(gi, 1.0, atol=1e-12)

    def test_diagonal_matrix_gives_expected_ratio(self):
        ctm = _diagonal_ctm(diag_value=1.0, off_value=0.5, n=5)
        gi = generalization_index(ctm)
        np.testing.assert_allclose(gi, 0.5, atol=1e-12)

    def test_zero_diagonal_gives_nan(self):
        ctm = np.ones((4, 4))
        np.fill_diagonal(ctm, 0.0)
        gi = generalization_index(ctm)
        assert np.all(np.isnan(gi))

    def test_output_shape(self):
        n = 10
        ctm = _uniform_ctm(0.7, n=n)
        gi = generalization_index(ctm)
        assert gi.shape == (n,)

    def test_rejects_non_square(self):
        with pytest.raises(ValueError, match="square"):
            generalization_index(np.ones((3, 4)))

    def test_rejects_1d_input(self):
        with pytest.raises(ValueError, match="square"):
            generalization_index(np.ones(5))

    def test_rejects_single_bin(self):
        with pytest.raises(ValueError, match="at least 2"):
            generalization_index(np.ones((1, 1)))

    def test_asymmetric_matrix(self):
        ctm = np.array([
            [1.0, 0.8],
            [0.4, 1.0],
        ])
        gi = generalization_index(ctm)
        # GI(0) = 0.8 / 1.0 = 0.8
        # GI(1) = 0.4 / 1.0 = 0.4
        np.testing.assert_allclose(gi, [0.8, 0.4], atol=1e-12)


# ------------------------------------------------------------------
# coding_dimensionality
# ------------------------------------------------------------------

class TestCodingDimensionality:
    def test_rank1_gives_dimensionality_one(self):
        ctm = _rank1_ctm(n=8)
        # Shift to avoid centring at zero for this test.
        dim = coding_dimensionality(ctm, threshold=0.0)
        assert abs(dim - 1.0) < 0.01

    def test_identity_gives_full_dimensionality(self):
        n = 6
        ctm = _identity_ctm(n)
        dim = coding_dimensionality(ctm, threshold=0.0)
        assert abs(dim - n) < 0.01

    def test_uniform_matrix_at_threshold_gives_zero(self):
        ctm = _uniform_ctm(0.5, n=5)
        dim = coding_dimensionality(ctm, threshold=0.5)
        assert dim == 0.0

    def test_output_is_positive_float(self):
        rng = np.random.default_rng(99)
        ctm = rng.uniform(0.4, 0.9, size=(6, 6))
        dim = coding_dimensionality(ctm)
        assert isinstance(dim, float)
        assert dim >= 0.0

    def test_rejects_non_square(self):
        with pytest.raises(ValueError, match="square"):
            coding_dimensionality(np.ones((3, 5)))


# ------------------------------------------------------------------
# temporal_stability_curve
# ------------------------------------------------------------------

class TestTemporalStabilityCurve:
    def test_uniform_matrix_gives_flat_curve(self):
        val = 0.75
        ctm = _uniform_ctm(val, n=5)
        result = temporal_stability_curve(ctm)
        assert isinstance(result, StabilityCurve)
        np.testing.assert_allclose(result.accuracy, val, atol=1e-12)

    def test_lag_zero_equals_diagonal_mean(self):
        rng = np.random.default_rng(12)
        ctm = rng.uniform(0.4, 0.9, size=(6, 6))
        result = temporal_stability_curve(ctm)
        np.testing.assert_allclose(
            result.accuracy[0], np.mean(np.diag(ctm)), atol=1e-12
        )

    def test_lags_shape(self):
        n = 8
        ctm = np.ones((n, n))
        result = temporal_stability_curve(ctm)
        assert result.lags.shape == (n,)
        assert result.accuracy.shape == (n,)
        np.testing.assert_array_equal(result.lags, np.arange(n))

    def test_diagonal_ctm_decreases_with_lag(self):
        ctm = _diagonal_ctm(diag_value=1.0, off_value=0.5, n=6)
        result = temporal_stability_curve(ctm)
        # Lag 0 should be 1.0 (diagonal), all others 0.5
        assert result.accuracy[0] == pytest.approx(1.0)
        for delta in range(1, 6):
            assert result.accuracy[delta] == pytest.approx(0.5)

    def test_rejects_non_square(self):
        with pytest.raises(ValueError, match="square"):
            temporal_stability_curve(np.ones((3, 4)))


# ------------------------------------------------------------------
# peak_decode_latency
# ------------------------------------------------------------------

class TestPeakDecodeLatency:
    def test_finds_correct_peak_bin(self):
        scores = np.array([0.5, 0.6, 0.9, 0.7, 0.55])
        result = peak_decode_latency(scores)
        assert isinstance(result, PeakLatency)
        assert result.peak_bin == 2

    def test_peak_time_uses_time_bins(self):
        scores = np.array([0.5, 0.6, 0.9, 0.7, 0.55])
        time_bins = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])
        result = peak_decode_latency(scores, time_bins)
        assert result.peak_time == pytest.approx(0.0)

    def test_peak_time_defaults_to_index(self):
        scores = np.array([0.5, 0.6, 0.9, 0.7])
        result = peak_decode_latency(scores)
        assert result.peak_time == pytest.approx(2.0)

    def test_ci_brackets_peak(self):
        scores = np.array([0.5, 0.55, 0.6, 0.95, 0.6, 0.55, 0.5])
        result = peak_decode_latency(scores, n_bootstrap=2000, seed=42)
        assert result.ci_low <= result.peak_time
        assert result.ci_high >= result.peak_time

    def test_deterministic(self):
        scores = np.array([0.5, 0.7, 0.9, 0.6])
        r1 = peak_decode_latency(scores, seed=1)
        r2 = peak_decode_latency(scores, seed=1)
        assert r1.ci_low == r2.ci_low
        assert r1.ci_high == r2.ci_high

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="non-empty"):
            peak_decode_latency(np.array([]))

    def test_rejects_2d(self):
        with pytest.raises(ValueError, match="1-D"):
            peak_decode_latency(np.ones((3, 3)))

    def test_rejects_mismatched_time_bins(self):
        with pytest.raises(ValueError, match="match"):
            peak_decode_latency(np.ones(5), np.ones(3))


# ------------------------------------------------------------------
# onset_latency
# ------------------------------------------------------------------

class TestOnsetLatency:
    def test_clear_onset(self):
        scores = np.array([0.4, 0.45, 0.6, 0.7, 0.8, 0.9])
        onset = onset_latency(scores, chance=0.5, n_consecutive=3)
        assert onset == 2

    def test_no_onset_returns_none(self):
        scores = np.array([0.4, 0.45, 0.48, 0.49])
        assert onset_latency(scores, chance=0.5) is None

    def test_interrupted_run_resets(self):
        # Two above-chance bins, then a dip, then three above-chance bins.
        scores = np.array([0.6, 0.7, 0.4, 0.6, 0.7, 0.8])
        onset = onset_latency(scores, chance=0.5, n_consecutive=3)
        assert onset == 3

    def test_exact_chance_does_not_count(self):
        scores = np.array([0.5, 0.5, 0.5, 0.5])
        assert onset_latency(scores, chance=0.5, n_consecutive=1) is None

    def test_single_consecutive(self):
        scores = np.array([0.4, 0.6, 0.4])
        onset = onset_latency(scores, chance=0.5, n_consecutive=1)
        assert onset == 1

    def test_onset_at_start(self):
        scores = np.array([0.8, 0.9, 0.85, 0.6])
        onset = onset_latency(scores, chance=0.5, n_consecutive=3)
        assert onset == 0

    def test_rejects_2d(self):
        with pytest.raises(ValueError, match="1-D"):
            onset_latency(np.ones((3, 3)))

    def test_rejects_zero_consecutive(self):
        with pytest.raises(ValueError, match="at least 1"):
            onset_latency(np.ones(5), n_consecutive=0)

    def test_too_short_run_returns_none(self):
        scores = np.array([0.6, 0.7, 0.4, 0.6, 0.7, 0.4])
        assert onset_latency(scores, chance=0.5, n_consecutive=3) is None


# ------------------------------------------------------------------
# Integration: realistic-ish cross-temporal matrix
# ------------------------------------------------------------------

class TestIntegration:
    """End-to-end checks on a synthetic matrix with structured signal."""

    @pytest.fixture
    def block_ctm(self) -> np.ndarray:
        """Cross-temporal matrix with a stable block and a transient block.

        Bins 0-3: at chance everywhere (pre-signal), GI = 1
        (trivially stable -- nothing to decode).
        Bins 4-7: high on-diagonal accuracy with moderate off-diagonal
        within the block.  GI < 1 (representation is present but
        partially transient).
        """
        n = 8
        ctm = np.full((n, n), 0.5)
        # Signal block: bins 4-7.
        ctm[4:, 4:] = 0.7  # off-diagonal within the block
        np.fill_diagonal(ctm, 0.5)  # baseline diagonal
        for i in range(4, 8):
            ctm[i, i] = 0.9  # strong on-diagonal in signal block
        return ctm

    def test_gi_lower_in_signal_block_than_uniform_baseline(
        self, block_ctm: np.ndarray
    ):
        gi = generalization_index(block_ctm)
        # Pre-signal bins are uniform at chance: GI = 1 (trivially
        # perfect generalization).  Signal-block bins have higher
        # on-diagonal than off-diagonal: GI < 1.
        mean_gi_signal = np.mean(gi[4:])
        mean_gi_pre = np.mean(gi[:4])
        assert mean_gi_pre == pytest.approx(1.0, abs=1e-12)
        assert mean_gi_signal < 1.0

    def test_stability_curve_highest_at_lag_zero(self, block_ctm: np.ndarray):
        curve = temporal_stability_curve(block_ctm)
        assert curve.accuracy[0] >= curve.accuracy[1]

    def test_dimensionality_is_reasonable(self, block_ctm: np.ndarray):
        dim = coding_dimensionality(block_ctm)
        assert 1.0 <= dim <= block_ctm.shape[0]

    def test_peak_and_onset_on_diagonal(self, block_ctm: np.ndarray):
        diag = np.diag(block_ctm)
        peak = peak_decode_latency(diag)
        assert 4 <= peak.peak_bin <= 7

        onset = onset_latency(diag, chance=0.5, n_consecutive=2)
        assert onset is not None
        assert onset >= 4
