"""Temporal dynamics of cross-temporal decoding representations.

Complements ``analysis.py`` (which produces a cross-temporal decode
matrix) with summary measures that quantify *how* neural
representations evolve over time:

1. **Generalization index** -- ratio of off-diagonal to on-diagonal
   accuracy in the cross-temporal matrix.  Values near 1 indicate a
   stable code that generalises across time; values near 0 indicate a
   transient code that is useful only at the moment it was trained.

2. **Coding dimensionality** -- effective number of independent
   temporal coding patterns, estimated from the participation ratio
   of the singular values of the centred cross-temporal matrix.

3. **Temporal stability curve** -- mean cross-temporal accuracy as a
   function of temporal lag, summarising how quickly representations
   change.

4. **Peak decode latency** -- time of maximum decoding accuracy with
   a bootstrap confidence interval.

5. **Onset latency** -- first time at which decoding exceeds chance
   for a run of consecutive bins.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ------------------------------------------------------------------
# Result containers
# ------------------------------------------------------------------

@dataclass(frozen=True)
class StabilityCurve:
    """Mean cross-temporal accuracy as a function of temporal lag.

    Attributes
    ----------
    lags : ndarray, shape (n_lags,)
        Lag values from 0 to n_bins - 1.
    accuracy : ndarray, shape (n_lags,)
        Mean accuracy at each lag.
    """

    lags: np.ndarray
    accuracy: np.ndarray


@dataclass(frozen=True)
class PeakLatency:
    """Peak decoding latency with bootstrap confidence interval.

    Attributes
    ----------
    peak_bin : int
        Index of the time bin with maximum accuracy.
    peak_time : float
        Time value at the peak bin (NaN when *time_bins* is not provided).
    ci_low : float
        Lower bound of the bootstrap CI on peak time.
    ci_high : float
        Upper bound of the bootstrap CI on peak time.
    """

    peak_bin: int
    peak_time: float
    ci_low: float
    ci_high: float


# ------------------------------------------------------------------
# 1. Generalization index
# ------------------------------------------------------------------

def generalization_index(ctm: np.ndarray) -> np.ndarray:
    """Ratio of off-diagonal to on-diagonal cross-temporal accuracy.

    For each time bin *t*:

        GI(t) = mean(CTM[t, t'] for t' != t) / CTM[t, t]

    A value near 1 means the representation trained at *t* generalises
    well to other time points (stable code).  A value near 0 means the
    representation is transient and useful only at the training time.

    Parameters
    ----------
    ctm : ndarray, shape (n_bins, n_bins)
        Cross-temporal decoding matrix (train-time x test-time).

    Returns
    -------
    ndarray, shape (n_bins,)
        Generalization index at each time bin.

    Raises
    ------
    ValueError
        If *ctm* is not a square 2-D array with at least 2 bins.
    """
    ctm = np.asarray(ctm, dtype=float)
    if ctm.ndim != 2 or ctm.shape[0] != ctm.shape[1]:
        raise ValueError("ctm must be a square 2-D array")
    n = ctm.shape[0]
    if n < 2:
        raise ValueError("ctm must have at least 2 time bins")

    diag = np.diag(ctm)
    row_sums = ctm.sum(axis=1)
    off_diag_mean = (row_sums - diag) / (n - 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        gi = np.where(
            np.abs(diag) > 1e-12,
            off_diag_mean / diag,
            np.nan,
        )
    return gi


# ------------------------------------------------------------------
# 2. Coding dimensionality (participation ratio)
# ------------------------------------------------------------------

def coding_dimensionality(
    ctm: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Effective temporal coding dimensionality from the cross-temporal matrix.

    Centres the matrix by subtracting *threshold* (typically chance
    level), computes the singular values, and returns the participation
    ratio:

        PR = (sum lambda_i)^2 / sum(lambda_i^2)

    The participation ratio equals 1 when a single component dominates
    and equals the rank when all components contribute equally.

    Parameters
    ----------
    ctm : ndarray, shape (n_bins, n_bins)
        Cross-temporal decoding matrix.
    threshold : float
        Baseline to subtract before SVD (default 0.5, i.e. chance for
        binary decoding).

    Returns
    -------
    float
        Participation ratio of the singular values.

    Raises
    ------
    ValueError
        If *ctm* is not a square 2-D array.
    """
    ctm = np.asarray(ctm, dtype=float)
    if ctm.ndim != 2 or ctm.shape[0] != ctm.shape[1]:
        raise ValueError("ctm must be a square 2-D array")

    centred = ctm - threshold
    sv = np.linalg.svd(centred, compute_uv=False)
    sv_sq = sv ** 2

    total = sv_sq.sum()
    if total < 1e-12:
        return 0.0

    return float((sv_sq.sum()) ** 2 / (sv_sq ** 2).sum())


# ------------------------------------------------------------------
# 3. Temporal stability curve
# ------------------------------------------------------------------

def temporal_stability_curve(ctm: np.ndarray) -> StabilityCurve:
    """Mean cross-temporal accuracy as a function of temporal lag.

    For each lag delta from 0 to n_bins - 1:

        S(delta) = mean(CTM[t, t + delta]) over valid t

    Lag 0 is the diagonal (same-time decoding); larger lags measure how
    well a decoder trained at one time generalises to progressively
    later or earlier times.

    Parameters
    ----------
    ctm : ndarray, shape (n_bins, n_bins)
        Cross-temporal decoding matrix.

    Returns
    -------
    StabilityCurve
        Lag values and corresponding mean accuracies.

    Raises
    ------
    ValueError
        If *ctm* is not a square 2-D array.
    """
    ctm = np.asarray(ctm, dtype=float)
    if ctm.ndim != 2 or ctm.shape[0] != ctm.shape[1]:
        raise ValueError("ctm must be a square 2-D array")

    n = ctm.shape[0]
    lags = np.arange(n)
    accuracy = np.zeros(n, dtype=float)

    for delta in range(n):
        # Collect all entries at this lag (both upper and lower triangle).
        values = []
        for t in range(n):
            if t + delta < n:
                values.append(ctm[t, t + delta])
            if delta > 0 and t + delta < n:
                values.append(ctm[t + delta, t])
        accuracy[delta] = np.mean(values)

    return StabilityCurve(lags=lags, accuracy=accuracy)


# ------------------------------------------------------------------
# 4. Peak decode latency with bootstrap CI
# ------------------------------------------------------------------

def peak_decode_latency(
    scores: np.ndarray,
    time_bins: np.ndarray | None = None,
    *,
    n_bootstrap: int = 1000,
    seed: int = 7,
    ci: float = 0.95,
) -> PeakLatency:
    """Find the time bin with peak decoding accuracy.

    Uses a bootstrap over the score vector to construct a confidence
    interval on the peak time.  Each bootstrap resample adds Gaussian
    noise scaled to the local standard error of the accuracy estimate
    and picks the argmax.

    Parameters
    ----------
    scores : ndarray, shape (n_bins,)
        Decoding accuracy at each time bin.
    time_bins : ndarray, shape (n_bins,), optional
        Physical time value for each bin (e.g. seconds relative to
        stimulus onset).  If *None*, bin indices are used.
    n_bootstrap : int
        Number of bootstrap resamples for the CI.
    seed : int
        Random seed.
    ci : float
        Confidence level (default 0.95).

    Returns
    -------
    PeakLatency

    Raises
    ------
    ValueError
        If *scores* is empty or not 1-D, or if *time_bins* does not
        match *scores* in length.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("scores must be a non-empty 1-D array")

    if time_bins is not None:
        time_bins = np.asarray(time_bins, dtype=float)
        if time_bins.shape != scores.shape:
            raise ValueError("time_bins must match scores in length")
    else:
        time_bins = np.arange(scores.size, dtype=float)

    peak_bin = int(np.argmax(scores))
    peak_time = float(time_bins[peak_bin])

    # Bootstrap CI: add noise proportional to a rough SE estimate and
    # record the argmax of each resample.
    rng = np.random.default_rng(seed)
    score_range = np.ptp(scores)
    noise_scale = max(score_range * 0.05, 1e-6)

    boot_peaks = np.zeros(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        noisy = scores + rng.normal(0, noise_scale, size=scores.size)
        boot_peaks[i] = time_bins[np.argmax(noisy)]

    alpha = 1.0 - ci
    ci_low = float(np.percentile(boot_peaks, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_peaks, 100 * (1 - alpha / 2)))

    return PeakLatency(
        peak_bin=peak_bin,
        peak_time=peak_time,
        ci_low=ci_low,
        ci_high=ci_high,
    )


# ------------------------------------------------------------------
# 5. Onset latency
# ------------------------------------------------------------------

def onset_latency(
    scores: np.ndarray,
    chance: float = 0.5,
    n_consecutive: int = 3,
) -> int | None:
    """Find when decoding first exceeds chance for consecutive bins.

    Scans *scores* and returns the index of the first bin where
    accuracy exceeds *chance* for at least *n_consecutive* bins in a
    row.  Returns ``None`` if no such run exists.

    Parameters
    ----------
    scores : ndarray, shape (n_bins,)
        Decoding accuracy at each time bin.
    chance : float
        Chance-level accuracy (default 0.5 for binary decoding).
    n_consecutive : int
        Number of consecutive above-chance bins required (default 3).

    Returns
    -------
    int or None
        Index of the onset bin, or *None* if decoding never exceeds
        chance for *n_consecutive* bins.

    Raises
    ------
    ValueError
        If *scores* is not 1-D or *n_consecutive* < 1.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1:
        raise ValueError("scores must be a 1-D array")
    if n_consecutive < 1:
        raise ValueError("n_consecutive must be at least 1")

    run_start: int | None = None
    run_length = 0

    for i, s in enumerate(scores):
        if s > chance:
            if run_length == 0:
                run_start = i
            run_length += 1
            if run_length >= n_consecutive:
                return run_start
        else:
            run_length = 0
            run_start = None

    return None
