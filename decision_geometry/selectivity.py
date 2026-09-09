"""Single-unit selectivity measures for decision-related neural activity.

Population decoding (``analysis.py``) answers whether the full population
encodes a variable; this module asks which individual units carry the
signal. The two complement each other: a region where many units show
weak selectivity might decode well (the population pools noisy signals),
while a region with a few strongly selective units might decode poorly
(the decoder cannot ignore the majority of noisy units).

Measures
--------
1. **d-prime (d')** -- signal detection metric. For each unit and time
   bin, d' = (mean_A - mean_B) / pooled_std. Positive d' means the
   unit fires more for condition A (e.g., rightward choice).

2. **Choice probability (CP)** -- the probability that a randomly drawn
   trial from condition A produces a higher firing rate than one from
   condition B. Equivalent to the area under the ROC curve. CP = 0.5
   means no selectivity; CP > 0.5 means the unit discriminates between
   conditions above chance.

3. **Selectivity index (SI)** -- abs(mean_A - mean_B) / (mean_A + mean_B),
   a normalized measure of how much the firing rate differs between
   conditions relative to the overall level. Ranges from 0 (identical)
   to 1 (one condition silent).

4. **Onset latency** -- the first time bin at which a unit's selectivity
   exceeds a significance threshold, estimated by shuffling trial labels.
   Units with shorter onset latencies carry the earliest decision signals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class UnitSelectivity:
    """Selectivity profile for one unit across time."""

    d_prime: np.ndarray
    choice_probability: np.ndarray
    selectivity_index: np.ndarray
    mean_a: np.ndarray
    mean_b: np.ndarray
    onset_bin: int | None


@dataclass(frozen=True)
class PopulationSelectivity:
    """Selectivity profiles for all units in a population."""

    d_prime: np.ndarray          # (n_units, n_bins)
    choice_probability: np.ndarray  # (n_units, n_bins)
    selectivity_index: np.ndarray   # (n_units, n_bins)
    onset_bins: np.ndarray       # (n_units,) -1 if no significant onset
    fraction_selective: np.ndarray  # (n_bins,) fraction of units significant

    @property
    def n_units(self) -> int:
        return self.d_prime.shape[0]

    @property
    def n_bins(self) -> int:
        return self.d_prime.shape[1]

    @property
    def peak_selectivity_bin(self) -> int:
        """Time bin with the highest mean absolute d-prime across units."""
        return int(np.argmax(np.mean(np.abs(self.d_prime), axis=0)))


def d_prime(
    rates_a: np.ndarray,
    rates_b: np.ndarray,
) -> float:
    """Compute d-prime between two groups of firing rates.

    Parameters
    ----------
    rates_a, rates_b
        1-D arrays of firing rates for the two conditions.

    Returns
    -------
    d' value. Positive means group A has higher mean firing rate.
    """
    n_a, n_b = len(rates_a), len(rates_b)
    if n_a < 2 or n_b < 2:
        return np.nan

    mean_a = np.mean(rates_a)
    mean_b = np.mean(rates_b)
    var_a = np.var(rates_a, ddof=1)
    var_b = np.var(rates_b, ddof=1)

    # Pooled standard deviation.
    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    pooled_std = np.sqrt(pooled_var)

    if pooled_std < 1e-12:
        if abs(mean_a - mean_b) < 1e-12:
            return 0.0
        return float(np.sign(mean_a - mean_b) * np.inf)
    return float((mean_a - mean_b) / pooled_std)


def choice_probability(
    rates_a: np.ndarray,
    rates_b: np.ndarray,
) -> float:
    """Compute choice probability (ROC area) between two conditions.

    Returns the probability that a randomly drawn rate from condition A
    exceeds a randomly drawn rate from condition B. CP = 0.5 means no
    selectivity.
    """
    n_a, n_b = len(rates_a), len(rates_b)
    if n_a < 1 or n_b < 1:
        return np.nan

    # Mann-Whitney U statistic normalized to [0, 1].
    concordant = 0.0
    for a in rates_a:
        concordant += np.sum(a > rates_b) + 0.5 * np.sum(a == rates_b)
    return float(concordant / (n_a * n_b))


def selectivity_index(
    rates_a: np.ndarray,
    rates_b: np.ndarray,
) -> float:
    """Normalized selectivity: |mean_A - mean_B| / (mean_A + mean_B).

    Returns 0 when both means are equal, 1 when one condition is silent.
    Returns nan if both means are zero.
    """
    mean_a = np.mean(rates_a)
    mean_b = np.mean(rates_b)
    denom = abs(mean_a) + abs(mean_b)
    if denom < 1e-12:
        return np.nan
    return float(abs(mean_a - mean_b) / denom)


def unit_selectivity_profile(
    rates: np.ndarray,
    labels: np.ndarray,
    unit_idx: int,
    n_shuffles: int = 200,
    alpha: float = 0.05,
    seed: int = 7,
) -> UnitSelectivity:
    """Compute the full selectivity profile for one unit.

    Parameters
    ----------
    rates : ndarray, shape (n_trials, n_units, n_bins)
        Population firing rate tensor.
    labels : ndarray, shape (n_trials,)
        Binary condition labels (0 or 1).
    unit_idx : int
        Which unit to analyze.
    n_shuffles : int
        Number of label shuffles for onset detection.
    alpha : float
        Significance level for onset detection.
    seed : int
        Random seed for shuffles.

    Returns
    -------
    UnitSelectivity for the specified unit.
    """
    valid = labels >= 0
    r = rates[valid, unit_idx, :]  # (n_valid_trials, n_bins)
    y = labels[valid]

    mask_a = y == 1
    mask_b = y == 0
    n_bins = r.shape[1]

    dp = np.zeros(n_bins)
    cp = np.zeros(n_bins)
    si = np.zeros(n_bins)
    mean_a = np.zeros(n_bins)
    mean_b = np.zeros(n_bins)

    for t in range(n_bins):
        ra = r[mask_a, t]
        rb = r[mask_b, t]
        dp[t] = d_prime(ra, rb)
        cp[t] = choice_probability(ra, rb)
        si[t] = selectivity_index(ra, rb)
        mean_a[t] = np.mean(ra) if len(ra) > 0 else np.nan
        mean_b[t] = np.mean(rb) if len(rb) > 0 else np.nan

    # Onset detection via shuffled label distribution.
    onset = _detect_onset(r, y, dp, n_shuffles=n_shuffles, alpha=alpha, seed=seed)

    return UnitSelectivity(
        d_prime=dp,
        choice_probability=cp,
        selectivity_index=si,
        mean_a=mean_a,
        mean_b=mean_b,
        onset_bin=onset,
    )


def population_selectivity(
    rates: np.ndarray,
    labels: np.ndarray,
    n_shuffles: int = 200,
    alpha: float = 0.05,
    seed: int = 7,
) -> PopulationSelectivity:
    """Compute selectivity profiles for all units in the population.

    Parameters
    ----------
    rates : ndarray, shape (n_trials, n_units, n_bins)
    labels : ndarray, shape (n_trials,)
    n_shuffles, alpha, seed
        Passed to ``unit_selectivity_profile`` for onset detection.

    Returns
    -------
    PopulationSelectivity with per-unit and population-level summaries.
    """
    if rates.ndim != 3:
        raise ValueError("rates must have shape (n_trials, n_units, n_bins)")
    if labels.ndim != 1 or labels.shape[0] != rates.shape[0]:
        raise ValueError("labels must match the trial dimension")

    n_units = rates.shape[1]
    n_bins = rates.shape[2]

    all_dp = np.zeros((n_units, n_bins))
    all_cp = np.zeros((n_units, n_bins))
    all_si = np.zeros((n_units, n_bins))
    onsets = np.full(n_units, -1, dtype=int)

    for u in range(n_units):
        profile = unit_selectivity_profile(
            rates, labels, u,
            n_shuffles=n_shuffles, alpha=alpha, seed=seed + u,
        )
        all_dp[u] = profile.d_prime
        all_cp[u] = profile.choice_probability
        all_si[u] = profile.selectivity_index
        if profile.onset_bin is not None:
            onsets[u] = profile.onset_bin

    # Fraction of units significant at each time bin.
    frac_sel = np.zeros(n_bins)
    for t in range(n_bins):
        frac_sel[t] = np.mean(onsets[onsets >= 0] <= t) if np.any(onsets >= 0) else 0.0

    return PopulationSelectivity(
        d_prime=all_dp,
        choice_probability=all_cp,
        selectivity_index=all_si,
        onset_bins=onsets,
        fraction_selective=frac_sel,
    )


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------
def _detect_onset(
    unit_rates: np.ndarray,
    labels: np.ndarray,
    observed_dp: np.ndarray,
    n_shuffles: int,
    alpha: float,
    seed: int,
) -> int | None:
    """Find the first bin where |d'| exceeds the shuffle null.

    Returns the bin index, or None if no bin is significant.
    """
    rng = np.random.default_rng(seed)
    n_bins = unit_rates.shape[1]

    # Build null distribution of max |d'| across all time bins.
    max_null = np.zeros(n_shuffles)
    for s in range(n_shuffles):
        shuffled = rng.permutation(labels)
        mask_a = shuffled == 1
        mask_b = shuffled == 0
        null_dp = np.zeros(n_bins)
        for t in range(n_bins):
            null_dp[t] = abs(d_prime(unit_rates[mask_a, t], unit_rates[mask_b, t]))
        max_null[s] = np.nanmax(null_dp) if np.any(np.isfinite(null_dp)) else 0.0

    threshold = float(np.quantile(max_null, 1.0 - alpha))

    # First bin exceeding the threshold.
    for t in range(n_bins):
        if abs(observed_dp[t]) > threshold:
            return t
    return None
