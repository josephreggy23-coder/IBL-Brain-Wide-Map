"""Effect size measures for time-resolved neural decoding.

Significance tests (``statistics.py``) answer whether decoding accuracy
differs from chance, but they do not answer *how much* it differs.
Effect sizes fill that gap: a large sample can make a tiny departure
from chance significant, while a small sample can hide a meaningful one.
Reporting both keeps the interpretation honest.

This module provides three complementary measures:

1. **Cohen's h** for proportions (decoding accuracy vs. chance).
   The arcsine transform stabilizes the variance of a proportion, so
   the difference between two proportions can be compared on a scale
   that does not depend on the base rate.

       h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))

   Conventions (Cohen 1988): |h| = 0.2 small, 0.5 medium, 0.8 large.

2. **Decoding advantage** over chance, expressed in units of the null
   standard deviation (like a z-score but referenced to the empirical
   null from permutation testing rather than a Gaussian assumption).

       delta(t) = (observed(t) - mean_null(t)) / sd_null(t)

3. **Omega-squared (omega^2)** for the proportion of variance in
   neural activity explained by the decoded variable, estimated from
   the decoding accuracy via the relationship between balanced accuracy
   and explained variance in the linear-discriminant case:

       omega^2 ~ (BA - 1/K)^2 / ((1/K)(1 - 1/K))

   where BA is balanced accuracy and K is the number of classes.
   This is an approximation that holds exactly for a linear Gaussian
   discriminant and serves as a lower bound on the true explained
   variance for nonlinear relationships.

References
----------
Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences.
Rosenthal, R. (1994). Parametric measures of effect size. In H. Cooper &
    L. V. Hedges (Eds.), The Handbook of Research Synthesis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EffectSizeResult:
    """Effect sizes at each time bin for a decoded variable.

    Attributes
    ----------
    cohens_h : ndarray, shape (n_bins,)
        Cohen's h comparing observed accuracy against chance.
    advantage : ndarray, shape (n_bins,)
        Decoding accuracy minus chance, expressed in null-SD units.
        Only available when a permutation null is provided (NaN otherwise).
    omega_squared : ndarray, shape (n_bins,)
        Approximate proportion of neural variance explained.
    label : str
        Name of the decoded variable (e.g., "choice", "stimulus").
    """

    cohens_h: np.ndarray
    advantage: np.ndarray
    omega_squared: np.ndarray
    label: str

    @property
    def peak_h(self) -> float:
        """Maximum absolute Cohen's h across time bins."""
        return float(np.nanmax(np.abs(self.cohens_h)))

    @property
    def peak_bin(self) -> int:
        """Time bin with the largest absolute Cohen's h."""
        return int(np.nanargmax(np.abs(self.cohens_h)))

    @property
    def mean_omega_squared(self) -> float:
        """Mean omega-squared across time bins."""
        return float(np.nanmean(self.omega_squared))


def cohens_h(p1: float | np.ndarray, p2: float | np.ndarray) -> np.ndarray:
    """Cohen's h effect size for the difference between two proportions.

    The arcsine transformation stabilizes the variance of a proportion,
    making the effect size comparable across different base rates.

        h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))

    Parameters
    ----------
    p1 : float or ndarray
        Observed proportion(s), in [0, 1].
    p2 : float or ndarray
        Reference proportion(s), in [0, 1].

    Returns
    -------
    ndarray
        Cohen's h. Positive when p1 > p2.

    Raises
    ------
    ValueError
        If any proportion is outside [0, 1].
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)

    if np.any(p1 < 0) or np.any(p1 > 1) or np.any(p2 < 0) or np.any(p2 > 1):
        raise ValueError("proportions must be in [0, 1]")

    return 2.0 * np.arcsin(np.sqrt(p1)) - 2.0 * np.arcsin(np.sqrt(p2))


def null_normalized_advantage(
    observed: np.ndarray,
    null_distribution: np.ndarray,
) -> np.ndarray:
    """Decoding advantage expressed in null-distribution standard deviations.

    For each time bin, computes:

        delta(t) = (observed(t) - mean_null(t)) / sd_null(t)

    This is analogous to a z-score but uses the empirical permutation
    null rather than assuming normality.

    Parameters
    ----------
    observed : ndarray, shape (n_bins,)
        Observed balanced accuracy at each time bin.
    null_distribution : ndarray, shape (n_permutations, n_bins)
        Null-distribution scores from label permutations.

    Returns
    -------
    ndarray, shape (n_bins,)
        Advantage in null-SD units. NaN where the null SD is zero.

    Raises
    ------
    ValueError
        If shapes are incompatible.
    """
    observed = np.asarray(observed, dtype=float)
    null_distribution = np.asarray(null_distribution, dtype=float)

    if observed.ndim != 1:
        raise ValueError("observed must be one-dimensional")
    if null_distribution.ndim != 2:
        raise ValueError("null_distribution must be two-dimensional")
    if null_distribution.shape[1] != observed.shape[0]:
        raise ValueError(
            "null_distribution columns must match the number of time bins"
        )

    null_mean = null_distribution.mean(axis=0)
    null_sd = null_distribution.std(axis=0, ddof=1)

    return np.where(
        null_sd > 1e-12,
        (observed - null_mean) / null_sd,
        np.nan,
    )


def omega_squared(
    accuracy: float | np.ndarray,
    n_classes: int = 2,
) -> np.ndarray:
    """Approximate omega-squared from balanced accuracy.

    In the linear Gaussian discriminant case, balanced accuracy and
    explained variance are monotonically related.  The approximation:

        omega^2 ~ (BA - 1/K)^2 / ((1/K)(1 - 1/K))

    maps chance-level accuracy (1/K) to zero and perfect accuracy (1.0)
    to (K-1)/K.  For K=2, perfect accuracy gives omega^2 = 1.

    Parameters
    ----------
    accuracy : float or ndarray
        Balanced accuracy values, in [0, 1].
    n_classes : int
        Number of label classes (default 2).

    Returns
    -------
    ndarray
        Approximate omega-squared, clipped to [0, 1].
    """
    accuracy = np.asarray(accuracy, dtype=float)
    chance = 1.0 / n_classes
    denom = chance * (1.0 - chance)
    if denom < 1e-12:
        return np.zeros_like(accuracy)
    raw = (accuracy - chance) ** 2 / denom
    return np.clip(raw, 0.0, 1.0)


def decoding_effect_sizes(
    observed: np.ndarray,
    *,
    n_classes: int = 2,
    null_distribution: np.ndarray | None = None,
    label: str = "",
) -> EffectSizeResult:
    """Compute effect sizes for time-resolved decoding accuracy.

    Parameters
    ----------
    observed : ndarray, shape (n_bins,)
        Balanced accuracy at each time bin.
    n_classes : int
        Number of label classes (for chance level).
    null_distribution : ndarray, shape (n_permutations, n_bins), optional
        Permutation null scores. If provided, the null-normalized
        advantage is computed; otherwise it is filled with NaN.
    label : str
        Name of the decoded variable for labeling.

    Returns
    -------
    EffectSizeResult
    """
    observed = np.asarray(observed, dtype=float)
    if observed.ndim != 1:
        raise ValueError("observed must be one-dimensional")

    chance = 1.0 / n_classes

    h = cohens_h(observed, chance)
    w2 = omega_squared(observed, n_classes)

    if null_distribution is not None:
        adv = null_normalized_advantage(observed, null_distribution)
    else:
        adv = np.full_like(observed, np.nan)

    return EffectSizeResult(
        cohens_h=h,
        advantage=adv,
        omega_squared=w2,
        label=label,
    )
