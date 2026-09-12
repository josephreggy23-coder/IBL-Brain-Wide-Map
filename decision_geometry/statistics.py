"""Statistical inference for time-resolved neural decoding results.

Complements ``analysis.py`` (cross-validated decoding) and
``selectivity.py`` (single-unit measures) with formal hypothesis tests
for population-level significance:

1. **Permutation test** -- shuffle trial labels to build a null
   distribution, then test whether each time bin's observed decoding
   accuracy exceeds the null.

2. **Cluster-based permutation test** (Maris & Oostenveld 2007) --
   controls the family-wise error rate across time bins by comparing
   observed cluster masses (summed test statistics in contiguous
   significant bins) against a null distribution of maximum cluster
   masses.

3. **Chance-level confidence interval** -- quantifies the expected
   accuracy under the null given finite sample sizes and (optionally)
   class imbalance, so that observed accuracy can be compared against
   a concrete interval rather than just 0.5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from .analysis import _classifier, _splits, _validated_decoding_inputs


# ------------------------------------------------------------------
# Result containers
# ------------------------------------------------------------------

@dataclass(frozen=True)
class PermutationResult:
    """Output of a label-permutation test on time-resolved decoding."""

    observed: np.ndarray        # (n_bins,) observed balanced accuracy
    p_values: np.ndarray        # (n_bins,) one-sided p-values
    null_distribution: np.ndarray  # (n_permutations, n_bins)


@dataclass(frozen=True)
class ClusterResult:
    """Output of a cluster-based permutation test."""

    observed: np.ndarray        # (n_bins,) observed balanced accuracy
    p_values: np.ndarray        # (n_bins,) cluster-corrected p-values
    clusters: list[np.ndarray]  # list of index arrays, one per cluster
    cluster_stats: np.ndarray   # cluster mass for each observed cluster
    cluster_p_values: np.ndarray  # p-value for each cluster
    null_cluster_max: np.ndarray  # (n_permutations,) max cluster mass


@dataclass(frozen=True)
class ChanceLevelCI:
    """Confidence interval for chance-level balanced accuracy."""

    expected: float   # point estimate (always 0.5 for balanced accuracy)
    ci_low: float
    ci_high: float
    n_trials: int
    n_classes: int


# ------------------------------------------------------------------
# 1. Label-permutation test
# ------------------------------------------------------------------

def _decode_with_labels(
    rates: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Decode at every time bin, returning mean balanced accuracy across folds."""
    splits = _splits(labels, seed)
    n_bins = rates.shape[2]
    scores = np.zeros(n_bins, dtype=float)
    for t in range(n_bins):
        fold_scores = []
        for train, test in splits:
            model = _classifier()
            model.fit(rates[train, :, t], labels[train])
            pred = model.predict(rates[test, :, t])
            fold_scores.append(balanced_accuracy_score(labels[test], pred))
        scores[t] = np.mean(fold_scores)
    return scores


def permutation_test(
    rates: np.ndarray,
    labels: np.ndarray,
    *,
    n_permutations: int = 200,
    seed: int = 7,
) -> PermutationResult:
    """Label-permutation test for time-resolved decoding significance.

    For each permutation, trial labels are shuffled (breaking the
    association between neural activity and condition) and the full
    cross-validated decoding pipeline is re-run.  The p-value at each
    time bin is computed as::

        p(t) = (1 + #{null >= observed}) / (1 + n_permutations)

    This yields a conservative, valid p-value for every finite
    *n_permutations* (Phipson & Smyth 2010).

    Parameters
    ----------
    rates : ndarray, shape (n_trials, n_units, n_bins)
        Population firing-rate tensor.
    labels : ndarray, shape (n_trials,)
        Binary condition labels (non-negative integers).
    n_permutations : int
        Number of label shuffles.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    PermutationResult
    """
    if n_permutations < 1:
        raise ValueError("n_permutations must be at least 1")

    x, y = _validated_decoding_inputs(rates, labels)

    # Observed decoding accuracy.
    observed = _decode_with_labels(x, y, seed=seed)

    # Null distribution.
    rng = np.random.default_rng(seed)
    null = np.zeros((n_permutations, x.shape[2]), dtype=float)
    for i in range(n_permutations):
        shuffled = rng.permutation(y)
        null[i] = _decode_with_labels(x, shuffled, seed=seed + i + 1)

    # One-sided p-values: fraction of null scores >= observed.
    n_geq = np.sum(null >= observed[np.newaxis, :], axis=0)
    p_values = (1 + n_geq) / (1 + n_permutations)

    return PermutationResult(
        observed=observed,
        p_values=p_values,
        null_distribution=null,
    )


# ------------------------------------------------------------------
# 2. Cluster-based permutation test (Maris & Oostenveld 2007)
# ------------------------------------------------------------------

def _find_clusters(
    significant: np.ndarray,
    stat: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Identify contiguous clusters and compute their mass.

    Parameters
    ----------
    significant : bool array (n_bins,)
        Which bins pass the per-bin threshold.
    stat : float array (n_bins,)
        Test statistic at each bin (e.g., observed - chance).

    Returns
    -------
    clusters : list of int arrays
        Indices belonging to each cluster.
    cluster_stats : float array
        Sum of *stat* within each cluster.
    """
    clusters: list[np.ndarray] = []
    cluster_stats: list[float] = []

    if not np.any(significant):
        return clusters, np.array(cluster_stats)

    # Walk through bins and collect runs of True.
    in_cluster = False
    current: list[int] = []
    for i, sig in enumerate(significant):
        if sig:
            current.append(i)
            in_cluster = True
        else:
            if in_cluster:
                idx = np.array(current)
                clusters.append(idx)
                cluster_stats.append(float(np.sum(stat[idx])))
                current = []
                in_cluster = False
            current = []
    # Final cluster if it extends to the last bin.
    if in_cluster and current:
        idx = np.array(current)
        clusters.append(idx)
        cluster_stats.append(float(np.sum(stat[idx])))

    return clusters, np.array(cluster_stats)


def cluster_permutation_test(
    rates: np.ndarray,
    labels: np.ndarray,
    *,
    n_permutations: int = 200,
    cluster_alpha: float = 0.05,
    seed: int = 7,
) -> ClusterResult:
    """Cluster-based permutation test for multiple-comparison correction.

    Implements the method of Maris & Oostenveld (2007):

    1. Decode at every time bin with observed labels and compute a test
       statistic (accuracy minus chance).
    2. Threshold each bin at *cluster_alpha* using the permutation null.
    3. Identify contiguous clusters of supra-threshold bins.
    4. Sum the test statistic within each cluster (cluster mass).
    5. Build a null distribution of the maximum cluster mass from
       shuffled data and compare observed cluster masses against it.

    This controls the family-wise error rate across all time bins.

    Parameters
    ----------
    rates : ndarray, shape (n_trials, n_units, n_bins)
    labels : ndarray, shape (n_trials,)
    n_permutations : int
    cluster_alpha : float
        Per-bin threshold for cluster formation.
    seed : int

    Returns
    -------
    ClusterResult
    """
    if n_permutations < 1:
        raise ValueError("n_permutations must be at least 1")

    x, y = _validated_decoding_inputs(rates, labels)
    chance = 0.5
    n_bins = x.shape[2]

    # Observed scores and test statistic (excess over chance).
    observed = _decode_with_labels(x, y, seed=seed)
    obs_stat = observed - chance

    # Build null distribution via permutations.
    rng = np.random.default_rng(seed)
    null_scores = np.zeros((n_permutations, n_bins), dtype=float)
    null_max_cluster = np.zeros(n_permutations, dtype=float)

    for i in range(n_permutations):
        shuffled = rng.permutation(y)
        null_scores[i] = _decode_with_labels(x, shuffled, seed=seed + i + 1)

    # Per-bin threshold from the null distribution.
    threshold = np.quantile(null_scores, 1.0 - cluster_alpha, axis=0)

    # Observed clusters.
    obs_significant = observed > threshold
    clusters, cluster_stats = _find_clusters(obs_significant, obs_stat)

    # Null cluster-mass distribution: for each permutation, form
    # clusters and record the maximum cluster mass.
    for i in range(n_permutations):
        null_stat = null_scores[i] - chance
        null_sig = null_scores[i] > threshold
        _, null_c_stats = _find_clusters(null_sig, null_stat)
        if null_c_stats.size > 0:
            null_max_cluster[i] = np.max(null_c_stats)
        else:
            null_max_cluster[i] = 0.0

    # Cluster-level p-values.
    cluster_p = np.ones(len(clusters), dtype=float)
    for ci, c_mass in enumerate(cluster_stats):
        n_geq = np.sum(null_max_cluster >= c_mass)
        cluster_p[ci] = (1 + n_geq) / (1 + n_permutations)

    # Map cluster p-values back to individual time bins.
    bin_p = np.ones(n_bins, dtype=float)
    for ci, idx in enumerate(clusters):
        bin_p[idx] = cluster_p[ci]

    return ClusterResult(
        observed=observed,
        p_values=bin_p,
        clusters=clusters,
        cluster_stats=cluster_stats,
        cluster_p_values=cluster_p,
        null_cluster_max=null_max_cluster,
    )


# ------------------------------------------------------------------
# 3. Chance-level confidence interval
# ------------------------------------------------------------------

def chance_level_ci(
    n_trials: int,
    n_classes: int = 2,
    *,
    alpha: float = 0.05,
) -> ChanceLevelCI:
    """Confidence interval for balanced accuracy under the null hypothesis.

    For *K* balanced classes with *n* trials, the expected balanced
    accuracy is 1/K.  With balanced classes and a 1-vs-rest or binary
    decision, each class's recall is approximately Binomial(n_k, 1/K)
    where n_k = n / K.  By the CLT the balanced accuracy is
    approximately::

        BA ~ N(1/K,  (1/K)(1 - 1/K) / n_k)

    For K = 2 this simplifies to BA ~ N(0.5, 1/(4 * n/2)) = N(0.5, 1/(2n)).

    The interval is conservative for small n and becomes exact as
    n grows.

    Parameters
    ----------
    n_trials : int
        Total number of trials.
    n_classes : int
        Number of label classes (default 2).
    alpha : float
        Significance level (two-sided CI is 1 - alpha).

    Returns
    -------
    ChanceLevelCI
    """
    if n_trials < 2:
        raise ValueError("n_trials must be at least 2")
    if n_classes < 2:
        raise ValueError("n_classes must be at least 2")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    from scipy.stats import norm

    expected = 1.0 / n_classes
    n_per_class = n_trials / n_classes

    # Variance of per-class recall: Binomial variance / n_per_class.
    var_recall = expected * (1 - expected) / n_per_class
    se = np.sqrt(var_recall)

    z = norm.ppf(1 - alpha / 2)
    ci_low = expected - z * se
    ci_high = expected + z * se

    return ChanceLevelCI(
        expected=expected,
        ci_low=ci_low,
        ci_high=ci_high,
        n_trials=n_trials,
        n_classes=n_classes,
    )
