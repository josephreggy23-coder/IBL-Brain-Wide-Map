"""Population geometry and cross-validated neural decoding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import PopulationDataset


@dataclass(frozen=True)
class AnalysisResult:
    pca_trajectories: dict[str, np.ndarray]
    explained_variance: np.ndarray
    decoding: dict[str, np.ndarray]
    decoding_ci: dict[str, np.ndarray]
    cross_temporal_choice: np.ndarray
    region_decoding: dict[str, np.ndarray]


@dataclass(frozen=True)
class DecodingEstimate:
    scores: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray


def _splits(labels: np.ndarray, seed: int, n_splits: int = 5):
    counts = np.bincount(labels.astype(int))
    folds = min(n_splits, int(counts.min()))
    if folds < 2:
        raise ValueError("each label needs at least two samples")
    return list(StratifiedKFold(folds, shuffle=True, random_state=seed).split(labels, labels))


def _classifier() -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000),
    )


def _validated_decoding_inputs(
    rates: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return valid observations after checking the decoder input contract."""
    if rates.ndim != 3:
        raise ValueError("rates must have shape trials x units x time")
    if labels.ndim != 1 or labels.shape[0] != rates.shape[0]:
        raise ValueError("labels must be one-dimensional and match the trial count")
    valid = labels >= 0
    if not np.any(valid):
        raise ValueError("at least one non-negative label is required")
    return rates[valid], labels[valid]


def _decode_timecourse_oof(
    rates: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y = _validated_decoding_inputs(rates, labels)
    scores = np.zeros(x.shape[2], dtype=float)
    predictions = np.empty((x.shape[0], x.shape[2]), dtype=y.dtype)
    splits = _splits(y, seed)
    for time_index in range(x.shape[2]):
        fold_scores = []
        for train, test in splits:
            model = _classifier()
            model.fit(x[train, :, time_index], y[train])
            prediction = model.predict(x[test, :, time_index])
            predictions[test, time_index] = prediction
            fold_scores.append(balanced_accuracy_score(y[test], prediction))
        scores[time_index] = np.mean(fold_scores)
    return y, scores, predictions


def _bootstrap_balanced_accuracy(
    labels: np.ndarray,
    predictions: np.ndarray,
    *,
    seed: int,
    n_resamples: int,
) -> tuple[np.ndarray, np.ndarray]:
    if n_resamples < 10:
        raise ValueError("n_resamples must be at least 10")

    classes = np.unique(labels)
    bootstrap_scores = np.zeros((n_resamples, predictions.shape[1]), dtype=float)
    rng = np.random.default_rng(seed)
    for label in classes:
        class_indices = np.flatnonzero(labels == label)
        sampled = rng.choice(
            class_indices,
            size=(n_resamples, class_indices.size),
            replace=True,
        )
        bootstrap_scores += (predictions[sampled] == label).mean(axis=1)
    bootstrap_scores /= classes.size
    low, high = np.quantile(bootstrap_scores, [0.025, 0.975], axis=0)
    return low, high


def decode_timecourse(
    rates: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int = 7,
) -> np.ndarray:
    """Return cross-validated balanced accuracy at every time bin."""
    _, scores, _ = _decode_timecourse_oof(rates, labels, seed=seed)
    return scores


def decode_timecourse_estimate(
    rates: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int = 7,
    n_resamples: int = 1000,
) -> DecodingEstimate:
    """Return time-resolved accuracy with a stratified bootstrap interval."""
    y, scores, predictions = _decode_timecourse_oof(rates, labels, seed=seed)
    low, high = _bootstrap_balanced_accuracy(
        y,
        predictions,
        seed=seed + 10_000,
        n_resamples=n_resamples,
    )
    return DecodingEstimate(scores=scores, ci_low=low, ci_high=high)


def cross_temporal_decode(
    rates: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int = 7,
) -> np.ndarray:
    """Train at each time and test at every other time bin."""
    x, y = _validated_decoding_inputs(rates, labels)
    splits = _splits(y, seed)
    n_bins = x.shape[2]
    scores = np.zeros((n_bins, n_bins), dtype=float)

    for train_time in range(n_bins):
        for train, test in splits:
            model = _classifier()
            model.fit(x[train, :, train_time], y[train])
            for test_time in range(n_bins):
                prediction = model.predict(x[test, :, test_time])
                scores[train_time, test_time] += balanced_accuracy_score(y[test], prediction)
    return scores / len(splits)


def _standardize(rates: np.ndarray) -> np.ndarray:
    mean = rates.mean(axis=(0, 2), keepdims=True)
    std = rates.std(axis=(0, 2), keepdims=True)
    return (rates - mean) / np.where(std < 1e-6, 1.0, std)


def _pca_trajectories(rates: np.ndarray, choice: np.ndarray):
    standardized = _standardize(rates)
    flat = standardized.transpose(0, 2, 1).reshape(-1, standardized.shape[1])
    pca = PCA(n_components=3, random_state=7).fit(flat)
    trajectories = {}
    for value, name in [(0, "clockwise"), (1, "counter-clockwise")]:
        condition_mean = standardized[choice == value].mean(axis=0).T
        trajectories[name] = pca.transform(condition_mean)
    return trajectories, pca.explained_variance_ratio_


def analyze_population(
    dataset: PopulationDataset,
    *,
    seed: int = 7,
    min_region_units: int = 5,
    bootstrap_resamples: int = 1000,
) -> AnalysisResult:
    rates = dataset.rates.astype(float)
    trajectories, explained = _pca_trajectories(dataset.rates, dataset.choice)
    labels = {
        "choice": dataset.choice,
        "stimulus": dataset.stimulus_side,
        "prior": dataset.prior_side,
    }
    estimates = {
        name: decode_timecourse_estimate(
            rates,
            values,
            seed=seed,
            n_resamples=bootstrap_resamples,
        )
        for name, values in labels.items()
    }
    decoding = {name: estimate.scores for name, estimate in estimates.items()}
    decoding_ci = {
        name: np.vstack((estimate.ci_low, estimate.ci_high))
        for name, estimate in estimates.items()
    }
    cross_temporal = cross_temporal_decode(rates, dataset.choice, seed=seed)

    region_decoding = {}
    for region in np.unique(dataset.unit_regions):
        unit_mask = dataset.unit_regions == region
        if unit_mask.sum() >= min_region_units:
            label = f"{region} (n={unit_mask.sum()})"
            region_decoding[label] = decode_timecourse(
                rates[:, unit_mask, :], dataset.choice, seed=seed
            )

    return AnalysisResult(
        pca_trajectories=trajectories,
        explained_variance=explained,
        decoding=decoding,
        decoding_ci=decoding_ci,
        cross_temporal_choice=cross_temporal,
        region_decoding=region_decoding,
    )
