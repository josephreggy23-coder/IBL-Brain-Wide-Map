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
    cross_temporal_choice: np.ndarray
    region_decoding: dict[str, np.ndarray]


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


def decode_timecourse(
    rates: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int = 7,
) -> np.ndarray:
    """Return cross-validated balanced accuracy at every time bin."""
    valid = labels >= 0
    x = rates[valid]
    y = labels[valid]
    scores = np.zeros(x.shape[2], dtype=float)
    splits = _splits(y, seed)
    for time_index in range(x.shape[2]):
        fold_scores = []
        for train, test in splits:
            model = _classifier()
            model.fit(x[train, :, time_index], y[train])
            prediction = model.predict(x[test, :, time_index])
            fold_scores.append(balanced_accuracy_score(y[test], prediction))
        scores[time_index] = np.mean(fold_scores)
    return scores


def cross_temporal_decode(
    rates: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int = 7,
) -> np.ndarray:
    """Train at each time and test at every other time bin."""
    valid = labels >= 0
    x = rates[valid]
    y = labels[valid]
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
) -> AnalysisResult:
    rates = dataset.rates.astype(float)
    trajectories, explained = _pca_trajectories(dataset.rates, dataset.choice)
    decoding = {
        "choice": decode_timecourse(rates, dataset.choice, seed=seed),
        "stimulus": decode_timecourse(rates, dataset.stimulus_side, seed=seed),
        "prior": decode_timecourse(rates, dataset.prior_side, seed=seed),
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
        cross_temporal_choice=cross_temporal,
        region_decoding=region_decoding,
    )
