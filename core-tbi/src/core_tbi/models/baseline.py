from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2

from core_tbi.schemas import RecoveryState


class RecoveryPlaneModel:
    """Transparent within-animal recovery plane with empirical tolerances.

    Each animal's baseline median and covariance define its personal movement
    reference. Tolerances are derived from baseline/sham distance distributions,
    never fixed by hand. This is an operational model, not a causal claim.
    """
    def __init__(self, feature_columns: list[str], performance_column: str = "task_success", baseline_timepoint: str = "baseline", confidence_level: float = 0.95):
        self.feature_columns = feature_columns
        self.performance_column = performance_column
        self.baseline_timepoint = baseline_timepoint
        self.confidence_level = confidence_level

    def fit(self, frame: pd.DataFrame) -> "RecoveryPlaneModel":
        self._check(frame)
        baseline = frame.loc[frame.timepoint == self.baseline_timepoint]
        self.references: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
        baseline_distances = []
        performance_residuals = []
        for animal, values in baseline.groupby("animal_id"):
            matrix = values[self.feature_columns].to_numpy(float)
            center = np.median(matrix, axis=0)
            covariance = np.cov(matrix.T) if len(matrix) > 1 else np.eye(len(self.feature_columns))
            covariance = np.atleast_2d(covariance) + np.eye(len(self.feature_columns)) * 1e-3
            inverse = np.linalg.pinv(covariance)
            perf_center = float(np.median(values[self.performance_column]))
            self.references[animal] = (center, inverse, perf_center)
            distances = self._distance(matrix, center, inverse)
            baseline_distances.extend(distances.tolist())
            performance_residuals.extend(np.abs(values[self.performance_column] - perf_center).tolist())
        sham = frame.loc[(frame.condition.astype(str).str.lower() == "sham") & (frame.timepoint != self.baseline_timepoint)]
        sham_distances = [self._row_distance(row) for _, row in sham.iterrows() if row.animal_id in self.references]
        self.kinematic_tolerance_ = float(np.quantile(baseline_distances + sham_distances, self.confidence_level))
        self.performance_tolerance_ = float(np.quantile(performance_residuals, self.confidence_level))
        self.uncertainty_distance_ = float(chi2.ppf(self.confidence_level, len(self.feature_columns)))
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "references"):
            raise RuntimeError("Fit the model before transform.")
        output = frame.copy()
        records = []
        for _, row in output.iterrows():
            if row.animal_id not in self.references:
                records.append((np.nan, np.nan, np.nan, RecoveryState.UNCERTAIN.value, "no_baseline", np.nan))
                continue
            center, inverse, perf_center = self.references[row.animal_id]
            distance = self._row_distance(row)
            perf_deviation = abs(float(row[self.performance_column]) - perf_center)
            quality = float(row.get("tracking_quality", 1.0))
            state, reason = self._state(distance, perf_deviation, quality)
            restitution = max(0.0, 1 - distance / max(self.kinematic_tolerance_, 1e-9))
            burden = max(0.0, distance / max(self.kinematic_tolerance_, 1e-9) - 1) if perf_deviation <= self.performance_tolerance_ else 0.0
            records.append((distance, perf_deviation, restitution, state.value, reason, burden))
        scored = pd.DataFrame(records, columns=["counterfactual_deviation", "performance_deviation", "restitution_score", "recovery_state", "state_reason", "compensation_burden"], index=output.index)
        return pd.concat([output, scored], axis=1)

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)

    def _state(self, distance: float, performance: float, quality: float) -> tuple[RecoveryState, str]:
        if quality < 0.8 or not np.isfinite(distance):
            return RecoveryState.UNCERTAIN, "insufficient_tracking_or_baseline"
        kin_normal, perf_normal = distance <= self.kinematic_tolerance_, performance <= self.performance_tolerance_
        if kin_normal and perf_normal:
            return RecoveryState.RESTITUTION, "movement_and_performance_within_empirical_tolerance"
        if not kin_normal and perf_normal:
            return RecoveryState.COMPENSATION, "operational_proxy_normal_performance_abnormal_movement"
        if not kin_normal and not perf_normal:
            return RecoveryState.PERSISTENT_DYSFUNCTION, "movement_and_performance_abnormal"
        return RecoveryState.UNCERTAIN, "mixed_evidence"

    def _row_distance(self, row: pd.Series) -> float:
        center, inverse, _ = self.references[row.animal_id]
        return float(self._distance(row[self.feature_columns].to_numpy(float)[None, :], center, inverse)[0])

    @staticmethod
    def _distance(matrix: np.ndarray, center: np.ndarray, inverse: np.ndarray) -> np.ndarray:
        delta = matrix - center
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", delta, inverse, delta), 0))

    def _check(self, frame: pd.DataFrame) -> None:
        missing = set(self.feature_columns + ["animal_id", "timepoint", "condition", self.performance_column]) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        if frame.loc[frame.timepoint == self.baseline_timepoint, "animal_id"].nunique() != frame.animal_id.nunique():
            raise ValueError("Every analysed animal needs at least one baseline observation.")
