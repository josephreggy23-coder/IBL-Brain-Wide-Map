from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core_tbi.evaluation.leakage_checks import assert_no_group_leakage
from core_tbi.evaluation.splitting import grouped_splits


def evaluate_conventional_models(frame: pd.DataFrame, features: list[str], label: str = "injury_label", n_splits: int = 5, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Out-of-fold predictions using grouped animal partitions only."""
    data = frame.copy()
    data[label] = (data.condition.astype(str).str.lower() != "sham").astype(int) if label not in data else data[label]
    models = {
        "pca_logistic": Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler()), ("pca", PCA(n_components=min(4, len(features)))), ("model", LogisticRegression(max_iter=1000, random_state=seed))]),
        "random_forest": Pipeline([("impute", SimpleImputer()), ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=seed, class_weight="balanced"))]),
    }
    predictions, assignments = [], []
    for fold, (train, test) in enumerate(grouped_splits(data, n_splits=n_splits)):
        assert_no_group_leakage(data, train, test)
        for animal in data.iloc[test].animal_id.unique():
            assignments.append({"animal_id": animal, "partition": f"fold_{fold}"})
        for name, model in models.items():
            model.fit(data.iloc[train][features], data.iloc[train][label])
            probability = model.predict_proba(data.iloc[test][features])[:, 1]
            predictions.append(pd.DataFrame({"row_id": data.index[test], "animal_id": data.iloc[test].animal_id.values, "model": name, "injury_label": data.iloc[test][label].values, "injury_probability": probability, "fold": fold}))
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(assignments).drop_duplicates()
