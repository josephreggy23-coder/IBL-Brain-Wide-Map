from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, average_precision_score, brier_score_loss


def animal_level_metrics(frame: pd.DataFrame, y_true: str = "injury_label", score: str = "injury_probability") -> dict:
    animal = frame.groupby("animal_id", as_index=False).agg({y_true: "max", score: "mean"})
    truth, probabilities = animal[y_true].astype(int), animal[score].clip(0, 1)
    predicted = (probabilities >= 0.5).astype(int)
    result = {"n_animals": len(animal), "balanced_accuracy": balanced_accuracy_score(truth, predicted), "macro_f1": f1_score(truth, predicted, average="macro"), "brier_score": brier_score_loss(truth, probabilities)}
    if truth.nunique() == 2:
        result.update({"auroc": roc_auc_score(truth, probabilities), "average_precision": average_precision_score(truth, probabilities)})
    return result
