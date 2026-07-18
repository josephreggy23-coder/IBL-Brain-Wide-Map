from __future__ import annotations
import pandas as pd
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut


def grouped_splits(frame: pd.DataFrame, n_splits: int = 5, group_by: str = "animal_id"):
    if group_by not in frame:
        raise ValueError(f"Grouping column {group_by!r} is missing.")
    groups = frame[group_by].to_numpy()
    unique = pd.unique(groups)
    if len(unique) < 2:
        raise ValueError("At least two animals are required for grouped validation.")
    splitter = GroupKFold(n_splits=min(n_splits, len(unique)))
    yield from splitter.split(frame, groups=groups)


def leave_one_animal_out(frame: pd.DataFrame, group_by: str = "animal_id"):
    groups = frame[group_by].to_numpy()
    yield from LeaveOneGroupOut().split(frame, groups=groups)
