from __future__ import annotations
import pandas as pd


def assert_no_group_leakage(frame: pd.DataFrame, train_index, test_index, group_by: str = "animal_id") -> None:
    overlap = set(frame.iloc[train_index][group_by]) & set(frame.iloc[test_index][group_by])
    if overlap:
        raise ValueError(f"Animal leakage detected across split: {sorted(overlap)}")


def assert_all_animal_observations_together(frame: pd.DataFrame, assignments: pd.DataFrame, group_by: str = "animal_id") -> None:
    merged = frame[[group_by]].merge(assignments[[group_by, "partition"]], on=group_by, how="left")
    leaking = merged.groupby(group_by).partition.nunique()
    if (leaking > 1).any():
        raise ValueError(f"Animals assigned to multiple partitions: {leaking[leaking > 1].index.tolist()}")
