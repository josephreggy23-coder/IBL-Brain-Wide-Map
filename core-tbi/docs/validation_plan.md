# Validation plan

All frames, strides, sessions, and time points from an animal belong to exactly one partition. Primary inference and metrics are animal-level. Required validation includes GroupKFold, leave-one-animal-out, and where suitable leave-one-dataset-out. Report balanced accuracy, AUROC, PR-AUC, macro F1, Brier score, calibration, uncertainty coverage, reliability, and speed/missingness/dataset ablations. Never call stride count biological sample size.
