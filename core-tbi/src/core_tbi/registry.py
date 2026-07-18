from __future__ import annotations

from pathlib import Path
import yaml


def load_registry(path: str | Path = "configs/datasets.yaml") -> dict:
    """Load the machine-readable registry without attempting data download."""
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)["datasets"]


def get_dataset(dataset_id: str, path: str | Path = "configs/datasets.yaml") -> dict:
    registry = load_registry(path)
    if dataset_id not in registry:
        raise KeyError(f"Unknown dataset '{dataset_id}'. Available: {', '.join(registry)}")
    return registry[dataset_id]
