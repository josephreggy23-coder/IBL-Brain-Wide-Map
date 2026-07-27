"""Export compact, exact analysis results for the interactive web explorer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from decision_geometry.analysis import analyze_population
from decision_geometry.config import (
    DEFAULT_BIN_SIZE,
    DEFAULT_MAX_UNITS,
    DEFAULT_WINDOW,
)
from decision_geometry.data import PopulationDataset, provenance


def _rounded(values: np.ndarray, digits: int = 4):
    return np.round(values, digits).tolist()


def main() -> None:
    cache_path = Path("data/cache/session_population.npz")
    if not cache_path.exists():
        raise SystemExit("Run decision-geometry once to create the session cache.")

    dataset = PopulationDataset.from_cache(cache_path)
    result = analyze_population(dataset)
    peak_choice_index = int(np.argmax(result.decoding["choice"]))
    payload = {
        "provenance": provenance(),
        "analysis": {
            "randomSeed": 7,
            "binSizeSeconds": DEFAULT_BIN_SIZE,
            "windowSeconds": list(DEFAULT_WINDOW),
            "maxUnitsRequested": DEFAULT_MAX_UNITS,
            "crossValidationFolds": 5,
            "bootstrapResamples": 1000,
            "metric": "balanced accuracy",
            "classifier": "standardized class-balanced logistic regression (C=1.0)",
        },
        "summary": {
            "trials": int(dataset.rates.shape[0]),
            "units": int(dataset.rates.shape[1]),
            "regions": int(np.unique(dataset.unit_regions).size),
            "peakChoiceAccuracy": float(result.decoding["choice"].max()),
            "peakChoiceTime": float(dataset.time[peak_choice_index]),
            "peakChoiceCi95": _rounded(
                result.decoding_ci["choice"][:, peak_choice_index]
            ),
        },
        "time": _rounded(dataset.time),
        "decoding": {
            name: _rounded(scores) for name, scores in result.decoding.items()
        },
        "decodingCi": {
            name: _rounded(interval)
            for name, interval in result.decoding_ci.items()
        },
        "crossTemporalChoice": _rounded(result.cross_temporal_choice),
        "regionDecoding": {
            name: _rounded(scores) for name, scores in result.region_decoding.items()
        },
        "trajectories": {
            name: _rounded(points)
            for name, points in result.pca_trajectories.items()
        },
        "explainedVariance": _rounded(result.explained_variance),
    }

    app_path = Path("web/app/analysis-data.json")
    receipt_path = Path("web/public/analysis-receipt.json")
    app_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    app_path.write_text(
        json.dumps(payload, separators=(",", ":")),
        encoding="utf-8",
    )
    receipt_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {app_path} and {receipt_path}")


if __name__ == "__main__":
    main()
