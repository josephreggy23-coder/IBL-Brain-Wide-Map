"""Export compact, exact analysis results for the interactive web explorer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from decision_geometry.analysis import analyze_population
from decision_geometry.data import PopulationDataset, provenance


def _rounded(values: np.ndarray, digits: int = 4):
    return np.round(values, digits).tolist()


def main() -> None:
    cache_path = Path("data/cache/session_population.npz")
    if not cache_path.exists():
        raise SystemExit("Run decision-geometry once to create the session cache.")

    dataset = PopulationDataset.from_cache(cache_path)
    result = analyze_population(dataset)
    payload = {
        "provenance": provenance(),
        "summary": {
            "trials": int(dataset.rates.shape[0]),
            "units": int(dataset.rates.shape[1]),
            "regions": int(np.unique(dataset.unit_regions).size),
            "peakChoiceAccuracy": float(result.decoding["choice"].max()),
            "peakChoiceTime": float(dataset.time[np.argmax(result.decoding["choice"])]),
        },
        "time": _rounded(dataset.time),
        "decoding": {
            name: _rounded(scores) for name, scores in result.decoding.items()
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

    output_path = Path("web/app/analysis-data.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
