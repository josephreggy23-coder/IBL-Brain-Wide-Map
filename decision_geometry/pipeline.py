"""End-to-end orchestration for the public-data analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .analysis import analyze_population
from .data import load_population, provenance
from .plotting import save_dashboard


def run_pipeline(
    *,
    cache_path: str | Path = "data/cache/session_population.npz",
    output_dir: str | Path = "results",
    force_stream: bool = False,
    bin_size: float = 0.05,
    max_units: int | None = 96,
) -> dict:
    dataset = load_population(
        cache_path,
        force_stream=force_stream,
        bin_size=bin_size,
        max_units=max_units,
    )
    result = analyze_population(dataset)
    output_dir = Path(output_dir)
    figure_path = save_dashboard(dataset, result, output_dir / "decision_geometry.png")

    choice_scores = result.decoding["choice"]
    stimulus_scores = result.decoding["stimulus"]
    prior_scores = result.decoding["prior"]
    summary = {
        "data": provenance(),
        "n_trials": int(dataset.rates.shape[0]),
        "n_units": int(dataset.rates.shape[1]),
        "n_regions": int(np.unique(dataset.unit_regions).size),
        "quality_filter": "ibl_quality_score=1, presence_ratio>=0.9, 0.1<=firing_rate<=100 Hz",
        "peak_choice_accuracy": float(choice_scores.max()),
        "peak_choice_time_s": float(dataset.time[np.argmax(choice_scores)]),
        "peak_stimulus_accuracy": float(stimulus_scores.max()),
        "peak_prior_accuracy": float(prior_scores.max()),
        "figure": str(figure_path),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    return summary
