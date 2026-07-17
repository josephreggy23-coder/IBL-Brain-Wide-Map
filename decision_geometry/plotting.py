"""Publication-style summary figure for the population analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .analysis import AnalysisResult
from .data import PopulationDataset


COLORS = {
    "choice": "#0072B2",
    "stimulus": "#D55E00",
    "prior": "#009E73",
    "clockwise": "#CC79A7",
    "counter-clockwise": "#0072B2",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.titlesize": 10,
            "figure.dpi": 150,
        }
    )


def save_dashboard(
    dataset: PopulationDataset,
    result: AnalysisResult,
    path: str | Path,
) -> Path:
    _style()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(13, 8), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.0, 1.12))

    ax = fig.add_subplot(grid[0, 0])
    for name, trajectory in result.pca_trajectories.items():
        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color=COLORS[name],
            linewidth=2.2,
            label=name,
        )
        ax.scatter(trajectory[0, 0], trajectory[0, 1], color=COLORS[name], marker="o", s=25)
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], color=COLORS[name], marker="s", s=28)
    ax.set_title("A  Choice-conditioned neural trajectories", loc="left")
    ax.set_xlabel(f"PC1 ({result.explained_variance[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({result.explained_variance[1] * 100:.1f}% variance)")
    ax.legend(frameon=False, title="wheel choice")
    ax.text(0.02, 0.02, "circle: -0.5 s   square: +1.0 s", transform=ax.transAxes, color="#555555")

    ax = fig.add_subplot(grid[0, 1])
    for name in ("stimulus", "prior", "choice"):
        ax.plot(dataset.time, result.decoding[name], color=COLORS[name], linewidth=2, label=name)
    ax.axhline(0.5, color="#777777", linewidth=1, linestyle="--")
    ax.axvline(0.0, color="#222222", linewidth=1)
    ax.set_ylim(0.4, 1.01)
    ax.set_title("B  Information becomes linearly readable over time", loc="left")
    ax.set_xlabel("time from stimulus onset (s)")
    ax.set_ylabel("cross-validated balanced accuracy")
    ax.legend(frameon=False, ncol=3)

    ax = fig.add_subplot(grid[1, 0])
    image = ax.imshow(
        result.cross_temporal_choice,
        origin="lower",
        aspect="auto",
        extent=(dataset.time[0], dataset.time[-1], dataset.time[0], dataset.time[-1]),
        vmin=0.45,
        vmax=max(0.75, float(result.cross_temporal_choice.max())),
        cmap="magma",
    )
    ax.axhline(0.0, color="white", linewidth=0.7, alpha=0.8)
    ax.axvline(0.0, color="white", linewidth=0.7, alpha=0.8)
    ax.set_title("C  Choice-code stability across time", loc="left")
    ax.set_xlabel("test time (s)")
    ax.set_ylabel("train time (s)")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.02)
    colorbar.set_label("balanced accuracy")

    ax = fig.add_subplot(grid[1, 1])
    region_items = sorted(
        result.region_decoding.items(),
        key=lambda item: float(np.max(item[1])),
        reverse=True,
    )
    region_names = [name for name, _ in region_items]
    region_matrix = np.vstack([scores for _, scores in region_items])
    image = ax.imshow(
        region_matrix,
        origin="upper",
        aspect="auto",
        extent=(dataset.time[0], dataset.time[-1], len(region_names) - 0.5, -0.5),
        vmin=0.45,
        vmax=max(0.75, float(region_matrix.max())),
        cmap="viridis",
    )
    ax.axvline(0.0, color="white", linewidth=0.7, alpha=0.8)
    ax.set_yticks(np.arange(len(region_names)), labels=region_names)
    ax.set_title("D  Choice information by anatomical region", loc="left")
    ax.set_xlabel("time from stimulus onset (s)")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.02)
    colorbar.set_label("balanced accuracy")

    fig.suptitle(
        "Decision geometry in a real IBL Neuropixels session\n"
        f"subject {dataset.subject_id} | {dataset.rates.shape[0]} trials | "
        f"{dataset.rates.shape[1]} quality-filtered units",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path
