from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_recovery_plane(scores: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    colors = {"restitution": "#2a9d8f", "compensation": "#e9c46a", "persistent_dysfunction": "#e76f51", "uncertain": "#8d99ae"}
    fig, ax = plt.subplots(figsize=(7, 5))
    for state, group in scores.groupby("recovery_state"):
        ax.scatter(group.performance_deviation, group.counterfactual_deviation, label=state.replace("_", " "), color=colors.get(state, "gray"), alpha=0.8)
    ax.set(xlabel="Performance deviation from own baseline", ylabel="Counterfactual movement deviation", title="CoRe-TBI recovery plane")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)
    return path
