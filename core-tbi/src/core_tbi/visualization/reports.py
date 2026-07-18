from __future__ import annotations
from pathlib import Path
import pandas as pd
import plotly.express as px


def animal_report(scores: pd.DataFrame, animal_id: str, output: str | Path) -> Path:
    subset = scores.loc[scores.animal_id == animal_id].sort_values("days_post_injury")
    if subset.empty:
        raise ValueError(f"No observations for animal {animal_id}")
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    figure = px.line(subset, x="days_post_injury", y="counterfactual_deviation", color="recovery_state", markers=True, hover_data=["session_id", "task_success", "compensation_burden"])
    disclaimer = "<p><strong>Interpretation:</strong> compensation is an operational computational proxy, not biological proof. This report may contain synthetic demonstration data.</p>"
    output.write_text(f"<h1>CoRe-TBI recovery report: {animal_id}</h1>{disclaimer}{figure.to_html(full_html=False, include_plotlyjs='cdn')}{subset.to_html(index=False)}", encoding="utf-8")
    return output


def cohort_report(scores: pd.DataFrame, output: str | Path) -> Path:
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    summary = scores.groupby(["timepoint", "recovery_state"], as_index=False).size()
    figure = px.scatter(scores, x="performance_deviation", y="counterfactual_deviation", color="recovery_state", hover_data=["animal_id", "timepoint"])
    output.write_text("<h1>CoRe-TBI cohort report</h1><p>Animal-level inference only; no causal or definitive compensation claim.</p>" + figure.to_html(full_html=False, include_plotlyjs="cdn") + summary.to_html(index=False), encoding="utf-8")
    return output
