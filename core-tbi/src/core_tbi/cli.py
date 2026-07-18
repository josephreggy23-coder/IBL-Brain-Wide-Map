from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
import typer
from core_tbi.data.inventory import inventory_directory
from core_tbi.data.download import download_figshare_file, fetch_figshare_inventory
from core_tbi.data.alma import extract_alma_feature_cycles, inventory_alma_workbook
from core_tbi.data.synthetic import FEATURES, make_synthetic_longitudinal_tbi
from core_tbi.models.baseline import RecoveryPlaneModel
from core_tbi.models.conventional import evaluate_conventional_models
from core_tbi.evaluation.metrics import animal_level_metrics
from core_tbi.registry import get_dataset, load_registry
from core_tbi.visualization.recovery_plane import plot_recovery_plane
from core_tbi.visualization.reports import animal_report, cohort_report
from core_tbi.reproducibility import write_run_manifest

app = typer.Typer(help="CoRe-TBI: animal-level counterfactual recovery modelling.")
datasets_app = typer.Typer(help="Dataset registry and non-destructive inventory tools.")
app.add_typer(datasets_app, name="datasets")


def _paths():
    root = Path.cwd(); return root / "data", root / "outputs"


@datasets_app.command("list")
def datasets_list():
    for name, entry in load_registry().items(): typer.echo(f"{name}: {entry['role']}")


@datasets_app.command("inspect")
def datasets_inspect(dataset: str):
    entry = get_dataset(dataset); data, outputs = _paths()
    table = inventory_directory(data / "raw" / dataset, outputs / "reports" / f"{dataset}_inventory.html") if (data / "raw" / dataset).exists() else pd.DataFrame()
    typer.echo(json.dumps(entry, indent=2)); typer.echo(f"Inventory rows: {len(table)}. No download or metadata inference performed.")


@datasets_app.command("official-inventory")
def official_inventory(dataset: str):
    """Save official Figshare metadata before requesting a file download."""
    data, _ = _paths()
    record = fetch_figshare_inventory(dataset, data / "raw")
    typer.echo(f"Official inventory saved for {record['title']}: {len(record['files'])} files.")
    for item in record["files"]:
        typer.echo(f"  {item['id']}: {item['name']} ({item['size']} bytes)")


@datasets_app.command("download")
def datasets_download(dataset: str, file_id: int = typer.Option(..., help="File ID listed by official-inventory.")):
    data, _ = _paths()
    typer.echo(download_figshare_file(dataset, file_id, data / "raw"))


@app.command("demo")
def demo():
    _, outputs = _paths(); outputs.mkdir(exist_ok=True)
    frame = make_synthetic_longitudinal_tbi()
    model = RecoveryPlaneModel(FEATURES).fit(frame)
    scores = model.transform(frame)
    scores.to_csv(outputs / "tables" / "synthetic_recovery_scores.csv", index=False)
    plot_recovery_plane(scores, outputs / "figures" / "synthetic_recovery_plane.png")
    cohort_report(scores, outputs / "reports" / "demo_cohort.html")
    animal_report(scores, scores.animal_id.iloc[0], outputs / "reports" / f"{scores.animal_id.iloc[0]}.html")
    post = frame.loc[frame.timepoint != "baseline"]
    predictions, assignments = evaluate_conventional_models(post, FEATURES)
    metrics = {name: animal_level_metrics(group) for name, group in predictions.groupby("model")}
    (outputs / "tables" / "group_assignments.csv").parent.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(outputs / "tables" / "group_assignments.csv", index=False)
    predictions.to_csv(outputs / "tables" / "conventional_oof_predictions.csv", index=False)
    (outputs / "tables" / "conventional_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_run_manifest(outputs / "reports" / "demo_manifest.json", command="core-tbi demo", inputs={"dataset": "synthetic_tbi", "is_synthetic": True})
    typer.echo("Synthetic demo complete. Open outputs/reports/demo_cohort.html")


@app.command("report")
def report(animal: str, input: Path = typer.Option(Path("outputs/tables/synthetic_recovery_scores.csv"))):
    _, outputs = _paths(); typer.echo(animal_report(pd.read_csv(input), animal, outputs / "reports" / f"{animal}.html"))


@app.command("prepare")
def prepare(dataset: str = typer.Option(...)):
    data, outputs = _paths()
    if dataset == "alma":
        files = list((data / "raw" / "alma").glob("*.xlsx"))
        if not files:
            raise typer.BadParameter("No ALMA workbook found. Run `core-tbi datasets official-inventory alma`, then download a listed file.")
        table = inventory_alma_workbook(files[0])
        target = outputs / "tables" / "alma_workbook_inventory.csv"
        target.parent.mkdir(parents=True, exist_ok=True); table.to_csv(target, index=False)
        cycles = extract_alma_feature_cycles(files[0])
        cycle_target = outputs / "processed" / "alma_feature_cycles.csv"
        cycle_target.parent.mkdir(parents=True, exist_ok=True); cycles.to_csv(cycle_target, index=False)
        write_run_manifest(outputs / "reports" / "alma_prepare_manifest.json", command="core-tbi prepare --dataset alma", inputs={"source_file": str(files[0]), "source_sheet": "Figure_2_Panel_G", "animal_id_available": False})
        typer.echo(f"ALMA workbook inventoried: {target}. Extracted {len(cycles)} source-table feature rows to {cycle_target}. Animal IDs are absent, so animal-level inference is intentionally blocked.")
        return
    typer.echo(f"Adapter placeholder for {dataset}: inspect actual files first with `core-tbi datasets inspect {dataset}`. Raw data are never overwritten.")


@app.command("evaluate")
def evaluate(group_by: str = "animal_id"):
    typer.echo(f"Evaluation requires saved model predictions. Grouping is fixed at {group_by}; frame- or stride-random splits are forbidden.")


@app.command("qc")
def qc(input: Path = typer.Option(Path("outputs/tables/synthetic_recovery_scores.csv")), output: Path = typer.Option(Path("outputs/tables/qc_summary.csv"))):
    """Export missingness and tracking-quality summary without deleting rows."""
    frame = pd.read_csv(input)
    result = pd.DataFrame({"column": frame.columns, "missing_values": [int(frame[column].isna().sum()) for column in frame], "missing_fraction": [float(frame[column].isna().mean()) for column in frame]})
    if "tracking_quality" in frame:
        result = pd.concat([result, pd.DataFrame([{"column": "tracking_quality_below_0.8", "missing_values": int((frame.tracking_quality < .8).sum()), "missing_fraction": float((frame.tracking_quality < .8).mean())}])], ignore_index=True)
    output.parent.mkdir(parents=True, exist_ok=True); result.to_csv(output, index=False)
    typer.echo(f"QC summary written to {output}; no observations were excluded.")


@app.command("train-feature")
def train_feature(input: Path = typer.Option(Path("data/processed/features.csv")), config: Path = Path("configs/feature_model.yaml")):
    """Run leakage-safe feature baselines when source data identify animals."""
    if not input.exists():
        raise typer.BadParameter(f"Input not found: {input}. Use `core-tbi demo` or supply a per-animal feature CSV.")
    frame = pd.read_csv(input)
    required = {"animal_id", "timepoint", "condition"}
    missing = required - set(frame)
    if missing or frame.animal_id.isna().any():
        raise typer.BadParameter("Animal-level training is blocked: input lacks stable animal_id, timepoint, or condition. ALMA figure-source extraction is descriptive until participant IDs are available.")
    features = [column for column in FEATURES if column in frame]
    if not features:
        raise typer.BadParameter("No configured standard feature columns were found in the input.")
    predictions, assignments = evaluate_conventional_models(frame.loc[frame.timepoint != "baseline"], features)
    _, outputs = _paths(); outputs.joinpath("tables").mkdir(parents=True, exist_ok=True)
    predictions.to_csv(outputs / "tables" / "feature_training_oof_predictions.csv", index=False)
    assignments.to_csv(outputs / "tables" / "feature_training_assignments.csv", index=False)
    write_run_manifest(outputs / "reports" / "feature_training_manifest.json", command="core-tbi train-feature", inputs={"input": str(input), "config": str(config), "features": features})
    typer.echo("Feature baseline complete with animal-grouped out-of-fold predictions.")


if __name__ == "__main__":
    app()
