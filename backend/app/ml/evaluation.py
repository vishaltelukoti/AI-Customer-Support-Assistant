"""Evaluation and readable result artifacts. No estimator is fitted here."""

import csv
from pathlib import Path
from textwrap import fill

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

METRICS = (
    "accuracy", "precision_macro", "recall_macro", "f1_macro",
    "precision_weighted", "recall_weighted", "f1_weighted",
)


def evaluate_classifier(model: Pipeline, texts: list[str], labels: list[str]) -> dict:
    """Calculate aggregate and per-class metrics with a fixed, complete label order."""
    predicted = model.predict(texts)
    classes = model.classes_.tolist()
    report = classification_report(labels, predicted, labels=classes, output_dict=True, zero_division=0)
    metrics = {"accuracy": float(accuracy_score(labels, predicted))}
    for average in ("macro", "weighted"):
        for name, report_key in (("precision", "precision"), ("recall", "recall"), ("f1", "f1-score")):
            metrics[f"{name}_{average}"] = float(report[f"{average} avg"][report_key])
    return {**metrics, "classification_report": report, "labels": classes,
            "confusion_matrix": confusion_matrix(labels, predicted, labels=classes).tolist(),
            "support": len(labels)}


def save_confusion_matrix(result: dict, target: str, output_dir: Path) -> None:
    """Save the final-test count matrix as both PNG and labelled CSV."""
    # Import plotting only for export; the API and inference do not need a GUI backend.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    import numpy as np

    labels = result["labels"]
    matrix = np.asarray(result["confusion_matrix"])
    size = (13, 11) if len(labels) > 3 else (7, 6)
    figure = Figure(figsize=size, layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    plotted = axis.imshow(matrix, cmap="Blues", vmin=0)
    wrapped = [fill(label, 19) for label in labels]
    axis.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=wrapped,
             yticklabels=wrapped, xlabel="Predicted label", ylabel="True label",
             title=f"{target.capitalize()} baseline — test set (counts)")
    axis.tick_params(axis="x", labelrotation=45, labelsize=9)
    axis.tick_params(axis="y", labelsize=9)
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")
    threshold = matrix.max() / 2
    for (row, column), value in np.ndenumerate(matrix):
        axis.text(column, row, str(value), ha="center", va="center",
                  color="white" if value > threshold else "#16232e", fontsize=9)
    figure.colorbar(plotted, ax=axis, shrink=.75, label="Tickets")
    figure.savefig(output_dir / f"{target}_confusion_matrix.png", dpi=160)
    with (output_dir / f"{target}_confusion_matrix.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["true_label / predicted_label", *labels])
        writer.writerows([[label, *row] for label, row in zip(labels, matrix.tolist())])


def metrics_markdown(results: dict) -> str:
    """Render measured results without manually transcribing any metric."""
    lines = ["# Baseline results", "", "Generated from baseline_results.json; scores are fractions (0–1).", ""]
    for target in ("category", "priority"):
        lines += [f"## {target.capitalize()}", "", "| Metric | Train | Validation | Test |",
                  "| --- | ---: | ---: | ---: |"]
        for metric in METRICS:
            values = [f"{results[target][split][metric]:.6f}" for split in ("train", "validation", "test")]
            lines.append(f"| {metric} | {' | '.join(values)} |")
        lines += ["", "### Test per-class report", "", "| Class | Precision | Recall | F1 | Support |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for label in results[target]["test"]["labels"]:
            row = results[target]["test"]["classification_report"][label]
            lines.append(f"| {label} | {row['precision']:.6f} | {row['recall']:.6f} | {row['f1-score']:.6f} | {int(row['support'])} |")
        lines.append("")
    return "\n".join(lines) + "\n"
