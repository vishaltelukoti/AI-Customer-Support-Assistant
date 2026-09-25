"""Subgroup performance diagnostics using existing non-sensitive metadata."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from ..ml.baseline_data import prepare_ticket_text
from ..ml.optimization import OPTIMIZED_MODEL_FILES
from ..ml.preprocessing import ROOT

EXPLAINABILITY_DIR = ROOT / "experiments/explainability"
OPTIMIZED_MODELS_DIR = ROOT / "experiments/optimization/models"
DATA_DIR = ROOT / "data/processed"
MIN_GROUP_SIZE = 50
NON_SENSITIVE_GROUP_FIELDS = ("version", "type")
SENSITIVE_ATTRIBUTES_NOT_INFERRED = ("race", "gender", "age", "religion", "disability")


@dataclass(frozen=True)
class GroupMetric:
    field: str
    group: str
    sample_count: int
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float


def load_test_metadata(data_dir: Path = DATA_DIR) -> dict[str, dict[str, str]]:
    with (data_dir / "historical_tickets.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        return {row["ticket_id"]: row for row in rows if row["split"] == "test"}


def load_historical_test_records(data_dir: Path = DATA_DIR) -> list[dict[str, str]]:
    with (data_dir / "historical_tickets.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            if row["split"] == "test":
                prepared = dict(row)
                prepared["ticket_text"] = prepare_ticket_text(row["ticket_text"])
                rows.append(prepared)
        if not rows:
            raise ValueError("No test records found in historical_tickets.csv.")
        return rows


def subgroup_metrics(y_true: list[str], y_pred: list[str], groups: list[str], field: str,
                     min_group_size: int = MIN_GROUP_SIZE) -> list[GroupMetric]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        if group:
            grouped[group].append(index)

    metrics = []
    for group in sorted(grouped):
        indices = grouped[group]
        if len(indices) < min_group_size:
            continue
        true = [y_true[index] for index in indices]
        pred = [y_pred[index] for index in indices]
        precision, recall, f1, _ = precision_recall_fscore_support(
            true, pred, average="macro", zero_division=0
        )
        metrics.append(GroupMetric(
            field=field,
            group=group,
            sample_count=len(indices),
            accuracy=float(accuracy_score(true, pred)),
            precision_macro=float(precision),
            recall_macro=float(recall),
            f1_macro=float(f1),
        ))
    return metrics


def largest_metric_differences(metrics: list[GroupMetric]) -> dict[str, dict[str, float | str]] | None:
    if len(metrics) < 2:
        return None
    differences: dict[str, dict[str, float | str]] = {}
    for metric_name in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
        low = min(metrics, key=lambda item: getattr(item, metric_name))
        high = max(metrics, key=lambda item: getattr(item, metric_name))
        differences[metric_name] = {
            "difference": float(getattr(high, metric_name) - getattr(low, metric_name)),
            "lowest_group": low.group,
            "highest_group": high.group,
        }
    return differences


def run_fairness_evaluation(data_dir: Path = DATA_DIR, model_dir: Path = OPTIMIZED_MODELS_DIR,
                            output_dir: Path = EXPLAINABILITY_DIR,
                            min_group_size: int = MIN_GROUP_SIZE) -> dict[str, Any]:
    test_records = load_historical_test_records(data_dir)

    model = joblib.load(model_dir / OPTIMIZED_MODEL_FILES["category"])
    texts = [row["ticket_text"] for row in test_records]
    labels = [row["category"] for row in test_records]
    predictions = list(model.predict(texts))

    fields = {}
    for field in NON_SENSITIVE_GROUP_FIELDS:
        values = [row[field] for row in test_records]
        field_metrics = subgroup_metrics(labels, predictions, values, field, min_group_size)
        excluded = {
            group: count for group, count in sorted(_counts(values).items())
            if count < min_group_size
        }
        fields[field] = {
            "evaluated_groups": [metric.__dict__ for metric in field_metrics],
            "excluded_groups": excluded,
            "largest_observed_differences": largest_metric_differences(field_metrics),
        }

    evaluation = {
        "model": "Day 3 optimized category TF-IDF + Logistic Regression",
        "dataset": {
            "split": "test",
            "sample_count": len(test_records),
            "metadata_source": str((data_dir / "historical_tickets.csv").relative_to(ROOT)),
        },
        "group_fields_considered": {
            "evaluated": list(NON_SENSITIVE_GROUP_FIELDS),
            "language": "All classification rows are English-only, so language is documented but not used for a valid subgroup comparison.",
            "sensitive_attributes_not_inferred": list(SENSITIVE_ATTRIBUTES_NOT_INFERRED),
        },
        "minimum_group_size": min_group_size,
        "fields": fields,
        "limitations": [
            "This is a subgroup performance diagnostic, not proof of demographic fairness.",
            "The dataset is English-only for classification, so it does not support an English-vs-German fairness comparison.",
            "No race, gender, age, religion, disability or other sensitive attributes are inferred.",
            "Version and type are dataset metadata fields, not protected demographic groups.",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "fairness_evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    (output_dir / "fairness_evaluation.md").write_text(format_fairness_markdown(evaluation), encoding="utf-8")
    return evaluation


def _counts(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        counts[value] += 1
    return dict(counts)


def format_fairness_markdown(evaluation: dict[str, Any]) -> str:
    lines = [
        "# Day 10 Subgroup Diagnostic",
        "",
        "This evaluates the saved Day 3 category model on the held-out test split using only available non-sensitive metadata.",
        "",
        f"Minimum group size: {evaluation['minimum_group_size']}",
        "",
    ]
    for field, payload in evaluation["fields"].items():
        lines.extend([
            f"## {field}",
            "",
            "| Group | Count | Accuracy | Macro precision | Macro recall | Macro F1 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for metric in payload["evaluated_groups"]:
            lines.append(
                f"| {metric['group']} | {metric['sample_count']} | {metric['accuracy']:.6f} | "
                f"{metric['precision_macro']:.6f} | {metric['recall_macro']:.6f} | {metric['f1_macro']:.6f} |"
            )
        if payload["excluded_groups"]:
            lines.extend(["", f"Excluded below threshold: `{payload['excluded_groups']}`"])
        if payload["largest_observed_differences"]:
            lines.extend(["", "Largest observed metric differences:"])
            for name, diff in payload["largest_observed_differences"].items():
                lines.append(
                    f"- {name}: {diff['difference']:.6f} "
                    f"({diff['lowest_group']} to {diff['highest_group']})"
                )
        lines.append("")
    lines.extend([
        "## Limitations",
        "",
        "- This is not proof that the model is fair or biased.",
        "- The classification data is English-only, so no valid English-vs-German fairness comparison is available.",
        "- No sensitive demographic attributes are inferred.",
        "- Version and ticket type may reflect synthetic data-generation patterns rather than real user populations.",
    ])
    return "\n".join(lines) + "\n"
