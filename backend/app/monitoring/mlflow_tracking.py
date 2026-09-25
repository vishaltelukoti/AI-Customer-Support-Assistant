"""Log existing Day 3 optimized model metrics to a local MLflow store."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any

from ..ml.preprocessing import ROOT
from .mlops_metrics import MLOPS_DIR, write_ml_metrics

MLFLOW_DIR = ROOT / "experiments/mlflow"


def run_mlflow_tracking(output_dir: Path = MLOPS_DIR, mlflow_dir: Path = MLFLOW_DIR) -> dict[str, Any]:
    mlflow = _import_mlflow()
    metrics = write_ml_metrics(output_dir)
    tracking_uri = (mlflow_dir / "mlruns").resolve().as_uri()
    mlflow_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("ai-customer-support-day11")

    category = metrics["classification"]["category"]
    priority = metrics["classification"]["priority"]
    retrieval = metrics["retrieval"]
    with mlflow.start_run(run_name="day3-optimized-classifiers-and-day6-retrieval") as run:
        mlflow.set_tag("model_name", "day3_optimized_tfidf_logreg")
        mlflow.set_tag("model_type", "TF-IDF + Logistic Regression")
        mlflow.set_tag("retrieval_index_type", retrieval["index_type"])
        mlflow.log_metric("category_test_accuracy", category["accuracy"])
        mlflow.log_metric("category_test_macro_f1", category["macro_f1"])
        mlflow.log_metric("priority_test_accuracy", priority["accuracy"])
        mlflow.log_metric("priority_test_macro_f1", priority["macro_f1"])
        mlflow.log_metric("retrieval_recall_at_1", retrieval["recall@1"])
        mlflow.log_metric("retrieval_recall_at_3", retrieval["recall@3"])
        mlflow.log_metric("retrieval_recall_at_5", retrieval["recall@5"])
        mlflow.log_metric("retrieval_mrr", retrieval["mrr"])
        for prefix, payload in (("category", category), ("priority", priority)):
            mlflow.log_param(f"{prefix}_selected_method", payload["selected_method"])
            for name, value in payload["parameters"].items():
                mlflow.log_param(f"{prefix}_{name}", str(value))
        mlflow.log_param("classification_source", metrics["classification"]["source_artifact"])
        mlflow.log_param("retrieval_source", metrics["retrieval"]["source_artifact"])
        run_id = run.info.run_id
        experiment_id = run.info.experiment_id

    config = {
        "tracking_uri": tracking_uri,
        "experiment_name": "ai-customer-support-day11",
        "local_only": True,
        "mlflow_ui_command": f"mlflow ui --backend-store-uri {tracking_uri}",
    }
    result = {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "tracked_metrics": {
            "category_test_accuracy": category["accuracy"],
            "category_test_macro_f1": category["macro_f1"],
            "priority_test_accuracy": priority["accuracy"],
            "priority_test_macro_f1": priority["macro_f1"],
            "retrieval_recall_at_1": retrieval["recall@1"],
            "retrieval_recall_at_3": retrieval["recall@3"],
            "retrieval_recall_at_5": retrieval["recall@5"],
            "retrieval_mrr": retrieval["mrr"],
        },
        "environment": {"python": platform.python_version()},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "mlflow_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "mlflow_tracking_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return {"config": config, "result": result, "metrics": metrics}


def _import_mlflow():
    """Import the installed MLflow package despite the repo's placeholder mlflow/ directory."""
    removed = []
    root_text = str(ROOT)
    for entry in list(sys.path):
        resolved = str(Path(entry or ".").resolve())
        if resolved == root_text:
            sys.path.remove(entry)
            removed.append(entry)
    try:
        sys.modules.pop("mlflow", None)
        import mlflow  # type: ignore
        if not hasattr(mlflow, "set_tracking_uri"):
            raise ImportError("Installed MLflow package was not found.")
        return mlflow
    finally:
        for entry in reversed(removed):
            sys.path.insert(0, entry)


if __name__ == "__main__":
    output = run_mlflow_tracking()
    print(json.dumps(output["result"], indent=2))
