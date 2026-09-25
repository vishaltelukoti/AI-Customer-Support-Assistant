"""Create compact MLOps metric summaries from existing experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..ml.preprocessing import ROOT

OPTIMIZATION_RESULTS = ROOT / "experiments/optimization/optimization_results.json"
RETRIEVAL_RESULTS = ROOT / "experiments/retrieval/retrieval_evaluation.json"
MLOPS_DIR = ROOT / "experiments/mlops"


def load_ml_metrics(optimization_path: Path = OPTIMIZATION_RESULTS,
                    retrieval_path: Path = RETRIEVAL_RESULTS) -> dict[str, Any]:
    optimization = json.loads(optimization_path.read_text(encoding="utf-8"))
    retrieval = json.loads(retrieval_path.read_text(encoding="utf-8"))
    return {
        "classification": {
            "source_artifact": str(optimization_path.relative_to(ROOT)),
            "category": {
                "accuracy": optimization["category"]["test"]["accuracy"],
                "macro_f1": optimization["category"]["test"]["f1_macro"],
                "selected_method": optimization["category"]["selected"]["method"],
                "parameters": optimization["category"]["selected"]["parameters"],
            },
            "priority": {
                "accuracy": optimization["priority"]["test"]["accuracy"],
                "macro_f1": optimization["priority"]["test"]["f1_macro"],
                "selected_method": optimization["priority"]["selected"]["method"],
                "parameters": optimization["priority"]["selected"]["parameters"],
            },
        },
        "retrieval": {
            "source_artifact": str(retrieval_path.relative_to(ROOT)),
            "recall@1": retrieval["metrics"]["recall@1"],
            "recall@3": retrieval["metrics"]["recall@3"],
            "recall@5": retrieval["metrics"]["recall@5"],
            "mrr": retrieval["metrics"]["mrr"],
            "index_type": retrieval["index_type"],
            "indexed_tickets": retrieval["leakage_checks"]["indexed_tickets"],
        },
    }


def write_ml_metrics(output_dir: Path = MLOPS_DIR) -> dict[str, Any]:
    metrics = load_ml_metrics()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ml_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "ml_metrics.md").write_text(format_metrics_markdown(metrics), encoding="utf-8")
    return metrics


def format_metrics_markdown(metrics: dict[str, Any]) -> str:
    category = metrics["classification"]["category"]
    priority = metrics["classification"]["priority"]
    retrieval = metrics["retrieval"]
    return "\n".join([
        "# Day 11 MLOps Metrics Summary",
        "",
        "Metrics are loaded from existing experiment artifacts; no model training or retrieval evaluation is rerun.",
        "",
        "## Classification",
        "",
        "| Target | Accuracy | Macro F1 | Source |",
        "| --- | ---: | ---: | --- |",
        f"| Category | {category['accuracy']:.6f} | {category['macro_f1']:.6f} | {metrics['classification']['source_artifact']} |",
        f"| Priority | {priority['accuracy']:.6f} | {priority['macro_f1']:.6f} | {metrics['classification']['source_artifact']} |",
        "",
        "## Retrieval",
        "",
        "| Metric | Value | Source |",
        "| --- | ---: | --- |",
        f"| Recall@1 | {retrieval['recall@1']:.6f} | {retrieval['source_artifact']} |",
        f"| Recall@3 | {retrieval['recall@3']:.6f} | {retrieval['source_artifact']} |",
        f"| Recall@5 | {retrieval['recall@5']:.6f} | {retrieval['source_artifact']} |",
        f"| MRR | {retrieval['mrr']:.6f} | {retrieval['source_artifact']} |",
        "",
    ])


if __name__ == "__main__":
    print(json.dumps(write_ml_metrics(), indent=2))
