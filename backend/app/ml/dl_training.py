"""Train Day 4 category-only CNN/RNN/LSTM comparison models."""

import argparse
import json
import logging
import platform
import time
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
from sklearn.metrics import accuracy_score, classification_report
from threadpoolctl import threadpool_limits

from .baseline_data import load_splits
from .dataset_audit import CATEGORIES
from .dl_models import DLConfig, build_dl_model
from .preprocessing import ROOT, SEED

LOGGER = logging.getLogger(__name__)
DL_DIR = ROOT / "experiments/dl_comparison"
MODEL_NAMES = ("cnn", "rnn", "lstm")


def _metrics(labels: list[str], predictions: list[str]) -> dict:
    report = classification_report(labels, predictions, labels=CATEGORIES, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_macro": float(report["macro avg"]["precision"]),
        "recall_macro": float(report["macro avg"]["recall"]),
        "f1_macro": float(report["macro avg"]["f1-score"]),
    }


def _results_markdown(results: dict) -> str:
    lines = [
        "# Day 4 DL comparison",
        "",
        "Category-only CNN/RNN/LSTM comparison on the unchanged train/validation/test split.",
        "",
        "| Model | Validation Macro-F1 | Test Accuracy | Test Precision Macro | Test Recall Macro | Test Macro-F1 | Training time seconds |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in MODEL_NAMES:
        validation = results["models"][name]["validation"]
        test = results["models"][name]["test"]
        lines.append(
            f"| {name.upper()} | {validation['f1_macro']:.6f} | {test['accuracy']:.6f} | "
            f"{test['precision_macro']:.6f} | {test['recall_macro']:.6f} | {test['f1_macro']:.6f} | "
            f"{results['models'][name]['training_time_seconds']:.3f} |"
        )
    selected = results["selected"]
    lines += [
        "",
        f"Selected model: {selected['model']} based on validation Macro-F1.",
        f"Final selected-model test Macro-F1: {selected['test_f1_macro']:.6f}.",
    ]
    return "\n".join(lines) + "\n"


def run_dl_comparison(data_dir: Path = ROOT / "data/processed", output_dir: Path = DL_DIR,
                      seed: int = SEED, config: DLConfig = DLConfig()) -> dict:
    splits, audit_verified = load_splits(data_dir)
    train, validation, test = splits["train"], splits["validation"], splits["test"]
    results = {
        "configuration": {
            "seed": seed,
            "task": "ticket_text_to_category",
            "models": list(MODEL_NAMES),
            "vocabulary_size": config.vocabulary_size,
            "sequence_length": config.sequence_length,
            "embedding_dim": config.embedding_dim,
            "hidden_dim": config.hidden_dim,
            "dense_max_iter": config.dense_max_iter,
            "hyperparameter_optimization": False,
        },
        "environment": {"python": platform.python_version(), **{name: version(name) for name in
                        ("scikit-learn", "numpy", "joblib", "threadpoolctl")}},
        "dataset": {name: {"rows": len(split.texts), "sha256": split.sha256} for name, split in splits.items()},
        "leakage_checks": {"train_validation_test_split_unchanged": True, "audit_hashes_verified": audit_verified,
                           "priority_models_trained": False, "test_used_during_training": False},
        "models": {},
    }
    trained = {}
    with threadpool_limits(limits=1):
        for name in MODEL_NAMES:
            LOGGER.info("Training %s category model.", name.upper())
            model = build_dl_model(name, config, seed)
            started = time.perf_counter()
            model.fit(train.texts, train.categories)
            training_time = time.perf_counter() - started
            results["models"][name] = {
                "validation": _metrics(validation.categories, model.predict(validation.texts).tolist()),
                "test": _metrics(test.categories, model.predict(test.texts).tolist()),
                "training_time_seconds": float(training_time),
            }
            trained[name] = model
    selected = max(MODEL_NAMES, key=lambda name: results["models"][name]["validation"]["f1_macro"])
    results["selected"] = {
        "model": selected,
        "selection_metric": "validation_f1_macro",
        "validation_f1_macro": results["models"][selected]["validation"]["f1_macro"],
        "test_f1_macro": results["models"][selected]["test"]["f1_macro"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".dl-comparison-", dir=output_dir) as temporary:
        staging = Path(temporary)
        (staging / "models").mkdir()
        joblib.dump(trained[selected], staging / "models" / "selected_model.joblib", compress=3)
        (staging / "dl_comparison_results.json").write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "dl_comparison_results.md").write_text(_results_markdown(results), encoding="utf-8")
        (output_dir / "models").mkdir(exist_ok=True)
        (staging / "models" / "selected_model.joblib").replace(output_dir / "models" / "selected_model.joblib")
        for path in staging.iterdir():
            if path.is_file():
                path.replace(output_dir / path.name)
    for name in MODEL_NAMES:
        print(f"{name.upper()} test Macro-F1: {results['models'][name]['test']['f1_macro']:.6f}")
    print(f"Selected model: {selected}")
    print(f"Saved DL comparison artifacts: {output_dir.resolve()}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--output-dir", type=Path, default=DL_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_dl_comparison(args.data_dir, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
