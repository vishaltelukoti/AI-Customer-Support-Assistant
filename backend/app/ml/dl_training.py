"""Train category-only DL comparison models through Day 5."""

import argparse
import json
import logging
import platform
import time
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from threadpoolctl import threadpool_limits

from .baseline_data import load_splits
from .dataset_audit import CATEGORIES
from .dl_models import DLConfig, build_dl_model
from .preprocessing import ROOT, SEED

LOGGER = logging.getLogger(__name__)
DL_DIR = ROOT / "experiments/dl_comparison"
CLASSICAL_DL_MODELS = ("cnn", "rnn", "lstm", "attention")
COMPARISON_MODELS = (*CLASSICAL_DL_MODELS, "distilbert")
DAY4_MODELS = ("cnn", "rnn", "lstm")
DISTILBERT_CHECKPOINT = "distilbert-base-uncased"
DISTILBERT_MAX_TRAIN_PER_CLASS = 60
DISTILBERT_SEQUENCE_LENGTH = 48


def _metrics(labels: list[str], predictions: list[str]) -> dict:
    report = classification_report(labels, predictions, labels=CATEGORIES, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_macro": float(report["macro avg"]["precision"]),
        "recall_macro": float(report["macro avg"]["recall"]),
        "f1_macro": float(report["macro avg"]["f1-score"]),
        "f1_weighted": float(report["weighted avg"]["f1-score"]),
    }


def _failed_metrics(reason: str) -> dict:
    return {"status": "failed", "failure_reason": reason}


def _distilbert_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def _train_distilbert(train, validation, test, output_dir: Path, seed: int) -> tuple[dict, object | None]:
    started = time.perf_counter()
    try:
        import numpy as np
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        reason = f"Missing pretrained-model dependency: {exc}"
        return {
            "validation": _failed_metrics(reason),
            "test": _failed_metrics(reason),
            "training_time_seconds": float(time.perf_counter() - started),
            "checkpoint": DISTILBERT_CHECKPOINT,
            "status": "failed",
            "failure_reason": reason,
        }, None

    torch.manual_seed(seed)
    np.random.seed(seed)
    label_to_id = {label: index for index, label in enumerate(CATEGORIES)}
    id_to_label = {index: label for label, index in label_to_id.items()}

    class TicketDataset(Dataset):
        def __init__(self, texts: list[str], labels: list[str], tokenizer, max_length: int = 64):
            self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
            self.labels = [label_to_id[label] for label in labels]

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, index: int) -> dict:
            item = {key: torch.tensor(values[index]) for key, values in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[index])
            return item

    try:
        tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_CHECKPOINT)
        model = AutoModelForSequenceClassification.from_pretrained(
            DISTILBERT_CHECKPOINT,
            num_labels=len(CATEGORIES),
            id2label=id_to_label,
            label2id=label_to_id,
        )
    except Exception as exc:  # network/cache/model errors should be recorded, not fabricated.
        reason = f"Could not load {DISTILBERT_CHECKPOINT}: {exc}"
        return {
            "validation": _failed_metrics(reason),
            "test": _failed_metrics(reason),
            "training_time_seconds": float(time.perf_counter() - started),
            "checkpoint": DISTILBERT_CHECKPOINT,
            "status": "failed",
            "failure_reason": reason,
        }, None

    for parameter in model.distilbert.parameters():
        parameter.requires_grad = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_texts, train_labels = [], []
    per_class_counts = {label: 0 for label in CATEGORIES}
    for text, label in zip(train.texts, train.categories):
        if per_class_counts[label] < DISTILBERT_MAX_TRAIN_PER_CLASS:
            train_texts.append(text)
            train_labels.append(label)
            per_class_counts[label] += 1
    train_loader = DataLoader(
        TicketDataset(train_texts, train_labels, tokenizer, DISTILBERT_SEQUENCE_LENGTH),
        batch_size=64,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=5e-4)
    model.train()
    for batch in train_loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        output = model(**batch)
        output.loss.backward()
        optimizer.step()

    def predict(texts: list[str]) -> list[str]:
        dataset = TicketDataset(texts, [CATEGORIES[0]] * len(texts), tokenizer, DISTILBERT_SEQUENCE_LENGTH)
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        predictions = []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = {key: value.to(device) for key, value in batch.items() if key != "labels"}
                logits = model(**batch).logits
                predictions.extend(logits.argmax(dim=1).cpu().tolist())
        return [id_to_label[index] for index in predictions]

    training_time = time.perf_counter() - started
    validation_predictions = predict(validation.texts)
    test_predictions = predict(test.texts)
    results = {
        "validation": _metrics(validation.categories, validation_predictions),
        "test": _metrics(test.categories, test_predictions),
        "training_time_seconds": float(training_time),
        "checkpoint": DISTILBERT_CHECKPOINT,
        "epochs": 1,
        "batch_size": 64,
        "sequence_length": DISTILBERT_SEQUENCE_LENGTH,
        "base_encoder_frozen": True,
        "train_examples_used": len(train_texts),
        "max_train_examples_per_class": DISTILBERT_MAX_TRAIN_PER_CLASS,
        "status": "completed",
        "_test_predictions": test_predictions,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return results, tokenizer


def _results_markdown(results: dict) -> str:
    lines = [
        "# Day 5 DL comparison",
        "",
        "Category-only comparison on the unchanged train/validation/test split.",
        "",
        "| Model | Validation Macro-F1 | Test Accuracy | Test Precision Macro | Test Recall Macro | Test Macro-F1 | Test Weighted-F1 | Training time seconds | Status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name in COMPARISON_MODELS:
        if name not in results["models"]:
            continue
        model_result = results["models"][name]
        if model_result.get("status") == "failed":
            lines.append(f"| {name.upper()} | n/a | n/a | n/a | n/a | n/a | n/a | {model_result['training_time_seconds']:.3f} | failed |")
            continue
        validation = results["models"][name]["validation"]
        test = results["models"][name]["test"]
        lines.append(
            f"| {name.upper()} | {validation['f1_macro']:.6f} | {test['accuracy']:.6f} | "
            f"{test['precision_macro']:.6f} | {test['recall_macro']:.6f} | {test['f1_macro']:.6f} | "
            f"{test['f1_weighted']:.6f} | {results['models'][name]['training_time_seconds']:.3f} | completed |"
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
            "models": list(COMPARISON_MODELS),
            "vocabulary_size": config.vocabulary_size,
            "sequence_length": config.sequence_length,
            "embedding_dim": config.embedding_dim,
            "hidden_dim": config.hidden_dim,
            "dense_max_iter": config.dense_max_iter,
            "distilbert_checkpoint": DISTILBERT_CHECKPOINT,
            "distilbert_sequence_length": DISTILBERT_SEQUENCE_LENGTH,
            "distilbert_max_train_examples_per_class": DISTILBERT_MAX_TRAIN_PER_CLASS,
            "hyperparameter_optimization": False,
        },
        "environment": {"python": platform.python_version(), **{name: version(name) for name in
                        ("scikit-learn", "numpy", "joblib", "threadpoolctl")
                        }, "torch_transformers_available": _distilbert_available()},
        "dataset": {name: {"rows": len(split.texts), "sha256": split.sha256} for name, split in splits.items()},
        "leakage_checks": {"train_validation_test_split_unchanged": True, "audit_hashes_verified": audit_verified,
                           "priority_models_trained": False, "test_used_during_training": False,
                           "model_selection_metric": "validation_f1_macro"},
        "models": {},
    }
    trained = {}
    with threadpool_limits(limits=1):
        for name in CLASSICAL_DL_MODELS:
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
    distilbert_dir = output_dir / "models" / "distilbert"
    distilbert_result, _ = _train_distilbert(train, validation, test, distilbert_dir, seed)
    results["models"]["distilbert"] = distilbert_result
    selectable = [name for name, model_result in results["models"].items() if model_result.get("status") != "failed"]
    selected = max(selectable, key=lambda name: results["models"][name]["validation"]["f1_macro"])
    results["selected"] = {
        "model": selected,
        "selection_metric": "validation_f1_macro",
        "validation_f1_macro": results["models"][selected]["validation"]["f1_macro"],
        "test_f1_macro": results["models"][selected]["test"]["f1_macro"],
    }
    selected_predictions = trained[selected].predict(test.texts).tolist() if selected in trained else []
    if selected == "distilbert":
        selected_predictions = results["models"]["distilbert"].pop("_test_predictions", [])
    if selected_predictions:
        results["selected"]["test_confusion_matrix"] = confusion_matrix(
            test.categories, selected_predictions, labels=CATEGORIES
        ).tolist()
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".dl-comparison-", dir=output_dir) as temporary:
        staging = Path(temporary)
        (staging / "models").mkdir()
        if selected in trained:
            joblib.dump(trained[selected], staging / "models" / "selected_model.joblib", compress=3)
            (staging / "selected_model_metadata.json").write_text(json.dumps({
                "model": selected,
                "artifact_type": "joblib",
                "labels": CATEGORIES,
                "selection_metric": "validation_f1_macro",
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        else:
            (staging / "selected_model_metadata.json").write_text(json.dumps({
                "model": selected,
                "artifact_type": "huggingface_transformers",
                "artifact_path": "models/distilbert",
                "labels": CATEGORIES,
                "selection_metric": "validation_f1_macro",
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "dl_comparison_results.json").write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "dl_comparison_results.md").write_text(_results_markdown(results), encoding="utf-8")
        (output_dir / "models").mkdir(exist_ok=True)
        if (staging / "models" / "selected_model.joblib").exists():
            (staging / "models" / "selected_model.joblib").replace(output_dir / "models" / "selected_model.joblib")
        for path in staging.iterdir():
            if path.is_file():
                path.replace(output_dir / path.name)
    for name in COMPARISON_MODELS:
        model_result = results["models"].get(name, {})
        if model_result.get("status") == "failed":
            print(f"{name.upper()} failed: {model_result['failure_reason']}")
        else:
            print(f"{name.upper()} test Macro-F1: {model_result['test']['f1_macro']:.6f}")
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
