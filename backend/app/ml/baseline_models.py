"""Train and evaluate two fixed Day 2 baselines; no hyperparameter search."""

import argparse
import hashlib
import json
import logging
import platform
import warnings
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from .baseline_data import BASELINE_DIR, MODEL_FILES, load_splits
from .evaluation import evaluate_classifier, metrics_markdown, save_confusion_matrix
from .preprocessing import ROOT, SEED

LOGGER = logging.getLogger(__name__)
TFIDF_SETTINGS = dict(lowercase=True, strip_accents="unicode", ngram_range=(1, 2),
                      min_df=2, max_df=.95, sublinear_tf=True, max_features=50000)
LOGREG_SETTINGS = dict(C=1.0, l1_ratio=0.0, solver="lbfgs", tol=1e-4,
                      max_iter=1000, class_weight=None, fit_intercept=True)


def build_pipeline(seed: int = SEED) -> Pipeline:
    """Independent sparse TF-IDF plus unweighted, L2 logistic regression."""
    return Pipeline([("tfidf", TfidfVectorizer(**TFIDF_SETTINGS)),
                     ("classifier", LogisticRegression(random_state=seed, **LOGREG_SETTINGS))])


def _train(texts: list[str], labels: list[str], seed: int) -> Pipeline:
    if not texts or len(texts) != len(labels) or len(set(labels)) < 2:
        raise ValueError("Training needs aligned nonempty text/labels and at least two classes.")
    model = build_pipeline(seed)
    # Prevent a silently unconverged baseline and avoid BLAS oversubscription.
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(texts, labels)
    return model


def train_category_classifier(texts: list[str], categories: list[str], seed: int = SEED) -> Pipeline:
    """Fit the category pipeline using training text/category labels only."""
    return _train(texts, categories, seed)


def train_priority_classifier(texts: list[str], priorities: list[str], seed: int = SEED) -> Pipeline:
    """Fit the priority pipeline using training text/priority labels only."""
    return _train(texts, priorities, seed)


def run_baseline(data_dir: Path = ROOT / "data/processed", output_dir: Path = BASELINE_DIR,
                 seed: int = SEED) -> dict:
    """Train once, evaluate fixed splits, then stage and save complete artifacts."""
    splits, audit_verified = load_splits(data_dir)
    train = splits["train"]
    LOGGER.info("Loaded train=%d, validation=%d, test=%d; only train is fitted.",
                len(train.texts), len(splits['validation'].texts), len(splits['test'].texts))
    models = {}
    LOGGER.info("Training category baseline (fixed settings, class_weight=None).")
    models["category"] = train_category_classifier(train.texts, train.categories, seed)
    LOGGER.info("Training priority baseline (fixed settings, class_weight=None).")
    models["priority"] = train_priority_classifier(train.texts, train.priorities, seed)
    results = {
        "configuration": {"seed": seed, "tfidf": TFIDF_SETTINGS, "logistic_regression": LOGREG_SETTINGS,
                          "numeric_threads": 1, "feature": "ticket_text", "hyperparameter_search": False,
                          "class_weight": None, "resampling": False},
        "environment": {"python": platform.python_version(), **{name: version(name) for name in
                        ("scikit-learn", "numpy", "scipy", "joblib", "matplotlib", "threadpoolctl")}},
        "dataset": {name: {"rows": len(split.texts), "sha256": split.sha256,
                           "categories": dict(sorted(Counter(split.categories).items())),
                           "priorities": dict(sorted(Counter(split.priorities).items()))}
                    for name, split in splits.items()},
        "leakage_checks": {"only_train_fitted": True, "audit_hashes_verified": audit_verified,
                           "duplicate_ids_or_text_across_splits": 0, "answer_loaded": False,
                           "test_used_for_selection": False},
    }
    with threadpool_limits(limits=1):
        for target, model in models.items():
            results[target] = {"model": "TF-IDF + Logistic Regression",
                               "vocabulary_size": len(model.named_steps['tfidf'].vocabulary_),
                               "iterations": model.named_steps['classifier'].n_iter_.tolist()}
            # Both models and configurations are fixed before any test evaluation.
            for name in ("train", "validation", "test"):
                split = splits[name]
                labels = split.categories if target == "category" else split.priorities
                results[target][name] = evaluate_classifier(model, split.texts, labels)
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".baseline-", dir=output_dir) as temporary:
        staging = Path(temporary)
        (staging / "models").mkdir()
        for target, model in models.items():
            joblib.dump(model, staging / "models" / MODEL_FILES[target], compress=3)
            save_confusion_matrix(results[target]["test"], target, staging)
        results["model_sha256"] = {filename: hashlib.sha256((staging / "models" / filename).read_bytes()).hexdigest()
                                   for filename in MODEL_FILES.values()}
        (staging / "baseline_results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "baseline_results.md").write_text(metrics_markdown(results), encoding="utf-8")
        (output_dir / "models").mkdir(exist_ok=True)
        for filename in MODEL_FILES.values():
            (staging / "models" / filename).replace(output_dir / "models" / filename)
        for path in staging.iterdir():
            if path.is_file():
                path.replace(output_dir / path.name)
    for target in models:
        print(f"{target.capitalize()}:\n  Validation Macro-F1: {results[target]['validation']['f1_macro']:.6f}"
              f"\n  Test Macro-F1: {results[target]['test']['f1_macro']:.6f}")
    print(f"Saved models and reports: {output_dir.resolve()}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--output-dir", type=Path, default=BASELINE_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        run_baseline(args.data_dir, args.output_dir, args.seed)
    except (ValueError, OSError, ConvergenceWarning) as exc:
        parser.exit(1, f"Baseline failed: {exc}\n")


if __name__ == "__main__":
    main()
