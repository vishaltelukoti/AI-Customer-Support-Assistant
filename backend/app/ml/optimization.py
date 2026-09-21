"""Day 3 hyperparameter optimization for TF-IDF plus Logistic Regression."""

import argparse
import hashlib
import json
import logging
import platform
import warnings
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import optuna
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score
from threadpoolctl import threadpool_limits

from .baseline_data import load_splits
from .baseline_models import build_pipeline
from .evaluation import METRICS, evaluate_classifier, save_confusion_matrix
from .preprocessing import ROOT, SEED

LOGGER = logging.getLogger(__name__)
OPTIMIZATION_DIR = ROOT / "experiments/optimization"
OPTIMIZED_MODEL_FILES = {
    "category": "category_optimized.joblib",
    "priority": "priority_optimized.joblib",
}
GRID_PARAMS = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__min_df": [2],
    "tfidf__max_df": [0.95],
    "tfidf__sublinear_tf": [True],
    "tfidf__max_features": [20000],
    "classifier__C": [1.0],
    "classifier__class_weight": [None, "balanced"],
}
RANDOM_PARAMS = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__min_df": [1, 2, 5],
    "tfidf__max_df": [0.95, 1.0],
    "tfidf__sublinear_tf": [True, False],
    "tfidf__max_features": [20000, 50000],
    "classifier__C": [0.1, 1.0, 10.0],
    "classifier__class_weight": [None, "balanced"],
}


def _jsonable(value):
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _validation_metrics(model, split, target: str) -> dict:
    labels = split.categories if target == "category" else split.priorities
    result = evaluate_classifier(model, split.texts, labels)
    return {metric: result[metric] for metric in METRICS}


def _search_result(search, validation, target: str) -> dict:
    return {
        "best_parameters": _jsonable(search.best_params_),
        "cv_macro_f1": float(search.best_score_),
        "validation": _validation_metrics(search.best_estimator_, validation, target),
    }


def _cv(seed: int, folds: int) -> StratifiedKFold:
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)


def run_grid_search(texts: list[str], labels: list[str], validation, target: str,
                    seed: int = SEED, cv: int = 3, param_grid: dict | None = None) -> dict:
    search = GridSearchCV(build_pipeline(seed), param_grid or GRID_PARAMS, scoring="f1_macro",
                          cv=_cv(seed, cv), n_jobs=1, refit=True)
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        search.fit(texts, labels)
    return _search_result(search, validation, target)


def run_random_search(texts: list[str], labels: list[str], validation, target: str,
                      seed: int = SEED, cv: int = 3, n_iter: int = 10,
                      param_distributions: dict | None = None) -> dict:
    search = RandomizedSearchCV(build_pipeline(seed), param_distributions or RANDOM_PARAMS, n_iter=n_iter,
                                scoring="f1_macro", cv=_cv(seed, cv), random_state=seed, n_jobs=1,
                                refit=True)
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        search.fit(texts, labels)
    return _search_result(search, validation, target)


def _trial_params(trial: optuna.Trial) -> dict:
    ngram = trial.suggest_categorical("tfidf__ngram_range", ["1-1", "1-2"])
    return {
        "tfidf__ngram_range": (1, 1) if ngram == "1-1" else (1, 2),
        "tfidf__min_df": trial.suggest_categorical("tfidf__min_df", [1, 2, 5]),
        "tfidf__max_df": trial.suggest_categorical("tfidf__max_df", [0.95, 1.0]),
        "tfidf__sublinear_tf": trial.suggest_categorical("tfidf__sublinear_tf", [True, False]),
        "tfidf__max_features": trial.suggest_categorical("tfidf__max_features", [20000, 50000]),
        "classifier__C": trial.suggest_categorical("classifier__C", [0.1, 1.0, 10.0]),
        "classifier__class_weight": trial.suggest_categorical("classifier__class_weight", [None, "balanced"]),
    }


def run_bayesian_search(texts: list[str], labels: list[str], validation, target: str,
                        seed: int = SEED, cv: int = 3, n_trials: int = 10) -> dict:
    splitter = _cv(seed, cv)

    def objective(trial: optuna.Trial) -> float:
        model = build_pipeline(seed).set_params(**_trial_params(trial))
        with threadpool_limits(limits=1), warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            scores = cross_val_score(model, texts, labels, scoring="f1_macro", cv=splitter, n_jobs=1)
        return float(scores.mean())

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = _trial_params(study.best_trial)
    model = build_pipeline(seed).set_params(**best_params)
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(texts, labels)
    return {
        "best_parameters": _jsonable(best_params),
        "objective_macro_f1": float(study.best_value),
        "validation": _validation_metrics(model, validation, target),
    }


def _select_method(results: dict) -> tuple[str, dict]:
    candidates = {name: result["validation"]["f1_macro"] for name, result in results.items()}
    best_name = max(candidates, key=lambda name: (candidates[name], name == "grid"))
    return best_name, deepcopy(results[best_name])


def _fit_selected(texts: list[str], labels: list[str], parameters: dict, seed: int):
    params = deepcopy(parameters)
    if isinstance(params.get("tfidf__ngram_range"), list):
        params["tfidf__ngram_range"] = tuple(params["tfidf__ngram_range"])
    model = build_pipeline(seed).set_params(**params)
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(texts, labels)
    return model


def optimization_markdown(results: dict) -> str:
    lines = ["# Day 3 optimization results", "", "Scores are fractions (0-1). Test is evaluated once after selection.", ""]
    for target in ("category", "priority"):
        lines += [f"## {target.capitalize()}", "", "| Method | CV/Objective Macro-F1 | Validation Macro-F1 |",
                  "| --- | ---: | ---: |"]
        for method in ("grid", "random", "bayesian"):
            score_key = "objective_macro_f1" if method == "bayesian" else "cv_macro_f1"
            result = results[target][method]
            lines.append(f"| {method.capitalize()} | {result[score_key]:.6f} | {result['validation']['f1_macro']:.6f} |")
        selected = results[target]["selected"]
        lines += ["", f"Selected method: {selected['method']}",
                  f"Selected parameters: `{json.dumps(selected['parameters'], sort_keys=True)}`", "",
                  "| Test metric | Value |", "| --- | ---: |"]
        for metric in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"):
            lines.append(f"| {metric} | {results[target]['test'][metric]:.6f} |")
        lines.append("")
    return "\n".join(lines)


def run_optimization(data_dir: Path = ROOT / "data/processed", output_dir: Path = OPTIMIZATION_DIR,
                     seed: int = SEED, cv: int = 3, random_iter: int = 10,
                     bayesian_trials: int = 10, grid_params: dict | None = None,
                     random_params: dict | None = None) -> dict:
    splits, audit_verified = load_splits(data_dir)
    train, validation, test = splits["train"], splits["validation"], splits["test"]
    LOGGER.info("Loaded train=%d, validation=%d, test=%d; searches use train only.",
                len(train.texts), len(validation.texts), len(test.texts))
    results = {
        "configuration": {"seed": seed, "cv": cv, "random_iterations": random_iter,
                          "bayesian_trials": bayesian_trials, "scoring": "f1_macro",
                          "selection_metric": "validation_f1_macro", "test_used_for_search": False,
                          "grid_parameters": _jsonable(grid_params or GRID_PARAMS),
                          "random_parameters": _jsonable(random_params or RANDOM_PARAMS)},
        "environment": {"python": platform.python_version(), **{name: version(name) for name in
                        ("scikit-learn", "numpy", "scipy", "joblib", "optuna", "threadpoolctl")}},
        "dataset": {name: {"rows": len(split.texts), "sha256": split.sha256} for name, split in splits.items()},
        "leakage_checks": {"search_uses_train_only": True, "validation_used_for_selection": True,
                           "test_used_for_selection": False, "audit_hashes_verified": audit_verified,
                           "answer_loaded": False},
    }
    models = {}
    for target in ("category", "priority"):
        labels = train.categories if target == "category" else train.priorities
        LOGGER.info("Running %s grid search.", target)
        target_results = {"grid": run_grid_search(train.texts, labels, validation, target, seed, cv, grid_params)}
        LOGGER.info("Running %s randomized search.", target)
        target_results["random"] = run_random_search(train.texts, labels, validation, target, seed, cv,
                                                     random_iter, random_params)
        LOGGER.info("Running %s Bayesian optimization.", target)
        target_results["bayesian"] = run_bayesian_search(train.texts, labels, validation, target, seed, cv,
                                                         bayesian_trials)
        method, selected = _select_method(target_results)
        selected_labels = (train.categories + validation.categories if target == "category"
                           else train.priorities + validation.priorities)
        model = _fit_selected(train.texts + validation.texts, selected_labels, selected["best_parameters"], seed)
        test_labels = test.categories if target == "category" else test.priorities
        target_results["selected"] = {"method": method, "parameters": selected["best_parameters"],
                                      "validation_f1_macro": selected["validation"]["f1_macro"]}
        target_results["test"] = evaluate_classifier(model, test.texts, test_labels)
        results[target] = target_results
        models[target] = model
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".optimization-", dir=output_dir) as temporary:
        staging = Path(temporary)
        (staging / "models").mkdir()
        for target, model in models.items():
            joblib.dump(model, staging / "models" / OPTIMIZED_MODEL_FILES[target], compress=3)
            save_confusion_matrix(results[target]["test"], f"{target}_optimized", staging)
        results["model_sha256"] = {
            filename: hashlib.sha256((staging / "models" / filename).read_bytes()).hexdigest()
            for filename in OPTIMIZED_MODEL_FILES.values()
        }
        (staging / "optimization_results.json").write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "optimization_results.md").write_text(optimization_markdown(results) + "\n", encoding="utf-8")
        (output_dir / "models").mkdir(exist_ok=True)
        for filename in OPTIMIZED_MODEL_FILES.values():
            (staging / "models" / filename).replace(output_dir / "models" / filename)
        for path in staging.iterdir():
            if path.is_file():
                path.replace(output_dir / path.name)
    for target in ("category", "priority"):
        print(f"{target.capitalize()} selected {results[target]['selected']['method']} "
              f"validation Macro-F1={results[target]['selected']['validation_f1_macro']:.6f} "
              f"test Macro-F1={results[target]['test']['f1_macro']:.6f}")
    print(f"Saved optimization artifacts: {output_dir.resolve()}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--output-dir", type=Path, default=OPTIMIZATION_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--random-iter", type=int, default=10)
    parser.add_argument("--bayesian-trials", type=int, default=10)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_optimization(args.data_dir, args.output_dir, args.seed, args.cv, args.random_iter, args.bayesian_trials)


if __name__ == "__main__":
    main()
