import csv
import json

import joblib
import pytest
from sklearn.pipeline import Pipeline

from app.ml.dataset_audit import CATEGORIES, PRIORITIES
from app.ml.optimization import OPTIMIZED_MODEL_FILES, run_optimization
from app.ml.preprocessing import COLUMNS


@pytest.fixture
def optimization_data_dir(tmp_path):
    directory = tmp_path / "data"
    directory.mkdir()
    for split in ("train", "validation", "test"):
        with (directory / f"{split}.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=COLUMNS)
            writer.writeheader()
            for category in CATEGORIES:
                for priority in PRIORITIES:
                    for i in range(3):
                        writer.writerow({
                            "ticket_id": f"{split}-{category}-{priority}-{i}",
                            "ticket_text": f"{category} {priority} ticket {i} {split}onlytoken",
                            "category": category,
                            "priority": priority,
                        })
    return directory


def tiny_grid():
    return {
        "tfidf__ngram_range": [(1, 1)],
        "tfidf__min_df": [1],
        "tfidf__max_df": [1.0],
        "tfidf__sublinear_tf": [True],
        "tfidf__max_features": [20000],
        "classifier__C": [1.0],
        "classifier__class_weight": [None],
    }


def test_all_search_methods_save_models_and_test_results(optimization_data_dir, tmp_path):
    results = run_optimization(optimization_data_dir, tmp_path / "optimization", cv=2,
                               random_iter=1, bayesian_trials=1,
                               grid_params=tiny_grid(), random_params=tiny_grid())
    saved = json.loads((tmp_path / "optimization" / "optimization_results.json").read_text(encoding="utf-8"))
    assert set(saved["category"]) >= {"grid", "random", "bayesian", "selected", "test"}
    assert set(saved["priority"]) >= {"grid", "random", "bayesian", "selected", "test"}
    for target, labels in (("category", CATEGORIES), ("priority", PRIORITIES)):
        for method in ("grid", "random", "bayesian"):
            assert saved[target][method]["best_parameters"]
            assert 0 <= saved[target][method]["validation"]["f1_macro"] <= 1
        assert saved[target]["selected"]["method"] in {"grid", "random", "bayesian"}
        assert 0 <= saved[target]["test"]["f1_macro"] <= 1
        model = joblib.load(tmp_path / "optimization" / "models" / OPTIMIZED_MODEL_FILES[target])
        assert model.predict([f"{target} validationonlytoken testonlytoken"])[0] in labels
    assert results["leakage_checks"]["test_used_for_selection"] is False
    assert (tmp_path / "optimization" / "optimization_results.md").is_file()


def test_search_does_not_fit_on_test_rows(optimization_data_dir, tmp_path, monkeypatch):
    fitted_text_batches = []
    original_fit = Pipeline.fit

    def recording_fit(self, texts, labels=None, *args, **kwargs):
        fitted_text_batches.append(list(texts))
        return original_fit(self, texts, labels, *args, **kwargs)

    monkeypatch.setattr(Pipeline, "fit", recording_fit)
    run_optimization(optimization_data_dir, tmp_path / "optimization", cv=2,
                     random_iter=1, bayesian_trials=1,
                     grid_params=tiny_grid(), random_params=tiny_grid())
    search_fits = fitted_text_batches[:12]
    assert search_fits
    assert all("testonlytoken" not in " ".join(batch) for batch in search_fits)
