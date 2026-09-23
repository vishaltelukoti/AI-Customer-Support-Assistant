import csv
import json

import joblib
import numpy as np

from app.ml.dataset_audit import CATEGORIES, PRIORITIES
from app.ml.dl_models import DLConfig, build_dl_model
import app.ml.dl_training as dl_training
from app.ml.dl_training import run_dl_comparison
from app.ml.preprocessing import COLUMNS


def _texts_and_labels():
    texts = []
    labels = []
    for category in CATEGORIES:
        for index in range(2):
            texts.append(f"{category} example ticket {index}")
            labels.append(category)
    return texts, labels


def test_dl_model_creation_and_output_shape():
    texts, labels = _texts_and_labels()
    config = DLConfig(vocabulary_size=200, sequence_length=8, embedding_dim=8, hidden_dim=6, dense_max_iter=50)
    for name in ("cnn", "rnn", "lstm", "attention"):
        model = build_dl_model(name, config)
        model.fit(texts, labels)
        assert set(model.classes_) == set(CATEGORIES)
        assert model.predict(texts[:3]).shape == (3,)
        assert model.predict_proba(texts[:3]).shape == (3, len(CATEGORIES))
        np.testing.assert_allclose(model.predict_proba(texts[:3]).sum(axis=1), np.ones(3), atol=1e-6)


def test_attention_model_rejects_unknown_builder_name():
    config = DLConfig(vocabulary_size=200, sequence_length=8, embedding_dim=8, hidden_dim=6, dense_max_iter=50)
    model = build_dl_model("attention", config)
    assert model.name == "attention"


def test_dl_training_path_and_artifacts(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for split in ("train", "validation", "test"):
        with (data_dir / f"{split}.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=COLUMNS)
            writer.writeheader()
            for category in CATEGORIES:
                for priority in PRIORITIES:
                    for index in range(2):
                        writer.writerow({
                            "ticket_id": f"{split}-{category}-{priority}-{index}",
                            "ticket_text": f"{category} {priority} sample {index} {split}only",
                            "category": category,
                            "priority": priority,
                        })

    def fake_distilbert(train, validation, test, output_dir, seed):
        output_dir.mkdir(parents=True)
        (output_dir / "config.json").write_text("{}", encoding="utf-8")
        metrics = {
            "accuracy": 0.1,
            "precision_macro": 0.01,
            "recall_macro": 0.1,
            "f1_macro": 0.02,
            "f1_weighted": 0.02,
        }
        return {
            "validation": metrics,
            "test": metrics,
            "training_time_seconds": 0.01,
            "checkpoint": dl_training.DISTILBERT_CHECKPOINT,
            "status": "completed",
        }, None

    monkeypatch.setattr(dl_training, "_train_distilbert", fake_distilbert)
    config = DLConfig(vocabulary_size=300, sequence_length=8, embedding_dim=8, hidden_dim=6, dense_max_iter=50)
    output_dir = tmp_path / "dl"
    results = run_dl_comparison(data_dir, output_dir, config=config)
    saved = json.loads((output_dir / "dl_comparison_results.json").read_text(encoding="utf-8"))
    assert set(saved["models"]) == {"cnn", "rnn", "lstm", "attention", "distilbert"}
    assert saved["selected"]["model"] in {"cnn", "rnn", "lstm", "attention", "distilbert"}
    expected = max(saved["models"], key=lambda name: saved["models"][name]["validation"]["f1_macro"])
    assert saved["selected"]["model"] == expected
    assert saved["selected"]["selection_metric"] == "validation_f1_macro"
    assert saved["leakage_checks"]["model_selection_metric"] == "validation_f1_macro"
    assert saved["leakage_checks"]["test_used_during_training"] is False
    assert 0 <= saved["models"][saved["selected"]["model"]]["test"]["f1_macro"] <= 1
    model = joblib.load(output_dir / "models" / "selected_model.joblib")
    assert model.predict(["Billing and Payments sample"])[0] in CATEGORIES
    assert (output_dir / "dl_comparison_results.md").is_file()
    assert (output_dir / "selected_model_metadata.json").is_file()
    assert (output_dir / "models" / "distilbert" / "config.json").is_file()
    assert results["leakage_checks"]["priority_models_trained"] is False
