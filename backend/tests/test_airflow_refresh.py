import csv
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DAG_PATH = ROOT / "airflow/dags/ticket_processing.py"


class TinyEmbedder:
    def encode(self, sentences, **_kwargs):
        vectors = []
        for text in sentences:
            lower = text.lower()
            vectors.append([float("payment" in lower), float("password" in lower)])
        return np.asarray(vectors, dtype=np.float32)


def _dag_module():
    spec = importlib.util.spec_from_file_location("ticket_refresh_dag", DAG_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows():
    values = [
        ("train-payment", "payment failed", "Billing and Payments", "high", "train"),
        ("train-login", "password reset", "Technical Support", "medium", "train"),
        ("validation-payment", "payment validation", "Billing and Payments", "high", "validation"),
        ("test-login", "password test", "Technical Support", "medium", "test"),
    ]
    return [{
        "ticket_id": ticket_id,
        "ticket_text": text,
        "category": category,
        "priority": priority,
        "answer": "historical answer",
        "source_record": ticket_id,
        "version": "400",
        "language": "en",
        "split": split,
        "type": "Incident",
        "tag_1": "tag",
    } for ticket_id, text, category, priority, split in values]


def _write_processed_data(data_dir):
    data_dir.mkdir()
    for name in ("train.csv", "validation.csv", "test.csv"):
        (data_dir / name).write_text("ticket_id,ticket_text,category,priority\n", encoding="utf-8")
    rows = _rows()
    with (data_dir / "historical_tickets.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_airflow_refresh_rebuilds_training_only_faiss_artifacts(tmp_path):
    module = _dag_module()
    data_dir = tmp_path / "processed"
    artifact_dir = tmp_path / "retrieval"
    _write_processed_data(data_dir)

    assert module.validate_data(data_dir) == "processed data artifacts present"
    assert "2 tickets" in module.prepare_ticket_data(data_dir)
    assert "rebuilt 2" in module.refresh_embeddings(data_dir, artifact_dir, TinyEmbedder())
    assert "validated 2" in module.validate_retrieval_index(data_dir, artifact_dir)

    metadata = [json.loads(line) for line in (artifact_dir / "metadata.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {item["ticket_id"] for item in metadata} == {"train-payment", "train-login"}
    assert all(item["split"] == "train" for item in metadata)

    assert "rebuilt 2" in module.refresh_embeddings(data_dir, artifact_dir, TinyEmbedder())
    rebuilt = [json.loads(line) for line in (artifact_dir / "metadata.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rebuilt) == 2
