import csv
import json

import numpy as np

from app.rag.retrieval import (
    RetrievalConfig,
    SimilarTicketRetriever,
    encode_texts,
    evaluate_retrieval,
    load_historical_tickets,
)


class TinyEmbedder:
    def __init__(self):
        self.calls = []

    def encode(self, sentences, **kwargs):
        self.calls.append(list(sentences))
        vectors = []
        for text in sentences:
            lower = text.lower()
            vectors.append([
                float("billing" in lower or "payment" in lower),
                float("password" in lower or "login" in lower),
                float("refund" in lower or "return" in lower),
                float("shipping" in lower or "delivery" in lower),
            ])
        array = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        return np.divide(array, norms, out=np.zeros_like(array), where=norms > 0)


def _rows():
    base = [
        ("train-1", "Billing payment failed", "Billing and Payments", "high", "Do billing answer", "train"),
        ("train-2", "Password login reset needed", "Technical Support", "medium", "Do password answer", "train"),
        ("train-3", "Refund return request", "Returns and Exchanges", "low", "Do refund answer", "train"),
        ("train-4", "Shipping delivery delayed", "Customer Service", "medium", "Do shipping answer", "train"),
        ("val-1", "Payment card billing issue", "Billing and Payments", "high", "Validation answer", "validation"),
        ("test-1", "Need login password help", "Technical Support", "medium", "Test answer", "test"),
    ]
    return [{
        "ticket_id": ticket_id,
        "ticket_text": text,
        "category": category,
        "priority": priority,
        "answer": answer,
        "source_record": str(index),
        "version": "400",
        "language": "en",
        "split": split,
        "type": "Incident",
        "tag_1": "tag",
    } for index, (ticket_id, text, category, priority, answer, split) in enumerate(base, start=1)]


def test_embedding_generation_dimensions_and_repeatability():
    embedder = TinyEmbedder()
    first = encode_texts(embedder, ["Billing payment failed", "Password login reset needed"])
    second = encode_texts(embedder, ["Billing payment failed", "Password login reset needed"])
    assert first.shape == (2, 4)
    np.testing.assert_allclose(first, second)


def test_faiss_index_search_top_k_metadata_and_answer_exclusion():
    embedder = TinyEmbedder()
    retriever = SimilarTicketRetriever.build_index(_rows(), embedder, RetrievalConfig(model_name="tiny"))
    assert retriever.index.ntotal == 4
    assert {item.split for item in retriever.metadata} == {"train"}
    assert retriever.metadata[0].index_position == 0
    assert retriever.metadata[0].ticket_id == "train-1"
    assert all("answer" not in text.lower() for call in embedder.calls for text in call)
    results = retriever.search_similar_tickets("Payment billing trouble", top_k=2)
    assert len(results) == 2
    assert results[0].ticket_id == "train-1"
    assert results[0].answer == "Do billing answer"
    assert results[0].score > 0


def test_index_save_load_and_no_self_retrieval(tmp_path):
    retriever = SimilarTicketRetriever.build_index(_rows(), TinyEmbedder(), RetrievalConfig(model_name="tiny"))
    retriever.save(tmp_path)
    loaded = SimilarTicketRetriever.load_index(tmp_path, TinyEmbedder())
    assert loaded.index.ntotal == retriever.index.ntotal
    assert loaded.metadata[1].ticket_id == "train-2"
    results = loaded.search_similar_tickets("Password login reset needed", top_k=3, exclude_ticket_id="train-2")
    assert all(result.ticket_id != "train-2" for result in results)
    assert (tmp_path / "tickets.faiss").is_file()
    assert (tmp_path / "metadata.jsonl").is_file()
    config = json.loads((tmp_path / "retrieval_config.json").read_text(encoding="utf-8"))
    assert config["answers_used_for_embeddings"] is False


def test_retrieval_evaluation_uses_held_out_queries_without_indexing_them():
    retriever = SimilarTicketRetriever.build_index(_rows(), TinyEmbedder(), RetrievalConfig(model_name="tiny"))
    evaluation = evaluate_retrieval(retriever, _rows(), limit_per_split=5)
    assert evaluation["queries_evaluated"] == 2
    assert evaluation["metrics"]["recall@1"] == 1.0
    assert evaluation["metrics"]["recall@3"] == 1.0
    assert evaluation["metrics"]["recall@5"] == 1.0
    assert evaluation["metrics"]["mrr"] == 1.0
    assert evaluation["leakage_checks"]["validation_or_test_ids_indexed"] is False
    assert evaluation["leakage_checks"]["answers_used_for_embeddings"] is False
    assert evaluation["leakage_checks"]["self_retrieval_observed"] is False


def test_load_historical_tickets_schema(tmp_path):
    path = tmp_path / "historical_tickets.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(_rows()[0]))
        writer.writeheader()
        writer.writerows(_rows())
    loaded = load_historical_tickets(path)
    assert len(loaded) == 6
    assert loaded[0]["ticket_id"] == "train-1"
