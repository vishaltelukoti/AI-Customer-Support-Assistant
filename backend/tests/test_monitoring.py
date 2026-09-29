import pytest

from app.monitoring.metrics import monitor
from app.services import ticket_service
from tests.test_tickets import FakeProcessor


class CountingProcessor(FakeProcessor):
    def process(self, ticket):
        monitor.record_model_prediction()
        monitor.record_retrieval(0.002)
        return super().process(ticket)


@pytest.fixture(autouse=True)
def reset_monitor():
    monitor.reset()
    yield
    monitor.reset()


def test_monitoring_endpoint_reports_runtime_and_artifact_metrics(client):
    response = client.get("/monitoring")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "request_count",
        "error_count",
        "average_latency_ms",
        "model_prediction_count",
        "retrieval_request_count",
        "average_retrieval_latency_ms",
        "model_quality",
        "retrieval_quality",
    }
    assert body["model_quality"] == {"metric": "macro_f1", "value": 0.6415383542430868}
    assert body["retrieval_quality"] == {"metric": "recall_at_3", "value": 0.764}


def test_ticket_request_updates_monitoring_counters(client, monkeypatch):
    monkeypatch.setattr(ticket_service, "_processor", CountingProcessor())
    ticket_response = client.post("/api/v1/tickets", json={
        "subject": "Payment failed",
        "body": "My payment failed during checkout.",
    })
    assert ticket_response.status_code == 200

    body = client.get("/monitoring").json()
    assert body["request_count"] >= 1
    assert body["error_count"] == 0
    assert body["average_latency_ms"] >= 0.0
    assert body["model_prediction_count"] == 1
    assert body["retrieval_request_count"] == 1
    assert body["average_retrieval_latency_ms"] >= 2.0


def test_invalid_ticket_increments_monitoring_error_count(client):
    response = client.post("/api/v1/tickets", json={})
    assert response.status_code == 422
    assert client.get("/monitoring").json()["error_count"] == 1


def test_health_endpoint_remains_available(client):
    body = client.get("/health").json()
    assert body["status"] == "healthy"
    assert set(body) >= {"classifier_artifacts_available", "retrieval_index_available"}
