import logging
import sys
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.agents.workflow import AgentConfig, RetrievalTool, SupportAgentWorkflow
from app.main import create_app
from app.ml import dl_training
from app.ml.baseline_data import MODEL_FILES
from app.schemas.ticket import ClassificationView, ExplanationView, TicketCreate
from app.services import ticket_service
from app.services.classification_service import ClassificationService


class _FailingRetriever:
    def search_similar_tickets(self, *_args, **_kwargs):
        raise RuntimeError("retrieval unavailable")


class _FailingRag:
    def generate_from_retrieved(self, *_args, **_kwargs):
        raise RuntimeError("generation unavailable")


def _agent_workflow(retriever, rag):
    return SupportAgentWorkflow(RetrievalTool(retriever), rag, AgentConfig(top_k=2))


def _processor():
    processor = ticket_service.TicketProcessor.__new__(ticket_service.TicketProcessor)
    processor.security = ticket_service.SecurityService()
    processor.classifier = SimpleNamespace(predict=lambda _text: ClassificationView(
        category="Billing and Payments", category_confidence=0.8, priority="medium", priority_confidence=0.7,
    ))
    processor._run_simple = lambda *_args: ("safe", [], "grounded", ["retrieval", "resolution"])
    processor._explain = lambda *_args: ExplanationView()
    return processor


def _assert_error_logged(caplog, message):
    assert any(record.levelno == logging.ERROR and message in record.getMessage() for record in caplog.records)


def test_agent_retrieval_failure_logs_and_preserves_fallback(caplog):
    state = _agent_workflow(_FailingRetriever(), _FailingRag()).retrieval_agent({"ticket_text": "payment", "workflow_trace": []})
    assert state["status"] == "retrieval_failed"
    _assert_error_logged(caplog, "Retrieval agent failed to fetch similar tickets")


def test_agent_resolution_failure_logs_and_preserves_fallback(caplog):
    state = _agent_workflow(_FailingRetriever(), _FailingRag()).resolution_agent({"ticket_text": "payment", "retrieved_tickets": [], "workflow_trace": []})
    assert state["status"] == "generation_failed"
    _assert_error_logged(caplog, "Resolution agent failed to generate a grounded response")


def test_distilbert_load_failure_logs_and_preserves_failure_result(tmp_path, monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *_: (_ for _ in ()).throw(RuntimeError("offline"))),
        AutoModelForSequenceClassification=SimpleNamespace(),
    ))
    result, model = dl_training._train_distilbert(None, None, None, tmp_path, 42)
    assert model is None and result["status"] == "failed"
    _assert_error_logged(caplog, "DistilBERT model loading failed")


def test_classification_artifact_failure_logs_and_preserves_exception(tmp_path, monkeypatch, caplog):
    (tmp_path / MODEL_FILES["category"]).touch()
    monkeypatch.setattr("app.services.classification_service.joblib.load", lambda _: (_ for _ in ()).throw(RuntimeError("bad artifact")))
    with pytest.raises(ValueError, match="Cannot load valid category model"):
        ClassificationService(tmp_path)
    _assert_error_logged(caplog, "Baseline category classification model validation failed")


def test_ticket_retrieval_failure_logs_and_preserves_empty_results(caplog):
    processor = _processor()
    processor.retriever = _FailingRetriever()
    response = processor.process(TicketCreate(subject="Payment", body="payment failed"))
    assert response.similar_tickets == []
    _assert_error_logged(caplog, "Similar-ticket retrieval failed")


@pytest.mark.parametrize(("method", "message", "expected"), [
    ("_run_simple", "Simple RAG resolution generation failed", "generation_failed"),
    ("_run_complex", "Complex agent workflow failed", "workflow_failed"),
])
def test_ticket_workflow_fallbacks_log(caplog, method, message, expected):
    processor = _processor()
    if method == "_run_simple":
        processor._run_simple = ticket_service.TicketProcessor._run_simple.__get__(processor)
        processor.rag_service = _FailingRag()
        result = processor._run_simple("payment", [], 3)
    else:
        processor.agent_workflow = SimpleNamespace(run=lambda _: (_ for _ in ()).throw(RuntimeError("workflow unavailable")))
        result = processor._run_complex("payment")
    assert result[2] == expected
    _assert_error_logged(caplog, message)


def test_ticket_explanation_failure_logs_and_preserves_fallback(caplog):
    processor = _processor()
    processor._explain = ticket_service.TicketProcessor._explain.__get__(processor)
    processor.explainer = SimpleNamespace(explain=lambda *_: (_ for _ in ()).throw(RuntimeError("explanation unavailable")))
    fallback = processor._explain("payment", ClassificationView(category="Billing and Payments", category_confidence=0.8))
    assert fallback.predicted_category == "Billing and Payments"
    _assert_error_logged(caplog, "Ticket explanation generation failed")


def test_receive_ticket_failure_logs_and_preserves_structured_fallback(monkeypatch, caplog):
    monkeypatch.setattr(ticket_service, "get_ticket_processor", lambda: SimpleNamespace(
        process=lambda _: (_ for _ in ()).throw(RuntimeError("component unavailable")),
    ))
    response = ticket_service.receive_ticket(TicketCreate(subject="Payment", body="payment failed"))
    assert response.status == "error"
    _assert_error_logged(caplog, "End-to-end ticket processing failed")


def test_http_middleware_rethrow_logs_failure(monkeypatch, caplog):
    monkeypatch.setattr(ticket_service, "initialize_ticket_processor", lambda: SimpleNamespace())
    app: FastAPI = create_app()

    @app.get("/logging-test-error")
    def fail_request():
        raise RuntimeError("request unavailable")

    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.get("/logging-test-error").status_code == 500
    _assert_error_logged(caplog, "HTTP request handling failed")
