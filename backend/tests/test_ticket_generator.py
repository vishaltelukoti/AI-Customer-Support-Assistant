from app.rag.rag_service import RAGConfig, _EvaluationCaseGenerator
from app.services import ticket_service


def test_local_generator_is_default(monkeypatch):
    generator = object()
    monkeypatch.delenv("USE_LOCAL_MODEL", raising=False)
    monkeypatch.setattr(ticket_service, "LocalHFGenerator", lambda *_: generator)

    assert ticket_service._create_generator(RAGConfig()) is generator


def test_template_generator_is_only_used_for_disabled_or_failed_local_model(monkeypatch, caplog):
    monkeypatch.setenv("USE_LOCAL_MODEL", "false")
    assert isinstance(ticket_service._create_generator(RAGConfig()), _EvaluationCaseGenerator)

    monkeypatch.setenv("USE_LOCAL_MODEL", "true")
    monkeypatch.setattr(ticket_service, "LocalHFGenerator", lambda *_: (_ for _ in ()).throw(RuntimeError("missing")))
    assert isinstance(ticket_service._create_generator(RAGConfig()), _EvaluationCaseGenerator)
    assert "could not be loaded" in caplog.text
