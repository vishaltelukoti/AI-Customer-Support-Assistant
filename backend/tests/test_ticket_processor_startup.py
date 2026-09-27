import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.services import ticket_service


def test_processor_accessor_requires_completed_startup(monkeypatch):
    monkeypatch.setattr(ticket_service, "_processor", None)

    with pytest.raises(RuntimeError, match="Ticket processor not initialized -- app startup did not complete"):
        ticket_service.get_ticket_processor()


def test_lifespan_initializes_and_stores_processor_once(monkeypatch):
    created = []

    class StubProcessor:
        def __init__(self):
            created.append(self)

    monkeypatch.setattr(ticket_service, "_processor", None)
    monkeypatch.setattr(ticket_service, "TicketProcessor", StubProcessor)
    application = create_app()

    with TestClient(application):
        assert application.state.ticket_processor is created[0]
        assert ticket_service.get_ticket_processor() is created[0]
        assert len(created) == 1
