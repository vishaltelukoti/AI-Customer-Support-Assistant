from uuid import UUID

import pytest

from app.schemas.ticket import TicketResponse


def test_valid_ticket_schema_and_placeholder(client):
    text = "I was charged twice for the same order."
    response = client.post("/api/v1/tickets", json={"ticket_text": text})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"ticket_id", "ticket_text", "category", "priority", "confidence", "status"}
    assert UUID(body["ticket_id"]).version == 4
    assert body == {
        "ticket_id": body["ticket_id"], "ticket_text": text,
        "category": None, "priority": None, "confidence": None, "status": "received",
    }
    TicketResponse.model_validate(body)


@pytest.mark.parametrize("payload", [
    {}, {"ticket_text": ""}, {"ticket_text": " \n\t "}, {"ticket_text": None},
    {"ticket_text": 123}, {"ticket_text": []}, {"ticket_text": "x" * 10001},
    {"ticket_text": "Help", "priority": "High"},
])
def test_invalid_ticket_rejected(client, payload):
    assert client.post("/api/v1/tickets", json=payload).status_code == 422


def test_trim_and_unique_ids(client):
    first = client.post("/api/v1/tickets", json={"ticket_text": "  Help me sign in.  "}).json()
    second = client.post("/api/v1/tickets", json={"ticket_text": "Help me sign in."}).json()
    assert first["ticket_text"] == "Help me sign in."
    assert first["ticket_id"] != second["ticket_id"]


def test_maximum_length_accepted(client):
    assert client.post("/api/v1/tickets", json={"ticket_text": "x" * 10000}).status_code == 200


def test_malformed_json_rejected(client):
    response = client.post("/api/v1/tickets", content="{", headers={"Content-Type": "application/json"})
    assert response.status_code == 422
