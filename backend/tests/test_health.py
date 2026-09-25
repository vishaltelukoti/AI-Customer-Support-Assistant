def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["service"] == "AI Customer Support Assistant"
    assert isinstance(body["classifier_artifacts_available"], bool)
    assert isinstance(body["retrieval_index_available"], bool)


def test_allowed_cors_origin(client):
    response = client.options(
        "/api/v1/tickets",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_unlisted_cors_origin(client):
    response = client.options(
        "/api/v1/tickets",
        headers={"Origin": "http://unlisted.example", "Access-Control-Request-Method": "POST"},
    )
    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers
