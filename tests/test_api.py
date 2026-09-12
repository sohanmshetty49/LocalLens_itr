from __future__ import annotations

from fastapi.testclient import TestClient

from locallens import api


def test_health_endpoint(service, monkeypatch):
    monkeypatch.setattr(api, "_service", service)
    client = TestClient(api.app)

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["chunks_indexed"] > 0
    assert body["places_indexed"] > 0


def test_answer_endpoint_returns_grounded_payload(service, monkeypatch):
    monkeypatch.setattr(api, "_service", service)
    client = TestClient(api.app)

    response = client.post(
        "/answer", json={"query": "What should I do in San Francisco if I want a practical first itinerary?"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert body["citations"] or body["place_cards"]


def test_answer_endpoint_carries_session_memory(service, monkeypatch):
    monkeypatch.setattr(api, "_service", service)
    client = TestClient(api.app)

    client.post(
        "/answer",
        json={"query": "What should I do in San Francisco?", "session_id": "api-test-session"},
    )
    response = client.post(
        "/answer",
        json={"query": "What about good coffee?", "session_id": "api-test-session"},
    )

    assert response.status_code == 200
    assert response.json()["filters_applied"].get("location") == "San Francisco"


def test_answer_endpoint_rejects_empty_query(service, monkeypatch):
    monkeypatch.setattr(api, "_service", service)
    client = TestClient(api.app)

    response = client.post("/answer", json={"query": "   "})

    assert response.status_code == 400
