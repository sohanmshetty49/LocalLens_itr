from __future__ import annotations


def test_service_loads_fixture_corpus(service):
    assert service.chunks, "expected the fixture corpus to have chunks"
    assert service.places, "expected the fixture corpus to have place records"
    assert service.retriever is not None


def test_answer_is_grounded_for_a_known_city(service):
    response = service.answer("What should I do in San Francisco if I want a practical first itinerary?")
    assert response.citations or response.place_cards
    assert response.answer
    assert response.confidence_note


def test_answer_returns_place_cards_for_structured_query(service):
    response = service.answer("Best rated spot in San Francisco over 4.0?")
    assert response.place_cards


def test_guardrail_flags_unsupported_location(service):
    # "Atlantis" is not in LocalLens's city catalog at all, and "safety in
    # Atlantis" matches the explicit-location regex, so this should hit the
    # dedicated unsupported-location guardrail rather than the generic
    # "no evidence" fallback.
    response = service.answer("What should I know about safety in Atlantis?")
    assert response.citations == []
    assert response.place_cards == []
    assert response.filters_applied.get("coverage") == "unsupported"


def test_fallback_when_no_evidence_is_grounded_but_empty(service):
    # A recognized location with a query style that yields no matches in a
    # single-city fixture (no data at all for this city) should hit the
    # "could not find enough grounded evidence" guardrail, not error out.
    response = service.answer("What should I know before moving to Seattle?")
    assert response.citations == []
    assert response.place_cards == []
    assert response.answer


def test_memory_carries_location_into_a_followup_query(service):
    session_id = "test-memory-session"
    service.answer("What should I do in San Francisco?", session_id=session_id)
    assert service.memory.last_location(session_id) == "San Francisco"

    # The follow-up doesn't mention any city, so the agent graph should pull
    # "San Francisco" from session memory before classifying intent.
    result = service._graph.invoke(
        {"query": "What about good coffee?", "session_id": session_id, "location": "", "topic": ""}
    )
    assert result["intent"].location == "San Francisco"


def test_memory_is_isolated_per_session(service):
    service.answer("What should I do in San Francisco?", session_id="session-a")
    assert service.memory.last_location("session-b") == ""
