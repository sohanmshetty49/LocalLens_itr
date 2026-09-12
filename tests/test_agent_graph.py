from __future__ import annotations


def test_graph_has_the_expected_orchestration_nodes(service):
    node_names = set(service._graph.get_graph().nodes)
    assert {
        "apply_memory",
        "classify_intent",
        "search_places",
        "retrieve_evidence",
        "broaden_relax_topic",
        "broaden_drop_filters",
        "quality_gates",
        "compose",
        "compose_unsupported",
    }.issubset(node_names)


def test_unsupported_location_short_circuits_before_retrieval(service):
    result = service._graph.invoke(
        {"query": "What should I know about safety in Atlantis?", "session_id": "", "location": "", "topic": ""}
    )
    # search_places/retrieve_evidence never ran for this branch.
    assert "retrieved" not in result
    assert "place_candidates" not in result
    assert result["answer"].filters_applied.get("coverage") == "unsupported"


def test_grounded_query_runs_the_full_chain(service):
    result = service._graph.invoke(
        {
            "query": "What should I do in San Francisco if I want a practical first itinerary?",
            "session_id": "",
            "location": "",
            "topic": "",
        }
    )
    assert result["intent"].location == "San Francisco"
    assert "retrieved" in result
    assert "place_candidates" in result
    assert result["answer"].citations or result["answer"].place_cards
