from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from locallens.generation import compose_answer
from locallens.schemas import AnswerPayload, PlaceCandidate, QueryIntent, SearchResult

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance only
    from locallens.service import LocalLensService

logger = logging.getLogger("locallens.agent")


class AgentState(TypedDict, total=False):
    """Shared state threaded through the LocalLens agent graph.

    Every node reads from and writes a partial update to this state. Keys
    not returned by a node are left untouched, so state accumulates as the
    graph progresses from raw query -> intent -> evidence -> answer.
    """

    query: str
    session_id: str
    location: str
    topic: str
    intent: QueryIntent
    unsupported_location: str
    retrieval_topic: str
    retrieval_query: str
    retrieval_filters: dict[str, str]
    search_top_k: int
    candidate_k: int
    retrieved: list[SearchResult]
    place_candidates: list[PlaceCandidate]
    gallery: list[dict[str, str]]
    answer: AnswerPayload


def build_agent_graph(service: "LocalLensService"):
    """Compile a LangGraph `StateGraph` that orchestrates LocalLens's answer
    pipeline as explicit, named steps instead of one inline procedure:

        apply_memory -> classify_intent -> [unsupported location? compose early]
            -> search_places -> retrieve_evidence
            -> decide_followup (a real tool-call-like decision: broaden the
               search when the first retrieval attempt came back empty)
            -> quality_gates -> compose

    The heavy-lifting business logic (intent inference, filtering, ranking)
    still lives in `LocalLensService`'s existing, well-tested private
    methods; this graph is the orchestration layer that decides *which*
    step runs next and carries state (including cross-turn memory) between
    them, which is what turns a single hard-coded call chain into an
    inspectable multi-step agent.
    """

    def classify_intent(state: AgentState) -> dict[str, Any]:
        intent = service._infer_intent(
            state["query"], location=state.get("location", ""), topic=state.get("topic", "")
        )
        unsupported_location = service._unsupported_requested_location(
            state["query"], intent.location, explicit_location=state.get("location", "")
        )
        return {"intent": intent, "unsupported_location": unsupported_location}

    def apply_memory(state: AgentState) -> dict[str, Any]:
        # Runs *after* classify_intent so it only fills in a location the
        # current turn genuinely didn't specify. If the query named a place
        # LocalLens doesn't recognize at all (unsupported_location is set),
        # that is a distinct guardrail and must NOT be silently overwritten
        # by a remembered city from an earlier turn.
        session_id = state.get("session_id") or "default"
        intent = state["intent"]
        if not intent.location and not state.get("unsupported_location"):
            remembered_location = service.memory.last_location(session_id)
            if remembered_location:
                logger.debug(
                    "agent.memory: carrying location=%s into session=%s",
                    remembered_location,
                    session_id,
                )
                intent.location = remembered_location
        return {"session_id": session_id, "intent": intent}

    def route_after_memory(state: AgentState) -> str:
        return "unsupported" if state.get("unsupported_location") else "search_places"

    def search_places(state: AgentState) -> dict[str, Any]:
        intent = state["intent"]
        place_candidates = service._search_places(intent) if intent.wants_places else []
        retrieval_topic = service._retrieval_topic_for_intent(intent)
        filters = {
            "location": intent.location if intent.location and not intent.wants_distance_expansion else "",
            "topic": retrieval_topic,
        }
        retrieval_query = service._expanded_query(
            state["query"], retrieval_topic or intent.topic, intent.activity_types
        )
        search_top_k = max(
            service.settings.top_k * (2 if intent.route == "structured" else 3), service.settings.top_k
        )
        candidate_k = (
            max(service.settings.top_k * 2, min(service.settings.candidate_k, 12))
            if intent.route == "structured"
            else service.settings.candidate_k
        )
        return {
            "place_candidates": place_candidates,
            "retrieval_topic": retrieval_topic,
            "retrieval_filters": {k: v for k, v in filters.items() if v},
            "retrieval_query": retrieval_query,
            "search_top_k": search_top_k,
            "candidate_k": candidate_k,
        }

    def retrieve_evidence(state: AgentState) -> dict[str, Any]:
        if not service.retriever:
            return {"retrieved": []}
        intent = state["intent"]
        retrieved = service.retriever.search(
            state["retrieval_query"],
            top_k=state["search_top_k"],
            candidate_k=state["candidate_k"],
            filters=state["retrieval_filters"],
        )
        retrieved = service._prune_retrieved_results(intent, retrieved)
        retrieved = service._diversify_results(retrieved, limit=service.settings.top_k)
        return {"retrieved": retrieved}

    def decide_followup(state: AgentState) -> str:
        """Tool-call-like branch point: if the first retrieval attempt came
        back empty, decide whether (and how) to broaden the search instead
        of giving up immediately."""
        if state.get("retrieved") or service.retriever is None:
            return "quality_gates"
        intent = state["intent"]
        if intent.location and state.get("retrieval_topic"):
            return "broaden_relax_topic"
        if intent.location:
            return "broaden_drop_filters"
        return "quality_gates"

    def broaden_relax_topic(state: AgentState) -> dict[str, Any]:
        intent = state["intent"]
        relaxed_filters = {"location": intent.location} if not intent.wants_distance_expansion else {}
        logger.debug("agent.broaden_search: relaxing topic filter for session=%s", state.get("session_id"))
        retrieved = service.retriever.search(
            state["retrieval_query"],
            top_k=state["search_top_k"],
            candidate_k=state["candidate_k"],
            filters=relaxed_filters,
        )
        retrieved = service._prune_retrieved_results(intent, retrieved)
        retrieved = service._diversify_results(retrieved, limit=service.settings.top_k)
        retrieval_filters = state["retrieval_filters"]
        if retrieved:
            retrieval_filters = {**relaxed_filters, "topic_relaxed_from": intent.topic}
        return {"retrieved": retrieved, "retrieval_filters": retrieval_filters}

    def broaden_drop_filters(state: AgentState) -> dict[str, Any]:
        intent = state["intent"]
        filters = {"location": intent.location} if not intent.wants_distance_expansion else {}
        logger.debug("agent.broaden_search: dropping filters for session=%s", state.get("session_id"))
        retrieved = service.retriever.search(
            state["retrieval_query"], top_k=state["search_top_k"], candidate_k=state["candidate_k"], filters=filters
        )
        retrieved = service._prune_retrieved_results(intent, retrieved)
        retrieved = service._diversify_results(retrieved, limit=service.settings.top_k)
        retrieval_filters = state["retrieval_filters"]
        if retrieved:
            retrieval_filters = {"location_fallback": intent.location}
        return {"retrieved": retrieved, "retrieval_filters": retrieval_filters}

    def quality_gates(state: AgentState) -> dict[str, Any]:
        intent = state["intent"]
        retrieved = state.get("retrieved", [])
        place_candidates = state.get("place_candidates", [])
        if (
            intent.wants_local_knowledge
            and retrieved
            and not service._has_high_signal_local_evidence(intent, retrieved)
        ):
            retrieved = []
        if intent.wants_places and not place_candidates and not service._has_grounded_place_evidence(intent, retrieved):
            retrieved = []
        return {"retrieved": retrieved}

    def compose(state: AgentState) -> dict[str, Any]:
        intent = state["intent"]
        gallery = service._gallery_for(intent.location, state.get("place_candidates", []))
        answer = compose_answer(
            state["query"],
            state.get("retrieved", []),
            filters_applied=state.get("retrieval_filters", {}),
            place_candidates=state.get("place_candidates", []),
            ollama_client=service.ollama_client,
            gallery_images=gallery,
        )
        service.memory.record(state.get("session_id", "default"), state["query"], intent.location, intent.topic)
        return {"answer": answer, "gallery": gallery}

    def compose_unsupported(state: AgentState) -> dict[str, Any]:
        unsupported_location = state["unsupported_location"]
        answer = AnswerPayload(
            answer=(
                f"I do not have grounded coverage for {unsupported_location} in the current LocalLens corpus."
            ),
            why_this_recommendation=(
                "The query names a location that is outside the set of cities and parks currently indexed by "
                "the system, so returning a recommendation would risk pulling evidence from the wrong place."
            ),
            key_tips=[
                "Ask about one of the supported LocalLens destinations.",
                "If you want a nearby covered city, try naming it directly.",
                "Treat this as a corpus-coverage limit rather than a recommendation.",
            ],
            confidence_note="Low confidence because the requested location is not available in the current LocalLens corpus.",
            citations=[],
            filters_applied={"requested_location": unsupported_location, "coverage": "unsupported"},
            used_local_llm=False,
            source_summary="No sources retrieved because the requested location is outside the current corpus.",
            place_cards=[],
            gallery_images=[],
        )
        return {"answer": answer}

    graph = StateGraph(AgentState)
    graph.add_node("apply_memory", apply_memory)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("search_places", search_places)
    graph.add_node("retrieve_evidence", retrieve_evidence)
    graph.add_node("broaden_relax_topic", broaden_relax_topic)
    graph.add_node("broaden_drop_filters", broaden_drop_filters)
    graph.add_node("quality_gates", quality_gates)
    graph.add_node("compose", compose)
    graph.add_node("compose_unsupported", compose_unsupported)

    graph.add_edge(START, "classify_intent")
    graph.add_edge("classify_intent", "apply_memory")
    graph.add_conditional_edges(
        "apply_memory",
        route_after_memory,
        {"unsupported": "compose_unsupported", "search_places": "search_places"},
    )
    graph.add_edge("search_places", "retrieve_evidence")
    graph.add_conditional_edges(
        "retrieve_evidence",
        decide_followup,
        {
            "quality_gates": "quality_gates",
            "broaden_relax_topic": "broaden_relax_topic",
            "broaden_drop_filters": "broaden_drop_filters",
        },
    )
    graph.add_edge("broaden_relax_topic", "quality_gates")
    graph.add_edge("broaden_drop_filters", "quality_gates")
    graph.add_edge("quality_gates", "compose")
    graph.add_edge("compose", END)
    graph.add_edge("compose_unsupported", END)

    return graph.compile()
