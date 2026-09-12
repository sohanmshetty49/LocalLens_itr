"""FastAPI layer for LocalLens.

Run locally with:

    PYTHONPATH=src uvicorn locallens.api:app --reload

This exposes the same `LocalLensService` used by the Streamlit UI as a small
REST API, so LocalLens can be called as a service (not just a UI) -- the
"build/test/deploy" slice of the SDLC that the Streamlit app alone doesn't
cover.
"""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from locallens.service import LocalLensService

app = FastAPI(
    title="LocalLens API",
    version="0.1.0",
    description="Local-first RAG assistant for travelers and recent movers.",
)

# Lazily constructed singleton: loading the corpus + embedding model is not
# free, so we do it once per process rather than per-request.
_service: LocalLensService | None = None


def get_service() -> LocalLensService:
    global _service
    if _service is None:
        _service = LocalLensService()
    return _service


class AnswerRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's natural-language question.")
    location: str = Field("", description="Optional explicit city/park override.")
    topic: str = Field("", description="Optional explicit topic override.")
    session_id: str = Field(
        "", description="Optional session identifier used to carry location/topic across turns."
    )


class AnswerResponse(BaseModel):
    answer: str
    why_this_recommendation: str
    key_tips: list[str]
    confidence_note: str
    citations: list[dict[str, Any]]
    filters_applied: dict[str, str]
    used_local_llm: bool
    source_summary: str
    place_cards: list[dict[str, Any]]
    gallery_images: list[dict[str, str]]


class HealthResponse(BaseModel):
    status: str
    chunks_indexed: int
    places_indexed: int
    ollama_available: bool


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    service = get_service()
    return HealthResponse(
        status="ok",
        chunks_indexed=len(service.chunks),
        places_indexed=len(service.places),
        ollama_available=service.ollama_client.is_available(),
    )


@app.post("/answer", response_model=AnswerResponse)
def answer(request: AnswerRequest) -> AnswerResponse:
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query must not be empty")
    service = get_service()
    payload = service.answer(
        query,
        location=request.location,
        topic=request.topic,
        session_id=request.session_id,
    )
    return AnswerResponse(**payload.to_dict())
