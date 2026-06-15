"""
chat_schemas.py
---------------
Pydantic schemas for AYU's RAG chat API.

These models only describe the backend chat endpoint contract. They do not add
conversation memory, authentication, frontend behavior, or report analysis.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


CHAT_DISCLAIMER = (
    "AYU provides educational information only. It does not diagnose, prescribe, "
    "or replace advice from a qualified healthcare professional."
)


class ChatQuestionRequest(BaseModel):
    """Request body for POST /api/v1/chat/ask."""

    question: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="Educational medical question to answer using AYU's knowledge base.",
        examples=["What is HbA1c?"],
    )

    @field_validator("question")
    @classmethod
    def question_must_have_content(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Question must not be empty.")
        if len(cleaned) < 5:
            raise ValueError("Question is too short. Please ask a complete question.")
        return cleaned


class ChatSource(BaseModel):
    """A knowledge-base chunk used to ground an answer."""

    source: str = Field(..., description="Source Markdown file name.")
    topic: str = Field(..., description="Knowledge base topic.")
    chunk_index: int | str = Field(..., description="Chunk index within the source document.")
    similarity_score: float | None = Field(
        None,
        description="Similarity-style score derived from vector distance, when available.",
    )


class ChatAskResponse(BaseModel):
    """Response body for successful RAG answers."""

    success: bool = Field(True, description="Whether AYU generated an answer successfully.")
    answer: str = Field(..., description="Grounded educational answer.")
    sources: list[ChatSource] = Field(
        default_factory=list,
        description="Retrieved knowledge chunks used as context.",
    )
    confidence: Literal["retrieval_based" ,"retrieval_based_report_aware"] = Field(
        "retrieval_based",
        description="Indicates the answer was generated from retrieved context.",
    )
    disclaimer: str = Field(CHAT_DISCLAIMER, description="Medical safety disclaimer.")


class ChatErrorResponse(BaseModel):
    """Consistent error response for chat failures."""

    success: bool = Field(False, description="Always false for chat errors.")
    error: str = Field(..., description="Human-readable error message.")
    disclaimer: str = Field(CHAT_DISCLAIMER, description="Medical safety disclaimer.")
    details: dict[str, Any] | None = Field(
        None,
        description="Optional non-sensitive diagnostic details.",
    )
