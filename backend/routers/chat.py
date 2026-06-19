"""
chat.py
-------
FastAPI route for AYU's RAG chat API.

This router exposes grounded educational answers from the existing RAG pipeline.
It intentionally does not implement memory, authentication, frontend behavior, or
changes to the report-analysis pipeline.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

try:
    from schemas.chat_schemas import CHAT_DISCLAIMER, ChatAskResponse, ChatQuestionRequest
except ModuleNotFoundError:
    from backend.schemas.chat_schemas import CHAT_DISCLAIMER, ChatAskResponse, ChatQuestionRequest

logger = logging.getLogger("ayu.router.chat")

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post(
    "/ask",
    response_model=ChatAskResponse,
    summary="Ask AYU an educational medical question",
    description=(
        "Retrieves relevant chunks from AYU's local medical knowledge base and "
        "uses Groq to generate a grounded educational answer. This endpoint does "
        "not diagnose, prescribe, store memory, or replace professional care."
    ),
    responses={
        400: {
            "description": "Invalid question.",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "error": "Question must not be empty.",
                        "disclaimer": CHAT_DISCLAIMER,
                    }
                }
            },
        },
        422: {"description": "Missing or malformed request body."},
        503: {
            "description": "Required RAG dependency is not configured or unavailable.",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "error": "Knowledge base is not indexed. Please run ingestion first.",
                        "disclaimer": CHAT_DISCLAIMER,
                    }
                }
            },
        },
        502: {"description": "LLM generation failed."},
    },
)
async def ask_chat_question(payload: ChatQuestionRequest) -> ChatAskResponse:
    """
    Answer a single educational medical question with retrieved context.

    Validation for missing, empty, or very short questions is handled by the
    request schema before this function runs.
    """
    try:
        from services.rag_generation_service import answer_question
    except ModuleNotFoundError:
        from backend.services.rag_generation_service import answer_question
        
    try:
        result = answer_question(payload.question)
    except ValueError as exc:
        logger.warning("Invalid chat question: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "success": False,
                "error": str(exc),
                "disclaimer": CHAT_DISCLAIMER,
            },
        ) from exc
    except FileNotFoundError as exc:
        logger.error("RAG knowledge base unavailable: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "success": False,
                "error": "Knowledge base is not indexed. Please run ingestion first.",
                "disclaimer": CHAT_DISCLAIMER,
            },
        ) from exc
    except EnvironmentError as exc:
        logger.error("RAG generation configuration error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "success": False,
                "error": "AI service is not configured. Please set GROQ_API_KEY.",
                "disclaimer": CHAT_DISCLAIMER,
            },
        ) from exc
    except RuntimeError as exc:
        logger.error("RAG generation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "success": False,
                "error": "The AI answer service failed. Please try again in a moment.",
                "disclaimer": CHAT_DISCLAIMER,
            },
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected chat endpoint failure")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "error": "An unexpected chat service error occurred.",
                "disclaimer": CHAT_DISCLAIMER,
            },
        ) from exc

    return ChatAskResponse(
        success=True,
        answer=result["answer"],
        sources=result.get("sources", []),
        confidence=result.get("confidence", "retrieval_based"),
        disclaimer=CHAT_DISCLAIMER,
    )
