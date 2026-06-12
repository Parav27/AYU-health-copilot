"""
reports.py  (router)
--------------------
FastAPI route definitions for the AYU report analysis pipeline.

Route design principles:
  - Each route does ONE thing: receive, validate, orchestrate, respond.
  - All business logic lives in services — this file is just wiring.
  - Errors are caught and converted to clean JSON responses with HTTP status codes.
  - The success=False pattern (rather than always raising HTTPException) gives the
    frontend a consistent envelope to parse regardless of outcome.

Endpoints:
  POST /reports/analyze   — Upload a PDF, get back structured health data
  GET  /reports/health    — Simple liveness check for the AI layer
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

try:
    from schemas.report_schemas import AnalysisResponse
    from backend.services.groq_service import extract_health_report
    from backend.services.pdf_service import extract_text_from_bytes
except ModuleNotFoundError:
    from backend.schemas.report_schemas import AnalysisResponse
    from backend.services.groq_service import extract_health_report
    from backend.services.pdf_service import extract_text_from_bytes

logger = logging.getLogger("ayu.router.reports")

router = APIRouter(prefix="/reports", tags=["Reports"])

# File constraints
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"application/pdf", "application/octet-stream"}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    summary="Upload a medical report PDF for AI analysis",
    description=(
        "Accepts a PDF medical report, extracts text, sends it to Groq for "
        "structured biomarker extraction, and returns a health summary. "
        "Educational use only — not a diagnostic tool."
    ),
    status_code=status.HTTP_200_OK,
)
async def analyze_report(
    file: UploadFile = File(..., description="PDF medical report to analyse"),
) -> AnalysisResponse:
    """
    Main analysis endpoint — the core of Phase 1.

    Flow:
      1. Validate file type and size.
      2. Read bytes into memory.
      3. Extract text via pdf_service.
      4. Send text to groq_service.
      5. Return structured AnalysisResponse.

    All errors return success=False with a clear error message rather than
    raising HTTPException — this gives the frontend a single response shape.
    """
    # --- Validate file extension ---
    filename = file.filename or "upload.pdf"
    if not filename.lower().endswith(".pdf"):
        return AnalysisResponse(
            success=False,
            filename=filename,
            page_count=0,
            char_count=0,
            error="Only PDF files are supported. Please upload a .pdf document.",
        )

    # --- Read file into memory ---
    try:
        pdf_bytes = await file.read()
    except Exception as exc:
        logger.error("File read error for '%s': %s", filename, exc)
        return AnalysisResponse(
            success=False,
            filename=filename,
            page_count=0,
            char_count=0,
            error="Failed to read the uploaded file. Please try again.",
        )

    # --- Size check ---
    file_size_mb = len(pdf_bytes) / (1024 * 1024)
    if len(pdf_bytes) > MAX_FILE_SIZE_BYTES:
        return AnalysisResponse(
            success=False,
            filename=filename,
            page_count=0,
            char_count=0,
            error=(
                f"File size ({file_size_mb:.1f} MB) exceeds the {MAX_FILE_SIZE_MB} MB limit. "
                "Please upload a smaller file."
            ),
        )

    logger.info("Processing '%s' (%.2f MB)", filename, file_size_mb)

    # --- PDF text extraction ---
    try:
        pdf_result = extract_text_from_bytes(pdf_bytes, filename)
    except ValueError as exc:
        # User-facing validation errors (wrong format, encrypted, etc.)
        logger.warning("PDF validation error for '%s': %s", filename, exc)
        return AnalysisResponse(
            success=False,
            filename=filename,
            page_count=0,
            char_count=0,
            error=str(exc),
        )
    except RuntimeError as exc:
        logger.error("PDF extraction runtime error for '%s': %s", filename, exc)
        return AnalysisResponse(
            success=False,
            filename=filename,
            page_count=0,
            char_count=0,
            error="An error occurred while reading the PDF. Please try a different file.",
        )

    page_count = pdf_result["page_count"]
    char_count = pdf_result["char_count"]
    report_text = pdf_result["text"]

    logger.info(
        "'%s' extracted: %d pages, %d characters",
        filename,
        page_count,
        char_count,
    )

    # --- Groq AI extraction ---
    try:
        extracted_report = extract_health_report(report_text)
    except EnvironmentError as exc:
        # Missing API key — this is a server config problem
        logger.critical("Groq config error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI service is not configured. Please contact support.",
        )
    except RuntimeError as exc:
        logger.error("Groq extraction failed for '%s': %s", filename, exc)
        message = str(exc)
        quota_message = (
            "The Groq service is currently unavailable or quota-limited. "
            "Please try again in a few minutes or update your API access."
        )
        if "429" in message or "quota" in message.lower() or "resource_exhausted" in message.lower():
            error_message = quota_message
        else:
            error_message = (
                "The AI analysis service encountered an error. "
                "Please try again in a moment."
            )
        return AnalysisResponse(
            success=False,
            filename=filename,
            page_count=page_count,
            char_count=char_count,
            error=error_message,
        )

    logger.info(
        "'%s' analysis complete — confidence: %.2f, biomarkers: %d",
        filename,
        extracted_report.extraction_confidence,
        len(extracted_report.biomarkers),
    )

    return AnalysisResponse(
        success=True,
        filename=filename,
        page_count=page_count,
        char_count=char_count,
        report=extracted_report,
    )


@router.get(
    "/health",
    summary="Check AI service health",
    description="Returns the status of the Groq integration.",
)
async def ai_health_check() -> dict:
    """
    Lightweight liveness endpoint — confirms env var is set without making an API call.
    Use the /analyze endpoint to do a full end-to-end smoke test.
    """
    import os
    api_key_set = bool(os.getenv("GROQ_API_KEY"))
    return {
        "status": "ok" if api_key_set else "degraded",
        "groq_api_key_configured": api_key_set,
        "message": (
            "Ready to analyse reports."
            if api_key_set
            else "GROQ_API_KEY is not set — analysis will fail."
        ),
    }