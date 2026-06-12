"""
report_schemas.py
-----------------
Pydantic models for the AYU health report pipeline.

These schemas serve three purposes:
  1. Validate incoming requests (FastAPI auto-validates)
  2. Define the exact JSON structure Gemini must return
  3. Type the API response the frontend consumes

Design rule: every field has a description so the schema doubles as documentation.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MetricStatus(str, Enum):
    """
    Whether a biomarker value falls within the reference range.
    Using an enum instead of raw strings catches typos at parse time.
    """
    NORMAL = "normal"
    HIGH = "high"
    LOW = "low"
    BORDERLINE = "borderline"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Gemini extraction output — what the LLM returns
# ---------------------------------------------------------------------------

class Biomarker(BaseModel):
    """
    A single measurable value from the report (e.g. HbA1c, Haemoglobin).
    All fields are optional because Gemini may not always extract everything.
    """
    name: str = Field(..., description="Human-readable biomarker name, e.g. 'Haemoglobin'")
    value: float | None = Field(None, description="Numeric value as extracted")
    unit: str | None = Field(None, description="Unit of measurement, e.g. 'g/dL'")
    reference_range: str | None = Field(None, description="Normal range string, e.g. '13.0–17.0'")
    status: MetricStatus = Field(
        MetricStatus.UNKNOWN,
        description="Whether the value is within, above, or below the reference range",
    )
    plain_explanation: str | None = Field(
        None,
        description="One-sentence lay-person explanation of what this marker means",
    )


class ReportCategory(str, Enum):
    """Broad category of the medical report."""
    BLOOD_TEST = "blood_test"
    LIPID_PANEL = "lipid_panel"
    THYROID = "thyroid"
    DIABETES = "diabetes"
    LIVER_FUNCTION = "liver_function"
    KIDNEY_FUNCTION = "kidney_function"
    COMPLETE_BLOOD_COUNT = "complete_blood_count"
    OTHER = "other"
    UNKNOWN = "unknown"


class ExtractedReport(BaseModel):
    """
    Full structured output from the Gemini extraction step.
    This is what gets stored and returned to the frontend.
    """
    report_type: ReportCategory = Field(
        ReportCategory.UNKNOWN,
        description="Best-guess category of the report",
    )
    patient_name: str | None = Field(None, description="Patient name if found in the document")
    report_date: str | None = Field(None, description="Report date if found, ISO-8601 preferred")
    lab_name: str | None = Field(None, description="Name of the lab or hospital")
    biomarkers: list[Biomarker] = Field(
        default_factory=list,
        description="All extracted biomarker values",
    )
    abnormal_flags: list[str] = Field(
        default_factory=list,
        description="Short list of concerning findings to highlight (biomarker names only)",
    )
    health_summary: str = Field(
        ...,
        description="2–4 sentence plain-English summary of the overall report",
    )
    educational_notes: list[str] = Field(
        default_factory=list,
        description="Up to 3 educational bullets the user might find useful",
    )
    extraction_confidence: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Model's self-reported confidence in the extraction (0–1)",
    )


# ---------------------------------------------------------------------------
# API request / response shapes
# ---------------------------------------------------------------------------

class AnalysisResponse(BaseModel):
    """
    The complete payload returned by POST /reports/analyze.
    Wraps ExtractedReport with metadata the frontend needs.
    """
    success: bool
    filename: str
    page_count: int
    char_count: int
    report: ExtractedReport | None = None
    error: str | None = Field(
        None,
        description="Human-readable error message when success=False",
    )
    disclaimer: str = Field(
        default=(
            "⚠️ AYU provides educational insights only. "
            "This is NOT a medical diagnosis. "
            "Always consult a qualified healthcare professional for medical advice."
        ),
        description="Medical disclaimer — always included in every response",
    )


class HealthQuestion(BaseModel):
    """Request body for the /ask endpoint (Phase 3 RAG — schema defined early)."""
    question: str = Field(..., min_length=5, max_length=1000)
    context: dict[str, Any] | None = Field(
        None,
        description="Optional dict of recent report metrics to ground the answer",
    )