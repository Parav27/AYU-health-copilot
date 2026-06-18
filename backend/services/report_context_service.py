"""
report_context_service.py
--------------------------
Lightweight in-memory "report context" store for Phase 6.

Design goals:
- Reuse the existing extracted report (ExtractedReport) from /reports/analyze.
- Do NOT persist to ChromaDB.
- Do NOT modify the PDF analysis pipeline.
- Keep only the latest report for the current server session.

This module is imported by the chat RAG generator.
"""

from __future__ import annotations

import threading
from typing import Any

try:
    from schemas.report_schemas import ExtractedReport
except ModuleNotFoundError:
    from backend.schemas.report_schemas import ExtractedReport



_lock = threading.Lock()
_latest_report: ExtractedReport | None = None


def set_latest_report(report: ExtractedReport) -> None:
    global _latest_report
    with _lock:
        _latest_report = report


def get_latest_report() -> ExtractedReport | None:
    with _lock:
        return _latest_report


def has_report() -> bool:
    with _lock:
        return _latest_report is not None


def format_report_context(report: ExtractedReport) -> str:
    """Create a compact, prompt-ready report context block.

    Requirements satisfied:
    - Include biomarkers that are abnormal OR have a numeric value.
    - Include name, value, unit, status.
    - Keep existing health summary and abnormal flags.
    """

    def _status_to_label(status: object) -> str:
        raw = str(status).lower() if status is not None else "unknown"
        # Pydantic enum strings often come as "MetricStatus.HIGH" or just "high"
        raw = raw.replace("metricstatus.", "")
        raw = raw.replace(".", "_")
        if raw in {"high", "low", "borderline", "normal", "unknown"}:
            return raw
        return "unknown"


    included: list[Any] = []
    for bm in report.biomarkers:
        status_label = _status_to_label(bm.status)
        has_value = bm.value is not None
        is_abnormal = status_label in {"high", "low", "borderline"}
        if is_abnormal or has_value:
            included.append(bm)

    biomarker_sections: list[str] = []
    for bm in included:
        status_label = _status_to_label(bm.status)
        value_str = "N/A" if bm.value is None else str(bm.value)
        unit_str = (bm.unit or "").strip()
        unit_part = unit_str if unit_str else "(unit not provided)"

        biomarker_sections.append(
            f"{bm.name}:\n"
            f"* value: {value_str}\n"
            f"* unit: {unit_part}\n"
            f"* status: {status_label}"
        )

    biomarkers_block = (
        "\n\n".join(biomarker_sections)
        if biomarker_sections
        else "(No extracted biomarker values available.)"
    )

    header_parts: list[str] = ["Extracted Report"]
    if report.report_type:
        header_parts.append(
            f"Type: {report.report_type.value if hasattr(report.report_type, 'value') else report.report_type}"
        )
    if report.report_date:
        header_parts.append(f"Date: {report.report_date}")
    if report.lab_name:
        header_parts.append(f"Lab: {report.lab_name}")
    if report.patient_name:
        header_parts.append(f"Patient: {report.patient_name}")

    header = " · ".join(header_parts)

    flags_block = ""
    if report.abnormal_flags:
        flags_block = "\n\nFlagged markers (names): " + ", ".join(report.abnormal_flags)

    return (
        f"REPORT CONTEXT\n{header}\n\nBiomarkers (from your uploaded report):\n{biomarkers_block}"
        f"{flags_block}"
        f"\n\nHealth summary (plain language):\n{report.health_summary}"
    )

