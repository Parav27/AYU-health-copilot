"""
groq_service.py
---------------
AYU's Groq integration layer.

Responsibilities:
  - Initialize the Groq client from environment config.
  - Send the extraction prompt and parse the structured JSON response.
  - Retry on transient failures.
  - Parse and validate the response into the existing Pydantic ExtractedReport model.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from groq import Groq
from pydantic import ValidationError

env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(env_path)

try:
    from prompts.health_prompts import SYSTEM_PROMPT, build_extraction_prompt
    from schemas.report_schemas import ExtractedReport, ReportCategory
except ModuleNotFoundError:
    from backend.prompts.health_prompts import SYSTEM_PROMPT, build_extraction_prompt
    from backend.schemas.report_schemas import ExtractedReport, ReportCategory

logger = logging.getLogger("ayu.groq")

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

_client_initialized = False
_client = None


def _init_client() -> None:
    """Configure the Groq client once."""
    global _client_initialized, _client
    if _client_initialized:
        return

    api_key = os.getenv("GROQ_API_KEY")
    print("REPORT ANALYZER GROQ KEY =", api_key)
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY environment variable is not set. "
            "Add it to your backend/.env file."
        )

    _client = Groq(api_key=api_key)
    _client_initialized = True
    logger.info("Groq client initialised (model: %s)", GROQ_MODEL)


def extract_health_report(report_text: str) -> ExtractedReport:
    """Send report text to Groq and return a validated ExtractedReport."""
    _init_client()

    prompt = build_extraction_prompt(report_text)
    raw_response = _call_groq_with_retry(prompt, max_retries=3)
    logger.debug("Raw Groq response length: %d chars", len(raw_response))
    # Help diagnose cases where biomarkers become empty after parsing/validation
    try:
        logger.info("Raw Groq response sample: %s", raw_response[:500])
    except Exception:
        pass

    extracted = _parse_groq_response(raw_response)

    if extracted:
        return extracted

    logger.warning("Full parse failed — attempting partial JSON salvage")
    extracted = _salvage_partial_response(raw_response)
    if extracted:
        return extracted

    logger.error("All parse attempts failed — returning fallback report")
    return _fallback_report("Groq returned an unparseable response.")


def _call_groq_with_retry(prompt: str, max_retries: int = 3) -> str:
    """Call Groq with exponential backoff on transient failures."""
    last_exc: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            _init_client()
            response = _client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=4096,
            )
            text = getattr(response.choices[0].message, "content", "") or ""
            if not text or not text.strip():
                raise ValueError("Groq returned empty text")
            return text
        except Exception as exc:
            last_exc = exc
            wait = 2 ** attempt
            logger.warning(
                "Groq call attempt %d/%d failed: %s — retrying in %ds",
                attempt,
                max_retries,
                exc,
                wait,
            )
            if attempt < max_retries:
                time.sleep(wait)

    raise RuntimeError(
        f"Groq API call failed after {max_retries} attempts. "
        f"Last error: {last_exc}"
    )


def _parse_groq_response(raw: str) -> ExtractedReport | None:
    """Parse the raw Groq response into an ExtractedReport."""
    cleaned = _extract_json_block(raw)
    if not cleaned:
        return None

    try:
        data = json.loads(cleaned)
        return _validate_and_coerce(data)
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.warning("JSON parse/validation error: %s", exc)
        return None


def _extract_json_block(text: str) -> str | None:
    """Extract the JSON object from a string that may contain surrounding prose."""
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        return _fix_trailing_commas(fence_match.group(1))

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        return _fix_trailing_commas(candidate)

    stripped = text.strip()
    if stripped.startswith("{"):
        return _fix_trailing_commas(stripped)

    return None


def _fix_trailing_commas(json_str: str) -> str:
    """Remove trailing commas before ] and }."""
    return re.sub(r",\s*([}\]])", r"\1", json_str)


def _normalize_extraction_json(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize LLM output into values accepted by Pydantic enums.

    This is applied BEFORE ExtractedReport(**data) to avoid validation failure
    that would otherwise drop biomarkers entirely.
    """

    # ---- report_type normalization ----
    # Map non-matching categories into schema-supported values.
    report_type = data.get("report_type")
    if isinstance(report_type, str):
        rt = report_type.strip().lower().replace(" ", "_")
        rt_map = {
            # Common LLM outputs
            "comprehensive_metabolic_panel": "blood_test",
            "cmp": "blood_test",
            "lipid_panel": "lipid_panel",
            "thyroid": "thyroid",
            "diabetes": "diabetes",
            "blood_test": "blood_test",
            # allow already-correct values through
            "blood": "blood_test",
        }
        data["report_type"] = rt_map.get(rt, rt)

    # ---- biomarkers normalization ----
    for bm in data.get("biomarkers", []) or []:
        if not isinstance(bm, dict):
            continue

        status = bm.get("status")
        if isinstance(status, str):
            s = status.strip().lower().replace(" ", "_")
            status_map = {
                "borderline_high": "borderline",
                "borderline_low": "borderline",
                "borderline": "borderline",
                "high": "high",
                "low": "low",
                "normal": "normal",
                "unknown": "unknown",
            }
            normalized = status_map.get(s, s)

            if normalized != status:
                try:
                    logger.info("Normalizing biomarker status: original=%s normalized=%s", status, normalized)
                except Exception:
                    pass

            bm["status"] = normalized

    # ---- abnormal_flags normalization ----
    # Leave as-is; these are plain strings.

    return data


def _validate_and_coerce(data: dict[str, Any]) -> ExtractedReport:
    """Normalize values and validate against the existing Pydantic schema."""
    normalized = _normalize_extraction_json(data)
    return ExtractedReport(**normalized)



def _salvage_partial_response(raw: str) -> ExtractedReport | None:
    """Build a minimal ExtractedReport from whatever JSON-like text we can recover."""
    try:
        summary_match = re.search(r'"health_summary"\s*:\s*"([^"]+)"', raw)
        summary = summary_match.group(1) if summary_match else (
            "The report was processed but detailed extraction encountered an issue. "
            "Please review your report manually or try re-uploading."
        )

        return ExtractedReport(
            report_type=ReportCategory.UNKNOWN,
            biomarkers=[],
            abnormal_flags=[],
            health_summary=summary,
            educational_notes=[
                "For accurate analysis, ensure the PDF contains selectable text rather than scanned images."
            ],
            extraction_confidence=0.1,
        )
    except Exception:
        return None


def _fallback_report(reason: str) -> ExtractedReport:
    """Returns a safe fallback when everything else fails."""
    return ExtractedReport(
        report_type=ReportCategory.UNKNOWN,
        biomarkers=[],
        abnormal_flags=[],
        health_summary=(
            f"Unable to fully analyse this report at this time. ({reason}) "
            "Please try again or consult your healthcare provider directly."
        ),
        educational_notes=[],
        extraction_confidence=0.0,
    )
