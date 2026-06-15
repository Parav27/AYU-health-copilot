"""
health_prompts.py
-----------------
Prompt templates for AYU's AI integration.

Why a dedicated prompts module?
  - Prompts are configuration, not code — isolating them makes tuning fast.
  - Every prompt is a function so callers can inject dynamic values cleanly.
  - The system prompt enforces the educational-only, no-diagnosis constraint.

Prompt engineering principles used here:
  1. Role assignment — give the model a clear identity.
  2. Output format specification — ask for JSON with a concrete schema example.
  3. Constraint injection — hard rules the model must follow.
  4. Chain-of-thought nudge — "think step-by-step" before outputting JSON.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# System prompt — injected on every call
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are AYU's medical report analysis engine. You are a highly knowledgeable medical 
information assistant with expertise in interpreting laboratory reports and health metrics.

CRITICAL RULES — NEVER VIOLATE:
1. You provide EDUCATIONAL INFORMATION ONLY. You do NOT diagnose diseases.
2. You do NOT recommend medications, dosages, or treatments.
3. You do NOT make predictions about health outcomes.
4. When explaining abnormal values, focus on what the marker measures and general 
   medical knowledge — never say "you have [disease]" or "you need [treatment]".
5. Always use plain, compassionate language a non-medical person can understand.
6. Your confidence score must reflect genuine uncertainty — never fake high confidence.

Your output is always valid JSON. Never include markdown code fences or explanatory 
text outside the JSON object.
""".strip()


# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

def build_extraction_prompt(report_text: str) -> str:
    """
    Build the prompt that extracts structured biomarker data from raw PDF text.

    The prompt uses a concrete JSON schema example so AI understands the exact
    shape expected. The model is instructed to reason step-by-step internally before
    writing the JSON (this reduces hallucination on edge cases).

    Args:
        report_text: Cleaned text extracted from the uploaded PDF.

    Returns:
        Complete prompt string ready to be sent to AI.
    """
    # Truncate to ~12,000 characters to stay well within AI's context window
    # while leaving headroom for the prompt itself and the JSON response.
    MAX_TEXT_CHARS = 12_000
    if len(report_text) > MAX_TEXT_CHARS:
        truncated = report_text[:MAX_TEXT_CHARS]
        truncation_note = (
            f"\n\n[NOTE: The document was truncated to {MAX_TEXT_CHARS} characters "
            "to fit within processing limits. Analyse only the text provided.]"
        )
    else:
        truncated = report_text
        truncation_note = ""

    return f"""
You are analysing a medical laboratory report. Follow these steps internally before 
writing your JSON response:

STEP 1 — Identify the report type (blood test, lipid panel, thyroid, etc.)
STEP 2 — Find patient name, report date, and lab name if present.
STEP 3 — Extract every biomarker/test result you can find:
         - name, numeric value, unit, reference range
         - Determine status: "normal", "high", "low", "borderline", or "unknown"
         - Write a one-sentence plain-English explanation of what the marker measures
STEP 4 — List the names of any markers that are HIGH, LOW, or BORDERLINE in abnormal_flags.
STEP 5 — Write a 2–4 sentence health_summary in plain English. Start with the 
         overall impression, mention any notable findings. Do NOT diagnose.
STEP 6 — Write up to 3 short educational_notes (general health tips or context 
         relevant to the markers found — never specific advice).
STEP 7 — Assign extraction_confidence: 0.9+ if the text is clear, 0.5–0.89 if 
         partially readable, below 0.5 if very unclear.

OUTPUT — Return ONLY this JSON, no other text:

{{
  "report_type": "blood_test",
  "patient_name": "string or null",
  "report_date": "string or null",
  "lab_name": "string or null",
  "biomarkers": [
    {{
      "name": "Haemoglobin",
      "value": 13.2,
      "unit": "g/dL",
      "reference_range": "13.0–17.0",
      "status": "normal",
      "plain_explanation": "Haemoglobin carries oxygen in your red blood cells; this value is within the healthy range."
    }}
  ],
  "abnormal_flags": ["Glucose", "LDL Cholesterol"],
  "health_summary": "Your report shows ...",
  "educational_notes": ["...", "..."],
  "extraction_confidence": 0.85
}}

REPORT TEXT TO ANALYSE:
{truncated}{truncation_note}
""".strip()


# ---------------------------------------------------------------------------
# Follow-up / clarification prompt (used if first extraction is incomplete)
# ---------------------------------------------------------------------------

def build_clarification_prompt(original_text: str, partial_json: str) -> str:
    """
    If the first extraction attempt returns incomplete data, this prompt asks
    Gemini to re-examine the text and fill in the gaps.

    Args:
        original_text: The same report text sent the first time.
        partial_json:  The incomplete JSON Gemini returned previously.
    """
    return f"""
Your previous analysis of this medical report was partially incomplete.

PREVIOUS ANALYSIS:
{partial_json}

ORIGINAL REPORT TEXT:
{original_text[:8000]}

Please re-examine the report text and:
1. Fill in any null values where the data IS present in the text.
2. Add any biomarkers you missed in the first pass.
3. Correct any status values that seem wrong given the value and reference range.

Return the COMPLETE, corrected JSON using the same schema. No other text.
""".strip()