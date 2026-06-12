"""
pdf_service.py
--------------
Handles all PDF processing for AYU.

Responsibilities:
  - Validate uploaded files are actually PDFs
  - Extract text from each page using PyMuPDF (fitz)
  - Clean and normalise the extracted text
  - Return page count + concatenated text

Design notes:
  - Pure functions — no side effects, easy to unit test
  - Raises specific ValueError / RuntimeError so callers can give users clear messages
  - Does NOT write to disk; works entirely from bytes in memory (safer for a web service)
"""

from __future__ import annotations

import io
import re

import fitz  # PyMuPDF


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_text_from_bytes(pdf_bytes: bytes, filename: str = "upload.pdf") -> dict:
    """
    Extract all text from a PDF supplied as raw bytes.

    Args:
        pdf_bytes: Raw PDF file content.
        filename:  Original filename for error messages.

    Returns:
        {
            "text":       str   — full concatenated and cleaned text,
            "page_count": int   — number of pages in the document,
            "char_count": int   — character count of extracted text,
            "per_page":   list  — list of per-page text strings (useful for debugging),
        }

    Raises:
        ValueError:   If the file is not a valid PDF or is encrypted.
        RuntimeError: If PyMuPDF encounters an unexpected error.
    """
    _validate_pdf_header(pdf_bytes, filename)

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except fitz.FileDataError as exc:
        raise ValueError(f"'{filename}' could not be opened as a PDF: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Unexpected error opening '{filename}': {exc}") from exc

    if doc.is_encrypted:
        raise ValueError(
            f"'{filename}' is password-protected. "
            "Please remove the password and re-upload."
        )

    page_count = len(doc)
    if page_count == 0:
        raise ValueError(f"'{filename}' contains no pages.")

    per_page: list[str] = []
    for page_num in range(page_count):
        try:
            page = doc[page_num]
            raw = page.get_text("text")          # plain text, preserving layout order
            cleaned = _clean_page_text(raw)
            per_page.append(cleaned)
        except Exception as exc:
            # Don't abort — log the bad page and continue
            per_page.append(f"[Page {page_num + 1} extraction failed: {exc}]")

    doc.close()

    full_text = "\n\n".join(p for p in per_page if p.strip())
    char_count = len(full_text)

    if char_count < 20:
        raise ValueError(
            f"Very little text was extracted from '{filename}'. "
            "The PDF may be a scanned image. OCR support is on the roadmap."
        )

    return {
        "text": full_text,
        "page_count": page_count,
        "char_count": char_count,
        "per_page": per_page,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_pdf_header(data: bytes, filename: str) -> None:
    """
    PDFs always start with %PDF-.
    Checking the magic bytes is faster and more reliable than trusting the extension.
    """
    if not data.startswith(b"%PDF-"):
        raise ValueError(
            f"'{filename}' does not appear to be a valid PDF file. "
            "Please upload a .pdf document."
        )


def _clean_page_text(raw: str) -> str:
    """
    Normalise whitespace and remove common PDF artifacts.

    Steps:
      1. Replace non-breaking spaces and other unicode spaces with regular spaces.
      2. Collapse runs of spaces within a line.
      3. Remove lines that are just page numbers, dashes, or underscores.
      4. Collapse more than two consecutive blank lines into two.
    """
    # Unicode space normalisation
    text = raw.replace("\u00a0", " ").replace("\u2003", " ").replace("\u2002", " ")

    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()

        # Drop lines that are purely decorative or are lone page numbers
        if re.fullmatch(r"[-_=]{3,}", stripped):
            continue
        if re.fullmatch(r"\d{1,3}", stripped):
            continue

        # Collapse internal whitespace runs
        cleaned_line = re.sub(r" {2,}", " ", stripped)
        lines.append(cleaned_line)

    # Join and collapse excess blank lines
    joined = "\n".join(lines)
    joined = re.sub(r"\n{3,}", "\n\n", joined)
    return joined.strip()