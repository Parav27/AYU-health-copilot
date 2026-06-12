"""
Smoke-test AYU Phase 3 RAG generation.

Run from the project root:
    python backend/scripts/test_rag_generation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

load_dotenv(BACKEND_DIR / ".env")

from backend.services.rag_generation_service import answer_question


TEST_QUESTIONS = [
    "What is HbA1c?",
    "What does high LDL mean?",
    "What is TSH?",
    "What is Vitamin D deficiency?",
]


def _format_score(score: float | None) -> str:
    if score is None:
        return "not available"
    return f"{score:.4f}"


def main() -> None:
    print("AYU RAG generation smoke test")

    for question in TEST_QUESTIONS:
        result = answer_question(question)

        print("\n" + "=" * 80)
        print(f"Question: {question}")

        print("\nRetrieved Sources:")
        for source in result["sources"]:
            print(
                " - "
                f"{source['source']} "
                f"(topic={source['topic']}, "
                f"chunk={source['chunk_index']}, "
                f"score={_format_score(source['similarity_score'])})"
            )

        print("\nGenerated Answer:")
        print(result["answer"])
        print(f"\nConfidence: {result['confidence']}")


if __name__ == "__main__":
    main()
