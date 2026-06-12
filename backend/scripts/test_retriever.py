"""
Smoke-test AYU Phase 2 retrieval against the local ChromaDB index.

Run from the project root:
    python backend/scripts/test_retriever.py
"""

from __future__ import annotations

import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from services.retriever_service import embedding_model_name, retrieve_context


TEST_QUERIES = [
    "What is HbA1c?",
    "What does high LDL mean?",
    "What is Vitamin D deficiency?",
    "What is TSH?",
]


def _preview(text: str, max_chars: int = 450) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[:max_chars].rstrip()}..."


def _format_score(score: float | None) -> str:
    if score is None:
        return "not available"
    return f"{score:.4f}"


def main() -> None:
    print("AYU retrieval smoke test")
    print(f"Embedding model: {embedding_model_name()}")
    print("Similarity search: k=3")

    for query in TEST_QUERIES:
        print("\n" + "=" * 80)
        print(f"Query: {query}")
        print("Retrieved Chunks:")

        results = retrieve_context(query)
        for index, result in enumerate(results, start=1):
            metadata = result["metadata"]
            print(f"\n{index}. Source File: {metadata.get('source', 'unknown')}")
            print(f"   Topic: {metadata.get('topic', 'unknown')}")
            print(f"   Chunk Index: {metadata.get('chunk_index', 'unknown')}")
            print(f"   Similarity Score: {_format_score(result['relevance_score'])}")
            print(f"   Chunk Text: {_preview(result['text'])}")


if __name__ == "__main__":
    main()
