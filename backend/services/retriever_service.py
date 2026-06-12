"""
retriever_service.py
--------------------
Phase 2 RAG retrieval utilities for AYU.

This module only retrieves relevant chunks from the local ChromaDB knowledge
base. It does not call Groq, construct answers, expose routes, or use chains.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from services.rag_ingestion_service import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    get_embedding_model,
)


DEFAULT_K = 3
_VECTOR_STORE = None


def _load_vector_store(
    persist_directory: Path = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
):
    """Load the existing ChromaDB collection with the same embedding model."""
    global _VECTOR_STORE
    if _VECTOR_STORE is not None:
        return _VECTOR_STORE

    if not persist_directory.exists():
        raise FileNotFoundError(
            f"ChromaDB directory not found: {persist_directory}. "
            "Run backend/scripts/ingest_knowledge.py first."
        )

    from langchain_chroma import Chroma

    _VECTOR_STORE = Chroma(
        collection_name=collection_name,
        persist_directory=str(persist_directory),
        embedding_function=get_embedding_model(),
    )
    return _VECTOR_STORE


def retrieve_context(question: str, k: int = DEFAULT_K) -> list[dict[str, Any]]:
    """
    Retrieve the most relevant medical knowledge chunks for a question.

    Returns:
        [
            {
                "text": "...",
                "metadata": {...},
                "relevance_score": 0.82,
            }
        ]
    """
    cleaned_question = question.strip()
    if not cleaned_question:
        raise ValueError("Question must not be empty.")

    vector_store = _load_vector_store()

    docs_and_scores = vector_store.similarity_search_with_score(cleaned_question, k=k)
    return [
        {
            "text": doc.page_content,
            "metadata": dict(doc.metadata),
            "relevance_score": _distance_to_similarity(distance),
            "distance": distance,
        }
        for doc, distance in docs_and_scores
    ]


def _distance_to_similarity(distance: float) -> float:
    """Convert Chroma distance to a readable 0-1 similarity-style score."""
    return 1.0 / (1.0 + max(distance, 0.0))


def embedding_model_name() -> str:
    """Expose the configured embedding model for scripts and diagnostics."""
    return EMBEDDING_MODEL_NAME
