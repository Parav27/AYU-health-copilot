"""
Build and verify the local AYU ChromaDB knowledge index.

Run from the project root:
    python backend/scripts/ingest_knowledge.py
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

from services.rag_ingestion_service import ingest_knowledge_base


def main() -> None:
    stats = ingest_knowledge_base()

    print("AYU knowledge ingestion complete")
    print(f"Documents loaded: {stats.documents_loaded}")
    print(f"Chunks created: {stats.chunks_created}")
    print(f"Vectors stored: {stats.vectors_stored}")
    print(f"Collection: {stats.collection_name}")
    print(f"ChromaDB path: {stats.persist_directory}")


if __name__ == "__main__":
    main()
