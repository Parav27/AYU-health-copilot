"""
Verify the local AYU knowledge base and ChromaDB vector index.

Run from the project root:
    python backend/scripts/verify_knowledge_index.py
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

import chromadb

from backend.services.rag_ingestion_service import (
    CHROMA_DIR,
    COLLECTION_NAME,
    chunk_documents,
    load_markdown_documents,
)


def main() -> None:
    documents = load_markdown_documents()
    chunks = chunk_documents(documents)

    vectors_stored = 0
    if CHROMA_DIR.exists():
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        existing_names = [
            collection.name if hasattr(collection, "name") else str(collection)
            for collection in client.list_collections()
        ]
        if COLLECTION_NAME in existing_names:
            vectors_stored = client.get_collection(COLLECTION_NAME).count()

    print("AYU knowledge index verification")
    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Vectors stored: {vectors_stored}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"ChromaDB path: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
