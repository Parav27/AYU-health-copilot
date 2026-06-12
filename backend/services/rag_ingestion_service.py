"""
rag_ingestion_service.py
------------------------
Phase 1 RAG ingestion utilities for AYU.

This module only prepares the local educational knowledge base for retrieval.
It does not implement chat endpoints, answer generation, or Groq calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


BACKEND_DIR = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = BACKEND_DIR / "knowledge"
CHROMA_DIR = BACKEND_DIR / "chroma_db"
COLLECTION_NAME = "ayu_medical_knowledge"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100


@dataclass(frozen=True)
class IngestionStats:
    documents_loaded: int
    chunks_created: int
    vectors_stored: int
    collection_name: str
    persist_directory: str


def load_markdown_documents(knowledge_dir: Path = KNOWLEDGE_DIR) -> list[Document]:
    """Load local Markdown files and preserve source metadata."""
    if not knowledge_dir.exists():
        raise FileNotFoundError(f"Knowledge directory not found: {knowledge_dir}")

    documents: list[Document] = []
    for path in sorted(knowledge_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": path.name,
                    "topic": path.stem,
                    "file_path": str(path),
                    "content_type": "medical_education_markdown",
                },
            )
        )

    return documents


def chunk_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Split documents into retrieval-sized chunks using LangChain."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n# ", "\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    source_counts: dict[str, int] = {}
    for chunk in chunks:
        source = str(chunk.metadata.get("source", "unknown"))
        chunk_index = source_counts.get(source, 0)
        source_counts[source] = chunk_index + 1
        chunk.metadata["chunk_index"] = chunk_index
        chunk.metadata["chunk_size"] = len(chunk.page_content)

    return chunks


def get_embedding_model():
    """Create the HuggingFace embedding model used by the Chroma index."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def build_chroma_index(
    chunks: list[Document],
    persist_directory: Path = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
) -> int:
    """Embed chunks and persist them in ChromaDB."""
    if not chunks:
        raise ValueError("No chunks were provided for indexing.")

    import chromadb
    from langchain_chroma import Chroma

    persist_directory.mkdir(parents=True, exist_ok=True)
    embeddings = get_embedding_model()
    client = chromadb.PersistentClient(path=str(persist_directory))

    existing_names = [
        collection.name if hasattr(collection, "name") else str(collection)
        for collection in client.list_collections()
    ]
    if collection_name in existing_names:
        client.delete_collection(name=collection_name)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        client=client,
    )

    return vector_store._collection.count()


def ingest_knowledge_base(
    knowledge_dir: Path = KNOWLEDGE_DIR,
    persist_directory: Path = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
) -> IngestionStats:
    """Load, chunk, embed, and store the AYU educational knowledge base."""
    documents = load_markdown_documents(knowledge_dir)
    chunks = chunk_documents(documents)
    vectors_stored = build_chroma_index(chunks, persist_directory, collection_name)

    return IngestionStats(
        documents_loaded=len(documents),
        chunks_created=len(chunks),
        vectors_stored=vectors_stored,
        collection_name=collection_name,
        persist_directory=str(persist_directory),
    )
