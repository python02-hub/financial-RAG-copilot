"""
vectorstore.py — Thin wrapper around a local, persistent ChromaDB collection.

ChromaDB is used in its embedded (no-server) mode, storing everything under
data/chroma_db/ on disk. This means the whole vector database is just files
next to the project — no external service, no Pinecone account needed, and
it survives across runs so you don't re-embed filings every time.
"""
from __future__ import annotations

import os
from typing import Optional

from .embeddings import embed_documents, embed_query
from .ingest import Chunk

DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
DEFAULT_COLLECTION = "filings"


class FilingVectorStore:
    def __init__(self, db_path: str = DEFAULT_DB_PATH, collection_name: str = DEFAULT_COLLECTION):
        import chromadb

        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: list[Chunk], batch_size: int = 64) -> int:
        """Embed and upsert chunks. Returns number of chunks added."""
        added = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.text for c in batch]
            embeddings = embed_documents(texts)
            self.collection.upsert(
                ids=[c.id for c in batch],
                embeddings=embeddings,
                documents=texts,
                metadatas=[c.to_metadata() for c in batch],
            )
            added += len(batch)
        return added

    def query(
        self,
        query_text: str,
        n_results: int = 6,
        company: Optional[str] = None,
        section_contains: Optional[str] = None,
        chunk_type: Optional[str] = None,
    ) -> list[dict]:
        """Semantic search with optional metadata filters. Returns a list of
        {text, metadata, distance} dicts ordered by relevance."""
        where = {}
        if company:
            where["company"] = company
        if chunk_type:
            where["chunk_type"] = chunk_type
        # NB: Chroma's `where` doesn't support substring match on metadata,
        # so section_contains is applied as a post-filter below.

        query_embedding = embed_query(query_text)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=max(n_results * 3, n_results) if section_contains else n_results,
            where=where or None,
        )

        out = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            if section_contains and section_contains.lower() not in (meta.get("section", "").lower()):
                continue
            out.append({"text": doc, "metadata": meta, "distance": dist})
            if len(out) >= n_results:
                break
        return out

    def count(self) -> int:
        return self.collection.count()

    def companies(self) -> list[str]:
        data = self.collection.get(include=["metadatas"])
        return sorted({m["company"] for m in data["metadatas"]}) if data["metadatas"] else []
