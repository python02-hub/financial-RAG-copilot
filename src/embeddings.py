"""
embeddings.py — Local, offline embedding model wrapper.

Uses sentence-transformers so the whole pipeline runs on CPU with no API
key and no network call at query/embedding time. This keeps the system
"run on any device" — a laptop with no GPU handles this fine at the scale
of a handful of filings.

Model: BAAI/bge-small-en-v1.5 (384-dim, strong retrieval quality for its
size, ~130MB). Swap EMBEDDING_MODEL_NAME below for a larger model if you
have the compute and want higher recall.
"""
from __future__ import annotations

from functools import lru_cache

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# bge models are trained to expect this instruction prefix on *queries*
# (not on the documents being indexed) — leaving it off measurably hurts
# retrieval quality for this model family.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_documents(texts: list[str]) -> list[list[float]]:
    model = _get_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    model = _get_model()
    vector = model.encode(
        QUERY_INSTRUCTION + text, normalize_embeddings=True, show_progress_bar=False
    )
    return vector.tolist()
