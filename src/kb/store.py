"""Thin Qdrant wrapper for the Knowledge Base.

Embedding model: sentence-transformers, multilingual (bge-m3 if available, else
paraphrase-multilingual-mpnet-base-v2 as a lighter CPU-only fallback).
"""
from __future__ import annotations

import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer

from src.kb.schema import KBEntry, ScoredKBEntry

EMBEDDING_DIMS = {
    "BAAI/bge-m3": 1024,
    "paraphrase-multilingual-mpnet-base-v2": 768,
}


def _load_embedding_model(primary: str, fallback: str) -> tuple[SentenceTransformer, str]:
    try:
        return SentenceTransformer(primary), primary
    except Exception:
        return SentenceTransformer(fallback), fallback


class KBStore:
    def __init__(
        self,
        host: str,
        port: int,
        collection: str,
        embedding_model: str,
        embedding_model_fallback: str,
        client: QdrantClient | None = None,
    ):
        self.host = host
        self.port = port
        self.collection = collection
        self.embedding_model_name_requested = embedding_model
        self.embedding_model_fallback = embedding_model_fallback

        self.model, self.embedding_model_name = _load_embedding_model(embedding_model, embedding_model_fallback)
        self.vector_size = EMBEDDING_DIMS.get(self.embedding_model_name) or self.model.get_sentence_embedding_dimension()

        self.client = client if client is not None else QdrantClient(host=host, port=port)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=qmodels.VectorParams(size=self.vector_size, distance=qmodels.Distance.COSINE),
            )

    def upsert(self, entries: list[KBEntry]) -> None:
        if not entries:
            return
        texts = [f"{entry.problem}\n{entry.solution}" for entry in entries]
        embeddings = self.model.encode(texts, show_progress_bar=False)

        points = [
            qmodels.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{entry.source_thread_id}:{entry.problem}:{entry.solution}")),
                vector=embedding.tolist(),
                payload=entry.model_dump(),
            )
            for entry, embedding in zip(entries, embeddings)
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query: str, top_k: int) -> list[ScoredKBEntry]:
        query_vector = self.model.encode([query], show_progress_bar=False)[0]
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector.tolist(),
            limit=top_k,
        ).points
        return [
            ScoredKBEntry(entry=KBEntry(**point.payload), score=point.score)
            for point in results
        ]
