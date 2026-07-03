"""Qdrant-backed retrieval over the approved Socure knowledge corpus.

Uses Qdrant Cloud when QDRANT_URL is set; otherwise an in-memory index
rebuilt on cold start (the corpus is tiny, so this stays cheap).
"""

import uuid

from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .config import COLLECTION_NAME, EMBEDDING_MODEL, KNOWLEDGE_DIR, QDRANT_API_KEY, QDRANT_URL

_client: QdrantClient | None = None
_embeddings: OpenAIEmbeddings | None = None


def _get_embeddings() -> OpenAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return _embeddings


def _load_chunks() -> list[dict]:
    """Split each knowledge doc on `## ` headings; one chunk per section."""
    chunks = []
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        doc_id = path.stem
        text = path.read_text()
        title = text.splitlines()[0].lstrip("# ").strip()
        sections = text.split("\n## ")
        for i, section in enumerate(sections):
            body = section if i == 0 else "## " + section
            chunks.append(
                {
                    "source_id": f"KB:{doc_id}",
                    "doc_title": title,
                    "text": body.strip(),
                }
            )
    return chunks


def _ensure_index(client: QdrantClient) -> None:
    chunks = _load_chunks()
    if client.collection_exists(COLLECTION_NAME):
        # Reindex when the corpus changed (chunk-count heuristic keeps cold starts
        # cheap while picking up added/removed knowledge docs).
        if client.count(COLLECTION_NAME).count == len(chunks):
            return
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )
    vectors = _get_embeddings().embed_documents([c["text"] for c in chunks])
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(id=str(uuid.uuid4()), vector=v, payload=c)
            for c, v in zip(chunks, vectors)
        ],
    )


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        if QDRANT_URL:
            _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        else:
            _client = QdrantClient(":memory:")
        _ensure_index(_client)
    return _client


def search_knowledge(query: str, k: int = 4) -> list[dict]:
    client = get_client()
    vector = _get_embeddings().embed_query(query)
    hits = client.query_points(collection_name=COLLECTION_NAME, query=vector, limit=k).points
    return [
        {
            "source_id": h.payload["source_id"],
            "doc_title": h.payload["doc_title"],
            "text": h.payload["text"],
            "score": round(h.score, 3),
        }
        for h in hits
    ]
