"""Knowledge retrieval — Qdrant + embeddings locally; lightweight keyword search on Vercel."""

import os
import re
import uuid

from .config import COLLECTION_NAME, EMBEDDING_MODEL, IS_VERCEL, KNOWLEDGE_DIR, QDRANT_API_KEY, QDRANT_URL

_client = None
_embeddings = None
_chunks_cache: list[dict] | None = None


def _use_keyword_rag() -> bool:
    """Vercel hobby memory caps make in-memory Qdrant + batch embeddings too heavy."""
    if os.getenv("USE_KEYWORD_RAG", "").lower() in ("1", "true", "yes"):
        return True
    return IS_VERCEL or not QDRANT_URL


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        from langchain_openai import OpenAIEmbeddings

        _embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return _embeddings


def _load_chunks() -> list[dict]:
    global _chunks_cache
    if _chunks_cache is not None:
        return _chunks_cache
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
    _chunks_cache = chunks
    return chunks


def _keyword_search(query: str, k: int = 4) -> list[dict]:
    terms = [t.lower() for t in re.findall(r"\w{3,}", query.lower())]
    if not terms:
        return []
    scored: list[tuple[int, dict]] = []
    for chunk in _load_chunks():
        hay = f"{chunk['doc_title']} {chunk['text']}".lower()
        score = sum(hay.count(t) for t in terms)
        if score:
            scored.append((score, chunk))
    scored.sort(key=lambda x: -x[0])
    return [
        {**chunk, "score": float(score)}
        for score, chunk in scored[:k]
    ]


def _qdrant_client():
    from qdrant_client import QdrantClient

    if QDRANT_URL:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return QdrantClient(":memory:")


def _ensure_index(client) -> None:
    from qdrant_client.models import Distance, PointStruct, VectorParams

    chunks = _load_chunks()
    if client.collection_exists(COLLECTION_NAME):
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


def get_client():
    global _client
    if _client is None:
        _client = _qdrant_client()
        _ensure_index(_client)
    return _client


def search_knowledge(query: str, k: int = 4) -> list[dict]:
    if _use_keyword_rag():
        return _keyword_search(query, k)

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
