"""
ChromaDB service — embeddings and semantic search only.
Caching is handled separately in cache_service.py (Redis).
"""

import logging
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from app.config import get_settings

logger = logging.getLogger(__name__)

MEETINGS_COLLECTION = "meeting_transcripts"
DECISIONS_COLLECTION = "decisions"

_client: Optional[chromadb.HttpClient] = None


def get_chroma_client() -> chromadb.HttpClient:
    global _client
    if _client is None:
        settings = get_settings()
        _client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
        )
    return _client


def _get_collection(name: str):
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=name,
        embedding_function=DefaultEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"},
    )


def store_meeting_embedding(meeting_id: int, transcript: str, metadata: Dict[str, Any]) -> None:
    """Store a full meeting transcript embedding for semantic search."""
    collection = _get_collection(MEETINGS_COLLECTION)
    collection.upsert(
        ids=[str(meeting_id)],
        documents=[transcript],
        metadatas=[metadata],
    )
    logger.info(f"[CHROMA] Stored meeting embedding id={meeting_id}")


def store_decision_embeddings(meeting_id: int, decisions: List[Dict], project: str) -> None:
    """Store individual decision embeddings for cross-meeting contradiction detection."""
    if not decisions:
        return
    collection = _get_collection(DECISIONS_COLLECTION)
    ids = [f"{meeting_id}_{i}" for i, _ in enumerate(decisions)]
    docs = [d["text"] for d in decisions]
    metas = [{"meeting_id": meeting_id, "project": project, "owner": d.get("owner", "")} for d in decisions]
    collection.upsert(ids=ids, documents=docs, metadatas=metas)
    logger.info(f"[CHROMA] Stored {len(decisions)} decision embeddings for meeting {meeting_id}")


def search_meetings(query: str, n_results: int = 5, project: Optional[str] = None) -> List[Dict]:
    """Semantic search over meeting transcripts."""
    collection = _get_collection(MEETINGS_COLLECTION)
    where = {"project": project} if project else None
    kwargs = {"query_texts": [query], "n_results": min(n_results, collection.count() or 1)}
    if where:
        kwargs["where"] = where
    results = collection.query(**kwargs)
    output = []
    for i, doc in enumerate(results["documents"][0]):
        output.append({
            "meeting_id": int(results["ids"][0][i]),
            "text": doc,
            "score": 1 - results["distances"][0][i],  # cosine distance → similarity
            "metadata": results["metadatas"][0][i],
        })
    return output


def find_contradictions(new_decision_text: str, project: str, exclude_meeting_id: int) -> List[Dict]:
    """
    Find past decisions that semantically conflict with a new decision.
    Returns candidates with similarity > 0.60 (semantically related) 
    from a different meeting — the agent then determines if they contradict.
    """
    collection = _get_collection(DECISIONS_COLLECTION)
    count = collection.count()
    if count == 0:
        return []

    results = collection.query(
        query_texts=[new_decision_text],
        n_results=min(5, count),
        where={"project": project},
    )

    candidates = []
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
        mid = meta.get("meeting_id", -1)
        if mid == exclude_meeting_id:
            continue
        similarity = 1 - results["distances"][0][i]
        if similarity > 0.60:
            candidates.append({
                "meeting_id": mid,
                "decision_text": doc,
                "similarity": similarity,
                "metadata": meta,
            })
    return candidates
