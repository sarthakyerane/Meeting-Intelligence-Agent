"""
Redis-backed semantic cache.

Design: embed query → fetch all cached vectors from Redis → 
compute cosine similarity in Python (numpy) → hit if score ≥ threshold.

Known limitation: O(n) over cache size. Fine for ~50-200 cached queries.
At scale → Redis Stack with built-in HNSW vector search.
"""

import json
import time
import logging
import hashlib
import numpy as np
from typing import Optional

import redis as redis_lib
from app.config import get_settings

logger = logging.getLogger(__name__)

CACHE_COLLECTION = "semantic_cache"
CACHE_TTL_SECONDS = 86400  # 24 hours
HIT_THRESHOLD = 0.92


def _get_redis() -> redis_lib.Redis:
    settings = get_settings()
    return redis_lib.from_url(settings.redis_url, decode_responses=True)


def _embed(text: str) -> np.ndarray:
    """
    Use ChromaDB's default embedding function (all-MiniLM-L6-v2) so the 
    same embedding space is used for cache and semantic search.
    """
    import chromadb
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    ef = DefaultEmbeddingFunction()
    vectors = ef([text])
    return np.array(vectors[0], dtype=np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def cache_lookup(query: str) -> tuple[Optional[str], float]:
    """
    Returns (cached_response | None, latency_ms).
    Fetches all cached vectors from Redis and computes cosine similarity
    in-memory to find the best match.
    """
    t0 = time.time()
    try:
        r=_get_redis()
        query_vec=_embed(query)

        # All cache keys are stored in a Redis Set "cache_index"
        keys = r.smembers("cache_index")
        if not keys:
            return None, (time.time() - t0) * 1000

        best_score = 0.0
        best_response = None

        for key in keys:
            entry_raw = r.get(f"cache:{key}")
            if not entry_raw:
                continue
            entry = json.loads(entry_raw)
            cached_vec = np.array(entry["vector"], dtype=np.float32)
            score = _cosine(query_vec, cached_vec)
            if score > best_score:
                best_score = score
                best_response = entry["response"]

        latency_ms = (time.time() - t0) * 1000
        if best_score >= HIT_THRESHOLD:
            logger.info(f"[CACHE HIT] similarity={best_score:.4f} latency={latency_ms:.0f}ms")
            return best_response, latency_ms

        logger.info(f"[CACHE MISS] best_sim={best_score:.4f} latency={latency_ms:.0f}ms")
        return None, latency_ms

    except Exception as e:
        logger.warning(f"[CACHE] Lookup error: {e}")
        return None, (time.time() - t0) * 1000


def cache_store(query: str, response: str) -> None:
    """Store query+response in Redis with 24h TTL."""
    try:
        r = _get_redis()
        query_vec = _embed(query)
        key = hashlib.sha256(query.encode()).hexdigest()[:16]

        entry = {
            "query": query,
            "vector": query_vec.tolist(),
            "response": response,
        }
        r.setex(f"cache:{key}", CACHE_TTL_SECONDS, json.dumps(entry))
        r.sadd("cache_index", key)
        logger.info(f"[CACHE] Stored key={key}")
    except Exception as e:
        logger.warning(f"[CACHE] Store error: {e}")
