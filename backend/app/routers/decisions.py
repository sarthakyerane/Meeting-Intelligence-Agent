"""Decisions router — filter, list, and semantic search with Redis cache."""

import time
import logging
from typing import Optional, List

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app import models, schemas
from app.services import chroma_service, cache_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/decisions", tags=["decisions"])


@router.get("", response_model=List[schemas.DecisionOut])
def list_decisions(
    project: Optional[str] = Query(None),
    owner: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    q = db.query(models.Decision).join(models.Meeting)
    if project:
        q = q.filter(models.Meeting.project == project)
    if owner:
        q = q.filter(models.Decision.owner.ilike(f"%{owner}%"))
    return q.order_by(models.Decision.created_at.desc()).all()


@router.get("/search", response_model=schemas.SearchResponse)
def search_decisions(
    q: str = Query(..., description="Natural language query"),
    project: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    t0 = time.time()

    # ── Check Redis semantic cache first ────────────────────────────────────
    cache_key = f"{q}|project:{project or 'all'}"
    cached_response, cache_latency = cache_service.cache_lookup(cache_key)

    if cached_response:
        import json
        results_data = json.loads(cached_response)
        total_latency = (time.time() - t0) * 1000
        return schemas.SearchResponse(
            query=q,
            results=[schemas.SearchResult(**r) for r in results_data],
            cache_hit=True,
            latency_ms=round(total_latency, 1),
        )

    # ── Cache miss — query ChromaDB ─────────────────────────────────────────
    raw_results = chroma_service.search_meetings(q, n_results=5, project=project)

    # Enrich with meeting metadata from MySQL
    search_results = []
    for r in raw_results:
        meeting = db.query(models.Meeting).filter(models.Meeting.id == r["meeting_id"]).first()
        if meeting:
            search_results.append(schemas.SearchResult(
                meeting_id=r["meeting_id"],
                meeting_title=meeting.title,
                project=meeting.project,
                text=r["text"][:500],
                score=round(r["score"], 4),
                type="transcript",
            ))

    total_latency = (time.time() - t0) * 1000

    # Store in Redis cache
    import json
    cache_service.cache_store(cache_key, json.dumps([r.model_dump() for r in search_results]))

    return schemas.SearchResponse(
        query=q,
        results=search_results,
        cache_hit=False,
        latency_ms=round(total_latency, 1),
    )
