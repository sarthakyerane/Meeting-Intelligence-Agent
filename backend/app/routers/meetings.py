"""Meetings router — upload (audio or text), retrieve, and analyze."""

import time
import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, Response
from sqlalchemy.orm import Session

from app.database import get_db
from app import models, schemas
from app.services import agent, chroma_service, whisper_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/meetings", tags=["meetings"])


@router.post("/upload", response_model=schemas.UploadResponse)
async def upload_meeting(
    response: Response,
    title: str = Form(...),
    project: str = Form(...),
    transcript: Optional[str] = Form(None),
    duration_seconds: Optional[int] = Form(None),
    audio: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
):
    t0 = time.time()

    # ── 1. Get transcript text ──────────────────────────────────────────────
    if audio is not None:
        logger.info(f"[UPLOAD] Audio file: {audio.filename}")
        audio_bytes = await audio.read()
        transcript_text = whisper_service.transcribe_audio(audio_bytes, audio.filename)
    elif transcript:
        transcript_text = transcript
    else:
        raise HTTPException(status_code=422, detail="Provide either 'transcript' text or an 'audio' file.")

    # ── 2. Persist meeting record ───────────────────────────────────────────
    meeting = models.Meeting(
        title=title,
        project=project,
        transcript=transcript_text,
        duration_seconds=duration_seconds,
    )
    db.add(meeting)
    db.commit()
    db.refresh(meeting)

    # ── 3. Run agentic pipeline ─────────────────────────────────────────────
    pipeline_result = agent.run_pipeline(transcript_text)
    provider = pipeline_result.get("llm_provider", "unknown")

    # ── 4. Persist extracted data ───────────────────────────────────────────
    decision_objs = []
    for d in pipeline_result.get("decisions", []):
        obj = models.Decision(meeting_id=meeting.id, text=d.get("text", ""), owner=d.get("owner"))
        db.add(obj)
        decision_objs.append(obj)

    action_objs = []
    for a in pipeline_result.get("action_items", []):
        obj = models.ActionItem(
            meeting_id=meeting.id,
            text=a.get("text", ""),
            owner=a.get("owner"),
            deadline=a.get("deadline"),
        )
        db.add(obj)
        action_objs.append(obj)

    conflict_objs = []
    for c in pipeline_result.get("conflicts", []):
        obj = models.Conflict(
            meeting_id=meeting.id,
            party_a=c.get("party_a"),
            party_b=c.get("party_b"),
            issue=c.get("issue", ""),
            resolved=1 if c.get("resolved", False) else 0,
        )
        db.add(obj)
        conflict_objs.append(obj)

    question_objs = []
    for q in pipeline_result.get("unresolved_questions", []):
        obj = models.UnresolvedQuestion(
            meeting_id=meeting.id,
            question=q.get("question", ""),
            asker=q.get("asker"),
        )
        db.add(obj)
        question_objs.append(obj)

    db.commit()
    for obj in decision_objs + action_objs + conflict_objs + question_objs:
        db.refresh(obj)

    # ── 5. Store embeddings in ChromaDB ─────────────────────────────────────
    chroma_service.store_meeting_embedding(
        meeting.id,
        transcript_text,
        {"title": title, "project": project, "meeting_id": str(meeting.id)},
    )
    chroma_service.store_decision_embeddings(
        meeting.id,
        [{"text": d.text, "owner": d.owner} for d in decision_objs],
        project,
    )

    processing_ms = (time.time() - t0) * 1000
    response.headers["X-LLM-Provider"] = provider
    response.headers["X-Processing-Time-Ms"] = str(round(processing_ms))

    return schemas.UploadResponse(
        meeting_id=meeting.id,
        message=f"Meeting processed successfully via {provider}",
        analysis=schemas.MeetingAnalysis(
            meeting=schemas.MeetingOut.model_validate(meeting),
            decisions=[schemas.DecisionOut.model_validate(d) for d in decision_objs],
            action_items=[schemas.ActionItemOut.model_validate(a) for a in action_objs],
            conflicts=[schemas.ConflictOut.model_validate(c) for c in conflict_objs],
            unresolved_questions=[schemas.UnresolvedQuestionOut.model_validate(q) for q in question_objs],
            llm_provider_used=provider,
            processing_time_ms=round(processing_ms, 1),
        ),
    )


@router.get("/{meeting_id}", response_model=schemas.MeetingOut)
def get_meeting(meeting_id: int, db: Session = Depends(get_db)):
    meeting = db.query(models.Meeting).filter(models.Meeting.id == meeting_id).first()
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    return meeting


@router.get("/{meeting_id}/analysis", response_model=schemas.MeetingAnalysis)
def get_meeting_analysis(meeting_id: int, db: Session = Depends(get_db)):
    meeting = db.query(models.Meeting).filter(models.Meeting.id == meeting_id).first()
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    return schemas.MeetingAnalysis(
        meeting=schemas.MeetingOut.model_validate(meeting),
        decisions=[schemas.DecisionOut.model_validate(d) for d in meeting.decisions],
        action_items=[schemas.ActionItemOut.model_validate(a) for a in meeting.action_items],
        conflicts=[schemas.ConflictOut.model_validate(c) for c in meeting.conflicts],
        unresolved_questions=[schemas.UnresolvedQuestionOut.model_validate(q) for q in meeting.unresolved_questions],
    )
