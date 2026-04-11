"""
Intelligence router:
- GET /contradictions  — cross-meeting semantic contradiction detection
- GET /project/{name}/history — full narrative summary for a project
"""

import logging
from typing import Optional, List

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app import models, schemas
from app.services import chroma_service, llm_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["intelligence"])


@router.get("/contradictions", response_model=List[schemas.ContradictionOut])
def find_contradictions(
    project: str = Query(..., description="Project name to search within"),
    db: Session = Depends(get_db),
):
    """
    For each decision stored in this project, check if any past decisions
    from OTHER meetings are semantically similar (potentially contradictory).
    Then use the LLM to confirm and explain the contradiction.
    """
    # Get all decisions for this project
    decisions = (
        db.query(models.Decision)
        .join(models.Meeting)
        .filter(models.Meeting.project == project)
        .all()
    )

    contradictions = []
    seen_pairs = set()

    for decision in decisions:
        candidates = chroma_service.find_contradictions(
            decision.text, project, exclude_meeting_id=decision.meeting_id
        )

        for candidate in candidates:
            pair_key = tuple(sorted([decision.id, candidate["meeting_id"]]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            # Ask LLM: do these two decisions actually contradict?
            prompt = f"""Two decisions were made in different meetings for the same project.
Decision A: "{decision.text}"
Decision B: "{candidate['decision_text']}"

Do these decisions CONTRADICT each other (i.e., they cannot both be true)?
Reply with JSON: {{"contradicts": true/false, "explanation": "one sentence explanation"}}"""

            try:
                raw, _ = llm_service.call_llm(prompt)
                result = llm_service.parse_json_response(raw)
                if not result.get("contradicts", False):
                    continue
            except Exception as e:
                logger.warning(f"[CONTRADICTIONS] LLM check failed: {e}")
                continue

            # Look up meeting B details from MySQL
            meeting_b = db.query(models.Meeting).filter(
                models.Meeting.id == candidate["meeting_id"]
            ).first()
            meeting_a = db.query(models.Meeting).filter(
                models.Meeting.id == decision.meeting_id
            ).first()

            if not meeting_a or not meeting_b:
                continue

            contradictions.append(schemas.ContradictionOut(
                meeting_id_a=meeting_a.id,
                meeting_title_a=meeting_a.title,
                decision_a=decision.text,
                meeting_id_b=meeting_b.id,
                meeting_title_b=meeting_b.title,
                decision_b=candidate["decision_text"],
                similarity_score=round(candidate["similarity"], 4),
                contradiction_explanation=result.get("explanation", ""),
            ))

    return contradictions


@router.get("/project/{name}/history", response_model=schemas.ProjectHistory)
def project_history(name: str, db: Session = Depends(get_db)):
    """Generate a summarised narrative of all decisions and open action items for a project."""
    meetings = db.query(models.Meeting).filter(models.Meeting.project == name).all()
    if not meetings:
        raise HTTPException(status_code=404, detail=f"No meetings found for project '{name}'")

    all_decisions = (
        db.query(models.Decision)
        .join(models.Meeting)
        .filter(models.Meeting.project == name)
        .order_by(models.Decision.created_at)
        .all()
    )
    open_actions = (
        db.query(models.ActionItem)
        .join(models.Meeting)
        .filter(models.Meeting.project == name, models.ActionItem.status != models.ActionStatus.done)
        .all()
    )
    unresolved_conflicts = (
        db.query(models.Conflict)
        .join(models.Meeting)
        .filter(models.Meeting.project == name, models.Conflict.resolved == 0)
        .all()
    )

    # Build narrative with LLM
    decisions_text = "\n".join([f"- {d.text}" for d in all_decisions]) or "None"
    actions_text = "\n".join([f"- {a.text} (owner: {a.owner}, due: {a.deadline})" for a in open_actions]) or "None"

    prompt = f"""Summarize the project history for "{name}" in 3-5 sentences.

Decisions made so far:
{decisions_text}

Open action items:
{actions_text}

Write a brief executive summary of where the project stands, what has been decided, and what remains open."""

    try:
        summary, _ = llm_service.call_llm(prompt)
        summary = summary.strip()
    except Exception as e:
        logger.warning(f"[HISTORY] LLM summary failed: {e}")
        summary = f"Project {name} has {len(all_decisions)} decisions and {len(open_actions)} open action items."

    return schemas.ProjectHistory(
        project=name,
        total_meetings=len(meetings),
        summary=summary,
        all_decisions=[schemas.DecisionOut.model_validate(d) for d in all_decisions],
        open_action_items=[schemas.ActionItemOut.model_validate(a) for a in open_actions],
        unresolved_conflicts=[schemas.ConflictOut.model_validate(c) for c in unresolved_conflicts],
    )
