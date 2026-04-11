"""Action items router — list, filter, and update status."""

from typing import Optional, List

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app import models, schemas

router = APIRouter(prefix="/action-items", tags=["action-items"])


@router.get("", response_model=List[schemas.ActionItemOut])
def list_action_items(
    owner: Optional[str] = Query(None),
    status: Optional[models.ActionStatus] = Query(None),
    project: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    q = db.query(models.ActionItem).join(models.Meeting)
    if owner:
        q = q.filter(models.ActionItem.owner.ilike(f"%{owner}%"))
    if status:
        q = q.filter(models.ActionItem.status == status)
    if project:
        q = q.filter(models.Meeting.project == project)
    return q.order_by(models.ActionItem.created_at.desc()).all()


@router.patch("/{item_id}", response_model=schemas.ActionItemOut)
def update_action_item(
    item_id: int,
    update: schemas.ActionItemUpdate,
    db: Session = Depends(get_db),
):
    item = db.query(models.ActionItem).filter(models.ActionItem.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Action item not found")
    item.status = update.status
    db.commit()
    db.refresh(item)
    return item
