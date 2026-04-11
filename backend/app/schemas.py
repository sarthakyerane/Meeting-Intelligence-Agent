from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel
from app.models import ActionStatus


# ─── Decision ───────────────────────────────────────────────────────────────

class DecisionBase(BaseModel):
    text: str
    owner: Optional[str] = None


class DecisionCreate(DecisionBase):
    pass


class DecisionOut(DecisionBase):
    id: int
    meeting_id: int
    created_at: datetime

    class Config:
        from_attributes = True


# ─── Action Item ─────────────────────────────────────────────────────────────

class ActionItemBase(BaseModel):
    text: str
    owner: Optional[str] = None
    deadline: Optional[str] = None


class ActionItemCreate(ActionItemBase):
    pass


class ActionItemUpdate(BaseModel):
    status: ActionStatus


class ActionItemOut(ActionItemBase):
    id: int
    meeting_id: int
    status: ActionStatus
    created_at: datetime

    class Config:
        from_attributes = True


# ─── Conflict ────────────────────────────────────────────────────────────────

class ConflictBase(BaseModel):
    party_a: Optional[str] = None
    party_b: Optional[str] = None
    issue: str
    resolved: bool = False


class ConflictOut(ConflictBase):
    id: int
    meeting_id: int
    created_at: datetime

    class Config:
        from_attributes = True


# ─── Unresolved Question ─────────────────────────────────────────────────────

class UnresolvedQuestionBase(BaseModel):
    question: str
    asker: Optional[str] = None


class UnresolvedQuestionOut(UnresolvedQuestionBase):
    id: int
    meeting_id: int
    created_at: datetime

    class Config:
        from_attributes = True


# ─── Meeting ─────────────────────────────────────────────────────────────────

class MeetingCreate(BaseModel):
    title: str
    project: str
    transcript: Optional[str] = None
    duration_seconds: Optional[int] = None


class MeetingOut(BaseModel):
    id: int
    title: str
    project: str
    date: datetime
    duration_seconds: Optional[int]
    created_at: datetime

    class Config:
        from_attributes = True


class MeetingAnalysis(BaseModel):
    meeting: MeetingOut
    decisions: List[DecisionOut]
    action_items: List[ActionItemOut]
    conflicts: List[ConflictOut]
    unresolved_questions: List[UnresolvedQuestionOut]
    llm_provider_used: Optional[str] = None
    processing_time_ms: Optional[float] = None


# ─── Upload Response ──────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    meeting_id: int
    message: str
    analysis: MeetingAnalysis


# ─── Search ──────────────────────────────────────────────────────────────────

class SearchResult(BaseModel):
    meeting_id: int
    meeting_title: str
    project: str
    text: str
    score: float
    type: str  # "decision", "action_item", etc.


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    cache_hit: bool
    latency_ms: float


# ─── Contradiction ───────────────────────────────────────────────────────────

class ContradictionOut(BaseModel):
    meeting_id_a: int
    meeting_title_a: str
    decision_a: str
    meeting_id_b: int
    meeting_title_b: str
    decision_b: str
    similarity_score: float
    contradiction_explanation: str


# ─── Project History ─────────────────────────────────────────────────────────

class ProjectHistory(BaseModel):
    project: str
    total_meetings: int
    summary: str
    all_decisions: List[DecisionOut]
    open_action_items: List[ActionItemOut]
    unresolved_conflicts: List[ConflictOut]


# ─── Eval ────────────────────────────────────────────────────────────────────

class EvalMetrics(BaseModel):
    category: str
    precision: float
    recall: float
    f1: float


class EvalReport(BaseModel):
    num_cases: int
    decisions: EvalMetrics
    action_items: EvalMetrics
    conflicts: EvalMetrics
    unresolved_questions: EvalMetrics
    latency_ms: List[float]
    avg_latency_ms: float
    llm_providers_used: List[str]
