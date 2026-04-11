from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, ForeignKey, Enum
)
from sqlalchemy.orm import relationship
from app.database import Base
import enum


class ActionStatus(str, enum.Enum):
    pending = "pending"
    in_progress = "in_progress"
    done = "done"


class Meeting(Base):
    __tablename__ = "meetings"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    project = Column(String(255), nullable=False, index=True)
    transcript = Column(Text, nullable=False)
    date = Column(DateTime, default=datetime.utcnow)
    duration_seconds = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    decisions = relationship("Decision", back_populates="meeting", cascade="all, delete")
    action_items = relationship("ActionItem", back_populates="meeting", cascade="all, delete")
    conflicts = relationship("Conflict", back_populates="meeting", cascade="all, delete")
    unresolved_questions = relationship("UnresolvedQuestion", back_populates="meeting", cascade="all, delete")


class Decision(Base):
    __tablename__ = "decisions"

    id = Column(Integer, primary_key=True, index=True)
    meeting_id = Column(Integer, ForeignKey("meetings.id", ondelete="CASCADE"), nullable=False, index=True)
    text = Column(Text, nullable=False)
    owner = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    meeting = relationship("Meeting", back_populates="decisions")


class ActionItem(Base):
    __tablename__ = "action_items"

    id = Column(Integer, primary_key=True, index=True)
    meeting_id = Column(Integer, ForeignKey("meetings.id", ondelete="CASCADE"), nullable=False, index=True)
    text = Column(Text, nullable=False)
    owner = Column(String(255), nullable=True)
    deadline = Column(String(100), nullable=True)
    status = Column(Enum(ActionStatus), default=ActionStatus.pending)
    created_at = Column(DateTime, default=datetime.utcnow)

    meeting = relationship("Meeting", back_populates="action_items")


class Conflict(Base):
    __tablename__ = "conflicts"

    id = Column(Integer, primary_key=True, index=True)
    meeting_id = Column(Integer, ForeignKey("meetings.id", ondelete="CASCADE"), nullable=False, index=True)
    party_a = Column(String(255), nullable=True)
    party_b = Column(String(255), nullable=True)
    issue = Column(Text, nullable=False)
    resolved = Column(Integer, default=0)  # 0=unresolved, 1=resolved
    created_at = Column(DateTime, default=datetime.utcnow)

    meeting = relationship("Meeting", back_populates="conflicts")


class UnresolvedQuestion(Base):
    __tablename__ = "unresolved_questions"

    id = Column(Integer, primary_key=True, index=True)
    meeting_id = Column(Integer, ForeignKey("meetings.id", ondelete="CASCADE"), nullable=False, index=True)
    question = Column(Text, nullable=False)
    asker = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    meeting = relationship("Meeting", back_populates="unresolved_questions")
